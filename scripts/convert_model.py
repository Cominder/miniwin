import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from model.model_miniwin import MiniWinConfig, MiniWinForCausalLM
from model.model_lora import apply_lora, merge_lora

warnings.filterwarnings('ignore', category=UserWarning)


# MoE模型需使用此函数转换
def convert_torch2transformers_miniwin(torch_path, transformers_path, dtype=torch.float16):
    MiniWinConfig.register_for_auto_class()
    MiniWinForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    lm_model = MiniWinForCausalLM(lm_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)  # 转换模型权重精度
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained('../model')
    tokenizer.save_pretrained(transformers_path)
    print(f"模型已保存为 Transformers-MiniWin 格式: {transformers_path}")


# LlamaForCausalLM结构兼容第三方生态
def convert_torch2transformers_llama(torch_path, transformers_path, dtype=torch.float16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    llama_config = LlamaConfig(
        vocab_size=lm_config.vocab_size,
        hidden_size=lm_config.hidden_size,
        intermediate_size=64 * ((int(lm_config.hidden_size * 8 / 3) + 64 - 1) // 64),
        num_hidden_layers=lm_config.num_hidden_layers,
        num_attention_heads=lm_config.num_attention_heads,
        num_key_value_heads=lm_config.num_key_value_heads,
        max_position_embeddings=lm_config.max_position_embeddings,
        rms_norm_eps=lm_config.rms_norm_eps,
        rope_theta=lm_config.rope_theta,
        tie_word_embeddings=True
    )
    llama_model = LlamaForCausalLM(llama_config)
    llama_model.load_state_dict(state_dict, strict=False)
    llama_model = llama_model.to(dtype)  # 转换模型权重精度
    llama_model.save_pretrained(transformers_path)
    model_params = sum(p.numel() for p in llama_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    tokenizer = AutoTokenizer.from_pretrained('../model')
    tokenizer.save_pretrained(transformers_path)
    print(f"模型已保存为 Transformers-Llama 格式: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path):
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save(model.state_dict(), torch_path)
    print(f"模型已保存为 PyTorch 格式: {torch_path}")


def run_merge_lora(base_pth, lora_pth, out_pth, lm_config, device=None, lora_rank=8):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = MiniWinForCausalLM(lm_config).to(device)
    model.load_state_dict(torch.load(base_pth, map_location=device), strict=False)
    apply_lora(model, rank=lora_rank)
    merge_lora(model, lora_pth, out_pth)
    print(f"已合并 LoRA 并保存: {out_pth}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MiniWin 权重转换 / LoRA 合并")
    parser.add_argument('--mode', choices=['export_miniwin', 'merge_lora'], default='export_miniwin')
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--use_moe', type=int, default=0, choices=[0, 1])
    parser.add_argument('--n_windows', type=int, default=2)
    parser.add_argument('--sliding_window_size', type=int, default=64)
    parser.add_argument('--attention_pattern', type=str, default='SSSL')
    parser.add_argument('--attention_pattern_force_last_global', type=int, default=1, choices=[0, 1])
    parser.add_argument('--partial_rotary_factor', type=float, default=1.0, help='MiMo 风格部分 RoPE，1.0=全头')
    parser.add_argument('--base_pth', type=str, default='', help='merge_lora: 基础 .pth')
    parser.add_argument('--lora_pth', type=str, default='', help='merge_lora: LoRA .pth')
    parser.add_argument('--out_pth', type=str, default='', help='merge_lora: 输出完整 .pth')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--torch_path', type=str, default='')
    parser.add_argument('--transformers_path', type=str, default='')
    args = parser.parse_args()

    lm_config = MiniWinConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len,
        use_moe=bool(args.use_moe),
        n_windows=args.n_windows,
        sliding_window_size=args.sliding_window_size,
        attention_pattern=args.attention_pattern,
        attention_pattern_force_last_global=bool(args.attention_pattern_force_last_global),
        partial_rotary_factor=args.partial_rotary_factor,
    )

    if args.mode == 'merge_lora':
        if not args.base_pth or not args.lora_pth or not args.out_pth:
            raise SystemExit('merge_lora 需要 --base_pth --lora_pth --out_pth')
        run_merge_lora(args.base_pth, args.lora_pth, args.out_pth, lm_config, lora_rank=args.lora_rank)
    else:
        moe = '_moe' if lm_config.use_moe else ''
        torch_path = args.torch_path or f"../out/full_sft_{lm_config.hidden_size}{moe}.pth"
        transformers_path = args.transformers_path or '../miniwin_hf'
        os.makedirs(transformers_path, exist_ok=True)
        convert_torch2transformers_miniwin(torch_path, transformers_path)
