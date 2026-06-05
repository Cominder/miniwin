import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
import re
import gc
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.model_miniwin import MiniWinConfig
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import (
    Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed,
    SkipBatchSampler, init_model, LMForRewardModel, init_wandb_run,
)
from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps

warnings.filterwarnings('ignore')

args = None
model = None
tokenizer = None
lm_config = None
autocast_ctx = None


def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses), device=args.device)
    scale = 3.0
    t_open = '<' + 'think' + '>'
    t_close = '</' + 'think' + '>'

    def reasoning_model_reward(rw):
        p1 = '^' + re.escape(t_open) + r'\n.*?\n' + re.escape(t_close) + r'\n<answer>\n.*?\n</answer>$'
        p2 = '^' + re.escape(t_open) + r'\n.*?\n' + re.escape(t_close) + r'\n\n<answer>\n.*?\n</answer>$'
        mp = [re.match(p1, r, re.S) for r in responses]
        mp2 = [re.match(p2, r, re.S) for r in responses]
        rw += torch.tensor([0.5 if (a or b) else 0.0 for a, b in zip(mp, mp2)], device=args.device)

        def mark_num(text):
            r = 0
            if text.count(t_open) == 1:
                r += 0.25
            if text.count(t_close) == 1:
                r += 0.25
            if text.count("<answer>") == 1:
                r += 0.25
            if text.count("</answer>") == 1:
                r += 0.25
            return r

        rw += torch.tensor([mark_num(rsp) for rsp in responses], device=args.device)
        return rw

    if getattr(args, 'reasoning', 0) == 1:
        rewards = reasoning_model_reward(rewards)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)
        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]
                pat = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pat, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                answer = response
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
                if t_close in response:
                    thinking_content, rest = response.split(t_close, 1)
                    answer = rest.strip()
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    rewards[response_idx] += 0.25 if response.count(t_close) == 1 else -0.25
                rewards[response_idx] -= rep_penalty(answer)
                score = reward_model.get_score(messages, answer)
                score = max(min(score, scale), -scale)
                if getattr(args, 'reasoning', 0) == 1:
                    am = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                    if am:
                        ac = am.group(1).strip()
                        ascore = reward_model.get_score(messages, ac)
                        ascore = max(min(ascore, scale), -scale)
                        score = score * 0.4 + ascore * 0.6
                reward_model_scores.append(score)
        rewards += torch.tensor(reward_model_scores, device=args.device)
    return rewards


def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, optimizer, scheduler,
                     start_step=0, wandb=None, use_sglang=False):
    global model, tokenizer, args, lm_config, autocast_ctx
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        outputs = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        completions = rollout_result.completions
        old_per_token_logps = rollout_result.per_token_logps.to(args.device)

        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        model_unwrapped = getattr(model_unwrapped, '_orig_mod', model_unwrapped)
        with autocast_ctx:
            if use_sglang or lm_config.use_moe:
                res = model_unwrapped(outputs)
                aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
                logits = res.logits[:, :-1, :]
                per_token_logps = F.log_softmax(logits, dim=-1).gather(
                    2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1)[:, -completion_ids.size(1):]
            else:
                aux_loss = torch.tensor(0.0, device=args.device)
                per_token_logps = rollout_result.per_token_logps

        with torch.no_grad():
            ref_per_token_logps = compute_per_token_logps(ref_model, outputs, completion_ids.size(1))
        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-' * 100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(completions[idx])
                    Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")
                Logger('=' * 100)

        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        is_eos = completion_ids == tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()

        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        if args.loss_type == "cispo":
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if is_main_process() and step % args.save_interval == 0:
                rollout_engine.update_policy(model)

        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            kl_ref_val = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / completion_mask.sum().item()
            advantages_mean_val = advantages.mean().item()
            advantages_std_val = advantages.std().item()
            current_lr = optimizer.param_groups[0]['lr']
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                   f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                   f'Actor Loss: {policy_loss_val:.4f}, Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')
            if wandb and is_main_process():
                wandb.log({
                    "reward": avg_reward_val,
                    "kl_ref": kl_ref_val,
                    "advantages_std": advantages_std_val,
                    "advantages_mean": advantages_mean_val,
                    "policy_loss": policy_loss_val,
                    "avg_response_len": avg_len_val,
                    "learning_rate": current_lr
                })

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask
        torch.cuda.empty_cache()
        gc.collect()

    if step > start_step and step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if is_main_process() and step % args.save_interval == 0:
            rollout_engine.update_policy(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniWin GRPO / CISPO")
    parser.add_argument("--save_dir", type=str, default="./out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="./dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--num_generations", type=int, default=6, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.1, help="KL惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="CISPO ratio 上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练（reasoning=1 时仍可用 reason 权重）")
    parser.add_argument("--reasoning", type=int, default=0, choices=[0, 1], help="是否启用推理格式额外奖励（0=否，1=是）")
    parser.add_argument("--reward_model_path", type=str, default="./internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")
    parser.add_argument("--use_wandb", action="store_true", help="启用 Weights & Biases（wandb.ai）实验跟踪")
    parser.add_argument("--wandb_project", type=str, default="MiniWin-GRPO", help="wandb 项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="torch.compile")
    parser.add_argument("--debug_mode", action="store_true", help="调试打印采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug 间隔 step")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="RLAIFDataset open_thinking 概率")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout 引擎")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8996", help="SGLang URL")
    parser.add_argument("--sglang_model_path", type=str, default="./model", help="SGLang tokenizer 路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo", help="SGLang 共享权重路径")
    parser.add_argument("--sliding_window_size", type=int, default=64)
    parser.add_argument("--sliding_pe_type", type=str, default="rope", choices=["rope", "alibi"])
    parser.add_argument("--attention_pattern", type=str, default="SSSL")
    parser.add_argument("--attention_pattern_force_last_global", type=int, default=1, choices=[0, 1])
    parser.add_argument("--partial_rotary_factor", type=float, default=1.0, help="MiMo 风格部分 RoPE，1.0=全头")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniWinConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len + args.max_gen_len,
        use_moe=bool(args.use_moe),
        sliding_window_size=args.sliding_window_size,
        sliding_pe_type=args.sliding_pe_type,
        attention_pattern=args.attention_pattern,
        attention_pattern_force_last_global=bool(args.attention_pattern_force_last_global),
        partial_rotary_factor=args.partial_rotary_factor,
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='./checkpoints') if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        wandb_run_name = (
            f"MiniWin-GRPO-E{args.epochs}-BS{args.batch_size}-LR{args.learning_rate}"
            f"-nw{args.n_windows}-sw{args.sliding_window_size}-pat{args.attention_pattern}"
        )
        wandb = init_wandb_run(args.wandb_project, wandb_run_name, ckp_data)

    base_weight = "reason" if args.reasoning == 1 else args.from_weight
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)

    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )

    train_ds = RLAIFDataset(
        args.data_path, tokenizer,
        max_length=lm_config.max_seq_len,
        thinking_ratio=args.thinking_ratio,
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    if is_main_process():
        rollout_engine.update_policy(model)

    use_sglang = args.rollout_engine == "sglang"
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step} 个 step')
            grpo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model,
                             optimizer, scheduler, start_step, wandb, use_sglang=use_sglang)
        else:
            grpo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model,
                             optimizer, scheduler, 0, wandb, use_sglang=use_sglang)

    if dist.is_initialized():
        dist.destroy_process_group()
