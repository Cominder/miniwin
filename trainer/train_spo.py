import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import gc
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model_miniwin import MiniWinConfig, MiniWinForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import (
    Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed,
    SkipBatchSampler, init_model, LMForRewardModel, init_wandb_run,
)
from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps

warnings.filterwarnings('ignore')


class AutoAdaptiveValueTracker:
    """SPO自适应价值追踪器"""
    def __init__(self, rho_mode='kl', rho_const=0.9, D_half=0.06, clip_lower=0.5, clip_upper=0.96):
        self.rho_mode = rho_mode
        self.rho_const = rho_const
        self.D_half = D_half
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper
        N_init = 1.0 / (1.0 - self.clip_lower)
        self.alpha = 0.5 * N_init
        self.beta = 0.5 * N_init
        self.old_mean_logprob = None

    def get_baselines(self, batch_size):
        baseline = self.alpha / (self.alpha + self.beta)
        return torch.full((batch_size,), baseline, dtype=torch.float32)

    def compute_rho(self, cur_mean_logprob):
        if self.rho_mode == 'constant':
            return self.rho_const
        if self.old_mean_logprob is None:
            return self.rho_const
        kl = abs(self.old_mean_logprob - cur_mean_logprob)
        rho = 2 ** (-kl / self.D_half)
        return max(min(rho, self.clip_upper), self.clip_lower)

    def update(self, rewards, cur_logprobs=None, response_masks=None):
        if cur_logprobs is not None and response_masks is not None:
            mean_logprob = ((cur_logprobs * response_masks).sum() / response_masks.sum()).item()
            rho = self.compute_rho(mean_logprob)
            self.old_mean_logprob = mean_logprob
        else:
            rho = self.rho_const

        scale = 3.0
        normalized_rewards = (rewards + scale) / (2 * scale)
        avg_normalized_reward = normalized_rewards.mean().item()
        self.alpha = rho * self.alpha + avg_normalized_reward
        self.beta = rho * self.beta + (1 - avg_normalized_reward)
        return rho


def rep_penalty_spo(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def calculate_rewards(prompts, responses, reward_model):
    """整合所有奖励函数计算总奖励"""
    def reasoning_model_reward(rewards):
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern or match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        def mark_num(text):
            reward = 0
            if text.count("<think>") == 1: reward += 0.25
            if text.count("</think>") == 1: reward += 0.25
            if text.count("<answer>") == 1: reward += 0.25
            if text.count("</answer>") == 1: reward += 0.25
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    rewards = torch.zeros(len(responses), device=args.device)
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    with torch.no_grad():
        reward_model_scores = []
        scale = 3.0

        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]
            answer = response
            reward_extra = 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
            _tc = '</' + 'think' + '>'
            if _tc in response:
                thinking_content, rest = response.split(_tc, 1)
                answer = rest.strip()
                reward_extra += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                reward_extra += 0.25 if response.count(_tc) == 1 else -0.25
            reward_extra -= rep_penalty_spo(answer)
            score = reward_model.get_score(messages, answer)
            score = max(min(score, scale), -scale) + reward_extra
            if args.reasoning == 1:
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    answer_score = reward_model.get_score(messages, answer_content)
                    answer_score = max(min(answer_score, scale), -scale)
                    score = score * 0.4 + answer_score * 0.6
            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def spo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, value_tracker, optimizer, scheduler,
                    start_step=0, wandb=None):
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
            num_generations=1,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        outputs = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        per_token_logps = rollout_result.per_token_logps
        with torch.no_grad():
            ref_per_token_logps = compute_per_token_logps(ref_model, outputs, completion_ids.size(1))

        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)

        baselines = value_tracker.get_baselines(len(prompts)).to(args.device)  # [B]

        scale = 3.0
        # Un-normalize baselines to be in the same scale as raw rewards [-3, 3]
        unnormalized_baselines = baselines * (2 * scale) - scale  # [B]
        advantages = rewards - unnormalized_baselines  # [B]

        # 直接使用 baseline 提供的优势估计，只做裁剪防止梯度爆炸。不再做 batch 内归一化，因为 baseline 已经提供了跨 batch 的稳定基线
        advantages = advantages.clamp(-5.0, 5.0)

        is_eos = completion_ids == tokenizer.eos_token_id  # [B, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)  # [B]
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()  # [B, R]

        kl_div = ref_per_token_logps - per_token_logps  # [B, R]
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B, R]
        per_token_loss = -per_token_logps * advantages.unsqueeze(1) + args.beta * per_token_kl  # [B, R]
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean() / args.accumulation_steps  # scalar
        loss.backward()

        response_masks = completion_mask.float()  # [B, R]
        rho = value_tracker.update(rewards, per_token_logps.detach(), response_masks)

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if is_main_process() and step % args.save_interval == 0:
                rollout_engine.update_policy(model)
            torch.cuda.empty_cache()

        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            kl_val = ((per_token_kl * completion_mask).sum() / (completion_mask.sum() + 1e-8)).item()
            avg_baseline_val = baselines.mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch: {epoch+1}, Step: {step}/{iters}, '
                   f'Actor Loss: {policy_loss_val:.6f}, Reward: {avg_reward_val:.6f}, '
                   f'Baseline: {avg_baseline_val:.4f}, KL: {kl_val:.4f}, Rho: {rho:.4f}, '
                   f'Avg Response Len: {avg_len_val:.2f}, LR: {current_lr:.2e}')

            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "reward": avg_reward_val,
                    "kl": kl_val,
                    "rho": float(rho),
                    "baseline": avg_baseline_val,
                    "advantages_mean": advantages.mean().item(),
                    "learning_rate": current_lr
                })

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_m = model.module if isinstance(model, DistributedDataParallel) else model
            raw_m = getattr(raw_m, '_orig_mod', raw_m)
            state_dict = raw_m.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, advantages, completion_mask, baselines, response_masks
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniWin SPO (Self-Play Optimization)")
    parser.add_argument("--save_dir", type=str, default="./out", help="模型保存目录")
    parser.add_argument('--save_weight', default='spo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="./dataset/rlaif-mini.jsonl", help="RLAIF数据路径")
    parser.add_argument("--beta", type=float, default=0.02, help="KL惩罚系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniWin-SPO", help="wandb项目名")
    parser.add_argument("--n_windows", type=int, default=2)
    parser.add_argument("--sliding_window_size", type=int, default=64)
    parser.add_argument("--attention_pattern", type=str, default="SSSL")
    parser.add_argument( "--attention_pattern_force_last_global", type=int, default=1, choices=[0, 1], help=" set last layer L")
    parser.add_argument("--partial_rotary_factor", type=float, default=1.0, help="MiMo 风格部分 RoPE，1.0=全头")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="RLAIFDataset open_thinking 概率")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"])
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8996")
    parser.add_argument("--sglang_model_path", type=str, default="./model")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_spo")
    parser.add_argument("--use_compile", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniWinConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe),
        n_windows=args.n_windows,
        sliding_window_size=args.sliding_window_size,
        attention_pattern=args.attention_pattern,
        attention_pattern_force_last_global=bool(args.attention_pattern_force_last_global),
        partial_rotary_factor=args.partial_rotary_factor,
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='./checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        wandb_run_name = f"MiniWin-SPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}-Nwindows-{args.n_windows}-SlidingWindowSize-{args.sliding_window_size}-AttentionPattern-{args.attention_pattern}-AttentionPatternForceLastGlobal-{args.attention_pattern_force_last_global}"
        wandb = init_wandb_run(args.wandb_project, wandb_run_name, ckp_data)
    
    # ========== 5. 初始化模型（Policy, Ref, Reward）和Value Tracker、数据 ==========
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    # Policy模型
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # Reference模型
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # Reward模型
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
    # Value Tracker
    value_tracker = AutoAdaptiveValueTracker(rho_mode='kl', rho_const=0.9, D_half=0.06, clip_lower=0.5, clip_upper=0.96)
    
    train_ds = RLAIFDataset(
        args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio,
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    
    # ========== 6. 从ckp恢复状态 ==========
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
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            spo_train_epoch(epoch, loader, len(loader) + start_step + 1, rollout_engine, ref_model, reward_model,
                            value_tracker, optimizer, scheduler, start_step, wandb)
        else:
            loader = DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True,
                              drop_last=False, shuffle=(train_sampler is None),
                              num_workers=args.num_workers, sampler=train_sampler)
            spo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model,
                            value_tracker, optimizer, scheduler, 0, wandb)
