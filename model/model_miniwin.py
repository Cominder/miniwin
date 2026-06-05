
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, PretrainedConfig

try:
    from transformers import GenerationMixin
except ImportError:
    try:
        from transformers.generation.utils import GenerationMixin
    except ImportError:
        GenerationMixin = object
from transformers.modeling_outputs import CausalLMOutputWithPast

from .utils import (
    apply_partial_rotary_pos_emb,
    flash_attn_func_miniwin,
    precompute_freqs_cis,
    repeat_kv,
    rope_dim_from_factor,
)


class MiniWinConfig(PretrainedConfig):
    model_type = "miniwin"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 768,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1e6,
        flash_attn: bool = True,
        sliding_window_size: int = 64,
        attention_pattern: str = "SSSL",
        attention_pattern_force_last_global: bool = True,
        sliding_pe_type: str = "rope",
        nope_logit_log_base: Optional[float] = None,
        use_moe: bool = False,
        num_experts_per_tok: int = 1,
        n_routed_experts: int = 4,
        n_shared_experts: int = 0,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        max_seq_len: Optional[int] = None,
        partial_rotary_factor: float = 1.0,
        tie_word_embeddings: bool = True,
        head_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = None
        self.flash_attn = flash_attn
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.num_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.sliding_window_size = sliding_window_size
        self.attention_pattern = attention_pattern
        self.attention_pattern_force_last_global = attention_pattern_force_last_global
        self.max_seq_len = max_seq_len if max_seq_len is not None else self.max_position_embeddings
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.tie_word_embeddings = tie_word_embeddings
        self.head_dim = head_dim or (hidden_size // num_attention_heads)

        sliding_pe_type = sliding_pe_type.lower()
        if sliding_pe_type not in ("rope", "alibi"):
            raise ValueError(f"sliding_pe_type must be 'rope' or 'alibi', got {sliding_pe_type!r}")
        self.sliding_pe_type = sliding_pe_type
        self.nope_logit_log_base = nope_logit_log_base


def _alibi_slopes_pow2(n: int) -> List[float]:
    start = 2 ** (-(2 ** -(math.log2(n) - 3)))
    return [start * (start ** i) for i in range(n)]


def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    if math.log2(n_heads).is_integer():
        slopes = _alibi_slopes_pow2(n_heads)
    else:
        closest = 2 ** int(math.floor(math.log2(n_heads)))
        extra = get_alibi_slopes(2 * closest).tolist()[0::2][: n_heads - closest]
        slopes = _alibi_slopes_pow2(closest) + extra
    return torch.tensor(slopes, dtype=torch.float32)


def build_sliding_alibi_bias(slopes, q_len, k_len, window_left, device, dtype):
    row = (k_len - q_len) + torch.arange(q_len, device=device).unsqueeze(1)
    col = torch.arange(k_len, device=device).unsqueeze(0)
    rel = row - col
    valid = (col <= row) & (rel <= window_left) & (rel >= 0)
    rel_pos = rel.clamp(min=0).to(dtype=dtype)
    bias = -slopes.to(device=device, dtype=dtype).view(-1, 1, 1) * rel_pos
    bias = bias.masked_fill(~valid.unsqueeze(0), float("-inf"))
    return bias.unsqueeze(0)


def layer_attention_mode(config: MiniWinConfig, layer_id: int) -> str:
    pattern = getattr(config, "attention_pattern", "SSSL").upper()
    if not pattern or any(c not in "SL" for c in pattern):
        raise ValueError(f"attention_pattern must be non-empty and only contain S and L, got {pattern!r}")
    ch = pattern[layer_id % len(pattern)]
    if getattr(config, "attention_pattern_force_last_global", True) and layer_id == config.num_hidden_layers - 1:
        ch = "L"
    return ch


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


class FeedForward(nn.Module):
    def __init__(self, config: MiniWinConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    def __init__(self, config: MiniWinConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        scores = logits.softmax(dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniWinConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_routed_experts)])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_shared_experts)])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_out = expert(x[exp_token_idx]).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)
        return expert_cache


class NoPEGlobalAttention(nn.Module):
    def __init__(self, args: MiniWinConfig):
        super().__init__()
        self.num_key_value_heads = (
            args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        )
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(F, "scaled_dot_product_attention") and args.flash_attn
        self.nope_logit_log_base = args.nope_logit_log_base

    def _logit_scale(self, kv_len: int) -> float:
        a = self.nope_logit_log_base
        if a is None:
            return 1.0
        a = float(a)
        return math.log(a + max(kv_len, 1)) / math.log(a + 1.0)

    def forward(self, x, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x).view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq_t = xq.transpose(1, 2)
        xk_t = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv_t = repeat_kv(xv, self.n_rep).transpose(1, 2)
        scale = self._logit_scale(xk_t.size(-2))
        if scale != 1.0:
            xq_t = xq_t * scale
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(
                xq_t, xk_t, xv_t, dropout_p=self.dropout if self.training else 0.0, is_causal=True,
            )
        elif self.flash and seq_len == 1:
            output = F.scaled_dot_product_attention(
                xq_t, xk_t, xv_t, dropout_p=self.dropout if self.training else 0.0, is_causal=False,
            )
        else:
            scores = (xq_t @ xk_t.transpose(-2, -1)) / math.sqrt(self.head_dim)
            tq, tk = scores.shape[-2], scores.shape[-1]
            mask = torch.triu(
                torch.full((tq, tk), float("-inf"), device=scores.device, dtype=scores.dtype),
                diagonal=tk - tq + 1,
            )
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq_t)
            scores = self.attn_dropout(scores)
            output = scores @ xv_t
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class PRoPESlidingAttention(nn.Module):
    def __init__(self, args: MiniWinConfig):
        super().__init__()
        self.num_key_value_heads = (
            args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        )
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(F, "scaled_dot_product_attention") and args.flash_attn
        self.sliding_span = max(1, args.sliding_window_size)
        self.window_left = self.sliding_span - 1
        self.pe_type = args.sliding_pe_type
        if self.pe_type == "rope":
            self.rope_dim = rope_dim_from_factor(self.head_dim, getattr(args, "partial_rotary_factor", 1.0))
            hd = args.hidden_size // args.num_attention_heads
            fc, fs = precompute_freqs_cis(dim=hd, end=args.sliding_window_size, rope_base=args.rope_theta)
            self.register_buffer("_freqs_cos", fc, persistent=False)
            self.register_buffer("_freqs_sin", fs, persistent=False)
        else:
            self.rope_dim = 0

    def _forward_prefill(self, x, position_embeddings, alibi_slopes, use_cache):
        bsz, seq_len, _ = x.shape
        hd = self.head_dim
        nh = self.n_local_heads
        nkv = self.n_local_kv_heads
        xq = self.q_proj(x).view(bsz, seq_len, nh, hd)
        xk = self.k_proj(x).view(bsz, seq_len, nkv, hd)
        xv = self.v_proj(x).view(bsz, seq_len, nkv, hd)
        if self.pe_type == "rope":
            cos, sin = position_embeddings
            if cos.dim() == 2:
                cos_s, sin_s = cos[:seq_len], sin[:seq_len]
                unsq = 1
            else:
                cos_s, sin_s = cos, sin
                unsq = 2
            xq_s, xk_s = apply_partial_rotary_pos_emb(
                xq, xk, cos_s, sin_s, self.rope_dim, unsqueeze_dim=unsq,
            )
            xk_h = repeat_kv(xk_s, self.n_rep)
            xv_h = repeat_kv(xv, self.n_rep)
            out_slide = flash_attn_func_miniwin(
                xq_s, xk_h, xv_h, causal=True, window_size=(self.window_left, 0),
            )
        else:
            q_t = xq.transpose(1, 2)
            k_t = repeat_kv(xk, self.n_rep).transpose(1, 2)
            v_t = repeat_kv(xv, self.n_rep).transpose(1, 2)
            bias = build_sliding_alibi_bias(
                alibi_slopes, q_t.size(-2), k_t.size(-2), self.window_left,
                device=q_t.device, dtype=q_t.dtype,
            )
            if self.flash:
                out_slide = F.scaled_dot_product_attention(
                    q_t, k_t, v_t, attn_mask=bias,
                    dropout_p=self.dropout if self.training else 0.0, is_causal=False,
                )
            else:
                scores = (q_t @ k_t.transpose(-2, -1)) / math.sqrt(hd) + bias
                scores = F.softmax(scores.float(), dim=-1).type_as(q_t)
                scores = self.attn_dropout(scores)
                out_slide = scores @ v_t
            out_slide = out_slide.transpose(1, 2).contiguous()
        out_slide = out_slide.reshape(bsz, seq_len, nh * hd)
        merged = out_slide
        out = self.resid_dropout(self.o_proj(merged))
        present = (xk, xv) if use_cache else None
        return out, present

    def _forward_decode(self, x, position_embeddings, alibi_slopes, past_kv, use_cache):
        bsz = x.size(0)
        hd = self.head_dim
        nh = self.n_local_heads
        nkv = self.n_local_kv_heads
        k_cache, v_cache = past_kv
        xq = self.q_proj(x).view(bsz, 1, nh, hd)
        xk = self.k_proj(x).view(bsz, 1, nkv, hd)
        xv = self.v_proj(x).view(bsz, 1, nkv, hd)
        k_all = torch.cat([k_cache, xk], dim=1)
        v_all = torch.cat([v_cache, xv], dim=1)
        T = k_all.size(1)
        if self.pe_type == "rope":
            cos, sin = position_embeddings
            cos_q = cos[:1] if cos.dim() == 2 else cos
            sin_q = sin[:1] if sin.dim() == 2 else sin
            xq_s, _ = apply_partial_rotary_pos_emb(
                xq, xq, cos_q, sin_q, self.rope_dim, unsqueeze_dim=1,
            )
            W = self.sliding_span
            start = max(0, T - W)
            k_slide_raw = k_all[:, start:, :, :]
            v_slide = v_all[:, start:, :, :]
            slide_pos = torch.arange(start, T, device=x.device, dtype=torch.long) % W
            cos_k = self._freqs_cos[slide_pos]
            sin_k = self._freqs_sin[slide_pos]
            _, xk_slide = apply_partial_rotary_pos_emb(
                k_slide_raw, k_slide_raw, cos_k, sin_k, self.rope_dim, unsqueeze_dim=1,
            )
            xq_t = xq_s.transpose(1, 2)
            xk_t = repeat_kv(xk_slide, self.n_rep).transpose(1, 2)
            xv_t = repeat_kv(v_slide, self.n_rep).transpose(1, 2)
            if self.flash:
                out_slide = F.scaled_dot_product_attention(xq_t, xk_t, xv_t, is_causal=False)
            else:
                scores = (xq_t @ xk_t.transpose(-2, -1)) / math.sqrt(hd)
                scores = F.softmax(scores.float(), dim=-1).type_as(xq_t)
                out_slide = scores @ xv_t
            out_slide = out_slide.transpose(1, 2)
        else:
            W = self.sliding_span
            start = max(0, T - W)
            q_t = xq.transpose(1, 2)
            k_t = repeat_kv(k_all[:, start:], self.n_rep).transpose(1, 2)
            v_t = repeat_kv(v_all[:, start:], self.n_rep).transpose(1, 2)
            bias = build_sliding_alibi_bias(
                alibi_slopes, 1, k_t.size(-2), self.window_left,
                device=q_t.device, dtype=q_t.dtype,
            )
            if self.flash:
                out_slide = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=bias, is_causal=False)
            else:
                scores = (q_t @ k_t.transpose(-2, -1)) / math.sqrt(hd) + bias
                scores = F.softmax(scores.float(), dim=-1).type_as(q_t)
                out_slide = scores @ v_t
            out_slide = out_slide.transpose(1, 2).contiguous()
        out_slide = out_slide.reshape(bsz, 1, nh * hd)
        merged = out_slide
        out = self.resid_dropout(self.o_proj(merged))
        present = (k_all, v_all) if use_cache else None
        return out, present

    def forward(self, x, position_embeddings=None, past_key_value=None,
                use_cache=False, attention_mask=None, alibi_slopes=None, start_pos=0):
        if x.size(1) > 1 or past_key_value is None:
            return self._forward_prefill(x, position_embeddings, alibi_slopes, use_cache)
        return self._forward_decode(x, position_embeddings, alibi_slopes, past_key_value, use_cache)


class MiniWinBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniWinConfig):
        super().__init__()
        self.layer_id = layer_id
        self.layer_mode = layer_attention_mode(config, layer_id)
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        if self.layer_mode == "S":
            self.shared_attn = PRoPESlidingAttention(config)
        else:
            self.self_attn = NoPEGlobalAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings=None, past_key_value=None,
                use_cache=False, attention_mask=None, alibi_slopes=None, start_pos=0):
        if self.layer_mode == "L":
            return self._forward_global(hidden_states, past_key_value, use_cache, attention_mask)
        return self._forward_swan_s(
            hidden_states, position_embeddings, past_key_value,
            use_cache, attention_mask, alibi_slopes, start_pos,
        )

    def _forward_global(self, hidden_states, past_key_value, use_cache, attention_mask):
        residual = hidden_states
        norm_h = self.input_layernorm(hidden_states)
        main_past = None
        if past_key_value is not None and isinstance(past_key_value, tuple):
            if len(past_key_value) == 2:
                first = past_key_value[0]
                if isinstance(first, torch.Tensor):
                    main_past = past_key_value
                elif first is not None:
                    main_past = first
        hidden, main_present = self.self_attn(
            norm_h, past_key_value=main_past, use_cache=use_cache, attention_mask=attention_mask,
        )
        hidden = hidden + residual
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        layer_present = (main_present, None) if use_cache else None
        return hidden, layer_present

    def _forward_swan_s(self, hidden_states, position_embeddings, past_key_value,
                        use_cache, attention_mask, alibi_slopes, start_pos):
        residual = hidden_states
        norm_h = self.input_layernorm(hidden_states)
        hidden, layer_present = self.shared_attn(
            norm_h, position_embeddings=position_embeddings,
            past_key_value=past_key_value, use_cache=use_cache,
            attention_mask=attention_mask, alibi_slopes=alibi_slopes, start_pos=start_pos,
        )
        hidden = hidden + residual
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden, layer_present


class MiniWin(nn.Module):
    def __init__(self, config: MiniWinConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniWinBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.sliding_pe_type == "rope":
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=config.hidden_size // config.num_attention_heads,
                end=config.sliding_window_size,
                rope_base=config.rope_theta,
            )
            self.register_buffer("freqs_cos", freqs_cos, persistent=False)
            self.register_buffer("freqs_sin", freqs_sin, persistent=False)
            self.register_buffer("alibi_slopes", torch.empty(0), persistent=False)
        else:
            self.register_buffer("freqs_cos", torch.empty(0), persistent=False)
            self.register_buffer("freqs_sin", torch.empty(0), persistent=False)
            self.register_buffer(
                "alibi_slopes", get_alibi_slopes(config.num_attention_heads), persistent=False,
            )

    def _resolve_start_pos(self, past_key_values: List) -> int:
        for pk in past_key_values:
            if pk is None:
                continue
            if isinstance(pk, tuple) and len(pk) == 2:
                first = pk[0]
                if isinstance(first, torch.Tensor):
                    return first.shape[1]
                if first is not None and isinstance(first, tuple):
                    return first[0].shape[1]
        return 0

    def _position_embeddings(self, seq_length, start_pos, device):
        if self.config.sliding_pe_type == "rope":
            W = self.config.sliding_window_size
            if self.freqs_cos.numel() == 0 or self.freqs_cos[0, 0] == 0:
                hd = self.config.hidden_size // self.config.num_attention_heads
                freqs_cos, freqs_sin = precompute_freqs_cis(
                    dim=hd, end=self.config.sliding_window_size, rope_base=self.config.rope_theta,
                )
                self.freqs_cos, self.freqs_sin = freqs_cos.to(device), freqs_sin.to(device)
            if seq_length > 1:
                local_ids = torch.arange(seq_length, device=device, dtype=torch.long) % W
            else:
                local_ids = torch.tensor([start_pos % W], device=device, dtype=torch.long)
            return (self.freqs_cos[local_ids], self.freqs_sin[local_ids]), None
        return None, self.alibi_slopes

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = self._resolve_start_pos(past_key_values)
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        position_embeddings, alibi_slopes = self._position_embeddings(
            seq_length, start_pos, hidden_states.device,
        )
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
                alibi_slopes=alibi_slopes,
                start_pos=start_pos,
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        _moe_aux = [
            layer.mlp.aux_loss for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        ]
        aux_loss = sum(_moe_aux) if _moe_aux else hidden_states.new_zeros((), dtype=hidden_states.dtype)
        return hidden_states, presents, aux_loss


class MiniWinForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniWinConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MiniWinConfig = None):
        self.config = config or MiniWinConfig()
        super().__init__(self.config)
        self.model = MiniWin(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight

    def _init_weights(self, module):
        pass

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, use_cache=False,
                logits_to_keep: Union[int, torch.Tensor] = 0, labels=None, **args):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, use_cache=use_cache, **args,
        )
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x = logits[..., :-1, :].contiguous()
            y = labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        output = CausalLMOutputWithPast(
            loss=loss, logits=logits,
            past_key_values=past_key_values, hidden_states=hidden_states,
        )
        output.aux_loss = aux_loss
        return output
