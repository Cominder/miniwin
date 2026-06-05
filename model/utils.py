import math
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import os

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def print_banner():
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    banner = """
                  ███              ███                   ███            
                 ░░░              ░░░                   ░░░             
 █████████████   ████  ████████   ████  █████ ███ █████ ████  ████████  
░░███░░███░░███ ░░███ ░░███░░███ ░░███ ░░███ ░███░░███ ░░███ ░░███░░███ 
 ░███ ░███ ░███  ░███  ░███ ░███  ░███  ░███ ░███ ░███  ░███  ░███ ░███ 
 ░███ ░███ ░███  ░███  ░███ ░███  ░███  ░░███████████   ░███  ░███ ░███ 
 █████░███ █████ █████ ████ █████ █████  ░░████░████    █████ ████ █████
░░░░░ ░░░ ░░░░░ ░░░░░ ░░░░ ░░░░░ ░░░░░    ░░░░ ░░░░    ░░░░░ ░░░░ ░░░░░ 
                                                                        
                                                                        
                                                                                                                                         
    """
    print0(banner)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def rope_dim_from_factor(head_dim: int, partial_rotary_factor: float) -> int:
    """仅前 rope_dim 维做 RoPE，其余为 nope。factor=1.0 表示全头 RoPE。"""
    if partial_rotary_factor >= 1.0 - 1e-9:
        return head_dim
    d = int(head_dim * float(partial_rotary_factor))
    return max(0, min(head_dim, d))


def apply_partial_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
    unsqueeze_dim: int = 1,
):
    """只对 q/k 的最后维前 rope_dim 应用 RoPE，其余保持不变。"""
    if rope_dim <= 0:
        return q, k
    hd = q.shape[-1]
    if rope_dim >= hd:
        return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
    q_r, q_n = q.split([rope_dim, hd - rope_dim], dim=-1)
    k_r, k_n = k.split([rope_dim, hd - rope_dim], dim=-1)
    cos_r = cos[..., :rope_dim]
    sin_r = sin[..., :rope_dim]
    q_r, k_r = apply_rotary_pos_emb(q_r, k_r, cos_r, sin_r, unsqueeze_dim=unsqueeze_dim)
    return torch.cat([q_r, q_n], dim=-1), torch.cat([k_r, k_n], dim=-1)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

def _manual_sliding_attention_bh_td(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_left: int) -> torch.Tensor:
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    B, H, Tq, Tk = scores.shape[0], scores.shape[1], scores.shape[2], scores.shape[3]
    w = window_left
    device = q.device
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    causal = col_idx <= row_idx
    if w >= 0 and w < Tk:
        mask = causal & ((row_idx - col_idx) <= w)
    else:
        mask = causal
    scores = scores.masked_fill(~mask.view(1, 1, Tq, Tk), float("-inf"))
    attn = F.softmax(scores.float(), dim=-1).type_as(q)
    return attn @ v

def _sdpa_sliding_window_bh_td(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_left: int) -> torch.Tensor:
    if not hasattr(F, "scaled_dot_product_attention"):
        return _manual_sliding_attention_bh_td(q, k, v, window_left)

    Tq = q.size(2)
    Tk = k.size(2)
    w = window_left
    enable_gqa = q.size(1) != k.size(1)

    def sdpa_call(**kwargs):
        try:
            return F.scaled_dot_product_attention(q, k, v, enable_gqa=enable_gqa, **kwargs)
        except TypeError:
            return F.scaled_dot_product_attention(q, k, v, **kwargs)

    if (w < 0 or w >= Tq) and Tq == Tk:
        return sdpa_call(is_causal=True)
    if Tq == 1:
        if w >= 0 and w < Tk:
            start = max(0, Tk - (w + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return sdpa_call(is_causal=False)
    device = q.device
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx
    if w >= 0 and w < Tk:
        mask = mask & ((row_idx - col_idx) <= w)
    return sdpa_call(attn_mask=mask)

def flash_attn_func_miniwin(
    q_bthd: torch.Tensor,
    k_bthd: torch.Tensor,
    v_bthd: torch.Tensor,
    causal: bool,
    window_size: Tuple[int, int],
) -> torch.Tensor:
    try:
        from .flash_attention import flash_attn as miniwin_flash

        return miniwin_flash.flash_attn_func(q_bthd, k_bthd, v_bthd, causal=causal, window_size=window_size)
    except ImportError:
        left, _ = window_size
        q = q_bthd.transpose(1, 2)
        k = k_bthd.transpose(1, 2)
        v = v_bthd.transpose(1, 2)
        y = _sdpa_sliding_window_bh_td(q, k, v, left)
        return y.transpose(1, 2)


def rtr(hidden_states: torch.Tensor, n_windows: int) -> torch.Tensor:
    
    batch, hidden_len, hidden_dim = hidden_states.shape
    if n_windows == 1:
        return hidden_states
    hidden_states = hidden_states.reshape(batch, -1, n_windows, hidden_dim).transpose(1, 2)
    hidden_states = hidden_states.reshape(batch, -1, hidden_dim)
    return hidden_states

def rtr_inverse(hidden_states: torch.Tensor, n_windows: int) -> torch.Tensor:
    
    batch, hidden_len, hidden_dim = hidden_states.shape
    if n_windows == 1:
        return hidden_states
    hidden_states = hidden_states.reshape(batch, n_windows, -1, hidden_dim).transpose(1, 2)
    hidden_states = hidden_states.reshape(batch, -1, hidden_dim)
    return hidden_states


def rtr_positions(pos_ids: torch.Tensor, n_windows: int) -> torch.Tensor:
    if n_windows == 1:
        return pos_ids
    batch, n = pos_ids.shape
    return pos_ids.reshape(batch, -1, n_windows).transpose(1, 2).reshape(batch, -1)


def window_partition_positions(pos_ids: torch.Tensor, n_window: int) -> torch.Tensor:
    B, N = pos_ids.shape
    x = pos_ids.view(B, n_window, N // n_window)
    return x.reshape(-1, N // n_window)

def window_partition(x, n_window: int):
    """
    Args:
        x: (B, N, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, N, C = x.shape
    x = x.view(B, n_window, N // n_window, C)
    windows = x.view(-1, N // n_window, C)
    return windows

def window_reverse(windows, n_window: int):
    """
    Args:
        windows: (num_windows*B, window_size, C)
        window_size (int): Window size
        N (int): Length of sequence

    Returns:
        x: (B, N, C)
    """
    B, N, C = windows.shape
    x = windows.view(B//n_window, n_window * N, C)
    return x


def _unpack_layer_past(past_key_value):
    if past_key_value is None:
        return None, None
    if isinstance(past_key_value[0], torch.Tensor):
        return past_key_value, None
    return past_key_value[0], past_key_value[1]






