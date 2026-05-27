# Copyright (c) 2024-2025 Microsoft
# Copyright (c) 2026 (NPU adaptation)
# Licensed under The MIT License [see LICENSE for details]
"""Streaming (A-shape) attention via Ascend hardware band FA + sink dense + LSE merge.

把上游 MInference 1.0 `ops/streaming_kernel.py:streaming_forward` 移植到昇腾 NPU。
算法语义：每个 query 位置 ``i`` 在 K 中的可见集合为：

* **sink**:    ``K[..., :n_init, :]`` — 前 ``n_init`` 个 token，对所有 query 都可见
* **sliding**: ``K[..., i_abs - n_local + 1 : i_abs + 1, :]`` — 以当前 query 绝对位置
  为右端、宽 ``n_local`` 的滑动窗口（``i_abs = k_len - q_len + i``）

主路径 = 两段硬件 native FA + LSE 合并：

1. Sliding window: ``npu_fusion_attention(sparse_mode=4, pre_tockens=n_local-1, next_tockens=0)``
   —— Ascend 硬件 band attention，kernel 级真稀疏，只算 band 内的 token。
2. Sink: ``npu_fusion_attention(sparse_mode=1, atten_mask=...)`` 但 K_len=n_init 极小，
   bool mask 形状 ``[S_q, n_init]`` 不爆炸。Mask 排除与 sliding window 的重叠位置。
3. LSE merge: 用两段返回的 ``softmax_max`` / ``softmax_sum`` 在线合并 softmax 概率。

**和已弃用的 path A 的区别**：path A = ``sparse_mode=1 + bool mask`` 把整个 dense QK
matmul 算完再 mask 出 -inf；本路径的 pass1 走 ``sparse_mode=4`` 是硬件 band，**只算
sliding window 内的 FLOPs**。pass2 sink 的 K 极小（n_init 通常 ≤ 256），bool mask 完全
可承受。整体 FLOPs ≈ O(S × (n_init + n_local))，与上游 streaming-LLM 一致。

**和已弃用的 TileLang path-B 的区别**：TileLang sparse_attention_fwd 是通用
"per-token gather" 稀疏 FA，对位置确定的 streaming 场景是 over-engineering：
(1) kernel ``kv_group=1, heads=1`` MVP 强制 wrapper 把 B*H 折叠到 batch 维，内部
``H_per_block=16`` 导致 15× 算力浪费；
(2) K 是按 token 散读，吃不到 HBM burst 带宽；
(3) 6 calls × per-layer JIT/dispatch 开销可观。
端到端 4K stream probe 比 dense 慢 7.5×（稳态），是设计错配，本次彻底丢弃。

签名严格对齐上游：

    streaming_forward(q, k, v, n_init, n_local) -> Tensor  # [B, H, S, D]

输入约定：q/k/v 形状 ``[B, H, S, D]``（BNSD），已 RoPE、已 ``repeat_kv``。
"""

from __future__ import annotations

import math
from typing import Optional

import torch

try:
    import torch_npu  # type: ignore[import-not-found]

    _HAS_TORCH_NPU = True
except ImportError:  # pragma: no cover — 非 NPU 环境
    torch_npu = None  # type: ignore[assignment]
    _HAS_TORCH_NPU = False


__all__ = ["streaming_forward"]


_COMPRESSED_MASK_SIZE = 2048
_COMPRESSED_CAUSAL_MASK_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


# ----------------------------------------------------------------------------
# head_dim 对齐（与上游一致）
# ----------------------------------------------------------------------------

# `npu_fusion_attention` 要求 head_dim ∈ {16,32,64,128,256,512}；其他取值需要 pad
# 到 2 的幂。推理完成后再截回原始 head_dim。
_ALLOWED_HEAD_DIMS = (16, 32, 64, 128, 256, 512)


def _pad_head_dim_to_pow2(t: torch.Tensor, target_d: int) -> torch.Tensor:
    cur_d = t.shape[-1]
    if cur_d == target_d:
        return t
    return torch.nn.functional.pad(t, (0, target_d - cur_d, 0, 0, 0, 0, 0, 0))


def _next_allowed_head_dim(d: int) -> int:
    for cand in _ALLOWED_HEAD_DIMS:
        if cand >= d:
            return cand
    raise ValueError(
        f"head_dim={d} 超过支持的最大值 {_ALLOWED_HEAD_DIMS[-1]}；MInference 上游同样不支持"
    )


# ----------------------------------------------------------------------------
# PyTorch 黄金参考（测试与非 NPU 兜底共用）
# ----------------------------------------------------------------------------


def _streaming_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_init: int,
    n_local: int,
    *,
    chunk_size_q: Optional[int] = None,
) -> torch.Tensor:
    """Sink + sliding-window 注意力的纯 PyTorch 实现。

    用作：
    1. NPU 不可用时（CPU / CUDA / import 失败）的兜底
    2. 单测的"黄金参考"

    沿 q 维分块以避免长上下文场景的 mask 内存爆炸：mask 形状 ``[chunk, S_k]``，把
    ``chunk * S_k`` 控制在 ~4M 元素以内。
    """
    bsz, n_heads, s_q, head_d = q.shape
    s_k = k.shape[2]
    scale = 1.0 / math.sqrt(head_d)

    if chunk_size_q is None:
        chunk_size_q = max(64, min(s_q, (4 * 1024 * 1024) // max(1, s_k)))

    j_all = torch.arange(s_k, device=q.device)

    out = torch.empty_like(q)
    in_dtype = q.dtype

    for chunk_start in range(0, s_q, chunk_size_q):
        chunk_end = min(chunk_start + chunk_size_q, s_q)

        abs_i = torch.arange(
            s_k - s_q + chunk_start, s_k - s_q + chunk_end, device=q.device
        )

        sink = j_all[None, :] < n_init
        sliding = (j_all[None, :] <= abs_i[:, None]) & (
            j_all[None, :] > abs_i[:, None] - n_local
        )
        causal = j_all[None, :] <= abs_i[:, None]
        visible = (sink | sliding) & causal

        q_chunk = q[:, :, chunk_start:chunk_end, :]
        logits = torch.matmul(q_chunk.float(), k.float().transpose(-2, -1)) * scale
        logits = logits.masked_fill(~visible[None, None, :, :], float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        out_chunk = torch.matmul(probs, v.float())
        out[:, :, chunk_start:chunk_end, :] = out_chunk.to(in_dtype)

    return out


# ----------------------------------------------------------------------------
# NPU 主路径：band + sink + LSE merge
# ----------------------------------------------------------------------------


def _take_first_lane(stat: torch.Tensor) -> torch.Tensor:
    """``npu_fusion_attention`` 返回的 ``softmax_max`` / ``softmax_sum`` 通常是
    ``[B, H, S, 8]``（最后维 8 是 tile 对齐填充，每行实际 max/sum 在各 lane 复制）。
    取第 0 个 lane 并保留维度，返回 ``[B, H, S, 1]`` 方便对 ``[B, H, S, D]`` 广播。
    """
    if stat.dim() == 4:
        return stat[..., :1]
    if stat.dim() == 3:
        return stat.unsqueeze(-1)
    raise RuntimeError(
        f"unexpected softmax stat shape {tuple(stat.shape)}; "
        "需要适配新的 torch_npu npu_fusion_attention 返回格式"
    )


def _unpack_fa_result(result, *, where: str):
    """校验 ``npu_fusion_attention`` 返回 ``(output, softmax_max, softmax_sum, ...)``。"""
    if not isinstance(result, (tuple, list)) or len(result) < 3:
        kind = type(result).__name__
        size = len(result) if hasattr(result, "__len__") else "n/a"
        raise RuntimeError(
            f"npu_fusion_attention at {where} 返回 {kind} (len={size})；"
            " streaming kernel 需要 (output, softmax_max, softmax_sum, ...) "
            "才能做跨段 LSE 合并。"
        )
    return result[0], _take_first_lane(result[1]), _take_first_lane(result[2])


def _compressed_causal_mask(device: torch.device) -> torch.Tensor:
    key = (device, torch.bool)
    cached = _COMPRESSED_CAUSAL_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    mask = torch.triu(
        torch.ones((_COMPRESSED_MASK_SIZE, _COMPRESSED_MASK_SIZE), device=device, dtype=torch.bool),
        diagonal=1,
    )
    _COMPRESSED_CAUSAL_MASK_CACHE[key] = mask
    return mask


def _call_band_fa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    n_heads: int,
    scale: float,
    pre_tokens: int,
    next_tokens: int,
):
    """调用 ``npu_fusion_attention`` 的 sparse_mode=4 band 路径。

    Ascend API 历史拼写是 ``pre_tockens`` / ``next_tockens``（typo 保留），新版部分
    torch_npu 同时接受 ``pre_tokens`` / ``next_tokens``。两个名字都尝试一次。
    """
    common = dict(
        head_num=n_heads,
        input_layout="BNSD",
        scale=scale,
        sparse_mode=4,
        atten_mask=_compressed_causal_mask(q.device),
    )
    try:
        return torch_npu.npu_fusion_attention(  # type: ignore[union-attr]
            q, k, v,
            pre_tockens=pre_tokens,
            next_tockens=next_tokens,
            **common,
        )
    except TypeError:
        return torch_npu.npu_fusion_attention(  # type: ignore[union-attr]
            q, k, v,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            **common,
        )


def _streaming_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_init: int,
    n_local: int,
) -> torch.Tensor:
    """Hardware-native band FA + sink dense + LSE merge.

    本函数假定调用前已经：
    * ``q.device.type == "npu"`` 且 ``torch_npu`` 可用
    * head_dim 已 pad 至 ``_ALLOWED_HEAD_DIMS`` 内
    * ``k_len > n_local``（``k_len <= n_local`` 在外层短路为 causal dense）

    返回 ``[B, H, S_q, D_padded]``；外层 ``streaming_forward`` 把 head_dim 截回原值。
    """
    assert _HAS_TORCH_NPU, "_streaming_npu 不能在非 NPU 环境调用（外层应已分流）"

    bsz, n_heads, s_q, head_d = q.shape
    s_k = k.shape[2]
    scale = 1.0 / math.sqrt(head_d)

    if s_q > s_k:
        raise ValueError(f"streaming attention requires S_q <= S_k, got S_q={s_q}, S_k={s_k}")

    # --- Pass 1: sliding window via hardware band FA (sparse_mode=4) ---
    # sparse_mode=4 按 query row id 对齐 key row id。对于 S_q < S_k 的增量/后缀
    # 形态，在 Q 前补 dummy rows，把真实 query row i 移到 abs_i = S_k-S_q+i。
    pre_tokens = max(int(n_local) - 1, 0)
    next_tokens = 0
    q_band = q
    band_offset = s_k - s_q
    if band_offset:
        q_prefix = q.new_zeros((bsz, n_heads, band_offset, head_d))
        q_band = torch.cat((q_prefix, q), dim=2).contiguous()
    pass1 = _call_band_fa(
        q_band, k, v,
        n_heads=n_heads,
        scale=scale,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
    )
    o1, m1, l1 = _unpack_fa_result(pass1, where="streaming pass1 (band)")
    if band_offset:
        o1 = o1[:, :, band_offset:, :]
        m1 = m1[:, :, band_offset:, :]
        l1 = l1[:, :, band_offset:, :]
    # 早期 query 在极端情况下可能整行被 band 屏蔽（abs_i < 0 不会发生于 prefill，
    # 但 n_local 极小且早 query 时可能 l1 == 0）；nan_to_num 防御 LSE 合并出 NaN。
    l1 = torch.nan_to_num(l1, nan=0.0)

    if int(n_init) <= 0:
        return o1

    # --- Pass 2: sink only，mask 出与 sliding 的重叠 ---
    n_init_clamped = min(int(n_init), s_k)
    k_sink = k[:, :, :n_init_clamped, :].contiguous()
    v_sink = v[:, :, :n_init_clamped, :].contiguous()

    # 对 query i（abs_i = s_k - s_q + i），允许的 sink key j 满足：
    #   j < n_init_clamped  AND  j NOT IN sliding window
    # 等价 j < max(0, abs_i - n_local + 1)。当 abs_i < n_local - 1 时整行无可见。
    abs_i = torch.arange(s_k - s_q, s_k, device=q.device)
    j = torch.arange(n_init_clamped, device=q.device)
    allowed = j[None, :] < (abs_i[:, None] - int(n_local) + 1)
    # NPU atten_mask 约定：True / 1 = 屏蔽
    attn_mask = (~allowed).to(torch.bool)[None, None, :, :]

    try:
        pass2 = torch_npu.npu_fusion_attention(  # type: ignore[union-attr]
            q, k_sink, v_sink,
            head_num=n_heads,
            input_layout="BNSD",
            scale=scale,
            sparse_mode=1,
            atten_mask=attn_mask,
        )
    except TypeError:
        pass2 = torch_npu.npu_fusion_attention(  # type: ignore[union-attr]
            q, k_sink, v_sink,
            head_num=n_heads,
            input_layout="BNSD",
            scale=scale,
            sparse_mode=1,
            atten_mask=attn_mask.to(torch.uint8),
        )
    o2, m2, l2 = _unpack_fa_result(pass2, where="streaming pass2 (sink)")
    l2 = torch.nan_to_num(l2, nan=0.0)

    # --- 跨段 log-sum-exp 合并 ---
    # NPU FA 返回：m = max(scaled_logits)，l = sum(exp(scaled_logits - m))。
    # 合并：
    #   m_new = max(m1, m2)
    #   alpha_k = exp(m_k - m_new) * l_k
    #   o_new = (alpha1 * o1 + alpha2 * o2) / (alpha1 + alpha2)
    # 当某段整行被屏蔽 (l == 0)，把对应 m clamp 到一个有限大负数，避免 -inf - (-inf) = NaN。
    neg_big = torch.finfo(m1.dtype).min / 4
    m2_eff = torch.where(l2 > 0, m2, torch.full_like(m2, neg_big))
    m1_eff = torch.where(l1 > 0, m1, torch.full_like(m1, neg_big))
    m_new = torch.maximum(m1_eff, m2_eff)

    alpha1 = torch.exp(m1_eff - m_new) * l1
    alpha2 = torch.exp(m2_eff - m_new) * l2
    denom = alpha1 + alpha2
    safe = denom > 0
    out_combined = (alpha1 * o1.float() + alpha2 * o2.float()) / denom
    out_combined = torch.where(safe, out_combined, o1.float())

    return out_combined.to(q.dtype)


# ----------------------------------------------------------------------------
# 顶层入口
# ----------------------------------------------------------------------------


def streaming_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_init: int,
    n_local: int,
) -> torch.Tensor:
    """Sink + sliding-window 注意力（A-shape / streaming-LLM）。

    Args:
        q, k, v: ``[B, H, S, D]``，已 RoPE、已 ``repeat_kv``。
        n_init:  sink 段长度（前 N 个 token 永远可见）。<=0 表示无 sink。
        n_local: sliding window 段长度。

    Returns:
        ``[B, H, S, D]``，与 ``q`` 同 dtype 同 device。

    短路：
    * ``k_len <= n_local``：等价于 causal dense。
    """
    assert q.dim() == 4, f"q 期望 4D [B,H,S,D]，得到 {q.dim()}D"
    assert (
        q.shape[0] == k.shape[0] and q.shape[1] == k.shape[1] and q.shape[-1] == k.shape[-1]
    ), f"q/k B/H/D 不匹配；q={tuple(q.shape)} k={tuple(k.shape)}"

    head_d = q.shape[-1]
    orig_head_d = head_d

    if head_d not in _ALLOWED_HEAD_DIMS:
        target_d = _next_allowed_head_dim(head_d)
        q = _pad_head_dim_to_pow2(q, target_d)
        k = _pad_head_dim_to_pow2(k, target_d)
        v = _pad_head_dim_to_pow2(v, target_d)

    s_k = k.shape[2]

    # 短路：sliding window 覆盖所有 K → causal dense
    if s_k <= int(n_local):
        from ..backend_npu import dense_attention

        out = dense_attention(q, k, v, causal=True)
        return out[..., :orig_head_d]

    if q.device.type == "npu" and _HAS_TORCH_NPU:
        out = _streaming_npu(q, k, v, int(n_init), int(n_local))
    else:
        out = _streaming_pytorch_ref(q, k, v, int(n_init), int(n_local))

    return out[..., :orig_head_d]
