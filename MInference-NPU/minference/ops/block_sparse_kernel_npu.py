# Copyright (c) 2024-2025 Microsoft
# Copyright (c) 2026 (NPU 适配 — M3 block-sparse kernel)
# Licensed under The MIT License [see LICENSE for details]
"""Block-sparse attention — NPU v1 实现（路径 A：npu_fusion_attention + bool mask）。

算法语义：把 KV 序列按固定 ``block_size`` 切块，每个 query block 只与得分最高的
``topk_blocks`` 个 key block 做 attention（因果约束：key block index ≤ query block index）。
块得分由平均池化的 Q×K 近似计算，token 级因果约束额外施加。

NPU 实现（v1 PoC，见 ``docs/migration_plan_v1.md §4`` 路径 A）：
  1. 平均池化 Q/K → block representatives
  2. block 级 QK 打分 + 因果 top-k 选 key blocks
  3. 展开为 token 级 bool mask ``[B, H, S_q, S_k]``
  4. 追加 per-token 因果约束（保证对角块内正确性）
  5. ``npu_fusion_attention(sparse_mode=1, atten_mask=mask)``

序列长度限制：mask 构建是 O(S_q × S_k)；当 ``max(S_q, S_k) > _MAX_SEQ_FOR_MASK`` 时
退化为 dense attention 并打印 WARNING。对 128k+ 超长上下文请改用路径 B（TileLang-Ascend
Triton kernel，见 ``docs/M3_block_sparse.md §6``）。

非 NPU 路径：同样构建 block-sparse mask，用纯 PyTorch masked softmax 实现，作为
测试黄金参考。两条路径共用 ``_build_block_sparse_mask``，仅 attention 计算步骤不同。

签名与上游 ``MInference/minference/ops/block_sparse_flash_attention.py:block_sparse_attention``
对齐（参数名做了语义化重命名）：

    block_sparse_attention(q, k, v, topk_blocks, block_size=64) -> Tensor  # [B, H, S, D]

输入约定：q/k/v 形状 ``[B, H, S, D]``（BNSD），已 RoPE、已 ``repeat_kv``（与上游一致）。
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import torch_npu  # type: ignore[import-not-found]

    _HAS_TORCH_NPU = True
except ImportError:  # pragma: no cover
    torch_npu = None  # type: ignore[assignment]
    _HAS_TORCH_NPU = False


__all__ = ["block_sparse_attention"]


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# npu_fusion_attention 要求 head_dim ∈ 此集合；否则需要 pad
_ALLOWED_HEAD_DIMS = (16, 32, 64, 128, 256, 512)

# bool mask 构建是 O(S²)；超过此阈值退化为 dense 并发 WARNING
# 路径 B（TileLang-Ascend）可覆盖更长序列
_MAX_SEQ_FOR_MASK = 16384

# 默认 block size（与上游 block_sparse_flash_attention.py 一致）
_DEFAULT_BLOCK_SIZE = 64


# ---------------------------------------------------------------------------
# head_dim 工具
# ---------------------------------------------------------------------------


def _next_allowed_head_dim(d: int) -> int:
    for c in _ALLOWED_HEAD_DIMS:
        if c >= d:
            return c
    raise ValueError(
        f"head_dim={d} 超过支持的最大值 {_ALLOWED_HEAD_DIMS[-1]}；MInference 上游同样不支持"
    )


def _pad_head_dim(t: torch.Tensor, target_d: int) -> torch.Tensor:
    cur = t.shape[-1]
    if cur == target_d:
        return t
    return F.pad(t, (0, target_d - cur))


# ---------------------------------------------------------------------------
# block-sparse mask 构建（NPU 路径与 PyTorch 路径共用）
# ---------------------------------------------------------------------------


def _select_block_sparse_topk_indices(
    q: torch.Tensor,
    k: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """返回每个 query block 选中的 K block 索引，形状 ``[B,H,n_bq,topk]``。"""
    B, H, S_q, D = q.shape
    S_k = k.shape[2]

    pad_q = (-S_q) % block_size  # 0 if S_q % block_size == 0
    pad_k = (-S_k) % block_size
    S_q_p = S_q + pad_q
    S_k_p = S_k + pad_k
    n_bq = S_q_p // block_size
    n_bk = S_k_p // block_size

    q_p = F.pad(q, (0, 0, 0, pad_q)) if pad_q else q  # [B, H, S_q_p, D]
    k_p = F.pad(k, (0, 0, 0, pad_k)) if pad_k else k  # [B, H, S_k_p, D]

    q_pool = q_p.reshape(B, H, n_bq, block_size, D).mean(dim=3).float()  # [B, H, n_bq, D]
    k_pool = k_p.reshape(B, H, n_bk, block_size, D).mean(dim=3).float()  # [B, H, n_bk, D]

    scale = D ** -0.5
    scores = torch.matmul(q_pool, k_pool.transpose(-2, -1)) * scale  # [B, H, n_bq, n_bk]

    bq_idx = torch.arange(n_bq, device=q.device)
    bk_idx = torch.arange(n_bk, device=q.device)
    causal_block = bq_idx[:, None] >= bk_idx[None, :]  # [n_bq, n_bk]
    scores.masked_fill_(~causal_block[None, None], float("-inf"))

    topk = min(topk_blocks, n_bk)
    return torch.topk(scores, topk, dim=-1).indices  # [B, H, n_bq, topk]


def _build_block_sparse_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """构建 block-sparse attention 的 token 级 bool mask。

    返回形状 ``[B, H, S_q, S_k]``，``True`` = 被遮蔽（不参与 attention），即 NPU 惯例。

    算法：
    1. 将 Q/K 按 block_size pad 并 mean-pool 到 block 级。
    2. 计算 block 级 QK 得分 + 施加因果约束（key block index <= query block index）。
    3. 每个 query block 选 top-k key blocks。
    4. 展开到 token 级（block_attend → token_attend）。
    5. 追加 per-token 因果约束（消除 top-k -inf tie-breaking 引入的误差）。

    注意：per-token 因果约束是最终正确性的保证；block 级 top-k 中的 -inf tie-breaking
    即使选中了超前 key block，也会被 per-token 因果约束过滤掉。
    """
    B, H, S_q, D = q.shape
    S_k = k.shape[2]
    topk_idx = _select_block_sparse_topk_indices(q, k, topk_blocks, block_size)

    # --- 1. 展开 block mask 到 token 级 [B, H, S_q_p, S_k_p] ---
    pad_q = (-S_q) % block_size  # 0 if S_q % block_size == 0
    pad_k = (-S_k) % block_size
    S_q_p = S_q + pad_q
    S_k_p = S_k + pad_k
    n_bq = S_q_p // block_size
    n_bk = S_k_p // block_size

    block_attend = torch.zeros(B, H, n_bq, n_bk, dtype=torch.bool, device=q.device)
    block_attend.scatter_(-1, topk_idx, True)

    # 数学验证：reshape 后 token[b,h,bq*bs+bqi, bk*bs+bki] = block_attend[b,h,bq,bk] ✓
    token_attend = (
        block_attend.unsqueeze(3).unsqueeze(5)            # [B, H, n_bq, 1, n_bk, 1]
        .expand(-1, -1, -1, block_size, -1, block_size)  # [B, H, n_bq, bs, n_bk, bs]
        .reshape(B, H, S_q_p, S_k_p)                     # [B, H, S_q_p, S_k_p]
    )
    token_attend = token_attend[:, :, :S_q, :S_k]  # trim padding

    # --- 2. 追加 per-token 因果约束 ---
    # query token i 的绝对位置 = (S_k - S_q) + i；可见 key token j 满足 j <= abs_i
    abs_i = torch.arange(S_k - S_q, S_k, device=q.device)  # [S_q]
    j_idx = torch.arange(S_k, device=q.device)              # [S_k]
    causal_token = abs_i[:, None] >= j_idx[None, :]  # [S_q, S_k]，True = 可见
    token_attend = token_attend & causal_token[None, None]

    # NPU 惯例：True = 被遮蔽
    return (~token_attend).contiguous()


# ---------------------------------------------------------------------------
# PyTorch 参考（非 NPU 兜底 + 测试黄金）
# ---------------------------------------------------------------------------


def _block_sparse_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Block-sparse attention 的纯 PyTorch 参考实现。

    与 NPU 路径共用 ``_build_block_sparse_mask``，差别仅在用 PyTorch masked softmax
    代替 ``npu_fusion_attention``。用作：
    1. 非 NPU 环境（CPU / CUDA / ``torch_npu`` 未安装）的兜底
    2. 单测黄金参考（与 NPU 输出对比）
    """
    mask = _build_block_sparse_mask(q, k, topk_blocks, block_size)  # [B, H, S_q, S_k] True=masked
    scale = q.shape[-1] ** -0.5

    # fp32 计算防 fp16 溢出
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale  # [B, H, S_q, S_k]
    logits.masked_fill_(mask, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)  # 整行被遮蔽时避免 NaN
    out = torch.matmul(probs, v.float())
    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# NPU 路径
# ---------------------------------------------------------------------------


def _block_sparse_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """NPU 路径：构建 token 级 bool mask → npu_fusion_attention(sparse_mode=1)。

    调用前提：
    * ``q.device.type == "npu"`` 且 ``_HAS_TORCH_NPU``
    * head_dim 已 pad 至 ``_ALLOWED_HEAD_DIMS`` 内
    * ``max(S_q, S_k) <= _MAX_SEQ_FOR_MASK``（外层已检查）

    返回 ``[B, H, S_q, D]``，dtype 与输入一致。
    """
    assert _HAS_TORCH_NPU, "_block_sparse_npu 不能在非 NPU 环境调用"

    n_heads = q.shape[1]
    scale = q.shape[-1] ** -0.5

    mask = _build_block_sparse_mask(q, k, topk_blocks, block_size)  # [B, H, S_q, S_k] bool

    try:
        result = torch_npu.npu_fusion_attention(  # type: ignore[union-attr]
            q,
            k,
            v,
            head_num=n_heads,
            input_layout="BNSD",
            scale=scale,
            sparse_mode=1,      # user-provided mask
            atten_mask=mask,    # True = masked out
        )
    except TypeError:
        result = torch_npu.npu_fusion_attention(  # type: ignore[union-attr]
            q,
            k,
            v,
            head_num=n_heads,
            input_layout="BNSD",
            scale=scale,
            sparse_mode=1,
            atten_mask=mask.to(torch.uint8),
        )
    # npu_fusion_attention 返回元组 (out, softmax_max, softmax_sum, ...)
    return result[0] if isinstance(result, (tuple, list)) else result


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------


def block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_blocks: int,
    block_size: int = _DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """Block-sparse causal attention（NPU v1，路径 A：npu_fusion_attention + bool mask）。

    Args:
        q, k, v:      ``[B, H, S, D]``，已 RoPE、已 ``repeat_kv``（与上游一致）。
        topk_blocks:  每个 query block 保留的 key block 数量（与上游 ``top_k`` 对应）。
                      来自 ``best_pattern`` 的 ``vertical_size`` 字段。
        block_size:   block 大小（默认 64，与上游一致）。

    Returns:
        ``[B, H, S, D]``，与 ``q`` 同 dtype 同 device。

    短路 / 降级：
    * ``max(S_q, S_k) > _MAX_SEQ_FOR_MASK``：发 WARNING，退化为 causal dense。
      超长序列请改用 TileLang-Ascend 路径（路径 B）。
    * 非 NPU 设备：走 ``_block_sparse_pytorch_ref`` 参考路径。

    Notes:
        head_dim 不在 ``{16,32,64,128,256,512}`` 时自动 pad 到 2 的幂，输出截回原始 head_dim。
        与上游 ``block_sparse_flash_attention.py`` 的主要差异：
        1. 不使用 Triton kernel；改用 npu_fusion_attention + bool mask。
        2. 入参为 ``topk_blocks``（块数绝对值），对应上游 ``top_k``。
        3. 追加 per-token 因果约束保证对角块内正确性。
    """
    assert q.dim() == 4, f"q 期望 4D [B,H,S,D]，得到 {q.dim()}D"
    assert (
        q.shape[0] == k.shape[0] and q.shape[1] == k.shape[1] and q.shape[-1] == k.shape[-1]
    ), f"q/k B/H/D 不匹配；q={tuple(q.shape)} k={tuple(k.shape)}"

    topk_blocks = max(1, int(topk_blocks))  # 防止 ≤ 0

    orig_head_d = q.shape[-1]
    head_d = orig_head_d

    # head_dim pad
    if head_d not in _ALLOWED_HEAD_DIMS:
        target_d = _next_allowed_head_dim(head_d)
        q = _pad_head_dim(q, target_d)
        k = _pad_head_dim(k, target_d)
        v = _pad_head_dim(v, target_d)
        head_d = target_d

    S = max(q.shape[2], k.shape[2])

    # 超长序列退化为 dense
    if S > _MAX_SEQ_FOR_MASK:
        warnings.warn(
            f"block_sparse_attention: 序列长度 {S} > {_MAX_SEQ_FOR_MASK}。"
            " bool mask 构建需 O(S²) 内存，退化为 causal dense。"
            " 超长序列请改用 TileLang-Ascend 路径（docs/M3_block_sparse.md §6）。",
            stacklevel=2,
        )
        from ..backend_npu import dense_attention

        out = dense_attention(q, k, v, causal=True)
        return out[..., :orig_head_d]

    # 主路径
    if q.device.type == "npu" and _HAS_TORCH_NPU:
        out = _block_sparse_npu(q, k, v, topk_blocks, block_size)
    else:
        out = _block_sparse_pytorch_ref(q, k, v, topk_blocks, block_size)

    return out[..., :orig_head_d]
