# Copyright (c) 2024-2025 Microsoft
# Copyright (c) 2026 (NPU 适配 — M4 vertical-slash kernel)
# Licensed under The MIT License [see LICENSE for details]
"""Vertical-slash sparse attention — NPU v1 实现（路径 A：npu_fusion_attention + bool mask）。

算法语义：把 KV 序列按固定 block_size 划分为行块。每个 query block 只与以下 KV
位置做 attention（因果约束：j ≤ i 始终成立）：
  1. **Vertical 列**：v_idx 中的列（列重要性 topk），所有 query block 共享。
  2. **Slash 段**：每个 query block 对应一组 slash（反对角线）覆盖的 KV 块区间，
     由双指针算法（convert_vertical_slash_indexes）计算 block_offset。

两者合并后得到 token 级 bool attention mask，喂给 npu_fusion_attention(sparse_mode=1)。

NPU v1 实现路径（路径 A，见 docs/migration_plan_v1.md §4）：
  1. CPU 侧调用 convert_vertical_slash_indexes → block_count/block_offset/column_count/column_index
  2. 根据上述索引，用 cumsum 区间标记法高效构建 token 级 bool mask
  3. npu_fusion_attention(sparse_mode=1, atten_mask=mask)

序列长度限制：mask 构建是 O(S²) 内存；当 max(S_q, S_k) > _MAX_SEQ_FOR_MASK 时退化
为 dense attention 并打印 WARNING。路径 B（Triton-Ascend 稀疏 kernel）可突破此限制。

非 NPU 路径：_vertical_slash_pytorch_ref，直接从 v_idx/s_idx 构建 token 级 mask 并
用 PyTorch masked softmax 实现，作为单测黄金参考。

签名与上游 `MInference/minference/ops/pit_sparse_flash_attention_v2.py:vertical_slash_sparse_attention`
对齐：

    vertical_slash_sparse_attention(q, k, v, v_idx, s_idx,
                                    block_size_M=64, block_size_N=64) -> Tensor  # [B, H, S, D]

输入约定：q/k/v 形状 [B, H, S, D]（BNSD），已 RoPE、已 repeat_kv。
v_idx 升序，s_idx 降序，均为 int32。
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
except ImportError:
    torch_npu = None  # type: ignore[assignment]
    _HAS_TORCH_NPU = False

from ..backend_npu.cuda_shim import convert_vertical_slash_indexes

__all__ = ["vertical_slash_sparse_attention"]


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_ALLOWED_HEAD_DIMS = (16, 32, 64, 128, 256, 512)

# bool mask 构建需 O(S²) 内存；超过此阈值退化为 dense
_MAX_SEQ_FOR_MASK = 16384

_DEFAULT_BLOCK_SIZE = 64


# ---------------------------------------------------------------------------
# head_dim 工具
# ---------------------------------------------------------------------------


def _next_allowed_head_dim(d: int) -> int:
    for c in _ALLOWED_HEAD_DIMS:
        if c >= d:
            return c
    raise ValueError(f"head_dim={d} 超过支持的最大值 {_ALLOWED_HEAD_DIMS[-1]}")


def _pad_head_dim(t: torch.Tensor, target_d: int) -> torch.Tensor:
    cur = t.shape[-1]
    if cur == target_d:
        return t
    return F.pad(t, (0, target_d - cur))


# ---------------------------------------------------------------------------
# mask 构建：从 convert_vertical_slash_indexes 输出 → token 级 bool mask
# ---------------------------------------------------------------------------


def _build_vs_mask_from_indexes(
    block_count: torch.Tensor,   # [B, H, num_rows]  CPU int32
    block_offset: torch.Tensor,  # [B, H, num_rows, NNZ_S]  CPU int32
    column_count: torch.Tensor,  # [B, H, num_rows]  CPU int32
    column_index: torch.Tensor,  # [B, H, num_rows, NNZ_V]  CPU int32
    S_q: int,
    S_k: int,
    device: torch.device,
    block_size: int = _DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """从 convert_vertical_slash_indexes 输出构建 token 级 attention mask。

    Returns:
        mask [B, H, S_q, S_k]，bool，True = 被遮蔽（NPU 惯例）。
    """
    B, H, num_rows = block_count.shape
    mask = torch.ones(B, H, S_q, S_k, dtype=torch.bool, device=device)

    j_range = torch.arange(S_k, device=device, dtype=torch.long)  # [S_k]

    for b in range(B):
        for h in range(H):
            for bq in range(num_rows):
                start_q = bq * block_size
                if start_q >= S_q:
                    break
                end_q = min(start_q + block_size, S_q)
                block_rows = end_q - start_q

                q_rows = torch.arange(start_q, end_q, device=device, dtype=torch.long)  # [br]

                # ---- slash KV 块：用 cumsum 区间标记法 ----
                nblk = int(block_count[b, h, bq].item())
                slash_cov = torch.zeros(S_k, dtype=torch.bool, device=device)
                if nblk > 0:
                    blk_starts = block_offset[b, h, bq, :nblk].long().to(device)  # [nblk]
                    blk_ends = (blk_starts + block_size).clamp(max=S_k)  # [nblk]
                    # cumsum 区间标记：+1 at start, -1 at end
                    cov_delta = torch.zeros(S_k + 1, dtype=torch.int32, device=device)
                    cov_delta.scatter_add_(
                        0, blk_starts.clamp(0, S_k),
                        torch.ones(nblk, dtype=torch.int32, device=device),
                    )
                    cov_delta.scatter_add_(
                        0, blk_ends.clamp(0, S_k),
                        -torch.ones(nblk, dtype=torch.int32, device=device),
                    )
                    slash_cov = cov_delta.cumsum(0)[:S_k] > 0  # [S_k]

                # ---- vertical 列 ----
                ncol = int(column_count[b, h, bq].item())
                vert_cov = torch.zeros(S_k, dtype=torch.bool, device=device)
                if ncol > 0:
                    cols = column_index[b, h, bq, :ncol].long().to(device).clamp(0, S_k - 1)
                    vert_cov[cols] = True  # scatter set

                # ---- 合并覆盖区域 ----
                combined = slash_cov | vert_cov  # [S_k]

                # ---- 因果约束：j <= i ----
                # attend[i, j] = combined[j] AND j <= q_rows[i]
                attend_2d = combined[None, :] & (j_range[None, :] <= q_rows[:, None])  # [br, S_k]

                # ---- 写入 mask（True=masked → unmask = &= ~attend） ----
                mask[b, h, start_q:end_q, :] &= ~attend_2d

    return mask


# ---------------------------------------------------------------------------
# PyTorch 参考实现（非 NPU 兜底 + 测试黄金）
# ---------------------------------------------------------------------------


def _vertical_slash_pytorch_ref(
    q: torch.Tensor,    # [B, H, S_q, D]
    k: torch.Tensor,    # [B, H, S_k, D]
    v: torch.Tensor,    # [B, H, S_k, D]
    v_idx: torch.Tensor,  # [B, H, NNZ_V]  int32 ascending
    s_idx: torch.Tensor,  # [B, H, NNZ_S]  int32 descending
    block_size: int = _DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """Vertical-slash attention 的纯 PyTorch 参考实现。

    逻辑：
    1. 通过 convert_vertical_slash_indexes 获取 block_count/block_offset/column_count/column_index。
    2. 调用 _build_vs_mask_from_indexes 构建 token 级 mask。
    3. PyTorch masked softmax + matmul。

    用作：
    - 非 NPU 环境（CPU / CUDA）的兜底
    - 单测黄金参考（与 NPU 输出对比）
    """
    B, H, S_q, D = q.shape
    S_k = k.shape[2]

    seqlens = torch.tensor([S_k] * B, dtype=torch.int32)
    block_count, block_offset, column_count, column_index = convert_vertical_slash_indexes(
        seqlens, v_idx.cpu(), s_idx.cpu(), S_k, block_size, block_size,
    )

    mask = _build_vs_mask_from_indexes(
        block_count, block_offset, column_count, column_index,
        S_q, S_k, device=q.device, block_size=block_size,
    )  # [B, H, S_q, S_k] True=masked

    scale = D ** -0.5
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale  # [B, H, S_q, S_k]
    logits.masked_fill_(mask, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    out = torch.matmul(probs, v.float())
    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# NPU 路径
# ---------------------------------------------------------------------------


def _vertical_slash_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_idx: torch.Tensor,
    s_idx: torch.Tensor,
    block_size: int = _DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """NPU 路径：构建 token 级 bool mask → npu_fusion_attention(sparse_mode=1)。

    调用前提：
    * q.device.type == "npu" 且 _HAS_TORCH_NPU
    * head_dim 已 pad 至 _ALLOWED_HEAD_DIMS 内
    * max(S_q, S_k) <= _MAX_SEQ_FOR_MASK（外层已检查）
    """
    assert _HAS_TORCH_NPU, "_vertical_slash_npu 不能在非 NPU 环境调用"

    B, H, S_q, D = q.shape
    S_k = k.shape[2]
    scale = D ** -0.5

    seqlens = torch.tensor([S_k] * B, dtype=torch.int32)
    block_count, block_offset, column_count, column_index = convert_vertical_slash_indexes(
        seqlens, v_idx.cpu(), s_idx.cpu(), S_k, block_size, block_size,
    )

    mask = _build_vs_mask_from_indexes(
        block_count, block_offset, column_count, column_index,
        S_q, S_k, device=q.device, block_size=block_size,
    )  # [B, H, S_q, S_k] True=masked，NPU 惯例

    try:
        result = torch_npu.npu_fusion_attention(  # type: ignore[union-attr]
            q, k, v,
            head_num=H,
            input_layout="BNSD",
            scale=scale,
            sparse_mode=1,      # user-provided mask
            atten_mask=mask,    # True = masked out
        )
    except TypeError:
        result = torch_npu.npu_fusion_attention(  # type: ignore[union-attr]
            q, k, v,
            head_num=H,
            input_layout="BNSD",
            scale=scale,
            sparse_mode=1,
            atten_mask=mask.to(torch.uint8),
        )
    return result[0] if isinstance(result, (tuple, list)) else result


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------


def vertical_slash_sparse_attention(
    query: torch.Tensor,  # [B, H, S, D]
    key: torch.Tensor,    # [B, H, S, D]
    value: torch.Tensor,  # [B, H, S, D]
    v_idx: torch.Tensor,  # [B, H, NNZ_V]  int32 or int64，升序
    s_idx: torch.Tensor,  # [B, H, NNZ_S]  int32 or int64，降序
    block_size_M: int = _DEFAULT_BLOCK_SIZE,
    block_size_N: int = _DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """Vertical-slash causal attention（NPU v1，路径 A：npu_fusion_attention + bool mask）。

    与上游 ``pit_sparse_flash_attention_v2.py:vertical_slash_sparse_attention`` 签名对齐。

    Args:
        query, key, value: [B, H, S, D]，已 RoPE、已 repeat_kv。
        v_idx:  vertical 列索引，[B, H, NNZ_V]，升序 int32。来自在线估计的 topk 列索引。
        s_idx:  slash 偏移，[B, H, NNZ_S]，降序 int32。s = q_len - 1 - diag_from_upper_right。
        block_size_M: query block 大小（必须 = 64）。
        block_size_N: KV block 大小（必须 = 64）。

    Returns:
        [B, H, S, D]，与 query 同 dtype 同 device。

    短路 / 降级：
    * max(S_q, S_k) > _MAX_SEQ_FOR_MASK：发 WARNING，退化为 causal dense。
    * 非 NPU 设备：走 _vertical_slash_pytorch_ref。

    Notes:
        head_dim 不在 {16,32,64,128,256,512} 时自动 pad 到 2 的幂，输出截回原始 head_dim。
        与上游差异：
        1. 不使用 Triton CUDA kernel；改用 npu_fusion_attention + bool mask（路径 A）。
        2. convert_vertical_slash_indexes 在 CPU 侧用 Python 实现（M4-a）。
        3. block_size_M/N 目前只支持 64。
    """
    assert query.dim() == 4, f"query 期望 4D [B,H,S,D]，得到 {query.dim()}D"

    B, H, S_q, orig_head_d = query.shape
    S_k = key.shape[2]

    # v1 仅支持 full prefill (S_q == S_k)。decode 路径 (S_q == 1) 已在外层 forward
    # 的 q_len==1 短路里走 decode_dense，不会落到这里。chunked prefill 在 v1 不支持：
    # _vertical_and_slash_kernel 中 s_idx = (q_len-1) - s_raw_idx 与 _build_vs_mask
    # 的 q_rows 都假设绝对 query 位置从 0 起，S_q != S_k 时会算错索引/因果。
    assert S_q == S_k, (
        f"vertical_slash_sparse_attention: v1 仅支持 full prefill (S_q == S_k)；"
        f" 得到 S_q={S_q}, S_k={S_k}。chunked prefill / prefill-with-cache 是 v1 排除项。"
    )

    # head_dim pad
    head_d = orig_head_d
    if head_d not in _ALLOWED_HEAD_DIMS:
        target_d = _next_allowed_head_dim(head_d)
        query = _pad_head_dim(query, target_d)
        key = _pad_head_dim(key, target_d)
        value = _pad_head_dim(value, target_d)
        head_d = target_d

    # index 类型标准化
    v_idx = v_idx.to(torch.int32)
    s_idx = s_idx.to(torch.int32)

    S = max(S_q, S_k)

    # 超长序列退化为 dense
    if S > _MAX_SEQ_FOR_MASK:
        warnings.warn(
            f"vertical_slash_sparse_attention: 序列长度 {S} > {_MAX_SEQ_FOR_MASK}。"
            " bool mask 构建需 O(S²) 内存，退化为 causal dense。"
            " 超长序列请改用 Triton-Ascend 路径（路径 B）。",
            stacklevel=2,
        )
        from ..backend_npu import dense_attention

        out = dense_attention(query, key, value, causal=True)
        return out[..., :orig_head_d]

    if query.device.type == "npu" and _HAS_TORCH_NPU:
        out = _vertical_slash_npu(query, key, value, v_idx, s_idx, block_size_M)
    else:
        out = _vertical_slash_pytorch_ref(query, key, value, v_idx, s_idx, block_size_M)

    return out[..., :orig_head_d]
