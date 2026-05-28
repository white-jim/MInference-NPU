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


def _build_vs_mask_from_indexes_loop(
    block_count: torch.Tensor,   # [B, H, num_rows]  CPU int32
    block_offset: torch.Tensor,  # [B, H, num_rows, NNZ_S]  CPU int32
    column_count: torch.Tensor,  # [B, H, num_rows]  CPU int32
    column_index: torch.Tensor,  # [B, H, num_rows, NNZ_V]  CPU int32
    S_q: int,
    S_k: int,
    device: torch.device,
    block_size: int = _DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """Loop 版（v1 实现）。保留作为 vec 版的黄金参考 + bit-identical 对照。

    v2 PR-2 起生产路径切到 :func:`_build_vs_mask_from_indexes_vec`，此函数仅在
    ``tests/test_minference_batched_vs.py::test_loop_vs_vec_bit_identical`` 中
    被显式调用。

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


def _build_vs_mask_from_indexes_vec(
    block_count: torch.Tensor,   # [B, H, num_rows]   int32（CPU 或 device）
    block_offset: torch.Tensor,  # [B, H, num_rows, NNZ_S]  int32
    column_count: torch.Tensor,  # [B, H, num_rows]   int32
    column_index: torch.Tensor,  # [B, H, num_rows, NNZ_V]  int32
    S_q: int,
    S_k: int,
    device: torch.device,
    block_size: int = _DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """向量化版（v2 PR-2）。消除 (b, h, bq) 三重 Python 循环 + 4096 次小 NPU launch。

    与 :func:`_build_vs_mask_from_indexes_loop` 数值 bit-identical（True 在每个
    位置完全一致），由 ``test_loop_vs_vec_bit_identical`` 验证。

    设计要点：
      1. **Slash cumsum 区间标记** —— 在 [B, H, num_rows, S_k+1] 张量上一次性
         ``scatter_add_(+1)`` 起点 + ``scatter_add_(-1)`` 终点，然后 ``cumsum > 0``。
         无效位（i >= block_count）的起/终点同放在 sentinel S_k 上，+1/-1 抵消。
      2. **Vertical scatter** —— 用 int32 ``scatter_add_(v_valid.int())`` + ``> 0``，
         避免 bool scatter_ 在重复列上的覆写歧义。无效位 column_index 为 0
         （由 convert 阶段的 ``torch.zeros`` 初始化），其 v_valid=0 散布 0，无副作用。
      3. **Block → Token 展开** —— ``combined.repeat_interleave(block_size, dim=-2)``
         把 num_rows 维展开成 S_q 维，再 ``[..., :S_q, :]`` 截断处理 num_rows*block
         略大于 S_q 的边界。
      4. **因果** —— 与 j_range[None,:] <= i_range[:,None] 一次性与起来。
    """
    B, H, num_rows = block_count.shape
    NNZ_S = block_offset.shape[-1]
    NNZ_V = column_index.shape[-1]

    # 统一搬到目标 device（convert_vertical_slash_indexes 在 CPU 产 int32）
    block_count_d = block_count.to(device=device, dtype=torch.long)         # [B,H,R]
    block_offset_d = block_offset.to(device=device, dtype=torch.long)       # [B,H,R,NNZ_S]
    column_count_d = column_count.to(device=device, dtype=torch.long)       # [B,H,R]
    column_index_d = column_index.to(device=device, dtype=torch.long)       # [B,H,R,NNZ_V]

    # ---- Slash 覆盖：cumsum 区间标记 ----
    s_arange = torch.arange(NNZ_S, device=device).view(1, 1, 1, NNZ_S)
    s_valid = s_arange < block_count_d.unsqueeze(-1)                        # [B,H,R,NNZ_S]

    blk_starts = block_offset_d.clamp(0, S_k)                               # [B,H,R,NNZ_S]
    blk_ends = (block_offset_d + block_size).clamp(0, S_k)                  # [B,H,R,NNZ_S]

    # 无效位的 start/end 都放在 sentinel=S_k；scatter +1/-1 互相抵消，不污染 [:S_k]
    SENTINEL = S_k
    sentinel_t = torch.full_like(blk_starts, SENTINEL)
    blk_starts = torch.where(s_valid, blk_starts, sentinel_t)
    blk_ends = torch.where(s_valid, blk_ends, sentinel_t)

    cov_delta = torch.zeros(B, H, num_rows, S_k + 1, dtype=torch.int32, device=device)
    ones_int = torch.ones_like(blk_starts, dtype=torch.int32)
    cov_delta.scatter_add_(-1, blk_starts, ones_int)
    cov_delta.scatter_add_(-1, blk_ends, -ones_int)
    slash_cov = cov_delta.cumsum(-1)[..., :S_k] > 0                         # [B,H,R,S_k]

    # ---- Vertical 覆盖：int scatter_add + > 0（避开 bool scatter 覆写歧义） ----
    v_arange = torch.arange(NNZ_V, device=device).view(1, 1, 1, NNZ_V)
    v_valid = (v_arange < column_count_d.unsqueeze(-1)).to(torch.int32)     # [B,H,R,NNZ_V]
    cols = column_index_d.clamp(0, S_k - 1)                                 # [B,H,R,NNZ_V]
    vert_int = torch.zeros(B, H, num_rows, S_k, dtype=torch.int32, device=device)
    vert_int.scatter_add_(-1, cols, v_valid)
    vert_cov = vert_int > 0                                                 # [B,H,R,S_k]

    combined = slash_cov | vert_cov                                         # [B,H,R,S_k]

    # ---- num_rows → S_q 展开 ----
    # combined[b, h, bq, j] 描述 "query block bq 对 KV 位置 j 是否可见（不含因果）"
    combined_per_q = combined.repeat_interleave(block_size, dim=-2)         # [B,H,R*BLK,S_k]
    combined_per_q = combined_per_q[..., :S_q, :]                           # [B,H,S_q,S_k]

    # ---- 因果约束 ----
    i_range = torch.arange(S_q, device=device).view(S_q, 1)
    j_range = torch.arange(S_k, device=device).view(1, S_k)
    causal = j_range <= i_range                                             # [S_q, S_k] bool

    attend = combined_per_q & causal                                        # [B,H,S_q,S_k]
    mask = ~attend                                                          # True = masked
    return mask


# v2 PR-2：生产路径走向量化版；旧 loop 实现仅在测试中显式调用。
_build_vs_mask_from_indexes = _build_vs_mask_from_indexes_vec


# ---------------------------------------------------------------------------
# v2 PR-3：直接从 (v_idx, s_idx) 构建 mask，跳过 convert_vertical_slash_indexes
# ---------------------------------------------------------------------------


def _build_vs_mask_direct(
    v_idx: torch.Tensor,        # [B, H, NNZ_V]  int32/int64，升序
    s_idx: torch.Tensor,        # [B, H, NNZ_S]  int32/int64，降序
    S_q: int,
    S_k: int,
    device: torch.device,
    block_size: int = _DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """直接从 (v_idx, s_idx) 构建 token 级 bool mask，跳过 :func:`convert_vertical_slash_indexes`。

    v2 PR-3：v1 / PR-2 路径走 `convert_vertical_slash_indexes`（CPU 双指针，每次
    prefill ~131072 次 Python 迭代）→ `_build_vs_mask_from_indexes_vec`。PR-3 把
    这两步合并为单个 NPU-side 张量算子链，**彻底消除整条 v2 路径里最后一处 host-
    bound CPU 循环**。

    Args:
        v_idx: vertical 列索引，[B, H, NNZ_V]，升序。
        s_idx: slash 偏移（= q_len - 1 - diag_idx），[B, H, NNZ_S]，降序。
        S_q, S_k: 序列长度（v1 假设 S_q == S_k，即 full prefill）。
        device: 目标 device（mask 落在此处）。
        block_size: query / KV 块大小，必须 = 64。

    Returns:
        mask [B, H, S_q, S_k]，bool，True = 被遮蔽（NPU 惯例）。

    数值正确性：
        与 ``convert_vertical_slash_indexes`` + ``_build_vs_mask_from_indexes_loop``
        bit-identical（由 ``test_build_mask_direct_vs_convert_loop_bit_identical``
        在多组 (B,H,S,NNZ_V,NNZ_S) 下 ``torch.equal == True`` 验证）。

    关键等价性：
        原 CUDA 双指针 kernel 做两件事 ——
          1. slash 区间合并（相邻/重叠 [range_start, range_end) 合并）
          2. vertical 列去重（排除已被 slash 覆盖的列 + 排除 v_val >= end_m 的因果外列）
        这两件事在最终 mask 层面用 ``slash_cov | vert_cov``（OR 语义）+ 因果 ``&``
        后**完全等价**：bool OR 对重叠/重复幂等，因果 mask 兜底处理 ``j >= end_m``。
        因此可以直接对每个 slash 段独立打 cumsum 区间标记、对 vertical 一次 scatter，
        无需合并/去重。

    设计要点：
      1. **Slash cumsum 区间标记** —— 对每个 (b,h,bq,k_s)：
           valid = s_raw < end_m，end_m = (bq+1)*block_size
           range_end = max(end_m - s_raw, block_size)
           range_start = range_end - block_size
         invalid 位的 start/end 都设为 sentinel=S_k，scatter +1/-1 互相抵消。
      2. **Vertical scatter 共享** —— v_idx 不依赖 bq，一次 ``scatter_add_(+1) > 0``
         得到 [B,H,S_k]，再 broadcast 到 num_rows 维度，省一份 [B,H,R,S_k] 显存。
      3. **Block → Token 展开** —— ``repeat_interleave(block_size, dim=-2)`` +
         ``[..., :S_q, :]`` 与 PR-2 vec 版完全一致。
      4. **因果** —— 与 ``j_range[None,:] <= i_range[:,None]`` 一次性 AND。

    显存复杂度：与 PR-2 vec 版同 —— O(B*H*num_rows*S_k) 中间 + O(B*H*S_q*S_k) 输出。
    S>16384 仍由外层 silent dense fallback 兜底（不在此函数处理）。
    """
    assert block_size == _DEFAULT_BLOCK_SIZE, (
        f"_build_vs_mask_direct 仅支持 block_size={_DEFAULT_BLOCK_SIZE}，got {block_size}"
    )
    B, H, NNZ_V = v_idx.shape
    NNZ_S = s_idx.shape[-1]
    num_rows = (S_k + block_size - 1) // block_size
    SENTINEL = S_k

    # 移到目标 device 并转 long（scatter_add_ 的 index 必须 long）
    v_idx_d = v_idx.to(device=device, dtype=torch.long)  # [B, H, NNZ_V]
    s_idx_d = s_idx.to(device=device, dtype=torch.long)  # [B, H, NNZ_S]

    # ---- Slash 覆盖：cumsum 区间标记 ----
    # end_m[bq] = (bq+1) * block_size
    bq_arange = torch.arange(num_rows, device=device, dtype=torch.long)  # [R]
    end_m = (bq_arange + 1) * block_size                                 # [R]
    end_m_view = end_m.view(1, 1, num_rows, 1)                           # [1,1,R,1]

    s_view = s_idx_d.unsqueeze(-2)                                       # [B,H,1,NNZ_S]
    valid_s = s_view < end_m_view                                        # [B,H,R,NNZ_S] bool

    # range_end = max(end_m - s_raw, block_size)
    # 对 invalid 位也算出一个数没关系，下面用 where 替换为 sentinel
    range_end_raw = (end_m_view - s_view).clamp(min=block_size).clamp(max=S_k)
    range_start_raw = (range_end_raw - block_size).clamp(min=0, max=S_k)

    sentinel_t = torch.full(
        (), SENTINEL, dtype=torch.long, device=device
    )  # 0-dim，broadcast 到 [B,H,R,NNZ_S]
    range_start = torch.where(valid_s, range_start_raw, sentinel_t)      # [B,H,R,NNZ_S]
    range_end = torch.where(valid_s, range_end_raw, sentinel_t)          # [B,H,R,NNZ_S]

    cov_delta = torch.zeros(B, H, num_rows, S_k + 1, dtype=torch.int32, device=device)
    ones_int = torch.ones_like(range_start, dtype=torch.int32)
    cov_delta.scatter_add_(-1, range_start, ones_int)
    cov_delta.scatter_add_(-1, range_end, -ones_int)
    slash_cov = cov_delta.cumsum(-1)[..., :S_k] > 0                      # [B,H,R,S_k]

    # ---- Vertical 覆盖：所有 bq 共享同一组 v_idx，先生成 [B,H,S_k] 再 broadcast ----
    v_cols = v_idx_d.clamp(0, S_k - 1)                                   # [B,H,NNZ_V]
    vert_int = torch.zeros(B, H, S_k, dtype=torch.int32, device=device)
    vert_int.scatter_add_(
        -1, v_cols, torch.ones_like(v_cols, dtype=torch.int32)
    )
    vert_cov = (vert_int > 0).unsqueeze(-2)                              # [B,H,1,S_k]

    # ---- 合并 ----
    combined = slash_cov | vert_cov                                      # [B,H,R,S_k] (broadcast)

    # ---- num_rows → S_q 展开 ----
    combined_per_q = combined.repeat_interleave(block_size, dim=-2)      # [B,H,R*BLK,S_k]
    combined_per_q = combined_per_q[..., :S_q, :]                        # [B,H,S_q,S_k]

    # ---- 因果约束 ----
    i_range = torch.arange(S_q, device=device).view(S_q, 1)
    j_range = torch.arange(S_k, device=device).view(1, S_k)
    causal = j_range <= i_range                                          # [S_q, S_k] bool

    attend = combined_per_q & causal                                     # [B,H,S_q,S_k]
    mask = ~attend                                                       # True = masked
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
    """Vertical-slash attention 的纯 PyTorch 参考实现（v2 PR-3 起 = direct 路径）。

    逻辑：
    1. 调用 :func:`_build_vs_mask_direct` 从 (v_idx, s_idx) 直接构建 token 级 mask
       （跳过 ``convert_vertical_slash_indexes`` 的 CPU Python 双指针）。
    2. PyTorch masked softmax + matmul。

    用作：
    - 非 NPU 环境（CPU / CUDA）的兜底
    - 单测黄金参考（与 NPU 输出对比）—— 与 NPU 路径同走 direct 算法，保 bit-identical

    v2 PR-3 算法语义变更：mask 不再做上游 CUDA 的"贪婪 blk 扩展"近似，改走真实
    slash 区间 OR。这是合法的 MInference 算法变种（块扩展是为 Triton CUDA 块对齐
    sparse kernel 服务，NPU token-level bool mask 路径无此需求）。与 v1 mask 的关系
    由 :func:`_vertical_slash_pytorch_ref_legacy` 配合的子集测试验证：
    direct ⊆ v1（visible 集合），即 v1 多覆盖一些块对齐扩展位置。
    """
    B, H, S_q, D = q.shape
    S_k = k.shape[2]

    mask = _build_vs_mask_direct(
        v_idx, s_idx, S_q, S_k, device=q.device, block_size=block_size,
    )  # [B, H, S_q, S_k] True=masked

    scale = D ** -0.5
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale  # [B, H, S_q, S_k]
    logits.masked_fill_(mask, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    out = torch.matmul(probs, v.float())
    return out.to(q.dtype)


def _vertical_slash_pytorch_ref_legacy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_idx: torch.Tensor,
    s_idx: torch.Tensor,
    block_size: int = _DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """v1 算法参考（convert_vertical_slash_indexes + loop mask 构建）。**仅测试用**。

    保留作为 PR-3 算法差异对比的对照基线。生产路径走 :func:`_vertical_slash_pytorch_ref`
    （direct）。

    v1 mask 继承上游 CUDA 的"贪婪 blk 扩展"近似，是 direct mask 的 visible 超集
    （direct mask True ⊇ v1 mask True）。在 ``test_minference_batched_vs.py`` 的
    T5 / T6 中分别验证子集关系 + attention 输出容差。
    """
    B, H, S_q, D = q.shape
    S_k = k.shape[2]

    seqlens = torch.tensor([S_k] * B, dtype=torch.int32)
    block_count, block_offset, column_count, column_index = convert_vertical_slash_indexes(
        seqlens, v_idx.cpu(), s_idx.cpu(), S_k, block_size, block_size,
    )

    mask = _build_vs_mask_from_indexes_loop(
        block_count, block_offset, column_count, column_index,
        S_q, S_k, device=q.device, block_size=block_size,
    )

    scale = D ** -0.5
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
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

    v2 PR-3：mask 构建走 :func:`_build_vs_mask_direct`，直接从 (v_idx, s_idx) 在
    NPU 上一步生成 [B,H,S_q,S_k] bool mask，跳过原 :func:`convert_vertical_slash_indexes`
    的 CPU Python 双指针（每次 prefill 32 层 × 32 head × 128 query block ≈ 131072
    次迭代）。等价性见 ``test_build_mask_direct_vs_convert_loop_bit_identical``。
    """
    assert _HAS_TORCH_NPU, "_vertical_slash_npu 不能在非 NPU 环境调用"

    B, H, S_q, D = q.shape
    S_k = k.shape[2]
    scale = D ** -0.5

    mask = _build_vs_mask_direct(
        v_idx, s_idx, S_q, S_k, device=q.device, block_size=block_size,
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
