# Copyright (c) 2026
# Licensed under The MIT License [see LICENSE for details]
"""上游 `minference.cuda` C++ 扩展的 NPU v1 替代实现（M4-a）。

上游仓库通过 `setup.py` 编译出 `minference.cuda` 子模块，导出
`convert_vertical_slash_indexes`（CUDA 索引展开 kernel）。NPU v1 不构建 CUDA
扩展，本文件用纯 Python + PyTorch CPU 实现同名接口。

实现对应上游 `csrc/vertical_slash_index.cu:27`（双指针 CUDA kernel）。
计算量极小（topk×1，不依赖序列长度），CPU 开销可忽略。

M4-a 设计决策：
- **方案 A（已落地）**：纯 Python/CPU，按 CUDA kernel 逻辑逐行翻译。
- **方案 B（备选）**：Triton-Ascend kernel 重写（仅当 CPU 成为吞吐瓶颈时考虑）。

算法概述（与 CUDA kernel 完全等价）：
  对每个 (batch, head, query_block)，用双指针同时扫描升序 vertical_indexes 和
  降序 slash_indexes：
  1. 跳过 slash 中 >= end_m 的无效值。
  2. 把有效 slash 值转换为 KV 区间端点 range_end = max(end_m - s_raw, BLK)。
  3. 相邻区间合并为连续 range；垂直列去重（落在 range 内不重复）。
  4. 输出 block_count/block_offset（slash 段）和 column_count/column_index（vertical 列）。

输出用途：喂给 M4-b 的 vertical-slash attention kernel，由后者构建 token 级 bool
attention mask 后调 `npu_fusion_attention(sparse_mode=1)`（路径 A，v1 已落地）。同一组
索引结构在 v2 也可直接复用为 Triton-Ascend 稀疏 kernel 的输入，无需重新设计。
"""

from __future__ import annotations

from typing import Tuple

import torch


__all__ = ["convert_vertical_slash_indexes"]

# CUDA kernel 硬编码了 block_size_M = block_size_N = 64；本文件同样只支持此值
_SUPPORTED_BLOCK_SIZE = 64


def _save_blocks(blk_buf: list, range_start: int, range_end: int, block_size: int) -> None:
    """将 [range_start, range_end) 内每隔 block_size 的起始位置追加到 blk_buf。

    对应 CUDA kernel 中的 ``save_blocks`` device 函数。
    """
    idx = range_start
    while idx < range_end:
        blk_buf.append(idx)
        idx += block_size


def _process_block(
    v_list: list,
    s_list: list,
    end_m: int,
    blk: int,
) -> Tuple[list, list]:
    """对单个 (batch, head, query_block) 运行双指针算法。

    Args:
        v_list: 升序 vertical 列索引列表，shape NNZ_V。
        s_list: 降序 slash 偏移列表，shape NNZ_S。
        end_m:  query block 的结束位置（= (block_idx_m + 1) * BLK）。
        blk:    block size（= 64）。

    Returns:
        (blk_buf, col_buf)：分别是 slash KV 块起始位置列表和 vertical 列索引列表。
    """
    NNZ_V_h = len(v_list)
    NNZ_S_h = len(s_list)
    blk_buf: list = []
    col_buf: list = []

    # 初始读取第一个 vertical 值（哨兵：无 vertical 列时设为 end_m+blk）
    if NNZ_V_h == 0:
        v = 0
        v_val = end_m + blk
    else:
        v = 1
        v_val = v_list[0]

    # 初始读取 slash，跳过 >= end_m 的无效值
    s = 0
    s_val_raw = None
    while s < NNZ_S_h:
        s_val_raw = s_list[s]
        s += 1
        if s_val_raw < end_m:
            break  # 找到第一个有效 slash
    else:
        s_val_raw = None  # 所有 slash 对该 block 无效

    if s_val_raw is None:
        # 无有效 slash：所有 vertical 列（直到哨兵）进 column_index
        while True:
            if v_val <= end_m + blk:  # 非哨兵且在合理范围
                if v_val < end_m:  # 仅取 < end_m 的（因果），与 CUDA 行为一致
                    col_buf.append(v_val)
            if v < NNZ_V_h:
                v_val = v_list[v]
                v += 1
            else:
                break
        return blk_buf, col_buf

    # 正常路径：初始化首个 range
    range_end = max(end_m - s_val_raw, blk)
    range_start = range_end - blk

    while True:
        if v_val < range_end:
            # vertical 值落在当前 range 之前（可能在 range 内或 range 之前）
            if v_val < range_start:
                # 不在 slash 覆盖区间 → 加入 column_index（去重）
                col_buf.append(v_val)
            # advance vertical 指针
            if v < NNZ_V_h:
                v_val = v_list[v]
                v += 1
            else:
                v_val = end_m + blk  # 哨兵：垂直列已耗尽
        else:
            # vertical 值 >= range_end → 处理下一个 slash
            if s < NNZ_S_h:
                new_s_raw = s_list[s]
                s += 1
                new_range_end = max(end_m - new_s_raw, blk)
            else:
                # 无更多 slash：保存当前 range 并退出循环
                _save_blocks(blk_buf, range_start, range_end, blk)
                break

            if new_range_end > range_end + blk:
                # 有间隔（new range 与 current range 不连续）：先保存当前 range
                _save_blocks(blk_buf, range_start, range_end, blk)
                range_start = new_range_end - blk
                range_end = new_range_end
            elif new_range_end > range_end:
                # 相邻或重叠：扩展当前 range（与 CUDA 一致，扩展 blk 而非 new_range_end）
                range_end += blk
            # else: new_range_end <= range_end，已被覆盖，跳过

    return blk_buf, col_buf


def convert_vertical_slash_indexes(
    seqlens: torch.Tensor,           # [BATCH]
    vertical_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]，升序，int32
    slash_indexes: torch.Tensor,     # [BATCH, N_HEADS, NNZ_S]，降序，int32
    context_size: int,
    block_size_M: int,
    block_size_N: int,
    causal: bool = True,             # v1 只支持 causal=True；参数保留对齐上游签名
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU Python 实现的 CUDA 双指针索引展开。

    将 vertical/slash 稀疏索引转换为 (block_count, block_offset, column_count,
    column_index) 四元组：v1 由 `ops/vertical_slash_kernel_npu.py` 据此构建 token
    级 bool mask 喂给 `npu_fusion_attention`；v2 若引入 Triton-Ascend 稀疏 kernel
    可直接复用同一组索引。

    Args:
        seqlens:          实际序列长度，shape [BATCH]，int32。
        vertical_indexes: vertical 列索引（升序），shape [BATCH, N_HEADS, NNZ_V]，int32。
        slash_indexes:    slash 偏移（降序，s = q_len - 1 - diag_idx），
                          shape [BATCH, N_HEADS, NNZ_S]，int32。
        context_size:     全序列长度（用于计算 num_rows = ceil(context_size / BLK)）。
        block_size_M:     query 块大小，必须等于 64（与上游 CUDA kernel 一致）。
        block_size_N:     KV 块大小，必须等于 64（与上游 CUDA kernel 一致）。
        causal:           是否因果（v1 恒 True，参数存在仅对齐上游签名）。

    Returns:
        (block_count, block_offset, column_count, column_index)，均为 int32 CPU tensor：
        - block_count:   [BATCH, N_HEADS, num_rows]        — 每个 query block 的 KV 块数量
        - block_offset:  [BATCH, N_HEADS, num_rows, NNZ_S] — KV 块起始位置
        - column_count:  [BATCH, N_HEADS, num_rows]        — 每个 query block 的 vertical 列数量
        - column_index:  [BATCH, N_HEADS, num_rows, NNZ_V] — vertical 列位置（不重叠于 slash 段）

    Raises:
        AssertionError: 若 block_size_M != 64 或 block_size_N != 64。
    """
    assert block_size_M == _SUPPORTED_BLOCK_SIZE, (
        f"block_size_M 必须等于 {_SUPPORTED_BLOCK_SIZE}，got {block_size_M}"
    )
    assert block_size_N == _SUPPORTED_BLOCK_SIZE, (
        f"block_size_N 必须等于 {_SUPPORTED_BLOCK_SIZE}，got {block_size_N}"
    )

    BLK = block_size_M  # 64

    batch_size = slash_indexes.size(0)
    num_heads = slash_indexes.size(1)
    nnz_slash = slash_indexes.size(2)
    nnz_vertical = vertical_indexes.size(2)
    num_rows = (context_size + BLK - 1) // BLK

    block_count = torch.zeros(batch_size, num_heads, num_rows, dtype=torch.int32)
    block_offset = torch.zeros(batch_size, num_heads, num_rows, nnz_slash, dtype=torch.int32)
    column_count = torch.zeros(batch_size, num_heads, num_rows, dtype=torch.int32)
    column_index = torch.zeros(batch_size, num_heads, num_rows, nnz_vertical, dtype=torch.int32)

    # 转 Python list，避免逐元素张量索引开销
    v_all = vertical_indexes.cpu().tolist()  # [B][H][NNZ_V]
    s_all = slash_indexes.cpu().tolist()     # [B][H][NNZ_S]
    seq_all = seqlens.cpu().tolist()         # [B]

    for b in range(batch_size):
        seqlen = int(seq_all[b])
        for h in range(num_heads):
            v_list = v_all[b][h]
            s_list = s_all[b][h]

            for block_idx_m in range(num_rows):
                start_m = block_idx_m * BLK
                if start_m >= seqlen:
                    break
                end_m = start_m + BLK  # 注意：不 clamp 到 seqlen，与 CUDA 一致

                blk_buf, col_buf = _process_block(v_list, s_list, end_m, BLK)

                n_blk = len(blk_buf)
                n_col = len(col_buf)

                block_count[b, h, block_idx_m] = n_blk
                column_count[b, h, block_idx_m] = n_col
                if n_blk:
                    block_offset[b, h, block_idx_m, :n_blk] = torch.tensor(
                        blk_buf, dtype=torch.int32
                    )
                if n_col:
                    column_index[b, h, block_idx_m, :n_col] = torch.tensor(
                        col_buf, dtype=torch.int32
                    )

    return block_count, block_offset, column_count, column_index
