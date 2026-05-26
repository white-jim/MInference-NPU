# Copyright (c) 2026 (NPU 适配 — PR-4 tilelang Indices 转换层)
# Licensed under The MIT License [see LICENSE for details]
"""MInference 稀疏 pattern → tilelang-ascend ``sparse_attention_fwd`` 的 Indices 格式转换。

tilelang ``sparse_attention_fwd`` 接受的 ``Indices`` 张量约定：

  形状 ``[B, S_q, kv_group, topk]``，int32
  语义   每个 Q token 看到的 K **token 位置**（不是 block 索引）的列表
  约束   ``topk % block_I == 0`` 以保证 kernel 的 block-load 路径对齐
  causal `is_causal=True` 由 kernel 自行处理，本模块不必在 Indices 里裁剪未来 K 位置

本模块提供两个构造函数：

  1. ``block_indices_to_tilelang`` —— 通用 block-sparse / VS 输入：
        给定每个 (b, h, q_block) 的一组 K-block 索引，展开为 Indices。
        ``minference/ops/block_sparse_kernel_npu.py`` 内部的 ``topk_idx`` 直接对接。

  2. ``stream_llm_to_tilelang`` —— A-shape (stream_llm) 专用：
        给定 (n_init, n_local)，按 anchor + sliding-window 直接构造，无须先打分。

GQA 注意事项：
  tilelang Indices 的第 3 维是 ``kv_group``，也就是 KV tensor 的第 3 维 ``g``：
  ``[B, S_k, g, D]``。一个 KV head 服务 ``H // g`` 个 Q head。在 MHA
  （``H == n_kv_heads``）时 ``kv_group = H``，每个 Q head 拥有独立索引列表。

  在真 GQA（``H = 32`` 个 Q head 共享 ``n_kv_heads = 8`` 个 KV head，``kv_group = 8``）
  时，同一 KV head 下的 4 个 Q head **必须**共享同一 Indices 行。MInference 上游
  对 GQA 通常采用 "Q head 内部各自打分但 KV 只算一份" 的实现，无法直接映射。

  v1 (PR-4-tl-BS) MVP 只支持 MHA（``H == n_kv_heads``）；GQA 留 v2，需要先做
  per-group union 或 reuse 设计。MHA 检查在 ``_assert_mha`` 里集中处理。
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "block_indices_to_tilelang",
    "stream_llm_to_tilelang",
    "TILELANG_PAD_VALUE",
]


# tilelang Indices 的 pad sentinel —— 用 -1 让 kernel 内部容易识别为 "无效位置"。
# kernel 收到 -1 时不应做 K 载入。若实测发现 tilelang 需要不同 sentinel，改这里一处。
TILELANG_PAD_VALUE: int = -1


def _assert_mha(H: int, kv_heads: int, context: str) -> None:
    if H != kv_heads:
        raise NotImplementedError(
            f"{context}: 检测到 H={H} != kv_heads={kv_heads}（GQA）。"
            " PR-4-tl-BS MVP 仅支持 MHA。GQA 留 v2。"
        )


def block_indices_to_tilelang(
    block_indices: torch.Tensor,
    S_q: int,
    block_size_M: int,
    block_size_N: int,
    kv_heads: int,
    block_count: Optional[torch.Tensor] = None,
    pad_value: int = TILELANG_PAD_VALUE,
) -> torch.Tensor:
    """Per-(head, q_block) 的 K-block 索引展开为 tilelang Indices。

    Args:
        block_indices: ``[B, H, n_q_blocks, max_blocks]`` int，K block 索引（**block 空间，
            不是 token 位置**）。比如值 5 表示 K 序列的第 5 个 block（含 token
            ``[5*block_size_N, (5+1)*block_size_N)``）。Padding 槽位的值可以是任意整数，
            通过 ``block_count`` 区分。
        S_q: Q 序列总长度（``n_q_blocks * block_size_M`` 可能 ≥ S_q，最后一个 Q block 可能
            部分超出，会在输出里裁掉）。
        block_size_M: Q block 大小（MInference 上游约定 64）。
        block_size_N: K block 大小（== tilelang ``block_I``，默认 64）。
        kv_heads: tilelang ``Indices`` 第 3 维大小。MHA 下必须 == H。
        block_count: ``[B, H, n_q_blocks]`` int，每个 Q block 有效 K block 数量。
            None 时 ``max_blocks`` 个槽位全部认为有效。
        pad_value: 无效槽位填充值（默认 ``TILELANG_PAD_VALUE``）。

    Returns:
        ``Indices`` ``[B, S_q, kv_heads, max_blocks * block_size_N]`` int32。
        每个 K block 索引展开为 ``block_size_N`` 个连续 token 位置；
        无效 block 的整段 ``block_size_N`` 位置全部填 ``pad_value``。

    Note:
        Q block 内的 ``block_size_M`` 个 token 共享同一组 K 索引（broadcast）；
        因果约束由 ``sparse_attention_fwd(is_causal=True)`` 在 kernel 里自动施加，
        本函数不裁未来 K 位置。
    """
    B, H, n_q_blocks, max_blocks = block_indices.shape
    _assert_mha(H, kv_heads, "block_indices_to_tilelang")

    topk = max_blocks * block_size_N
    device = block_indices.device
    dtype = block_indices.dtype

    # 每个 block 索引展开为 block_size_N 个连续 K token 位置
    arange_N = torch.arange(block_size_N, device=device, dtype=dtype)
    # [B, H, n_q_blocks, max_blocks, block_size_N]
    token_positions = block_indices.unsqueeze(-1) * block_size_N + arange_N
    # [B, H, n_q_blocks, topk]
    token_positions = token_positions.reshape(B, H, n_q_blocks, topk)

    # 应用 block_count：超出 valid 数量的 K block 整段填 pad_value
    if block_count is not None:
        block_idx_range = torch.arange(max_blocks, device=device).reshape(1, 1, 1, max_blocks, 1)
        valid = block_idx_range < block_count.reshape(B, H, n_q_blocks, 1, 1)
        valid = valid.expand(B, H, n_q_blocks, max_blocks, block_size_N).reshape(
            B, H, n_q_blocks, topk
        )
        token_positions = torch.where(
            valid,
            token_positions,
            torch.full_like(token_positions, pad_value),
        )

    # Q block 内 broadcast 到每个 Q token
    # [B, H, n_q_blocks, topk] → [B, H, n_q_blocks * block_size_M, topk]
    token_positions = token_positions.repeat_interleave(block_size_M, dim=2)
    # 裁到 S_q（最后一个 Q block 可能部分超出）
    token_positions = token_positions[:, :, :S_q, :]

    # 转置到 tilelang 约定的 [B, S_q, kv_heads, topk]
    indices = token_positions.permute(0, 2, 1, 3).contiguous().to(torch.int32)
    return indices


def stream_llm_to_tilelang(
    B: int,
    S_q: int,
    kv_heads: int,
    n_init: int,
    n_local: int,
    block_size_N: int = 64,
    pad_value: int = TILELANG_PAD_VALUE,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """A-shape (stream_llm) 模式的 tilelang Indices 直接构造。

    每个 Q token ``s_q`` 看到：
      - **Anchor**：token ``[0, n_init)`` （所有 Q 共享）
      - **Local**：token ``[s_q - n_local + 1, s_q]`` （滑窗，越界部分填 pad_value）

    Anchor 与 Local 重叠的位置会在 Local 段填 ``pad_value``，避免同一个 K token
    被 sparse kernel 重复计入 softmax。Reference 的 scatter mask 会天然去重，但
    真正的 sparse kernel 通常按 Indices 列表逐项计算，重复项会改变概率分布。

    Args:
        B, S_q, kv_heads: 输出 Indices 形状的前三维。
        n_init: anchor token 数量（必须是 ``block_size_N`` 的倍数）。
        n_local: 滑窗长度（必须是 ``block_size_N`` 的倍数）。
        block_size_N: tilelang ``block_I``，验证 topk 整除用。
        pad_value: 越界 / 无效位置填充值。
        device, dtype: 输出张量配置。

    Returns:
        ``[B, S_q, kv_heads, n_init + n_local]`` int32。
    """
    if n_init % block_size_N != 0 or n_local % block_size_N != 0:
        raise ValueError(
            f"n_init={n_init} 与 n_local={n_local} 必须是 block_size_N={block_size_N} 的倍数"
        )
    topk = n_init + n_local

    # Anchor：[0, 1, ..., n_init-1]，所有 Q token 共享
    anchor = torch.arange(n_init, device=device, dtype=dtype).reshape(1, 1, 1, n_init)

    # Local：第 s_q 个 Q token 的滑窗 = [s_q - n_local + 1, ..., s_q]
    s_q_idx = torch.arange(S_q, device=device, dtype=dtype).reshape(1, S_q, 1, 1)
    local_off = torch.arange(-n_local + 1, 1, device=device, dtype=dtype).reshape(1, 1, 1, n_local)
    local_pos = s_q_idx + local_off  # [1, S_q, 1, n_local]
    # 越过序列起点（< 0）或与 anchor 重叠的位置填 pad_value，保持 Indices 唯一。
    # anchor 本身始终保留；未来位置由 kernel 的 causal 逻辑屏蔽。
    local_pos = torch.where(
        (local_pos >= 0) & (local_pos >= n_init),
        local_pos,
        torch.full_like(local_pos, pad_value),
    )

    anchor_full = anchor.expand(B, S_q, kv_heads, n_init).contiguous()
    local_full = local_pos.expand(B, S_q, kv_heads, n_local).contiguous()
    return torch.cat([anchor_full, local_full], dim=-1)
