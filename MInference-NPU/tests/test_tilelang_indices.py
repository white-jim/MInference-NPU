# Copyright (c) 2026 (NPU 适配 — PR-4-tl-BS Indices 转换层单测)
# Licensed under The MIT License [see LICENSE for details]
"""``minference/ops/tilelang_indices.py`` 的 CPU-only 单测。

只验证 Indices 张量的形状 / dtype / 内容语义，不依赖 NPU / tilelang，
所以可以在任何带 torch 的机器上 ``python -m pytest`` 直接跑。

运行::

    python -m pytest tests/test_tilelang_indices.py -v
"""

from __future__ import annotations

import pytest
import torch

from minference.ops.tilelang_indices import (
    TILELANG_PAD_VALUE,
    block_indices_to_tilelang,
    stream_llm_to_tilelang,
)


# ---------------------------------------------------------------------------
# block_indices_to_tilelang
# ---------------------------------------------------------------------------


class TestBlockIndicesToTilelang:
    def test_shape_and_dtype(self):
        B, H, n_q_blocks, max_blocks = 2, 4, 3, 5
        block_size_M = 64
        block_size_N = 64
        S_q = n_q_blocks * block_size_M

        block_indices = torch.zeros(B, H, n_q_blocks, max_blocks, dtype=torch.int64)
        out = block_indices_to_tilelang(
            block_indices,
            S_q=S_q,
            block_size_M=block_size_M,
            block_size_N=block_size_N,
            kv_heads=H,
        )
        assert out.shape == (B, S_q, H, max_blocks * block_size_N)
        assert out.dtype == torch.int32

    def test_block_to_token_expansion(self):
        """K block 索引 b → token 位置 [b*block_size_N, (b+1)*block_size_N)."""
        B, H, n_q_blocks, max_blocks = 1, 1, 1, 2
        block_size_M, block_size_N = 4, 4  # 小尺寸方便人工核对
        S_q = block_size_M

        # Q block 0 看 K block [3, 5]
        block_indices = torch.tensor([[[[3, 5]]]], dtype=torch.int64)
        out = block_indices_to_tilelang(
            block_indices, S_q, block_size_M, block_size_N, kv_heads=H
        )
        # 形状 [1, 4, 1, 8]
        assert out.shape == (1, S_q, 1, 8)

        # 第一个 Q token (s_q=0) 看到的 K token：[12,13,14,15, 20,21,22,23]
        expected = torch.tensor([12, 13, 14, 15, 20, 21, 22, 23], dtype=torch.int32)
        assert torch.equal(out[0, 0, 0], expected)

    def test_q_block_broadcast(self):
        """同一 Q block 内的 block_size_M 个 token 共享同一组 K 索引。"""
        B, H, n_q_blocks, max_blocks = 1, 1, 2, 1
        block_size_M, block_size_N = 4, 4
        S_q = n_q_blocks * block_size_M

        # Q block 0 看 K block [2]; Q block 1 看 K block [7]
        block_indices = torch.tensor([[[[2], [7]]]], dtype=torch.int64)
        out = block_indices_to_tilelang(
            block_indices, S_q, block_size_M, block_size_N, kv_heads=H
        )
        # Q block 0 的 4 个 token 都应看到 K tokens [8,9,10,11]
        for s in range(block_size_M):
            assert torch.equal(
                out[0, s, 0], torch.tensor([8, 9, 10, 11], dtype=torch.int32)
            )
        # Q block 1 的 4 个 token 都应看到 K tokens [28,29,30,31]
        for s in range(block_size_M, S_q):
            assert torch.equal(
                out[0, s, 0], torch.tensor([28, 29, 30, 31], dtype=torch.int32)
            )

    def test_block_count_masks_pad(self):
        """block_count 标记的无效槽位整段填 pad_value。"""
        B, H, n_q_blocks, max_blocks = 1, 1, 1, 3
        block_size_M, block_size_N = 4, 4
        S_q = block_size_M

        # 三个槽位有值，但只有前 2 个有效
        block_indices = torch.tensor([[[[1, 2, 99]]]], dtype=torch.int64)
        block_count = torch.tensor([[[2]]], dtype=torch.int64)
        out = block_indices_to_tilelang(
            block_indices,
            S_q,
            block_size_M,
            block_size_N,
            kv_heads=H,
            block_count=block_count,
        )
        # 前 2 个 block 的 token 正常；第 3 个 block 整段 pad
        expected = torch.tensor(
            [4, 5, 6, 7, 8, 9, 10, 11]
            + [TILELANG_PAD_VALUE] * 4,
            dtype=torch.int32,
        )
        assert torch.equal(out[0, 0, 0], expected)

    def test_s_q_trim(self):
        """最后一个 Q block 部分超出 S_q 时输出按 S_q 裁剪。"""
        B, H, n_q_blocks, max_blocks = 1, 1, 2, 1
        block_size_M, block_size_N = 4, 4
        S_q = 6  # 不是 block_size_M 的整数倍

        block_indices = torch.tensor([[[[0], [1]]]], dtype=torch.int64)
        out = block_indices_to_tilelang(
            block_indices, S_q, block_size_M, block_size_N, kv_heads=H
        )
        assert out.shape == (1, S_q, 1, 4)
        # 前 4 行（Q block 0）看 K block 0 → [0,1,2,3]
        for s in range(4):
            assert torch.equal(
                out[0, s, 0], torch.tensor([0, 1, 2, 3], dtype=torch.int32)
            )
        # 第 5/6 行（Q block 1 的前 2 个）看 K block 1 → [4,5,6,7]
        for s in range(4, S_q):
            assert torch.equal(
                out[0, s, 0], torch.tensor([4, 5, 6, 7], dtype=torch.int32)
            )

    def test_gqa_raises(self):
        block_indices = torch.zeros(1, 8, 1, 1, dtype=torch.int64)
        with pytest.raises(NotImplementedError, match="GQA"):
            block_indices_to_tilelang(
                block_indices,
                S_q=64,
                block_size_M=64,
                block_size_N=64,
                kv_heads=2,  # H=8 != kv_heads=2 → GQA
            )


# ---------------------------------------------------------------------------
# stream_llm_to_tilelang
# ---------------------------------------------------------------------------


class TestStreamLlmToTilelang:
    def test_shape_and_dtype(self):
        B, S_q, kv_heads = 2, 256, 4
        n_init, n_local = 64, 128
        out = stream_llm_to_tilelang(B, S_q, kv_heads, n_init, n_local, block_size_N=64)
        assert out.shape == (B, S_q, kv_heads, n_init + n_local)
        assert out.dtype == torch.int32

    def test_anchor_content(self):
        """前 n_init 列固定为 [0, n_init) 且对所有 Q token 一致。"""
        B, S_q, kv_heads = 1, 64, 1
        n_init, n_local = 64, 64
        out = stream_llm_to_tilelang(B, S_q, kv_heads, n_init, n_local, block_size_N=64)
        anchor_ref = torch.arange(n_init, dtype=torch.int32)
        for s in range(S_q):
            assert torch.equal(out[0, s, 0, :n_init], anchor_ref)

    def test_local_sliding_window_far(self):
        """s_q 远大于 n_local 时 local 段应为 [s_q-n_local+1 .. s_q]."""
        B, S_q, kv_heads = 1, 256, 1
        n_init, n_local = 64, 64
        out = stream_llm_to_tilelang(B, S_q, kv_heads, n_init, n_local, block_size_N=64)
        s_q = 200
        expected_local = torch.arange(
            s_q - n_local + 1, s_q + 1, dtype=torch.int32
        )
        assert torch.equal(out[0, s_q, 0, n_init:], expected_local)

    def test_local_sliding_window_edge(self):
        """s_q < n_local-1 或 local 与 anchor 重叠时 → local 段填 pad_value。"""
        B, S_q, kv_heads = 1, 64, 1
        n_init, n_local = 64, 64
        out = stream_llm_to_tilelang(B, S_q, kv_heads, n_init, n_local, block_size_N=64)
        # s_q = 0：local 段原本是 [-63..-1, 0]，0 已在 anchor 里，因此全填 pad。
        s_q = 0
        local = out[0, s_q, 0, n_init:]
        assert (local == TILELANG_PAD_VALUE).all()

        # s_q = 10：local 段原本是 [-53..-1, 0..10]，0..10 都已在 anchor 里。
        s_q = 10
        local = out[0, s_q, 0, n_init:]
        assert (local == TILELANG_PAD_VALUE).all()

    def test_local_anchor_overlap_is_padded(self):
        """Local 段和 anchor 重叠的 token 置 pad，避免 sparse kernel 重复计数。"""
        B, S_q, kv_heads = 1, 128, 1
        n_init, n_local = 64, 64
        out = stream_llm_to_tilelang(B, S_q, kv_heads, n_init, n_local, block_size_N=64)

        s_q = 64
        local = out[0, s_q, 0, n_init:]
        assert (local[: n_local - 1] == TILELANG_PAD_VALUE).all()
        assert local[-1].item() == 64

    def test_topk_divisible_validation(self):
        with pytest.raises(ValueError, match="block_size_N"):
            stream_llm_to_tilelang(1, 64, 1, n_init=63, n_local=64, block_size_N=64)
        with pytest.raises(ValueError, match="block_size_N"):
            stream_llm_to_tilelang(1, 64, 1, n_init=64, n_local=65, block_size_N=64)

    def test_kv_heads_broadcast(self):
        """所有 kv_head 共享同一份 anchor + local（MHA 下也成立，每个 Q head 同样视野）。"""
        B, S_q, kv_heads = 1, 128, 4
        n_init, n_local = 64, 64
        out = stream_llm_to_tilelang(B, S_q, kv_heads, n_init, n_local, block_size_N=64)
        for h in range(1, kv_heads):
            assert torch.equal(out[0, :, 0, :], out[0, :, h, :])
