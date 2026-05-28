# Copyright (c) 2026 (NPU adapter - PR-4 TileLang sparse attention)
# Licensed under The MIT License [see LICENSE for details]
"""Experimental H=1 block-sparse TileLang attention kernel.

This module is intentionally isolated from ``block_sparse_attention`` for now.
It tests the next path-B direction: when the wrapper folds heads into the batch
dimension, compute one real head without paying the old 16 padded-head slots.

The kernel processes a small block of query tokens for one head
(``block_M`` rows, default 16).  That keeps the vector softmax lanes large
enough for Ascend while replacing the old "16 head rows for one query token"
shape with "16 query rows for one real head".
"""

import torch

__all__ = [
    "build_sparse_attention_h1_block_fwd",
    "build_sparse_attention_h1_block_index_fwd",
    "build_sparse_attention_mh_block_index_fwd",
    "clear_sparse_attention_h1_kernel_cache",
]


_KERNEL_CACHE: dict[tuple[object, ...], object] = {}


def clear_sparse_attention_h1_kernel_cache() -> None:
    """Clear compiled TileLang H=1 sparse-attention callables."""
    _KERNEL_CACHE.clear()


def _require_tilelang():
    import tilelang  # type: ignore[import-not-found]
    from tilelang import DataType, language as T  # type: ignore[import-not-found]

    tilelang.disable_cache()
    return tilelang, DataType, T


def build_sparse_attention_h1_block_fwd(
    *,
    dim: int,
    topk: int,
    sm_scale: float | None = None,
    block_M: int = 16,
    block_I: int = 64,
    q_start_index_s: int = 0,
    pad_value: int = -1,
    dtype: str = "float16",
    core_num: int = 24,
    cache_device: object | None = None,
):
    """Build an isolated H=1 sparse attention kernel.

    Returned callable signature: ``kernel(q, k, v, indices) -> out``.

    Shapes:
      * ``q``: ``[B, S_q, 1, D]``
      * ``k``/``v``: ``[B, S_k, 1, D]``
      * ``indices``: ``[B, S_q, 1, topk]``

    First-stage constraints are deliberately narrow:
      * fp16 only
      * causal only
      * ``block_M`` must divide the runtime ``S_q``
      * indices must be block-sparse style, i.e. all rows inside one
        ``block_M`` query tile share the same K block list
    """

    if dtype != "float16":
        raise NotImplementedError("H=1 experimental kernel only supports dtype='float16'")
    if dim != 1 << (dim - 1).bit_length():
        raise ValueError(f"dim={dim} must be a power of two")
    if topk % block_I != 0:
        raise ValueError(f"topk={topk} must be divisible by block_I={block_I}")
    if block_M % 2 != 0 or block_M < 16:
        raise ValueError("block_M must be even and at least 16 for Ascend vector lanes")
    if pad_value != -1:
        raise NotImplementedError("H=1 experimental kernel only supports pad_value == -1")

    cache_key = (
        dim,
        topk,
        sm_scale,
        block_M,
        block_I,
        q_start_index_s,
        pad_value,
        dtype,
        core_num,
        cache_device,
    )
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    tilelang, _DataType, T = _require_tilelang()

    pass_configs = {
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    }

    @tilelang.jit(out_idx=[4], workspace_idx=[5, 6, 7, 8, 9], pass_configs=pass_configs)
    def _kernel_factory(
        dim,
        topk,
        sm_scale=None,
        block_M=16,
        block_I=64,
        q_start_index_s=0,
        pad_value=-1,
        core_num=24,
    ):
        sm_scale = (1.0 / dim) ** 0.5 if sm_scale is None else sm_scale

        batch = T.symbolic("batch")
        seq_len = T.symbolic("seq_len")
        seq_len_kv = T.symbolic("seq_len_kv")

        q_shape = [batch, seq_len, 1, dim]
        kv_shape = [batch, seq_len_kv, 1, dim]
        indices_shape = [batch, seq_len, 1, topk]
        o_shape = [batch, seq_len, 1, dim]

        BM = block_M
        BI = block_I
        NI = tilelang.cdiv(topk, block_I)
        D = dim
        v_block = BM // 2
        dtype = "float16"
        accum_dtype = "float"

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, "float16"),  # type: ignore
            K: T.Tensor(kv_shape, "float16"),  # type: ignore
            V: T.Tensor(kv_shape, "float16"),  # type: ignore
            Indices: T.Tensor(indices_shape, "int32"),  # type: ignore
            Output: T.Tensor(o_shape, "float16"),  # type: ignore
            workspace_k: T.Tensor([core_num, BI, D], "float16"),
            workspace_v: T.Tensor([core_num, BI, D], "float16"),
            workspace_s: T.Tensor([core_num, BM, BI], "float"),
            workspace_p: T.Tensor([core_num, BM, BI], "float16"),
            workspace_o: T.Tensor([core_num, BM, D], "float"),
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                q_l1 = T.alloc_L1([BM, D], dtype)
                k_l1 = T.alloc_L1([BI, D], dtype)
                v_l1 = T.alloc_L1([BI, D], dtype)
                p_l1 = T.alloc_L1([BM, BI], dtype)

                acc_s_l0c = T.alloc_L0C([BM, BI], accum_dtype)
                acc_o_l0c = T.alloc_L0C([BM, D], accum_dtype)

                acc_o = T.alloc_ub([v_block, D], accum_dtype)
                sumexp = T.alloc_ub([v_block], accum_dtype)
                m_i = T.alloc_ub([v_block], accum_dtype)
                m_i_prev = T.alloc_ub([v_block], accum_dtype)
                sumexp_i = T.alloc_ub([v_block], accum_dtype)

                indices_ub = T.alloc_ub([BI], "int32")
                indices_float = T.alloc_ub([BI], "float")
                k_ub_gather = T.alloc_ub([BI // 2, D], dtype)
                v_ub_gather = T.alloc_ub([BI // 2, D], dtype)

                acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
                acc_s_mask = T.alloc_ub([v_block, BI], accum_dtype)
                acc_s_from_cube = T.alloc_ub([v_block, BI], accum_dtype)
                acc_p_half = T.alloc_ub([v_block, BI], dtype)
                acc_o_ub = T.alloc_ub([v_block, D], accum_dtype)
                acc_o_half = T.alloc_ub([v_block, D], dtype)
                mask_ub = T.alloc_ub([BI // 8], "uint8")
                mask_pad_ub = T.alloc_ub([BI // 8], "uint8")

                q_tiles = T.ceildiv(seq_len, BM)
                for core_index in T.serial(T.ceildiv(q_tiles * batch, core_num)):
                    pid = core_index * core_num + cid
                    if pid < q_tiles * batch:
                        tile_i = pid % q_tiles
                        b_i = pid // q_tiles
                        s_base = tile_i * BM

                        with T.Scope("C"):
                            T.copy(Q[b_i, s_base : s_base + BM, 0, :D], q_l1)
                            T.barrier_all()

                            for _ in T.serial(NI):
                                T.wait_cross_flag(0)
                                T.barrier_all()
                                T.copy(workspace_k[cid, 0:BI, 0:D], k_l1)
                                T.barrier_all()

                                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                                T.barrier_all()

                                T.copy(acc_s_l0c, workspace_s[cid, 0:BM, 0:BI])
                                T.barrier_all()
                                T.set_cross_flag("FIX", 1)

                                T.wait_cross_flag(2)
                                T.barrier_all()

                                T.copy(workspace_p[cid, 0:BM, 0:BI], p_l1)
                                T.copy(workspace_v[cid, 0:BI, 0:D], v_l1)
                                T.barrier_all()

                                T.gemm_v0(p_l1, v_l1, acc_o_l0c, init=True)
                                T.barrier_all()

                                T.copy(acc_o_l0c, workspace_o[cid, 0:BM, 0:D])
                                T.barrier_all()

                                T.set_cross_flag("FIX", 3)
                                T.wait_cross_flag(4)

                        with T.Scope("V"):
                            T.tile.fill(acc_o, 0.0)
                            T.tile.fill(sumexp, 0.0)
                            T.tile.fill(m_i, -(2.0**30))
                            T.barrier_all()

                            for i_i in range(NI):
                                T.copy(Indices[b_i, s_base, 0, i_i * BI : i_i * BI + BI], indices_ub)
                                T.barrier_all()
                                T.copy(indices_ub, indices_float)
                                T.barrier_all()

                                block_start = indices_ub[0]
                                T.copy(
                                    K[
                                        b_i,
                                        block_start + vid * BI // 2 : block_start + (vid + 1) * BI // 2,
                                        0,
                                        :D,
                                    ],
                                    k_ub_gather,
                                )
                                T.copy(
                                    V[
                                        b_i,
                                        block_start + vid * BI // 2 : block_start + (vid + 1) * BI // 2,
                                        0,
                                        :D,
                                    ],
                                    v_ub_gather,
                                )
                                T.barrier_all()
                                T.copy(
                                    k_ub_gather,
                                    workspace_k[cid, vid * BI // 2 : (vid + 1) * BI // 2, :],
                                )
                                T.copy(
                                    v_ub_gather,
                                    workspace_v[cid, vid * BI // 2 : (vid + 1) * BI // 2, :],
                                )
                                T.barrier_all()

                                T.set_cross_flag("MTE3", 0)

                                T.tile.fill(acc_s_mask, 0.0)
                                T.copy(m_i, m_i_prev)
                                T.barrier_all()

                                T.wait_cross_flag(1)
                                T.copy(
                                    workspace_s[cid, vid * v_block : vid * v_block + v_block, :],
                                    acc_s_from_cube,
                                )
                                T.barrier_all()

                                for row in T.serial(v_block):
                                    q_abs = s_base + vid * v_block + row + q_start_index_s
                                    T.tile.compare(mask_ub, indices_float, T.float32(q_abs), "LE")
                                    T.tile.compare(
                                        mask_pad_ub,
                                        indices_float,
                                        T.float32(pad_value),
                                        "NE",
                                    )
                                    T.tile.bitwise_and(mask_ub, mask_ub, mask_pad_ub)
                                    T.tile.select(
                                        acc_s_ub[row, :],
                                        mask_ub,
                                        acc_s_from_cube[row, :],
                                        -T.infinity(accum_dtype),
                                        "VSEL_TENSOR_SCALAR_MODE",
                                    )
                                    T.barrier_all()

                                T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                                T.barrier_all()

                                T.reduce_max(acc_s_ub, m_i, dim=-1)
                                T.tile.max(m_i, m_i, m_i_prev)
                                T.tile.sub(m_i_prev, m_i_prev, m_i)
                                T.tile.exp(m_i_prev, m_i_prev)
                                T.barrier_all()

                                for row in range(v_block):
                                    T.tile.sub(acc_s_ub[row, :], acc_s_ub[row, :], m_i[row])
                                    T.barrier_all()

                                T.tile.exp(acc_s_ub, acc_s_ub)
                                T.reduce_sum(acc_s_ub, sumexp_i, dim=-1)
                                T.tile.mul(sumexp, sumexp, m_i_prev)
                                T.tile.add(sumexp, sumexp, sumexp_i)
                                T.barrier_all()

                                for row in range(v_block):
                                    T.tile.mul(acc_o[row, :], acc_o[row, :], m_i_prev[row])
                                    T.barrier_all()

                                T.copy(acc_s_ub, acc_p_half)
                                T.copy(
                                    acc_p_half,
                                    workspace_p[cid, vid * v_block : vid * v_block + v_block, :],
                                )
                                T.barrier_all()

                                T.set_cross_flag("MTE3", 2)

                                T.wait_cross_flag(3)
                                T.copy(
                                    workspace_o[cid, vid * v_block : vid * v_block + v_block, :],
                                    acc_o_ub,
                                )
                                T.tile.add(acc_o, acc_o, acc_o_ub)
                                T.barrier_all()

                                T.set_cross_flag("V", 4)

                            for row in range(v_block):
                                T.tile.div(acc_o[row, :], acc_o[row, :], sumexp[row])
                                T.barrier_all()

                            T.copy(acc_o, acc_o_half)
                            T.copy(
                                acc_o_half,
                                Output[
                                    b_i,
                                    s_base + vid * v_block : s_base + (vid + 1) * v_block,
                                    0,
                                    :D,
                                ],
                            )

        return main

    kernel = _kernel_factory(
        dim=dim,
        topk=topk,
        sm_scale=sm_scale,
        block_M=block_M,
        block_I=block_I,
        q_start_index_s=q_start_index_s,
        pad_value=pad_value,
        core_num=core_num,
    )
    _KERNEL_CACHE[cache_key] = kernel
    return kernel


def build_sparse_attention_h1_block_index_fwd(
    *,
    dim: int,
    topk_blocks: int,
    sm_scale: float | None = None,
    block_M: int = 16,
    block_I: int = 64,
    q_start_index_s: int = 0,
    dtype: str = "float16",
    core_num: int = 24,
    cache_device: object | None = None,
):
    """Build an H=1 sparse attention kernel that consumes block indices.

    Returned callable signature: ``kernel(q, k, v, block_indices) -> out``.

    Shapes:
      * ``q``: ``[B, S_q, 1, D]``
      * ``k``/``v``: ``[B, S_k, 1, D]``
      * ``block_indices``: ``[B, n_q_blocks, 1, topk_blocks]``

    ``block_indices`` are K-block ids, not token ids. The kernel expands each
    selected block to ``block_I`` contiguous K/V tokens internally, avoiding
    the wrapper-side ``[B, S_q, 1, topk_blocks * block_I]`` materialization.
    """

    if dtype != "float16":
        raise NotImplementedError("H=1 block-index kernel only supports dtype='float16'")
    if dim != 1 << (dim - 1).bit_length():
        raise ValueError(f"dim={dim} must be a power of two")
    if topk_blocks <= 0:
        raise ValueError(f"topk_blocks={topk_blocks} must be positive")
    if block_M % 2 != 0 or block_M < 16:
        raise ValueError("block_M must be even and at least 16 for Ascend vector lanes")
    if block_I % block_M != 0:
        raise ValueError(f"block_I={block_I} must be divisible by block_M={block_M}")

    cache_key = (
        "block_index",
        dim,
        topk_blocks,
        sm_scale,
        block_M,
        block_I,
        q_start_index_s,
        dtype,
        core_num,
        cache_device,
    )
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    tilelang, _DataType, T = _require_tilelang()

    pass_configs = {
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    }

    @tilelang.jit(out_idx=[4], workspace_idx=[5, 6, 7, 8, 9], pass_configs=pass_configs)
    def _kernel_factory(
        dim,
        topk_blocks,
        sm_scale=None,
        block_M=16,
        block_I=64,
        q_start_index_s=0,
        core_num=24,
    ):
        sm_scale = (1.0 / dim) ** 0.5 if sm_scale is None else sm_scale

        batch = T.symbolic("batch")
        seq_len = T.symbolic("seq_len")
        seq_len_kv = T.symbolic("seq_len_kv")
        q_blocks = T.symbolic("q_blocks")

        q_shape = [batch, seq_len, 1, dim]
        kv_shape = [batch, seq_len_kv, 1, dim]
        block_indices_shape = [batch, q_blocks, 1, topk_blocks]
        o_shape = [batch, seq_len, 1, dim]

        BM = block_M
        BI = block_I
        NI = topk_blocks
        D = dim
        v_block = BM // 2
        dtype = "float16"
        accum_dtype = "float"

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, "float16"),  # type: ignore
            K: T.Tensor(kv_shape, "float16"),  # type: ignore
            V: T.Tensor(kv_shape, "float16"),  # type: ignore
            BlockIndices: T.Tensor(block_indices_shape, "int32"),  # type: ignore
            Output: T.Tensor(o_shape, "float16"),  # type: ignore
            workspace_k: T.Tensor([core_num, BI, D], "float16"),
            workspace_v: T.Tensor([core_num, BI, D], "float16"),
            workspace_s: T.Tensor([core_num, BM, BI], "float"),
            workspace_p: T.Tensor([core_num, BM, BI], "float16"),
            workspace_o: T.Tensor([core_num, BM, D], "float"),
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                q_l1 = T.alloc_L1([BM, D], dtype)
                k_l1 = T.alloc_L1([BI, D], dtype)
                v_l1 = T.alloc_L1([BI, D], dtype)
                p_l1 = T.alloc_L1([BM, BI], dtype)

                acc_s_l0c = T.alloc_L0C([BM, BI], accum_dtype)
                acc_o_l0c = T.alloc_L0C([BM, D], accum_dtype)

                acc_o = T.alloc_ub([v_block, D], accum_dtype)
                sumexp = T.alloc_ub([v_block], accum_dtype)
                m_i = T.alloc_ub([v_block], accum_dtype)
                m_i_prev = T.alloc_ub([v_block], accum_dtype)
                sumexp_i = T.alloc_ub([v_block], accum_dtype)

                indices_ub = T.alloc_ub([BI], "int32")
                indices_float = T.alloc_ub([BI], "float")
                k_ub_gather = T.alloc_ub([BI // 2, D], dtype)
                v_ub_gather = T.alloc_ub([BI // 2, D], dtype)

                acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
                acc_s_mask = T.alloc_ub([v_block, BI], accum_dtype)
                acc_s_from_cube = T.alloc_ub([v_block, BI], accum_dtype)
                acc_p_half = T.alloc_ub([v_block, BI], dtype)
                acc_o_ub = T.alloc_ub([v_block, D], accum_dtype)
                acc_o_half = T.alloc_ub([v_block, D], dtype)
                mask_ub = T.alloc_ub([BI // 8], "uint8")

                q_tiles = T.ceildiv(seq_len, BM)
                for core_index in T.serial(T.ceildiv(q_tiles * batch, core_num)):
                    pid = core_index * core_num + cid
                    if pid < q_tiles * batch:
                        tile_i = pid % q_tiles
                        b_i = pid // q_tiles
                        s_base = tile_i * BM
                        q_block_i = s_base // BI

                        with T.Scope("C"):
                            T.copy(Q[b_i, s_base : s_base + BM, 0, :D], q_l1)
                            T.barrier_all()

                            for _ in T.serial(NI):
                                T.wait_cross_flag(0)
                                T.barrier_all()
                                T.copy(workspace_k[cid, 0:BI, 0:D], k_l1)
                                T.barrier_all()

                                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                                T.barrier_all()

                                T.copy(acc_s_l0c, workspace_s[cid, 0:BM, 0:BI])
                                T.barrier_all()
                                T.set_cross_flag("FIX", 1)

                                T.wait_cross_flag(2)
                                T.barrier_all()

                                T.copy(workspace_p[cid, 0:BM, 0:BI], p_l1)
                                T.copy(workspace_v[cid, 0:BI, 0:D], v_l1)
                                T.barrier_all()

                                T.gemm_v0(p_l1, v_l1, acc_o_l0c, init=True)
                                T.barrier_all()

                                T.copy(acc_o_l0c, workspace_o[cid, 0:BM, 0:D])
                                T.barrier_all()

                                T.set_cross_flag("FIX", 3)
                                T.wait_cross_flag(4)

                        with T.Scope("V"):
                            T.tile.fill(acc_o, 0.0)
                            T.tile.fill(sumexp, 0.0)
                            T.tile.fill(m_i, -(2.0**30))
                            T.barrier_all()

                            for i_i in range(NI):
                                block_start = BlockIndices[b_i, q_block_i, 0, i_i] * BI

                                T.copy(
                                    K[
                                        b_i,
                                        block_start + vid * BI // 2 : block_start + (vid + 1) * BI // 2,
                                        0,
                                        :D,
                                    ],
                                    k_ub_gather,
                                )
                                T.copy(
                                    V[
                                        b_i,
                                        block_start + vid * BI // 2 : block_start + (vid + 1) * BI // 2,
                                        0,
                                        :D,
                                    ],
                                    v_ub_gather,
                                )
                                T.barrier_all()
                                T.copy(
                                    k_ub_gather,
                                    workspace_k[cid, vid * BI // 2 : (vid + 1) * BI // 2, :],
                                )
                                T.copy(
                                    v_ub_gather,
                                    workspace_v[cid, vid * BI // 2 : (vid + 1) * BI // 2, :],
                                )
                                T.barrier_all()

                                T.set_cross_flag("MTE3", 0)

                                T.tile.fill(acc_s_mask, 0.0)
                                T.copy(m_i, m_i_prev)
                                T.barrier_all()

                                T.wait_cross_flag(1)
                                T.copy(
                                    workspace_s[cid, vid * v_block : vid * v_block + v_block, :],
                                    acc_s_from_cube,
                                )
                                T.barrier_all()

                                if block_start + BI - 1 <= s_base + vid * v_block + q_start_index_s:
                                    T.copy(acc_s_from_cube, acc_s_ub)
                                else:
                                    for col in T.serial(BI):
                                        indices_ub[col] = block_start + col
                                    T.barrier_all()
                                    T.copy(indices_ub, indices_float)
                                    T.barrier_all()

                                    for row in T.serial(v_block):
                                        q_abs = s_base + vid * v_block + row + q_start_index_s
                                        T.tile.compare(mask_ub, indices_float, T.float32(q_abs), "LE")
                                        T.tile.select(
                                            acc_s_ub[row, :],
                                            mask_ub,
                                            acc_s_from_cube[row, :],
                                            -T.infinity(accum_dtype),
                                            "VSEL_TENSOR_SCALAR_MODE",
                                        )
                                        T.barrier_all()

                                T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                                T.barrier_all()

                                T.reduce_max(acc_s_ub, m_i, dim=-1)
                                T.tile.max(m_i, m_i, m_i_prev)
                                T.tile.sub(m_i_prev, m_i_prev, m_i)
                                T.tile.exp(m_i_prev, m_i_prev)
                                T.barrier_all()

                                for row in range(v_block):
                                    T.tile.sub(acc_s_ub[row, :], acc_s_ub[row, :], m_i[row])
                                    T.barrier_all()

                                T.tile.exp(acc_s_ub, acc_s_ub)
                                T.reduce_sum(acc_s_ub, sumexp_i, dim=-1)
                                T.tile.mul(sumexp, sumexp, m_i_prev)
                                T.tile.add(sumexp, sumexp, sumexp_i)
                                T.barrier_all()

                                for row in range(v_block):
                                    T.tile.mul(acc_o[row, :], acc_o[row, :], m_i_prev[row])
                                    T.barrier_all()

                                T.copy(acc_s_ub, acc_p_half)
                                T.copy(
                                    acc_p_half,
                                    workspace_p[cid, vid * v_block : vid * v_block + v_block, :],
                                )
                                T.barrier_all()

                                T.set_cross_flag("MTE3", 2)

                                T.wait_cross_flag(3)
                                T.copy(
                                    workspace_o[cid, vid * v_block : vid * v_block + v_block, :],
                                    acc_o_ub,
                                )
                                T.tile.add(acc_o, acc_o, acc_o_ub)
                                T.barrier_all()

                                T.set_cross_flag("V", 4)

                            for row in range(v_block):
                                T.tile.div(acc_o[row, :], acc_o[row, :], sumexp[row])
                                T.barrier_all()

                            T.copy(acc_o, acc_o_half)
                            T.copy(
                                acc_o_half,
                                Output[
                                    b_i,
                                    s_base + vid * v_block : s_base + (vid + 1) * v_block,
                                    0,
                                    :D,
                                ],
                            )

        return main

    kernel = _kernel_factory(
        dim=dim,
        topk_blocks=topk_blocks,
        sm_scale=sm_scale,
        block_M=block_M,
        block_I=block_I,
        q_start_index_s=q_start_index_s,
        core_num=core_num,
    )
    _KERNEL_CACHE[cache_key] = kernel
    return kernel


def build_sparse_attention_mh_block_index_fwd(
    *,
    dim: int,
    heads: int,
    topk_blocks: int,
    sm_scale: float | None = None,
    block_M: int = 16,
    block_I: int = 64,
    q_start_index_s: int = 0,
    dtype: str = "float16",
    core_num: int = 24,
    cache_device: object | None = None,
):
    """Build a multi-head sparse-attention TileLang kernel that consumes block indices.

    Callable signature: ``kernel(q, k, v, block_indices) -> out``.

    Shapes:
      * ``q``: ``[B, S_q, H, D]``
      * ``k``/``v``: ``[B, S_k, H, D]``  (MHA only; ``H_kv = H`` for this MVP)
      * ``block_indices``: ``[B, n_q_blocks, H, topk_blocks]``

    ``heads`` 是 **compile-time** 参数（不是 symbolic），原因：
    早期把 ``heads`` 作为 symbolic 时，``q_tiles * heads * batch`` 的三层符号相乘 +
    modulo 让 Ascend kernel 出现全 NaN 输出（isolated smoke 4/4 MH cases 全失败）。
    改为 compile-time int 后 stride 与 work unit 分解都是编译期常量，且 ``cache_key``
    带上 ``heads`` 实现按 H 分别 JIT，运行时小开销。

    Differences vs ``build_sparse_attention_h1_block_index_fwd``:
      * 不再依赖 wrapper 把 ``B*H`` 折到 batch 维。kernel 直接接 BSHD 自然形态。
      * 工作单元分解为 ``(b, h, q_tile)``，``q_tiles*heads`` 在 JIT 期已是 ``int*symbolic``，
        不会出现 symbolic*symbolic 的多层嵌套。
      * cube/vector pipeline 与 H=1 block-index kernel 完全一致；每个 work item
        仍处理 ``block_M=16`` query token × 1 head。这一版只解 wrapper 侧的
        fold-into-batch + head_chunk 开销；后续若需进一步把多 head 堆到一次 cube
        计算里，是 Tier 2 的改造。
    """

    if dtype != "float16":
        raise NotImplementedError("MH block-index kernel only supports dtype='float16'")
    if dim != 1 << (dim - 1).bit_length():
        raise ValueError(f"dim={dim} must be a power of two")
    if topk_blocks <= 0:
        raise ValueError(f"topk_blocks={topk_blocks} must be positive")
    if heads <= 0:
        raise ValueError(f"heads={heads} must be positive")
    if block_M % 2 != 0 or block_M < 16:
        raise ValueError("block_M must be even and at least 16 for Ascend vector lanes")
    if block_I % block_M != 0:
        raise ValueError(f"block_I={block_I} must be divisible by block_M={block_M}")

    cache_key = (
        "mh_block_index",
        dim,
        heads,
        topk_blocks,
        sm_scale,
        block_M,
        block_I,
        q_start_index_s,
        dtype,
        core_num,
        cache_device,
    )
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    tilelang, _DataType, T = _require_tilelang()

    pass_configs = {
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    }

    @tilelang.jit(out_idx=[4], workspace_idx=[5, 6, 7, 8, 9], pass_configs=pass_configs)
    def _kernel_factory(
        dim,
        heads,
        topk_blocks,
        sm_scale=None,
        block_M=16,
        block_I=64,
        q_start_index_s=0,
        core_num=24,
    ):
        sm_scale = (1.0 / dim) ** 0.5 if sm_scale is None else sm_scale

        batch = T.symbolic("batch")
        seq_len = T.symbolic("seq_len")
        seq_len_kv = T.symbolic("seq_len_kv")
        q_blocks = T.symbolic("q_blocks")
        # ``heads`` 是闭包外的 compile-time int，下面 shape / 工作单元都用 int 字面量参与。
        H = heads

        q_shape = [batch, seq_len, H, dim]
        kv_shape = [batch, seq_len_kv, H, dim]
        block_indices_shape = [batch, q_blocks, H, topk_blocks]
        o_shape = [batch, seq_len, H, dim]

        BM = block_M
        BI = block_I
        NI = topk_blocks
        D = dim
        v_block = BM // 2
        dtype = "float16"
        accum_dtype = "float"

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, "float16"),  # type: ignore
            K: T.Tensor(kv_shape, "float16"),  # type: ignore
            V: T.Tensor(kv_shape, "float16"),  # type: ignore
            BlockIndices: T.Tensor(block_indices_shape, "int32"),  # type: ignore
            Output: T.Tensor(o_shape, "float16"),  # type: ignore
            workspace_k: T.Tensor([core_num, BI, D], "float16"),
            workspace_v: T.Tensor([core_num, BI, D], "float16"),
            workspace_s: T.Tensor([core_num, BM, BI], "float"),
            workspace_p: T.Tensor([core_num, BM, BI], "float16"),
            workspace_o: T.Tensor([core_num, BM, D], "float"),
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                q_l1 = T.alloc_L1([BM, D], dtype)
                k_l1 = T.alloc_L1([BI, D], dtype)
                v_l1 = T.alloc_L1([BI, D], dtype)
                p_l1 = T.alloc_L1([BM, BI], dtype)

                acc_s_l0c = T.alloc_L0C([BM, BI], accum_dtype)
                acc_o_l0c = T.alloc_L0C([BM, D], accum_dtype)

                acc_o = T.alloc_ub([v_block, D], accum_dtype)
                sumexp = T.alloc_ub([v_block], accum_dtype)
                m_i = T.alloc_ub([v_block], accum_dtype)
                m_i_prev = T.alloc_ub([v_block], accum_dtype)
                sumexp_i = T.alloc_ub([v_block], accum_dtype)

                indices_ub = T.alloc_ub([BI], "int32")
                indices_float = T.alloc_ub([BI], "float")
                k_ub_gather = T.alloc_ub([BI // 2, D], dtype)
                v_ub_gather = T.alloc_ub([BI // 2, D], dtype)

                acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
                acc_s_mask = T.alloc_ub([v_block, BI], accum_dtype)
                acc_s_from_cube = T.alloc_ub([v_block, BI], accum_dtype)
                acc_p_half = T.alloc_ub([v_block, BI], dtype)
                acc_o_ub = T.alloc_ub([v_block, D], accum_dtype)
                acc_o_half = T.alloc_ub([v_block, D], dtype)
                mask_ub = T.alloc_ub([BI // 8], "uint8")

                q_tiles = T.ceildiv(seq_len, BM)
                bh_tiles = q_tiles * H  # int * symbolic → symbolic（同 H=1 的 q_tiles * batch）
                total_tiles = bh_tiles * batch
                for core_index in T.serial(T.ceildiv(total_tiles, core_num)):
                    pid = core_index * core_num + cid
                    if pid < total_tiles:
                        tile_i = pid % q_tiles
                        h_i = (pid // q_tiles) % H
                        b_i = pid // bh_tiles
                        s_base = tile_i * BM
                        q_block_i = s_base // BI

                        with T.Scope("C"):
                            T.copy(Q[b_i, s_base : s_base + BM, h_i, :D], q_l1)
                            T.barrier_all()

                            for _ in T.serial(NI):
                                T.wait_cross_flag(0)
                                T.barrier_all()
                                T.copy(workspace_k[cid, 0:BI, 0:D], k_l1)
                                T.barrier_all()

                                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                                T.barrier_all()

                                T.copy(acc_s_l0c, workspace_s[cid, 0:BM, 0:BI])
                                T.barrier_all()
                                T.set_cross_flag("FIX", 1)

                                T.wait_cross_flag(2)
                                T.barrier_all()

                                T.copy(workspace_p[cid, 0:BM, 0:BI], p_l1)
                                T.copy(workspace_v[cid, 0:BI, 0:D], v_l1)
                                T.barrier_all()

                                T.gemm_v0(p_l1, v_l1, acc_o_l0c, init=True)
                                T.barrier_all()

                                T.copy(acc_o_l0c, workspace_o[cid, 0:BM, 0:D])
                                T.barrier_all()

                                T.set_cross_flag("FIX", 3)
                                T.wait_cross_flag(4)

                        with T.Scope("V"):
                            T.tile.fill(acc_o, 0.0)
                            T.tile.fill(sumexp, 0.0)
                            T.tile.fill(m_i, -(2.0**30))
                            T.barrier_all()

                            for i_i in range(NI):
                                block_start = BlockIndices[b_i, q_block_i, h_i, i_i] * BI

                                T.copy(
                                    K[
                                        b_i,
                                        block_start + vid * BI // 2 : block_start + (vid + 1) * BI // 2,
                                        h_i,
                                        :D,
                                    ],
                                    k_ub_gather,
                                )
                                T.copy(
                                    V[
                                        b_i,
                                        block_start + vid * BI // 2 : block_start + (vid + 1) * BI // 2,
                                        h_i,
                                        :D,
                                    ],
                                    v_ub_gather,
                                )
                                T.barrier_all()
                                T.copy(
                                    k_ub_gather,
                                    workspace_k[cid, vid * BI // 2 : (vid + 1) * BI // 2, :],
                                )
                                T.copy(
                                    v_ub_gather,
                                    workspace_v[cid, vid * BI // 2 : (vid + 1) * BI // 2, :],
                                )
                                T.barrier_all()

                                T.set_cross_flag("MTE3", 0)

                                T.tile.fill(acc_s_mask, 0.0)
                                T.copy(m_i, m_i_prev)
                                T.barrier_all()

                                T.wait_cross_flag(1)
                                T.copy(
                                    workspace_s[cid, vid * v_block : vid * v_block + v_block, :],
                                    acc_s_from_cube,
                                )
                                T.barrier_all()

                                if block_start + BI - 1 <= s_base + vid * v_block + q_start_index_s:
                                    T.copy(acc_s_from_cube, acc_s_ub)
                                else:
                                    for col in T.serial(BI):
                                        indices_ub[col] = block_start + col
                                    T.barrier_all()
                                    T.copy(indices_ub, indices_float)
                                    T.barrier_all()

                                    for row in T.serial(v_block):
                                        q_abs = s_base + vid * v_block + row + q_start_index_s
                                        T.tile.compare(mask_ub, indices_float, T.float32(q_abs), "LE")
                                        T.tile.select(
                                            acc_s_ub[row, :],
                                            mask_ub,
                                            acc_s_from_cube[row, :],
                                            -T.infinity(accum_dtype),
                                            "VSEL_TENSOR_SCALAR_MODE",
                                        )
                                        T.barrier_all()

                                T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                                T.barrier_all()

                                T.reduce_max(acc_s_ub, m_i, dim=-1)
                                T.tile.max(m_i, m_i, m_i_prev)
                                T.tile.sub(m_i_prev, m_i_prev, m_i)
                                T.tile.exp(m_i_prev, m_i_prev)
                                T.barrier_all()

                                for row in range(v_block):
                                    T.tile.sub(acc_s_ub[row, :], acc_s_ub[row, :], m_i[row])
                                    T.barrier_all()

                                T.tile.exp(acc_s_ub, acc_s_ub)
                                T.reduce_sum(acc_s_ub, sumexp_i, dim=-1)
                                T.tile.mul(sumexp, sumexp, m_i_prev)
                                T.tile.add(sumexp, sumexp, sumexp_i)
                                T.barrier_all()

                                for row in range(v_block):
                                    T.tile.mul(acc_o[row, :], acc_o[row, :], m_i_prev[row])
                                    T.barrier_all()

                                T.copy(acc_s_ub, acc_p_half)
                                T.copy(
                                    acc_p_half,
                                    workspace_p[cid, vid * v_block : vid * v_block + v_block, :],
                                )
                                T.barrier_all()

                                T.set_cross_flag("MTE3", 2)

                                T.wait_cross_flag(3)
                                T.copy(
                                    workspace_o[cid, vid * v_block : vid * v_block + v_block, :],
                                    acc_o_ub,
                                )
                                T.tile.add(acc_o, acc_o, acc_o_ub)
                                T.barrier_all()

                                T.set_cross_flag("V", 4)

                            for row in range(v_block):
                                T.tile.div(acc_o[row, :], acc_o[row, :], sumexp[row])
                                T.barrier_all()

                            T.copy(acc_o, acc_o_half)
                            T.copy(
                                acc_o_half,
                                Output[
                                    b_i,
                                    s_base + vid * v_block : s_base + (vid + 1) * v_block,
                                    h_i,
                                    :D,
                                ],
                            )

        return main

    kernel = _kernel_factory(
        dim=dim,
        heads=heads,
        topk_blocks=topk_blocks,
        sm_scale=sm_scale,
        block_M=block_M,
        block_I=block_I,
        q_start_index_s=q_start_index_s,
        core_num=core_num,
    )
    _KERNEL_CACHE[cache_key] = kernel
    return kernel
