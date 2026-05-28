# Copyright (c) 2026 (NPU adapter - PR-4 TileLang sparse attention)
# Licensed under The MIT License [see LICENSE for details]
"""TileLang sparse attention MVP with separate Q/K/V tensors.

This is the first PR-4 path-B production-shaped kernel.  It deliberately
keeps the scope narrow:

* BSHD layout: Q ``[B, S_q, H, D]``, K/V ``[B, S_k, kv_group, D]``.
* ``kv_group == 1`` only, so all Q heads share one sparse index list.
* causal fp16 forward only.
* ``indices`` is ``[B, S_q, kv_group, topk]`` and may use ``-1`` as a pad
  sentinel.  Pad slots are masked out and never used for K/V loads.

The implementation is adapted from tilelang-ascend's sparse flash attention
examples, but unlike the official NSA/DeepSeek-style example it does not pack
K and V into the same tensor.
"""

import torch

__all__ = [
    "build_sparse_attention_qkv_fwd",
    "clear_sparse_attention_qkv_kernel_cache",
    "sparse_attention_qkv_reference",
]


_KERNEL_CACHE: dict[tuple[object, ...], object] = {}


def clear_sparse_attention_qkv_kernel_cache() -> None:
    """Clear compiled TileLang sparse-attention callables cached by this module."""
    _KERNEL_CACHE.clear()


def _require_tilelang():
    import tilelang  # type: ignore[import-not-found]
    from tilelang import DataType, language as T  # type: ignore[import-not-found]

    tilelang.disable_cache()
    return tilelang, DataType, T


def build_sparse_attention_qkv_fwd(
    *,
    heads: int,
    dim: int,
    topk: int,
    kv_group: int = 1,
    sm_scale: float | None = None,
    is_causal: bool = True,
    block_I: int = 64,
    q_start_index_s: int = 0,
    pad_value: int = -1,
    dtype: str = "float16",
    core_num: int = 24,
    use_contiguous_range_load: bool = False,
    cache_device: object | None = None,
):
    """Build a TileLang sparse attention kernel for separate Q/K/V tensors.

    The returned callable has the signature ``kernel(q, k, v, indices)`` and
    returns ``out``.  Shapes are inferred dynamically by TileLang, while
    ``heads``, ``dim`` and ``topk`` are compile-time constants.
    """

    if kv_group != 1:
        raise NotImplementedError("PR-4 MVP only supports kv_group == 1")
    if not is_causal:
        raise NotImplementedError("PR-4 MVP only supports causal attention")
    if topk % block_I != 0:
        raise ValueError(f"topk={topk} must be divisible by block_I={block_I}")
    if dim != 1 << (dim - 1).bit_length():
        raise ValueError(f"dim={dim} must be a power of two for the TileLang MVP")
    if heads % kv_group != 0:
        raise ValueError(f"heads={heads} must be divisible by kv_group={kv_group}")
    if dtype != "float16":
        raise NotImplementedError("PR-4 MVP only supports dtype='float16'")
    if pad_value != -1:
        raise NotImplementedError("PR-4 MVP only supports pad_value == -1")
    cache_key = (
        heads,
        dim,
        topk,
        kv_group,
        sm_scale,
        is_causal,
        block_I,
        q_start_index_s,
        pad_value,
        dtype,
        core_num,
        use_contiguous_range_load,
        cache_device,
    )
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    tilelang, DataType, T = _require_tilelang()

    pass_configs = {
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    }

    @tilelang.jit(out_idx=[4], workspace_idx=[5, 6, 7, 8, 9], pass_configs=pass_configs)
    def _kernel_factory(
        heads,
        dim,
        topk,
        kv_group=1,
        sm_scale=None,
        block_I=64,
        q_start_index_s=0,
        pad_value=-1,
        core_num=24,
        use_contiguous_range_load=False,
    ):
        sm_scale = (1.0 / dim) ** 0.5 if sm_scale is None else sm_scale

        batch = T.symbolic("batch")
        seq_len = T.symbolic("seq_len")
        seq_len_kv = T.symbolic("seq_len_kv")

        head_kv = heads // kv_group
        q_shape = [batch, seq_len, heads, dim]
        kv_shape = [batch, seq_len_kv, kv_group, dim]
        o_shape = [batch, seq_len, heads, dim]
        indices_shape = [batch, seq_len, kv_group, topk]
        indices_dtype = "int32"
        dtype = "float16"
        accum_dtype = "float"

        BI = block_I
        NI = tilelang.cdiv(topk, block_I)
        D = dim

        padded_head_kv = max(tilelang.math.next_power_of_2(head_kv), 16)
        if padded_head_kv != head_kv and kv_group != 1:
            raise AssertionError("automatic small-H padding is only supported for kv_group == 1")

        if head_kv > 64:
            assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
            REPLICATE_H = head_kv // 64
        else:
            REPLICATE_H = 1

        H_per_block = padded_head_kv if REPLICATE_H == 1 else 64
        v_block = H_per_block // 2
        ub_len = max(32 // (DataType(accum_dtype).bits // 8), v_block)

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, "float16"),  # type: ignore
            K: T.Tensor(kv_shape, "float16"),  # type: ignore
            V: T.Tensor(kv_shape, "float16"),  # type: ignore
            Indices: T.Tensor(indices_shape, "int32"),  # type: ignore
            Output: T.Tensor(o_shape, "float16"),  # type: ignore
            workspace_k: T.Tensor([core_num, BI, D], "float16"),
            workspace_v: T.Tensor([core_num, BI, D], "float16"),
            workspace_s: T.Tensor([core_num, H_per_block, BI], "float"),
            workspace_p: T.Tensor([core_num, H_per_block, BI], "float16"),
            workspace_o: T.Tensor([core_num, H_per_block, D], "float"),
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                q_l1 = T.alloc_L1([H_per_block, D], dtype)
                k_l1 = T.alloc_L1([BI, D], dtype)
                v_l1 = T.alloc_L1([BI, D], dtype)
                acc_s_l1 = T.alloc_L1([H_per_block, BI], dtype)

                acc_s_l0c = T.alloc_L0C([H_per_block, BI], accum_dtype)
                acc_o_l0c = T.alloc_L0C([H_per_block, D], accum_dtype)

                acc_o = T.alloc_ub([v_block, D], accum_dtype)
                sumexp = T.alloc_ub([ub_len], accum_dtype)
                m_i = T.alloc_ub([ub_len], accum_dtype)
                indices_ub = T.alloc_ub([BI], indices_dtype)
                indices_float = T.alloc_ub([BI], "float")
                k_ub = T.alloc_ub([D], dtype)
                v_ub = T.alloc_ub([D], dtype)
                k_ub_gather = T.alloc_ub([BI // 2, D], dtype)
                v_ub_gather = T.alloc_ub([BI // 2, D], dtype)
                acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
                acc_s_zero = T.alloc_ub([v_block, BI], accum_dtype)
                m_i_prev = T.alloc_ub([ub_len], accum_dtype)
                acc_s_from_cube = T.alloc_ub([v_block, BI], accum_dtype)
                sumexp_i = T.alloc_ub([ub_len], accum_dtype)
                acc_p_half = T.alloc_ub([v_block, BI], dtype)
                acc_o_ub = T.alloc_ub([v_block, D], accum_dtype)
                acc_o_half = T.alloc_ub([v_block, D], dtype)
                mask_ub = T.alloc_ub([BI // 8], "uint8")
                mask_pad_ub = T.alloc_ub([BI // 8], "uint8")

                for core_index in T.serial(T.ceildiv(seq_len * REPLICATE_H * batch * kv_group, core_num)):
                    pid = core_index * core_num + cid
                    if pid < seq_len * REPLICATE_H * batch * kv_group:
                        bx = pid % (seq_len * REPLICATE_H)
                        by = pid // (seq_len * REPLICATE_H) % batch
                        bz = pid // (seq_len * REPLICATE_H) // batch % kv_group

                        b_i = by
                        g_i = bz
                        s_i = bx // REPLICATE_H
                        h_i = bx % REPLICATE_H

                        heads_per_group = heads // kv_group
                        H0 = g_i * padded_head_kv
                        H1 = H0 + H_per_block

                        if REPLICATE_H != 1:
                            H0 = g_i * heads_per_group + (bx % REPLICATE_H) * H_per_block
                            H1 = H0 + H_per_block

                        with T.Scope("C"):
                            T.copy(Q[b_i, s_i, H0:H1, :D], q_l1)
                            T.barrier_all()
                            for _ in T.serial(NI):
                                T.wait_cross_flag(0)
                                T.barrier_all()
                                T.copy(workspace_k[cid, 0:BI, 0:D], k_l1)
                                T.barrier_all()

                                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                                T.barrier_all()

                                T.copy(acc_s_l0c, workspace_s[cid, 0:H_per_block, 0:BI])
                                T.barrier_all()
                                T.set_cross_flag("FIX", 1)

                                T.wait_cross_flag(2)
                                T.barrier_all()

                                T.copy(workspace_p[cid, 0:H_per_block, 0:BI], acc_s_l1)
                                T.copy(workspace_v[cid, 0:BI, 0:D], v_l1)
                                T.barrier_all()

                                T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                                T.barrier_all()

                                T.copy(acc_o_l0c, workspace_o[cid, 0:H_per_block, 0:D])
                                T.barrier_all()

                                T.set_cross_flag("FIX", 3)
                                T.wait_cross_flag(4)

                        with T.Scope("V"):
                            T.tile.fill(acc_o, 0.0)
                            T.tile.fill(sumexp, 0.0)
                            T.tile.fill(m_i, -(2.0**30))
                            T.barrier_all()

                            for i_i in range(NI):
                                T.copy(Indices[b_i, s_i, g_i, i_i * BI : i_i * BI + BI], indices_ub)
                                T.barrier_all()
                                T.copy(indices_ub, indices_float)
                                T.barrier_all()
                                T.tile.compare(
                                    mask_ub,
                                    indices_float,
                                    T.float32(s_i + q_start_index_s),
                                    "LE",
                                )
                                T.tile.compare(
                                    mask_pad_ub,
                                    indices_float,
                                    T.float32(pad_value),
                                    "NE",
                                )
                                T.barrier_all()
                                T.tile.bitwise_and(mask_ub, mask_ub, mask_pad_ub)
                                T.barrier_all()

                                if use_contiguous_range_load:
                                    block_start = indices_ub[0]
                                    T.copy(
                                        K[
                                            b_i,
                                            block_start + vid * BI // 2 : block_start + (vid + 1) * BI // 2,
                                            g_i,
                                            :D,
                                        ],
                                        k_ub_gather,
                                    )
                                    T.copy(
                                        V[
                                            b_i,
                                            block_start + vid * BI // 2 : block_start + (vid + 1) * BI // 2,
                                            g_i,
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
                                else:
                                    for bi_i in range(BI // 2):
                                        pos = indices_ub[bi_i + vid * BI // 2]
                                        if pos != pad_value:
                                            T.copy(K[b_i, pos, g_i, :D], k_ub)
                                            T.copy(V[b_i, pos, g_i, :D], v_ub)
                                        else:
                                            T.tile.fill(k_ub, 0.0)
                                            T.tile.fill(v_ub, 0.0)
                                        T.barrier_all()
                                        T.copy(k_ub, workspace_k[cid, bi_i + vid * BI // 2, :])
                                        T.copy(v_ub, workspace_v[cid, bi_i + vid * BI // 2, :])
                                        T.barrier_all()

                                T.set_cross_flag("MTE3", 0)

                                T.tile.fill(acc_s_zero, 0.0)
                                T.barrier_all()

                                for row in T.serial(v_block):
                                    T.tile.select(
                                        acc_s_ub[row, :],
                                        mask_ub,
                                        acc_s_zero[row, :],
                                        -T.infinity(accum_dtype),
                                        "VSEL_TENSOR_SCALAR_MODE",
                                    )
                                    T.barrier_all()

                                T.copy(m_i, m_i_prev)
                                T.barrier_all()

                                T.wait_cross_flag(1)
                                T.copy(
                                    workspace_s[cid, vid * v_block : vid * v_block + v_block, :],
                                    acc_s_from_cube,
                                )
                                T.barrier_all()

                                T.tile.add(acc_s_ub, acc_s_ub, acc_s_from_cube)
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
                                    s_i,
                                    H0 + vid * v_block : H0 + v_block + vid * v_block,
                                    :,
                                ],
                            )

        return main

    kernel = _kernel_factory(
        heads=heads,
        dim=dim,
        topk=topk,
        kv_group=kv_group,
        sm_scale=sm_scale,
        block_I=block_I,
        q_start_index_s=q_start_index_s,
        pad_value=pad_value,
        core_num=core_num,
        use_contiguous_range_load=use_contiguous_range_load,
    )
    _KERNEL_CACHE[cache_key] = kernel
    return kernel


def sparse_attention_qkv_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    *,
    q_start_index_s: int = 0,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """PyTorch reference for ``build_sparse_attention_qkv_fwd``.

    ``q`` is BSHD, ``k``/``v`` are BSGD and ``indices`` is BSgtopk.  The MVP
    supports ``G == 1``; the implementation keeps the group dimension to match
    TileLang's interface and future GQA work.
    """

    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4 or indices.dim() != 4:
        raise ValueError("q/k/v/indices must all be 4D tensors")
    if k.shape != v.shape:
        raise ValueError(f"k and v shapes must match, got {tuple(k.shape)} vs {tuple(v.shape)}")

    bsz, s_q, heads, dim = q.shape
    b_k, s_k, kv_group, dim_k = k.shape
    if b_k != bsz or dim_k != dim:
        raise ValueError(f"q/k shape mismatch: q={tuple(q.shape)} k={tuple(k.shape)}")
    if indices.shape[:3] != (bsz, s_q, kv_group):
        raise ValueError(
            f"indices shape must start with {(bsz, s_q, kv_group)}, got {tuple(indices.shape)}"
        )
    if heads % kv_group != 0:
        raise ValueError(f"heads={heads} must be divisible by kv_group={kv_group}")

    q_f = q.float().view(bsz, s_q, kv_group, heads // kv_group, dim)
    k_f = k.float()
    v_f = v.float()

    valid_addr = (indices >= 0) & (indices < s_k)
    safe_indices = torch.where(valid_addr, indices, torch.zeros_like(indices)).long()

    gathered_k = []
    gathered_v = []
    for g in range(kv_group):
        idx_g = safe_indices[:, :, g, :]
        k_g = k_f[:, :, g, :]
        v_g = v_f[:, :, g, :]
        batch_idx = torch.arange(bsz, device=q.device).view(bsz, 1, 1)
        gathered_k.append(k_g[batch_idx, idx_g])
        gathered_v.append(v_g[batch_idx, idx_g])
    # [B, S_q, G, topk, D]
    k_sel = torch.stack(gathered_k, dim=2)
    v_sel = torch.stack(gathered_v, dim=2)

    logits = torch.einsum("bsgnd,bsgtd->bsgnt", q_f, k_sel)
    scale = dim ** -0.5 if sm_scale is None else sm_scale
    logits = logits * scale

    q_abs = torch.arange(s_q, device=q.device, dtype=indices.dtype) + int(q_start_index_s)
    causal = indices <= q_abs.view(1, s_q, 1, 1)
    visible = valid_addr & causal
    logits = logits.masked_fill(~visible.unsqueeze(3), float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    out = torch.einsum("bsgnt,bsgtd->bsgnd", probs, v_sel)
    return out.reshape(bsz, s_q, heads, dim).to(q.dtype)
