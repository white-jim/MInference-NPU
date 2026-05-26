# Copyright (c) 2026 (NPU 适配 — PR-4 PoC dense FA kernel)
# Licensed under The MIT License [see LICENSE for details]
"""Triton-Ascend dense Flash Attention PoC（PR-4-poc，路径 B 可行性 gate）。

这是 PR-4 的第一步：用 triton-ascend 写一个最简 dense causal FA kernel，
与 `torch_npu.npu_fusion_attention` (sparse_mode=1 + 显式 causal mask) 对比：

  1. 精度：fp16 max_abs_diff 应在 ~1e-2 量级（softmax 累积噪声）
  2. 性能：8K/16K 上 triton-ascend 写的 dense 比 `npu_fusion_attention` 慢 1.5-3× 可接受
          —— 这是 high-level DSL 的天然代价，AscendC 才能打平/超过
          只要 latency 在合理量级，就证明真稀疏 kernel 的方案可行

PR-4-poc **不做**的事：
  - 稀疏（vertical_slash / block_sparse → 留给 PR-4-VS / PR-4-BS）
  - GQA（先 MHA — repeat_kv 提前做完即可）
  - 反向 / dropout / RoPE 融合
  - 自动 tune（用 fixed BLOCK_M/N/D，后续调）

算法：Flash Attention 2 online softmax（标准实现），causal mask 在内循环 token 级处理。
保守起见，内循环 loop 所有 K block，causal 通过 `tl.where` 掩 -inf，不做"end_n
随 pid_m 递减" 的 skip 优化（triton-ascend 是否支持 runtime computed loop bound
留到 PR-4-VS 阶段验证；这里先求最稳）。

输入约定：BNSD layout，与 `vertical_slash_kernel_npu` / `npu_fusion_attention` 对齐：
    q/k/v: [B, N, S, D]  fp16 / bf16
    返回 : [B, N, S, D]  与输入同 dtype
"""

from __future__ import annotations

import math
from typing import Optional

import torch

# Module 级 import — @triton.jit 走 func.__globals__ 解析 `tl.*`，
# 放函数内会报 NameError('tl is not defined')（triton-ascend 3.2.0 实测）。
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception as _triton_err:  # noqa: BLE001 — Windows / 无 triton 环境 fallback
    triton = None
    tl = None
    _HAS_TRITON = False
    _TRITON_IMPORT_ERR = _triton_err

__all__ = ["triton_ascend_fa_dense", "has_triton"]


def has_triton() -> bool:
    """Triton-Ascend 是否可用（test / benchmark 提前 gate 用）。"""
    return _HAS_TRITON


# ---------------------------------------------------------------------------
# Triton-Ascend kernel（仅在 triton 可用时定义；module 级）
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _fa_dense_causal_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        sm_scale,
        # strides（按 element 计，PyTorch tensor.stride() 即可）
        stride_qb,
        stride_qh,
        stride_qs,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vs,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_os,
        stride_od,
        # 形状
        S_q,
        S_k,
        # 编译期常量
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """Dense flash-attention causal forward.

        Grid: (B * H, ceil(S_q / BLOCK_M))
        每个 program 处理一个 (b, h) 的一个 Q block。
        """
        pid_bh = tl.program_id(0)
        pid_m = tl.program_id(1)

        # 解出 (b, h)：H 通过 stride 推不出来，需要从 grid 外读 — 这里用 strides 直接寻址
        # 即可不显式拆 b/h（因为 batch、head 维度的 stride 已分别传入）。但为了
        # 让 ptr 计算清晰，仍按 b * stride_b + h * stride_h 写法，所以需要 H 维度。
        # 简化：把 (b, h) 合并到 pid_bh，按 stride_qh = D * S_q (BNSD contiguous) 计算时，
        # b * stride_qb + h * stride_qh ≡ pid_bh * stride_qh 当 stride_qb = H * stride_qh。
        # 这一般成立，因为我们假设 q/k/v BNSD contiguous。
        bh_offset_q = pid_bh * stride_qh
        bh_offset_k = pid_bh * stride_kh
        bh_offset_v = pid_bh * stride_vh
        bh_offset_o = pid_bh * stride_oh

        # Q block: [BLOCK_M, BLOCK_D]
        q_offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        q_ptrs = (
            Q_ptr
            + bh_offset_q
            + q_offs_m[:, None] * stride_qs
            + offs_d[None, :] * stride_qd
        )
        q_mask_row = q_offs_m < S_q
        q = tl.load(q_ptrs, mask=q_mask_row[:, None], other=0.0)

        # Online softmax 累加器（fp32）
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        # 内循环：所有 K block。causal 用 tl.where 掩 -inf（不做 loop bound skip）。
        # n_blocks 用编译期已知的 ceil(S_k / BLOCK_N)，必须在 host 侧保证 S_k 是 BLOCK_N 整数倍
        # 或在 kernel 内做 padding 的 boundary mask。这里走后者。
        for start_n in range(0, S_k, BLOCK_N):
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
            k_mask_col = k_offs_n < S_k

            k_ptrs = (
                K_ptr
                + bh_offset_k
                + k_offs_n[:, None] * stride_ks
                + offs_d[None, :] * stride_kd
            )
            k = tl.load(k_ptrs, mask=k_mask_col[:, None], other=0.0)

            # QK^T: [BLOCK_M, BLOCK_N]
            qk = tl.dot(q, tl.trans(k)) * sm_scale

            # 边界 + causal 掩码
            valid = q_mask_row[:, None] & k_mask_col[None, :]
            if IS_CAUSAL:
                causal = q_offs_m[:, None] >= k_offs_n[None, :]
                qk = tl.where(valid & causal, qk, float("-inf"))
            else:
                qk = tl.where(valid, qk, float("-inf"))

            # Online softmax 更新
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_ij)
            p = tl.exp(qk - m_ij[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)

            # V block + acc
            v_ptrs = (
                V_ptr
                + bh_offset_v
                + k_offs_n[:, None] * stride_vs
                + offs_d[None, :] * stride_vd
            )
            v = tl.load(v_ptrs, mask=k_mask_col[:, None], other=0.0)

            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_ij

        # 归一化 + 写回
        # 防御性除零：m_i = -inf 的行（全被 mask 掉，理论上只在 padding 区）
        # 这里只在 q_mask_row 范围内写回，所以 padding 行 store 时被 mask 屏蔽
        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_safe[:, None]

        o_ptrs = (
            O_ptr
            + bh_offset_o
            + q_offs_m[:, None] * stride_os
            + offs_d[None, :] * stride_od
        )
        tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_mask_row[:, None])


# ---------------------------------------------------------------------------
# Python entry
# ---------------------------------------------------------------------------


def triton_ascend_fa_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: Optional[float] = None,
    causal: bool = True,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """Dense flash-attention via triton-ascend kernel.

    Args:
        q/k/v: [B, N, S, D]，BNSD layout，fp16 / bf16，已 repeat_kv 到 MHA。
               q.shape[2] == S_q（可与 S_k 不同，但 PoC 阶段假设 S_q == S_k）。
        sm_scale: 默认 1/sqrt(D)。
        causal: True 时上三角掩 -inf。
        block_m/n: 内核 tile size。默认 64×64（NPU L1 cache 友好的起点；后续 PR-4-poc
                   bench 阶段可调）。head_dim 作为 BLOCK_D 直接传入（要求是 2 的幂，
                   且 <= 头维度，128 是 Llama-3 标准）。

    Returns:
        out: [B, N, S_q, D]，与 q 同 dtype。
    """
    if not _HAS_TRITON:
        raise RuntimeError(
            f"triton-ascend 不可用（module 级 import 失败）：{_TRITON_IMPORT_ERR}"
        )

    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "q/k/v 必须是 4-D BNSD"
    assert q.shape[:2] == k.shape[:2] == v.shape[:2], "B / N 维度必须一致（MHA，先 repeat_kv）"
    assert q.shape[3] == k.shape[3] == v.shape[3], "head_dim 必须一致"
    assert k.shape[2] == v.shape[2], "S_k 必须一致"

    B, N, S_q, D = q.shape
    S_k = k.shape[2]

    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), (
        "PR-4-poc 阶段强制 contiguous（stride 简化）。如需 non-contiguous，调用方先 .contiguous()。"
    )

    # BLOCK_D 必须是 2 的幂且 >= head_dim — Triton tl.arange 要求 constexpr power-of-2
    block_d = 1
    while block_d < D:
        block_d *= 2
    assert block_d == D, f"head_dim={D} 必须是 2 的幂（Triton constexpr 要求）"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(q)

    grid = (B * N, triton.cdiv(S_q, block_m))

    _fa_dense_causal_kernel[grid](
        q,
        k,
        v,
        out,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        S_q,
        S_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        IS_CAUSAL=causal,
    )
    return out
