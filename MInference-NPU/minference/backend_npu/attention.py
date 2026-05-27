# Copyright (c) 2026
# Licensed under The MIT License [see LICENSE for details]
"""Thin wrappers around Ascend NPU dense attention operators.

把 `torch_npu` 提供的几个 attention API 包成与 MInference 内部约定一致的接口：
  - dense_attention(q, k, v, causal=True)         —— dense baseline and fallback
  - prefill_dense(q, k, v, causal=True)           —— 显式 prefill（npu_prompt_flash_attention 优先）
  - decode_dense(q, k_cache, v_cache, ...)        —— q_len=1 decode（npu_incre_flash_attention）

输入约定：q/k/v 形状 `[B, H, S, D]`（BNSD layout），与 MInference 上游 Triton kernel 接口一致。

注意：
- 所有函数 **device-agnostic** —— 不写死 `npu:0`，device 跟随 q.device。
- 输出 dtype 与输入一致（fp16 / bf16）；内部若需要 fp32 中间计算，由 torch_npu 算子自行决定。
- 不在这里做 GQA repeat_kv —— MInference 上游 forward 已经在调 kernel 前把 K/V 复制成与 Q 同 head 数，所以本层假定 num_heads == k.num_heads。
"""

from __future__ import annotations

import math
from typing import Optional

import torch

# torch_npu 在非 NPU 环境会 ImportError；用 try-except 包住，让代码可以在 CPU/CUDA
# 机器上 import minference（仅 import 阶段不挂）。实际调用 dense_attention 等接口时
# 若没有 torch_npu 会抛 NotImplementedError，由上层 fallback 处理。
try:
    import torch_npu  # type: ignore[import-not-found]

    _HAS_TORCH_NPU = True
except ImportError:  # pragma: no cover — CI 上没 NPU 时走这里
    torch_npu = None  # type: ignore[assignment]
    _HAS_TORCH_NPU = False


__all__ = [
    "dense_attention",
    "prefill_dense",
    "decode_dense",
    "is_npu_available",
]


def is_npu_available() -> bool:
    """`torch_npu` 是否就绪、且当前进程能看到至少一张 NPU。"""
    if not _HAS_TORCH_NPU:
        return False
    try:
        return torch.npu.is_available() and torch.npu.device_count() > 0
    except AttributeError:
        return False


def _eager_attention_cpu_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool,
) -> torch.Tensor:
    """非 NPU 设备上的纯 PyTorch 参考实现，仅用于在开发机上做语义对照。

    本函数 **不应** 在生产路径被调用；它只用于非 NPU 环境下的 import/smoke/unit 兜底。
    """
    in_dtype = q.dtype
    qf = q.float()
    kf = k.float()
    vf = v.float()
    attn = torch.matmul(qf, kf.transpose(-2, -1)) * scale
    if causal:
        s_q, s_k = qf.shape[-2], kf.shape[-2]
        mask = torch.ones(s_q, s_k, device=qf.device, dtype=torch.bool).tril(
            diagonal=s_k - s_q
        )
        attn = attn.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(attn, dim=-1)
    return torch.matmul(probs, vf).to(in_dtype)


def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: Optional[float] = None,
    causal: bool = True,
) -> torch.Tensor:
    """Dense causal attention for baseline and fallback paths.

    Args:
        q, k, v: `[B, H, S, D]`，dtype 必须一致（fp16 / bf16 推荐）。
        scale:   QK 缩放系数，默认 `1 / sqrt(D)`。
        causal:  是否启用因果 mask（prefill 全部为 True；decode q_len=1 时无所谓）。

    Returns:
        `[B, H, S, D]`，与 q 同 dtype 同 device。
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))

    if q.device.type != "npu" or not _HAS_TORCH_NPU:
        # 开发机 / CI 兜底：用纯 PyTorch 实现，确保上层代码可以在非 NPU 上跑通 unit test
        return _eager_attention_cpu_ref(q, k, v, scale, causal)

    num_heads = q.size(1)

    # Use an explicit bool mask for causal semantics; torch_npu sparse_mode
    # behavior has differed across CANN releases.
    if not causal:
        result = torch_npu.npu_fusion_attention(  # type: ignore[union-attr]
            q,
            k,
            v,
            head_num=num_heads,
            input_layout="BNSD",
            scale=scale,
            sparse_mode=0,
        )
        return result[0] if isinstance(result, (tuple, list)) else result

    s_q = q.size(-2)
    s_k = k.size(-2)
    # causal: 允许 j <= i + (s_k - s_q)；True means masked in torch_npu.
    causal_mask = torch.ones(s_q, s_k, device=q.device, dtype=torch.bool).triu(
        diagonal=s_k - s_q + 1
    )
    try:
        result = torch_npu.npu_fusion_attention(  # type: ignore[union-attr]
            q,
            k,
            v,
            head_num=num_heads,
            input_layout="BNSD",
            scale=scale,
            sparse_mode=1,
            atten_mask=causal_mask,
        )
    except TypeError:
        # 个别 torch_npu 小版本只接受 uint8 mask
        result = torch_npu.npu_fusion_attention(  # type: ignore[union-attr]
            q,
            k,
            v,
            head_num=num_heads,
            input_layout="BNSD",
            scale=scale,
            sparse_mode=1,
            atten_mask=causal_mask.to(torch.uint8),
        )
    return result[0] if isinstance(result, (tuple, list)) else result


def prefill_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: Optional[float] = None,
    causal: bool = True,
) -> torch.Tensor:
    """Prefill 阶段的 dense attention。

    与 `dense_attention` 接口相同；保留独立名字是为了让 patch.py 里的调用点显式区分
    "prefill 路径" 与 "稀疏分支临时退化"。
    """
    return dense_attention(q, k, v, scale=scale, causal=causal)


def decode_dense(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """q_len=1 的 decode 路径。

    上游 MInference 在 `q_len == 1` 时短路成 dense flash_attn；NPU 上对应
    `npu_incre_flash_attention`（增量推理专用，针对 q_len=1 优化）。

    Args:
        q:       `[B, H, 1, D]`
        k_cache: `[B, H, S_kv, D]`
        v_cache: `[B, H, S_kv, D]`
        scale:   默认 `1 / sqrt(D)`
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))

    if q.device.type != "npu" or not _HAS_TORCH_NPU:
        return _eager_attention_cpu_ref(q, k_cache, v_cache, scale, causal=False)

    num_heads = q.size(1)
    # npu_incre_flash_attention 签名（按 torch_npu 2.6 文档）：
    #   (query, key, value, num_heads, input_layout, scale_value, ...)
    # 不同小版本签名略有差异；如遇 TypeError 上层 patch 会捕获并 fallback 到 npu_fusion_attention
    try:
        result = torch_npu.npu_incre_flash_attention(  # type: ignore[union-attr]
            q,
            k_cache,
            v_cache,
            num_heads=num_heads,
            input_layout="BNSD",
            scale_value=scale,
        )
        return result[0] if isinstance(result, (tuple, list)) else result
    except (AttributeError, TypeError):
        # API 名 / 签名不可用时退到 npu_fusion_attention
        return dense_attention(q, k_cache, v_cache, scale=scale, causal=False)
