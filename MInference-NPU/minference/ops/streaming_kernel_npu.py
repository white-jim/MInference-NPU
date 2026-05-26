# Copyright (c) 2024-2025 Microsoft
# Copyright (c) 2026 (NPU 适配 — M2 streaming kernel)
# Licensed under The MIT License [see LICENSE for details]
"""Streaming (A-shape) attention — NPU v1 实现。

把上游 MInference 1.0 `ops/streaming_kernel.py:streaming_forward` 从 Triton/CUDA 移植
到昇腾 NPU 上。算法语义：每个 query 位置 ``i`` 在 K 中的可见集合为：

* **sink**:    ``K[..., :n_init, :]`` — 前 ``n_init`` 个 token，对所有 query 都可见
* **sliding**: ``K[..., i_abs - n_local + 1 : i_abs + 1, :]`` — 以当前 query 绝对位置
  为右端、宽 ``n_local`` 的滑动窗口（``i_abs = k_len - q_len + i``）

NPU 实现路径（v1 PoC，与 `docs/migration_plan_v1.md §4` 一致）：

* **段 1** — sliding-window-only：``npu_fusion_attention(sparse_mode=1, atten_mask=...)``，
  显式 ``[1,1,S_q,S_k]`` bool mask 表达 sliding window（``True=屏蔽``）。
  历史上想用 ``sparse_mode=4 + pre/next_tockens`` 的 band 快路径，但 CANN 8.1.RC1
  下 sparse_mode=2/4 实测均不生效（退化成 full attention，见 docs §9.1/§9.3），
  v1 全部 causal/band 调用走 sparse_mode=1 + 显式 mask。
* **段 2** — sink-only：仅前 ``n_init`` 个 key，用自定义 ``atten_mask`` 排除 sliding
  window 已经覆盖到的位置（complement-sliding-window 思路；mask 形状
  ``[1, 1, S_q, n_init]``，n_init=128 时内存可忽略）。
* **合并** — 跨段 log-sum-exp，用 ``npu_fusion_attention`` 返回的 ``softmax_max`` /
  ``softmax_sum`` 做 online softmax 合并，等价完整 softmax。

非 NPU 路径（CPU / CUDA / 无 torch_npu）：纯 PyTorch 黄金参考（按 q 分块构造 sink+
sliding mask）。两条路径接口签名一致，测试用 PyTorch ref 做对照。

短路情形：

* ``k_len <= n_local``：全可见，退化为 causal dense（一段路径，无 sink 复杂度）。
* ``n_init <= 0``：只有 sliding window，跑段 1 后直接返回。

签名严格对齐上游 ``MInference/minference/ops/streaming_kernel.py:streaming_forward``：

    streaming_forward(q, k, v, n_init, n_local) -> Tensor  # [B, H, S, D]

输入约定：q/k/v 形状 ``[B, H, S, D]``（BNSD），已经做过 RoPE、已经 ``repeat_kv``
对齐 head 数（与上游一致）。
"""

from __future__ import annotations

import math
import importlib.util as _importlib_util
import os
import sys
import warnings
from typing import Optional

import torch

try:
    import torch_npu  # type: ignore[import-not-found]

    _HAS_TORCH_NPU = True
except ImportError:  # pragma: no cover — 非 NPU 环境
    torch_npu = None  # type: ignore[assignment]
    _HAS_TORCH_NPU = False


__all__ = ["streaming_forward"]


_SIBLING_MODULE_CACHE: dict[str, object] = {}


# ----------------------------------------------------------------------------
# head_dim 对齐（与上游一致）
# ----------------------------------------------------------------------------

# `npu_fusion_attention` 与上游 Triton kernel 都要求 head_dim ∈ {16,32,64,128,256,512}。
# 其他取值需要 pad 到 2 的幂；推理完成后再截回原始 head_dim。
_ALLOWED_HEAD_DIMS = (16, 32, 64, 128, 256, 512)
_TILELANG_BLOCK_SIZE = 64


def _load_sibling_module(module_name: str, filename: str):
    cached = _SIBLING_MODULE_CACHE.get(filename)
    if cached is not None:
        return cached
    if module_name in sys.modules:
        module = sys.modules[module_name]
        _SIBLING_MODULE_CACHE[filename] = module
        return module
    path = os.path.join(os.path.dirname(__file__), filename)
    spec = _importlib_util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename} from {path}")
    module = _importlib_util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _SIBLING_MODULE_CACHE[filename] = module
    return module


def _pad_head_dim_to_pow2(t: torch.Tensor, target_d: int) -> torch.Tensor:
    cur_d = t.shape[-1]
    if cur_d == target_d:
        return t
    return torch.nn.functional.pad(t, (0, target_d - cur_d, 0, 0, 0, 0, 0, 0))


def _next_allowed_head_dim(d: int) -> int:
    for cand in _ALLOWED_HEAD_DIMS:
        if cand >= d:
            return cand
    raise ValueError(
        f"head_dim={d} 超过支持的最大值 {_ALLOWED_HEAD_DIMS[-1]}；MInference 上游同样不支持"
    )


# ----------------------------------------------------------------------------
# PyTorch 黄金参考（**测试与非 NPU 兜底共用一份实现**）
# ----------------------------------------------------------------------------


def _streaming_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_init: int,
    n_local: int,
    *,
    chunk_size_q: Optional[int] = None,
) -> torch.Tensor:
    """Sink + sliding-window 注意力的纯 PyTorch 实现。

    用作：
    1. NPU 不可用时（CPU / CUDA / import 失败）的兜底
    2. 单测的"黄金参考"

    沿 q 维分块以避免长上下文场景的 mask 内存爆炸：mask 形状 ``[chunk, S_k]``，把
    ``chunk * S_k`` 控制在 ~4M 元素以内。

    数值实现：fp32 softmax，再 cast 回输入 dtype；与上游 Triton kernel 的 `tl.exp` /
    fp32 累加路径等价。
    """
    bsz, n_heads, s_q, head_d = q.shape
    s_k = k.shape[2]
    scale = 1.0 / math.sqrt(head_d)

    if chunk_size_q is None:
        # 把每个 chunk 的 attention 张量内存控制在 ~4M 元素（约 16MB fp32）：
        # chunk = clip(4*1024*1024 // s_k, 64, s_q)
        chunk_size_q = max(64, min(s_q, (4 * 1024 * 1024) // max(1, s_k)))

    j_all = torch.arange(s_k, device=q.device)  # [S_k]

    out = torch.empty_like(q)
    in_dtype = q.dtype

    for chunk_start in range(0, s_q, chunk_size_q):
        chunk_end = min(chunk_start + chunk_size_q, s_q)
        c = chunk_end - chunk_start

        # 绝对 query 位置（与上游 sliding_window_offset = k_len - q_len 一致）
        abs_i = torch.arange(
            s_k - s_q + chunk_start, s_k - s_q + chunk_end, device=q.device
        )  # [c]

        # sink: 永远可见（j < n_init）
        sink = j_all[None, :] < n_init  # [c-broadcast, S_k]

        # sliding: j ∈ [abs_i - n_local + 1, abs_i]
        sliding = (j_all[None, :] <= abs_i[:, None]) & (
            j_all[None, :] > abs_i[:, None] - n_local
        )  # [c, S_k]

        # causal 守门（与上游 `tl.where(mask)` 一致；sink 段不需要单独 causal，
        # 因为 sink 内的 j 都 < n_init ≤ abs_i 在所有非短路情形下；但 j_all 也包含
        # 超过 abs_i 的位置，所以需要显式截一刀）
        causal = j_all[None, :] <= abs_i[:, None]
        visible = (sink | sliding) & causal  # [c, S_k]

        q_chunk = q[:, :, chunk_start:chunk_end, :]  # [B, H, c, D]
        # 用 fp32 累计避免 fp16 溢出（与上游 m_i/l_i fp32 一致）
        logits = torch.matmul(q_chunk.float(), k.float().transpose(-2, -1)) * scale
        logits = logits.masked_fill(~visible[None, None, :, :], float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        # 处理"整行被 mask 掉"（理论上不会发生，因为每个 query 至少看到自己；防御性兜底）
        probs = torch.nan_to_num(probs, nan=0.0)
        out_chunk = torch.matmul(probs, v.float())  # [B, H, c, D]
        out[:, :, chunk_start:chunk_end, :] = out_chunk.to(in_dtype)

    return out


# ----------------------------------------------------------------------------
# NPU 路径（两段 attention + log-sum-exp 合并）
# ----------------------------------------------------------------------------


def _take_first_lane(stat: torch.Tensor) -> torch.Tensor:
    """`npu_fusion_attention` 返回的 ``softmax_max`` / ``softmax_sum`` 形状通常为
    ``[B, H, S, 8]``（最后维 8 是 tile 对齐填充，每行实际 max/sum 在各 lane 中复制）。

    取第 0 个 lane 并保留维度，返回 ``[B, H, S, 1]`` 方便对 ``[B, H, S, D]`` 广播。

    若 NPU 改成了 ``[B, H, S]`` 三维返回（不同 torch_npu 版本）也兼容：直接 unsqueeze。
    """
    if stat.dim() == 4:
        return stat[..., :1]
    if stat.dim() == 3:
        return stat.unsqueeze(-1)
    raise RuntimeError(
        f"unexpected softmax stat shape {tuple(stat.shape)}; "
        "需要适配新的 torch_npu npu_fusion_attention 返回格式"
    )


def _unpack_fa_result(result, *, where: str):
    """校验 `npu_fusion_attention` 返回 (output, softmax_max, softmax_sum, ...)。

    streaming 的两段 LSE 合并强依赖 softmax_max/softmax_sum；不同 torch_npu 小版本
    返回格式可能是裸 Tensor 或更短的 tuple。这里在入口处 fail-fast，给出可定位的错误。
    """
    if not isinstance(result, (tuple, list)) or len(result) < 3:
        kind = type(result).__name__
        size = len(result) if hasattr(result, "__len__") else "n/a"
        raise RuntimeError(
            f"npu_fusion_attention at {where} 返回 {kind} (len={size})；"
            " streaming kernel 需要 (output, softmax_max, softmax_sum, ...)"
            " 才能做跨段 LSE 合并。请确认 torch_npu 版本，或切到 dense 路径绕过 streaming。"
        )
    return result[0], _take_first_lane(result[1]), _take_first_lane(result[2])


def _streaming_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_init: int,
    n_local: int,
) -> torch.Tensor:
    """NPU 路径（v1）：两段 `npu_fusion_attention` + log-sum-exp 合并。

    本函数假定调用前已经：

    * 保证 ``q.device.type == "npu"`` 且 ``torch_npu`` 可用
    * head_dim 已 pad 至 ``_ALLOWED_HEAD_DIMS`` 内
    * ``k_len > n_local``（``k_len <= n_local`` 在外层短路）

    返回 ``[B, H, S_q, D_padded]``；外层 ``streaming_forward`` 负责把 head_dim 截回原始值。
    """
    assert _HAS_TORCH_NPU, "_streaming_npu 不能在非 NPU 环境调用（外层应已分流）"

    bsz, n_heads, s_q, head_d = q.shape
    s_k = k.shape[2]
    scale = 1.0 / math.sqrt(head_d)

    # --- 段 1：sliding window only（sparse_mode=1 + 显式 bool mask） ---
    # §9.1 实测：CANN 8.1.RC1 下 sparse_mode=2/4 不按文档语义生效（退化成 full
    # attention），唯一可靠的稀疏路径是 sparse_mode=1 + 用户显式 atten_mask。
    # sliding 语义：query i 看 [i_abs - n_local + 1, i_abs]，i_abs = s_k - s_q + i。
    abs_i = torch.arange(s_k - s_q, s_k, device=q.device)  # [s_q]
    j_all = torch.arange(s_k, device=q.device)  # [s_k]
    sliding_vis = (j_all[None, :] <= abs_i[:, None]) & (
        j_all[None, :] > abs_i[:, None] - int(n_local)
    )  # [s_q, s_k]，True=可见
    sliding_mask = (~sliding_vis).to(torch.bool)[None, None, :, :]  # True=屏蔽

    try:
        pass1 = torch_npu.npu_fusion_attention(  # type: ignore[union-attr]
            q,
            k,
            v,
            head_num=n_heads,
            input_layout="BNSD",
            scale=scale,
            sparse_mode=1,
            atten_mask=sliding_mask,
        )
    except TypeError:
        pass1 = torch_npu.npu_fusion_attention(  # type: ignore[union-attr]
            q,
            k,
            v,
            head_num=n_heads,
            input_layout="BNSD",
            scale=scale,
            sparse_mode=1,
            atten_mask=sliding_mask.to(torch.uint8),
        )
    # 文档定义返回元组：(attention_out, softmax_max, softmax_sum, softmax_out, seed, offset, numel)
    o1, m1, l1 = _unpack_fa_result(pass1, where="streaming pass1 (sliding window)")
    # 极端情形下 sliding window 可能整行被屏蔽（s_q > s_k 等非法输入），与段 2 一致用
    # nan_to_num 保护 LSE 合并；正常 prefill (s_q <= s_k) 不会触发。
    l1 = torch.nan_to_num(l1, nan=0.0)

    if n_init <= 0:
        # 没有 sink，单段直接返回（仍要 cast 回输入 dtype；npu_fusion_attention 输出已对齐）
        return o1

    # --- 段 2：sink only（前 n_init 个 key），mask 出 sliding window 重叠 ---
    n_init_clamped = min(int(n_init), s_k)
    k_sink = k[:, :, :n_init_clamped, :].contiguous()
    v_sink = v[:, :, :n_init_clamped, :].contiguous()

    # 对 query i（绝对 = s_k - s_q + i），允许的 sink key j 满足：
    #   j < n_init_clamped  AND  j NOT IN sliding window
    # 等价于 j < max(0, i_abs - n_local + 1)。当 i_abs < n_local 时整行无可见 → 段 2 贡献 0。
    abs_i = torch.arange(s_k - s_q, s_k, device=q.device)  # [s_q]
    j = torch.arange(n_init_clamped, device=q.device)  # [n_init_clamped]
    allowed = j[None, :] < (abs_i[:, None] - int(n_local) + 1)  # [s_q, n_init_clamped]
    # NPU atten_mask 约定：True / 1 = 屏蔽
    attn_mask = (~allowed).to(torch.bool)[None, None, :, :]  # [1, 1, s_q, n_init_clamped]

    try:
        pass2 = torch_npu.npu_fusion_attention(  # type: ignore[union-attr]
            q,
            k_sink,
            v_sink,
            head_num=n_heads,
            input_layout="BNSD",
            scale=scale,
            sparse_mode=1,  # user-provided mask
            atten_mask=attn_mask,
        )
    except TypeError:
        pass2 = torch_npu.npu_fusion_attention(  # type: ignore[union-attr]
            q,
            k_sink,
            v_sink,
            head_num=n_heads,
            input_layout="BNSD",
            scale=scale,
            sparse_mode=1,
            atten_mask=attn_mask.to(torch.uint8),
        )
    o2, m2, l2 = _unpack_fa_result(pass2, where="streaming pass2 (sink)")
    # 全行被 mask 时 torch_npu 可能返回 NaN；nan_to_num 使 LSE 合并退化为 o1
    l2 = torch.nan_to_num(l2, nan=0.0)

    # --- 跨段 log-sum-exp 合并 ---
    # NPU FA 返回：m = max(scaled_logits)，l = sum(exp(scaled_logits - m))。
    # 合并公式：
    #   m_new = max(m1, m2)
    #   o_new = (o1 * l1 * exp(m1 - m_new) + o2 * l2 * exp(m2 - m_new))
    #         / (l1 * exp(m1 - m_new) + l2 * exp(m2 - m_new))
    # 段 2 全行被 mask 时 l2 == 0，公式自然退化为 o1；为防 -inf - (-inf) = NaN，先把无效 m
    # clamp 到一个有限大负数（仅当对应 l == 0 时无效）。
    neg_big = torch.finfo(m1.dtype).min / 4
    m2_eff = torch.where(l2 > 0, m2, torch.full_like(m2, neg_big))
    m1_eff = torch.where(l1 > 0, m1, torch.full_like(m1, neg_big))
    m_new = torch.maximum(m1_eff, m2_eff)

    alpha1 = torch.exp(m1_eff - m_new) * l1
    alpha2 = torch.exp(m2_eff - m_new) * l2
    denom = alpha1 + alpha2
    # 防御性兜底：若 denom 仍 = 0（理论上不会发生，因为每个 query 至少看到 sliding window
    # 内的自己），直接退到 o1
    safe = denom > 0
    out_combined = (alpha1 * o1.float() + alpha2 * o2.float()) / denom
    out_combined = torch.where(safe, out_combined, o1.float())

    return out_combined.to(q.dtype)


def _streaming_tilelang_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_init: int,
    n_local: int,
) -> torch.Tensor:
    """Path B: TileLang true sparse A-shape attention for per-head ``H==1`` calls."""
    if q.dtype != torch.float16:
        raise NotImplementedError("TileLang streaming MVP only supports fp16")
    if q.shape[1] != 1:
        raise NotImplementedError("TileLang streaming MVP is currently only wired for H==1")
    if k.shape[1] != 1 or v.shape[1] != 1:
        raise NotImplementedError("TileLang streaming MVP expects repeated per-head K/V")
    if int(n_init) % _TILELANG_BLOCK_SIZE != 0 or int(n_local) % _TILELANG_BLOCK_SIZE != 0:
        raise ValueError(
            f"n_init={n_init} and n_local={n_local} must be multiples of {_TILELANG_BLOCK_SIZE}"
        )

    B, H, S_q, D = q.shape
    S_k = k.shape[2]
    q_start = S_k - S_q

    tilelang_indices = _load_sibling_module(
        "tilelang_indices_streaming_standalone",
        "tilelang_indices.py",
    )
    tilelang_sparse_attention = _load_sibling_module(
        "tilelang_sparse_attention_streaming_standalone",
        "tilelang_sparse_attention.py",
    )

    indices = tilelang_indices.stream_llm_to_tilelang(
        B=B,
        S_q=S_q,
        kv_heads=H,
        n_init=int(n_init),
        n_local=int(n_local),
        block_size_N=_TILELANG_BLOCK_SIZE,
        q_start_index_s=q_start,
        device="cpu",
    )
    indices = torch.where(
        (indices >= 0) & (indices < S_k),
        indices,
        torch.full_like(indices, tilelang_indices.TILELANG_PAD_VALUE),
    ).to(device=q.device, dtype=torch.int32)

    kernel = tilelang_sparse_attention.build_sparse_attention_qkv_fwd(
        heads=H,
        dim=D,
        topk=indices.shape[-1],
        kv_group=1,
        block_I=_TILELANG_BLOCK_SIZE,
        q_start_index_s=q_start,
    )

    q_bshd = q.transpose(1, 2).contiguous()
    k_bsgd = k.transpose(1, 2).contiguous()
    v_bsgd = v.transpose(1, 2).contiguous()
    out_bshd = kernel(q_bshd, k_bsgd, v_bsgd, indices)
    return out_bshd.transpose(1, 2).contiguous()


# ----------------------------------------------------------------------------
# 顶层入口
# ----------------------------------------------------------------------------


def streaming_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_init: int,
    n_local: int,
) -> torch.Tensor:
    """Sink + sliding-window 注意力（A-shape / streaming-LLM）。

    Args:
        q, k, v: ``[B, H, S, D]``，已 RoPE、已 ``repeat_kv``（与上游一致）。
        n_init:  sink 段长度（前 N 个 token 永远可见）。<=0 表示无 sink。
        n_local: sliding window 段长度。

    Returns:
        ``[B, H, S, D]``，与 ``q`` 同 dtype 同 device。

    短路：
    * ``k_len <= n_local``：等价于 causal dense（所有 K 都在 sliding window 内）。
      实现上转到 ``backend_npu.dense_attention`` / PyTorch dense。
    """
    assert q.dim() == 4, f"q 期望 4D [B,H,S,D]，得到 {q.dim()}D"
    assert (
        q.shape[0] == k.shape[0] and q.shape[1] == k.shape[1] and q.shape[-1] == k.shape[-1]
    ), f"q/k B/H/D 不匹配；q={tuple(q.shape)} k={tuple(k.shape)}"

    head_d = q.shape[-1]
    orig_head_d = head_d

    # head_dim pad 到允许集
    if head_d not in _ALLOWED_HEAD_DIMS:
        target_d = _next_allowed_head_dim(head_d)
        q = _pad_head_dim_to_pow2(q, target_d)
        k = _pad_head_dim_to_pow2(k, target_d)
        v = _pad_head_dim_to_pow2(v, target_d)

    s_k = k.shape[2]

    # 短路：sliding window 覆盖所有 K → causal dense
    if s_k <= int(n_local):
        # 复用 backend_npu.dense_attention（NPU 上 npu_fusion_attention sparse_mode=1
        # + 显式 causal mask；非 NPU 上自动走 PyTorch eager）；避免循环 import 用延迟导入
        from ..backend_npu import dense_attention

        out = dense_attention(q, k, v, causal=True)
        return out[..., :orig_head_d]

    # 主路径：PR-4 H==1 fp16 走 TileLang 真稀疏；其他 NPU 场景保留 path-A 两段合并。
    if q.device.type == "npu" and _HAS_TORCH_NPU:
        if q.shape[1] == 1 and q.dtype == torch.float16:
            try:
                out = _streaming_tilelang_npu(q, k, v, int(n_init), int(n_local))
            except ImportError as exc:
                warnings.warn(
                    "streaming_forward: TileLang path B is unavailable "
                    f"({exc}); falling back to path A bool-mask attention.",
                    stacklevel=2,
                )
                out = _streaming_npu(q, k, v, int(n_init), int(n_local))
            except ValueError as exc:
                warnings.warn(
                    "streaming_forward: TileLang path B is not applicable "
                    f"({exc}); falling back to path A bool-mask attention.",
                    stacklevel=2,
                )
                out = _streaming_npu(q, k, v, int(n_init), int(n_local))
        else:
            out = _streaming_npu(q, k, v, int(n_init), int(n_local))
    else:
        out = _streaming_pytorch_ref(q, k, v, int(n_init), int(n_local))

    return out[..., :orig_head_d]
