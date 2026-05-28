# Copyright (c) 2024-2025 Microsoft
# Copyright (c) 2026 (NPU adaptation)
# Licensed under The MIT License [see LICENSE for details]
"""Block-sparse attention for the PR-4 path-B workspace.

算法语义：把 KV 序列按固定 ``block_size`` 切块，每个 query block 只与得分最高的
``topk_blocks`` 个 key block 做 attention（因果约束：key block index ≤ query block index）。
块得分由平均池化的 Q×K 近似计算，token 级因果约束额外施加。

主路径是 TileLang true sparse attention。保留 bool-mask NPU 实现和 PyTorch 参考，
用于非 fp16/不可用场景的 fallback 与单测对照。

非 NPU 路径：同样构建 block-sparse mask，用纯 PyTorch masked softmax 实现，作为
测试黄金参考。两条路径共用 ``_build_block_sparse_mask``，仅 attention 计算步骤不同。

签名与上游 ``MInference/minference/ops/block_sparse_flash_attention.py:block_sparse_attention``
对齐（参数名做了语义化重命名）：

    block_sparse_attention(q, k, v, topk_blocks, block_size=64) -> Tensor  # [B, H, S, D]

输入约定：q/k/v 形状 ``[B, H, S, D]``（BNSD），已 RoPE、已 ``repeat_kv``（与上游一致）。
"""

from __future__ import annotations

import math
import importlib.util as _importlib_util
import os
import sys
import warnings
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import torch_npu  # type: ignore[import-not-found]

    _HAS_TORCH_NPU = True
except ImportError:  # pragma: no cover
    torch_npu = None  # type: ignore[assignment]
    _HAS_TORCH_NPU = False


__all__ = ["block_sparse_attention"]


_SIBLING_MODULE_CACHE: dict[str, object] = {}


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# npu_fusion_attention 要求 head_dim ∈ 此集合；否则需要 pad
_ALLOWED_HEAD_DIMS = (16, 32, 64, 128, 256, 512)

# bool mask 构建是 O(S²)；超过此阈值退化为 dense 并发 WARNING
# 路径 B（TileLang-Ascend）可覆盖更长序列
_MAX_SEQ_FOR_MASK = 16384

# 短上下文优先走 bool-mask NPU path A，作为 4K-16K 过渡路径与 TileLang path-B
# 的性能对照。设为 0 可强制所有 fp16 NPU block_sparse 先尝试 TileLang。
_PREFER_MASK_MAX_SEQ_ENV = "MINFERENCE_BLOCK_SPARSE_MASK_MAX_SEQ"
_PREFER_MASK_MAX_HEADS_ENV = "MINFERENCE_BLOCK_SPARSE_MASK_MAX_HEADS"
_DEFAULT_PREFER_MASK_MAX_SEQ = 16384
_DEFAULT_PREFER_MASK_MAX_HEADS = 16

# TileLang path-B 默认使用 H=1 query-block kernel。设为 0 可回到旧 padded-H kernel，
# 便于做 A/B benchmark 或快速规避新 kernel 的边界问题。
_TILELANG_H1_QUERY_BLOCK_ENV = "MINFERENCE_BLOCK_SPARSE_TILELANG_H1"
_TILELANG_H1_BLOCK_INDEX_ENV = "MINFERENCE_BLOCK_SPARSE_TILELANG_BLOCK_INDEX"
_TILELANG_H1_BLOCK_M = 16

# 默认 block size（与上游 block_sparse_flash_attention.py 一致）
_DEFAULT_BLOCK_SIZE = 64


# ---------------------------------------------------------------------------
# head_dim 工具
# ---------------------------------------------------------------------------


def _next_allowed_head_dim(d: int) -> int:
    for c in _ALLOWED_HEAD_DIMS:
        if c >= d:
            return c
    raise ValueError(
        f"head_dim={d} 超过支持的最大值 {_ALLOWED_HEAD_DIMS[-1]}；MInference 上游同样不支持"
    )


def _pad_head_dim(t: torch.Tensor, target_d: int) -> torch.Tensor:
    cur = t.shape[-1]
    if cur == target_d:
        return t
    return F.pad(t, (0, target_d - cur))


def _preferred_mask_max_seq() -> int:
    raw = os.environ.get(_PREFER_MASK_MAX_SEQ_ENV)
    if raw is None or raw == "":
        return _DEFAULT_PREFER_MASK_MAX_SEQ
    try:
        return max(0, int(raw))
    except ValueError:
        warnings.warn(
            f"{_PREFER_MASK_MAX_SEQ_ENV}={raw!r} 不是合法整数；"
            f"回退默认值 {_DEFAULT_PREFER_MASK_MAX_SEQ}。",
            stacklevel=2,
        )
        return _DEFAULT_PREFER_MASK_MAX_SEQ


def _preferred_mask_max_heads() -> int:
    raw = os.environ.get(_PREFER_MASK_MAX_HEADS_ENV)
    if raw is None or raw == "":
        return _DEFAULT_PREFER_MASK_MAX_HEADS
    try:
        return max(0, int(raw))
    except ValueError:
        warnings.warn(
            f"{_PREFER_MASK_MAX_HEADS_ENV}={raw!r} 不是合法整数；"
            f"回退默认值 {_DEFAULT_PREFER_MASK_MAX_HEADS}。",
            stacklevel=2,
        )
        return _DEFAULT_PREFER_MASK_MAX_HEADS


def _should_prefer_mask_npu(seq_len: int, num_heads: int = 1) -> bool:
    return (
        seq_len <= min(_MAX_SEQ_FOR_MASK, _preferred_mask_max_seq())
        and num_heads <= _preferred_mask_max_heads()
    )


def _should_use_tilelang_h1_query_block(seq_len_q: int, seq_len_k: int, block_size: int) -> bool:
    raw = os.environ.get(_TILELANG_H1_QUERY_BLOCK_ENV)
    if raw is not None and raw.strip().lower() in {"0", "false", "off", "no"}:
        return False

    return (
        seq_len_q % _TILELANG_H1_BLOCK_M == 0
        and seq_len_q % block_size == 0
        and seq_len_k % block_size == 0
        and block_size % _TILELANG_H1_BLOCK_M == 0
    )


def _should_use_tilelang_h1_block_index(seq_len_q: int, seq_len_k: int, block_size: int) -> bool:
    raw = os.environ.get(_TILELANG_H1_BLOCK_INDEX_ENV)
    if raw is not None and raw.strip().lower() in {"0", "false", "off", "no"}:
        return False
    return _should_use_tilelang_h1_query_block(seq_len_q, seq_len_k, block_size)


# ---------------------------------------------------------------------------
# block-sparse mask 构建（NPU 路径与 PyTorch 路径共用）
# ---------------------------------------------------------------------------


def _select_block_sparse_topk_indices(
    q: torch.Tensor,
    k: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """返回每个 query block 选中的 K block 索引，形状 ``[B,H,n_bq,topk]``。"""
    B, H, S_q, D = q.shape
    S_k = k.shape[2]

    pad_q = (-S_q) % block_size  # 0 if S_q % block_size == 0
    pad_k = (-S_k) % block_size
    S_q_p = S_q + pad_q
    S_k_p = S_k + pad_k
    n_bq = S_q_p // block_size
    n_bk = S_k_p // block_size

    q_p = F.pad(q, (0, 0, 0, pad_q)) if pad_q else q  # [B, H, S_q_p, D]
    k_p = F.pad(k, (0, 0, 0, pad_k)) if pad_k else k  # [B, H, S_k_p, D]

    q_pool = q_p.reshape(B, H, n_bq, block_size, D).mean(dim=3).float()  # [B, H, n_bq, D]
    k_pool = k_p.reshape(B, H, n_bk, block_size, D).mean(dim=3).float()  # [B, H, n_bk, D]

    scale = D ** -0.5
    scores = torch.matmul(q_pool, k_pool.transpose(-2, -1)) * scale  # [B, H, n_bq, n_bk]

    bq_idx = torch.arange(n_bq, device=q.device)
    bk_idx = torch.arange(n_bk, device=q.device)
    causal_block = bq_idx[:, None] >= bk_idx[None, :]  # [n_bq, n_bk]
    scores.masked_fill_(~causal_block[None, None], float("-inf"))

    topk = min(topk_blocks, n_bk)
    return torch.topk(scores, topk, dim=-1).indices.sort(dim=-1).values  # [B, H, n_bq, topk]


def _build_block_sparse_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """构建 block-sparse attention 的 token 级 bool mask。

    返回形状 ``[B, H, S_q, S_k]``，``True`` = 被遮蔽（不参与 attention），即 NPU 惯例。

    算法：
    1. 将 Q/K 按 block_size pad 并 mean-pool 到 block 级。
    2. 计算 block 级 QK 得分 + 施加因果约束（key block index <= query block index）。
    3. 每个 query block 选 top-k key blocks。
    4. 展开到 token 级（block_attend → token_attend）。
    5. 追加 per-token 因果约束（消除 top-k -inf tie-breaking 引入的误差）。

    注意：per-token 因果约束是最终正确性的保证；block 级 top-k 中的 -inf tie-breaking
    即使选中了超前 key block，也会被 per-token 因果约束过滤掉。
    """
    B, H, S_q, D = q.shape
    S_k = k.shape[2]
    topk_idx = _select_block_sparse_topk_indices(q, k, topk_blocks, block_size)

    # --- 1. 展开 block mask 到 token 级 [B, H, S_q_p, S_k_p] ---
    pad_q = (-S_q) % block_size  # 0 if S_q % block_size == 0
    pad_k = (-S_k) % block_size
    S_q_p = S_q + pad_q
    S_k_p = S_k + pad_k
    n_bq = S_q_p // block_size
    n_bk = S_k_p // block_size

    block_attend = torch.zeros(B, H, n_bq, n_bk, dtype=torch.bool, device=q.device)
    block_attend.scatter_(-1, topk_idx, True)

    # 数学验证：reshape 后 token[b,h,bq*bs+bqi, bk*bs+bki] = block_attend[b,h,bq,bk] ✓
    token_attend = (
        block_attend.unsqueeze(3).unsqueeze(5)            # [B, H, n_bq, 1, n_bk, 1]
        .expand(-1, -1, -1, block_size, -1, block_size)  # [B, H, n_bq, bs, n_bk, bs]
        .reshape(B, H, S_q_p, S_k_p)                     # [B, H, S_q_p, S_k_p]
    )
    token_attend = token_attend[:, :, :S_q, :S_k]  # trim padding

    # --- 2. 追加 per-token 因果约束 ---
    # query token i 的绝对位置 = (S_k - S_q) + i；可见 key token j 满足 j <= abs_i
    abs_i = torch.arange(S_k - S_q, S_k, device=q.device)  # [S_q]
    j_idx = torch.arange(S_k, device=q.device)              # [S_k]
    causal_token = abs_i[:, None] >= j_idx[None, :]  # [S_q, S_k]，True = 可见
    token_attend = token_attend & causal_token[None, None]

    # NPU 惯例：True = 被遮蔽
    return (~token_attend).contiguous()


# ---------------------------------------------------------------------------
# PyTorch 参考（非 NPU 兜底 + 测试黄金）
# ---------------------------------------------------------------------------


def _block_sparse_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Block-sparse attention 的纯 PyTorch 参考实现。

    与 NPU 路径共用 ``_build_block_sparse_mask``，差别仅在用 PyTorch masked softmax
    代替 ``npu_fusion_attention``。用作：
    1. 非 NPU 环境（CPU / CUDA / ``torch_npu`` 未安装）的兜底
    2. 单测黄金参考（与 NPU 输出对比）
    """
    mask = _build_block_sparse_mask(q, k, topk_blocks, block_size)  # [B, H, S_q, S_k] True=masked
    scale = q.shape[-1] ** -0.5

    # fp32 计算防 fp16 溢出
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale  # [B, H, S_q, S_k]
    logits.masked_fill_(mask, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)  # 整行被遮蔽时避免 NaN
    out = torch.matmul(probs, v.float())
    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# NPU 路径
# ---------------------------------------------------------------------------


def _block_sparse_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """NPU 路径：构建 token 级 bool mask → npu_fusion_attention(sparse_mode=1)。

    调用前提：
    * ``q.device.type == "npu"`` 且 ``_HAS_TORCH_NPU``
    * head_dim 已 pad 至 ``_ALLOWED_HEAD_DIMS`` 内
    * ``max(S_q, S_k) <= _MAX_SEQ_FOR_MASK``（外层已检查）

    返回 ``[B, H, S_q, D]``，dtype 与输入一致。
    """
    assert _HAS_TORCH_NPU, "_block_sparse_npu 不能在非 NPU 环境调用"

    n_heads = q.shape[1]
    scale = q.shape[-1] ** -0.5

    mask = _build_block_sparse_mask(q, k, topk_blocks, block_size)  # [B, H, S_q, S_k] bool

    try:
        result = torch_npu.npu_fusion_attention(  # type: ignore[union-attr]
            q,
            k,
            v,
            head_num=n_heads,
            input_layout="BNSD",
            scale=scale,
            sparse_mode=1,      # user-provided mask
            atten_mask=mask,    # True = masked out
        )
    except TypeError:
        result = torch_npu.npu_fusion_attention(  # type: ignore[union-attr]
            q,
            k,
            v,
            head_num=n_heads,
            input_layout="BNSD",
            scale=scale,
            sparse_mode=1,
            atten_mask=mask.to(torch.uint8),
        )
    # npu_fusion_attention 返回元组 (out, softmax_max, softmax_sum, ...)
    return result[0] if isinstance(result, (tuple, list)) else result


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


def _set_current_npu_device_for(tensor: torch.Tensor) -> str | None:
    """Align torch_npu's current device with the tensor before TileLang JIT/call.

    TileLang callables are compiled and launched against the current NPU
    context. In HF ``device_map=auto`` runs different layers may live on
    different NPU devices, so the Python kernel cache must also be keyed by
    device to avoid reusing an npu:0 callable on npu:1 tensors.
    """
    if tensor.device.type != "npu" or not hasattr(torch, "npu"):
        return None
    device_index = tensor.device.index
    if device_index is None:
        return None
    torch.npu.set_device(device_index)
    return f"npu:{device_index}"


def _block_sparse_tilelang_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Path B: TileLang true sparse attention.

    The separate-Q/K/V TileLang kernel currently keeps ``heads=1, kv_group=1``.
    To support grouped scheduling with independent per-head K/V tensors, fold
    the selected heads into the batch dimension and run a single TileLang
    launch for the whole group.
    """
    if q.dtype != torch.float16:
        raise NotImplementedError("TileLang block-sparse MVP only supports fp16")
    if k.shape[1] != q.shape[1] or v.shape[1] != q.shape[1]:
        raise NotImplementedError("TileLang block-sparse MVP expects repeated per-head K/V")

    B, H, S_q, D = q.shape
    S_k = k.shape[2]
    flat_B = B * H
    cache_device = _set_current_npu_device_for(q)

    q_flat = q.contiguous().reshape(flat_B, 1, S_q, D)
    k_flat = k.contiguous().reshape(flat_B, 1, S_k, D)
    v_flat = v.contiguous().reshape(flat_B, 1, S_k, D)

    block_indices = _select_block_sparse_topk_indices(q_flat, k_flat, topk_blocks, block_size)

    q_bshd = q_flat.transpose(1, 2).contiguous()
    k_bsgd = k_flat.transpose(1, 2).contiguous()
    v_bsgd = v_flat.transpose(1, 2).contiguous()

    if _should_use_tilelang_h1_block_index(S_q, S_k, block_size):
        tilelang_sparse_attention_h1 = _load_sibling_module(
            "tilelang_sparse_attention_h1_block_sparse_standalone",
            "tilelang_sparse_attention_h1.py",
        )
        block_indices_bsh = block_indices.permute(0, 2, 1, 3).contiguous().to(
            device=q.device,
            dtype=torch.int32,
        )
        kernel = tilelang_sparse_attention_h1.build_sparse_attention_h1_block_index_fwd(
            dim=D,
            topk_blocks=block_indices_bsh.shape[-1],
            block_M=_TILELANG_H1_BLOCK_M,
            block_I=block_size,
            q_start_index_s=S_k - S_q,
            cache_device=cache_device,
        )
        out_bshd = kernel(q_bshd, k_bsgd, v_bsgd, block_indices_bsh)
        return out_bshd.transpose(1, 2).contiguous().reshape(B, H, S_q, D)

    tilelang_indices = _load_sibling_module(
        "tilelang_indices_block_sparse_standalone",
        "tilelang_indices.py",
    )
    tilelang_sparse_attention = _load_sibling_module(
        "tilelang_sparse_attention_block_sparse_standalone",
        "tilelang_sparse_attention.py",
    )
    indices = tilelang_indices.block_indices_to_tilelang(
        block_indices,
        S_q=S_q,
        block_size_M=block_size,
        block_size_N=block_size,
        kv_heads=1,
    )
    indices = torch.where(
        (indices >= 0) & (indices < S_k),
        indices,
        torch.full_like(indices, tilelang_indices.TILELANG_PAD_VALUE),
    ).to(device=q.device, dtype=torch.int32)

    if _should_use_tilelang_h1_query_block(S_q, S_k, block_size):
        tilelang_sparse_attention_h1 = _load_sibling_module(
            "tilelang_sparse_attention_h1_block_sparse_standalone",
            "tilelang_sparse_attention_h1.py",
        )
        kernel = tilelang_sparse_attention_h1.build_sparse_attention_h1_block_fwd(
            dim=D,
            topk=indices.shape[-1],
            block_M=_TILELANG_H1_BLOCK_M,
            block_I=block_size,
            q_start_index_s=S_k - S_q,
            cache_device=cache_device,
        )
    else:
        kernel = tilelang_sparse_attention.build_sparse_attention_qkv_fwd(
            heads=1,
            dim=D,
            topk=indices.shape[-1],
            kv_group=1,
            block_I=block_size,
            q_start_index_s=S_k - S_q,
            use_contiguous_range_load=(S_k % block_size == 0),
            cache_device=cache_device,
        )

    out_bshd = kernel(q_bshd, k_bsgd, v_bsgd, indices)
    return out_bshd.transpose(1, 2).contiguous().reshape(B, H, S_q, D)


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------


def block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_blocks: int,
    block_size: int = _DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """Block-sparse causal attention.

    Args:
        q, k, v:      ``[B, H, S, D]``，已 RoPE、已 ``repeat_kv``（与上游一致）。
        topk_blocks:  每个 query block 保留的 key block 数量（与上游 ``top_k`` 对应）。
                      来自 ``best_pattern`` 的 ``vertical_size`` 字段。
        block_size:   block 大小（默认 64，与上游一致）。

    Returns:
        ``[B, H, S, D]``，与 ``q`` 同 dtype 同 device。

    短路 / 降级：
    * fp16 NPU 短序列默认优先走 bool-mask NPU path A，便于 4K-16K 过渡和对照。
      阈值由 ``MINFERENCE_BLOCK_SPARSE_MASK_MAX_SEQ`` 控制，设为 0 可强制 TileLang。
    * 长序列 fp16 NPU 优先走 TileLang path-B。
    * TileLang 不适用时，短序列回退 bool-mask NPU；过长序列回退 causal dense。
    * 非 NPU 设备：走 ``_block_sparse_pytorch_ref`` 参考路径。

    Notes:
        head_dim 不在 ``{16,32,64,128,256,512}`` 时自动 pad 到 2 的幂，输出截回原始 head_dim。
        TileLang path-B 当前仍把多 head 折叠到 batch 维；这是下一步优化重点。
        2. 入参为 ``topk_blocks``（块数绝对值），对应上游 ``top_k``。
        3. 追加 per-token 因果约束保证对角块内正确性。
    """
    assert q.dim() == 4, f"q 期望 4D [B,H,S,D]，得到 {q.dim()}D"
    assert (
        q.shape[0] == k.shape[0] and q.shape[1] == k.shape[1] and q.shape[-1] == k.shape[-1]
    ), f"q/k B/H/D 不匹配；q={tuple(q.shape)} k={tuple(k.shape)}"

    topk_blocks = max(1, int(topk_blocks))  # 防止 ≤ 0

    orig_head_d = q.shape[-1]
    head_d = orig_head_d

    # head_dim pad
    if head_d not in _ALLOWED_HEAD_DIMS:
        target_d = _next_allowed_head_dim(head_d)
        q = _pad_head_dim(q, target_d)
        k = _pad_head_dim(k, target_d)
        v = _pad_head_dim(v, target_d)
        head_d = target_d

    S = max(q.shape[2], k.shape[2])

    # 主路径
    if q.device.type == "npu" and _HAS_TORCH_NPU:
        if q.dtype == torch.float16:
            if _should_prefer_mask_npu(S, q.shape[1]):
                try:
                    out = _block_sparse_npu(q, k, v, topk_blocks, block_size)
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(
                        "block_sparse_attention: short-sequence bool-mask path failed "
                        f"({exc}); falling back to TileLang path B.",
                        stacklevel=2,
                    )
                    out = _block_sparse_tilelang_npu(q, k, v, topk_blocks, block_size)
            else:
                try:
                    out = _block_sparse_tilelang_npu(q, k, v, topk_blocks, block_size)
                except ImportError as exc:
                    warnings.warn(
                        "block_sparse_attention: TileLang path B is unavailable "
                        f"({exc}); falling back to path A bool-mask attention.",
                        stacklevel=2,
                    )
                    if S > _MAX_SEQ_FOR_MASK:
                        warnings.warn(
                            f"block_sparse_attention: 序列长度 {S} > {_MAX_SEQ_FOR_MASK}。"
                            " bool mask 构建需 O(S²) 内存，退化为 causal dense。",
                            stacklevel=2,
                        )
                        from ..backend_npu import dense_attention

                        out = dense_attention(q, k, v, causal=True)
                    else:
                        out = _block_sparse_npu(q, k, v, topk_blocks, block_size)
        else:
            if S > _MAX_SEQ_FOR_MASK:
                warnings.warn(
                    f"block_sparse_attention: 序列长度 {S} > {_MAX_SEQ_FOR_MASK}。"
                    " bool mask 构建需 O(S²) 内存，退化为 causal dense。"
                    " 当前 TileLang path-B 仅接入 fp16 形态。",
                    stacklevel=2,
                )
                from ..backend_npu import dense_attention

                out = dense_attention(q, k, v, causal=True)
            else:
                out = _block_sparse_npu(q, k, v, topk_blocks, block_size)
    else:
        out = _block_sparse_pytorch_ref(q, k, v, topk_blocks, block_size)

    return out[..., :orig_head_d]
