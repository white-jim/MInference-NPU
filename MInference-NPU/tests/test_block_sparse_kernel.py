# Copyright (c) 2026 (NPU 适配 — M3 block-sparse kernel 单测)
# Licensed under The MIT License [see LICENSE for details]
"""M3 block-sparse kernel 单测。

测试目标
--------
* ``_build_block_sparse_mask`` 正确性：
  - topk=all_blocks 时 mask 与纯因果 mask 一致（block-sparse 退化为 dense causal）
  - 因果约束：mask[..., i, j] == False 时必有 i >= j（即不会 attend 未来 token）
* ``_block_sparse_pytorch_ref`` 与 naive causal dense 在 topk=all_blocks 时数值一致
* 顶层 ``block_sparse_attention`` shape / dtype 保持
* head_dim pad/截回透明性（非 2 幂 head_dim）
* NPU 路径与 PyTorch ref 数值对比（仅在 NPU 环境运行）
* 参数扫描（不同 topk / block_size / seq_len 不崩溃）

运行方法
--------
::

    # CPU/非 NPU 环境（跑所有非 skip 测试）：
    python -m pytest tests/test_block_sparse_kernel.py -v

    # NPU 环境（额外跑 NPU vs ref 对比）：
    python -m pytest tests/test_block_sparse_kernel.py -v
"""

from __future__ import annotations

import importlib.util as _ilu
import math
import os

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(module_name: str, relative_path: str):
    path = os.path.join(_REPO_ROOT, *relative_path.split("/"))
    spec = _ilu.spec_from_file_location(module_name, path)
    module = _ilu.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


block_sparse_kernel_npu = _load_module(
    "block_sparse_kernel_npu_standalone",
    "minference/ops/block_sparse_kernel_npu.py",
)

_block_sparse_pytorch_ref = block_sparse_kernel_npu._block_sparse_pytorch_ref
_build_block_sparse_mask = block_sparse_kernel_npu._build_block_sparse_mask
_should_prefer_mask_npu = block_sparse_kernel_npu._should_prefer_mask_npu
_should_use_tilelang_h1_query_block = block_sparse_kernel_npu._should_use_tilelang_h1_query_block
_should_use_tilelang_h1_block_index = block_sparse_kernel_npu._should_use_tilelang_h1_block_index
_should_use_tilelang_mh_block_index = block_sparse_kernel_npu._should_use_tilelang_mh_block_index
block_sparse_attention = block_sparse_kernel_npu.block_sparse_attention

try:
    import torch_npu  # type: ignore[import-not-found]  # noqa: F401

    _block_sparse_npu = block_sparse_kernel_npu._block_sparse_npu
    _HAS_NPU = True
except ImportError:
    _HAS_NPU = False


# ---------------------------------------------------------------------------
# 黄金参考：标准 causal dense（用于 topk=all_blocks 时对照）
# ---------------------------------------------------------------------------


def _causal_dense_ref(q, k, v):
    """因果 dense attention，fp32 softmax，无 block 约束。用于校验 topk=all_blocks 时的等价性。"""
    B, H, S_q, D = q.shape
    S_k = k.shape[2]
    scale = D ** -0.5

    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale  # [B, H, S_q, S_k]
    # causal: abs_i = (S_k - S_q) + i
    abs_i = torch.arange(S_k - S_q, S_k, device=q.device)  # [S_q]
    j = torch.arange(S_k, device=q.device)                  # [S_k]
    causal = abs_i[:, None] >= j[None, :]                   # [S_q, S_k]
    logits.masked_fill_(~causal[None, None], float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.matmul(probs, v.float()).to(q.dtype)


# ---------------------------------------------------------------------------
# 工具：生成随机 qkv
# ---------------------------------------------------------------------------


def _make_qkv(bsz, n_heads, s_q, s_k, head_d, dtype=torch.float32, device="cpu", seed=0):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    q = torch.randn(bsz, n_heads, s_q, head_d, dtype=dtype, device=device, generator=g)
    k = torch.randn(bsz, n_heads, s_k, head_d, dtype=dtype, device=device, generator=g)
    v = torch.randn(bsz, n_heads, s_k, head_d, dtype=dtype, device=device, generator=g)
    return q, k, v


# ---------------------------------------------------------------------------
# 1. mask 正确性：topk=all_blocks → 等价于纯因果 mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s,block_size", [
    (64, 16),
    (128, 64),
    (256, 64),
    (300, 64),   # 非整除 block_size
    (512, 128),
])
def test_mask_full_topk_equals_causal(s, block_size):
    """topk_blocks >= n_bk 时，block-sparse mask 应等价于纯因果 mask。"""
    B, H, D = 1, 1, 64
    q, k, v = _make_qkv(B, H, s, s, D)
    n_bk = math.ceil(s / block_size)

    mask_sparse = _build_block_sparse_mask(q, k, topk_blocks=n_bk, block_size=block_size)
    # 纯因果 mask（True = masked）
    i_idx = torch.arange(s)
    j_idx = torch.arange(s)
    causal_mask = ~(i_idx[:, None] >= j_idx[None, :])  # True = masked
    causal_mask_4d = causal_mask[None, None]  # [1, 1, s, s]

    assert mask_sparse.shape == (B, H, s, s)
    assert torch.equal(mask_sparse, causal_mask_4d), (
        "block-sparse mask with full topk 应与纯因果 mask 相同"
    )


@pytest.mark.parametrize("s,block_size", [
    (64, 16),
    (256, 64),
])
def test_mask_causal_invariant(s, block_size):
    """mask[..., i, j] == False（可见）时必有 abs_i >= j（因果约束不可违反）。"""
    B, H, D = 1, 1, 64
    q, k, v = _make_qkv(B, H, s, s, D)
    topk = max(1, math.ceil(s / block_size) // 2)  # 取一半 blocks

    mask = _build_block_sparse_mask(q, k, topk_blocks=topk, block_size=block_size)
    # False = 可见；要求 i >= j（因 S_q == S_k，abs_i = i）
    attended = ~mask[0, 0]  # [S_q, S_k], True = attended
    i_idx = torch.arange(s)
    j_idx = torch.arange(s)
    future_attended = attended & (i_idx[:, None] < j_idx[None, :])
    assert not future_attended.any(), "block-sparse mask 中存在未来 token 被 attend 的情况（因果约束违反）"


# ---------------------------------------------------------------------------
# 2. PyTorch ref：topk=all_blocks → 与 causal dense 数值一致
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s,block_size", [
    (64, 16),
    (128, 64),
    (256, 64),
    (300, 64),
])
def test_pytorch_ref_full_topk_vs_dense(s, block_size):
    """topk_blocks = n_bk 时，_block_sparse_pytorch_ref 应与 causal dense 数值一致。"""
    B, H, D = 1, 2, 64
    q, k, v = _make_qkv(B, H, s, s, D, torch.float32)
    n_bk = math.ceil(s / block_size)

    ref = _block_sparse_pytorch_ref(q, k, v, topk_blocks=n_bk, block_size=block_size)
    gold = _causal_dense_ref(q, k, v)

    assert ref.shape == gold.shape == (B, H, s, D)
    max_diff = (ref.float() - gold.float()).abs().max().item()
    assert max_diff < 1e-4, (
        f"s={s},block_size={block_size}: pytorch_ref vs dense: max_diff={max_diff:.2e}"
    )


# ---------------------------------------------------------------------------
# 3. 顶层 block_sparse_attention：shape / dtype 保持
# ---------------------------------------------------------------------------

_SHAPE_CASES = [
    # (s_q, s_k, topk, block_size, dtype, description)
    (128, 128, 4,  64, torch.float32,  "f32_basic"),
    (128, 128, 4,  64, torch.float16,  "f16_basic"),
    (256, 256, 8,  64, torch.float32,  "f32_larger"),
    (300, 300, 5,  64, torch.float32,  "f32_nonalign"),
    (64,  64,  16, 16, torch.float32,  "f32_small_block"),
    (128, 128, 2,  64, torch.float32,  "f32_small_topk"),
]


@pytest.mark.parametrize("case", _SHAPE_CASES, ids=[c[5] for c in _SHAPE_CASES])
def test_shape_dtype(case):
    s_q, s_k, topk, block_size, dtype, _desc = case
    B, H, D = 1, 2, 64
    q, k, v = _make_qkv(B, H, s_q, s_k, D, dtype)

    out = block_sparse_attention(q, k, v, topk_blocks=topk, block_size=block_size)
    assert out.shape == (B, H, s_q, D), (
        f"[{_desc}] output shape {out.shape} != expected {(B, H, s_q, D)}"
    )
    assert out.dtype == dtype, f"[{_desc}] output dtype {out.dtype} != input dtype {dtype}"


# ---------------------------------------------------------------------------
# 4. head_dim pad 透明性（非 2 幂 head_dim）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_d", [48, 80, 96])
def test_head_dim_pad_roundtrip(head_d):
    """非标准 head_dim 应透明通过 block_sparse_attention（形状 / dtype 不变）。"""
    B, H, S = 1, 2, 128
    topk, block_size = 4, 64
    q, k, v = _make_qkv(B, H, S, S, head_d, torch.float32)

    out = block_sparse_attention(q, k, v, topk_blocks=topk, block_size=block_size)
    assert out.shape == (B, H, S, head_d), (
        f"head_d={head_d}: output shape {out.shape} 不等于输入 shape"
    )
    assert out.dtype == torch.float32


def test_short_seq_mask_path_policy_default(monkeypatch):
    """默认 4K-16K block_sparse 应优先走 bool-mask NPU 对照路径。"""
    monkeypatch.delenv("MINFERENCE_BLOCK_SPARSE_MASK_MAX_SEQ", raising=False)
    monkeypatch.delenv("MINFERENCE_BLOCK_SPARSE_MASK_MAX_HEADS", raising=False)
    assert _should_prefer_mask_npu(4096)
    assert _should_prefer_mask_npu(16384)
    assert not _should_prefer_mask_npu(16385)
    assert _should_prefer_mask_npu(16384, num_heads=16)
    assert not _should_prefer_mask_npu(16384, num_heads=17)


def test_short_seq_mask_path_policy_env_disable(monkeypatch):
    """阈值设为 0 时禁用短序列 path A，便于强制测 TileLang path-B。"""
    monkeypatch.setenv("MINFERENCE_BLOCK_SPARSE_MASK_MAX_SEQ", "0")
    assert not _should_prefer_mask_npu(1)
    assert not _should_prefer_mask_npu(4096)


def test_short_seq_mask_path_policy_env_invalid(monkeypatch):
    """非法阈值回退默认值，避免 benchmark 环境变量拼错后悄悄改变行为。"""
    monkeypatch.setenv("MINFERENCE_BLOCK_SPARSE_MASK_MAX_SEQ", "oops")
    with pytest.warns(UserWarning, match="不是合法整数"):
        assert _should_prefer_mask_npu(4096)


def test_short_seq_mask_path_policy_head_env(monkeypatch):
    """短序列 path A 还要限制一次 mask 覆盖的 head 数，避免 H=32 构造超大 mask。"""
    monkeypatch.delenv("MINFERENCE_BLOCK_SPARSE_MASK_MAX_SEQ", raising=False)
    monkeypatch.setenv("MINFERENCE_BLOCK_SPARSE_MASK_MAX_HEADS", "8")
    assert _should_prefer_mask_npu(4096, num_heads=8)
    assert not _should_prefer_mask_npu(4096, num_heads=9)


def test_short_seq_mask_path_policy_head_env_invalid(monkeypatch):
    """非法 head 阈值回退默认值。"""
    monkeypatch.setenv("MINFERENCE_BLOCK_SPARSE_MASK_MAX_HEADS", "oops")
    with pytest.warns(UserWarning, match="不是合法整数"):
        assert _should_prefer_mask_npu(4096, num_heads=16)


def test_tilelang_h1_query_block_policy_default(monkeypatch):
    """完整 block 且 S_q 可按 16 行切分时，TileLang path-B 默认启用 H=1 query-block kernel。"""
    monkeypatch.delenv("MINFERENCE_BLOCK_SPARSE_TILELANG_H1", raising=False)
    assert _should_use_tilelang_h1_query_block(4096, 4096, 64)
    assert not _should_use_tilelang_h1_query_block(4097, 4096, 64)
    assert not _should_use_tilelang_h1_query_block(4096, 4097, 64)
    assert not _should_use_tilelang_h1_query_block(4096, 4096, 24)


def test_tilelang_h1_query_block_policy_env_disable(monkeypatch):
    """设为 0 可回退旧 padded-H TileLang kernel，方便 A/B benchmark。"""
    monkeypatch.setenv("MINFERENCE_BLOCK_SPARSE_TILELANG_H1", "0")
    assert not _should_use_tilelang_h1_query_block(4096, 4096, 64)


def test_tilelang_h1_block_index_policy_default(monkeypatch):
    """block-index H=1 kernel 默认跟随 H=1 query-block 安全边界。"""
    monkeypatch.delenv("MINFERENCE_BLOCK_SPARSE_TILELANG_H1", raising=False)
    monkeypatch.delenv("MINFERENCE_BLOCK_SPARSE_TILELANG_BLOCK_INDEX", raising=False)
    assert _should_use_tilelang_h1_block_index(4096, 4096, 64)
    assert not _should_use_tilelang_h1_block_index(4097, 4096, 64)


def test_tilelang_h1_block_index_policy_env_disable(monkeypatch):
    """设为 0 可保留旧 token-index 展开路径，便于隔离 wrapper 开销。"""
    monkeypatch.delenv("MINFERENCE_BLOCK_SPARSE_TILELANG_H1", raising=False)
    monkeypatch.setenv("MINFERENCE_BLOCK_SPARSE_TILELANG_BLOCK_INDEX", "0")
    assert not _should_use_tilelang_h1_block_index(4096, 4096, 64)


def test_tilelang_mh_block_index_policy_default(monkeypatch):
    """MH path 默认启用，安全边界同 H=1 block-index。"""
    monkeypatch.delenv("MINFERENCE_BLOCK_SPARSE_TILELANG_H1", raising=False)
    monkeypatch.delenv("MINFERENCE_BLOCK_SPARSE_TILELANG_BLOCK_INDEX", raising=False)
    monkeypatch.delenv("MINFERENCE_BLOCK_SPARSE_TILELANG_MH", raising=False)
    assert _should_use_tilelang_mh_block_index(4096, 4096, 64)
    assert not _should_use_tilelang_mh_block_index(4097, 4096, 64)


def test_tilelang_mh_block_index_policy_env_disable(monkeypatch):
    """MINFERENCE_BLOCK_SPARSE_TILELANG_MH=0 强制回退 H=1 fold-into-batch。"""
    monkeypatch.delenv("MINFERENCE_BLOCK_SPARSE_TILELANG_H1", raising=False)
    monkeypatch.delenv("MINFERENCE_BLOCK_SPARSE_TILELANG_BLOCK_INDEX", raising=False)
    monkeypatch.setenv("MINFERENCE_BLOCK_SPARSE_TILELANG_MH", "0")
    assert not _should_use_tilelang_mh_block_index(4096, 4096, 64)


def test_tilelang_mh_block_index_follows_block_index_disable(monkeypatch):
    """MH path 依赖 block-index 路径；显式关闭 block-index 时 MH 也不应启用。"""
    monkeypatch.delenv("MINFERENCE_BLOCK_SPARSE_TILELANG_H1", raising=False)
    monkeypatch.setenv("MINFERENCE_BLOCK_SPARSE_TILELANG_BLOCK_INDEX", "0")
    monkeypatch.delenv("MINFERENCE_BLOCK_SPARSE_TILELANG_MH", raising=False)
    assert not _should_use_tilelang_mh_block_index(4096, 4096, 64)


# ---------------------------------------------------------------------------
# 5. NPU 路径 vs PyTorch ref（仅在 NPU 环境运行）
# ---------------------------------------------------------------------------

_NPU_CASES = [
    (256, 256, 4,  64, "npu_standard"),
    (128, 128, 8,  32, "npu_small_block"),
    (300, 300, 3,  64, "npu_nonalign"),
]


@pytest.mark.skipif(not _HAS_NPU, reason="需要 torch_npu 且有 NPU 设备")
@pytest.mark.parametrize("case", _NPU_CASES, ids=[c[4] for c in _NPU_CASES])
@pytest.mark.parametrize("dtype", [torch.float16], ids=["f16"])
def test_npu_vs_pytorch_ref(case, dtype):
    """NPU 上 _block_sparse_npu 与 _block_sparse_pytorch_ref 数值对比，容差 1e-2。"""
    s_q, s_k, topk, block_size, _desc = case
    B, H, D = 1, 2, 128
    device = torch.device("npu:0")

    q, k, v = _make_qkv(B, H, s_q, s_k, D, dtype, device=str(device))

    # PyTorch ref 在 CPU fp32 上跑
    q_cpu = q.cpu().float()
    k_cpu = k.cpu().float()
    v_cpu = v.cpu().float()
    ref = _block_sparse_pytorch_ref(q_cpu, k_cpu, v_cpu, topk_blocks=topk, block_size=block_size).to(dtype)

    npu_out = _block_sparse_npu(q, k, v, topk_blocks=topk, block_size=block_size).cpu()

    assert npu_out.shape == ref.shape, (
        f"[{_desc}] shape mismatch: npu={npu_out.shape} ref={ref.shape}"
    )
    max_diff = (npu_out.float() - ref.float()).abs().max().item()
    assert max_diff < 1e-2, (
        f"[{_desc}] npu vs pytorch_ref: max_abs_diff={max_diff:.2e} >= 1e-2"
    )


@pytest.mark.skipif(not _HAS_NPU, reason="需要 torch_npu 且有 NPU 设备")
def test_tilelang_path_b_forced_vs_pytorch_ref(monkeypatch):
    """强制短序列走 TileLang path-B，覆盖 block_sparse wrapper 的 range-load 接入。"""
    monkeypatch.setenv("MINFERENCE_BLOCK_SPARSE_MASK_MAX_SEQ", "0")
    B, H, S, D = 1, 2, 256, 64
    topk, block_size = 2, 64
    device = torch.device("npu:0")

    q, k, v = _make_qkv(B, H, S, S, D, torch.float16, device=str(device), seed=11)
    ref = _block_sparse_pytorch_ref(
        q.cpu().float(),
        k.cpu().float(),
        v.cpu().float(),
        topk_blocks=topk,
        block_size=block_size,
    ).to(torch.float16)

    out = block_sparse_attention(q, k, v, topk_blocks=topk, block_size=block_size).cpu()
    max_diff = (out.float() - ref.float()).abs().max().item()
    assert max_diff < 6e-2, f"TileLang path-B forced vs ref: max_abs_diff={max_diff:.2e}"


# ---------------------------------------------------------------------------
# 6. 参数扫描（不同 topk / block_size / s，保证不崩溃）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topk_blocks,block_size,s", [
    (1,  64, 256),   # 最少 block
    (16, 64, 256),   # 正常
    (100, 64, 256),  # topk > n_bk（应 clamp）
    (4,  32, 128),   # 较小 block_size
    (4,  64, 64),    # s == block_size（单个 block）
    (4,  64, 65),    # s = block_size + 1（刚好需要 pad）
])
def test_param_sweep(topk_blocks, block_size, s):
    B, H, D = 1, 1, 64
    q, k, v = _make_qkv(B, H, s, s, D, torch.float32)

    out = block_sparse_attention(q, k, v, topk_blocks=topk_blocks, block_size=block_size)
    assert out.shape == (B, H, s, D), (
        f"topk={topk_blocks},block_size={block_size},s={s}: shape {out.shape}"
    )
    assert not out.isnan().any(), "输出不应包含 NaN"
    assert not out.isinf().any(), "输出不应包含 Inf"
