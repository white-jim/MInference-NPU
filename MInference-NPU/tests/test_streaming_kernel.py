# Copyright (c) 2026 (NPU 适配 — M2 streaming kernel 单测)
# Licensed under The MIT License [see LICENSE for details]
"""M2 streaming kernel 单测。

测试目标
--------
* CPU / 非 NPU 路径（``_streaming_pytorch_ref``）的正确性：与"逐 row naive 实现"对比，
  max_abs_diff < 1e-3（fp32 内部计算，不会有精度损失）。
* NPU 路径（``_streaming_npu``）与 PyTorch ref 对比，max_abs_diff < 1e-2（fp16 误差容限）。
* 顶层 ``streaming_forward`` 短路逻辑（k_len <= n_local → dense）。
* head_dim pad/截回（非 2 幂 head_dim 透明传入透明传出）。

运行方法
--------
::

    # CPU/非 NPU 环境（仅跑 ref 正确性 + 短路 + head_dim pad）：
    python -m pytest tests/test_streaming_kernel.py -v

    # NPU 环境（额外跑 NPU vs ref 对比）：
    python -m pytest tests/test_streaming_kernel.py -v
    # NPU 测试在非 NPU 机器上自动 skip，无需单独开关。
"""

from __future__ import annotations

import importlib.util as _ilu
import math
import os
import sys
import types
import pytest
import torch

# ---------------------------------------------------------------------------
# 导入被测模块（streaming_kernel_npu 的内部函数 + 顶层接口）
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pkg = types.ModuleType("minference")
pkg.__path__ = [os.path.join(_REPO_ROOT, "minference")]
sys.modules.setdefault("minference", pkg)
ops_pkg = types.ModuleType("minference.ops")
ops_pkg.__path__ = [os.path.join(_REPO_ROOT, "minference", "ops")]
sys.modules.setdefault("minference.ops", ops_pkg)


def _load_module(module_name: str, relative_path: str):
    path = os.path.join(_REPO_ROOT, *relative_path.split("/"))
    spec = _ilu.spec_from_file_location(module_name, path)
    module = _ilu.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


streaming_kernel_npu = _load_module(
    "minference.ops.streaming_kernel_npu",
    "minference/ops/streaming_kernel_npu.py",
)

_streaming_pytorch_ref = streaming_kernel_npu._streaming_pytorch_ref
streaming_forward = streaming_kernel_npu.streaming_forward

try:
    import torch_npu  # type: ignore[import-not-found]  # noqa: F401

    _streaming_npu = streaming_kernel_npu._streaming_npu
    _HAS_NPU = True
except ImportError:
    _HAS_NPU = False

# ---------------------------------------------------------------------------
# 黄金参考（独立 naive 实现，用于校验 _streaming_pytorch_ref）
# ---------------------------------------------------------------------------


def _naive_streaming(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_init: int,
    n_local: int,
) -> torch.Tensor:
    """逐 query-row 的 naive 实现，用作对照黄金。

    遍历每个 query 位置，手工构造 bool 可见 mask，再做 fp32 softmax 加权求和。
    正确性显而易见（唯一算法路径），但速度慢；仅用于小规模测试。
    """
    bsz, n_heads, s_q, head_d = q.shape
    s_k = k.shape[2]
    scale = 1.0 / math.sqrt(head_d)
    out = torch.zeros_like(q, dtype=torch.float32)

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    for i in range(s_q):
        abs_i = s_k - s_q + i  # query i 对应的绝对 key 位置
        # sink: j < n_init
        # sliding: j ∈ [abs_i - n_local + 1, abs_i]
        # causal: j <= abs_i（已被 sliding 和 sink 隐式包含，显式保留防止边界问题）
        j = torch.arange(s_k, device=q.device)
        sink_mask = j < n_init
        slide_mask = (j <= abs_i) & (j > abs_i - n_local)
        visible = (sink_mask | slide_mask) & (j <= abs_i)

        q_i = q_f[:, :, i : i + 1, :]  # [B, H, 1, D]
        logits = torch.matmul(q_i, k_f.transpose(-2, -1)) * scale  # [B, H, 1, S_k]
        logits = logits.masked_fill(~visible[None, None, None, :], float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        out[:, :, i : i + 1, :] = torch.matmul(probs, v_f)

    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# 测试用例参数（n_init, n_local, s_q, s_k, head_dim, dtype）
# ---------------------------------------------------------------------------

# (n_init, n_local, s_q, s_k, description)
_STREAMING_CASES = [
    # 短路：k_len <= n_local → dense（streaming_forward 走 dense 分支）
    # 注意：prefill 场景 s_q == s_k；s_q > s_k 会让 causal 下前 (s_q-s_k) 个 query
    # 看不到任何 key，softmax(-inf) → nan，并非被测代码 bug 而是用例本身不合法。
    (64, 512, 256, 256, "short_circuit_klen_eq_nlocal"),
    (64, 512, 128, 128, "short_circuit_klen_lt_nlocal"),
    # 边界：k_len = n_local + 1（刚好触发 streaming 路径）
    (4, 8, 9, 9, "boundary_klen_eq_nlocal_p1"),
    # n_init = 0：无 sink，只有 sliding window
    (0, 16, 32, 64, "no_sink"),
    # 标准长上下文
    (128, 512, 512, 1024, "standard_long_ctx"),
    # 默认配置（n_init=128, n_local=3968，缩短版）
    (128, 256, 256, 512, "default_like_config"),
    # n_init 覆盖大半 k_len（sink 段与 sliding 大量重叠）
    (400, 128, 128, 512, "large_sink_overlap"),
]

_DTYPES = [torch.float32, torch.float16]
_HEAD_DIMS = [64, 128]


def _make_qkv(bsz, n_heads, s_q, s_k, head_d, dtype, device="cpu", seed=42):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    q = torch.randn(bsz, n_heads, s_q, head_d, dtype=dtype, device=device, generator=g)
    k = torch.randn(bsz, n_heads, s_k, head_d, dtype=dtype, device=device, generator=g)
    v = torch.randn(bsz, n_heads, s_k, head_d, dtype=dtype, device=device, generator=g)
    return q, k, v


# ---------------------------------------------------------------------------
# 1. _streaming_pytorch_ref vs naive（CPU, fp32）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _STREAMING_CASES, ids=[c[4] for c in _STREAMING_CASES])
def test_pytorch_ref_vs_naive(case):
    n_init, n_local, s_q, s_k, _desc = case
    bsz, n_heads, head_d = 1, 2, 64
    q, k, v = _make_qkv(bsz, n_heads, s_q, s_k, head_d, torch.float32)

    ref = _streaming_pytorch_ref(q, k, v, n_init, n_local)
    gold = _naive_streaming(q, k, v, n_init, n_local)

    assert ref.shape == gold.shape == (bsz, n_heads, s_q, head_d)
    max_diff = (ref.float() - gold.float()).abs().max().item()
    assert max_diff < 1e-3, (
        f"[{_desc}] pytorch_ref vs naive: max_abs_diff={max_diff:.2e} >= 1e-3"
    )


# ---------------------------------------------------------------------------
# 2. streaming_forward 顶层接口正确性（CPU, fp32 + fp16）
#    包含短路分支（k_len <= n_local → dense）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f16"])
@pytest.mark.parametrize("case", _STREAMING_CASES, ids=[c[4] for c in _STREAMING_CASES])
def test_streaming_forward_vs_naive(case, dtype):
    n_init, n_local, s_q, s_k, _desc = case
    bsz, n_heads, head_d = 1, 2, 64
    q, k, v = _make_qkv(bsz, n_heads, s_q, s_k, head_d, dtype)

    out = streaming_forward(q, k, v, n_init=n_init, n_local=n_local)
    assert out.shape == (bsz, n_heads, s_q, head_d), (
        f"output shape mismatch: {out.shape} vs expected {(bsz, n_heads, s_q, head_d)}"
    )
    assert out.dtype == dtype, f"output dtype {out.dtype} != input dtype {dtype}"

    # 短路情形用 causal dense 黄金比对
    if s_k <= n_local:
        # causal dense golden：逐 row naive（仅限因果 visible = j <= abs_i）
        gold = _naive_streaming(q.float(), k.float(), v.float(), n_init=0, n_local=s_k + 1)
        max_diff = (out.float() - gold.float()).abs().max().item()
        tol = 1e-2 if dtype == torch.float16 else 1e-3
        assert max_diff < tol, (
            f"[{_desc}/short_circuit] max_abs_diff={max_diff:.2e} >= {tol}"
        )
    else:
        gold = _naive_streaming(q.float(), k.float(), v.float(), n_init, n_local)
        max_diff = (out.float() - gold.float()).abs().max().item()
        tol = 1e-2 if dtype == torch.float16 else 1e-3
        assert max_diff < tol, (
            f"[{_desc}] max_abs_diff={max_diff:.2e} >= {tol}"
        )


# ---------------------------------------------------------------------------
# 3. head_dim pad/截回透明性（非 2 幂 head_dim）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_d", [48, 96, 80])
def test_head_dim_pad_roundtrip(head_d):
    """非标准 head_dim 进入 streaming_forward 应得到正确形状、且不改变 dtype。"""
    bsz, n_heads, s_q, s_k = 1, 2, 64, 256
    n_init, n_local = 32, 64
    q, k, v = _make_qkv(bsz, n_heads, s_q, s_k, head_d, torch.float32)

    out = streaming_forward(q, k, v, n_init=n_init, n_local=n_local)
    assert out.shape == (bsz, n_heads, s_q, head_d), (
        f"head_dim={head_d}: output shape {out.shape} 不等于输入形状"
    )
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# 4. NPU 路径 vs PyTorch ref（仅在 NPU 环境运行）
# ---------------------------------------------------------------------------

_NPU_CASES = [
    (128, 512, 256, 1024, "npu_standard"),
    (0, 256, 128, 512, "npu_no_sink"),
    (512, 128, 64, 1024, "npu_large_sink"),
]


@pytest.mark.skipif(not _HAS_NPU, reason="需要 torch_npu 且有 NPU 设备")
@pytest.mark.parametrize("case", _NPU_CASES, ids=[c[4] for c in _NPU_CASES])
@pytest.mark.parametrize("dtype", [torch.float16], ids=["f16"])
def test_npu_vs_pytorch_ref(case, dtype):
    """NPU 上 _streaming_npu 与 _streaming_pytorch_ref 的数值对比，容差 1e-2。"""
    n_init, n_local, s_q, s_k, _desc = case
    bsz, n_heads, head_d = 1, 2, 128
    device = torch.device("npu:0")

    q, k, v = _make_qkv(bsz, n_heads, s_q, s_k, head_d, dtype, device=str(device))

    # PyTorch ref 在 CPU 上跑（NPU 上 PyTorch eager 可能更慢/有限制）
    q_cpu = q.cpu().float()
    k_cpu = k.cpu().float()
    v_cpu = v.cpu().float()
    ref = _streaming_pytorch_ref(q_cpu, k_cpu, v_cpu, n_init, n_local).to(dtype)

    npu_out = _streaming_npu(q, k, v, n_init, n_local).cpu()

    assert npu_out.shape == ref.shape, (
        f"[{_desc}] shape mismatch: npu={npu_out.shape} ref={ref.shape}"
    )
    max_diff = (npu_out.float() - ref.float()).abs().max().item()
    assert max_diff < 1e-2, (
        f"[{_desc}] npu vs pytorch_ref: max_abs_diff={max_diff:.2e} >= 1e-2"
    )


# ---------------------------------------------------------------------------
# 5. 多组 (n_init, n_local) 参数覆盖（CPU, fp32, 确保参数边界不崩）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_init,n_local",
    [
        (0, 64),    # 无 sink
        (1, 64),    # 最小 sink
        (64, 64),   # n_init == n_local
        (128, 64),  # n_init > n_local（sink 超出 sliding 宽度）
        (64, 1),    # 极窄 sliding window
    ],
    ids=["no_sink", "min_sink", "eq", "sink_gt_local", "narrow_window"],
)
def test_param_sweep(n_init, n_local):
    bsz, n_heads, s_q, s_k, head_d = 1, 1, 128, 256, 64
    q, k, v = _make_qkv(bsz, n_heads, s_q, s_k, head_d, torch.float32)

    ref = _streaming_pytorch_ref(q, k, v, n_init, n_local)
    gold = _naive_streaming(q, k, v, n_init, n_local)

    assert ref.shape == (bsz, n_heads, s_q, head_d)
    max_diff = (ref.float() - gold.float()).abs().max().item()
    assert max_diff < 1e-3, f"n_init={n_init},n_local={n_local}: max_diff={max_diff:.2e}"
