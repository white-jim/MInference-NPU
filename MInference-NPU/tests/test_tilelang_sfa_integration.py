# Copyright (c) 2026 (NPU 适配 — PR-4-tl-sfa 集成精度闸门)
# Licensed under The MIT License [see LICENSE for details]
"""tilelang ``sparse_attention_fwd`` × 我们的 ``tilelang_indices.py`` 联调精度测试。

这是 PR-4-tl-BS / PR-4-tl-SL 之前的最后一道闸门：
    用 ``block_indices_to_tilelang`` / ``stream_llm_to_tilelang`` 构造 Indices →
    喂给 tilelang ``sparse_attention_fwd`` → 与 example 自带 reference 对比。

接口理解（probe 出来的真实情况，不是文档约定）：
    * Q layout: BSHD ``[B, S_q, H, dim+tail_dim]`` fp16
    * KV layout: BSGD ``[B, S_k, kv_group, dim+tail_dim]`` fp16 —— K 和 V packed
                 在同一张量；reference 里 ``k = kv``，``v = kv[..., :dim]``
                 （NSA / DeepSeek-V4 风格，V 是 K 的前 dim 段）
    * Indices: ``[B, S_q, kv_group, topk]`` int32 —— 与我们 tilelang_indices.py 一致
    * Pad sentinel: **``S_k``（kv 序列长）**，不是 -1。reference 用 ``mask.scatter`` 到
                    宽度 ``S_k + 1`` 然后裁掉最后一列，让 ≥ S_k 的索引自动失效
    * 调用：``func(q, kv, indices)`` 三参数。output 和 5 个 workspace 由
            ``@tilelang.jit(out_idx=[3], workspace_idx=[4..8])`` 自动分配
    * sm_scale 默认 = ``(dim + tail_dim) ** -0.5``（不是 dim ** -0.5）
    * 当前官方 example 源码断言 ``kv_group == 1``。因此本闸门先验证 4 个 Q heads
      共享一组 KV/Indices 的路径；per-head MHA/GQA 映射留到 PR-4-tl-BS kernel 适配。

PR-4-tl-sfa 闸门策略：
    测试就用 NSA 风格输入（dim + tail_dim packed KV），不试图把 standard MInference
    的 K/V 分离对接到这个 packed kernel。那是 PR-4-tl-BS 阶段的工作。
    本闸门只验证：我们 ``tilelang_indices.py`` 输出的 Indices 张量喂给 kernel 后，
    结果与官方 reference 一致。

运行（NPU 机器，flexhead-tl conda env, PYTHONPATH=~/tilelang-ascend）::

    python tests/test_tilelang_sfa_integration.py --probe
    python tests/test_tilelang_sfa_integration.py --case sanity
    python tests/test_tilelang_sfa_integration.py --case bs
    python tests/test_tilelang_sfa_integration.py --case sl
    python tests/test_tilelang_sfa_integration.py  # 跑全部
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util as _ilu
import inspect
import os
import sys
import traceback
from typing import Callable

import torch

# 自洽：把 MInference-NPU 仓库根加进 sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 绕开 minference/__init__.py（它 eager import transformers，flexhead-tl 里没装）
_TI_PATH = os.path.join(_REPO_ROOT, "minference", "ops", "tilelang_indices.py")
_ti_spec = _ilu.spec_from_file_location("tilelang_indices_standalone", _TI_PATH)
tilelang_indices = _ilu.module_from_spec(_ti_spec)
_ti_spec.loader.exec_module(tilelang_indices)
TILELANG_PAD_VALUE = tilelang_indices.TILELANG_PAD_VALUE
block_indices_to_tilelang = tilelang_indices.block_indices_to_tilelang
stream_llm_to_tilelang = tilelang_indices.stream_llm_to_tilelang

__test__ = False  # 不让 pytest 当 collection target

# ---------------------------------------------------------------------------
# Import sparse_attention_fwd
# ---------------------------------------------------------------------------

_CANDIDATE_IMPORT_PATHS = [
    "tilelang.examples.sparse_flash_attention.example_sparse_flash_attn",
    "examples.sparse_flash_attention.example_sparse_flash_attn",
    "tilelang.ops.sparse_flash_attention",
    "tilelang.kernels.sparse_flash_attention",
    "tilelang.sparse_flash_attention",
]


def _load_sparse_attention_fwd_from_example_source(path: str) -> tuple[Callable, str] | None:
    """Disabled source-prefix loader for the official example.

    Executing only the prefix looks attractive because the example has a heavy top-level
    smoke test, but tilelang's JIT depends on the function being imported as part of the
    real module. Prefix execution can trip a TVM TIR builder error around Python globals
    such as ``REPLICATE_H``. Keep this hook disabled and use normal import.
    """
    return None


def _probe_sparse_attention_fwd() -> tuple[Callable, str]:
    errors = []
    for path in _CANDIDATE_IMPORT_PATHS:
        try:
            loaded = _load_sparse_attention_fwd_from_example_source(path)
            if loaded is not None:
                return loaded
            mod = importlib.import_module(path)
            if hasattr(mod, "sparse_attention_fwd"):
                return mod.sparse_attention_fwd, path
            errors.append(f"  {path}: module imported but has no sparse_attention_fwd")
        except ImportError as e:
            errors.append(f"  {path}: ImportError({e})")
    raise ImportError("无法找到 sparse_attention_fwd：\n" + "\n".join(errors))


def _print_signature(func: Callable, source: str) -> None:
    print(f"[probe] sparse_attention_fwd from: {source}")
    print(f"[probe] type: {type(func).__name__}")
    try:
        sig = inspect.signature(func)
        print(f"[probe] signature: sparse_attention_fwd{sig}")
    except (TypeError, ValueError) as e:
        print(f"[probe] inspect.signature failed: {e}")


# ---------------------------------------------------------------------------
# Reference 实现（从 example 改写：去掉 hardcoded assert，参数化 dim）
# ---------------------------------------------------------------------------


def _ref_sparse_attention_fwd_interface(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    dim: int,
    q_start_index_s: int = 0,
    kv_stride: int = 1,
    sm_scale: float | None = None,
    is_casual: bool = True,
) -> torch.Tensor:
    """example ``ref_sparse_attention_fwd_interface`` 的参数化版本。

    与 example 完全等价，区别仅在 ``dim`` 是参数（example 是硬编码 512）。
    """
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    _, sk, g, _ = kv.shape

    k = kv
    v = kv[..., :dim]

    dim_v = v.shape[-1]
    g_index = g
    h_index = h // g

    ref_device = q.device
    compressed_casual_mask = torch.arange(
        q_start_index_s, sq + q_start_index_s, dtype=torch.int32, device=ref_device
    ).view(-1, 1) >= torch.arange(
        kv_stride - 1, sk * kv_stride, kv_stride, dtype=torch.int32, device=ref_device
    ).view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(
        3, indices.long(), 1
    )
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    if kv_stride > 1:
        mask[:, :, : kv_stride - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(torch.float16)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pad_to_kernel_sentinel(indices: torch.Tensor, s_k: int) -> torch.Tensor:
    """tilelang_indices.py 用 -1 作 pad；kernel 期望 pad == S_k。"""
    return torch.where(
        indices == TILELANG_PAD_VALUE,
        torch.full_like(indices, s_k),
        indices,
    )


def _build_kernel(
    sfa: Callable,
    heads: int,
    dim: int,
    tail_dim: int,
    topk: int,
    kv_stride: int = 1,
    kv_group: int = 1,
    is_causal: bool = True,
    block_I: int = 64,
):
    return sfa(
        heads=heads,
        dim=dim,
        tail_dim=tail_dim,
        topk=topk,
        kv_stride=kv_stride,
        kv_group=kv_group,
        is_causal=is_causal,
        block_I=block_I,
    )


def _compare(name: str, out_tl: torch.Tensor, out_ref: torch.Tensor, threshold: float = 5e-2) -> bool:
    diff = (out_tl.float() - out_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ok = max_diff < threshold
    print(f"[{name}] max_abs_diff={max_diff:.4e}  mean_abs_diff={mean_diff:.4e}")
    print(f"[{name}] threshold={threshold:.0e}  result={'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Test parameters（小尺寸快速跑，保持 NSA 接口约束）
# ---------------------------------------------------------------------------

B = 1
S_Q = 128        # 官方 example kernel 的 Q 序列长度固定为 128
S_K = 512        # 小尺寸 KV 长度，用于验证 Indices 语义
H = 4            # Q heads
KV_GROUP = 1     # 官方 example 当前 assert kv_group == 1；4 个 Q heads 共享同一 KV/Indices
DIM = 128        # V 的维度 == output 维度
TAIL_DIM = 128   # 已知 tilelang example 在 heads=4 时可编译的组合：Q/KV last dim = 256
BLOCK_I = 64
KV_STRIDE = 1
Q_START = 0
TOPK = 256       # 已知可编译；且给 early tokens 留足 pad sentinel 槽位


def _make_qkv(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    q = torch.randn(B, S_Q, H, DIM + TAIL_DIM, dtype=torch.float16, device=device)
    kv = torch.randn(B, S_K, KV_GROUP, DIM + TAIL_DIM, dtype=torch.float16, device=device)
    return q, kv


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def run_sanity_case(sfa: Callable, device: torch.device) -> bool:
    """Sanity：用 example 风格的随机 indices（不通过我们的转换层），验证 kernel + ref 兼容。"""
    print("\n" + "=" * 60)
    print("[case] SANITY (example-style random indices)")
    print("=" * 60)

    q, kv = _make_qkv(device)
    topk = TOPK  # 必须 % BLOCK_I == 0

    # 与 example 同样的初始化方式：先全部填 S（作 pad sentinel），再随机选 valid
    indices = torch.full((B, S_Q, KV_GROUP, topk), S_K, dtype=torch.int32)
    for b in range(B):
        for t in range(S_Q):
            for g in range(KV_GROUP):
                # 因果：Q at position t 看 K 位置 [0, t]，topk 个候选
                max_valid = max(1, (t + Q_START) // KV_STRIDE)
                k_pos = torch.randperm(max_valid)[:topk]
                indices[b, t, g, : len(k_pos)] = k_pos.to(torch.int32)
    indices = indices.to(device)
    print(f"[sanity] q={tuple(q.shape)} kv={tuple(kv.shape)} indices={tuple(indices.shape)}")

    kernel = _build_kernel(sfa, H, DIM, TAIL_DIM, topk, kv_group=KV_GROUP, block_I=BLOCK_I)
    try:
        out_tl = kernel(q, kv, indices)
    except Exception:
        print("[sanity] kernel 调用失败：")
        traceback.print_exc()
        return False
    print(f"[sanity] kernel out shape={tuple(out_tl.shape)} dtype={out_tl.dtype}")

    out_ref = _ref_sparse_attention_fwd_interface(
        q.cpu(), kv.cpu(), indices.cpu(),
        dim=DIM, q_start_index_s=Q_START, kv_stride=KV_STRIDE,
    ).to(device)

    return _compare("sanity", out_tl, out_ref)


def run_block_sparse_case(sfa: Callable, device: torch.device) -> bool:
    """用 ``block_indices_to_tilelang`` 生成 indices，验证 block-sparse 转换语义。"""
    print("\n" + "=" * 60)
    print("[case] BLOCK-SPARSE (via block_indices_to_tilelang)")
    print("=" * 60)

    q, kv = _make_qkv(device)
    block_size_M = BLOCK_I
    block_size_N = BLOCK_I
    n_q_blocks = S_Q // block_size_M
    max_blocks = TOPK // block_size_N

    # 每个 (b, h, q_block) 选 K block（anchor + trailing window）。
    # 用 block_count 屏蔽重复/越界槽位，避免 kernel 重复计算同一段 K。
    block_indices = torch.zeros(
        B, KV_GROUP, n_q_blocks, max_blocks, dtype=torch.int32
    )
    block_count = torch.full((B, KV_GROUP, n_q_blocks), 2, dtype=torch.int32)
    for b in range(B):
        for h in range(KV_GROUP):
            for q_blk in range(n_q_blocks):
                choices = [0]
                for k_blk in range(max(0, q_blk - max_blocks + 2), q_blk + 1):
                    if k_blk not in choices:
                        choices.append(k_blk)
                block_indices[b, h, q_blk, : len(choices)] = torch.tensor(
                    choices, dtype=torch.int32
                )
                block_count[b, h, q_blk] = len(choices)

    indices = block_indices_to_tilelang(
        block_indices,
        S_q=S_Q,
        block_size_M=block_size_M,
        block_size_N=block_size_N,
        kv_heads=KV_GROUP,
        block_count=block_count,
    )
    indices = _pad_to_kernel_sentinel(indices, s_k=S_K).to(device)
    topk = indices.shape[-1]
    print(f"[bs] indices shape={tuple(indices.shape)} topk={topk} pad→{S_K}")

    kernel = _build_kernel(sfa, H, DIM, TAIL_DIM, topk, kv_group=KV_GROUP, block_I=BLOCK_I)
    try:
        out_tl = kernel(q, kv, indices)
    except Exception:
        print("[bs] kernel 调用失败：")
        traceback.print_exc()
        return False

    out_ref = _ref_sparse_attention_fwd_interface(
        q.cpu(), kv.cpu(), indices.cpu(),
        dim=DIM, q_start_index_s=Q_START, kv_stride=KV_STRIDE,
    ).to(device)

    return _compare("bs", out_tl, out_ref)


def run_stream_llm_case(sfa: Callable, device: torch.device) -> bool:
    """用 ``stream_llm_to_tilelang`` 生成 indices，验证 A-shape 转换语义。"""
    print("\n" + "=" * 60)
    print("[case] STREAM-LLM (via stream_llm_to_tilelang)")
    print("=" * 60)

    q, kv = _make_qkv(device)
    n_init, n_local = 64, TOPK - 64

    indices = stream_llm_to_tilelang(
        B=B, S_q=S_Q, kv_heads=KV_GROUP,
        n_init=n_init, n_local=n_local, block_size_N=BLOCK_I,
        device="cpu",
    )
    indices = _pad_to_kernel_sentinel(indices, s_k=S_K).to(device)
    topk = indices.shape[-1]
    print(f"[sl] indices shape={tuple(indices.shape)} topk={topk} pad→{S_K}")

    kernel = _build_kernel(sfa, H, DIM, TAIL_DIM, topk, kv_group=KV_GROUP, block_I=BLOCK_I)
    try:
        out_tl = kernel(q, kv, indices)
    except Exception:
        print("[sl] kernel 调用失败：")
        traceback.print_exc()
        return False

    out_ref = _ref_sparse_attention_fwd_interface(
        q.cpu(), kv.cpu(), indices.cpu(),
        dim=DIM, q_start_index_s=Q_START, kv_stride=KV_STRIDE,
    ).to(device)

    return _compare("sl", out_tl, out_ref)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="tilelang sparse_attention_fwd 集成精度测试")
    parser.add_argument("--probe", action="store_true", help="仅 probe API 签名后退出")
    parser.add_argument("--case", choices=["sanity", "bs", "sl", "all"], default="all")
    args = parser.parse_args()

    try:
        import torch_npu  # noqa: F401
        device = torch.device("npu:0")
        print(f"[env] torch_npu OK，device={device}")
    except ImportError as e:
        print(f"[env] torch_npu 不可用：{e}")
        return 1

    try:
        sfa, source = _probe_sparse_attention_fwd()
    except ImportError as e:
        print(f"[env] sparse_attention_fwd import 失败：\n{e}")
        return 1
    _print_signature(sfa, source)

    if args.probe:
        print("\n[probe] done.")
        return 0

    results: dict[str, bool] = {}
    if args.case in ("sanity", "all"):
        results["sanity"] = run_sanity_case(sfa, device)
    if args.case in ("bs", "all"):
        results["block-sparse"] = run_block_sparse_case(sfa, device)
    if args.case in ("sl", "all"):
        results["stream-llm"] = run_stream_llm_case(sfa, device)

    print("\n" + "=" * 60)
    print("[summary]")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\n  overall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
