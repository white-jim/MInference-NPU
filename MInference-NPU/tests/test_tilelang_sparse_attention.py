# Copyright (c) 2026 (NPU adapter - PR-4 TileLang sparse attention)
# Licensed under The MIT License [see LICENSE for details]
"""Precision smoke tests for the PR-4 separate-Q/K/V TileLang sparse kernel.

Run on an Ascend server with::

    source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh
    PYTHONPATH=~/tilelang-ascend conda run -n flexhead-tl \
        python tests/test_tilelang_sparse_attention.py --case all
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import os
import sys
import traceback

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_module(module_name: str, relative_path: str):
    path = os.path.join(_REPO_ROOT, *relative_path.split("/"))
    spec = _ilu.spec_from_file_location(module_name, path)
    module = _ilu.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tilelang_indices = _load_module(
    "tilelang_indices_standalone",
    "minference/ops/tilelang_indices.py",
)
tilelang_sparse_attention = _load_module(
    "tilelang_sparse_attention_standalone",
    "minference/ops/tilelang_sparse_attention.py",
)

block_indices_to_tilelang = tilelang_indices.block_indices_to_tilelang
stream_llm_to_tilelang = tilelang_indices.stream_llm_to_tilelang
TILELANG_PAD_VALUE = tilelang_indices.TILELANG_PAD_VALUE
build_sparse_attention_qkv_fwd = tilelang_sparse_attention.build_sparse_attention_qkv_fwd
sparse_attention_qkv_reference = tilelang_sparse_attention.sparse_attention_qkv_reference

__test__ = False

B = 1
S_Q = 128
S_K = 512
S_K_STREAM = 512
H = 16
KV_GROUP = 1
D = 64
BLOCK = 64


def _device() -> torch.device:
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(f"torch_npu unavailable: {exc}") from exc
    return torch.device("npu:0")


def _make_qkv(device: torch.device, s_q: int = S_Q, s_k: int = S_K):
    torch.manual_seed(123)
    q = torch.randn(B, s_q, H, D, dtype=torch.float16, device=device)
    k = torch.randn(B, s_k, KV_GROUP, D, dtype=torch.float16, device=device)
    v = torch.randn(B, s_k, KV_GROUP, D, dtype=torch.float16, device=device)
    return q, k, v


def _make_block_indices(max_blocks: int) -> torch.Tensor:
    n_q_blocks = S_Q // BLOCK
    block_indices = torch.zeros(B, KV_GROUP, n_q_blocks, max_blocks, dtype=torch.int32)
    if max_blocks == 1:
        block_indices[..., 0] = 0
    elif max_blocks == 2:
        # Include the future block for the first Q block.  The TileLang kernel
        # must remove future tokens with its causal mask.
        block_indices[:, :, :, 0] = 0
        block_indices[:, :, :, 1] = 1
    else:
        raise ValueError(f"unsupported max_blocks={max_blocks}")
    return block_indices


def _compare(name: str, out_tl: torch.Tensor, out_ref: torch.Tensor, threshold: float = 6e-2) -> bool:
    if torch.isnan(out_tl).any() or torch.isnan(out_ref).any():
        print(f"[{name}] NaN detected")
        return False
    diff = (out_tl.float() - out_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ok = max_diff < threshold
    print(f"[{name}] max_abs_diff={max_diff:.4e} mean_abs_diff={mean_diff:.4e}")
    print(f"[{name}] threshold={threshold:.0e} result={'PASS' if ok else 'FAIL'}")
    return ok


def run_case(name: str, max_blocks: int, device: torch.device) -> bool:
    print("\n" + "=" * 60)
    print(f"[case] {name}")
    print("=" * 60)

    q, k, v = _make_qkv(device)
    indices = block_indices_to_tilelang(
        _make_block_indices(max_blocks),
        S_q=S_Q,
        block_size_M=BLOCK,
        block_size_N=BLOCK,
        kv_heads=KV_GROUP,
    ).to(device)
    topk = indices.shape[-1]
    print(
        f"[{name}] q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} "
        f"indices={tuple(indices.shape)} topk={topk}"
    )

    kernel = build_sparse_attention_qkv_fwd(
        heads=H,
        dim=D,
        topk=topk,
        kv_group=KV_GROUP,
        block_I=BLOCK,
        q_start_index_s=0,
    )

    try:
        out_tl = kernel(q, k, v, indices)
    except Exception:
        print(f"[{name}] kernel call failed:")
        traceback.print_exc()
        return False

    out_ref = sparse_attention_qkv_reference(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        indices.cpu(),
        q_start_index_s=0,
    ).to(device)
    return _compare(name, out_tl, out_ref)


def run_stream_llm_case(device: torch.device) -> bool:
    name = "stream-llm"
    print("\n" + "=" * 60)
    print(f"[case] {name}")
    print("=" * 60)

    q_start = S_K_STREAM - S_Q
    q, k, v = _make_qkv(device, s_q=S_Q, s_k=S_K_STREAM)
    indices = stream_llm_to_tilelang(
        B=B,
        S_q=S_Q,
        kv_heads=KV_GROUP,
        n_init=64,
        n_local=64,
        block_size_N=BLOCK,
        q_start_index_s=q_start,
        device="cpu",
    ).to(device)
    topk = indices.shape[-1]
    print(
        f"[{name}] q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} "
        f"indices={tuple(indices.shape)} topk={topk} q_start={q_start}"
    )

    kernel = build_sparse_attention_qkv_fwd(
        heads=H,
        dim=D,
        topk=topk,
        kv_group=KV_GROUP,
        block_I=BLOCK,
        q_start_index_s=q_start,
    )

    try:
        out_tl = kernel(q, k, v, indices)
    except Exception:
        print(f"[{name}] kernel call failed:")
        traceback.print_exc()
        return False

    out_ref = sparse_attention_qkv_reference(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        indices.cpu(),
        q_start_index_s=q_start,
    ).to(device)
    return _compare(name, out_tl, out_ref)


def run_stream_llm_pad_case(device: torch.device) -> bool:
    name = "stream-llm-pad"
    print("\n" + "=" * 60)
    print(f"[case] {name}")
    print("=" * 60)

    q_start = 0
    q, k, v = _make_qkv(device, s_q=S_Q, s_k=S_Q)
    indices = stream_llm_to_tilelang(
        B=B,
        S_q=S_Q,
        kv_heads=KV_GROUP,
        n_init=64,
        n_local=64,
        block_size_N=BLOCK,
        q_start_index_s=q_start,
        device="cpu",
    ).to(device)
    topk = indices.shape[-1]
    pad_count = int((indices.cpu() == TILELANG_PAD_VALUE).sum().item())
    print(
        f"[{name}] q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} "
        f"indices={tuple(indices.shape)} topk={topk} q_start={q_start} pad_count={pad_count}"
    )

    kernel = build_sparse_attention_qkv_fwd(
        heads=H,
        dim=D,
        topk=topk,
        kv_group=KV_GROUP,
        block_I=BLOCK,
        q_start_index_s=q_start,
    )

    try:
        out_tl = kernel(q, k, v, indices)
    except Exception:
        print(f"[{name}] kernel call failed:")
        traceback.print_exc()
        return False

    out_ref = sparse_attention_qkv_reference(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        indices.cpu(),
        q_start_index_s=q_start,
    ).to(device)
    return _compare(name, out_tl, out_ref)


def main() -> int:
    parser = argparse.ArgumentParser(description="TileLang separate-Q/K/V sparse attention smoke")
    parser.add_argument(
        "--case",
        choices=["one-block", "two-block", "stream-llm", "stream-llm-pad", "all"],
        default="all",
    )
    args = parser.parse_args()

    try:
        device = _device()
    except RuntimeError as exc:
        print(f"[env] {exc}")
        return 1
    print(f"[env] device={device}")

    results = {}
    if args.case in ("one-block", "all"):
        results["one-block"] = run_case("one-block", max_blocks=1, device=device)
    if args.case in ("two-block", "all"):
        results["two-block"] = run_case("two-block", max_blocks=2, device=device)
    if args.case in ("stream-llm", "all"):
        results["stream-llm"] = run_stream_llm_case(device)
    if args.case in ("stream-llm-pad", "all"):
        results["stream-llm-pad"] = run_stream_llm_pad_case(device)

    print("\n" + "=" * 60)
    print("[summary]")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    overall = all(results.values())
    print(f"\n  overall: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
