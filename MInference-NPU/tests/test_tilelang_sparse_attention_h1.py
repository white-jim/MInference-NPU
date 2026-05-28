# Copyright (c) 2026 (NPU adapter - PR-4 TileLang sparse attention)
# Licensed under The MIT License [see LICENSE for details]
"""Smoke tests for the experimental H=1 query-block sparse attention kernel.

Run on an Ascend server with::

    source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh
    PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl \
        python tests/test_tilelang_sparse_attention_h1.py --case all
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
    "tilelang_indices_h1_standalone",
    "minference/ops/tilelang_indices.py",
)
tilelang_sparse_attention = _load_module(
    "tilelang_sparse_attention_h1_ref_standalone",
    "minference/ops/tilelang_sparse_attention.py",
)
tilelang_sparse_attention_h1 = _load_module(
    "tilelang_sparse_attention_h1_standalone",
    "minference/ops/tilelang_sparse_attention_h1.py",
)

block_indices_to_tilelang = tilelang_indices.block_indices_to_tilelang
sparse_attention_qkv_reference = tilelang_sparse_attention.sparse_attention_qkv_reference
build_sparse_attention_h1_block_fwd = tilelang_sparse_attention_h1.build_sparse_attention_h1_block_fwd
build_sparse_attention_h1_block_index_fwd = (
    tilelang_sparse_attention_h1.build_sparse_attention_h1_block_index_fwd
)
build_sparse_attention_mh_block_index_fwd = (
    tilelang_sparse_attention_h1.build_sparse_attention_mh_block_index_fwd
)

__test__ = False

B = 1
S_Q = 128
S_K = 512
H = 1
KV_GROUP = 1
D = 64
BLOCK = 64
BLOCK_M = 16


def _device() -> torch.device:
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(f"torch_npu unavailable: {exc}") from exc
    return torch.device("npu:0")


def _make_qkv(device: torch.device):
    torch.manual_seed(123)
    q = torch.randn(B, S_Q, H, D, dtype=torch.float16, device=device)
    k = torch.randn(B, S_K, KV_GROUP, D, dtype=torch.float16, device=device)
    v = torch.randn(B, S_K, KV_GROUP, D, dtype=torch.float16, device=device)
    return q, k, v


def _make_block_indices(max_blocks: int) -> torch.Tensor:
    n_q_blocks = S_Q // BLOCK
    block_indices = torch.zeros(B, H, n_q_blocks, max_blocks, dtype=torch.int32)
    if max_blocks == 1:
        block_indices[..., 0] = 0
    elif max_blocks == 2:
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
        f"indices={tuple(indices.shape)} topk={topk} block_M={BLOCK_M}"
    )

    kernel = build_sparse_attention_h1_block_fwd(
        dim=D,
        topk=topk,
        block_M=BLOCK_M,
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


def run_block_index_case(name: str, max_blocks: int, device: torch.device) -> bool:
    print("\n" + "=" * 60)
    print(f"[case] {name}")
    print("=" * 60)

    q, k, v = _make_qkv(device)
    block_indices_cpu = _make_block_indices(max_blocks)
    token_indices = block_indices_to_tilelang(
        block_indices_cpu,
        S_q=S_Q,
        block_size_M=BLOCK,
        block_size_N=BLOCK,
        kv_heads=KV_GROUP,
    ).to(device)
    block_indices = block_indices_cpu.permute(0, 2, 1, 3).contiguous().to(device)
    print(
        f"[{name}] q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} "
        f"block_indices={tuple(block_indices.shape)} topk_blocks={max_blocks} block_M={BLOCK_M}"
    )

    kernel = build_sparse_attention_h1_block_index_fwd(
        dim=D,
        topk_blocks=max_blocks,
        block_M=BLOCK_M,
        block_I=BLOCK,
        q_start_index_s=0,
    )

    try:
        out_tl = kernel(q, k, v, block_indices)
    except Exception:
        print(f"[{name}] kernel call failed:")
        traceback.print_exc()
        return False

    out_ref = sparse_attention_qkv_reference(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        token_indices.cpu(),
        q_start_index_s=0,
    ).to(device)
    return _compare(name, out_tl, out_ref)


def _make_qkv_mh(device: torch.device, h_test: int):
    torch.manual_seed(123 + h_test)
    q = torch.randn(B, S_Q, h_test, D, dtype=torch.float16, device=device)
    k = torch.randn(B, S_K, h_test, D, dtype=torch.float16, device=device)
    v = torch.randn(B, S_K, h_test, D, dtype=torch.float16, device=device)
    return q, k, v


def _make_block_indices_mh(h_test: int, max_blocks: int) -> torch.Tensor:
    """Per-head block indices: each head gets a different causal-visible subset."""
    n_q_blocks = S_Q // BLOCK
    block_indices = torch.zeros(B, h_test, n_q_blocks, max_blocks, dtype=torch.int32)
    if max_blocks == 1:
        for h in range(h_test):
            for qb in range(n_q_blocks):
                block_indices[:, h, qb, 0] = h % (qb + 1)
    elif max_blocks == 2:
        for h in range(h_test):
            for qb in range(n_q_blocks):
                visible_blocks = qb + 1
                block_indices[:, h, qb, 0] = h % visible_blocks
                block_indices[:, h, qb, 1] = (h + 1) % visible_blocks
    else:
        raise ValueError(f"unsupported max_blocks={max_blocks}")
    return block_indices


def run_mh_block_index_case(name: str, h_test: int, max_blocks: int, device: torch.device) -> bool:
    print("\n" + "=" * 60)
    print(f"[case] {name}  H={h_test}  max_blocks={max_blocks}")
    print("=" * 60)

    q, k, v = _make_qkv_mh(device, h_test)  # q: [B, S_Q, h_test, D]
    block_indices_cpu = _make_block_indices_mh(h_test, max_blocks)  # [B, h_test, n_q_blocks, max_blocks]
    token_indices = block_indices_to_tilelang(
        block_indices_cpu,
        S_q=S_Q,
        block_size_M=BLOCK,
        block_size_N=BLOCK,
        kv_heads=h_test,
    ).to(device)

    flat_B = B * h_test
    n_q_blocks = S_Q // BLOCK

    # Fold [B, S, H, D] → [B*H, S, 1, D] so T.copy sees contiguous stride_S = D
    q_flat = q.permute(0, 2, 1, 3).contiguous().reshape(flat_B, 1, S_Q, D).transpose(1, 2).contiguous()
    k_flat = k.permute(0, 2, 1, 3).contiguous().reshape(flat_B, 1, S_K, D).transpose(1, 2).contiguous()
    v_flat = v.permute(0, 2, 1, 3).contiguous().reshape(flat_B, 1, S_K, D).transpose(1, 2).contiguous()

    # [B, h_test, n_q_blocks, max_blocks] → [B*H, n_q_blocks, 1, max_blocks]
    block_indices_flat = (
        block_indices_cpu
        .reshape(flat_B, n_q_blocks, max_blocks)
        .unsqueeze(2)
        .to(device=device, dtype=torch.int32)
    )

    print(
        f"[{name}] q_flat={tuple(q_flat.shape)} k_flat={tuple(k_flat.shape)} "
        f"block_indices_flat={tuple(block_indices_flat.shape)} topk_blocks={max_blocks} block_M={BLOCK_M}"
    )

    kernel = build_sparse_attention_mh_block_index_fwd(
        dim=D,
        topk_blocks=max_blocks,
        block_M=BLOCK_M,
        block_I=BLOCK,
        q_start_index_s=0,
    )

    try:
        out_flat = kernel(q_flat, k_flat, v_flat, block_indices_flat)  # [B*H, S, 1, D]
    except Exception:
        print(f"[{name}] kernel call failed:")
        traceback.print_exc()
        return False

    # Unfold: [B*H, S, 1, D] → [B, h_test, S, D] → [B, S, h_test, D]
    out_tl = out_flat.squeeze(2).reshape(B, h_test, S_Q, D).permute(0, 2, 1, 3).contiguous()

    out_ref = sparse_attention_qkv_reference(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        token_indices.cpu(),
        q_start_index_s=0,
    ).to(device)
    return _compare(name, out_tl, out_ref)


def main() -> int:
    parser = argparse.ArgumentParser(description="TileLang H=1 / MH sparse attention smoke")
    parser.add_argument(
        "--case",
        choices=[
            "one-block",
            "two-block",
            "mh-h2-one-block",
            "mh-h4-two-block",
            "mh-h8-one-block",
            "all",
        ],
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
    if args.case == "all":
        results["block-index-one-block"] = run_block_index_case(
            "block-index-one-block",
            max_blocks=1,
            device=device,
        )
        results["block-index-two-block"] = run_block_index_case(
            "block-index-two-block",
            max_blocks=2,
            device=device,
        )
    if args.case in ("mh-h2-one-block", "all"):
        results["mh-h2-one-block"] = run_mh_block_index_case(
            "mh-h2-one-block", h_test=2, max_blocks=1, device=device,
        )
    if args.case in ("mh-h4-two-block", "all"):
        results["mh-h4-two-block"] = run_mh_block_index_case(
            "mh-h4-two-block", h_test=4, max_blocks=2, device=device,
        )
    if args.case in ("mh-h8-one-block", "all"):
        results["mh-h8-one-block"] = run_mh_block_index_case(
            "mh-h8-one-block", h_test=8, max_blocks=1, device=device,
        )

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
