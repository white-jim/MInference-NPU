# Copyright (c) 2026 (NPU adapter - PR-4 TileLang sparse attention)
# Licensed under The MIT License [see LICENSE for details]
"""Micro-benchmark old padded-H=1 vs experimental query-block H=1 kernels.

Run on an Ascend server with::

    source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh
    PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl \
        python benchmarks/bench_tilelang_h1_sparse_attention.py

This benchmark is intentionally isolated: it does not exercise MInference's
grouped wrapper or HF model path.  It answers one question only: after heads
are folded into batch and the kernel sees H=1, does the query-block H=1 kernel
beat the old padded-head TileLang kernel for the same q/k/v/indices?
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from minference.ops.tilelang_indices import block_indices_to_tilelang
from minference.ops.tilelang_sparse_attention import build_sparse_attention_qkv_fwd
from minference.ops.tilelang_sparse_attention_h1 import (
    build_sparse_attention_h1_block_fwd,
    build_sparse_attention_h1_block_index_fwd,
    build_sparse_attention_mh_block_index_fwd,
)


def _device() -> torch.device:
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(f"torch_npu unavailable: {exc}") from exc
    return torch.device("npu:0")


def _make_block_indices(
    *,
    seq_len: int,
    block_size: int,
    topk_blocks: int,
) -> torch.Tensor:
    n_blocks = seq_len // block_size
    block_indices = torch.empty(1, 1, n_blocks, topk_blocks, dtype=torch.int32)
    for bq in range(n_blocks):
        first = max(0, bq - topk_blocks + 1)
        visible = list(range(first, bq + 1))
        while len(visible) < topk_blocks:
            visible.insert(0, visible[0])
        block_indices[0, 0, bq, :] = torch.tensor(visible[-topk_blocks:], dtype=torch.int32)

    return block_indices


def _make_indices(
    *,
    seq_len: int,
    block_size: int,
    block_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return block_indices_to_tilelang(
        block_indices,
        S_q=seq_len,
        block_size_M=block_size,
        block_size_N=block_size,
        kv_heads=1,
    ).to(device)


def _time_kernel(name: str, fn, *, warmup: int, iters: int) -> tuple[float, torch.Tensor]:
    out = None
    for _ in range(warmup):
        out = fn()
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.npu.synchronize()
    elapsed = (time.perf_counter() - start) / max(1, iters)
    assert out is not None
    print(f"    {name:<18} {elapsed * 1000.0:9.3f} ms/call")
    return elapsed, out


def run_case(
    *,
    seq_len: int,
    dim: int,
    block_size: int,
    topk_blocks: int,
    block_m: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float]:
    if seq_len % block_size != 0:
        raise ValueError(f"seq_len={seq_len} must be divisible by block_size={block_size}")
    if seq_len % block_m != 0:
        raise ValueError(f"seq_len={seq_len} must be divisible by block_m={block_m}")

    torch.manual_seed(123)
    q = torch.randn(1, seq_len, 1, dim, dtype=torch.float16, device=device)
    k = torch.randn(1, seq_len, 1, dim, dtype=torch.float16, device=device)
    v = torch.randn(1, seq_len, 1, dim, dtype=torch.float16, device=device)
    block_indices_cpu = _make_block_indices(
        seq_len=seq_len,
        block_size=block_size,
        topk_blocks=topk_blocks,
    )
    indices = _make_indices(
        seq_len=seq_len,
        block_size=block_size,
        block_indices=block_indices_cpu,
        device=device,
    )
    block_indices = block_indices_cpu.permute(0, 2, 1, 3).contiguous().to(device)
    topk = indices.shape[-1]

    old_kernel = build_sparse_attention_qkv_fwd(
        heads=1,
        dim=dim,
        topk=topk,
        kv_group=1,
        block_I=block_size,
        q_start_index_s=0,
        use_contiguous_range_load=True,
    )
    new_kernel = build_sparse_attention_h1_block_fwd(
        dim=dim,
        topk=topk,
        block_M=block_m,
        block_I=block_size,
        q_start_index_s=0,
    )
    block_index_kernel = build_sparse_attention_h1_block_index_fwd(
        dim=dim,
        topk_blocks=topk_blocks,
        block_M=block_m,
        block_I=block_size,
        q_start_index_s=0,
    )

    print(
        f"\n[case] S={seq_len} D={dim} block={block_size} "
        f"topk_blocks={topk_blocks} topk_tokens={topk} block_M={block_m}"
    )
    old_t, old_out = _time_kernel("old_padded_h1", lambda: old_kernel(q, k, v, indices), warmup=warmup, iters=iters)
    new_t, new_out = _time_kernel("new_query_h1", lambda: new_kernel(q, k, v, indices), warmup=warmup, iters=iters)
    block_t, block_out = _time_kernel(
        "block_index_h1",
        lambda: block_index_kernel(q, k, v, block_indices),
        warmup=warmup,
        iters=iters,
    )

    diff = (old_out.float() - new_out.float()).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    block_diff = (new_out.float() - block_out.float()).abs()
    block_max_diff = float(block_diff.max().item())
    block_mean_diff = float(block_diff.mean().item())
    speedup = old_t / new_t if new_t > 0 else float("inf")
    block_speedup = new_t / block_t if block_t > 0 else float("inf")
    print(f"    speedup            {speedup:9.3f} x")
    print(f"    block_idx_speedup  {block_speedup:9.3f} x")
    print(f"    old_vs_new_diff    max={max_diff:.4e} mean={mean_diff:.4e}")
    print(f"    token_vs_block     max={block_max_diff:.4e} mean={block_mean_diff:.4e}")
    return {
        "old_ms": old_t * 1000.0,
        "new_ms": new_t * 1000.0,
        "block_index_ms": block_t * 1000.0,
        "speedup": speedup,
        "block_index_speedup": block_speedup,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "block_index_max_diff": block_max_diff,
        "block_index_mean_diff": block_mean_diff,
    }


def _make_block_indices_mh(
    *,
    seq_len: int,
    block_size: int,
    topk_blocks: int,
    heads: int,
) -> torch.Tensor:
    """[1, heads, n_q_blocks, topk_blocks] — 每 head 独立选 K block 子集。"""
    n_blocks = seq_len // block_size
    block_indices = torch.empty(1, heads, n_blocks, topk_blocks, dtype=torch.int32)
    for h in range(heads):
        for bq in range(n_blocks):
            first = max(0, bq - topk_blocks + 1)
            visible = list(range(first, bq + 1))
            while len(visible) < topk_blocks:
                visible.insert(0, visible[0])
            block_indices[0, h, bq, :] = torch.tensor(visible[-topk_blocks:], dtype=torch.int32)
    return block_indices


def run_mh_vs_h1_case(
    *,
    seq_len: int,
    heads: int,
    dim: int,
    block_size: int,
    topk_blocks: int,
    block_m: int,
    head_chunk: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float]:
    """MH kernel 一次 launch vs H=1 fold-into-batch（含 head_chunk 多次 launch）的对比。

    Args:
        head_chunk: 模拟 wrapper 侧 ``MINFERENCE_BLOCK_SPARSE_HEAD_CHUNK``。
            ``<= 0`` 表示一次 launch 处理全部 ``heads``；正值则按该 chunk 切分多次 launch。
    """
    if seq_len % block_size != 0:
        raise ValueError(f"seq_len={seq_len} must be divisible by block_size={block_size}")
    if seq_len % block_m != 0:
        raise ValueError(f"seq_len={seq_len} must be divisible by block_m={block_m}")

    torch.manual_seed(123)
    q_bshd = torch.randn(1, seq_len, heads, dim, dtype=torch.float16, device=device)
    k_bshd = torch.randn(1, seq_len, heads, dim, dtype=torch.float16, device=device)
    v_bshd = torch.randn(1, seq_len, heads, dim, dtype=torch.float16, device=device)
    block_indices_bhnt = _make_block_indices_mh(
        seq_len=seq_len,
        block_size=block_size,
        topk_blocks=topk_blocks,
        heads=heads,
    )  # [1, heads, n_q_blocks, topk_blocks]
    n_q_blocks = seq_len // block_size

    # MH kernel: pre-fold [1, S, H, D] → [H, S, 1, D] (same fold-into-batch as H=1 path)
    q_flat = q_bshd.permute(0, 2, 1, 3).contiguous().reshape(heads, 1, seq_len, dim).transpose(1, 2).contiguous()
    k_flat = k_bshd.permute(0, 2, 1, 3).contiguous().reshape(heads, 1, seq_len, dim).transpose(1, 2).contiguous()
    v_flat = v_bshd.permute(0, 2, 1, 3).contiguous().reshape(heads, 1, seq_len, dim).transpose(1, 2).contiguous()
    # [1, H, n_q, topk] → [H, n_q, 1, topk]
    block_indices_flat = block_indices_bhnt.reshape(heads, n_q_blocks, topk_blocks).unsqueeze(2).contiguous().to(device)

    mh_kernel = build_sparse_attention_mh_block_index_fwd(
        dim=dim,
        topk_blocks=topk_blocks,
        block_M=block_m,
        block_I=block_size,
        q_start_index_s=0,
    )

    # H=1 fold-into-batch 模拟：把 (1, S, H, D) reshape 为 (H, S, 1, D)
    # 然后按 head_chunk 切多次 launch（复现 wrapper 行为）
    h1_block_index_kernel = build_sparse_attention_h1_block_index_fwd(
        dim=dim,
        topk_blocks=topk_blocks,
        block_M=block_m,
        block_I=block_size,
        q_start_index_s=0,
    )

    # 折叠版输入: BSHD → [H, S, 1, D]（与 MH kernel 输入完全相同）
    h1_indices_full = block_indices_flat  # [H, n_q_blocks, 1, topk]

    def _h1_one_launch():
        return h1_block_index_kernel(q_flat, k_flat, v_flat, h1_indices_full)

    def _h1_chunked():
        if head_chunk <= 0:
            return _h1_one_launch()
        outs = []
        for start in range(0, heads, head_chunk):
            end = min(start + head_chunk, heads)
            q_chunk = q_flat[start:end].contiguous()
            k_chunk = k_flat[start:end].contiguous()
            v_chunk = v_flat[start:end].contiguous()
            idx_chunk = h1_indices_full[start:end].contiguous()
            outs.append(h1_block_index_kernel(q_chunk, k_chunk, v_chunk, idx_chunk))
        return torch.cat(outs, dim=0)

    print(
        f"\n[mh-case] S={seq_len} H={heads} D={dim} block={block_size} "
        f"topk_blocks={topk_blocks} block_M={block_m} head_chunk={head_chunk}"
    )

    mh_t, mh_out = _time_kernel(
        "mh_one_launch", lambda: mh_kernel(q_flat, k_flat, v_flat, block_indices_flat),
        warmup=warmup, iters=iters,
    )
    h1_t, h1_out = _time_kernel(
        "h1_chunked", _h1_chunked, warmup=warmup, iters=iters,
    )

    # 数值对比：mh_out 和 h1_out 都是 [H, S, 1, D]
    diff = (mh_out.float() - h1_out.float()).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    speedup = h1_t / mh_t if mh_t > 0 else float("inf")
    print(f"    mh_vs_h1_speedup   {speedup:9.3f} x")
    print(f"    mh_vs_h1_diff      max={max_diff:.4e} mean={mean_diff:.4e}")
    return {
        "mh_ms": mh_t * 1000.0,
        "h1_chunked_ms": h1_t * 1000.0,
        "mh_vs_h1_speedup": speedup,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark H=1 TileLang sparse attention kernels")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--topk-blocks", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--mh-heads",
        nargs="+",
        type=int,
        default=[],
        help="若给定，跑 MH vs H=1 fold-into-batch 对比；每个 head 数额外跑一组。",
    )
    parser.add_argument(
        "--mh-head-chunks",
        nargs="+",
        type=int,
        default=[8],
        help="模拟 wrapper 侧 head_chunk；多个值都跑一遍（0 表示一次 launch 全部 head）。",
    )
    args = parser.parse_args()

    device = _device()
    print(f"[env] device={device}")

    results = []
    for seq_len in args.seq_lens:
        for topk_blocks in args.topk_blocks:
            results.append(
                run_case(
                    seq_len=seq_len,
                    dim=args.dim,
                    block_size=args.block_size,
                    topk_blocks=topk_blocks,
                    block_m=args.block_m,
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                )
            )

    positive = [row for row in results if row["speedup"] > 1.0]
    print(f"\n[summary] positive_cases={len(positive)}/{len(results)}")

    if args.mh_heads:
        print("\n" + "=" * 60)
        print("[mh-vs-h1] MH kernel 一次 launch vs H=1 fold-into-batch 多次 launch")
        print("=" * 60)
        mh_results = []
        for seq_len in args.seq_lens:
            for topk_blocks in args.topk_blocks:
                for heads in args.mh_heads:
                    for head_chunk in args.mh_head_chunks:
                        mh_results.append(
                            run_mh_vs_h1_case(
                                seq_len=seq_len,
                                heads=heads,
                                dim=args.dim,
                                block_size=args.block_size,
                                topk_blocks=topk_blocks,
                                block_m=args.block_m,
                                head_chunk=head_chunk,
                                warmup=args.warmup,
                                iters=args.iters,
                                device=device,
                            )
                        )
        mh_positive = [row for row in mh_results if row["mh_vs_h1_speedup"] > 1.0]
        print(f"\n[mh-summary] positive_cases={len(mh_positive)}/{len(mh_results)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
