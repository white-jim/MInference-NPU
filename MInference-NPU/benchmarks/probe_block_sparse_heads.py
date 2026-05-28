#!/usr/bin/env python3
"""Probe block_sparse_attention at fixed sequence length across head counts."""

from __future__ import annotations

import argparse
import time

import torch

from minference.ops.block_sparse_kernel_npu import (
    _block_sparse_tilelang_npu_with_indices,
    _select_block_sparse_topk_indices,
    block_sparse_attention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topk-blocks", type=int, default=1)
    parser.add_argument("--heads", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--profile-components",
        action="store_true",
        help="Also time selector-only and kernel-with-precomputed-indices components.",
    )
    return parser.parse_args()


def _time_call(fn, warmup: int, repeats: int) -> tuple[object, list[float]]:
    out = None
    for _ in range(warmup):
        out = fn()
        torch.npu.synchronize()
    times = []
    for _ in range(repeats):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return out, times


def _fmt_times(times: list[float]) -> str:
    return (
        f"mean_ms={sum(times) / len(times):.2f} "
        f"min_ms={min(times):.2f} max_ms={max(times):.2f}"
    )


def main() -> int:
    args = parse_args()
    device = torch.device("npu:0")
    for heads in args.heads:
        torch.manual_seed(1000 + heads)
        q = torch.randn(
            1, heads, args.seq_len, args.head_dim, device=device, dtype=torch.float16
        )
        k = torch.randn(
            1, heads, args.seq_len, args.head_dim, device=device, dtype=torch.float16
        )
        v = torch.randn(
            1, heads, args.seq_len, args.head_dim, device=device, dtype=torch.float16
        )
        out, times = _time_call(
            lambda: block_sparse_attention(
                q,
                k,
                v,
                topk_blocks=args.topk_blocks,
                block_size=64,
            ),
            args.warmup,
            args.repeats,
        )
        print(f"H={heads:<2} full ok {_fmt_times(times)} out={tuple(out.shape)}")

        if args.profile_components:
            block_indices, selector_times = _time_call(
                lambda: _select_block_sparse_topk_indices(
                    q,
                    k,
                    topk_blocks=args.topk_blocks,
                    block_size=64,
                ),
                args.warmup,
                args.repeats,
            )
            block_indices = block_indices.detach()
            out_kernel, kernel_times = _time_call(
                lambda: _block_sparse_tilelang_npu_with_indices(
                    q,
                    k,
                    v,
                    block_indices,
                    block_size=64,
                ),
                args.warmup,
                args.repeats,
            )
            print(
                f"H={heads:<2} components selector {_fmt_times(selector_times)} "
                f"kernel {_fmt_times(kernel_times)} out={tuple(out_kernel.shape)}"
            )
        del q, k, v, out
        if args.profile_components:
            del block_indices, out_kernel
        torch.npu.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
