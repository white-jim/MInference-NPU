#!/usr/bin/env python3
"""Probe block_sparse_attention at fixed sequence length across head counts."""

from __future__ import annotations

import argparse
import time

import torch

from minference.ops.block_sparse_kernel_npu import block_sparse_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topk-blocks", type=int, default=1)
    parser.add_argument("--heads", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    return parser.parse_args()


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
        torch.npu.synchronize()
        t0 = time.perf_counter()
        out = block_sparse_attention(
            q,
            k,
            v,
            topk_blocks=args.topk_blocks,
            block_size=64,
        )
        torch.npu.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"H={heads:<2} ok time_ms={dt_ms:.2f} out={tuple(out.shape)}")
        del q, k, v, out
        torch.npu.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
