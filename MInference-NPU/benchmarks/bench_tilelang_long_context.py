#!/usr/bin/env python3
"""Long-context speed/memory benchmark for PR-4 TileLang path-B.

This script intentionally loads the target op files directly instead of
importing the top-level ``minference`` package.  The ``flexhead-tl`` server
environment may not contain the full transformers stack.

Recommended server command:

    source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh
    PYTHONPATH=~/tilelang-ascend conda run -n flexhead-tl python \
      benchmarks/bench_tilelang_long_context.py --seq-lens 131072 262144
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BenchResult:
    mode: str
    seq_len: int
    shape: str
    sparse_topk_tokens: int
    first_call_ms: float | None
    mean_ms: float
    tokens_per_s: float
    path_b_hits_first_call: int
    path_b_hits_timed: int
    peak_allocated_mb: float | None
    peak_reserved_mb: float | None
    end_allocated_mb: float | None
    end_reserved_mb: float | None


class HitCounter:
    def __init__(self, module, attr: str):
        self.module = module
        self.attr = attr
        self.count = 0
        self.original = getattr(module, attr)

    def install(self) -> None:
        def wrapped(*args, **kwargs):
            self.count += 1
            return self.original(*args, **kwargs)

        setattr(self.module, self.attr, wrapped)

    def uninstall(self) -> None:
        setattr(self.module, self.attr, self.original)


def _ensure_minference_package_shim() -> None:
    pkg = sys.modules.get("minference")
    if pkg is None:
        pkg = types.ModuleType("minference")
        pkg.__path__ = [str(REPO_ROOT / "minference")]
        sys.modules["minference"] = pkg

    ops = sys.modules.get("minference.ops")
    if ops is None:
        ops = types.ModuleType("minference.ops")
        ops.__path__ = [str(REPO_ROOT / "minference" / "ops")]
        sys.modules["minference.ops"] = ops


def _load_op_module(name: str):
    _ensure_minference_package_shim()
    path = REPO_ROOT / "minference" / "ops" / f"{name}.py"
    module_name = f"minference.ops.{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _require_npu() -> torch.device:
    try:
        import torch_npu  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "torch_npu is unavailable. Did you source CANN and use conda env flexhead-tl?"
        ) from exc
    device = torch.device("npu:0")
    try:
        probe = torch.empty(1, device=device)
        _sync()
        del probe
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("failed to allocate a probe tensor on npu:0") from exc
    return device


def _sync() -> None:
    torch.npu.synchronize()


def _empty_cache() -> None:
    empty = getattr(torch.npu, "empty_cache", None)
    if empty is not None:
        empty()


def _reset_peak() -> None:
    reset = getattr(torch.npu, "reset_peak_memory_stats", None)
    if reset is not None:
        try:
            reset()
        except Exception:
            pass


def _memory_mb(name: str) -> float | None:
    fn = getattr(torch.npu, name, None)
    if fn is None:
        return None
    try:
        return float(fn()) / 1024.0 / 1024.0
    except Exception:
        return None


def _make_qkv(seq_len: int, head_dim: int, device: torch.device, seed: int):
    torch.manual_seed(seed)
    q = torch.randn(1, 1, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(1, 1, seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(1, 1, seq_len, head_dim, device=device, dtype=torch.float16)
    return q, k, v


def _run_timed(
    fn: Callable[[], torch.Tensor],
    *,
    seq_len: int,
    warmup: int,
    iters: int,
    counter: HitCounter,
    include_first_call: bool,
) -> tuple[float | None, float, int, int, float | None, float | None, float | None, float | None]:
    first_call_ms = None
    first_start_count = counter.count
    if include_first_call:
        _reset_peak()
        t0 = time.perf_counter()
        out = fn()
        _sync()
        first_call_ms = (time.perf_counter() - t0) * 1000.0
        del out
        gc.collect()
    first_hits = counter.count - first_start_count

    for _ in range(warmup):
        out = fn()
    _sync()
    if warmup > 0:
        del out
    gc.collect()
    _empty_cache()

    _reset_peak()
    timed_start_count = counter.count
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    _sync()
    elapsed = time.perf_counter() - t0
    timed_hits = counter.count - timed_start_count

    # Keep output alive until after sync and memory sampling.
    mean_ms = elapsed / float(iters) * 1000.0
    peak_alloc = _memory_mb("max_memory_allocated")
    peak_reserved = _memory_mb("max_memory_reserved")
    end_alloc = _memory_mb("memory_allocated")
    end_reserved = _memory_mb("memory_reserved")
    del out
    gc.collect()
    return (
        first_call_ms,
        mean_ms,
        first_hits,
        timed_hits,
        peak_alloc,
        peak_reserved,
        end_alloc,
        end_reserved,
    )


def bench_block_sparse(args, module, device: torch.device, seq_len: int) -> BenchResult:
    q, k, v = _make_qkv(seq_len, args.head_dim, device, args.seed + seq_len)
    counter = HitCounter(module, "_block_sparse_tilelang_npu")
    counter.install()
    try:
        def call():
            return module.block_sparse_attention(
                q,
                k,
                v,
                topk_blocks=args.topk_blocks,
                block_size=args.block_size,
            )

        values = _run_timed(
            call,
            seq_len=seq_len,
            warmup=args.warmup,
            iters=args.iters,
            counter=counter,
            include_first_call=not args.skip_first_call,
        )
    finally:
        counter.uninstall()

    first_ms, mean_ms, first_hits, timed_hits, peak_alloc, peak_reserved, end_alloc, end_reserved = values
    return BenchResult(
        mode="block_sparse",
        seq_len=seq_len,
        shape=str(tuple(q.shape)),
        sparse_topk_tokens=args.topk_blocks * args.block_size,
        first_call_ms=first_ms,
        mean_ms=mean_ms,
        tokens_per_s=seq_len / (mean_ms / 1000.0),
        path_b_hits_first_call=first_hits,
        path_b_hits_timed=timed_hits,
        peak_allocated_mb=peak_alloc,
        peak_reserved_mb=peak_reserved,
        end_allocated_mb=end_alloc,
        end_reserved_mb=end_reserved,
    )


def bench_stream_llm(args, module, device: torch.device, seq_len: int) -> BenchResult:
    q, k, v = _make_qkv(seq_len, args.head_dim, device, args.seed + 17 + seq_len)
    counter = HitCounter(module, "_streaming_tilelang_npu")
    counter.install()
    try:
        def call():
            return module.streaming_forward(
                q,
                k,
                v,
                n_init=args.n_init,
                n_local=args.n_local,
            )

        values = _run_timed(
            call,
            seq_len=seq_len,
            warmup=args.warmup,
            iters=args.iters,
            counter=counter,
            include_first_call=not args.skip_first_call,
        )
    finally:
        counter.uninstall()

    first_ms, mean_ms, first_hits, timed_hits, peak_alloc, peak_reserved, end_alloc, end_reserved = values
    return BenchResult(
        mode="stream_llm",
        seq_len=seq_len,
        shape=str(tuple(q.shape)),
        sparse_topk_tokens=args.n_init + args.n_local,
        first_call_ms=first_ms,
        mean_ms=mean_ms,
        tokens_per_s=seq_len / (mean_ms / 1000.0),
        path_b_hits_first_call=first_hits,
        path_b_hits_timed=timed_hits,
        peak_allocated_mb=peak_alloc,
        peak_reserved_mb=peak_reserved,
        end_allocated_mb=end_alloc,
        end_reserved_mb=end_reserved,
    )


def _print_result(result: BenchResult) -> None:
    first = "n/a" if result.first_call_ms is None else f"{result.first_call_ms:.2f}"
    peak = "n/a" if result.peak_allocated_mb is None else f"{result.peak_allocated_mb:.1f}"
    reserved = "n/a" if result.peak_reserved_mb is None else f"{result.peak_reserved_mb:.1f}"
    print(
        f"{result.mode:<13} S={result.seq_len:<7} topk={result.sparse_topk_tokens:<5} "
        f"first_ms={first:<10} mean_ms={result.mean_ms:<10.2f} "
        f"tok/s={result.tokens_per_s:<10.1f} pathB(first/timed)="
        f"{result.path_b_hits_first_call}/{result.path_b_hits_timed} "
        f"peak_alloc_mb={peak:<9} peak_reserved_mb={reserved}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[131072, 262144])
    parser.add_argument("--modes", nargs="+", choices=["block_sparse", "stream_llm"], default=["block_sparse", "stream_llm"])
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--topk-blocks", type=int, default=16, help="block_sparse top-k blocks; topk tokens = topk_blocks * block_size")
    parser.add_argument("--n-init", type=int, default=128)
    parser.add_argument("--n-local", type=int, default=896)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-first-call", action="store_true", help="skip the separate first-call/JIT timing")
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_init % args.block_size != 0 or args.n_local % args.block_size != 0:
        raise SystemExit("--n-init and --n-local must be multiples of --block-size")
    if args.topk_blocks < 1:
        raise SystemExit("--topk-blocks must be >= 1")

    os.chdir(REPO_ROOT)
    device = _require_npu()
    print(f"[env] repo={REPO_ROOT}")
    print(f"[env] device={device} torch={torch.__version__}")
    print(
        f"[config] seq_lens={args.seq_lens} modes={args.modes} "
        f"D={args.head_dim} block={args.block_size} block_topk_tokens="
        f"{args.topk_blocks * args.block_size} stream_topk={args.n_init + args.n_local} "
        f"warmup={args.warmup} iters={args.iters}"
    )

    block_module = _load_op_module("block_sparse_kernel_npu") if "block_sparse" in args.modes else None
    stream_module = _load_op_module("streaming_kernel_npu") if "stream_llm" in args.modes else None

    results: list[BenchResult] = []
    for seq_len in args.seq_lens:
        for mode in args.modes:
            _empty_cache()
            gc.collect()
            if mode == "block_sparse":
                result = bench_block_sparse(args, block_module, device, seq_len)
            elif mode == "stream_llm":
                result = bench_stream_llm(args, stream_module, device, seq_len)
            else:
                raise AssertionError(mode)
            _print_result(result)
            if result.path_b_hits_timed != args.iters:
                print(
                    f"[warn] {mode} S={seq_len}: path-B hit count during timed loop "
                    f"is {result.path_b_hits_timed}, expected {args.iters}. "
                    "This run may include a fallback."
                )
            results.append(result)

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n")
        print(f"[json] wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
