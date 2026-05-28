#!/usr/bin/env python3
"""Create compact Phi-3 PR-4 sparse probe configs.

Only the 43 target heads are kept as sparse.  All other heads are written as
``dense`` so the current smoke/profiling runs measure only the active
``stream_llm`` or ``block_sparse`` sparse code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "minference" / "configs"
STREAM_DENSE_OTHERS_OUTPUT = (
    CONFIG_DIR / "Phi_3_mini_128k_instruct_pathb_stream_llm_aligned_dense_others.json"
)
BLOCK_DENSE_OTHERS_OUTPUT = (
    CONFIG_DIR / "Phi_3_mini_128k_instruct_pathb_block_sparse_probe_dense_others.json"
)
BLOCK_ALL_HEADS_OUTPUT = (
    CONFIG_DIR / "Phi_3_mini_128k_instruct_pathb_block_sparse_all_heads_latency.json"
)
BLOCK_LAYERS8_13_ALL_HEADS_TOPK1_OUTPUT = (
    CONFIG_DIR / "Phi_3_mini_128k_instruct_pathb_block_sparse_layers8_13_all_heads_topk1_latency.json"
)
BLOCK_LAYERS8_13_ALL_HEADS_TOPK2_OUTPUT = (
    CONFIG_DIR / "Phi_3_mini_128k_instruct_pathb_block_sparse_layers8_13_all_heads_topk2_latency.json"
)
BLOCK_LAYERS8_13_ALL_HEADS_TOPK4_OUTPUT = (
    CONFIG_DIR / "Phi_3_mini_128k_instruct_pathb_block_sparse_layers8_13_all_heads_topk4_latency.json"
)
NUM_LAYERS = 32
NUM_HEADS = 32
TARGET_LAYERS = tuple(range(8, 14))

STREAM_TARGET_HEADS = [
    (8, 1, 0.96484375),
    (8, 8, 0.9921875),
    (8, 10, 0.97265625),
    (8, 23, 0.98828125),
    (9, 0, 0.984375),
    (9, 11, 0.94921875),
    (9, 12, 0.59375),
    (9, 15, 0.9765625),
    (9, 17, 0.859375),
    (9, 18, 0.93359375),
    (9, 20, 0.96875),
    (9, 21, 0.9296875),
    (9, 31, 0.91796875),
    (10, 3, 0.984375),
    (10, 5, 0.62109375),
    (10, 7, 0.578125),
    (10, 9, 0.8671875),
    (10, 10, 0.98828125),
    (10, 11, 0.80078125),
    (10, 15, 1.0),
    (10, 16, 0.93359375),
    (10, 18, 0.96875),
    (10, 21, 0.94921875),
    (10, 22, 0.91796875),
    (10, 25, 0.953125),
    (10, 28, 0.98046875),
    (11, 6, 0.8046875),
    (11, 7, 0.9921875),
    (11, 10, 0.65234375),
    (11, 12, 0.69140625),
    (11, 20, 0.85546875),
    (11, 23, 0.99609375),
    (11, 25, 0.93359375),
    (11, 29, 0.59765625),
    (11, 30, 0.98046875),
    (12, 12, 0.82421875),
    (12, 16, 0.83203125),
    (12, 20, 0.84765625),
    (12, 25, 0.984375),
    (12, 26, 0.8125),
    (13, 5, 0.98828125),
    (13, 25, 0.9921875),
    (13, 27, 0.921875),
]


def _build_config(
    *,
    mode: str,
    n_init: int,
    n_local: int,
    topk_blocks: int,
) -> tuple[list[dict[str, list]], int]:
    out = [
        {str(head): ["dense", 0, 0, 1.0] for head in range(NUM_HEADS)}
        for _ in range(NUM_LAYERS)
    ]
    for layer, head, score in STREAM_TARGET_HEADS:
        if mode == "stream_llm":
            out[layer][str(head)] = ["stream_llm", n_init, n_local, score]
        elif mode == "block_sparse":
            out[layer][str(head)] = ["block_sparse", topk_blocks, 0, score]
        else:
            raise AssertionError(mode)
    count = len(STREAM_TARGET_HEADS)
    return out, count


def _build_all_block_sparse_config(*, topk_blocks: int) -> tuple[list[dict[str, list]], int]:
    out = [
        {str(head): ["block_sparse", topk_blocks, 0, 1.0] for head in range(NUM_HEADS)}
        for _ in range(NUM_LAYERS)
    ]
    return out, NUM_LAYERS * NUM_HEADS


def _build_layer_range_block_sparse_config(
    *,
    layers: tuple[int, ...],
    topk_blocks: int,
) -> tuple[list[dict[str, list]], int]:
    out = [
        {
            str(head): (
                ["block_sparse", topk_blocks, 0, 1.0]
                if layer in layers
                else ["dense", 0, 0, 1.0]
            )
            for head in range(NUM_HEADS)
        }
        for layer in range(NUM_LAYERS)
    ]
    return out, len(layers) * NUM_HEADS


def _write_json(path: Path, data: list[dict[str, list]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _parse_layer_range(spec: str) -> tuple[int, int]:
    """Parse ``"8-13"`` into ``(8, 13)`` inclusive. Single int ``"8"`` -> ``(8, 8)``."""
    spec = spec.strip()
    if "-" in spec:
        lo_s, hi_s = spec.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
    else:
        lo = hi = int(spec)
    if not (0 <= lo <= hi < NUM_LAYERS):
        raise ValueError(
            f"layer range {spec!r} out of [0, {NUM_LAYERS - 1}] or lo > hi"
        )
    return lo, hi


def _parse_layer_ranges_csv(raw: str) -> list[tuple[int, int]]:
    if not raw:
        return []
    return [_parse_layer_range(item) for item in raw.split(",") if item.strip()]


def _parse_topks_csv(raw: str) -> list[int]:
    if not raw:
        return []
    topks: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        v = int(item)
        if v < 1:
            raise ValueError(f"topk must be >= 1, got {v}")
        topks.append(v)
    return topks


def _layer_range_config_name(lo: int, hi: int, topk: int) -> str:
    return (
        f"Phi_3_mini_128k_instruct_pathb_block_sparse_"
        f"layers{lo}_{hi}_all_heads_topk{topk}_latency.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-dense-others-output", type=Path, default=STREAM_DENSE_OTHERS_OUTPUT)
    parser.add_argument("--block-dense-others-output", type=Path, default=BLOCK_DENSE_OTHERS_OUTPUT)
    parser.add_argument("--block-all-heads-output", type=Path, default=BLOCK_ALL_HEADS_OUTPUT)
    parser.add_argument("--block-layers8-13-topk1-output", type=Path, default=BLOCK_LAYERS8_13_ALL_HEADS_TOPK1_OUTPUT)
    parser.add_argument("--block-layers8-13-topk2-output", type=Path, default=BLOCK_LAYERS8_13_ALL_HEADS_TOPK2_OUTPUT)
    parser.add_argument("--block-layers8-13-topk4-output", type=Path, default=BLOCK_LAYERS8_13_ALL_HEADS_TOPK4_OUTPUT)
    parser.add_argument(
        "--extra-layer-ranges",
        type=str,
        default="",
        help='额外 block_sparse layer range 集合，逗号分隔。例如 "7-14,8-12,9-12,8-14"。'
             "每个 range 与 --extra-topks 做 cartesian 生成配置。",
    )
    parser.add_argument(
        "--extra-topks",
        type=str,
        default="",
        help='额外 block_sparse topk_blocks 集合，逗号分隔。例如 "1,2,4"。'
             "需配合 --extra-layer-ranges 使用。",
    )
    parser.add_argument(
        "--extra-output-dir",
        type=Path,
        default=CONFIG_DIR,
        help="额外 block_sparse latency configs 的输出目录。",
    )
    parser.add_argument("--n-init", type=int, default=128)
    parser.add_argument("--n-local", type=int, default=896)
    parser.add_argument("--topk-blocks", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_init % 64 != 0 or args.n_local % 64 != 0:
        raise SystemExit("--n-init and --n-local must be multiples of 64")
    if args.topk_blocks < 1:
        raise SystemExit("--topk-blocks must be >= 1")

    stream_dense_others_data, stream_dense_others_count = _build_config(
        mode="stream_llm",
        n_init=args.n_init,
        n_local=args.n_local,
        topk_blocks=args.topk_blocks,
    )
    block_dense_others_data, block_dense_others_count = _build_config(
        mode="block_sparse",
        n_init=args.n_init,
        n_local=args.n_local,
        topk_blocks=args.topk_blocks,
    )
    block_all_heads_data, block_all_heads_count = _build_all_block_sparse_config(
        topk_blocks=args.topk_blocks,
    )
    layer_range_outputs = [
        (args.block_layers8_13_topk1_output, 1),
        (args.block_layers8_13_topk2_output, 2),
        (args.block_layers8_13_topk4_output, 4),
    ]
    _write_json(args.stream_dense_others_output, stream_dense_others_data)
    _write_json(args.block_dense_others_output, block_dense_others_data)
    _write_json(args.block_all_heads_output, block_all_heads_data)
    for output_path, topk in layer_range_outputs:
        layer_range_data, layer_range_count = _build_layer_range_block_sparse_config(
            layers=TARGET_LAYERS,
            topk_blocks=topk,
        )
        _write_json(output_path, layer_range_data)
        print(
            f"[block+layers8-13]  rewrote {layer_range_count} heads "
            f"(topk={topk}) -> {output_path}"
        )

    extra_ranges = _parse_layer_ranges_csv(args.extra_layer_ranges)
    extra_topks = _parse_topks_csv(args.extra_topks)
    if (extra_ranges and not extra_topks) or (extra_topks and not extra_ranges):
        raise SystemExit(
            "--extra-layer-ranges 和 --extra-topks 必须同时给定才能生成额外配置"
        )
    for lo, hi in extra_ranges:
        layer_tuple = tuple(range(lo, hi + 1))
        for topk in extra_topks:
            data, count = _build_layer_range_block_sparse_config(
                layers=layer_tuple,
                topk_blocks=topk,
            )
            out_path = args.extra_output_dir / _layer_range_config_name(lo, hi, topk)
            _write_json(out_path, data)
            print(
                f"[block+layers{lo}-{hi}] rewrote {count} heads "
                f"(topk={topk}) -> {out_path}"
            )

    print(
        f"[stream+dense-others] rewrote {stream_dense_others_count} heads "
        f"-> {args.stream_dense_others_output}"
    )
    print(
        f"[block+dense-others]  rewrote {block_dense_others_count} heads "
        f"-> {args.block_dense_others_output}"
    )
    print(
        f"[block+all-heads]     rewrote {block_all_heads_count} heads "
        f"-> {args.block_all_heads_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
