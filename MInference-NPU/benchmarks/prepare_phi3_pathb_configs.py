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
NUM_LAYERS = 32
NUM_HEADS = 32

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


def _write_json(path: Path, data: list[dict[str, list]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-dense-others-output", type=Path, default=STREAM_DENSE_OTHERS_OUTPUT)
    parser.add_argument("--block-dense-others-output", type=Path, default=BLOCK_DENSE_OTHERS_OUTPUT)
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
    _write_json(args.stream_dense_others_output, stream_dense_others_data)
    _write_json(args.block_dense_others_output, block_dense_others_data)

    print(
        f"[stream+dense-others] rewrote {stream_dense_others_count} heads "
        f"-> {args.stream_dense_others_output}"
    )
    print(
        f"[block+dense-others]  rewrote {block_dense_others_count} heads "
        f"-> {args.block_dense_others_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
