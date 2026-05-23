# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import os
import time

from .merge_utils import STORE_DIR, copy_configs_to_merged_dir, merge_ckpts

STABILITY_DELAY = 5  # seconds
POLL_INTERVAL = 10  # seconds


def print_args(args):
    print("-" * 60)
    print(f"Arguments:\n{args.__dict__}")
    print("-" * 60)


def is_checkpoint_complete(checkpoint_dir, expected_shards):
    """
    Check if all expected checkpoint shards are present in the directory and are stable.
    """
    # List current files in the directory
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    if len(files) < expected_shards:
        # if there are fewer files than expected, the checkpoint is not complete
        return False

    # Optionally: check for stability by comparing file sizes over a delay period
    sizes_initial = {f: os.path.getsize(os.path.join(checkpoint_dir, f)) for f in files}
    time.sleep(STABILITY_DELAY)
    sizes_later = {f: os.path.getsize(os.path.join(checkpoint_dir, f)) for f in files}

    # Ensure each file's size has not changed
    return all(sizes_initial[f] == sizes_later[f] for f in files)


def monitor_and_merge(args, expr_data_dir: str):
    """
    Monitor the base checkpoint directory and trigger merging for completed iterations.
    """
    base_ckpt_dir = os.path.join(expr_data_dir, "checkpoints")

    merged_iterations = set()
    err_iterations = {}
    while True:
        # List iteration directories (assuming format 'epoch_idx-iter_idx')
        if not os.path.isdir(base_ckpt_dir):
            continue

        for iteration in os.listdir(base_ckpt_dir):
            # example `iteration`: 0000-0005
            if not iteration.count("-") == 1:
                continue

            if err_iterations.get(iteration, 0) > 3:
                continue

            try:
                epoch_idx, iter_idx = iteration.split("-")
                epoch_idx, iter_idx = int(epoch_idx), int(iter_idx)

                if epoch_idx < args.start_epoch or iter_idx < args.start_iter:
                    continue

                iter_dir = os.path.join(base_ckpt_dir, iteration)
                if not os.path.isdir(iter_dir) or iteration in merged_iterations:
                    continue

                # Check if the checkpoint is complete in this shard directory
                if is_checkpoint_complete(iter_dir, args.num_gpus):
                    # Merge the checkpoints (you may want to merge across all ranks,
                    # or handle each rank separately depending on your setup)

                    print(f"-" * 60)
                    print(f"Merging checkpoint for iteration {iteration}...")
                    merged = merge_ckpts(
                        expr_data_dir,
                        epoch_idx,
                        iter_idx,
                        args.override,
                    )
                    if not merged:
                        print(f"Error merging checkpoints for {iteration}. Continue.")
                        continue

                    # Copy config files to the merged directory
                    copied = copy_configs_to_merged_dir(
                        expr_data_dir,
                        epoch_idx,
                        iter_idx,
                        args.model_id,
                        args.override,
                    )
                    if not copied:
                        print(f"Error copying config fiiles for {iteration}. Continue.")
                        continue

                    # Mark this iteration as merged to avoid reprocessing.
                    merged_iterations.add(iteration)
            except Exception as e:
                print(f"Error processing iteration {iteration}: {e}")
                err_iterations[iteration] = err_iterations.get(iteration, 0) + 1

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expr_data_dir",
        type=str,
        default=None,
        help="Path to the experiment data directory",
    )

    parser.add_argument("--gpu_set", type=str, default=None)
    parser.add_argument("--expr_dir", type=str, default=None)
    parser.add_argument("--expr_name", type=str, help="name of the experiment")

    parser.add_argument("--num_gpus", type=int, help="number of gpus")
    parser.add_argument(
        "--model_id", type=str, default="Qwen/Qwen2.5-3B", help="transformers model id"
    )
    parser.add_argument(
        "--use_ring_attn", action="store_true", help="use ring attention"
    )

    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--start_epoch", type=int, default=0)

    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    if args.expr_data_dir is None and (
        args.gpu_set is None or args.expr_dir is None or args.expr_name is None
    ):
        raise ValueError(
            "Either use expr_data_dir or (gpu_set, expr_dir, and expr_name) to present the experiment data dir"
        )
    print_args(args)

    expr_data_dir = (
        args.expr_data_dir
        if args.expr_data_dir
        else os.path.join(STORE_DIR, args.gpu_set, args.expr_dir, args.expr_name)
    )
    monitor_and_merge(args, expr_data_dir)
