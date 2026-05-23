# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Standalone distributed correctness checks for XAttention raw kernels."""
from __future__ import annotations

import random
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from minference.dist_ops.test.raw_test_utils import (
    SEED_BASE,
    check_forward_and_qkv_grads,
    create_full_inputs,
    gather_sequence_shards,
    init_process_group,
    slice_local_inputs,
)
from minference.dist_ops.xattn_zigzag import xattn_zigzag_func
from minference.ops.utils import set_seed
from minference.ops.xattention_fa import xattn_flash_attn_func

# ------------- constants ------------------------------------------------------
_ATOL = 1e-1
_RTOL = 1e-1
_WORLD_SIZE = 4


def _run_worker(
    rank: int,
    world_size: int,
    port: str,
    cfg: SimpleNamespace,
) -> None:
    """Worker function executed in every spawned GPU process."""
    init_process_group(rank, world_size, port)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    set_seed(SEED_BASE + rank)
    q, k, v, dout = create_full_inputs(rank, cfg, device, dtype)
    q_local, k_local, v_local, dout_local = slice_local_inputs(
        rank, world_size, q, k, v, dout
    )

    # ----------------- forward / backward on the candidate kernel ------------
    out_local = xattn_zigzag_func(
        q_local,
        k_local,
        v_local,
        layer_idx=0,
        xattn_params=cfg.xattn_params,
        granularity=128,
    )
    torch.autograd.backward(out_local, dout_local)

    final_out = gather_sequence_shards(out_local, world_size)
    grads = tuple(
        gather_sequence_shards(grad, world_size)
        for grad in (q_local.grad, k_local.grad, v_local.grad)
    )
    dist.barrier()
    torch.cuda.synchronize()

    if rank == 0:
        q_ref = q.detach().clone().requires_grad_()
        k_ref = k.detach().clone().requires_grad_()
        v_ref = v.detach().clone().requires_grad_()

        single_machine_params = cfg.xattn_params.copy()
        single_machine_params["chunk_size"] = cfg.seq_len // _WORLD_SIZE
        out_ref = xattn_flash_attn_func(
            q_ref,
            k_ref,
            v_ref,
            head_indices=list(range(cfg.num_qo_heads)),
            xattn_params=single_machine_params,
            granularity=128,
        )
        torch.autograd.backward(out_ref, dout)
        ref_grads = (q_ref.grad, k_ref.grad, v_ref.grad)

        check_forward_and_qkv_grads(
            cfg.seq_len,
            final_out,
            out_ref,
            grads,
            ref_grads,
            atol=_ATOL,
            rtol=_RTOL,
        )

    dist.destroy_process_group()


def run_xattention_kernel_test(
    seq_len: int = 4096,
    batch_sz: int = 1,
    head_dim: int = 64,
    ones: bool = True,
    num_qo_heads: int = 2,
    num_kv_heads: int = 2,
    stride: int = 16,
    threshold: float = 0.9,
):
    """Compare distributed XAttention-Zigzag outputs with dense reference."""
    port = str(random.randint(12000, 20000))
    xattn_params = {
        "stride": stride,
        "norm": 1,
        "softmax": True,
        "threshold": threshold,
        "select_mode": "inverse",
        "use_triton": True,
        "causal": True,
        "kdb": 1,
        "keep_sink": False,
        "keep_recent": False,
    }
    cfg = SimpleNamespace(
        batch_size=batch_sz,
        seq_len=seq_len,
        head_dim=head_dim,
        ones=ones,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        xattn_params=xattn_params,
    )

    print("=" * 80)
    print(f"Testing XAttention (w. Zigzag) with configuration:\n{cfg}")
    print("=" * 80)
    mp.spawn(
        _run_worker,
        args=(_WORLD_SIZE, port, cfg),
        nprocs=_WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    run_xattention_kernel_test(
        seq_len=512 * 1024,
        batch_sz=1,
        head_dim=64,
        ones=False,
        num_qo_heads=4,
        num_kv_heads=1,
        stride=16,
        threshold=0.95,
    )
