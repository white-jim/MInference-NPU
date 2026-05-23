# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Standalone distributed correctness checks for Minference raw kernels."""
from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from minference.dist_ops.minfer_dr_striped import minfer_dr_stripe_func
from minference.dist_ops.minfer_striped import minfer_stripe_func
from minference.dist_ops.minfer_zigzag import minfer_zigzag_func
from minference.dist_ops.test.raw_test_utils import (
    SEED_BASE,
    check_forward_and_qkv_grads,
    create_full_inputs,
    gather_sequence_shards,
    init_process_group,
    slice_local_inputs,
)
from minference.ops.pit_sparse_flash_attention_v3 import minference_flash_attn_func
from minference.ops.utils import set_seed

# ------------- constants ------------------------------------------------------
_ATOL = 1e-2
_RTOL = 1e-2
_WORLD_SIZE = 4

_ATTENTION_IMPLS: dict[str, Callable] = {
    "minfer_zigzag": minfer_zigzag_func,
    "minfer_stripe": minfer_stripe_func,
    "minfer_dr_stripe": minfer_dr_stripe_func,
}


def _run_worker(
    rank: int,
    world_size: int,
    port: str,
    cfg: SimpleNamespace,
    attn_op_name: str,
) -> None:
    """Worker function executed in every spawned GPU process."""
    init_process_group(rank, world_size, port)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    set_seed(SEED_BASE + rank)

    attn_op: Callable = _ATTENTION_IMPLS[attn_op_name]

    q, k, v, dout = create_full_inputs(rank, cfg, device, dtype)
    q_local, k_local, v_local, dout_local = slice_local_inputs(
        rank, world_size, q, k, v, dout
    )

    # ----------------- forward / backward on the candidate kernel ------------
    out_local = attn_op(
        q_local,
        k_local,
        v_local,
        cfg.v_size,
        cfg.s_size,
        layer_idx=0,
    )
    torch.autograd.backward(out_local, dout_local)

    final_out = gather_sequence_shards(out_local, world_size)
    grads = tuple(
        gather_sequence_shards(grad, world_size)
        for grad in (q_local.grad, k_local.grad, v_local.grad)
    )
    torch.distributed.barrier()
    torch.cuda.synchronize()

    # ----------------- reference: single machine MInference Forward/Backward  ----------------------
    if rank == 0:
        print(f"Rank {rank} | Running reference forward/backward", flush=True)
        q_ref = q.detach().clone().requires_grad_()
        k_ref = k.detach().clone().requires_grad_()
        v_ref = v.detach().clone().requires_grad_()

        print(
            f"Rank {rank} | Running reference forward with q_ref.shape={q_ref.shape}, k_ref.shape={k_ref.shape}, v_ref.shape={v_ref.shape}",
            flush=True,
        )
        out_ref = minference_flash_attn_func(
            q_ref,
            k_ref,
            v_ref,
            cfg.v_size,
            cfg.s_size,
            causal=True,
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


def run_minfer_kernel_test(
    seq_len: int,
    batch_sz: int,
    head_dim: int,
    sparsity: float,
    ones: bool,
    num_qo_heads: int,
    num_kv_heads: int,
    attn_op_name: str,
):
    """Compare a distributed Minference kernel against dense reference outputs."""
    port = str(random.randint(12000, 20000))

    cfg = SimpleNamespace(
        batch_size=batch_sz,
        seq_len=seq_len,
        head_dim=head_dim,
        sparsity=sparsity,
        ones=ones,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
    )
    # derived sizes used by both candidate and reference kernels
    cfg.v_size = [int((1 - cfg.sparsity) * 0.1 * cfg.seq_len)] * cfg.num_qo_heads
    cfg.s_size = [int((1 - cfg.sparsity) * 0.2 * cfg.seq_len)] * cfg.num_qo_heads

    print("=" * 80)
    print(f"Testing {attn_op_name} with configuration:\n{cfg}")
    print("=" * 80)
    mp.spawn(
        _run_worker,
        args=(_WORLD_SIZE, port, cfg, attn_op_name),
        nprocs=_WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    run_minfer_kernel_test(
        seq_len=512 * 1024,
        batch_sz=1,
        head_dim=128,
        sparsity=0.9,
        ones=False,
        num_qo_heads=4,
        num_kv_heads=1,
        attn_op_name="minfer_stripe",
    )
