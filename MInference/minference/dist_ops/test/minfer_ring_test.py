# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import os
import random
from types import SimpleNamespace
from typing import Callable

import pytest
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
    # ----------------- reference: dense Flash-Attention ----------------------
    if rank == 0:
        q_ref = q.detach().clone().requires_grad_()
        k_ref = k.detach().clone().requires_grad_()
        v_ref = v.detach().clone().requires_grad_()

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
            raise_on_fail=True,
        )
    dist.destroy_process_group()


# ------------- pytest entry-point --------------------------------------------
@pytest.mark.skipif(torch.cuda.device_count() < _WORLD_SIZE, reason="Not enough GPUs")
@pytest.mark.parametrize("seq_len", [131072, 262144, 524288])
@pytest.mark.parametrize("batch_sz", [1])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("sparsity", [0.9, 0.95])
@pytest.mark.parametrize("num_qkv_head_pair", [(4, 1), (4, 4)])
@pytest.mark.parametrize("use_triton", [True, False])
@pytest.mark.parametrize(
    "attn_op_name", ["minfer_zigzag", "minfer_stripe", "minfer_dr_stripe"]
)
def test_sparse_attention_kernels(
    seq_len: int,
    batch_sz: int,
    head_dim: int,
    sparsity: float,
    num_qkv_head_pair: tuple[int, int],
    use_triton: bool,
    attn_op_name: str,
):
    """
    Compare every sparse kernel against the dense Flash-Attention reference on
    both forward pass and input-gradient w.r.t Q/K/V.
    """
    port = str(random.randint(12000, 20000))
    if attn_op_name == "minfer_zigzag" and use_triton:
        pytest.skip("minfer_zigzag is not implemented with the Triton path")

    cfg = SimpleNamespace(
        batch_size=batch_sz,
        seq_len=seq_len,
        head_dim=head_dim,
        sparsity=sparsity,
        ones=False,
        num_qo_heads=num_qkv_head_pair[0],
        num_kv_heads=num_qkv_head_pair[1],
    )
    # derived sizes used by both candidate and reference kernels
    cfg.v_size = [int((1 - cfg.sparsity) * 0.1 * cfg.seq_len)] * cfg.num_qo_heads
    cfg.s_size = [int((1 - cfg.sparsity) * 0.2 * cfg.seq_len)] * cfg.num_qo_heads

    print(f"=" * 80)
    print(f"Testing {attn_op_name} with configuration:\n{cfg}")
    print(f"=" * 80)
    prev_force_triton = os.environ.get("FORCE_TRITON")
    try:
        if use_triton:
            os.environ["FORCE_TRITON"] = "1"
        else:
            os.environ.pop("FORCE_TRITON", None)

        mp.spawn(
            _run_worker,
            args=(_WORLD_SIZE, port, cfg, attn_op_name),
            nprocs=_WORLD_SIZE,
            join=True,
        )
    finally:
        if prev_force_triton is None:
            os.environ.pop("FORCE_TRITON", None)
        else:
            os.environ["FORCE_TRITON"] = prev_force_triton
