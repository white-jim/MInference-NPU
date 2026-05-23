# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import functools
import os
import random
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessRaisedException

from minference.dist_ops.moba_zigzag import moba_zigzag_func
from minference.dist_ops.test.raw_test_utils import (
    SEED_BASE,
    check_forward_and_qkv_grads,
    create_full_inputs,
    gather_sequence_shards,
    init_process_group,
    slice_local_inputs,
)
from minference.ops.moba import moba_attn_func
from minference.ops.utils import set_seed

# ------------- constants ------------------------------------------------------
_ATOL = 1e-2
_RTOL = 1e-2
_WORLD_SIZE = 4


# ------------- helpers --------------------------------------------------------
def skip_if_cuda_oom(test_func):
    """Decorator: convert worker OOM raised by spawn into a pytest skip."""

    @functools.wraps(test_func)
    def _wrapper(*args, **kwargs):
        try:
            return test_func(*args, **kwargs)
        except (torch.OutOfMemoryError, ProcessRaisedException) as exc:
            if "OutOfMemoryError" not in str(exc):
                raise
            torch.cuda.empty_cache()
            pytest.skip("skipped because the GPU ran out of memory")

    return _wrapper


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
    out_local = moba_zigzag_func(
        q_local,
        k_local,
        v_local,
        layer_idx=0,
        global_seq_len=cfg.seq_len,
        moba_chunk_size=cfg.moba_chunk_size,
        moba_topk=cfg.moba_topk,
    )
    torch.autograd.backward(out_local, dout_local)

    final_out = gather_sequence_shards(out_local, world_size)
    grads = tuple(
        gather_sequence_shards(grad, world_size)
        for grad in (q_local.grad, k_local.grad, v_local.grad)
    )
    torch.distributed.barrier()
    torch.cuda.synchronize()
    # ---------------------------------------
    if rank == 0:
        q_ref = q.detach().clone().requires_grad_()
        k_ref = k.detach().clone().requires_grad_()
        v_ref = v.detach().clone().requires_grad_()

        out_ref = moba_attn_func(
            q_ref,
            k_ref,
            v_ref,
            global_seq_len=cfg.seq_len,
            moba_chunk_size=cfg.moba_chunk_size,
            moba_topk=cfg.moba_topk,
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
@skip_if_cuda_oom
@pytest.mark.skipif(torch.cuda.device_count() < _WORLD_SIZE, reason="Not enough GPUs")
@pytest.mark.parametrize("seq_len", [16384, 32768])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_qkv_head_pair", [(4, 1), (4, 4)])
@pytest.mark.parametrize("moba_chunk_size", [128, 256])
@pytest.mark.parametrize("moba_topk", [8, 16])
def test_moba_kernels(
    seq_len: int,
    head_dim: int,
    num_qkv_head_pair: tuple[int, int],
    moba_chunk_size: int,
    moba_topk: int,
):
    """
    Compare every sparse kernel against the dense Flash-Attention reference on
    both forward pass and input-gradient w.r.t Q/K/V.
    """

    port = str(random.randint(12000, 20000))
    cfg = SimpleNamespace(
        batch_size=1,
        seq_len=seq_len,
        head_dim=head_dim,
        ones=False,
        num_qo_heads=num_qkv_head_pair[0],
        num_kv_heads=num_qkv_head_pair[1],
        moba_chunk_size=moba_chunk_size,
        moba_topk=moba_topk,
    )

    print(f"=" * 80)
    print(f"Testing MoBA (w. Zigzag) with configuration:\n{cfg}")
    print(f"=" * 80)
    mp.spawn(
        _run_worker,
        args=(_WORLD_SIZE, port, cfg),
        nprocs=_WORLD_SIZE,
        join=True,
    )
