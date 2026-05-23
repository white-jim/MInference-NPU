# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Shared helpers for standalone distributed raw-kernel tests."""
from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from minference.ops.utils import check_by_correct_rate

SEED_BASE = 2025


def init_process_group(rank: int, world_size: int, port: str) -> None:
    """Initialize NCCL backend for the current worker."""
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": port,
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank % min(world_size, torch.cuda.device_count())),
            "LOCAL_WORLD_SIZE": str(min(world_size, torch.cuda.device_count())),
        }
    )
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def create_full_inputs(
    rank: int,
    cfg: SimpleNamespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create full-sequence inputs on rank 0 and broadcast to all ranks."""
    if rank == 0:
        rand_or_one = (
            torch.randn if not cfg.ones else lambda s, **k: torch.ones(*s, **k)
        )
        q = rand_or_one(
            (cfg.batch_size, cfg.seq_len, cfg.num_qo_heads, cfg.head_dim),
            dtype=dtype,
            device=device,
        )
        k = rand_or_one(
            (cfg.batch_size, cfg.seq_len, cfg.num_kv_heads, cfg.head_dim),
            dtype=dtype,
            device=device,
        )
        v = rand_or_one(
            (cfg.batch_size, cfg.seq_len, cfg.num_kv_heads, cfg.head_dim),
            dtype=dtype,
            device=device,
        )
        dout = rand_or_one(
            (cfg.batch_size, cfg.seq_len, cfg.num_qo_heads, cfg.head_dim),
            dtype=dtype,
            device=device,
        )
    else:
        shape_q = (cfg.batch_size, cfg.seq_len, cfg.num_qo_heads, cfg.head_dim)
        shape_kv = (cfg.batch_size, cfg.seq_len, cfg.num_kv_heads, cfg.head_dim)
        q = torch.empty(shape_q, device=device, dtype=dtype)
        k = torch.empty(shape_kv, device=device, dtype=dtype)
        v = torch.empty(shape_kv, device=device, dtype=dtype)
        dout = torch.empty(shape_q, device=device, dtype=dtype)

    for tensor in (q, k, v, dout):
        dist.broadcast(tensor, src=0)

    return q, k, v, dout


def slice_local_inputs(
    rank: int,
    world_size: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice local sequence shard and set gradients."""
    local_ctx = q.size(1) // world_size
    sl = slice(rank * local_ctx, (rank + 1) * local_ctx)
    q_local = q[:, sl].clone().detach().requires_grad_()
    k_local = k[:, sl].clone().detach().requires_grad_()
    v_local = v[:, sl].clone().detach().requires_grad_()
    dout_local = dout[:, sl].clone()
    return q_local, k_local, v_local, dout_local


def gather_sequence_shards(local_tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """All-gather sequence shards and concatenate along sequence dim."""
    gathered = [torch.empty_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor)
    return torch.cat(gathered, dim=1)


def check_forward_and_qkv_grads(
    seq_len: int,
    final_out: torch.Tensor,
    out_ref: torch.Tensor,
    grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ref_grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    atol: float,
    rtol: float,
    raise_on_fail: bool = False,
) -> bool:
    """Run a unified forward/QKV check flow with detailed error analysis."""

    def _format_index(flat_index: int, shape: torch.Size) -> tuple[int, ...]:
        coords = []
        stride = 1
        for size in reversed(shape):
            coords.append((flat_index // stride) % size)
            stride *= size
        return tuple(reversed(coords))

    def _analyze_tensor(name: str, got: torch.Tensor, ref: torch.Tensor) -> bool:
        # got_fp = got.float()
        # ref_fp = ref.float()
        diff = (got - ref).abs()
        max_diff = diff.max()
        mean_diff = diff.mean()
        min_diff = diff.min()

        ok = check_by_correct_rate(got, ref, ATOL=atol, RTOL=rtol)
        status = "PASS" if ok else "FAIL"
        print(
            (
                f"[{status}] {name}: max_diff={max_diff.item():.6e}, "
                f"mean_diff={mean_diff.item():.6e}, "
                f" min_diff={min_diff.item():.6e}"
            ),
            flush=True,
        )

        if not ok:
            flat_idx = int(diff.argmax().item())
            idx = _format_index(flat_idx, diff.shape)
            got_val = got[idx].item()
            ref_val = ref[idx].item()
            print(
                (
                    f"    worst_idx={idx}, got={got_val:.6e}, ref={ref_val:.6e}, "
                    f"abs_diff={max_diff.item():.6e}"
                ),
                flush=True,
            )
        return ok

    _ = seq_len  # reserved for optional sequence-wise diagnostics
    overall_ok = True
    forward_ok = _analyze_tensor("forward output", final_out, out_ref)
    if not forward_ok:
        overall_ok = False
        if raise_on_fail:
            raise AssertionError(
                "forward output mismatch; see printed diff analysis for details"
            )
        return overall_ok

    for name, grad, ref_grad in zip(("Q-grad", "K-grad", "V-grad"), grads, ref_grads):
        grad_ok = _analyze_tensor(name, grad, ref_grad)
        overall_ok = overall_ok and grad_ok

    if raise_on_fail and not overall_ok:
        raise AssertionError("gradient mismatch; see printed diff analysis for details")

    return overall_ok
