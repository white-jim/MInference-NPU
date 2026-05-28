# Copyright (c) 2026 (NPU adapter)
# Licensed under The MIT License [see LICENSE for details]
"""Smoke test for dense causal attention's hardware-band path.

Run on Ascend with::

    source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh
    PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl \
        python tests/test_dense_attention_band.py
"""

from __future__ import annotations

import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from minference.backend_npu.attention import dense_attention, _eager_attention_cpu_ref


def main() -> int:
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        print(f"[env] torch_npu unavailable: {exc}")
        return 1

    for seq_len in (128, 512, 1024):
        torch.manual_seed(123 + seq_len)
        q = torch.randn(1, 2, seq_len, 64, dtype=torch.float16, device="npu")
        k = torch.randn(1, 2, seq_len, 64, dtype=torch.float16, device="npu")
        v = torch.randn(1, 2, seq_len, 64, dtype=torch.float16, device="npu")
        out = dense_attention(q, k, v, causal=True).cpu()
        ref = _eager_attention_cpu_ref(q.cpu(), k.cpu(), v.cpu(), 1.0 / (64**0.5), True)
        diff = (out.float() - ref.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"[dense-band] S={seq_len} max_abs_diff={max_diff:.4e} mean_abs_diff={mean_diff:.4e}")
        if max_diff >= 6e-2:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
