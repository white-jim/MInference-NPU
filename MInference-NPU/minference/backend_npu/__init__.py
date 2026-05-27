# Copyright (c) 2026
# Licensed under The MIT License [see LICENSE for details]
"""Dense/decode attention helpers for Ascend NPU."""

from .attention import (
    dense_attention,
    decode_dense,
    is_npu_available,
    prefill_dense,
)

__all__ = [
    "dense_attention",
    "decode_dense",
    "is_npu_available",
    "prefill_dense",
]
