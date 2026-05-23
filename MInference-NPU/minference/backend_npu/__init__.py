# Copyright (c) 2026
# Licensed under The MIT License [see LICENSE for details]
"""NPU backend 适配层。

- attention.py: dense / prefill / decode 三个 npu_fusion_attention 系列的薄封装
- cuda_shim.py: 上游 minference.cuda 扩展的占位（M4-a 时替换为真实现）
"""

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
