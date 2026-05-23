# Copyright (c) 2026
# Licensed under The MIT License [see LICENSE for details]
"""上游 `minference.cuda` C++ 扩展的占位 shim。

上游仓库通过 `setup.py` 编译出 `minference.cuda` 子模块，导出
`convert_vertical_slash_indexes`（CUDA 索引展开 kernel）。NPU 版 v1 不构建 CUDA
扩展，本文件提供同名占位接口：

- M1 阶段：直接抛 `NotImplementedError`，因为三种稀疏分支会临时全部走 dense fallback，
  调不到这个函数。
- M4-a 阶段：把这里替换为 Triton-Ascend / 或临时 host CPU 实现。

上游 `ops/pit_sparse_flash_attention_v2.py` 里 `from ..cuda import convert_vertical_slash_indexes`
的路径会在 M4-a 时改为 `from ..backend_npu.cuda_shim import convert_vertical_slash_indexes`。
"""

from __future__ import annotations

import torch


__all__ = ["convert_vertical_slash_indexes"]


def convert_vertical_slash_indexes(
    seqlens: torch.Tensor,
    vertical_indexes: torch.Tensor,
    slash_indexes: torch.Tensor,
    context_size: int,
    block_size_M: int,
    block_size_N: int,
    causal: bool = True,
):
    """占位实现。M1 阶段稀疏分支全部走 dense fallback，不会调到这里。

    Raises:
        NotImplementedError: 一定会抛。M4-a 时再用 Triton-Ascend 实现替换。
    """
    raise NotImplementedError(
        "convert_vertical_slash_indexes 在 v1 M1 阶段未实现。"
        "M1 三种稀疏分支已全部退化为 dense（backend_npu.dense_attention），不应触达此函数。"
        "若你看到此异常，说明 patch.py / minference_forward.py 的 dense fallback 没生效，请排查。"
    )
