# Copyright (c) 2026
# Licensed under The MIT License [see LICENSE for details]
#
# MInference-NPU 安装脚本（v1）
# - 不构建 CUDA 扩展：上游 `csrc/vertical_slash_index.cu` 由 `backend_npu/cuda_shim.py`
#   的纯 Python/CPU 双指针实现顶替；三种稀疏 kernel 走 `npu_fusion_attention` + bool mask。
# - 与上游 MInference 的 setup.py 区别：去掉 CUDAExtension/cmdclass

from setuptools import find_packages, setup

with open("minference/version.py", "r") as f:
    exec(f.read())  # noqa: S102 — 注入 VERSION

setup(
    name="minference-npu",
    version=VERSION,  # noqa: F821 — 来自 exec 注入
    description="MInference 1.0 ported to Huawei Ascend NPU (v1)",
    long_description=(
        "Ascend NPU port of Microsoft MInference 1.0 long-context prefill "
        "acceleration. Targets CANN 8.1+ / torch_npu 2.5+ / npu_fusion_attention. "
        "v1 scope: vertical_and_slash + block_sparse + stream_llm + dense fallback."
    ),
    author="MInference-NPU contributors",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    include_package_data=True,
    package_data={
        "minference.configs": ["*.json", "leank/*"],
    },
    install_requires=[
        "torch>=2.5.1,<2.8",
        # torch_npu / CANN 走昇腾官方渠道，版本需按实机 CANN 匹配。
        "transformers>=4.45",
        "accelerate>=0.28",
        "numpy",
        # 上游 MInference 还依赖 flash-attn / sgl_kernel / vllm-flash-attn —— NPU 上不可用，
        # 已在 ops / patch 层用 try-except 守护，不写入 install_requires
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-xdist",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
