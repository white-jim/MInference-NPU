# Copyright (c) 2026
# Licensed under The MIT License [see LICENSE for details]
#
# MInference-NPU 安装脚本。
# 当前工作区只保留 PR-4 Ascend NPU 稀疏注意力 smoke/profiling 所需 Python 包。

from setuptools import find_packages, setup

with open("minference/version.py", "r") as f:
    exec(f.read())  # noqa: S102 — 注入 VERSION

setup(
    name="minference-npu",
    version=VERSION,  # noqa: F821 — 来自 exec 注入
    description="MInference PR-4 sparse attention adaptation for Huawei Ascend NPU",
    long_description=(
        "Trimmed Ascend NPU workspace for Phi-3 long-context smoke/profiling. "
        "stream_llm uses hardware band attention plus sink/LSE merge; "
        "block_sparse remains on the TileLang path-B track, with dense as "
        "the baseline and fallback."
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
