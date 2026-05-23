#!/usr/bin/bash
# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)" # path/to/MInference/mtraining
PROJECT_ROOT="$(cd "${BASE_DIR}/.." && pwd)" # path/to/MInference
PIP="$(which pip)"

sudo $PIP install -U pip setuptools wheel
sudo $PIP install ninja cmake pybind11 packaging psutil pytest
sudo $PIP install -r "${BASE_DIR}/requirements.txt"

sudo $PIP install git+https://github.com/microsoft/nnscaler.git@2368540417bc3b77b7e714d3f1a0de8a51bb66e8
sudo $PIP install "rotary-emb @ git+https://github.com/Dao-AILab/flash-attention.git@9356a1c0389660d7e231ff3163c1ac17d9e3824a#subdirectory=csrc/rotary" --no-build-isolation
sudo $PIP install "block_sparse_attn @ git+https://github.com/HalberdOfPineapple/flash-attention.git@block-sparse" --no-build-isolation
sudo $PIP install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.4.post1 --no-build-isolation
sudo $PIP install torch==2.3.1 torchvision==0.18.1
sudo $PIP install triton==3.0.0

# Get the path to nnscaler and write its path to PYTHONPATH in ~/.profile
NNSCALER_HOME=$(python -c "import nnscaler; print(nnscaler.__path__[0])")
echo "export NNSCALER_HOME=${NNSCALER_HOME}" >> ~/.profile
echo "export PYTHONPATH=${NNSCALER_HOME}:${PROJECT_ROOT}:\${PYTHONPATH}" >> ~/.profile
source ~/.profile

cd $PROJECT_ROOT
sudo MINFERENCE_FORCE_BUILD=TRUE $PIP install -e . --no-build-isolation

cd $BASE_DIR
sudo $PIP install -e $BASE_DIR

cp -r $PROJECT_ROOT/mtraining/utils/comm_prof/NVIDIA_A100-SXM4-40GB/* $NNSCALER_HOME/resources/profile/mi200/comm/
