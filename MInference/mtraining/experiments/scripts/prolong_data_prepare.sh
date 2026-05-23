#!/usr/bin/bash

# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

export HF_TRUST_REMOTE_CODE=1
export HF_HOME=/scratch/hf_cache/huggingface
export HF_TOKEN_PATH="/scratch/.hf_access_token"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${BASE_DIR}/.." && pwd)"

# ------------------------------------------
# Download data
# Prerequisite: sudo apt-get install git-lfs && git lfs install
RAW_DATASET_DIR="/scratch/datasets" # "/path/to/datasets"
mkdir -p $RAW_DATASET_DIR


# Check whether the data is already downloaded
if [ -d $RAW_DATASET_DIR/long-context-524288 ]; then
    echo "Data already downloaded"
else
    echo "Downloading data..."
    git clone https://huggingface.co/datasets/princeton-nlp/prolong-data-512K $RAW_DATASET_DIR/long-context-524288
    cd $RAW_DATASET_DIR/long-context-524288
    git lfs fetch
    git lfs checkout
fi

# ------------------------------------------
# Data Processing
cd $BASE_DIR
MODEL_ID="Qwen/Qwen2.5-7B"
PROCESSED_DATA_DIR="/scratch/datasets/processed_datasets"
mkdir -p $PROCESSED_DATA_DIR

torchrun --nproc_per_node=1 \
	utils/data_utils/prolong.py \
    --model_id $MODEL_ID \
    --dataset_mix fixed_524288 \
    --dataset_path $RAW_DATASET_DIR/long-context-524288 \
    --save_path $PROCESSED_DATA_DIR/long-context-524288 \
    --sample_interval 4
