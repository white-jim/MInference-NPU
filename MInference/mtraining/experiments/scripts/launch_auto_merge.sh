#!/usr/bin/bash

# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

i=$(hostname | awk -F'-' '{print $2}')
NODE_RANK=$i
export NUM_NODES=4
export REUSE_TYPE="match"
export FORCE_TRITON=1

export HF_HOME=/scratch/hf_cache/huggingface
mkdir -p $HF_HOME
export HF_TRUST_REMOTE_CODE=true
export HF_DATASETS_TRUST_REMOTE_CODE=true

export MASTER_ADDR="node-0"
export MASTER_PORT="12345"

export NNSCALER_HOME="${HOME}/.conda/envs/ptca/lib/python3.10/site-packages/nnscaler/"
export PYTHONPATH="${NNSCALER_HOME}:${PYTHONPATH}"

# -----------------------------------------------
# TODO: Basic Environment Settings
SEQUENCE_LENGTH=524288
export GPU_NAME=A100
export GPU_PER_NODE=8
export WORLD_SIZE=32
export GPU_SET="${GPU_NAME}_${WORLD_SIZE}"

export SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export EXPR_HOME="$(cd "${SCRIPT_DIR}/../.." && pwd)" # .../mtraining
export EXPR_DATA_STORE="/blob/mtrain_expr_data_store/${GPU_SET}"
mkdir -p $EXPR_DATA_STORE
cd $EXPR_HOME

export EXPR_DIR="mtrain_qwen" # Name for the experiment set
export EXPR_NAME="qwen_3B_fp090_512K" # Name for the single experiment run
export MODEL_ID="Qwen/Qwen2.5-3B"

# -----------------------------------------------
export MERGE_CKPT_DIR="${EXPR_DATA_STORE}/${EXPR_DIR}/${EXPR_NAME}/merged_ckpts"
mkdir -p $MERGE_CKPT_DIR

export LOG_PATH="${MERGE_CKPT_DIR}/auto_merge.log"
echo "log path: $LOG_PATH"

python -m utils.auto_merge_ckpt \
    --gpu_set ${GPU_NAME}_${WORLD_SIZE} \
    --expr_dir $EXPR_DIR \
    --expr_name $EXPR_NAME \
    --model_id $MODEL_ID \
    --num_gpus $WORLD_SIZE \
    > $LOG_PATH 2>&1 &
