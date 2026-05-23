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

# ------------------------------------------
export EXPR_DIR="mtrain_qwen" # Name for the experiment set
export EXPR_NAME="qwen_3B_fp090_512K" # Name for the single experiment run
export MODEL_ID="Qwen/Qwen2.5-3B"
export DATASET_PATH="/scratch/datasets/processed_datasets/long-context-524288"
export MODEL_CONFIG_PATH="${EXPR_HOME}/model_configs/qwen2/lc_config_3B"
echo "Using model config path: $MODEL_CONFIG_PATH"
TRANSFER_CONFIG_DIR="none"
export TRAIN_ATTN_CONFIG_PATH="${EXPR_HOME}/train_attn_configs/qwen_flex_090.yaml"
export ATTN_TYPE="minfer"

# ------------------------------------------
# Training Path settings
export TF_LOG_PATH="$EXPR_DATA_STORE/$EXPR_DIR/tf_logs"
export CKPT_PATH="$EXPR_DATA_STORE/$EXPR_DIR/$EXPR_NAME/checkpoints"
export COMPILE_PATH="$EXPR_DATA_STORE/compile_config/rank_${NODE_RANK}"
export PAS_PROFILE_DIR="$EXPR_DATA_STORE/$EXPR_DIR/$EXPR_NAME/pas_profile"
mkdir -p $TF_LOG_PATH
mkdir -p $CKPT_PATH
mkdir -p $COMPILE_PATH
mkdir -p $PAS_PROFILE_DIR

# -------------------------------------------
# Training Settings
export SOLVER="dp"
export TRACE_STRATEGY="reuse_cache"
export FORCE_BROADCAST_ALL=0
if [ "$FORCE_BROADCAST_ALL" -eq 1 ]; then
    FORCE_BROADCAST_ALL_FLAG="--force_broadcast_all"
else
    FORCE_BROADCAST_ALL_FLAG=""
fi

export GLOBAL_BATCH_SIZE=64
export MICRO_BATCH_SIZE=1
export MEM_CONSTRAINT=40

export NUM_ITER=0
export NUM_EPOCH=1

export CKPT_SAVE_STEP=1
export CKPT_SAVE_EPOCH=0

export CHECK_RESUME=0
if [ "$CHECK_RESUME" -eq 1 ]; then
    CHECK_RESUME="--check_resume"
else
    CHECK_RESUME=""
fi

# -------------------------------------------
# Logging Path

export LOG_PATH="${EXPR_DATA_STORE}/${EXPR_DIR}/${EXPR_NAME}/rank_${NODE_RANK}"
mkdir -p $LOG_PATH
max=0
# Loop through matching files (if no match, the glob expands to itself, so check existence)
for file in "$LOG_PATH"/train*.log; do
    [ -e "$file" ] || continue  # Skip if no file exists

    base=$(basename "$file")
    if [ "$base" = "train.log" ]; then
        num=0
    elif [[ "$base" =~ train_([0-9]+)\.log ]]; then
        num="${BASH_REMATCH[1]}"
    else
        continue
    fi

    if (( num > max )); then
        max=$num
    fi
done
next=$((max + 1))
echo "Logging directed to $LOG_PATH/train_${next}.log"

torchrun --nproc_per_node=$GPU_PER_NODE \
        --nnodes=$NUM_NODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train.py  --plan_ngpus $WORLD_SIZE \
                    --runtime_ngpus $WORLD_SIZE \
                    --name $EXPR_NAME \
                    --seq_len $SEQUENCE_LENGTH \
                    --attn_type $ATTN_TYPE \
                    --solver $SOLVER \
                    --train_attn_config_path $TRAIN_ATTN_CONFIG_PATH \
                    --reuse_type $REUSE_TYPE \
                    --model_id $MODEL_ID \
                    --n_iter $NUM_ITER \
                    --n_epochs $NUM_EPOCH \
                    --global_batch_size $GLOBAL_BATCH_SIZE \
                    --micro_batch_size $MICRO_BATCH_SIZE \
                    --pas_profile_dir $PAS_PROFILE_DIR \
                    --dataset_path $DATASET_PATH \
                    --compile_save_path $COMPILE_PATH \
                    --tf_log_dir $TF_LOG_PATH \
                    --model_config_path $MODEL_CONFIG_PATH \
                    --ckpt_save_dir $CKPT_PATH \
                    --ckpt_n_step $CKPT_SAVE_STEP \
                    --ckpt_n_epoch $CKPT_SAVE_EPOCH \
                    --trace_strategy $TRACE_STRATEGY \
                    --transfer_config_dir $TRANSFER_CONFIG_DIR \
                    --mem_constraint $MEM_CONSTRAINT \
                    $FORCE_BROADCAST_ALL_FLAG \
                    $CHECK_RESUME > $LOG_PATH/train_${next}.log 2>&1
echo "Log saved to $LOG_PATH/train_${next}.log"
