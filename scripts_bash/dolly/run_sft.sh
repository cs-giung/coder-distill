#!/bin/bash

# Usage:
# ./scripts_bash/dolly/run_sft.sh <CUDA_VISIBLE_DEVICES> <MODEL_PATH> [MICRO_BATCH_SIZE]

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <CUDA_VISIBLE_DEVICES> <MODEL_PATH> [MICRO_BATCH_SIZE]"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$1
MODEL_PATH=$2
MICRO_BATCH_SIZE=${3:-8}

# Calculate NUM_PROCESSES
NUM_PROCESSES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Running on devices: $CUDA_VISIBLE_DEVICES (Count: $NUM_PROCESSES)"

# Extract Model Name
# Expected format: checkpoints/dolly/<model_name>/... 
MODEL_NAME=$(echo $MODEL_PATH | awk -F'/' '{print $3}')

if [ -z "$MODEL_NAME" ]; then
    echo "Error: Could not extract model name from path '$MODEL_PATH'."
    echo "Expected path format: checkpoints/dolly/<model_name>/..."
    exit 1
fi

# Paths
DATASET_NAME="dolly_en"
INITIAL_DATA_JSONL="data/dolly/${DATASET_NAME}.jsonl"
EVAL_DATA_JSONL="data/dolly_eval/${DATASET_NAME}.jsonl"
OUTPUT_ROOT="checkpoints/dolly/${MODEL_NAME}/${DATASET_NAME}/SFT"
LIMIT=13312
EPOCHS=10

echo "Output Directory: $OUTPUT_ROOT"
echo "Dataset: $INITIAL_DATA_JSONL"

set -e

# Loop
CURRENT_MODEL="$MODEL_PATH"
RUNNING_STATE=""

for (( epoch=0; epoch<EPOCHS; epoch++ ))
do    
    EPOCH_DIR="$OUTPUT_ROOT/epoch_$epoch"
    mkdir -p "$EPOCH_DIR"
    
    echo "========================================"
    echo "Epoch $epoch: SFT Training"
    echo "Data: $INITIAL_DATA_JSONL"
    echo "========================================"
    
    CMD_ARGS=(
        "--data_jsonl" "$INITIAL_DATA_JSONL"
        "--model" "$CURRENT_MODEL"
        "--limit" "$LIMIT"
        "--output_dir" "$EPOCH_DIR"
        "--max_length" "1024"
        "--use_alpaca_prompt"
        "--micro_batch_size" "$MICRO_BATCH_SIZE"
    )

    if [ ! -z "$RUNNING_STATE" ]; then
        echo "Resuming from state: $RUNNING_STATE"
        CMD_ARGS+=("--resume_from_checkpoint" "$RUNNING_STATE")
    fi
    
    .venv/bin/accelerate launch \
        --num_processes $NUM_PROCESSES \
        --main_process_port 0 \
        scripts/train_sft.py "${CMD_ARGS[@]}"
    
    RUNNING_STATE="$EPOCH_DIR/final_state"
    CURRENT_MODEL="$EPOCH_DIR/final_model"

    echo "========================================"
    echo "Epoch $epoch: Evaluation"
    echo "Model: $CURRENT_MODEL"
    echo "========================================"
    
    .venv/bin/python scripts/eval_dolly.py \
        --model $CURRENT_MODEL \
        --data_path $EVAL_DATA_JSONL \
        --output_file $CURRENT_MODEL/dolly_eval/${DATASET_NAME}.jsonl

done

echo "SFT Loop Complete."
