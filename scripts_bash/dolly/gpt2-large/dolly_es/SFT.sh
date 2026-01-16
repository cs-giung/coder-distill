#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
NUM_PROCESSES=1

# Models
MODEL="checkpoints/dolly/gpt2-large/init"

# Datasets
INITIAL_DATA_JSONL="data/dolly/dolly_es.jsonl"
LIMIT=13312

# Parameters
EPOCHS=10
OUTPUT_ROOT="checkpoints/dolly/gpt2-large/dolly_es/SFT"

set -e

# Loop
CURRENT_MODEL="$MODEL"
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
        --data_path data/dolly_eval/dolly_es.jsonl \
        --output_file $CURRENT_MODEL/dolly_eval/dolly_es.jsonl

done

echo "SFT Loop Complete."

