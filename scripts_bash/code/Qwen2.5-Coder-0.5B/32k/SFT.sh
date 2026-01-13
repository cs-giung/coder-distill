#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5
NUM_MACHINES=1
NUM_PROCESSES=2
TENSOR_PARALLEL_SIZE=2

# Models
STUDENT_MODEL="checkpoints/code/Qwen2.5-Coder-0.5B/init"

# Datasets
INITIAL_DATA_JSONL="data/32k/code.jsonl"
LIMIT=32768

# Parameters
EPOCHS=4
OUTPUT_ROOT="checkpoints/code/Qwen2.5-Coder-0.5B/32k/SFT"

set -e

# Loop
CURRENT_STUDENT_MODEL="$STUDENT_MODEL"
RUNNING_STUDENT_STATE=""

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
        "--student_model" "$CURRENT_STUDENT_MODEL"
        "--limit" "$LIMIT"
        "--lr" "1e-05"
        "--output_dir" "$EPOCH_DIR"
    )
    
    # Resume logic if needed (optional, from previous script)
    if [ ! -z "$RUNNING_STUDENT_STATE" ]; then
        echo "Resuming from state: $RUNNING_STUDENT_STATE"
        CMD_ARGS+=("--resume_from_checkpoint" "$RUNNING_STUDENT_STATE")
    fi
    
    .venv/bin/accelerate launch \
        --num_processes $NUM_PROCESSES \
        --main_process_port 0 \
        scripts/train_student_sft.py "${CMD_ARGS[@]}"
    
    RUNNING_STUDENT_STATE="$EPOCH_DIR/final_state"
    CURRENT_STUDENT_MODEL="$EPOCH_DIR/final_model" # Update model for next iter or eval

    echo "========================================"
    echo "Epoch $epoch: Evaluation"
    echo "Model: $CURRENT_STUDENT_MODEL"
    echo "========================================"
    
    .venv/bin/evalplus.evaluate \
        --model $CURRENT_STUDENT_MODEL \
        --root $CURRENT_STUDENT_MODEL \
        --dataset humaneval \
        --backend vllm \
        --tp $TENSOR_PARALLEL_SIZE \
        --greedy
    
    .venv/bin/evalplus.evaluate \
        --model $CURRENT_STUDENT_MODEL \
        --root $CURRENT_STUDENT_MODEL \
        --dataset mbpp \
        --backend vllm \
        --tp $TENSOR_PARALLEL_SIZE \
        --greedy

done

echo "SFT Loop Complete."
