#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
NUM_PROCESSES=1

# Models
TEACHER_MODEL="checkpoints/dolly/gpt2-large/dolly_en/SFT/epoch_9/final_model"
STUDENT_MODEL="checkpoints/dolly/gpt2/dolly_en/SFT/epoch_9/final_model"

# Datasets
INITIAL_TEACHER_JSONL="data/dolly/dolly_en/42/$TEACHER_MODEL.jsonl"
INITIAL_STUDENT_JSONL="data/dolly/dolly_en/42/$STUDENT_MODEL.jsonl"

# Parameters
EPOCHS=10
OUTPUT_ROOT="checkpoints/dolly/gpt2/dolly_en/gpt2-large_OD_RKL"

set -e

# Loop
CURRENT_TEACHER_JSONL="$INITIAL_TEACHER_JSONL"
CURRENT_STUDENT_JSONL="$INITIAL_STUDENT_JSONL"
CURRENT_STUDENT_MODEL="$STUDENT_MODEL"
RUNNING_STUDENT_STATE=""

for (( epoch=0; epoch<EPOCHS; epoch++ ))
do    
    EPOCH_DIR="$OUTPUT_ROOT/epoch_$epoch"
    mkdir -p "$EPOCH_DIR"
    
    echo "========================================"
    echo "Epoch $epoch: Distillation Training"
    echo "Student Data: $CURRENT_STUDENT_JSONL"
    echo "Teacher Data: $CURRENT_TEACHER_JSONL"
    echo "========================================"
    
    CMD_ARGS=(
        "--teacher_jsonl" "$CURRENT_TEACHER_JSONL"
        "--student_jsonl" "$CURRENT_STUDENT_JSONL"
        "--teacher_model" "$TEACHER_MODEL"
        "--student_model" "$CURRENT_STUDENT_MODEL"
        "--lr" "1e-05"
        "--output_dir" "$EPOCH_DIR"
        "--fwd_kl_student" "0.0"
        "--rev_kl_student" "1.0"
        "--fwd_kl_teacher" "0.0"
        "--rev_kl_teacher" "0.0"
        "--max_length" "1024"
        "--use_alpaca_prompt"
        "--micro_batch_size" "16"
    )
    
    if [ ! -z "$RUNNING_STUDENT_STATE" ]; then
        echo "Resuming from state: $RUNNING_STUDENT_STATE"
        CMD_ARGS+=("--resume_from_checkpoint" "$RUNNING_STUDENT_STATE")
    fi
    
    .venv/bin/accelerate launch \
        --num_processes $NUM_PROCESSES \
        --main_process_port 0 \
        scripts/train_student.py "${CMD_ARGS[@]}"
    
    RUNNING_STUDENT_STATE="$EPOCH_DIR/final_state"
    CURRENT_STUDENT_MODEL="$EPOCH_DIR/final_model"

    echo "========================================"
    echo "Epoch $epoch: Evaluation"
    echo "Model: $CURRENT_STUDENT_MODEL"
    echo "========================================"
    
    .venv/bin/python scripts/eval_dolly.py \
        --model $CURRENT_STUDENT_MODEL \
        --data_path data/dolly_eval/dolly_en.jsonl \
        --output_file $CURRENT_STUDENT_MODEL/dolly_eval/dolly_en.jsonl
    
    echo "========================================"
    echo "Epoch $epoch: Generating Data from Student"
    echo "Model: $CURRENT_STUDENT_MODEL"
    echo "========================================"
    
    CURRENT_STUDENT_JSONL="$EPOCH_DIR/generated_student.jsonl"
    .venv/bin/python scripts/generate_data.py \
        --input "$INITIAL_STUDENT_JSONL" \
        --output "$CURRENT_STUDENT_JSONL" \
        --model "$CURRENT_STUDENT_MODEL" \
        --seed $epoch
    
    CURRENT_TEACHER_JSONL="data/dolly/dolly_en/$epoch/$TEACHER_MODEL.jsonl"

done

echo "Distillation Loop Complete."
