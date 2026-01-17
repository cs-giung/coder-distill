#!/bin/bash

# Usage:
# ./scripts_bash/dolly/run_distillation.sh <CUDA_VISIBLE_DEVICES> <TEACHER_MODEL_PATH> <STUDENT_MODEL_PATH> <METHOD> [MICRO_BATCH_SIZE]

# Model:
# checkpoints/dolly/gpt2/dolly_en/SFT/epoch_9/final_model
# checkpoints/dolly/gpt2-medium/dolly_en/SFT/epoch_8/final_model
# checkpoints/dolly/gpt2-large/dolly_en/SFT/epoch_9/final_model
# checkpoints/dolly/gpt2-xl/dolly_en/SFT/epoch_6/final_model

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <CUDA_VISIBLE_DEVICES> <TEACHER_MODEL_PATH> <STUDENT_MODEL_PATH> <METHOD> [MICRO_BATCH_SIZE]"
    echo "METHOD options: KD_FKL, KD_RKL, OD_FKL, OD_RKL"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$1
TEACHER_MODEL=$2
STUDENT_MODEL=$3
METHOD=$4
MICRO_BATCH_SIZE=${5:-8}

# Calculate NUM_PROCESSES
NUM_PROCESSES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Running on devices: $CUDA_VISIBLE_DEVICES (Count: $NUM_PROCESSES)"

# Determine Model Names (Assuming checkpoints/dolly/<name>/...)
# Expected path: checkpoints/dolly/gpt2-medium/dolly_en/...
TEACHER_NAME=$(echo $TEACHER_MODEL | awk -F'/' '{print $3}')
STUDENT_NAME=$(echo $STUDENT_MODEL | awk -F'/' '{print $3}')

if [ -z "$TEACHER_NAME" ] || [ -z "$STUDENT_NAME" ]; then
    echo "Error: Could not extract model names from paths."
    echo "Expected path format: checkpoints/dolly/<model_name>/..."
    exit 1
fi

# Determine OUTPUT_ROOT
# Output path: checkpoints/dolly/<student_name>/dolly_en/<teacher_name>_<METHOD>
OUTPUT_ROOT="checkpoints/dolly/$STUDENT_NAME/dolly_en/${TEACHER_NAME}_${METHOD}"
echo "Output Directory: $OUTPUT_ROOT"

# Determine KL Parameters
FWD_KL_STUDENT="0.0"
REV_KL_STUDENT="0.0"
FWD_KL_TEACHER="0.0"
REV_KL_TEACHER="0.0"

case $METHOD in
    "KD_FKL")
        FWD_KL_TEACHER="1.0"
        ;;
    "KD_RKL")
        REV_KL_TEACHER="1.0"
        ;;
    "OD_FKL")
        FWD_KL_STUDENT="1.0"
        ;;
    "OD_RKL")
        REV_KL_STUDENT="1.0"
        ;;
    *)
        echo "Error: Unknown METHOD '$METHOD'. Supported: KD_FKL, KD_RKL, OD_FKL, OD_RKL"
        exit 1
        ;;
esac

# Other Constants
INITIAL_TEACHER_JSONL="data/dolly/dolly_en/42/$TEACHER_MODEL.jsonl"
INITIAL_STUDENT_JSONL="data/dolly/dolly_en/42/$STUDENT_MODEL.jsonl"
EPOCHS=10

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
    echo "Method: $METHOD"
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
        "--fwd_kl_student" "$FWD_KL_STUDENT"
        "--rev_kl_student" "$REV_KL_STUDENT"
        "--fwd_kl_teacher" "$FWD_KL_TEACHER"
        "--rev_kl_teacher" "$REV_KL_TEACHER"
        "--max_length" "1024"
        "--use_alpaca_prompt"
        "--micro_batch_size" "$MICRO_BATCH_SIZE"
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
