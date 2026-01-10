#!/bin/bash
export CUDA_VISIBLE_DEVICES=6,7
NUM_MACHINES=1
NUM_PROCESSES=2
TENSOR_PARALLEL_SIZE=2

# Models
TEACHER_MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
STUDENT_MODEL="checkpoints/math/Qwen2.5-0.5B/init"

# Datasets
INITIAL_TEACHER_JSONL="data/32k/math/42/$TEACHER_MODEL.jsonl"
INITIAL_STUDENT_JSONL="data/32k/math/42/$STUDENT_MODEL.jsonl"
LIMIT=32768

# Parameters
EPOCHS=4
OUTPUT_ROOT="checkpoints/math/Qwen2.5-0.5B/32k/Qwen2.5-Math-7B-Instruct_KD_RKL"

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
        "--limit" "$LIMIT"
        "--lr" "1e-05"
        "--output_dir" "$EPOCH_DIR"
        "--fwd_kl_student" "0.0"
        "--rev_kl_student" "0.0"
        "--fwd_kl_teacher" "0.0"
        "--rev_kl_teacher" "1.0"
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
    
    .venv/bin/lm_eval \
        --model vllm \
        --model_args pretrained="$CURRENT_STUDENT_MODEL",tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=$NUM_PROCESSES \
        --tasks minerva_math,minerva_math500,gsm8k \
        --batch_size 1 \
        --fewshot_as_multiturn \
        --apply_chat_template \
        --gen_kwargs max_gen_toks=2048
    
    echo "========================================"
    echo "Epoch $epoch: Generating Data from Student"
    echo "Model: $CURRENT_STUDENT_MODEL"
    echo "========================================"
    
    CURRENT_STUDENT_JSONL="$EPOCH_DIR/generated_student.jsonl"
    .venv/bin/python scripts/generate_data.py \
        --input "$INITIAL_STUDENT_JSONL" \
        --output "$CURRENT_STUDENT_JSONL" \
        --model "$CURRENT_STUDENT_MODEL" \
        --limit $LIMIT \
        --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
        --seed $epoch
    
    CURRENT_TEACHER_JSONL="data/32k/math/$epoch/$TEACHER_MODEL.jsonl"

done

echo "Distillation Loop Complete."
