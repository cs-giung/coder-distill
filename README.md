# coder-distill

A personal playground for exploring LLM distillation.

## Overview

The basic form of LLM distillation involves finding a student model $p_{\text{stu}}$ that minimizes the divergence relative to a given teacher model $p_{\text{tea}}$:

$$\mathbb{E}_{x \sim \mathcal{D}}
\mathbb{E}_{y_{1:T} \sim p_{\text{gen}}(\cdot|x)}
\sum_{t=1}^{T} D(
p_{\text{tea}}(\cdot|x,y_{<t})
\parallel
p_{\text{stu}}(\cdot|x,y_{<t}))$$

1. For $x \sim \mathcal{D}$: We utilize the first 32k samples from `nickrosh/Evol-Instruct-Code-80k-v1` and `meta-math/MetaMathQA`. Refer to `data/32k/code.jsonl` and `data/32k/math.jsonl` for the specific instances.

2. For $y_{1:T} \sim p_{\text{gen}}(\cdot | x)$: We distinguish between two distillation settings based on the generator:
    - Knowledge Distillation (KD): $p_{\text{gen}}$ is the teacher model $p_{\text{tea}}$.
    - On-policy Distillation (OD): $p_{\text{gen}}$ is the student model $p_{\text{stu}}$.

3. For $D$: We consider the following fundamental divergence measures:
    - Forward KL (FKL): $D(p_{\text{tea}} \parallel p_{\text{stu}})$, which encourages the student to cover the teacher's entire distribution.
    - Reverse KL (RKL): $D(p_{\text{stu}} \parallel p_{\text{tea}})$, which encourages the student to focus on the teacher's high-probability modes.

## Distilled Models

### Prepare initial student model
```bash
# e.g., Qwen2.5-0.5B (student)
python scripts/generate_checkpoints.py
```
We prepare student models to have the same tokenizer and template as the teacher;
- [`checkpoints/code/Qwen2.5-0.5B/init`](checkpoints/code/Qwen2.5-0.5B/init)
- [`checkpoints/math/Qwen2.5-0.5B/init`](checkpoints/math/Qwen2.5-0.5B/init)

### Prepare training sequences
```bash
# e.g., Qwen2.5-Coder-3B-Instruct (teacher)
export SEED={42,0,1,2}
export MODEL="Qwen/Qwen2.5-Coder-3B-Instruct"
python scripts/generate_data.py --input data/32k/code.jsonl --output data/32k/code/$SEED/$MODEL.jsonl --model "$MODEL" --tensor_parallel_size 1 --limit 32768 --seed $SEED

# e.g., Qwen2.5-0.5B (student)
export SEED=42
export MODEL="checkpoints/code/Qwen2.5-0.5B"
python scripts/generate_data.py --input data/32k/code.jsonl --output data/32k/code/$SEED/$MODEL.jsonl --model "$MODEL" --tensor_parallel_size 1 --limit 32768 --seed $SEED
```
Since the teacher model is frozen, we pre-generate its sequences to save computational resources. However, the student model should generate sequences on-the-fly during training (except the very first epoch with the initial student model; `SEED=42` data).
- [`scripts_bash/code/prepare_data.txt`](scripts_bash/code/prepare_data.txt)
- [`scripts_bash/math/prepare_data.txt`](scripts_bash/math/prepare_data.txt)

### Code
```bash
# e.g., KD_FKL
mkdir -p checkpoints/code/Qwen2.5-0.5B/32k/Qwen2.5-Coder-3B-Instruct_KD_FKL
./scripts_bash/code/Qwen2.5-0.5B/32k/Qwen2.5-Coder-3B-Instruct_KD_FKL.sh > checkpoints/code/Qwen2.5-0.5B/32k/Qwen2.5-Coder-3B-Instruct_KD_FKL/log.txt
```

| model                                 | HumanEval   | MBPP        | AVG  | Ref |
| :-                                    | :-:         | :-:         | :-:  | :-: |
| Qwen2.5-0.5B (Student)                | 29.9 / 26.8 | 45.2 / 37.6 | 34.9 |
| SFT                                   | 31.7 / 27.4 | 39.4 / 33.6 | 33.0 |
||
| Qwen2.5-Coder-3B-Instruct (Teacher)   | 84.8 / 79.3 | 75.7 / 64.3 | 76.0 |
| KD_FKL                                | 34.1 / 28.7 | 41.3 / 34.4 | 34.6 | [🤗](https://huggingface.co/cs-giung/Qwen2.5-0.5B-Distilled/tree/main/32k/Qwen2.5-Coder-3B-Instruct_KD_FKL/epoch_0/final_model) [📉](checkpoints/code/Qwen2.5-0.5B/32k/Qwen2.5-Coder-3B-Instruct_KD_FKL/log.png)
| KD_RKL                                | 38.4 / 32.9 | 44.2 / 38.1 | 38.4 | [🤗](https://huggingface.co/cs-giung/Qwen2.5-0.5B-Distilled/tree/main/32k/Qwen2.5-Coder-3B-Instruct_KD_RKL/epoch_3/final_model) [📉](checkpoints/code/Qwen2.5-0.5B/32k/Qwen2.5-Coder-3B-Instruct_KD_RKL/log.png)
| OD_FKL                                | 36.0 / 32.3 | 42.1 / 35.7 | 36.5 | [🤗](https://huggingface.co/cs-giung/Qwen2.5-0.5B-Distilled/tree/main/32k/Qwen2.5-Coder-3B-Instruct_OD_FKL/epoch_3/final_model) [📉](checkpoints/code/Qwen2.5-0.5B/32k/Qwen2.5-Coder-3B-Instruct_OD_FKL/log.png)
| OD_RKL                                | 36.6 / 30.5 | 41.3 / 34.9 | 35.8 | [🤗](https://huggingface.co/cs-giung/Qwen2.5-0.5B-Distilled/tree/main/32k/Qwen2.5-Coder-3B-Instruct_OD_RKL/epoch_1/final_model) [📉](checkpoints/code/Qwen2.5-0.5B/32k/Qwen2.5-Coder-3B-Instruct_OD_RKL/log.png)
||
| Qwen2.5-Coder-1.5B-Instruct (Teacher) | 69.5 / 63.4 | 68.8 / 59.0 | 65.2 |
| KD_FKL                                | 35.4 / 31.1 | 42.6 / 35.4 | 36.1 | [🤗](https://huggingface.co/cs-giung/Qwen2.5-0.5B-Distilled/tree/main/32k/Qwen2.5-Coder-1.5B-Instruct_KD_FKL/epoch_2/final_model) [📉](checkpoints/code/Qwen2.5-0.5B/32k/Qwen2.5-Coder-1.5B-Instruct_KD_FKL/log.png)
| KD_RKL                                | 33.5 / 28.0 | 40.5 / 34.7 | 34.2 | [🤗](https://huggingface.co/cs-giung/Qwen2.5-0.5B-Distilled/tree/main/32k/Qwen2.5-Coder-1.5B-Instruct_KD_RKL/epoch_2/final_model) [📉](checkpoints/code/Qwen2.5-0.5B/32k/Qwen2.5-Coder-1.5B-Instruct_KD_RKL/log.png)
| OD_FKL                                | 34.1 / 31.7 | 43.4 / 35.7 | 36.2 | [🤗](https://huggingface.co/cs-giung/Qwen2.5-0.5B-Distilled/tree/main/32k/Qwen2.5-Coder-1.5B-Instruct_OD_FKL/epoch_0/final_model) [📉](checkpoints/code/Qwen2.5-0.5B/32k/Qwen2.5-Coder-1.5B-Instruct_OD_FKL/log.png)
| OD_RKL                                | 31.7 / 28.7 | 42.1 / 35.4 | 34.5 | [🤗](https://huggingface.co/cs-giung/Qwen2.5-0.5B-Distilled/tree/main/32k/Qwen2.5-Coder-1.5B-Instruct_OD_RKL/epoch_0/final_model) [📉](checkpoints/code/Qwen2.5-0.5B/32k/Qwen2.5-Coder-1.5B-Instruct_OD_RKL/log.png)

### Math
```bash

```

| model                                | GSM8K       | Minerva     | MATH500     |
| :-                                   | :-:         | :-:         | :-:         |
| Qwen2.5-Math-1.5B-Instruct (Teacher) | 75.2 / 74.4 | 18.9 / 53.2 | 17.8 / 53.4 | 
| Qwen2.5-0.5B (Student)               | 30.5 / 30.4 | 10.4 / 16.5 | 11.0 / 18.4 |
||
| SFT                                  | 43.6 / 43.6 | 0.00 / 19.0 | 0.00 / 19.6 |
||
| KD_FKL                               |
| KD_RKL                               |
| OD_FKL                               |
| OD_RKL                               |

## Pre-trained Models

### Code
```bash
export MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
evalplus.evaluate --model $MODEL --root $MODEL --dataset humaneval,mbpp --backend vllm --tp 1 --greedy
```

| model                       | HumanEval   | MBPP        |
| :-                          | :-          | :-          |
| Qwen2.5-Coder-14B-Instruct  | 91.5 / 86.0 | 
| Qwen2.5-Coder-7B-Instruct   | 90.9 / 83.5 | 82.8 / 71.7 |
| Qwen2.5-Coder-3B-Instruct   | 84.8 / 79.3 | 75.7 / 64.3 |
| Qwen2.5-Coder-1.5B-Instruct | 69.5 / 63.4 | 68.8 / 59.0 |
| Qwen2.5-Coder-0.5B-Instruct | 59.8 / 55.5 |
| Qwen2.5-7B-Instruct         | 82.3 / 73.2 | 78.8 / 67.5 |
| Qwen2.5-3B-Instruct         | 74.4 / 67.7 | 
| Qwen2.5-1.5B-Instruct       | 54.9 / 49.4 | 65.3 / 56.6 |
| Qwen2.5-0.5B-Instruct       | 37.8 / 32.3 | 49.5 / 42.1 |
| Qwen2.5-1.5B (init)         | 51.2 / 46.3 | 59.0 / 48.9 |
| Qwen2.5-1.5B                | 37.2 / 31.1 | 60.6 / 50.3 |
| Qwen2.5-0.5B (init)         | 29.9 / 26.8 | 45.2 / 37.6 |
| Qwen2.5-0.5B                | 30.5 / 26.2 | 41.0 / 35.2 |

### Math
```bash
export MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct"
lm_eval --model vllm --model_args pretrained="$MODEL",tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 --tasks minerva_math,minerva_math500,gsm8k --batch_size 1 --apply_chat_template --fewshot_as_multiturn --gen_kwargs max_gen_toks=2048
```

| model                      | GSM8K       | Minerva     | MATH500     |
| :-                         | :-          | :-          | :-          |
| Qwen2.5-Math-7B-Instruct   | 91.4 / 89.2 | 44.2 / 80.4 | 39.8 / 80.6 |
| Qwen2.5-Math-1.5B-Instruct | 75.2 / 74.4 | 18.9 / 53.2 | 17.8 / 53.4 |
| Qwen2.5-1.5B               | 61.9 / 61.4 | 28.0 / 30.5 | 30.0 / 32.8 |
| Qwen2.5-1.5B (init)        | 61.3 / 61.6 | 25.3 / 32.1 | 22.0 / 30.4 |
| Qwen2.5-0.5B               | 35.0 / 34.6 | 11.2 / 17.2 | 10.2 / 15.6 |
| Qwen2.5-0.5B (init)        | 30.5 / 30.4 | 10.4 / 16.5 | 11.0 / 18.4 |
