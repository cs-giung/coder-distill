import argparse
import json
import math
import os
import socket
from datetime import datetime

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

progress_console = Console(stderr=True, force_terminal=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_jsonl", type=str, required=True)
    parser.add_argument("--student_jsonl", type=str, required=True)
    parser.add_argument("--teacher_model", type=str, required=True)
    parser.add_argument("--student_model", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-05)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--jsd_beta", type=float, default=0.9)
    parser.add_argument("--jsd_lambda", type=float, default=1.0)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--use_alpaca_prompt", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["all-linear"],
        help="List of module names or 'all-linear'",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )

    # Include DeepSpeed arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


class GeneratedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def print_fn(log, output_dir):
    if dist.get_rank() == 0:
        log_msg = f"{datetime.now()}: {log}\n"
        print(log_msg, end="", flush=True)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "console.log"), "a") as f:
            f.write(log_msg)


def load_jsonl(path, limit=None, rank=0):
    data = []
    if path and os.path.exists(path):
        if rank == 0:
            print(f"Loading data: {path}")
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                inp = item.get("input", "")
                data.append((item["instruction"], inp, item["response"]))
        if limit:
            data = data[:limit]
    return data


def main():
    args = get_args()

    # Initialize Distributed
    # Get local rank and set device to prevent memory imbalance (context on GPU 0)
    if args.local_rank == -1:
        args.local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)

    # Fallback for single-process execution
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        os.environ["MASTER_PORT"] = str(port)
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"

    deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False)

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Calculate Gradient Accumulation Steps
    need_backward_loss_s = args.jsd_lambda > 0
    need_backward_loss_t = args.jsd_lambda < 1
    assert need_backward_loss_s or need_backward_loss_t

    micro_batch_size = args.micro_batch_size
    gradient_accumulation_steps = max(
        1, args.batch_size // (world_size * micro_batch_size)
    )
    if need_backward_loss_s and need_backward_loss_t:
        gradient_accumulation_steps *= 2

    if rank == 0:
        print_fn(
            f"world_size={world_size}, "
            f"micro_batch_size={micro_batch_size}, "
            f"gradient_accumulation_steps={gradient_accumulation_steps}",
            args.output_dir,
        )

    # ------------------------------------------------------------------------ #
    # Load and Patch DeepSpeed Configs
    # ------------------------------------------------------------------------ #
    with open("ds_configs/ds_config_student.json", "r") as f:
        student_ds_config = json.load(f)
    with open("ds_configs/ds_config_teacher.json", "r") as f:
        teacher_ds_config = json.load(f)

    # Patch Student Config
    student_ds_config["train_batch_size"] = (
        micro_batch_size * gradient_accumulation_steps * world_size
    )
    student_ds_config["train_micro_batch_size_per_gpu"] = micro_batch_size
    student_ds_config["gradient_accumulation_steps"] = gradient_accumulation_steps

    # Patch Teacher Config
    teacher_ds_config["train_batch_size"] = (
        micro_batch_size * gradient_accumulation_steps * world_size
    )
    teacher_ds_config["train_micro_batch_size_per_gpu"] = micro_batch_size
    teacher_ds_config["gradient_accumulation_steps"] = gradient_accumulation_steps

    # ------------------------------------------------------------------------ #
    # Model Setup
    # ------------------------------------------------------------------------ #
    # NOTE: make sure the tokenizer is shared across student and teacher models
    assert (
        AutoTokenizer.from_pretrained(args.teacher_model).vocab
        == AutoTokenizer.from_pretrained(args.student_model).vocab
    )
    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Student Model
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    student_model.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules[0]
            if len(args.lora_target_modules) == 1
            and args.lora_target_modules[0] == "all-linear"
            else args.lora_target_modules,
        )
        student_model = get_peft_model(student_model, peft_config)
    else:
        student_model.gradient_checkpointing_enable()

    # Teacher Model
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    teacher_model.resize_token_embeddings(len(tokenizer))
    for p in teacher_model.parameters():
        p.requires_grad = False

    # Optimizer
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=0.0,
        fused=True,
    )

    # ------------------------------------------------------------------------ #
    # DeepSpeed Initialization
    # ------------------------------------------------------------------------ #
    student_engine, optimizer, _, _ = deepspeed.initialize(
        model=student_model, optimizer=optimizer, config=student_ds_config
    )
    teacher_engine, _, _, _ = deepspeed.initialize(
        model=teacher_model, config=teacher_ds_config
    )

    # Load Checkpoint if requested
    if args.resume_from_checkpoint:
        if rank == 0:
            print_fn(
                f"Resuming from checkpoint: {args.resume_from_checkpoint}",
                args.output_dir,
            )
        load_path, _ = student_engine.load_checkpoint(
            os.path.dirname(args.resume_from_checkpoint),
            tag="final_state",
            load_module_strict=False,
        )

    # ------------------------------------------------------------------------ #
    # Data Setup
    # ------------------------------------------------------------------------ #
    combined_data = [
        {
            "instruction": s_ins,
            "input": s_inp,
            "student_response": s_res,
            "teacher_response": t_res,
        }
        for (s_ins, s_inp, s_res), (t_ins, t_inp, t_res) in zip(
            load_jsonl(args.student_jsonl, limit=args.limit, rank=rank),
            load_jsonl(args.teacher_jsonl, limit=args.limit, rank=rank),
        )
        if s_ins == t_ins and s_inp == t_inp
    ]

    if rank == 0:
        print_fn(f"Loaded {len(combined_data)} combined samples.", args.output_dir)

    dataset = GeneratedDataset(combined_data)
    sampler = DistributedSampler(dataset)
    dloader = DataLoader(
        dataset, batch_size=micro_batch_size, sampler=sampler, shuffle=False
    )  # Shuffle handled by sampler

    # ------------------------------------------------------------------------ #
    # Training Loop
    # ------------------------------------------------------------------------ #
    def process_batch_chunk(
        prompts, source_label, instructions_for_mask, inputs_for_mask
    ):
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=args.max_length,
            truncation=True,
        ).to(student_engine.device)  # use engine device

        # Loss Mask logic (same as legacy)
        prompt_only_texts = []

        if args.use_alpaca_prompt:
            PROMPT_NO_INPUT = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            PROMPT_INPUT = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."

            for inst, inp in zip(instructions_for_mask, inputs_for_mask):
                sys_msg = PROMPT_INPUT if inp else PROMPT_NO_INPUT
                user_content = inst
                if inp:
                    user_content += f"\n\n### Input:\n{inp}"

                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_content},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt_only_texts.append(text)
        else:
            for inst, inp in zip(instructions_for_mask, inputs_for_mask):
                user_content = inst
                if inp:
                    user_content += f"\n\n{inp}"
                messages = [{"role": "user", "content": user_content}]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt_only_texts.append(text)

        prompt_inputs = tokenizer(
            prompt_only_texts, return_tensors=None, padding=False, truncation=True
        )

        loss_mask = torch.zeros_like(inputs.input_ids, dtype=torch.float32)

        for b in range(inputs.input_ids.shape[0]):
            p_len = len(prompt_inputs["input_ids"][b])
            non_pad_len = torch.sum(inputs.attention_mask[b]).item()
            pad_len = inputs.input_ids.shape[1] - non_pad_len
            start_response = pad_len + p_len
            if start_response < inputs.input_ids.shape[1]:
                loss_mask[b, start_response:] = 1.0

        loss_mask = loss_mask * inputs.attention_mask.float()
        loss_mask = loss_mask.to(student_engine.device)

        # Forward pass Teacher
        with torch.no_grad():
            teacher_outputs = teacher_engine(**inputs)
            teacher_logits = teacher_outputs.logits

        # Forward pass Student
        student_outputs = student_engine(**inputs)
        student_logits = student_outputs.logits

        teacher_logits_shifted = teacher_logits[:, :-1, :]
        student_logits_shifted = student_logits[:, :-1, :]
        loss_mask_shifted = loss_mask[:, 1:]

        # Shifted probabilities
        teacher_probs = F.softmax(teacher_logits_shifted, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits_shifted, dim=-1)
        student_log_probs = F.log_softmax(student_logits_shifted, dim=-1)
        student_probs = F.softmax(student_logits_shifted, dim=-1)

        fwd_kl_pt = torch.sum(
            teacher_probs * (teacher_log_probs - student_log_probs), dim=-1
        )
        rev_kl_pt = torch.sum(
            student_probs * (student_log_probs - teacher_log_probs), dim=-1
        )

        if source_label == "student":
            jsd_coeff = args.jsd_lambda
        elif source_label == "teacher":
            jsd_coeff = 1 - args.jsd_lambda

        mixed_log_probs = torch.logsumexp(
            torch.stack(
                [
                    math.log(args.jsd_beta) + teacher_log_probs,
                    math.log(1 - args.jsd_beta) + student_log_probs,
                ],
                dim=0,
            ),
            dim=0,
        )
        total_loss_map = args.jsd_beta * torch.sum(
            teacher_probs * (teacher_log_probs - mixed_log_probs), dim=-1
        ) + (1 - args.jsd_beta) * torch.sum(
            student_probs * (student_log_probs - mixed_log_probs), dim=-1
        )

        total_loss_map = total_loss_map * jsd_coeff
        masked_loss = total_loss_map * loss_mask_shifted
        num_valid_tokens = torch.sum(loss_mask_shifted)
        loss = torch.sum(masked_loss) / (num_valid_tokens + 1e-9)

        return loss, fwd_kl_pt, rev_kl_pt, loss_mask_shifted

    student_engine.train()
    teacher_engine.eval()

    start_time = datetime.now()
    if rank == 0:
        print_fn(f"Training started at {start_time}", args.output_dir)

    global_step = 0
    if args.resume_from_checkpoint:
        global_step = student_engine.global_steps

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        BarColumn(bar_width=30),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        SpinnerColumn(),
        console=progress_console,
        disable=rank != 0,
    ) as progress:
        for step, batch_data in progress.track(
            enumerate(dloader),
            description="Training",
            total=len(dloader),
        ):
            instructions = batch_data["instruction"]
            inputs_data = batch_data["input"]
            student_resps = batch_data["student_response"]
            teacher_resps = batch_data["teacher_response"]

            def build_prompts(resps):
                prompts = []
                for inst, inp, resp in zip(instructions, inputs_data, resps):
                    if args.use_alpaca_prompt:
                        PROMPT_NO_INPUT = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
                        PROMPT_INPUT = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
                        sys_msg = PROMPT_INPUT if inp else PROMPT_NO_INPUT
                        user_content = inst
                        if inp:
                            user_content += f"\n\n### Input:\n{inp}"

                        messages = [
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": resp},
                        ]
                    else:
                        user_content = inst
                        if inp:
                            user_content += f"\n\n{inp}"
                        messages = [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": resp},
                        ]
                    prompts.append(
                        tokenizer.apply_chat_template(messages, tokenize=False)
                    )
                return prompts

            student_prompts = build_prompts(student_resps)
            teacher_prompts = build_prompts(teacher_resps)

            is_log_boundary = (step + 1) % gradient_accumulation_steps == 0
            do_process_s = need_backward_loss_s or is_log_boundary
            do_process_t = need_backward_loss_t or is_log_boundary

            # --- Micro-Step 1: Student Data ---
            if do_process_s:
                with torch.set_grad_enabled(need_backward_loss_s):
                    loss_s, fwd_s, rev_s, mask_s = process_batch_chunk(
                        student_prompts, "student", instructions, inputs_data
                    )
                if need_backward_loss_s:
                    student_engine.backward(loss_s)
                    student_engine.step()

            # --- Micro-Step 2: Teacher Data ---
            if do_process_t:
                with torch.set_grad_enabled(need_backward_loss_t):
                    loss_t, fwd_t, rev_t, mask_t = process_batch_chunk(
                        teacher_prompts, "teacher", instructions, inputs_data
                    )
                if need_backward_loss_t:
                    student_engine.backward(loss_t)
                    student_engine.step()

            # Logging logic
            current_global_step = student_engine.global_steps
            if current_global_step > global_step:
                global_step = current_global_step
                # Log
                if rank == 0:

                    def mean_metric(val_map, mask):
                        valid_mask = mask > 0
                        if valid_mask.sum() > 0:
                            return (
                                val_map * valid_mask.float()
                            ).sum() / valid_mask.sum()
                        return 0.0

                    m_fwd_s = mean_metric(fwd_s, mask_s).item()
                    m_rev_s = mean_metric(rev_s, mask_s).item()
                    m_fwd_t = mean_metric(fwd_t, mask_t).item()
                    m_rev_t = mean_metric(rev_t, mask_t).item()

                    log_str = f"[{datetime.now() - start_time}] Step {global_step}"
                    log_str += f" | Loss(S): {loss_s.item():.3e}"
                    log_str += f" | Loss(T): {loss_t.item():.3e}"
                    log_str += f" | fwd_S: {m_fwd_s:.3e} | rev_S: {m_rev_s:.3e}"
                    log_str += f" | fwd_T: {m_fwd_t:.3e} | rev_T: {m_rev_t:.3e}"
                    print_fn(log_str, args.output_dir)

    # Save
    if rank == 0:
        print_fn("Training finished. Saving...", args.output_dir)

    # 1. Save DeepSpeed Checkpoint (for resuming)
    # This saves to args.output_dir/final_state
    student_engine.save_checkpoint(args.output_dir, tag="final_state")
    if rank == 0:
        print_fn(
            f"Saved DeepSpeed checkpoint to {os.path.join(args.output_dir, 'final_state')}",
            args.output_dir,
        )

    # 2. Save HF Model (for evaluation/generation)
    # Ensure all processes verify save is done if needed, but save_pretrained is usually rank 0 only or handles it.
    # But wait, student_engine is wrapped.
    if rank == 0:
        final_model_path = os.path.join(args.output_dir, "final_model")
        # Unwrapped model via DeepSpeed
        # For ZeRO-2, module access is fine.
        unwrapped_model = student_engine.module
        unwrapped_model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print_fn(f"Saved HF model to {final_model_path}", args.output_dir)

    # Wait for all processes to ensure saving is complete before exiting (important for loop scripts)
    dist.barrier()


if __name__ == "__main__":
    main()
