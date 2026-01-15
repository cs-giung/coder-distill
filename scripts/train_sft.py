import argparse
import json
import os
import socket
from datetime import datetime

import deepspeed
import torch
import torch.distributed as dist
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
    parser.add_argument("--data_jsonl", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-05)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=2048)
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


class SFTDataset(Dataset):
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
    micro_batch_size = 1
    gradient_accumulation_steps = args.batch_size // (world_size * micro_batch_size)

    if rank == 0:
        print_fn(
            f"world_size={world_size}, "
            f"micro_batch_size={micro_batch_size}, "
            f"gradient_accumulation_steps={gradient_accumulation_steps}",
            args.output_dir,
        )

    # ------------------------------------------------------------------------ #
    # Load and Patch DeepSpeed Config
    # ------------------------------------------------------------------------ #
    with open("ds_configs/ds_config_student.json", "r") as f:
        ds_config = json.load(f)

    # Patch Config
    ds_config["train_batch_size"] = (
        micro_batch_size * gradient_accumulation_steps * world_size
    )
    ds_config["train_micro_batch_size_per_gpu"] = micro_batch_size
    ds_config["gradient_accumulation_steps"] = gradient_accumulation_steps

    # ------------------------------------------------------------------------ #
    # Model Setup
    # ------------------------------------------------------------------------ #
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.resize_token_embeddings(len(tokenizer))

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
        model = get_peft_model(model, peft_config)
    else:
        model.gradient_checkpointing_enable()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=0.0,
        fused=True,
    )

    # ------------------------------------------------------------------------ #
    # DeepSpeed Initialization
    # ------------------------------------------------------------------------ #
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config=ds_config
    )

    # Load Checkpoint if requested
    if args.resume_from_checkpoint:
        if rank == 0:
            print_fn(
                f"Resuming from checkpoint: {args.resume_from_checkpoint}",
                args.output_dir,
            )
        load_path, _ = model_engine.load_checkpoint(
            os.path.dirname(args.resume_from_checkpoint),
            tag="final_state",
            load_module_strict=False,
        )

    # ------------------------------------------------------------------------ #
    # Data Setup
    # ------------------------------------------------------------------------ #
    raw_data = load_jsonl(args.data_jsonl, limit=args.limit, rank=rank)

    if rank == 0:
        print_fn(f"Loaded {len(raw_data)} samples.", args.output_dir)

    dataset = SFTDataset(raw_data)
    sampler = DistributedSampler(dataset)
    dloader = DataLoader(
        dataset, batch_size=micro_batch_size, sampler=sampler, shuffle=False
    )  # Shuffle handled by sampler

    # ------------------------------------------------------------------------ #
    # Training Loop
    # ------------------------------------------------------------------------ #
    def process_batch(instructions, inputs_data, responses):
        # 1. Prepare full text for input
        full_texts = []

        if args.use_alpaca_prompt:
            PROMPT_NO_INPUT = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            PROMPT_INPUT = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."

            for inst, inp, resp in zip(instructions, inputs_data, responses):
                sys_msg = PROMPT_INPUT if inp else PROMPT_NO_INPUT
                user_content = inst
                if inp:
                    user_content += f"\n\n### Input:\n{inp}"

                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": resp},
                ]
                full_texts.append(
                    tokenizer.apply_chat_template(messages, tokenize=False)
                )
        else:
            for inst, inp, resp in zip(instructions, inputs_data, responses):
                user_content = inst
                if inp:
                    user_content += f"\n\n{inp}"
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": resp},
                ]
                full_texts.append(
                    tokenizer.apply_chat_template(messages, tokenize=False)
                )

        inputs = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=args.max_length,
            truncation=True,
        ).to(model_engine.device)

        # 2. Prepare labels with masking
        # Instructions only (for masking)
        prompt_only_texts = []
        if args.use_alpaca_prompt:
            for inst, inp in zip(instructions, inputs_data):
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
            for inst, inp in zip(instructions, inputs_data):
                user_content = inst
                if inp:
                    user_content += f"\n\n{inp}"
                messages = [{"role": "user", "content": user_content}]
                # add_generation_prompt=True ensures we get the prompt part exactly as it appears before generation
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt_only_texts.append(text)

        prompt_inputs = tokenizer(
            prompt_only_texts, return_tensors="pt", padding=False, truncation=True
        )

        labels = inputs.input_ids.clone()

        # Mask out padding and instructions
        for b in range(labels.shape[0]):
            # Mask padding
            # padding_side="left", so padding is at the beginning
            # attention_mask == 0 -> padding
            pad_mask = inputs.attention_mask[b] == 0
            labels[b, pad_mask] = -100  # Ignore padding in loss

            # Mask instruction
            # Find the length of the prompt in tokens.
            # Note: prompt_inputs might not align perfectly if left-padding vs no-padding changes tokenization slightly
            # (usually fine for sentencepiece, but we need to be careful).
            # A robust way is to measure length of prompt tokens.
            p_len = len(prompt_inputs.input_ids[b])

            # Since inputs are left-padded, the actual content starts after the padding.
            non_pad_len = torch.sum(inputs.attention_mask[b]).item()
            pad_len = inputs.input_ids.shape[1] - non_pad_len

            # The prompt corresponds to the first p_len tokens valid tokens?
            # Wait. prompt_inputs was tokenized without padding.
            # So its length is the pure length of the prompt.
            # In 'inputs', the prompt starts at index 'pad_len'.

            # Mask [pad_len : pad_len + p_len]
            start_response = pad_len + p_len

            if start_response < labels.shape[1]:
                labels[b, :start_response] = -100
            else:
                # Should not happen if response exists, but safety.
                labels[b, :] = -100

        # 3. Forward Pass
        outputs = model_engine(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels,
            use_cache=False,
        )
        return outputs.loss

    model_engine.train()

    start_time = datetime.now()
    if rank == 0:
        print_fn(f"Training started at {start_time}", args.output_dir)

    global_step = 0
    if args.resume_from_checkpoint:
        global_step = model_engine.global_steps

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
            instructions, inputs_data, responses = batch_data

            loss = process_batch(instructions, inputs_data, responses)
            model_engine.backward(loss)
            model_engine.step()

            # Logging logic
            current_global_step = model_engine.global_steps
            if current_global_step > global_step:
                global_step = current_global_step
                if rank == 0:
                    log_str = f"[{datetime.now() - start_time}] Step {global_step}"
                    log_str += f" | Loss: {loss.item():.4f}"
                    print_fn(log_str, args.output_dir)

    # Save
    if rank == 0:
        print_fn("Training finished. Saving...", args.output_dir)

    model_engine.save_checkpoint(args.output_dir, tag="final_state")
    if rank == 0:
        print_fn(
            f"Saved DeepSpeed checkpoint to {os.path.join(args.output_dir, 'final_state')}",
            args.output_dir,
        )

    if rank == 0:
        final_model_path = os.path.join(args.output_dir, "final_model")
        unwrapped_model = model_engine.module
        unwrapped_model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print_fn(f"Saved HF model to {final_model_path}", args.output_dir)

    dist.barrier()


if __name__ == "__main__":
    main()
