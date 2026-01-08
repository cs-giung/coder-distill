import argparse
import gzip
import json
import os

import vllm
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest


def write_jsonl(filename: str, data, append: bool = False):
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/32k/code.jsonl")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Check for LoRA
    adapter_path = None
    base_model_path = args.model
    enable_lora = False

    if os.path.exists(os.path.join(args.model, "adapter_config.json")):
        print(f"Detected LoRA adapter at {args.model}")
        adapter_path = args.model
        enable_lora = True
        # Read base model from config
        with open(os.path.join(args.model, "adapter_config.json"), "r") as f:
            t = json.load(f)
            base_model_path = t.get("base_model_name_or_path")
        print(f"Base model for LoRA: {base_model_path}")

    llm = vllm.LLM(
        model=base_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        dtype="bfloat16",
        enable_lora=enable_lora,
    )
    print(f"Loaded model {base_model_path} with vLLM (LoRA={enable_lora}).")

    instructions = []
    with open(args.input, "r") as fp:
        for line in fp:
            instructions.append(json.loads(line)["instruction"])
    if args.limit:
        instructions = instructions[: args.limit]

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    prompts = []
    for inst in instructions:
        messages = [{"role": "user", "content": inst}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)

    sampling_params = vllm.SamplingParams(
        top_p=args.top_p,
        min_p=args.min_p,
        temperature=args.temperature,
        max_tokens=1024,
        seed=args.seed,
    )
    print(f"Sampling params: {sampling_params}")
    print(f"Generating responses for {len(prompts)} prompts...")

    lora_request = None
    if enable_lora and adapter_path:
        lora_request = LoRARequest("adapter", 1, lora_path=adapter_path)

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    data_to_save = []
    for inst, output in zip(instructions, outputs):
        generated_text = output.outputs[0].text
        data_to_save.append(
            {
                "instruction": inst,
                "response": generated_text,
                "top_p": sampling_params.top_p,
                "min_p": sampling_params.min_p,
                "temperature": sampling_params.temperature,
            }
        )

    print(f"Saving {len(data_to_save)} records to {args.output}...")
    write_jsonl(args.output, data_to_save)
    print("Done.")


if __name__ == "__main__":
    main()
