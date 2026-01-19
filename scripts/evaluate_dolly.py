import argparse
import json
import os
import evaluate
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--use_alpaca_prompt", action="store_true")
    args = parser.parse_args()

    # Load Data
    print(f"Loading test data from {args.data_path}")
    try:
        test_data = load_jsonl(args.data_path)
    except:
        # monkey-patch
        print("ROUGE-L: \\NA")
        return False

    references = []
    for item in test_data:
        response = item["response"]
        if isinstance(response, list):
            references.append(response)
        else:
            references.append([response])
    generated_texts = []

    # Check if output_file exists and try to load predictions
    if args.output_file and os.path.exists(args.output_file):
        print(f"Output file {args.output_file} exists. Attempting to load...")
        try:
            with open(args.output_file, "r") as f:
                saved_data = [json.loads(line) for line in f]

            # Basic validation: check length and presence of generated_text
            if len(saved_data) == len(test_data) and "generated_text" in saved_data[0]:
                generated_texts = [item["generated_text"] for item in saved_data]
                print("Loaded predictions from file. Skipping generation.")
            else:
                print(
                    "Output file content format mismatch or incomplete. Regenerating..."
                )
        except Exception as e:
            print(f"Error loading output file: {e}. Regenerating...")

    if not generated_texts:
        prompts = []
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        for item in test_data:
            inst = item["instruction"]
            inp = item.get("input", "")

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
                ]
            else:
                user_content = inst
                if inp:
                    user_content += f"\n\n{inp}"
                messages = [{"role": "user", "content": user_content}]

            # Check if chat_template exists
            if tokenizer.chat_template is None:
                raise AssertionError
            else:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            prompts.append(prompt)

        # vLLM
        print(f"Loading vLLM model: {args.model}")
        llm = LLM(
            model=args.model,
            tensor_parallel_size=1,
            trust_remote_code=True,
            dtype="bfloat16",
        )

        sampling_params = SamplingParams(temperature=0.0, max_tokens=256)  # Greedy

        print("Generating...")
        outputs = llm.generate(prompts, sampling_params)

        generated_texts = [o.outputs[0].text.strip() for o in outputs]

        # Save predictions to output_file
        if args.output_file:
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            output_data = []
            for i, item in enumerate(test_data):
                new_item = item.copy()
                new_item["generated_text"] = generated_texts[i]
                output_data.append(new_item)

            with open(args.output_file, "w") as f:
                for item in output_data:
                    f.write(json.dumps(item) + "\n")
            print(f"Saved predictions to {args.output_file}")

    # Evaluate
    print("Computing ROUGE...")
    rouge = evaluate.load("rouge", seed=42)
    results = rouge.compute(predictions=generated_texts, references=references)

    print("Computing BLEU and GLEU...")
    bleu = evaluate.load("bleu")
    google_bleu = evaluate.load("google_bleu")

    # BLEU and GLEU expect list of lists for references
    # references is already normalized to list of lists
    bleu_results = bleu.compute(predictions=generated_texts, references=references)
    google_bleu_results = google_bleu.compute(
        predictions=generated_texts, references=references
    )

    results["bleu"] = bleu_results["bleu"]
    results["google_bleu"] = google_bleu_results["google_bleu"]

    print("--------------------------------------------------")
    print(f"Model: {args.model}")
    print(f"Data: {args.data_path}")
    print(f"ROUGE-1: {results['rouge1']:.4f}")
    print(f"ROUGE-2: {results['rouge2']:.4f}")
    print(f"ROUGE-L: {results['rougeL']:.4f}")
    print(f"BLEU: {results['bleu']:.4f}")
    print(f"GLEU: {results['google_bleu']:.4f}")
    print("--------------------------------------------------")

    if args.output_file:
        # Save evaluation results to a separate file
        results_file = args.output_file.replace(".jsonl", "") + "_eval_results.json"
        if results_file == args.output_file:
            results_file = args.output_file + "_eval_results.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved eval results to {results_file}")


if __name__ == "__main__":
    main()
