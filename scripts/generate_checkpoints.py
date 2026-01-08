from transformers import AutoTokenizer, AutoModelForCausalLM


if __name__ == "__main__":
    model = "Qwen2.5-0.5B"
    # model = "Qwen2.5-1.5B"

    math_tokenizer = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    hf_math_tokenizer = AutoTokenizer.from_pretrained(math_tokenizer)
    hf_math_tokenizer.save_pretrained(f"checkpoints/math/{model}/init")
    hf_model = AutoModelForCausalLM.from_pretrained(f"Qwen/{model}")
    hf_model.resize_token_embeddings(len(hf_math_tokenizer))
    hf_model.save_pretrained(f"checkpoints/math/{model}/init")

    code_tokenizer = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    hf_code_tokenizer = AutoTokenizer.from_pretrained(code_tokenizer)
    hf_code_tokenizer.save_pretrained(f"checkpoints/code/{model}/init")
    hf_model = AutoModelForCausalLM.from_pretrained(f"Qwen/{model}")
    hf_model.resize_token_embeddings(len(hf_code_tokenizer))
    hf_model.save_pretrained(f"checkpoints/code/{model}/init")
