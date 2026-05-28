import os
import json
import torch
import argparse
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm
from peft import PeftModel, PeftConfig

from utils import *

def evaluate_model(
    model_name_or_path: str,
    dataset_path: str,
    temperature: float,
    batch_size: int = 4,
    num_samples: int = None,
    save_results: bool = True,
):
    print(f"Loading tokenizer and model for {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto"
    if torch.backends.mps.is_available():
        device_map = "mps"
        print("Using MPS (Metal Performance Shaders) for Mac.")
    elif not torch.cuda.is_available():
        device_map = "cpu"
        print("CUDA/MPS not found. Using CPU. This will be very slow.")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    except OSError as e:
        if "does not appear to have a file named" in str(e):
            print(f"[{model_name_or_path}] appears to be a LoRA adapter, not a full model.")
            config = PeftConfig.from_pretrained(model_name_or_path)
            base_model_path = config.base_model_name_or_path
            print(f"Loading base model [{base_model_path}] first...")

            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map=device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            print(f"Applying LoRA adapter [{model_name_or_path}]...")
            model = PeftModel.from_pretrained(model, model_name_or_path)
        else:
            raise

    model.eval()

    if dataset_path == "openai/gsm8k":
        dataset = load_dataset('openai/gsm8k', 'main')['test']
    elif dataset_path.endswith('.parquet'):
        dataset = load_dataset('parquet', data_files=dataset_path)['train']
    elif dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=dataset_path)['train']
    else:
        dataset = load_dataset(dataset_path)['train']

    if num_samples and len(dataset) > num_samples:
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
    total_samples = len(dataset)
    print(f"Loaded {total_samples} samples from {dataset_path}")

    results = []
    correct = 0
    total = 0

    progress_bar = tqdm(
        total=total_samples,
        desc="Processing samples",
        unit="examples",
        dynamic_ncols=True,
    )
    progress_bar.set_postfix({'acc': '0.00%', 'correct': '0'})

    # Process samples in batches
    for i in range(0, total_samples, batch_size):
        batch_data = dataset[i:i + batch_size]
        current_batch_size = len(batch_data['question'])

        # Prepare prompts
        prompts = [
            [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': q.strip()},
            ]
            for q in batch_data['question']
        ]

        # Convert chat prompts to the required format
        formatted_prompts = [
            tokenizer.apply_chat_template(
                p,
                tokenize=False,
                add_generation_prompt=True
            )
            for p in prompts
        ]

        prompt_inputs = tokenizer(
            formatted_prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        ).to(model.device)

        prompt_length = prompt_inputs["input_ids"].size(1)

        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                **prompt_inputs,
                generation_config=GenerationConfig(
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else 1.0,
                    max_new_tokens=2048,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            )

        # Process each generated response
        for j, output in enumerate(outputs):
            response = tokenizer.decode(output[prompt_length:], skip_special_tokens=True)

            # Extract the generated answer
            extracted = extract_from_response(response)
            generated_answer = process_gsm8k_answer(extracted)

            # Extract true answer
            raw_answer = batch_data['answer'][j] if 'answer' in batch_data and batch_data['answer'][j] is not None else None

            if raw_answer is not None:
                true_answer = extract_hash_answer(raw_answer)
                true_answer = process_gsm8k_answer(true_answer)
            elif 'answer_number' in batch_data and batch_data['answer_number'][j] is not None:
                true_answer = str(batch_data['answer_number'][j])
                true_answer = process_gsm8k_answer(true_answer)
            else:
                true_answer = ""

            is_correct = (generated_answer == true_answer)

            # Store the result
            result = {
                'question': batch_data['question'][j],
                'true_answer': true_answer,
                'generated_answer': generated_answer,
                'full_response': response,
                'correct': is_correct
            }
            results.append(result)

            if is_correct:
                correct += 1
            total += 1

        progress_bar.update(current_batch_size)
        progress_bar.set_postfix({
            'acc': f'{(correct/total)*100:.2f}%',
            'correct': f'{correct}/{total}',
        })

    progress_bar.close()
    accuracy = correct / total if total > 0 else 0
    metrics = {
        'model': model_name_or_path,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'temperature': temperature,
        'timestamp': datetime.now().isoformat()
    }

    if save_results:
        safe_model_name = model_name_or_path.replace("/", "_")
        save_path = f"eval_results_{safe_model_name}.json"
        with open(save_path, 'w') as f:
            json.dump({'metrics': metrics, 'results': results}, f, indent=2)
        print(f"\nResults saved to {save_path}")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--dataset", type=str, default="openai/gsm8k", help="Hugging Face dataset ID or local path to a .parquet/.jsonl file")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation (0.0 for greedy)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (for quick test)")
    args = parser.parse_args()

    print(f"Starting evaluation of {args.model} on {args.dataset}")
    evaluate_model(
        model_name_or_path=args.model,
        dataset_path=args.dataset,
        temperature=args.temperature,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
    )
