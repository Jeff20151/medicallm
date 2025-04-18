#!/usr/bin/env python
import os
import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import argparse
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from pathlib import Path
# Import vllm
from vllm import LLM, SamplingParams

# Download necessary NLTK data
nltk.download('punkt_tab', quiet=True)

# Constants for prompt components
PROMPT_COMPONENT = {
    'chq': {
            'prefix': 'query',
            'suffix': 'summarized question',
            'instruction': 'summarize the patient health query into one question of 15 words or less',
           },
    'opi': {
            'prefix': 'finding',
            'suffix': 'impression',
            'instruction': 'summarize the radiology report findings into an impression with minimal text',
           },
    'd2n': {
            'prefix': 'patient/doctor dialogue',
            'suffix': 'assessment and plan',
            'instruction': 'summarize the patient/doctor dialogue into an assessment and plan',
           },
}

# Model paths
MODEL_PATHS = [
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "Henrychur/MMed-Llama-3-8B",
    "TsinghuaC3I/Llama-3-8B-UltraMedical",
    "/home/user/medicallm/merged_models/slerp_ultra_mmed",
    "/home/user/medicallm/merged_models/sce_deepseek_ultramed_mmed",  # The newly merged one
]

# Dataset paths
DATASET_PATHS = [
    "/home/user/medicallm/clin-summ/data/chq/test.jsonl",
    "/home/user/medicallm/clin-summ/data/d2n/test.jsonl",
    "/home/user/medicallm/clin-summ/data/opi/test.jsonl",
]

def load_data(dataset_path):
    """Load data from a JSONL file."""
    try:
        dataset_name = Path(dataset_path).parent.name
        
        # Load data
        with open(dataset_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        data = [json.loads(line) for line in lines]
        
        # Convert to DataFrames
        inputs_df = pd.DataFrame([{'inputs': item['inputs']} for item in data])
        targets_df = pd.DataFrame([{'target': item['target']} for item in data])
        
        return inputs_df, targets_df
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {e}")
        return None, None

def create_prompt(input_text, dataset_name):
    """Create a prompt for the model based on the dataset."""
    component = PROMPT_COMPONENT[dataset_name]
    prompt = f"Below is a {component['prefix']}:\n\n{input_text}\n\nPlease {component['instruction']}:\n"
    return prompt

def generate_text(model, tokenizer, prompt, device="cuda", max_new_tokens=128, use_vllm=False):
    """Generate text from a prompt using the model."""
    if use_vllm:
        # vllm LLM
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=max_new_tokens,
            top_p=1.0,
        )
        outputs = model.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        if prompt in generated_text:
            answer = generated_text[len(prompt):].strip()
        else:
            answer = generated_text
    else:
        # transformers
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
            "temperature": 1.0
        }
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                **gen_kwargs
            )
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if prompt in full_output:
            answer = full_output[len(prompt):].strip()
        else:
            answer = full_output
    return answer

def calculate_metrics(predictions, references):
    """Calculate ROUGE-L and BLEU scores."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1
    
    rouge_l_scores = []
    bleu_scores = []
    
    for pred, ref in zip(predictions, references):
        # Calculate ROUGE-L
        rouge_score = scorer.score(ref, pred)
        rouge_l_scores.append(rouge_score['rougeL'].fmeasure)
        
        # Calculate BLEU with error handling
        try:
            ref_tokens = nltk.word_tokenize(ref.lower())
            pred_tokens = nltk.word_tokenize(pred.lower())
            
            # Ensure we have some tokens to compare
            if len(ref_tokens) == 0:
                ref_tokens = ['dummy']
            if len(pred_tokens) == 0:
                pred_tokens = ['dummy']
                
            bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
            bleu_scores.append(bleu)
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
            bleu_scores.append(0.0)
    
    return {
        'ROUGE-L': sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0,
        'BLEU': sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    }

def evaluate_model(model_path, dataset_paths, n_samples=3, output_dir="comparison_results", device="cuda", use_vllm=True, use_int8=True):
    """Evaluate a model on multiple datasets."""
    print(f"\n=== Evaluating: {model_path} ===")
    start_time = time.time()
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Configure pad_token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if use_vllm:
            print("Using vllm for inference")
            # Load model with vllm and int8 quantization if specified
            # quantization = "int8" if use_int8 else None
            model = LLM(
                model=model_path,
                tensor_parallel_size=1,  # Adjust based on GPU count
                # quantization=quantization,
                trust_remote_code=True
            )
        else:
            print("Using transformers for inference")
            # Load model with transformers and bfloat16/int8 if specified
            if use_int8:
                print("Using int8 quantization")
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    quantization_config=quantization_config,
                    low_cpu_mem_usage=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            model.eval()
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return {'results': {}, 'evaluation_time': 0}
    
    model_results = {}
    model_name = Path(model_path).name if not model_path.endswith('/') else Path(model_path).parent.name
    
    for dataset_path in dataset_paths:
        dataset_name = Path(dataset_path).parent.name
        print(f"\nProcessing dataset: {dataset_name}")
        
        if dataset_name not in PROMPT_COMPONENT:
            print(f"Skipping {dataset_name}: not supported.")
            continue
        
        inputs_df, targets_df = load_data(dataset_path)
        if inputs_df is None or targets_df is None:
            continue
        
        # Limit samples for testing
        n_samples = min(n_samples, len(inputs_df))
        inputs_df = inputs_df.head(n_samples)
        targets_df = targets_df.head(n_samples)
        
        predictions = []
        references = []
        
        print(f"Generating predictions for {n_samples} samples...")
        for i, (_, row) in enumerate(tqdm(inputs_df.iterrows(), total=n_samples)):
            try:
                input_text = row['inputs']
                reference = targets_df.iloc[i]['target']
                
                prompt = create_prompt(input_text, dataset_name)
                prediction = generate_text(model, tokenizer, prompt, device=device, use_vllm=use_vllm)
                
                predictions.append(prediction.strip())
                references.append(reference.strip())
                
                # Save intermediate results
                model_output_dir = os.path.join(output_dir, model_name, dataset_name)
                os.makedirs(model_output_dir, exist_ok=True)
                
                with open(f"{model_output_dir}/sample_{i}.json", 'w') as f:
                    json.dump({
                        'input': input_text,
                        'prompt': prompt,
                        'reference': reference,
                        'prediction': prediction
                    }, f, indent=2)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        # Calculate metrics
        if predictions and references:
            metrics = calculate_metrics(predictions, references)
            model_results[dataset_name] = metrics
            
            print(f"Results for {dataset_name}:")
            print(f"ROUGE-L: {metrics['ROUGE-L']:.4f}")
            print(f"BLEU: {metrics['BLEU']:.4f}")
        else:
            print(f"No valid predictions for {dataset_name}")
            model_results[dataset_name] = {'ROUGE-L': 0, 'BLEU': 0}
    
    elapsed_time = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed_time:.2f} seconds")
    
    # Save summary results
    summary_dir = os.path.join(output_dir, model_name)
    os.makedirs(summary_dir, exist_ok=True)
    
    with open(f"{summary_dir}/metrics_summary.json", 'w') as f:
        json.dump({
            'model': model_path,
            'results': model_results,
            'evaluation_time': elapsed_time
        }, f, indent=2)
    
    return {
        'results': model_results,
        'evaluation_time': elapsed_time
    }

def plot_results(all_results, output_dir="comparison_results"):
    """Plot the comparative results of all models."""
    os.makedirs(output_dir, exist_ok=True)
    
    if not all_results:
        print("No results to plot")
        return
    
    # Get all dataset names
    dataset_names = []
    for model_results in all_results.values():
        for dataset in model_results['results'].keys():
            if dataset not in dataset_names:
                dataset_names.append(dataset)
    
    if not dataset_names:
        print("No datasets found in results")
        return
    
    # Get all model names
    model_names = [Path(model_path).name if not model_path.endswith('/') else Path(model_path).parent.name 
                  for model_path in all_results.keys()]
    
    # Create plots for ROUGE-L and BLEU
    for metric_name in ['ROUGE-L', 'BLEU']:
        plt.figure(figsize=(12, 8))
        
        # For each dataset, create a group of bars (one for each model)
        bar_width = 0.8 / len(model_names) if model_names else 0.8
        index = np.arange(len(dataset_names))
        
        for i, (model_path, model_data) in enumerate(all_results.items()):
            model_name = Path(model_path).name if not model_path.endswith('/') else Path(model_path).parent.name
            values = []
            
            for dataset in dataset_names:
                if dataset in model_data['results']:
                    values.append(model_data['results'][dataset][metric_name])
                else:
                    values.append(0)
            
            offset = bar_width * i - bar_width * len(model_names) / 2 + bar_width / 2
            plt.bar(index + offset, values, bar_width, label=model_name)
        
        plt.xlabel('Dataset')
        plt.ylabel(f'{metric_name} Score')
        plt.title(f'{metric_name} Comparison Across Models and Datasets')
        plt.xticks(index, dataset_names)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'{metric_name}_comparison.png'))
        plt.close()
    
    # Create a plot for evaluation time
    if all('evaluation_time' in data for data in all_results.values()):
        plt.figure(figsize=(10, 6))
        times = [data['evaluation_time'] for data in all_results.values()]
        plt.bar(model_names, times, color='skyblue')
        plt.xlabel('Model')
        plt.ylabel('Evaluation Time (seconds)')
        plt.title('Evaluation Time Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'evaluation_time_comparison.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Run inference with multiple models on medical datasets')
    parser.add_argument('--n_samples', type=int, default=np.inf, help='Number of samples to process per dataset')
    parser.add_argument('--output_dir', type=str, default='model_comparison_results', help='Directory to save results')
    parser.add_argument('--cpu_only', action='store_true', help='Use CPU only for inference')
    parser.add_argument('--models', nargs='+', choices=['deepseek', 'mmed', 'ultramed', 'merged'], 
                        help='Specific models to evaluate (default: all)')
    parser.add_argument('--no_vllm', action='store_true', help='Disable vllm and use transformers instead')
    parser.add_argument('--no_int8', action='store_true', help='Disable int8 quantization')
    args = parser.parse_args()
    
    # Check if CUDA is available
    device = 'cpu' if args.cpu_only or not torch.cuda.is_available() else 'cuda'
    print(f"Using device: {device}")
    
    # Filter models based on user input
    model_paths_to_use = MODEL_PATHS
    if args.models:
        model_map = {
            'deepseek': "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            'mmed': "Henrychur/MMed-Llama-3-8B",
            'ultramed': "TsinghuaC3I/Llama-3-8B-UltraMedical",
            'merged': "./merged-model",
        }
        model_paths_to_use = [model_map[model] for model in args.models if model in model_map]
    
    all_results = {}
    
    for model_path in model_paths_to_use:
        try:
            results = evaluate_model(
                model_path=model_path,
                dataset_paths=DATASET_PATHS,
                n_samples=args.n_samples,
                output_dir=args.output_dir,
                device=device,
                use_vllm=not args.no_vllm,
                use_int8=not args.no_int8
            )
            all_results[model_path] = results
        except Exception as e:
            print(f"Error evaluating model {model_path}: {e}")
            continue
    
    if all_results:
        plot_results(all_results, args.output_dir)
        
        # Save a summary of all results
        with open(f"{args.output_dir}/all_results_summary.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nAll evaluations completed. Results saved to {args.output_dir}")
    else:
        print("No results were generated for any model.")

if __name__ == "__main__":
    main() 