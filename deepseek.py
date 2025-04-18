#!/usr/bin/env python
# Load model directly
import os
import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import glob
import csv
from tqdm import tqdm
import argparse

# Download necessary NLTK data
nltk.download('punkt')

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

def load_data(dataset_path):
    """Load data from CSV files in the dataset directory."""
    input_file = os.path.join(dataset_path, "inputs.csv")
    target_file = os.path.join(dataset_path, "target.csv")
    
    if os.path.exists(input_file) and os.path.exists(target_file):
        inputs_df = pd.read_csv(input_file)
        targets_df = pd.read_csv(target_file)
        
        # Ensure we have matching indices
        common_indices = set(inputs_df['idx']).intersection(set(targets_df['idx']))
        inputs_df = inputs_df[inputs_df['idx'].isin(common_indices)]
        targets_df = targets_df[targets_df['idx'].isin(common_indices)]
        
        # Sort by idx to ensure alignment
        inputs_df = inputs_df.sort_values('idx').reset_index(drop=True)
        targets_df = targets_df.sort_values('idx').reset_index(drop=True)
        
        return inputs_df, targets_df
    else:
        print(f"Error: Input or target file not found in {dataset_path}")
        return None, None

def generate_text(model, tokenizer, prompt, max_new_tokens=256, device='cuda'):
    """Generate text using the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]

def create_prompt(input_text, dataset_name):
    """Create prompt for the specific dataset."""
    component = PROMPT_COMPONENT[dataset_name]
    instruction = component['instruction']
    prefix = component['prefix']
    suffix = component['suffix']
    
    prompt = f"""You are a knowledgeable medical professional. Below is a {prefix}. Please {instruction}.

{prefix}:
{input_text}

{suffix}:
"""
    return prompt

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
        
        # Calculate BLEU
        ref_tokens = nltk.word_tokenize(ref.lower())
        pred_tokens = nltk.word_tokenize(pred.lower())
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
        bleu_scores.append(bleu)
    
    return {
        'ROUGE-L': sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0,
        'BLEU': sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    }

def main():
    parser = argparse.ArgumentParser(description='Run inference with DeepSeek-R1 on medical datasets')
    parser.add_argument('--dataset_paths', nargs='+', required=True, help='Paths to datasets')
    parser.add_argument('--n_samples', type=int, default=10, help='Number of samples to process')
    parser.add_argument('--cpu_only', action='store_true', help='Use CPU only for inference')
    parser.add_argument('--output_dir', type=str, default='results_deepseek', help='Directory to save results')
    args = parser.parse_args()
    
    # Check if CUDA is available
    device = 'cpu' if args.cpu_only or not torch.cuda.is_available() else 'cuda'
    print(f"Using device: {device}")
    
    print("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", trust_remote_code=True)
        
        # 嘗試不同的加載方法
        try:
            # 方法 1: 使用 device_map="auto" (需要 accelerate)
            try:
                import accelerate
                model = AutoModelForCausalLM.from_pretrained(
                    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
                    torch_dtype=torch.float16, 
                    device_map="auto",
                    trust_remote_code=True
                )
                print("使用 accelerate 載入模型成功")
            except (ImportError, ValueError):
                # 方法 2: 使用指定 device
                print("無法使用 accelerate，嘗試直接載入模型到設備")
                model = AutoModelForCausalLM.from_pretrained(
                    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                    trust_remote_code=True
                ).to(device)
        except Exception as e:
            print(f"載入模型發生錯誤: {e}")
            # 方法 3: 最基本的加載方式
            print("嘗試使用基本方法加載模型")
            model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", trust_remote_code=True)
            model = model.to(device)
    except Exception as e:
        print(f"模型載入失敗: {e}")
        return
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    
    for dataset_path in args.dataset_paths:
        dataset_name = os.path.basename(dataset_path)
        print(f"\nProcessing dataset: {dataset_name}")
        
        if dataset_name not in PROMPT_COMPONENT:
            print(f"Skipping {dataset_name}: not supported.")
            continue
        
        inputs_df, targets_df = load_data(dataset_path)
        if inputs_df is None or targets_df is None:
            continue
        
        # Limit samples for testing
        n_samples = min(args.n_samples, len(inputs_df))
        inputs_df = inputs_df.head(n_samples)
        targets_df = targets_df.head(n_samples)
        
        predictions = []
        references = []
        
        print(f"Generating predictions for {n_samples} samples...")
        for i, (_, row) in enumerate(tqdm(inputs_df.iterrows(), total=n_samples)):
            input_text = row['inputs']
            reference = targets_df.iloc[i]['target']
            
            prompt = create_prompt(input_text, dataset_name)
            prediction = generate_text(model, tokenizer, prompt, device=device)
            
            predictions.append(prediction.strip())
            references.append(reference.strip())
            
            # Save intermediate results
            output_dir = f"{args.output_dir}/{dataset_name}"
            os.makedirs(output_dir, exist_ok=True)
            
            with open(f"{output_dir}/sample_{i}.json", 'w') as f:
                json.dump({
                    'input': input_text,
                    'prompt': prompt,
                    'reference': reference,
                    'prediction': prediction
                }, f, indent=2)
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, references)
        results[dataset_name] = metrics
        
        print(f"Results for {dataset_name}:")
        print(f"ROUGE-L: {metrics['ROUGE-L']:.4f}")
        print(f"BLEU: {metrics['BLEU']:.4f}")
        
        # Save all predictions and metrics
        with open(f"{output_dir}/predictions.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['input', 'reference', 'prediction'])
            for i in range(len(predictions)):
                writer.writerow([inputs_df.iloc[i]['inputs'], references[i], predictions[i]])
        
        with open(f"{output_dir}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Save overall results
    print("\nOverall Results:")
    for dataset_name, metrics in results.items():
        print(f"{dataset_name}:")
        print(f"  ROUGE-L: {metrics['ROUGE-L']:.4f}")
        print(f"  BLEU: {metrics['BLEU']:.4f}")
    
    with open(f"{args.output_dir}/overall_metrics.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()