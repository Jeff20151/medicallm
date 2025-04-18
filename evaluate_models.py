import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import matplotlib.pyplot as plt
import time
import os

###############################################################################
# 1) LOAD DATASET
###############################################################################
dataset_name = "TsinghuaC3I/UltraMedical"
ultra_ds = load_dataset(dataset_name, split="train")
# Optionally take a small sample for quick tests:
ultra_ds = ultra_ds.select(range(1000))

###############################################################################
# 2) MODEL LIST
###############################################################################
model_paths = [
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "Henrychur/MMed-Llama-3-8B",
    "TsinghuaC3I/Llama-3-8B-UltraMedical",
    "./merged-model",  # The newly merged one
]

###############################################################################
# 3) EVALUATION
###############################################################################
def evaluate_model(model_path, dataset, device="cuda"):
    print(f"\n=== Evaluating: {model_path} ===")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Configure pad_token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()
    
    # Create a clean generation config to avoid warnings
    generation_config = GenerationConfig(
        max_new_tokens=128,
        do_sample=False,
        num_beams=1,  # Greedy decoding
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # Explicitly unset problematic parameters
        temperature=None,
        top_p=None
    )

    total = 0
    correct = 0

    for idx, sample in enumerate(dataset):
        # Extract question from the conversations field (first message from human)
        question = ""
        for conv in sample["conversations"]:
            if conv["from"] == "human":
                question = conv["value"]
                break
        
        # Get the answer from the dataset
        gold_answer = sample["answer"]

        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                generation_config=generation_config
            )

        pred = tokenizer.decode(output[0], skip_special_tokens=True)
        predicted_answer = pred.split("Answer:")[-1].strip()

        print(f"Sample {idx+1}/{len(dataset)}")
        print(f"Gold answer: {gold_answer}")
        print(f"Predicted: {predicted_answer[:50]}...\n")
        
        total += 1
        if gold_answer.strip().lower() in predicted_answer.lower():
            correct += 1

    accuracy = correct / total if total else 0.0
    elapsed_time = time.time() - start_time
    
    print(f"Model: {model_path}")
    print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
    print(f"Evaluation time: {elapsed_time:.1f} seconds")
    
    model_result = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "evaluation_time": elapsed_time
    }
    return model_result

def plot_results(results):
    """Plot the accuracy and evaluation time for each model."""
    model_names = [os.path.basename(path) for path in results.keys()]
    accuracies = [results[model]["accuracy"] * 100 for model in results.keys()]
    eval_times = [results[model]["evaluation_time"] / 60 for model in results.keys()]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot accuracies
    bars1 = ax1.bar(model_names, accuracies, color='skyblue')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylim(0, 100)
    
    # Add accuracy values on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Plot evaluation times
    bars2 = ax2.bar(model_names, eval_times, color='lightgreen')
    ax2.set_ylabel('Evaluation Time (minutes)')
    ax2.set_title('Model Evaluation Time')
    
    # Add time values on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f} min', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
    print("Results visualization saved to 'model_comparison_results.png'")


device = "cuda" if torch.cuda.is_available() else "cpu"

results = {}
for mp in model_paths:
    model_result = evaluate_model(mp, ultra_ds, device=device)
    results[mp] = model_result

print("\nResults summary:")
for k, v in results.items():
    print(f"  {k}: Accuracy = {v['accuracy']:.3f}, Time = {v['evaluation_time']:.1f}s")

# Plot and save the results
plot_results(results)
