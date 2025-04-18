#!/usr/bin/env python
# 評估合併模型的性能
import os
import json
import pandas as pd
import numpy as np
import sys
import random
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import traceback

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

# 定義錯誤處理裝飾器
def safe_import(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ImportError as e:
            module_name = str(e).split("'")[1]
            print(f"錯誤: 未找到必要的依賴 {module_name}")
            print(f"請安裝相關依賴: pip install {module_name}")
            sys.exit(1)
    return wrapper

@safe_import
def load_dependencies():
    global torch, AutoTokenizer, AutoModelForCausalLM, evaluate
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import evaluate
    except ImportError as e:
        print(f"錯誤: 缺少必要的依賴: {e}")
        print("請執行: pip install transformers torch evaluate")
        raise

def generate_simulated_metrics(model_name: str) -> Dict[str, Dict[str, float]]:
    """生成模擬的評估指標，用於乾跑模式"""
    random.seed(42)  # 確保可重複結果
    
    # 模擬不同數據集的結果
    datasets = ["chq", "d2n", "opi"]
    metrics = {}
    
    for dataset in datasets:
        # 基於模型名稱產生不同的基準值
        if "UltraMedical" in model_name:
            base_rouge = 0.38
            base_bleu = 0.25
        elif "MMed" in model_name:
            base_rouge = 0.36
            base_bleu = 0.23
        elif "DeepSeek" in model_name:
            base_rouge = 0.40
            base_bleu = 0.27
        elif "merged" in model_name.lower() or "stock" in model_name.lower():
            base_rouge = 0.41
            base_bleu = 0.28
        else:
            base_rouge = 0.35
            base_bleu = 0.22
            
        # 添加一些隨機變化
        rouge_l = base_rouge + random.uniform(-0.05, 0.05)
        bleu = base_bleu + random.uniform(-0.05, 0.05)
        
        # 確保值在合理範圍內
        rouge_l = max(0.1, min(0.7, rouge_l))
        bleu = max(0.05, min(0.5, bleu))
        
        metrics[dataset] = {
            "rouge_l": round(rouge_l, 4),
            "bleu": round(bleu, 4)
        }
    
    return metrics

class ModelEvaluator:
    def __init__(self, model_name_or_path: str, device: str = None, dry_run: bool = False):
        self.model_name = model_name_or_path
        self.dry_run = dry_run
        
        if self.dry_run:
            print(f"[乾跑模式] 模擬加載模型: {model_name_or_path}")
            return
            
        # 檢查 CUDA 可用性
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"使用設備: {self.device}")
        print(f"加載模型: {model_name_or_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
        except Exception as e:
            print(f"加載模型時出錯: {e}")
            print(traceback.format_exc())
            raise RuntimeError(f"無法加載模型 {model_name_or_path}")
        
        # 設置 chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            self.has_template = True
        else:
            self.has_template = False
            print("警告: 模型沒有聊天模板，將使用基本提示")
        
        # 加載評估指標
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")
    
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """加載數據集"""
        if self.dry_run:
            # 返回一些模擬數據
            return [
                {"source": "患者主訴頭痛三天", "target": "診斷為偏頭痛"} 
                for _ in range(5)
            ]
            
        dataset = []
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        if "source" in item and "target" in item:
                            dataset.append(item)
                        else:
                            print(f"警告: 數據項缺少 source 或 target: {item}")
                    except json.JSONDecodeError:
                        print(f"警告: 無法解析 JSON 行: {line.strip()}")
        except Exception as e:
            print(f"加載數據集時出錯: {e}")
            return []
        
        print(f"加載了 {len(dataset)} 條數據項")
        return dataset
    
    def generate_text(self, prompt: str, max_length: int = 512) -> str:
        """生成回應文本"""
        if self.dry_run:
            # 返回一個簡單的模擬響應
            return "診斷為偏頭痛，建議服用止痛藥並休息"
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 設置生成參數
        generation_config = {
            "max_new_tokens": max_length,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # 生成響應
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)
        
        # 解碼輸出
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 移除輸入提示
        input_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        response = generated_text[len(input_text):].strip()
        
        return response
    
    def format_prompt(self, source_text: str) -> str:
        """格式化輸入提示"""
        system_message = "你是一個醫療助手。請使用專業且簡潔的語言總結以下醫療記錄的要點。"
        user_message = source_text
        
        if self.has_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            prompt = f"{system_message}\n\n{user_message}"
        
        return prompt
    
    def evaluate_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """評估模型在數據集上的表現"""
        if self.dry_run:
            # 返回模擬的評估結果
            dataset_name = "unknown"
            return {"rouge_l": 0.38, "bleu": 0.25}
            
        if not dataset:
            print("警告: 數據集為空，無法進行評估")
            return {"rouge_l": 0.0, "bleu": 0.0}
        
        predictions = []
        references = []
        
        for item in tqdm(dataset, desc="生成響應"):
            try:
                prompt = self.format_prompt(item["source"])
                prediction = self.generate_text(prompt)
                predictions.append(prediction)
                references.append(item["target"])
            except Exception as e:
                print(f"生成響應時出錯: {e}")
                # 添加一個空預測以保持索引一致性
                predictions.append("")
                references.append(item["target"])
        
        # 計算 ROUGE 指標
        rouge_results = self.rouge.compute(predictions=predictions, references=references)
        
        # 計算 BLEU 指標
        tokenized_predictions = [pred.split() for pred in predictions]
        tokenized_references = [[ref.split()] for ref in references]
        bleu_results = self.bleu.compute(predictions=tokenized_predictions, references=tokenized_references)
        
        metrics = {
            "rouge_l": rouge_results["rougeL"],
            "bleu": bleu_results["bleu"]
        }
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description="評估合併模型的性能")
    parser.add_argument("--model", type=str, required=True, help="模型名稱或路徑")
    parser.add_argument("--datasets", nargs="+", required=True, help="數據集路徑列表")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="結果輸出目錄")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None, help="運行設備")
    parser.add_argument("--dry-run", action="store_true", help="乾跑模式，不實際加載模型或處理數據")
    args = parser.parse_args()
    
    # 加載依賴，如果在乾跑模式下跳過
    if not args.dry_run:
        try:
            load_dependencies()
        except:
            if args.dry_run:
                print("警告: 雖然依賴加載失敗，但因為處於乾跑模式，將繼續執行")
            else:
                sys.exit(1)
    
    # 創建輸出目錄
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except Exception as e:
        print(f"創建輸出目錄時出錯: {e}")
        sys.exit(1)
    
    # 獲取模型名稱（不含路徑）
    model_name = os.path.basename(args.model)
    
    if args.dry_run:
        print(f"[乾跑模式] 模擬評估模型: {args.model}")
        evaluator = ModelEvaluator(args.model, device=args.device, dry_run=True)
        all_metrics = generate_simulated_metrics(args.model)
    else:
        try:
            # 創建評估器
            evaluator = ModelEvaluator(args.model, device=args.device, dry_run=args.dry_run)
            
            # 評估每個數據集
            all_metrics = {}
            for dataset_path in args.datasets:
                try:
                    dataset_name = os.path.basename(os.path.dirname(dataset_path))
                    print(f"評估數據集: {dataset_name}")
                    
                    dataset = evaluator.load_dataset(dataset_path)
                    metrics = evaluator.evaluate_dataset(dataset)
                    all_metrics[dataset_name] = metrics
                    
                    print(f"數據集 {dataset_name} 的指標:")
                    print(f"  ROUGE-L: {metrics['rouge_l']:.4f}")
                    print(f"  BLEU: {metrics['bleu']:.4f}")
                except Exception as e:
                    print(f"評估數據集 {dataset_path} 時出錯: {e}")
                    print(traceback.format_exc())
        except Exception as e:
            if args.dry_run:
                print(f"警告: 評估過程中出錯: {e}")
                print("繼續乾跑模式，生成模擬結果")
                all_metrics = generate_simulated_metrics(args.model)
            else:
                print(f"評估過程中出錯: {e}")
                print(traceback.format_exc())
                sys.exit(1)
    
    # 保存結果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(args.output_dir, f"{model_name}_{timestamp}.json")
    
    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                "model": args.model,
                "metrics": all_metrics,
                "timestamp": timestamp,
                "dry_run": args.dry_run
            }, f, indent=2)
        
        print(f"結果已保存到 {result_file}")
    except Exception as e:
        print(f"保存結果時出錯: {e}")
    
    # 輸出摘要表格
    print("\n評估摘要:")
    print("-" * 50)
    print(f"模型: {args.model}")
    print("-" * 50)
    print(f"{'數據集':<10} {'ROUGE-L':<10} {'BLEU':<10}")
    print("-" * 50)
    
    for dataset_name, metrics in all_metrics.items():
        print(f"{dataset_name:<10} {metrics['rouge_l']:.4f}     {metrics['bleu']:.4f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 