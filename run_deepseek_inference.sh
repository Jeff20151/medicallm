#!/bin/bash

# 啟動conda環境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate medicalllm

# 建立結果目錄
mkdir -p results_deepseek

# 安裝必要的依賴
pip install rouge-score tqdm nltk datasets torch transformers accelerate

# 防止記憶體問題的環境變數設定
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TRANSFORMERS_OFFLINE=0
export CUDA_VISIBLE_DEVICES=0

# 釋放記憶體
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches || echo "無法釋放緩存，繼續運行..."

# 運行DeepSeek-R1模型的推理腳本
python deepseek.py --dataset_paths /home/user/medicallm/clin-summ/data/chq /home/user/medicallm/clin-summ/data/d2n /home/user/medicallm/clin-summ/data/opi --n_samples 5 --output_dir results_deepseek

# 如果第一次失敗，嘗試使用 CPU 運行
if [ $? -ne 0 ]; then
    echo "GPU 運行失敗，嘗試使用 CPU..."
    python deepseek.py --dataset_paths /home/user/medicallm/clin-summ/data/chq /home/user/medicallm/clin-summ/data/d2n /home/user/medicallm/clin-summ/data/opi --n_samples 5 --output_dir results_deepseek --cpu_only
fi

# 顯示結果摘要
echo "=================================================="
echo "推理完成，結果摘要:"
cat results_deepseek/overall_metrics.json 2>/dev/null || echo "未生成結果文件"
echo "=================================================="
echo "詳細結果保存在 results_deepseek/ 目錄中"

# 如果其他模型已運行完成，顯示比較結果
echo ""
echo "=== 模型比較 ==="

if [ -f "results/overall_metrics.json" ]; then
    echo "UltraMedical模型結果:"
    cat results/overall_metrics.json | grep -E "ROUGE-L|BLEU"
    echo ""
fi

if [ -f "results_mmed/overall_metrics.json" ]; then
    echo "MMed-Llama模型結果:"
    cat results_mmed/overall_metrics.json | grep -E "ROUGE-L|BLEU"
    echo ""
fi

if [ -f "results_deepseek/overall_metrics.json" ]; then
    echo "DeepSeek-R1模型結果:"
    cat results_deepseek/overall_metrics.json | grep -E "ROUGE-L|BLEU"
fi

# 如果所有模型都已完成，創建比較報告
if [ -f "results/overall_metrics.json" ] && [ -f "results_mmed/overall_metrics.json" ] && [ -f "results_deepseek/overall_metrics.json" ]; then
    echo ""
    echo "生成模型比較報告..."
    python -c "
import json
import pandas as pd
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))

# 讀取三個模型的結果
with open('results/overall_metrics.json', 'r') as f:
    ultra_results = json.load(f)
with open('results_mmed/overall_metrics.json', 'r') as f:
    mmed_results = json.load(f)
with open('results_deepseek/overall_metrics.json', 'r') as f:
    deepseek_results = json.load(f)

# 準備數據
datasets = list(set(ultra_results.keys()) & set(mmed_results.keys()) & set(deepseek_results.keys()))
metrics = ['ROUGE-L', 'BLEU']
models = ['UltraMedical', 'MMed-Llama', 'DeepSeek-R1']
colors = ['blue', 'green', 'red']

# 構建數據框
data = []
for dataset in datasets:
    for metric in metrics:
        data.append({
            'Dataset': dataset,
            'Metric': metric,
            'UltraMedical': ultra_results[dataset][metric],
            'MMed-Llama': mmed_results[dataset][metric],
            'DeepSeek-R1': deepseek_results[dataset][metric]
        })

df = pd.DataFrame(data)

# 創建圖表
for i, metric in enumerate(metrics):
    plt.subplot(2, 1, i+1)
    metric_data = df[df['Metric'] == metric]
    
    bar_width = 0.25
    index = range(len(datasets))
    
    for j, model in enumerate(models):
        plt.bar([x + j*bar_width for x in index], 
                metric_data[model], 
                bar_width, 
                label=model, 
                color=colors[j])
    
    plt.title(f'{metric} Score Comparison')
    plt.xlabel('Dataset')
    plt.ylabel('Score')
    plt.xticks([x + bar_width for x in index], datasets)
    plt.legend()

plt.tight_layout()
plt.savefig('model_comparison.png')
print('比較圖表已保存到 model_comparison.png')
" || echo "無法生成比較報告"
fi 