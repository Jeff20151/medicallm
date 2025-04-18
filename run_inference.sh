#!/bin/bash

# 啟動conda環境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate medicalllm

# 建立結果目錄
mkdir -p results

# 安裝必要的依賴
pip install rouge-score tqdm nltk datasets torch transformers accelerate

# 防止記憶體問題的環境變數設定
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TRANSFORMERS_OFFLINE=0
export CUDA_VISIBLE_DEVICES=0

# 釋放記憶體
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches || echo "無法釋放緩存，繼續運行..."

# 運行推理腳本 (默認處理10個樣本，如需處理更多，可調整--n_samples參數)
python llama_8b.py --dataset_paths /home/user/medicallm/clin-summ/data/chq /home/user/medicallm/clin-summ/data/d2n /home/user/medicallm/clin-summ/data/opi --n_samples 5

# 如果第一次失敗，嘗試使用 CPU 運行
if [ $? -ne 0 ]; then
    echo "GPU 運行失敗，嘗試使用 CPU..."
    python llama_8b.py --dataset_paths /home/user/medicallm/clin-summ/data/chq /home/user/medicallm/clin-summ/data/d2n /home/user/medicallm/clin-summ/data/opi --n_samples 5 --cpu_only
fi

# 顯示結果摘要
echo "=================================================="
echo "推理完成，結果摘要:"
cat results/overall_metrics.json 2>/dev/null || echo "未生成結果文件"
echo "=================================================="
echo "詳細結果保存在 results/ 目錄中" 