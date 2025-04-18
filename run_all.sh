#!/bin/bash

# 執行整個工作流程：模型合併、評估和比較
# 用法: ./run_all.sh [--skip-merge] [--skip-eval] [--dry-run]

# 解析命令列參數
SKIP_MERGE=false
SKIP_EVAL=false
DRY_RUN=false

for arg in "$@"; do
  case $arg in
    --skip-merge)
      SKIP_MERGE=true
      shift
      ;;
    --skip-eval)
      SKIP_EVAL=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
  esac
done

# 設置環境變數
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# 檢查 conda 環境
if [ -n "$CONDA_PREFIX" ]; then
  echo "使用現有 conda 環境: $CONDA_PREFIX"
else
  echo "警告: 未檢測到活躍的 conda 環境，可能會導致依賴問題"
fi

# 設置目錄
MERGE_CONFIG_DIR="merge_configs"
MERGED_MODELS_DIR="merged_models"
EVAL_RESULTS_DIR="evaluation_results"
COMPARISON_DIR="model_comparisons"

# 創建目錄
mkdir -p "$MERGE_CONFIG_DIR"
mkdir -p "$MERGED_MODELS_DIR"
mkdir -p "$EVAL_RESULTS_DIR"
mkdir -p "$COMPARISON_DIR"

################################
# 步驟1: 安裝必要的依賴
################################
echo "步驟1: 安裝必要的依賴..."
pip install -q transformers accelerate rouge-score tqdm nltk datasets torch matplotlib pandas pyyaml

# 如要進行模型合併，需要安裝 mergekit（需 Rust 編譯器）
if [ "$SKIP_MERGE" = false ] && [ "$DRY_RUN" = false ]; then
  echo "安裝 mergekit (可能需要 Rust 編譯器)..."
  pip install -q mergekit
fi

################################
# 步驟2: 合併模型（如果需要）
################################
if [ "$SKIP_MERGE" = false ]; then
  echo "步驟2: 生成並執行模型合併..."

  # 在此呼叫我們的 Python 腳本 "merge_medical_models.py"
  # 根據 --dry-run 決定要不要真正執行合併
  if [ "$DRY_RUN" = true ]; then
    python merge_medical_models.py \
      --method ties \
      --config-dir "$MERGE_CONFIG_DIR" \
      --output-dir "$MERGED_MODELS_DIR" \
      --run --dry-run
  else
    python merge_medical_models.py \
      --method ties \
      --config-dir "$MERGE_CONFIG_DIR" \
      --output-dir "$MERGED_MODELS_DIR" \
      --run
  fi
else
  echo "步驟2: 跳過模型合併（--skip-merge）"
fi

################################
# 步驟3: 評估原始模型和合併後的模型
################################
if [ "$SKIP_EVAL" = false ]; then
  echo "步驟3: 評估所有模型..."

  # 定義數據集路徑
  DATASETS=(
      "/home/user/medicallm/clin-summ/data/chq"
      "/home/user/medicallm/clin-summ/data/d2n"
      "/home/user/medicallm/clin-summ/data/opi"
  )

  # 檢查數據集路徑是否存在，若不存在則嘗試使用相對路徑
  DATASETS_OK=true
  for dataset in "${DATASETS[@]}"; do
      if [ ! -d "$dataset" ]; then
          echo "警告: 數據集目錄不存在: $dataset"
          DATASETS_OK=false
      fi
  done

  if [ "$DATASETS_OK" = false ]; then
      echo "嘗試切換為相對路徑..."
      DATASETS=(
          "clin-summ/data/chq"
          "clin-summ/data/d2n"
          "clin-summ/data/opi"
      )
  fi

  # 評估原始模型
  ORIGINAL_MODELS=(
      "TsinghuaC3I/Llama-3-8B-UltraMedical"
      "Henrychur/MMed-Llama-3-8B"
      "deepseek-ai/DeepSeek-R1"
  )

  # 若乾跑模式則加上 --dry-run 標記
  DRY_RUN_FLAG=""
  if [ "$DRY_RUN" = true ]; then
      DRY_RUN_FLAG="--dry-run"
  fi

  for model in "${ORIGINAL_MODELS[@]}"; do
      echo "評估原始模型: $model"
      python evaluate_merged_model.py \
          --model-path "$model" \
          --dataset-paths "${DATASETS[@]}" \
          --n-samples 3 \
          $DRY_RUN_FLAG
  done

  # 評估合併後的模型
  if [ -d "$MERGED_MODELS_DIR" ]; then
      for model_dir in "$MERGED_MODELS_DIR"/*; do
          if [ -d "$model_dir" ]; then
              echo "評估合併模型: $model_dir"
              python evaluate_merged_model.py \
                  --model-path "$model_dir" \
                  --dataset-paths "${DATASETS[@]}" \
                  --n-samples 3 \
                  $DRY_RUN_FLAG
          fi
      done
  else
      echo "找不到合併模型目錄: $MERGED_MODELS_DIR"
  fi
else
  echo "步驟3: 跳過模型評估（--skip-eval）"
fi

################################
# 步驟4: 比較所有模型性能
################################
echo "步驟4: 比較所有模型性能..."

# 如果是乾跑模式且沒有評估結果，就創建模擬的評估結果以便測試比較
if [ "$DRY_RUN" = true ] && [ ! -d "eval_results_Llama-3-8B-UltraMedical" ]; then
  echo "乾跑模式: 創建模擬評估結果..."
  for model in "Llama-3-8B-UltraMedical" "MMed-Llama-3-8B" "DeepSeek-R1" "UltraMed-MMed-DeepSeek-8B-ties"; do
      results_dir="eval_results_$model"
      mkdir -p "$results_dir"
      cat > "$results_dir/overall_metrics.json" << EOF
{
  "chq": {
    "ROUGE-L": $(awk -v min=0.3 -v max=0.8 'BEGIN{srand(); print min+rand()*(max-min)}'),
    "BLEU": $(awk -v min=0.1 -v max=0.5 'BEGIN{srand(); print min+rand()*(max-min)}')
  },
  "d2n": {
    "ROUGE-L": $(awk -v min=0.3 -v max=0.8 'BEGIN{srand(); print min+rand()*(max-min)}'),
    "BLEU": $(awk -v min=0.1 -v max=0.5 'BEGIN{srand(); print min+rand()*(max-min)}')
  },
  "opi": {
    "ROUGE-L": $(awk -v min=0.3 -v max=0.8 'BEGIN{srand(); print min+rand()*(max-min)}'),
    "BLEU": $(awk -v min=0.1 -v max=0.5 'BEGIN{srand(); print min+rand()*(max-min)}')
  }
}
EOF
  done
fi

# 執行比較腳本
python compare_models.py --result-dirs "eval_results_*" --output-dir "$COMPARISON_DIR"

echo "工作流程完成！"
echo "請檢查 $COMPARISON_DIR 目錄中的比較結果。"
echo "也可以在以下目錄查看詳細結果:"
echo "- 模型合併配置: $MERGE_CONFIG_DIR"
echo "- 合併後的模型: $MERGED_MODELS_DIR"
echo "- 評估結果: eval_results_*"

# 如果是乾跑模式，提醒用戶
if [ "$DRY_RUN" = true ]; then
  echo ""
  echo "注意：這是乾跑模式，未實際下載或合併模型。"
  echo "所有結果僅為測試目的而生成的範例數據。"
fi
