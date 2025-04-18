# Medical LLM評估工具

此項目用於評估醫療領域大型語言模型（LLM）在臨床摘要任務上的性能。

## 功能特點

- 支持多種醫學LLM模型的評估
- 測試不同模型在各種醫療摘要數據集上的表現
- 使用ROUGE-L和BLEU指標進行評估
- 生成比較圖表以視覺化不同模型的性能
- 支持vllm加速（可選）和int8量化以減少內存占用

## 主要模型

此工具支持以下模型：
- DeepSeek-R1-Distill-Llama-8B
- MMed-Llama-3-8B
- Llama-3-8B-UltraMedical
- 自定義合併模型

## 數據集

評估基於以下摘要任務：
- chq: 病人健康查詢摘要
- opi: 放射學報告摘要
- d2n: 醫生-病人對話摘要

## 安裝與設置

1. 克隆此倉庫：
   ```bash
   git clone [your-repo-url]
   cd medicallm
   ```

2. 安裝依賴：
   ```bash
   python install_dependencies.py
   ```
   
   或手動安裝：
   ```bash
   pip install transformers nltk rouge-score pandas matplotlib tqdm
   # 可選依賴
   pip install vllm bitsandbytes
   ```

3. 下載NLTK資源：
   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

## 使用方法

### 基本使用

```bash
python run_inference.py --n_samples 3 --models deepseek mmed ultramed
```

### 指定所有樣本

```bash
python run_inference.py --models deepseek
```

### 禁用vllm（如果安裝出現問題）

```bash
python run_inference.py --n_samples 3 --models deepseek --no_vllm
```

### 禁用int8量化

```bash
python run_inference.py --n_samples 3 --models deepseek --no_int8
```

### 僅使用CPU（較慢）

```bash
python run_inference.py --n_samples 3 --models deepseek --cpu_only
```

## 結果

結果將保存在`model_comparison_results`目錄中，包括：
- 每個樣本的預測和參考摘要
- 每個模型和數據集的指標摘要
- 比較不同模型性能的圖表

## 自定義

通過修改`run_inference.py`中的`MODEL_PATHS`和`DATASET_PATHS`變量來自定義模型和數據集。

## 貢獻

歡迎提交問題報告和功能請求。如需貢獻代碼，請先創建一個Issue討論您想要更改的內容。 