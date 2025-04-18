#!/usr/bin/env python
# 使用 MergeKit 合併醫療大語言模型
import os
import sys
import yaml
import subprocess
import argparse

def create_merge_config(
    merge_method,
    models,
    base_model=None,
    parameters=None,
    dtype="float16"
):
    """
    創建合併配置文件（只存放合併所需的欄位）
    注意：並不在此直接放 output_model，因為我們會
         直接在 CLI 裡把 output_model_dir 當作第二參數傳給 mergekit-yaml
    """
    config = {
        "merge_method": merge_method,
        "models": [{"model": m} for m in models],
        "dtype": dtype,
    }
    
    if base_model:
        config["base_model"] = base_model
    
    if parameters:
        config["parameters"] = parameters
    
    # 簡化tokenizer設定，只使用union來源
    config["tokenizer"] = {
        "source": "union"
    }
    
    # chat_template，可使用 auto 或其他內建模板（alpaca, chatml, llama3 等）
    config["chat_template"] = "auto"
    
    return config

def save_config(config, config_file_path):
    """將合併配置寫到 YAML 檔"""
    try:
        with open(config_file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, sort_keys=False, allow_unicode=True)
        print(f"已儲存 YAML 配置檔到: {config_file_path}")
        return True
    except Exception as e:
        print(f"無法儲存配置檔：{e}")
        return False

def run_merge(config_file, output_dir, use_cuda=True, allow_crimes=True, dry_run=False):
    """
    呼叫 mergekit-yaml 進行合併。
    根據 README: mergekit-yaml path/to/your/config.yml ./output-model-directory [--cuda] [其他參數...]
    """
    cmd = ["mergekit-yaml", config_file, output_dir]
    
    # 加上可選參數
    if use_cuda:
        cmd.append("--cuda")
    if allow_crimes:
        cmd.append("--allow-crimes")
    
    # 建議加上 --lazy-unpickle 提升大型權重讀取的效率
    cmd.append("--lazy-unpickle")
    
    print("準備執行指令：", " ".join(cmd))
    
    if dry_run:
        print("[乾跑模式] 不執行實際合併。")
        return True
    
    try:
        subprocess.run(cmd, check=True)
        print("合併完成。")
        return True
    except FileNotFoundError:
        print("找不到 mergekit-yaml 指令，請確認 mergekit 是否已安裝 (pip install mergekit)。")
    except subprocess.CalledProcessError as e:
        print(f"呼叫合併時失敗: {e}")
    
    return False

def main():
    parser = argparse.ArgumentParser(description="示例：使用 MergeKit 進行多種方法之模型合併")
    parser.add_argument("--method", type=str, choices=[
        "slerp", "ties", "task_arithmetic", "dare_ties", "model_stock", "all"
    ], default="slerp", help="選擇要執行的合併方法")
    parser.add_argument("--output-dir", default="merged_models", help="輸出模型的資料夾路徑")
    parser.add_argument("--config-dir", default="merge_configs", help="用來存放 YAML 配置檔的資料夾")
    parser.add_argument("--no-cuda", action="store_true", help="若指定則不使用 CUDA")
    parser.add_argument("--run", action="store_true", help="若指定則實際執行 mergekit-yaml 合併")
    parser.add_argument("--dry-run", action="store_true", help="乾跑模式，顯示指令但不執行")
    args = parser.parse_args()
    
    # 只使用兩個較小的醫療模型，避免使用DeepSeek-R1
    models = [
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "TsinghuaC3I/Llama-3-8B-UltraMedical",
        "Henrychur/MMed-Llama-3-8B"
    ]
    
    # 只提供一種最簡單的合併方法
    all_methods = {
        # "slerp_ultra_mmed": {
        #     "method": "slerp",
        #     "models": models,             # 只使用兩個較小的模型
        #     "base_model": models[0],
        #     "parameters": {"t": 0.5},
        #     "desc": "ULtraMedical & MMed 的 SLERP"
        # }
        # Use SCE 
        "sce_deepseek_ultramed_mmed": {
            "method": "sce",
            "models": models,
            "base_model": models[0],
            "desc": "DeepSeek, UltraMed, and MMed 的 SCE"
        }
    }
    
    # 永遠選擇 slerp 合併方法
    # selected_methods = ["slerp_ultra_mmed"]
    selected_methods = ["sce_deepseek_ultramed_mmed"]
    # 建立輸出與配置檔資料夾
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.config_dir, exist_ok=True)
    
    # 逐個合併
    for key in selected_methods:
        if key not in all_methods:
            print(f"方法 {key} 不存在於預設清單，跳過。")
            continue
        
        info = all_methods[key]
        
        # 1. 建立 YAML 的 dict
        merge_cfg = create_merge_config(
            merge_method=info["method"],
            models=info["models"],
            base_model=info.get("base_model"),
            parameters=info.get("parameters")
        )
        
        # 2. 寫到 config_dir 下
        config_file_path = os.path.join(args.config_dir, f"{key}.yaml")
        if not save_config(merge_cfg, config_file_path):
            continue
        
        # 3. 為了讓輸出的資料夾更容易看懂，可用 key 當名稱
        output_model_dir = os.path.join(args.output_dir, key)
        
        # 4. 實際執行 mergekit-yaml
        if args.run:
            print(f"\n=== 準備合併: {info['desc']} ===")
            run_merge(
                config_file=config_file_path,
                output_dir=output_model_dir,
                use_cuda=(not args.no_cuda),
                allow_crimes=True,
                dry_run=args.dry_run
            )
        else:
            print(f"[僅建立 YAML] {key} 的配置檔已生成至 {config_file_path}，未執行合併。")
    
    print("\n所有指定的配置都已處理完成。")
    return 0

if __name__ == "__main__":
    sys.exit(main())
