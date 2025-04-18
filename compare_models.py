#!/usr/bin/env python
# 比較所有模型在醫療數據集上的表現
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import glob

def load_metrics(result_dirs):
    """從多個結果目錄加載指標"""
    all_metrics = {}
    for result_dir in result_dirs:
        model_name = os.path.basename(result_dir.rstrip("/"))
        if model_name.startswith("eval_results_"):
            model_name = model_name[len("eval_results_"):]
        
        metrics_file = os.path.join(result_dir, "overall_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                all_metrics[model_name] = metrics
        else:
            print(f"警告: 找不到指標文件 {metrics_file}")
    
    return all_metrics

def create_comparison_dataframe(all_metrics):
    """創建用於比較的數據框架"""
    rows = []
    
    # 遍歷所有模型和數據集
    for model_name, metrics in all_metrics.items():
        for dataset_name, dataset_metrics in metrics.items():
            for metric_name, value in dataset_metrics.items():
                rows.append({
                    "Model": model_name,
                    "Dataset": dataset_name,
                    "Metric": metric_name,
                    "Value": value
                })
    
    return pd.DataFrame(rows)

def plot_comparison(df, output_dir):
    """繪製比較圖表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 獲取唯一的數據集和指標
    datasets = df['Dataset'].unique()
    metrics = df['Metric'].unique()
    
    # 為每個指標創建一個圖表，每個圖表包含所有數據集
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        # 過濾該指標的數據
        metric_df = df[df['Metric'] == metric]
        
        # 按模型分組並按數據集進行透視
        pivot_df = metric_df.pivot(index='Model', columns='Dataset', values='Value')
        
        # 繪製條形圖
        ax = pivot_df.plot(kind='bar', figsize=(12, 8))
        
        plt.title(f'{metric} Comparison Across Models and Datasets', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Dataset', fontsize=12)
        
        # 添加數值標籤
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_comparison.png"), dpi=300)
        plt.close()
    
    # 為每個數據集創建一個圖表，每個圖表包含所有指標
    for dataset in datasets:
        plt.figure(figsize=(12, 8))
        
        # 過濾該數據集的數據
        dataset_df = df[df['Dataset'] == dataset]
        
        # 按模型和指標進行透視
        pivot_df = dataset_df.pivot(index='Model', columns='Metric', values='Value')
        
        # 繪製條形圖
        ax = pivot_df.plot(kind='bar', figsize=(12, 8))
        
        plt.title(f'Performance on {dataset} Dataset', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Metric', fontsize=12)
        
        # 添加數值標籤
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset}_comparison.png"), dpi=300)
        plt.close()
    
    # 創建熱力圖，顯示所有模型在所有數據集上的表現（對每個指標）
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        # 過濾該指標的數據
        metric_df = df[df['Metric'] == metric]
        
        # 創建透視表
        pivot_table = metric_df.pivot_table(values='Value', index='Model', columns='Dataset')
        
        # 繪製熱力圖
        ax = plt.pcolor(pivot_table, cmap='YlGnBu')
        
        # 設置坐標軸標籤
        plt.colorbar(ax)
        plt.yticks(np.arange(0.5, len(pivot_table.index), 1), pivot_table.index)
        plt.xticks(np.arange(0.5, len(pivot_table.columns), 1), pivot_table.columns, rotation=45, ha='right')
        
        plt.title(f'Heatmap of {metric} Scores', fontsize=16)
        
        # 添加數值標籤
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                plt.text(j + 0.5, i + 0.5, f'{pivot_table.iloc[i, j]:.3f}',
                        ha='center', va='center', color='black')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_heatmap.png"), dpi=300)
        plt.close()
    
    return

def create_summary_table(df, output_dir):
    """創建摘要表格"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 計算每個模型在每個指標上的平均分數
    avg_scores = df.groupby(['Model', 'Metric'])['Value'].mean().reset_index()
    pivot_avg = avg_scores.pivot(index='Model', columns='Metric', values='Value')
    
    # 添加平均列
    pivot_avg['Average'] = pivot_avg.mean(axis=1)
    
    # 排序
    pivot_avg = pivot_avg.sort_values('Average', ascending=False)
    
    # 保存為CSV
    pivot_avg.to_csv(os.path.join(output_dir, "summary_table.csv"))
    
    # 創建摘要表圖形
    plt.figure(figsize=(10, len(pivot_avg) * 0.5 + 2))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    # 打印表格
    table = plt.table(
        cellText=np.round(pivot_avg.values, 3),
        rowLabels=pivot_avg.index,
        colLabels=pivot_avg.columns,
        cellLoc='center',
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.title("Model Performance Summary", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_table.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return pivot_avg

def main():
    parser = argparse.ArgumentParser(description="比較多個模型的性能")
    parser.add_argument("--result-dirs", nargs='+', required=True, help="評估結果目錄")
    parser.add_argument("--output-dir", type=str, default="model_comparisons", help="輸出目錄")
    args = parser.parse_args()
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加載所有指標
    all_metrics = load_metrics(args.result_dirs)
    
    # 如果沒有找到任何指標，嘗試使用通配符
    if not all_metrics:
        expanded_dirs = []
        for pattern in args.result_dirs:
            expanded_dirs.extend(glob.glob(pattern))
        all_metrics = load_metrics(expanded_dirs)
    
    if not all_metrics:
        print("未找到任何指標文件")
        return
    
    # 創建比較數據框
    comparison_df = create_comparison_dataframe(all_metrics)
    
    # 繪製比較圖表
    plot_comparison(comparison_df, args.output_dir)
    
    # 創建摘要表格
    summary = create_summary_table(comparison_df, args.output_dir)
    
    # 打印摘要
    print("\n模型性能摘要:")
    print(summary)
    print(f"\n比較結果已保存到 {args.output_dir} 目錄")

if __name__ == "__main__":
    main() 