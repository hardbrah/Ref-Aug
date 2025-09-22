from datasets import load_dataset
import pandas as pd
import json
import os

# 创建 dataset 文件夹（如果不存在）
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# 下载数据集并保存到 dataset 文件夹
dataset = load_dataset("Maxwell-Jia/AIME_2024", cache_dir="./dataset")

# 查看数据集信息
print(dataset)

# 将 DatasetDict 中的每个分割转换为 JSON
for split_name, split_dataset in dataset.items():
    # 将数据集转换为 pandas DataFrame
    df = split_dataset.to_pandas()
    
    # 确保输出目录存在
    output_dir = "./dataset/aime_2024_json"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存为 JSON 文件
    json_path = f"{output_dir}/{split_name}.json"
    df.to_json(json_path, orient="records", lines=False, force_ascii=False, indent=4)
    print(f"已将 {split_name} 数据保存到 {json_path}")