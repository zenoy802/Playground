#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载实验所需的所有数据集 (使用新的抽象接口)
"""

import os
import argparse
from dataset_downloader import DatasetManager

os.environ['HF_HOME'] = '/root/autodl-tmp/.cache/'

def main(args):
    """使用新的抽象接口下载所有数据集"""
    print("开始下载数据集...")
    
    # 创建数据集管理器
    manager = DatasetManager(args.output_dir)
    
    # 定义要下载的数据集和参数
    datasets_config = {
        'belle': {'sample_size': args.belle_sample_size},
        'ceval': {},
        'cmmlu': {},
        'mmlu': {},
        'gsm8k': {},
        'hellaswag': {}
    }
    
    # 下载所有数据集
    success_count = 0
    total_count = len(datasets_config)
    
    for dataset_name, kwargs in datasets_config.items():
        print(f"\n[{success_count + 1}/{total_count}] 正在下载: {dataset_name}")
        try:
            success = manager.download_dataset(dataset_name, **kwargs)
            if success:
                success_count += 1
                print(f"✓ {dataset_name} 下载成功")
            else:
                print(f"✗ {dataset_name} 下载失败")
        except Exception as e:
            print(f"✗ {dataset_name} 下载出错: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"下载完成! 成功: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("所有数据集下载成功!")
    elif success_count > 0:
        print("部分数据集下载成功，请检查失败的数据集")
    else:
        print("所有数据集下载失败，请检查网络连接和配置")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载实验所需的所有数据集")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/data", help="数据保存目录")
    parser.add_argument("--belle_sample_size", type=int, default=50000, 
                        help="从BELLE数据集中采样的样本数，设为-1表示使用全部数据")
    args = parser.parse_args()
    
    main(args)