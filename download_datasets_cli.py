#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集下载命令行工具
支持按名称下载指定的数据集
"""

import os
import argparse
import sys
from typing import List
from dataset_downloader import DatasetManager

# 设置Hugging Face缓存目录
os.environ['HF_HOME'] = '/root/autodl-tmp/.cache/'


def main():
    parser = argparse.ArgumentParser(
        description="数据集下载工具 - 支持下载预定义数据集和任意Hugging Face数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --list                           # 列出所有预定义数据集
  %(prog)s --datasets belle                 # 下载BELLE数据集
  %(prog)s --datasets belle ceval mmlu      # 下载多个数据集
  %(prog)s --datasets all                   # 下载所有预定义数据集
  %(prog)s --datasets belle --belle-sample-size 10000  # 下载BELLE数据集并设置采样大小
  %(prog)s --huggingface microsoft/DialoGPT-medium     # 下载任意Hugging Face数据集
  %(prog)s --huggingface squad --split train --sample-size 1000  # 下载指定分割和采样
        """
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="/root/autodl-tmp/data", 
        help="数据保存目录 (默认: data)"
    )
    
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        help="要下载的数据集名称，支持: belle, ceval, cmmlu, mmlu, gsm8k, hellaswag, all"
    )
    
    parser.add_argument(
        "--huggingface", 
        type=str,
        help="下载任意Hugging Face数据集，指定数据集ID（如'microsoft/DialoGPT-medium'）"
    )
    
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="列出所有预定义的数据集名称"
    )
    
    # 通用Hugging Face数据集参数
    parser.add_argument(
        "--split", 
        type=str,
        help="指定要下载的数据集分割（如'train', 'test', 'validation'）"
    )
    
    parser.add_argument(
        "--config-name", 
        type=str,
        help="指定数据集配置名称（某些数据集有多个配置）"
    )
    
    parser.add_argument(
        "--sample-size", 
        type=int,
        help="从数据集中采样的样本数，不指定则使用全部数据"
    )
    
    parser.add_argument(
        "--custom-name", 
        type=str,
        help="为Hugging Face数据集指定自定义名称（用于保存目录）"
    )
    
    parser.add_argument(
        "--trust-remote-code", 
        action="store_true",
        default=True,
        help="是否信任远程代码（默认: True）"
    )
    
    # BELLE数据集特定参数
    parser.add_argument(
        "--belle-sample-size", 
        type=int, 
        default=50000,
        help="从BELLE数据集中采样的样本数，设为-1表示使用全部数据 (默认: 50000)"
    )
    
    # C-Eval数据集特定参数
    parser.add_argument(
        "--ceval-tasks", 
        nargs="*",
        help="指定要下载的C-Eval任务列表，不指定则下载所有任务"
    )
    
    # CMMLU数据集特定参数
    parser.add_argument(
        "--cmmlu-tasks", 
        nargs="*",
        help="指定要下载的CMMLU任务列表，不指定则下载所有任务"
    )
    
    args = parser.parse_args()
    
    # 创建数据集管理器
    manager = DatasetManager(args.output_dir)
    
    # 如果用户要求列出数据集
    if args.list:
        print("预定义的数据集:")
        for dataset_name in manager.list_predefined_datasets():
            print(f"  - {dataset_name}")
        print("\n注意: 除了预定义数据集外，还可以使用 --huggingface 参数下载任意Hugging Face数据集")
        return
    
    # 处理Hugging Face数据集下载
    if args.huggingface:
        print(f"开始下载Hugging Face数据集: {args.huggingface}")
        print(f"保存到目录: {args.output_dir}")
        
        # 准备Hugging Face数据集参数
        hf_kwargs = {}
        if args.split:
            hf_kwargs["split"] = args.split
        if args.config_name:
            hf_kwargs["config_name"] = args.config_name
        if args.sample_size:
            hf_kwargs["sample_size"] = args.sample_size
        if args.trust_remote_code is not None:
            hf_kwargs["trust_remote_code"] = args.trust_remote_code
        
        try:
            success = manager.download_huggingface_dataset(
                dataset_id=args.huggingface,
                custom_name=args.custom_name,
                **hf_kwargs
            )
            if success:
                print(f"✓ Hugging Face数据集 {args.huggingface} 下载成功")
            else:
                print(f"✗ Hugging Face数据集 {args.huggingface} 下载失败")
                sys.exit(1)
        except Exception as e:
            print(f"✗ Hugging Face数据集 {args.huggingface} 下载出错: {str(e)}")
            sys.exit(1)
        return
    
    # 检查是否指定了数据集
    if not args.datasets:
        print("错误: 请指定要下载的数据集名称或使用 --huggingface 下载Hugging Face数据集")
        print("使用 --list 查看所有预定义数据集")
        print("使用 --help 查看详细帮助信息")
        sys.exit(1)
    
    # 处理"all"特殊情况
    if "all" in args.datasets:
        datasets_to_download = manager.list_predefined_datasets()
    else:
        datasets_to_download = args.datasets
    
    # 验证数据集名称（允许任意Hugging Face数据集ID）
    predefined_datasets = manager.list_predefined_datasets()
    unknown_datasets = []
    for name in datasets_to_download:
        if name not in predefined_datasets:
            # 检查是否看起来像Hugging Face数据集ID
            if "/" not in name and "-" not in name:
                unknown_datasets.append(name)
    
    if unknown_datasets:
        print(f"警告: 以下数据集不在预定义列表中: {', '.join(unknown_datasets)}")
        print("将尝试作为Hugging Face数据集ID下载...")
        print(f"预定义的数据集: {', '.join(predefined_datasets)}")
        print("-" * 50)
    
    print(f"开始下载数据集到目录: {args.output_dir}")
    print(f"要下载的数据集: {', '.join(datasets_to_download)}")
    print("-" * 50)
    
    # 下载数据集
    success_count = 0
    total_count = len(datasets_to_download)
    
    for dataset_name in datasets_to_download:
        print(f"\n[{success_count + 1}/{total_count}] 正在下载: {dataset_name}")
        
        # 准备数据集特定的参数
        kwargs = {}
        if dataset_name == "belle":
            kwargs["sample_size"] = args.belle_sample_size
        elif dataset_name == "ceval" and args.ceval_tasks is not None:
            kwargs["tasks"] = args.ceval_tasks
        elif dataset_name == "cmmlu" and args.cmmlu_tasks is not None:
            kwargs["tasks"] = args.cmmlu_tasks
        
        # 下载数据集
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
        sys.exit(1)


if __name__ == "__main__":
    main()