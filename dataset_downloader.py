#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可配置的数据集下载器
支持按名称下载不同的数据集
"""

import os
import time
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datasets import load_dataset
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def setup_network_config():
    """配置网络设置，包括重试机制和超时"""
    # 设置环境变量以改善网络连接
    os.environ.setdefault('HF_HUB_TIMEOUT', '60')  # 增加超时时间到60秒
    os.environ.setdefault('REQUESTS_TIMEOUT', '60')
    
    # 配置requests的重试策略
    session = requests.Session()
    retry_strategy = Retry(
        total=5,  # 总重试次数
        backoff_factor=2,  # 退避因子
        status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的HTTP状态码
        allowed_methods=["HEAD", "GET", "OPTIONS"]  # 允许重试的HTTP方法
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


def download_with_retry(func, max_retries=3, delay=5, **kwargs):
    """带重试机制的下载函数"""
    for attempt in range(max_retries):
        try:
            return func(**kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            # 检查是否是网络相关错误
            if any(keyword in error_msg for keyword in ['timeout', 'connection', 'network', 'httperror']):
                if attempt < max_retries - 1:
                    print(f"网络错误，{delay}秒后进行第{attempt + 2}次尝试...")
                    time.sleep(delay)
                    delay *= 2  # 指数退避
                    continue
            raise e
    return None


class DatasetDownloader(ABC):
    """数据集下载器抽象基类"""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    @abstractmethod
    def download(self, **kwargs) -> bool:
        """下载数据集的抽象方法"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """数据集名称"""
        pass


class GenericHuggingFaceDownloader(DatasetDownloader):
    """通用的Hugging Face数据集下载器"""
    
    def __init__(self, dataset_id: str, output_dir: str = "data", custom_name: Optional[str] = None):
        super().__init__(output_dir)
        self.dataset_id = dataset_id
        self.custom_name = custom_name or dataset_id.replace("/", "_").replace("-", "_")
        # 初始化网络配置
        setup_network_config()
    
    @property
    def name(self) -> str:
        return self.custom_name
    
    def _load_dataset_with_fallback(self, **load_args):
        """使用镜像源回退机制加载数据集"""
        # 首先尝试使用默认源
        try:
            print(f"尝试从默认源下载: {self.dataset_id}")
            return load_dataset(**load_args)
        except Exception as e:
            print(f"默认源下载失败: {str(e)}")
            
            # 尝试使用HF Mirror镜像源
            try:
                print(f"尝试从HF Mirror镜像源下载: {self.dataset_id}")
                # 设置镜像源环境变量
                original_endpoint = os.environ.get('HF_ENDPOINT')
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                
                result = load_dataset(**load_args)
                
                # 恢复原始环境变量
                if original_endpoint:
                    os.environ['HF_ENDPOINT'] = original_endpoint
                else:
                    os.environ.pop('HF_ENDPOINT', None)
                
                return result
            except Exception as mirror_e:
                print(f"镜像源下载也失败: {str(mirror_e)}")
                
                # 恢复原始环境变量
                if original_endpoint:
                    os.environ['HF_ENDPOINT'] = original_endpoint
                else:
                    os.environ.pop('HF_ENDPOINT', None)
                
                # 检查本地缓存
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets", self.dataset_id.replace("/", "___"))
                if os.path.exists(cache_dir):
                    print(f"尝试从本地缓存加载: {cache_dir}")
                    try:
                        # 尝试离线模式
                        load_args_offline = load_args.copy()
                        load_args_offline['download_mode'] = 'reuse_cache_if_exists'
                        return load_dataset(**load_args_offline)
                    except Exception as cache_e:
                        print(f"本地缓存加载失败: {str(cache_e)}")
                
                raise Exception(f"所有下载方式都失败了。原始错误: {str(e)}, 镜像源错误: {str(mirror_e)}")
    
    def download(self, 
                 split: Optional[str] = None,
                 config_name: Optional[str] = None,
                 sample_size: Optional[int] = None,
                 trust_remote_code: bool = True,
                 **kwargs) -> bool:
        """下载任意Hugging Face数据集
        
        Args:
            split: 数据集分割（如'train', 'test', 'validation'）
            config_name: 数据集配置名称
            sample_size: 采样大小，None表示使用全部数据
            trust_remote_code: 是否信任远程代码
            **kwargs: 其他传递给load_dataset的参数
        """
        print(f"下载Hugging Face数据集: {self.dataset_id}")
        
        try:
            # 准备load_dataset参数
            load_args = {
                'path': self.dataset_id,
                'trust_remote_code': trust_remote_code,
                **kwargs
            }
            
            if config_name:
                load_args['name'] = config_name
            if split:
                load_args['split'] = split
            
            # 使用重试机制和镜像源回退加载数据集
            dataset = download_with_retry(
                self._load_dataset_with_fallback,
                max_retries=3,
                delay=5,
                **load_args
            )
            
            if dataset is None:
                return False
                
            print(f"成功加载数据集: {self.dataset_id}")
            
            # 如果是DatasetDict，处理每个分割
            if hasattr(dataset, 'keys'):  # DatasetDict
                for split_name, split_dataset in dataset.items():
                    if sample_size and len(split_dataset) > sample_size:
                        split_dataset = split_dataset.shuffle(seed=42).select(range(sample_size))
                    
                    save_path = os.path.join(self.output_dir, f"{self.custom_name}_{split_name}")
                    split_dataset.save_to_disk(save_path)
                    print(f"数据集分割 {split_name} 已保存到 {save_path}，共{len(split_dataset)}条样本")
            else:  # Dataset
                if sample_size and len(dataset) > sample_size:
                    dataset = dataset.shuffle(seed=42).select(range(sample_size))
                
                save_path = os.path.join(self.output_dir, self.custom_name)
                dataset.save_to_disk(save_path)
                print(f"数据集已保存到 {save_path}，共{len(dataset)}条样本")
            
            return True
            
        except Exception as e:
            print(f"下载数据集 {self.dataset_id} 失败: {str(e)}")
            return False


class BelleDownloader(DatasetDownloader):
    """BELLE中文指令微调数据集下载器"""
    
    def __init__(self, output_dir: str = "data"):
        super().__init__(output_dir)
        # 初始化网络配置
        setup_network_config()
    
    @property
    def name(self) -> str:
        return "belle"
    
    def download(self, sample_size: int = 50000, **kwargs) -> bool:
        """下载BELLE数据集"""
        print("下载BELLE中文指令微调数据集...")
        
        try:
            # 尝试下载BELLE-2M数据集
            dataset = download_with_retry(
                lambda: load_dataset("BelleGroup/train_2M_CN", split="train"),
                max_retries=3,
                delay=5
            )
            if dataset:
                print("成功加载BELLE-2M数据集")
            else:
                raise Exception("重试后仍无法加载BELLE-2M")
        except Exception as e:
            print(f"无法加载BELLE-2M，尝试加载BELLE-1M: {str(e)}")
            try:
                dataset = download_with_retry(
                    lambda: load_dataset("BelleGroup/train_1M_CN", split="train"),
                    max_retries=3,
                    delay=5
                )
                if dataset:
                    print("成功加载BELLE-1M数据集")
                else:
                    raise Exception("重试后仍无法加载BELLE-1M")
            except Exception as e2:
                print(f"无法加载BELLE-1M，尝试加载更小的BELLE数据集: {str(e2)}")
                dataset = download_with_retry(
                    lambda: load_dataset("BelleGroup/generated_chat_0.4M", split="train"),
                    max_retries=3,
                    delay=5
                )
                if dataset:
                    print("成功加载BELLE-0.4M数据集")
                else:
                    return False
        
        # 采样处理
        if sample_size > 0 and len(dataset) > sample_size:
            dataset = dataset.shuffle(seed=42).select(range(sample_size))
        
        # 保存数据集
        save_path = os.path.join(self.output_dir, "belle_dataset")
        dataset.save_to_disk(save_path)
        print(f"BELLE数据集已保存到 {save_path}，共{len(dataset)}条样本")
        return True


class CEvalDownloader(DatasetDownloader):
    """C-Eval中文评估数据集下载器"""
    
    def __init__(self, output_dir: str = "data"):
        super().__init__(output_dir)
        # 初始化网络配置
        setup_network_config()
    
    @property
    def name(self) -> str:
        return "ceval"
    
    def download(self, tasks: Optional[List[str]] = None, **kwargs) -> bool:
        """下载C-Eval数据集"""
        print("下载C-Eval数据集...")
        
        if tasks is None:
            tasks = ['computer_network', 'operating_system', 'computer_architecture', 
                    'college_programming', 'college_physics', 'college_chemistry', 
                    'advanced_mathematics', 'probability_and_statistics', 'discrete_mathematics',
                    'electrical_engineer', 'metrology_engineer', 'high_school_mathematics',
                    'high_school_physics', 'high_school_chemistry', 'high_school_biology',
                    'middle_school_mathematics', 'middle_school_biology', 'middle_school_physics',
                    'middle_school_chemistry', 'veterinary_medicine', 'college_economics',
                    'business_administration', 'marxism', 'mao_zedong_thought', 'education_science',
                    'teacher_qualification', 'high_school_politics', 'high_school_geography',
                    'middle_school_politics', 'middle_school_geography', 'modern_chinese_history',
                    'ideological_and_moral_cultivation', 'logic', 'law', 'chinese_language_and_literature',
                    'art_studies', 'professional_tour_guide', 'legal_professional', 'high_school_chinese',
                    'high_school_history', 'middle_school_history', 'civil_servant', 'sports_science',
                    'plant_protection', 'basic_medicine', 'clinical_medicine', 'urban_and_rural_planner',
                    'accountant', 'fire_engineer', 'environmental_impact_assessment_engineer',
                    'tax_accountant', 'physician']
        
        success_count = 0
        for task in tasks:
            try:
                dataset = download_with_retry(
                    lambda: load_dataset("ceval/ceval-exam", name=task, trust_remote_code=True),
                    max_retries=3,
                    delay=5
                )
                if dataset:
                    save_path = os.path.join(self.output_dir, f"ceval_dataset/{task}")
                    dataset.save_to_disk(save_path)
                    print(f"C-Eval数据集 {task} 已保存到 {save_path}")
                    success_count += 1
                else:
                    print(f"重试后仍无法加载C-Eval任务 {task}")
            except Exception as e:
                print(f"无法加载C-Eval任务 {task}: {str(e)}")
        
        print(f"C-Eval数据集下载完成，成功下载 {success_count}/{len(tasks)} 个任务")
        return success_count > 0


class CMMluDownloader(DatasetDownloader):
    """CMMLU中文评估数据集下载器"""
    
    def __init__(self, output_dir: str = "data"):
        super().__init__(output_dir)
        # 初始化网络配置
        setup_network_config()
    
    @property
    def name(self) -> str:
        return "cmmlu"
    
    def download(self, tasks: Optional[List[str]] = None, **kwargs) -> bool:
        """下载CMMLU数据集"""
        print("下载CMMLU数据集...")
        
        if tasks is None:
            tasks = ['agronomy', 'anatomy', 'ancient_chinese', 'arts', 'astronomy', 'business_ethics',
                    'chinese_civil_service_exam', 'chinese_driving_rule', 'chinese_food_culture',
                    'chinese_foreign_policy', 'chinese_history', 'chinese_literature',
                    'chinese_teacher_qualification', 'clinical_knowledge', 'college_actuarial_science',
                    'college_education', 'college_engineering_hydrology', 'college_law',
                    'college_mathematics', 'college_medical_statistics', 'college_medicine',
                    'computer_science', 'computer_security', 'conceptual_physics',
                    'construction_project_management', 'economics', 'education', 'electrical_engineering',
                    'elementary_chinese', 'elementary_commonsense', 'elementary_information_and_technology',
                    'elementary_mathematics', 'ethnology', 'food_science', 'genetics', 'global_facts',
                    'high_school_biology', 'high_school_chemistry', 'high_school_geography',
                    'high_school_mathematics', 'high_school_physics', 'high_school_politics',
                    'human_sexuality', 'international_law', 'journalism', 'jurisprudence',
                    'legal_and_moral_basis', 'logical', 'machine_learning', 'management', 'marketing',
                    'marxist_theory', 'modern_chinese', 'nutrition', 'philosophy', 'professional_accounting',
                    'professional_law', 'professional_medicine', 'professional_psychology',
                    'public_relations', 'security_study', 'sociology', 'sports_science',
                    'traditional_chinese_medicine', 'virology', 'world_history', 'world_religions']
        
        success_count = 0
        for task in tasks:
            try:
                dataset = download_with_retry(
                    lambda: load_dataset("haonan-li/cmmlu", task, trust_remote_code=True),
                    max_retries=3,
                    delay=5
                )
                if dataset:
                    save_path = os.path.join(self.output_dir, f"cmmlu_dataset/{task}")
                    dataset.save_to_disk(save_path)
                    print(f"CMMLU数据集 {task} 已保存到 {save_path}")
                    success_count += 1
                else:
                    print(f"重试后仍无法加载CMMLU任务 {task}")
            except Exception as e:
                print(f"无法加载CMMLU任务 {task}: {str(e)}")
        
        print(f"CMMLU数据集下载完成，成功下载 {success_count}/{len(tasks)} 个任务")
        return success_count > 0


class MMLUDownloader(DatasetDownloader):
    """MMLU英文评估数据集下载器"""
    
    def __init__(self, output_dir: str = "data"):
        super().__init__(output_dir)
        # 初始化网络配置
        setup_network_config()
    
    @property
    def name(self) -> str:
        return "mmlu"
    
    def download(self, **kwargs) -> bool:
        """下载MMLU数据集"""
        print("下载MMLU数据集...")
        
        try:
            dataset = download_with_retry(
                lambda: load_dataset("cais/mmlu", "all", trust_remote_code=True),
                max_retries=3,
                delay=5
            )
            if not dataset:
                raise Exception("重试后仍无法加载MMLU")
        except Exception as e:
            print(f"无法以默认配置加载MMLU，尝试备用方式: {str(e)}")
            try:
                dataset = download_with_retry(
                    lambda: load_dataset("lukaemon/mmlu", "all"),
                    max_retries=3,
                    delay=5
                )
                if not dataset:
                    raise Exception("重试后仍无法加载备用MMLU")
            except Exception as e2:
                print(f"备用MMLU也无法加载，尝试加载MMLU的一个子集: {str(e2)}")
                dataset = download_with_retry(
                    lambda: load_dataset("cais/mmlu", "abstract_algebra"),
                    max_retries=3,
                    delay=5
                )
                if not dataset:
                    return False
        
        save_path = os.path.join(self.output_dir, "mmlu_dataset")
        dataset.save_to_disk(save_path)
        print(f"MMLU数据集已保存到 {save_path}")
        return True


class GSM8KDownloader(DatasetDownloader):
    """GSM8K数学推理数据集下载器"""
    
    def __init__(self, output_dir: str = "data"):
        super().__init__(output_dir)
        # 初始化网络配置
        setup_network_config()
    
    @property
    def name(self) -> str:
        return "gsm8k"
    
    def download(self, **kwargs) -> bool:
        """下载GSM8K数据集"""
        print("下载GSM8K数据集...")
        
        try:
            dataset = download_with_retry(
                lambda: load_dataset("gsm8k", "main", trust_remote_code=True),
                max_retries=3,
                delay=5
            )
            if dataset:
                save_path = os.path.join(self.output_dir, "gsm8k_dataset")
                dataset.save_to_disk(save_path)
                print(f"GSM8K数据集已保存到 {save_path}")
                return True
            else:
                print("重试后仍无法加载GSM8K数据集")
                return False
        except Exception as e:
            print(f"无法加载GSM8K数据集: {str(e)}")
            return False


class HellaSwagDownloader(DatasetDownloader):
    """HellaSwag常识推理数据集下载器"""
    
    def __init__(self, output_dir: str = "data"):
        super().__init__(output_dir)
        # 初始化网络配置
        setup_network_config()
    
    @property
    def name(self) -> str:
        return "hellaswag"
    
    def download(self, **kwargs) -> bool:
        """下载HellaSwag数据集"""
        print("下载HellaSwag数据集...")
        
        try:
            dataset = download_with_retry(
                lambda: load_dataset("Rowan/hellaswag", trust_remote_code=True),
                max_retries=3,
                delay=5
            )
            if dataset:
                save_path = os.path.join(self.output_dir, "hellaswag_dataset")
                dataset.save_to_disk(save_path)
                print(f"HellaSwag数据集已保存到 {save_path}")
                return True
            else:
                print("重试后仍无法加载HellaSwag数据集")
                return False
        except Exception as e:
            print(f"无法加载HellaSwag数据集: {str(e)}")
            return False


class DatasetManager:
    """数据集管理器，统一管理所有数据集下载器"""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        self.downloaders: Dict[str, DatasetDownloader] = {}
        self._register_downloaders()
    
    def _register_downloaders(self):
        """注册所有数据集下载器"""
        downloaders = [
            BelleDownloader(self.output_dir),
            CEvalDownloader(self.output_dir),
            CMMluDownloader(self.output_dir),
            MMLUDownloader(self.output_dir),
            GSM8KDownloader(self.output_dir),
            HellaSwagDownloader(self.output_dir)
        ]
        
        for downloader in downloaders:
            self.downloaders[downloader.name] = downloader
    
    def list_datasets(self) -> List[str]:
        """列出所有可用的数据集名称"""
        return list(self.downloaders.keys())
    
    def download_dataset(self, dataset_name: str, **kwargs) -> bool:
        """下载指定名称的数据集"""
        if dataset_name not in self.downloaders:
            # 如果不是预定义的数据集，尝试作为Hugging Face数据集ID处理
            print(f"未找到预定义数据集: {dataset_name}")
            print(f"尝试作为Hugging Face数据集ID下载...")
            return self.download_huggingface_dataset(dataset_name, **kwargs)
        
        print(f"开始下载数据集: {dataset_name}")
        return self.downloaders[dataset_name].download(**kwargs)
    
    def download_huggingface_dataset(self, dataset_id: str, custom_name: Optional[str] = None, **kwargs) -> bool:
        """下载任意Hugging Face数据集
        
        Args:
            dataset_id: Hugging Face数据集ID（如'microsoft/DialoGPT-medium'）
            custom_name: 自定义数据集名称，用于保存目录
            **kwargs: 传递给GenericHuggingFaceDownloader的参数
        """
        try:
            downloader = GenericHuggingFaceDownloader(
                dataset_id=dataset_id,
                output_dir=self.output_dir,
                custom_name=custom_name
            )
            return downloader.download(**kwargs)
        except Exception as e:
            print(f"下载Hugging Face数据集 {dataset_id} 失败: {str(e)}")
            return False
    
    def register_custom_downloader(self, downloader: DatasetDownloader):
        """注册自定义数据集下载器"""
        self.downloaders[downloader.name] = downloader
        print(f"已注册自定义数据集下载器: {downloader.name}")
    
    def list_predefined_datasets(self) -> List[str]:
        """列出所有预定义的数据集名称"""
        return list(self.downloaders.keys())
    
    def download_multiple_datasets(self, dataset_names: List[str], **kwargs) -> Dict[str, bool]:
        """下载多个数据集"""
        results = {}
        for dataset_name in dataset_names:
            results[dataset_name] = self.download_dataset(dataset_name, **kwargs)
        return results
    
    def download_all_datasets(self, **kwargs) -> Dict[str, bool]:
        """下载所有数据集"""
        return self.download_multiple_datasets(self.list_datasets(), **kwargs)