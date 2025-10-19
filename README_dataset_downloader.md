# 数据集下载工具

这是一个功能强大的数据集下载工具，支持下载预定义的数据集以及任意的Hugging Face数据集。

## 文件结构

- `dataset_downloader.py` - 核心数据集下载器模块，包含抽象基类和具体实现
- `download_datasets_cli.py` - 命令行接口，提供便捷的数据集下载功能
- `download_data.py` - 重构后的主程序，使用新的抽象接口
- `README_dataset_downloader.md` - 本文档

## 支持的预定义数据集

- **belle** - BELLE中文指令微调数据集
- **ceval** - C-Eval中文评估数据集
- **cmmlu** - CMMLU中文评估数据集
- **mmlu** - MMLU英文评估数据集
- **gsm8k** - GSM8K数学推理数据集
- **hellaswag** - HellaSwag常识推理数据集

## 新功能：任意Hugging Face数据集支持

除了预定义数据集外，现在还支持下载任意的Hugging Face数据集！

### 支持的功能
- 下载任意Hugging Face Hub上的数据集
- 支持指定数据集分割（train/test/validation等）
- 支持指定数据集配置（对于有多个配置的数据集）
- 支持数据采样（指定样本数量）
- 支持自定义保存名称
- 自动处理数据集元数据和缓存

## 使用方法

### 1. 命令行使用

#### 列出所有预定义数据集
```bash
python download_datasets_cli.py --list
```

#### 下载预定义数据集
```bash
# 下载单个数据集
python download_datasets_cli.py --datasets belle

# 下载多个数据集
python download_datasets_cli.py --datasets belle ceval mmlu

# 下载所有预定义数据集
python download_datasets_cli.py --datasets all

# 指定输出目录
python download_datasets_cli.py --datasets belle --output-dir ./my_data
```

#### 下载任意Hugging Face数据集
```bash
# 下载完整数据集
python download_datasets_cli.py --huggingface microsoft/DialoGPT-medium

# 下载指定分割
python download_datasets_cli.py --huggingface squad --split train

# 下载并采样
python download_datasets_cli.py --huggingface imdb --split train --sample-size 1000

# 指定配置和自定义名称
python download_datasets_cli.py --huggingface glue --config-name cola --custom-name glue_cola

# 组合多个参数
python download_datasets_cli.py --huggingface wikitext --config-name wikitext-2-raw-v1 --split train --sample-size 5000 --custom-name wikitext2_sample
```

#### 预定义数据集的特定参数
```bash
# BELLE数据集采样
python download_datasets_cli.py --datasets belle --belle-sample-size 10000

# C-Eval指定任务
python download_datasets_cli.py --datasets ceval --ceval-tasks computer_network operating_system

# CMMLU指定任务
python download_datasets_cli.py --datasets cmmlu --cmmlu-tasks anatomy clinical_medicine
```

### 2. 编程使用

#### 使用DatasetManager下载预定义数据集
```python
from dataset_downloader import DatasetManager

# 创建数据集管理器
manager = DatasetManager(output_dir="./data")

# 下载单个数据集
success = manager.download_dataset("belle", sample_size=10000)

# 下载多个数据集
results = manager.download_multiple_datasets(["belle", "ceval", "mmlu"])

# 下载所有数据集
all_results = manager.download_all_datasets()
```

#### 使用DatasetManager下载任意Hugging Face数据集
```python
from dataset_downloader import DatasetManager

manager = DatasetManager(output_dir="./data")

# 下载完整数据集
success = manager.download_huggingface_dataset("microsoft/DialoGPT-medium")

# 下载指定分割和采样
success = manager.download_huggingface_dataset(
    dataset_id="squad",
    split="train",
    sample_size=1000,
    custom_name="squad_train_sample"
)

# 下载指定配置
success = manager.download_huggingface_dataset(
    dataset_id="glue",
    config_name="cola",
    custom_name="glue_cola"
)
```

#### 直接使用GenericHuggingFaceDownloader
```python
from dataset_downloader import GenericHuggingFaceDownloader

# 创建下载器
downloader = GenericHuggingFaceDownloader(
    dataset_id="imdb",
    output_dir="./data",
    custom_name="imdb_reviews"
)

# 下载数据集
success = downloader.download(
    split="train",
    sample_size=5000,
    trust_remote_code=True
)
```

### 3. 扩展：添加自定义下载器

```python
from dataset_downloader import DatasetDownloader, DatasetManager

class MyCustomDownloader(DatasetDownloader):
    @property
    def name(self) -> str:
        return "my_custom_dataset"
    
    def download(self, **kwargs) -> bool:
        # 实现自定义下载逻辑
        print(f"下载自定义数据集到: {self.output_dir}")
        return True

# 注册自定义下载器
manager = DatasetManager()
manager.register_custom_downloader(MyCustomDownloader("./data"))

# 使用自定义下载器
success = manager.download_dataset("my_custom_dataset")
```

## 命令行参数说明

### 基本参数
- `--output-dir` - 数据保存目录（默认：data）
- `--list` - 列出所有预定义数据集
- `--datasets` - 指定要下载的预定义数据集名称（支持多个）
- `--huggingface` - 指定要下载的Hugging Face数据集ID

### Hugging Face数据集参数
- `--split` - 指定数据集分割（如train、test、validation）
- `--config-name` - 指定数据集配置名称（某些数据集有多个配置）
- `--sample-size` - 从数据集中采样的样本数
- `--custom-name` - 为数据集指定自定义保存名称
- `--trust-remote-code` - 是否信任远程代码（默认：True）

### 预定义数据集特定参数
- `--belle-sample-size` - BELLE数据集采样大小（默认：50000，-1表示全部）
- `--ceval-tasks` - C-Eval数据集指定任务列表
- `--cmmlu-tasks` - CMMLU数据集指定任务列表

## 常见Hugging Face数据集示例

### 文本分类数据集
```bash
# IMDb电影评论情感分析
python download_datasets_cli.py --huggingface imdb --split train

# AG News新闻分类
python download_datasets_cli.py --huggingface ag_news --split train --sample-size 10000
```

### 问答数据集
```bash
# SQuAD阅读理解
python download_datasets_cli.py --huggingface squad --split train

# MS MARCO问答
python download_datasets_cli.py --huggingface ms_marco --config-name v1.1 --split train
```

### 语言建模数据集
```bash
# WikiText-2
python download_datasets_cli.py --huggingface wikitext --config-name wikitext-2-raw-v1

# OpenWebText
python download_datasets_cli.py --huggingface openwebtext --sample-size 50000
```

### 多任务数据集
```bash
# GLUE基准测试
python download_datasets_cli.py --huggingface glue --config-name cola
python download_datasets_cli.py --huggingface glue --config-name sst2
python download_datasets_cli.py --huggingface glue --config-name mrpc
```

## 环境要求

- Python 3.7+
- datasets库
- 网络连接（用于下载数据集）

## 注意事项

1. 首次下载可能需要较长时间，取决于网络速度和数据集大小
2. 确保有足够的磁盘空间存储数据集
3. 某些数据集可能需要Hugging Face账户认证
4. 如果下载失败，工具会自动尝试备用数据源

## 错误处理

工具内置了完善的错误处理机制：
- 网络连接失败时会显示详细错误信息
- 数据集不存在时会尝试备用数据源
- 部分下载失败不会影响其他数据集的下载
- 提供详细的下载进度和结果统计