import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# 从prepare_datasets.py导入数据集
from prepare_datasets import prepare_new_knowledge_dataset
from lora_finetune import preprocess_function  # 复用数据预处理函数

# 加载模型和tokenizer
model_name = "baichuan-inc/Baichuan2-13B-Base"  # 或其他中文大模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# 准备训练数据
new_knowledge_dataset = prepare_new_knowledge_dataset()
processed_dataset = new_knowledge_dataset.map(preprocess_function, batched=True)

# 设置训练参数 - 注意学习率比LoRA低
training_args = TrainingArguments(
    output_dir="/Users/zenoy/Documents/Playground/full_finetune_output",
    per_device_train_batch_size=2,  # 更小的批量大小以适应内存
    gradient_accumulation_steps=8,  # 更大的梯度累积步数
    num_train_epochs=2,
    learning_rate=5e-5,  # 全参数微调使用更小的学习率
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
)

# 开始训练
trainer.train()

# 保存模型
model.save_pretrained("/Users/zenoy/Documents/Playground/full_finetuned_model")