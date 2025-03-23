import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

# 从prepare_datasets.py导入数据集
from prepare_datasets import prepare_new_knowledge_dataset

# 加载模型和tokenizer
model_name = "baichuan-inc/Baichuan2-13B-Base"  # 或其他中文大模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# 配置LoRA
lora_config = LoraConfig(
    r=8,                      # LoRA的秩
    lora_alpha=16,            # LoRA的alpha参数
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 要应用LoRA的模块
    lora_dropout=0.05,        # LoRA的dropout率
    bias="none",              # 是否训练偏置项
    task_type=TaskType.CAUSAL_LM  # 任务类型
)

# 应用LoRA配置
model = get_peft_model(model, lora_config)
print(f"可训练参数数量: {model.num_parameters(True)}")
print(f"总参数数量: {model.num_parameters()}")

# 准备训练数据
new_knowledge_dataset = prepare_new_knowledge_dataset()

# 数据预处理函数
def preprocess_function(examples):
    # 将指令和输入组合成提示
    prompts = [f"指令: {instruction}\n输入: {inp}\n输出:" 
              for instruction, inp in zip(examples["instruction"], examples["input"])]
    
    # 编码提示和输出
    inputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=512)
    outputs = tokenizer(examples["output"], padding="max_length", truncation=True, max_length=512)
    
    # 准备输入IDs和标签
    input_ids = inputs.input_ids
    labels = outputs.input_ids.copy()
    
    # 将填充token的标签设为-100，以便在计算损失时忽略
    for i in range(len(labels)):
        labels[i] = [-100 if token == tokenizer.pad_token_id else token for token in labels[i]]
    
    return {
        "input_ids": input_ids,
        "attention_mask": inputs.attention_mask,
        "labels": labels
    }

# 预处理数据集
processed_dataset = new_knowledge_dataset.map(preprocess_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="/Users/zenoy/Documents/Playground/lora_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
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
model.save_pretrained("/Users/zenoy/Documents/Playground/lora_finetuned_model")