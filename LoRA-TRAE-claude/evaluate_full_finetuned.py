import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# 从evaluate_baseline.py导入评估函数
from evaluate_baseline import evaluate_model
from prepare_datasets import load_old_knowledge_dataset, prepare_new_knowledge_dataset

# 加载全参数微调模型
model_path = "/Users/zenoy/Documents/Playground/full_finetuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
full_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
full_model.to("cuda")

# 加载数据集
old_knowledge_datasets = load_old_knowledge_dataset()
new_knowledge_dataset = prepare_new_knowledge_dataset()

# 评估旧知识
full_old_knowledge_results = {}
for subject, dataset in old_knowledge_datasets.items():
    accuracy = evaluate_model(full_model, tokenizer, dataset)
    full_old_knowledge_results[subject] = accuracy
    print(f"全参数微调后模型在{subject}上的准确率: {accuracy:.4f}")

# 评估新知识
full_new_knowledge_accuracy = evaluate_model(full_model, tokenizer, new_knowledge_dataset)
print(f"全参数微调后模型在新知识领域的准确率: {full_new_knowledge_accuracy:.4f}")

# 保存结果
full_results = {
    "old_knowledge": full_old_knowledge_results,
    "new_knowledge": full_new_knowledge_accuracy
}

with open("/Users/zenoy/Documents/Playground/full_finetune_results.json", "w") as f:
    json.dump(full_results, f)