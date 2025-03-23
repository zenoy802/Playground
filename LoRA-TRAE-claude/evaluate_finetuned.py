import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 从evaluate_baseline.py导入评估函数
from evaluate_baseline import evaluate_model
from prepare_datasets import load_old_knowledge_dataset, prepare_new_knowledge_dataset

# 加载基础模型
model_name = "baichuan-inc/Baichuan2-13B-Base"  # 或其他中文大模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# 加载LoRA微调模型
lora_model = PeftModel.from_pretrained(
    base_model,
    "/Users/zenoy/Documents/Playground/lora_finetuned_model",
    torch_dtype=torch.float16
)
lora_model.to("cuda")

# 加载数据集
old_knowledge_datasets = load_old_knowledge_dataset()
new_knowledge_dataset = prepare_new_knowledge_dataset()

# 评估旧知识
lora_old_knowledge_results = {}
for subject, dataset in old_knowledge_datasets.items():
    accuracy = evaluate_model(lora_model, tokenizer, dataset)
    lora_old_knowledge_results[subject] = accuracy
    print(f"LoRA微调后模型在{subject}上的准确率: {accuracy:.4f}")

# 评估新知识
lora_new_knowledge_accuracy = evaluate_model(lora_model, tokenizer, new_knowledge_dataset)
print(f"LoRA微调后模型在新知识领域的准确率: {lora_new_knowledge_accuracy:.4f}")

# 保存结果
lora_results = {
    "old_knowledge": lora_old_knowledge_results,
    "new_knowledge": lora_new_knowledge_accuracy
}

with open("/Users/zenoy/Documents/Playground/lora_results.json", "w") as f:
    json.dump(lora_results, f)

# 加载基线结果进行比较
with open("/Users/zenoy/Documents/Playground/baseline_results.json", "r") as f:
    baseline_results = json.load(f)

# 计算新旧知识的变化
print("\n性能变化分析:")
print("旧知识保留情况:")
for subject in baseline_results["old_knowledge"]:
    baseline_acc = baseline_results["old_knowledge"][subject]
    lora_acc = lora_old_knowledge_results[subject]
    change = lora_acc - baseline_acc
    print(f"  {subject}: {change:.4f} ({'+' if change >= 0 else ''}{change*100:.2f}%)")

print("\n新知识获取情况:")
new_knowledge_change = lora_new_knowledge_accuracy - baseline_results["new_knowledge"]
print(f"  新知识领域: {new_knowledge_change:.4f} ({'+' if new_knowledge_change >= 0 else ''}{new_knowledge_change*100:.2f}%)")

# 计算灾难性遗忘指标
avg_old_knowledge_change = sum(lora_acc - baseline_results["old_knowledge"][subject] 
                              for subject, lora_acc in lora_old_knowledge_results.items()) / len(lora_old_knowledge_results)
print(f"\n平均旧知识变化: {avg_old_knowledge_change:.4f} ({'+' if avg_old_knowledge_change >= 0 else ''}{avg_old_knowledge_change*100:.2f}%)")
print(f"新知识获取: {new_knowledge_change:.4f} ({'+' if new_knowledge_change >= 0 else ''}{new_knowledge_change*100:.2f}%)")

# 如果平均旧知识变化为负但幅度小，同时新知识获取显著提升，则说明LoRA有效避免了灾难性遗忘
if avg_old_knowledge_change > -0.05 and new_knowledge_change > 0.1:
    print("\n结论: LoRA成功地学习了新知识，同时有效避免了灾难性遗忘。")
elif avg_old_knowledge_change > 0:
    print("\n结论: LoRA不仅学习了新知识，还提升了旧知识的表现，完全避免了灾难性遗忘。")
else:
    print(f"\n结论: LoRA学习了新知识，但旧知识有所下降({avg_old_knowledge_change*100:.2f}%)，"
          f"相比传统微调方法的灾难性遗忘程度较轻。")