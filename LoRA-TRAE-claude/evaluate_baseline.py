import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def evaluate_model(model, tokenizer, dataset, max_length=512):
    model.eval()
    correct = 0
    total = 0
    
    for item in tqdm(dataset):
        question = item["question"]
        options = item["choices"]
        answer = item["answer"]
        
        # 为每个选项生成完整的问题
        prompts = [f"问题: {question}\n选项: {option}\n这个选项正确吗?" for option in options]
        
        scores = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
            # 计算"是"的概率作为分数
            yes_token_id = tokenizer.encode("是")[0]
            yes_score = logits[0, -1, yes_token_id].item()
            scores.append(yes_score)
        
        # 选择分数最高的选项
        predicted = scores.index(max(scores))
        correct_idx = options.index(answer)
        
        if predicted == correct_idx:
            correct += 1
        total += 1
    
    return correct / total

# 加载预训练模型
model_name = "baichuan-inc/Baichuan2-13B-Base"  # 或其他中文大模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.to("cuda")

# 从prepare_datasets.py导入数据集加载函数
from prepare_datasets import load_old_knowledge_dataset, prepare_new_knowledge_dataset

# 评估旧知识
old_knowledge_datasets = load_old_knowledge_dataset()
old_knowledge_results = {}
for subject, dataset in old_knowledge_datasets.items():
    accuracy = evaluate_model(model, tokenizer, dataset)
    old_knowledge_results[subject] = accuracy
    print(f"基线模型在{subject}上的准确率: {accuracy:.4f}")

# 评估新知识领域
new_knowledge_dataset = prepare_new_knowledge_dataset()
new_knowledge_accuracy = evaluate_model(model, tokenizer, new_knowledge_dataset)
print(f"基线模型在新知识领域的准确率: {new_knowledge_accuracy:.4f}")

# 保存基线结果
import json
with open("/Users/zenoy/Documents/Playground/baseline_results.json", "w") as f:
    json.dump({
        "old_knowledge": old_knowledge_results,
        "new_knowledge": new_knowledge_accuracy
    }, f)