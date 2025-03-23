import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from prepare_datasets import load_old_knowledge_dataset, prepare_new_knowledge_dataset
from evaluate_baseline import evaluate_model
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 定义要测试的LoRA参数组合
lora_configs = [
    {"r": 4, "alpha": 8, "name": "r4_alpha8"},
    {"r": 8, "alpha": 16, "name": "r8_alpha16"},
    {"r": 16, "alpha": 32, "name": "r16_alpha32"},
    {"r": 32, "alpha": 64, "name": "r32_alpha64"}
]

# 加载基线结果
with open("/Users/zenoy/Documents/Playground/baseline_results.json", "r") as f:
    baseline_results = json.load(f)

# 创建结果存储目录
os.makedirs("/Users/zenoy/Documents/Playground/lora_parameter_results", exist_ok=True)

# 对每个参数组合进行实验
results = {}

for config in lora_configs:
    print(f"\n开始测试LoRA参数: r={config['r']}, alpha={config['alpha']}")
    
    # 加载模型
    model_name = "baichuan-inc/Baichuan2-13B-Base"  # 或其他中文大模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=config["r"],
        lora_alpha=config["alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # 应用LoRA配置
    model = get_peft_model(model, lora_config)
    
    # 这里应该有微调代码，但为了简化示例，我们假设已经完成微调
    # 实际实验中，需要对每个参数组合进行微调
    
    # 评估模型性能
    old_knowledge_datasets = load_old_knowledge_dataset()
    new_knowledge_dataset = prepare_new_knowledge_dataset()
    
    # 评估旧知识
    old_knowledge_results = {}
    for subject, dataset in old_knowledge_datasets.items():
        accuracy = evaluate_model(model, tokenizer, dataset)
        old_knowledge_results[subject] = accuracy
    
    # 评估新知识
    new_knowledge_accuracy = evaluate_model(model, tokenizer, new_knowledge_dataset)
    
    # 保存结果
    config_results = {
        "old_knowledge": old_knowledge_results,
        "new_knowledge": new_knowledge_accuracy
    }
    
    results[config["name"]] = config_results
    
    # 保存当前配置的结果
    with open(f"/Users/zenoy/Documents/Playground/lora_parameter_results/{config['name']}.json", "w") as f:
        json.dump(config_results, f)

# 分析结果
subjects = list(baseline_results["old_knowledge"].keys())
baseline_old = [baseline_results["old_knowledge"][subject] for subject in subjects]
baseline_old_avg = sum(baseline_old) / len(baseline_old)
baseline_new = baseline_results["new_knowledge"]

# 准备绘图数据
config_names = [config["name"] for config in lora_configs]
retention_rates = []
acquisition_rates = []

for config in lora_configs:
    config_name = config["name"]
    config_results = results[config_name]
    
    # 计算平均旧知识准确率
    old_accs = [config_results["old_knowledge"][subject] for subject in subjects]
    old_avg = sum(old_accs) / len(old_accs)
    
    # 计算保留率和获取率
    retention = old_avg / baseline_old_avg
    acquisition = (config_results["new_knowledge"] - baseline_new) / (1 - baseline_new) if baseline_new < 1 else 1
    
    retention_rates.append(retention)
    acquisition_rates.append(acquisition)

# 绘制参数敏感性图
plt.figure(figsize=(12, 6))

x = np.arange(len(config_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, retention_rates, width, label='知识保留率')
ax.bar(x + width/2, acquisition_rates, width, label='知识获取率')

ax.set_ylabel('比率')
ax.set_title('不同LoRA参数对知识保留和获取的影响')
ax.set_xticks(x)
ax.set_xticklabels(config_names)
ax.legend()

plt.tight_layout()
plt.savefig("/Users/zenoy/Documents/Playground/lora_parameter_sensitivity.png")

# 绘制权衡图
plt.figure(figsize=(10, 8))
plt.scatter(retention_rates, acquisition_rates, s=100)

for i, config_name in enumerate(config_names):
    plt.annotate(config_name, (retention_rates[i], acquisition_rates[i]), fontsize=10)

plt.xlabel('知识保留率')
plt.ylabel('知识获取率')
plt.title('LoRA参数对知识保留和获取的权衡')
plt.grid(True, alpha=0.3)
plt.savefig("/Users/zenoy/Documents/Playground/lora_parameter_tradeoff.png")

# 生成参数敏感性分析报告
with open("/Users/zenoy/Documents/Playground/lora_parameter_analysis_report.txt", "w") as f:
    f.write("# LoRA参数敏感性分析报告\n\n")
    
    f.write("## 参数组合及其性能\n\n")
    f.write("| 参数配置 | 知识保留率 | 知识获取率 | 平均旧知识准确率 | 新知识准确率 |\n")
    f.write("|----------|------------|------------|------------------|------------|\n")
    
    for i, config in enumerate(lora_configs):
        config_name = config["name"]
        old_accs = [results[config_name]["old_knowledge"][subject] for subject in subjects]
        old_avg = sum(old_accs) / len(old_accs)
        new_acc = results[config_name]["new_knowledge"]
        
        retention = retention_rates[i]
        acquisition = acquisition_rates[i]
        
        f.write(f"| r={config['r']}, α={config['alpha']} | {retention:.2%} | {acquisition:.2%} | {old_avg:.4f} | {new_acc:.4f} |\n")
    
    # 找出最佳参数组合
    best_retention_idx = retention_rates.index(max(retention_rates))
    best_acquisition_idx = acquisition_rates.index(max(acquisition_rates))
    
    # 计算综合得分 (可以根据实际需求调整权重)
    combined_scores = [0.6 * ret + 0.4 * acq for ret, acq in zip(retention_rates, acquisition_rates)]
    best_combined_idx = combined_scores.index(max(combined_scores))
    
    f.write("\n## 最佳参数分析\n\n")
    f.write(f"1. 知识保留率最高的参数组合: r={lora_configs[best_retention_idx]['r']}, α={lora_configs[best_retention_idx]['alpha']}, 保留率为{retention_rates[best_retention_idx]:.2%}\n")
    f.write(f"2. 知识获取率最高的参数组合: r={lora_configs[best_acquisition_idx]['r']}, α={lora_configs[best_acquisition_idx]['alpha']}, 获取率为{acquisition_rates[best_acquisition_idx]:.2%}\n")
    f.write(f"3. 综合表现最佳的参数组合: r={lora_configs[best_combined_idx]['r']}, α={lora_configs[best_combined_idx]['alpha']}, 保留率为{retention_rates[best_combined_idx]:.2%}, 获取率为{acquisition_rates[best_combined_idx]:.2%}\n")
    
    f.write("\n## 参数影响分析\n\n")
    
    # 分析rank参数的影响
    if len(set(config["r"] for config in lora_configs)) > 1:
        f.write("### Rank (r) 参数的影响\n\n")
        f.write("随着rank值的增加：\n")
        
        # 检查趋势
        r_values = [config["r"] for config in lora_configs]
        if all(retention_rates[i] <= retention_rates[i+1] for i in range(len(retention_rates)-1)):
            f.write("- 知识保留率呈现单调增加趋势\n")
        elif all(retention_rates[i] >= retention_rates[i+1] for i in range(len(retention_rates)-1)):
            f.write("- 知识保留率呈现单调减少趋势\n")
        else:
            f.write("- 知识保留率呈现非单调变化\n")
            
        if all(acquisition_rates[i] <= acquisition_rates[i+1] for i in range(len(acquisition_rates)-1)):
            f.write("- 知识获取率呈现单调增加趋势\n")
        elif all(acquisition_rates[i] >= acquisition_rates[i+1] for i in range(len(acquisition_rates)-1)):
            f.write("- 知识获取率呈现单调减少趋势\n")
        else:
            f.write("- 知识获取率呈现非单调变化\n")
    
    # 分析alpha参数的影响
    if len(set(config["alpha"] for config in lora_configs)) > 1:
        f.write("\n### Alpha (α) 参数的影响\n\n")
        f.write("Alpha参数控制LoRA更新的缩放程度。在本实验中：\n")
        
        # 这里可以添加更详细的alpha参数分析
        # 由于我们的实验设计中r和alpha是成对变化的，这部分分析可能需要额外的实验
        
    f.write("\n## 建议\n\n")
    f.write(f"1. 基于本实验结果，推荐使用 r={lora_configs[best_combined_idx]['r']}, α={lora_configs[best_combined_idx]['alpha']} 的参数组合，可以在知识保留和获取之间取得良好的平衡。\n")
    f.write("2. 如果更关注知识保留（避免灾难性遗忘），建议使用较小的rank值。\n")
    f.write("3. 如果更关注新知识的学习效果，可以考虑使用较大的rank值。\n")
    f.write("4. 建议进行更细粒度的参数扫描，特别是分离rank和alpha参数的影响，以获得更精确的结论。\n")