import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 加载所有结果
with open("/Users/zenoy/Documents/Playground/baseline_results.json", "r") as f:
    baseline_results = json.load(f)

with open("/Users/zenoy/Documents/Playground/lora_results.json", "r") as f:
    lora_results = json.load(f)

with open("/Users/zenoy/Documents/Playground/full_finetune_results.json", "r") as f:
    full_results = json.load(f)

# 计算每个知识领域的保留率
subjects = list(baseline_results["old_knowledge"].keys())
retention_data = []

for subject in subjects:
    baseline_acc = baseline_results["old_knowledge"][subject]
    lora_acc = lora_results["old_knowledge"][subject]
    full_acc = full_results["old_knowledge"][subject]
    
    lora_retention = lora_acc / baseline_acc if baseline_acc > 0 else 0
    full_retention = full_acc / baseline_acc if baseline_acc > 0 else 0
    
    retention_data.append({
        "subject": subject,
        "baseline_acc": baseline_acc,
        "lora_acc": lora_acc,
        "full_acc": full_acc,
        "lora_retention": lora_retention,
        "full_retention": full_retention,
        "lora_change": lora_acc - baseline_acc,
        "full_change": full_acc - baseline_acc
    })

# 绘制热图显示不同知识领域的保留情况
plt.figure(figsize=(12, 8))
heatmap_data = np.zeros((len(subjects), 2))

for i, subject_data in enumerate(retention_data):
    heatmap_data[i, 0] = subject_data["lora_change"]
    heatmap_data[i, 1] = subject_data["full_change"]

sns.heatmap(
    heatmap_data, 
    annot=True, 
    fmt=".3f", 
    cmap="RdYlGn", 
    center=0,
    yticklabels=subjects,
    xticklabels=["LoRA微调", "全参数微调"],
    cbar_kws={"label": "准确率变化"}
)
plt.title("不同微调方法在各知识领域的准确率变化")
plt.tight_layout()
plt.savefig("/Users/zenoy/Documents/Playground/knowledge_retention_heatmap.png")

# 绘制散点图比较LoRA和全参数微调的知识保留
plt.figure(figsize=(10, 8))
x = [data["lora_retention"] for data in retention_data]
y = [data["full_retention"] for data in retention_data]
labels = [data["subject"] for data in retention_data]

plt.scatter(x, y, s=100, alpha=0.7)
for i, label in enumerate(labels):
    plt.annotate(label, (x[i], y[i]), fontsize=9)

plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=1, color='r', linestyle='--', alpha=0.5)
plt.plot([0, 2], [0, 2], 'g--', alpha=0.5)  # 对角线

plt.xlim(min(x) * 0.9, max(x) * 1.1)
plt.ylim(min(y) * 0.9, max(y) * 1.1)
plt.xlabel('LoRA微调知识保留率')
plt.ylabel('全参数微调知识保留率')
plt.title('LoRA vs 全参数微调的知识保留率比较')
plt.grid(True, alpha=0.3)
plt.savefig("/Users/zenoy/Documents/Playground/retention_comparison_scatter.png")

# 生成详细的知识保留分析报告
with open("/Users/zenoy/Documents/Playground/knowledge_retention_report.txt", "w") as f:
    f.write("# 知识保留详细分析报告\n\n")
    
    f.write("## 各知识领域保留情况\n\n")
    f.write("| 知识领域 | 基线准确率 | LoRA准确率 | 全参数准确率 | LoRA变化 | 全参数变化 | LoRA保留率 | 全参数保留率 |\n")
    f.write("|----------|------------|------------|--------------|----------|------------|------------|------------|\n")
    
    for data in retention_data:
        f.write(f"| {data['subject']} | {data['baseline_acc']:.4f} | {data['lora_acc']:.4f} | {data['full_acc']:.4f} | ")
        f.write(f"{data['lora_change']:.4f} | {data['full_change']:.4f} | {data['lora_retention']:.2%} | {data['full_retention']:.2%} |\n")
    
    # 计算平均值
    avg_lora_change = sum(data["lora_change"] for data in retention_data) / len(retention_data)
    avg_full_change = sum(data["full_change"] for data in retention_data) / len(retention_data)
    avg_lora_retention = sum(data["lora_retention"] for data in retention_data) / len(retention_data)
    avg_full_retention = sum(data["full_retention"] for data in retention_data) / len(retention_data)
    
    f.write(f"\n**平均变化**: LoRA: {avg_lora_change:.4f}, 全参数: {avg_full_change:.4f}\n")
    f.write(f"**平均保留率**: LoRA: {avg_lora_retention:.2%}, 全参数: {avg_full_retention:.2%}\n\n")
    
    # 分析结果
    f.write("## 分析结论\n\n")
    
    if avg_lora_retention > avg_full_retention:
        f.write("1. LoRA微调在知识保留方面整体表现优于全参数微调，证明其在减轻灾难性遗忘方面的有效性。\n")
    else:
        f.write("1. 在本实验中，全参数微调在知识保留方面表现优于LoRA微调，这可能与具体的实验设置有关。\n")
    
    # 分析各领域的差异
    best_lora_subject = max(retention_data, key=lambda x: x["lora_retention"])
    worst_lora_subject = min(retention_data, key=lambda x: x["lora_retention"])
    
    f.write(f"\n2. LoRA微调在'{best_lora_subject['subject']}'领域保留效果最好，保留率为{best_lora_subject['lora_retention']:.2%}。\n")
    f.write(f"3. LoRA微调在'{worst_lora_subject['subject']}'领域保留效果最差，保留率为{worst_lora_subject['lora_retention']:.2%}。\n")
    
    # 提供建议
    f.write("\n## 建议\n\n")
    f.write("1. 针对保留率较低的知识领域，可以考虑在微调数据中添加少量相关样本，以减轻遗忘。\n")
    f.write("2. 调整LoRA的rank和alpha参数可能会影响知识保留效果，建议进行参数敏感性分析。\n")
    f.write("3. 对于特别重要的知识领域，可以考虑使用知识蒸馏或正则化技术来进一步减轻遗忘。\n")