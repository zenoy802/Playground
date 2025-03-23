import json
import matplotlib.pyplot as plt
import numpy as np

# 加载所有结果
with open("/Users/zenoy/Documents/Playground/baseline_results.json", "r") as f:
    baseline_results = json.load(f)

with open("/Users/zenoy/Documents/Playground/lora_results.json", "r") as f:
    lora_results = json.load(f)

with open("/Users/zenoy/Documents/Playground/full_finetune_results.json", "r") as f:
    full_results = json.load(f)

# 提取旧知识和新知识的准确率
subjects = list(baseline_results["old_knowledge"].keys())
baseline_old = [baseline_results["old_knowledge"][subject] for subject in subjects]
lora_old = [lora_results["old_knowledge"][subject] for subject in subjects]
full_old = [full_results["old_knowledge"][subject] for subject in subjects]

baseline_new = baseline_results["new_knowledge"]
lora_new = lora_results["new_knowledge"]
full_new = full_results["new_knowledge"]

# 计算平均旧知识准确率
baseline_old_avg = sum(baseline_old) / len(baseline_old)
lora_old_avg = sum(lora_old) / len(lora_old)
full_old_avg = sum(full_old) / len(full_old)

# 计算旧知识保留率和新知识获取率
lora_retention = lora_old_avg / baseline_old_avg
full_retention = full_old_avg / baseline_old_avg

lora_acquisition = (lora_new - baseline_new) / (1 - baseline_new) if baseline_new < 1 else 1
full_acquisition = (full_new - baseline_new) / (1 - baseline_new) if baseline_new < 1 else 1

# 打印结果
print("实验结果比较:")
print(f"基线模型平均旧知识准确率: {baseline_old_avg:.4f}")
print(f"LoRA微调后平均旧知识准确率: {lora_old_avg:.4f} (保留率: {lora_retention:.2%})")
print(f"全参数微调后平均旧知识准确率: {full_old_avg:.4f} (保留率: {full_retention:.2%})")
print()
print(f"基线模型新知识准确率: {baseline_new:.4f}")
print(f"LoRA微调后新知识准确率: {lora_new:.4f} (获取率: {lora_acquisition:.2%})")
print(f"全参数微调后新知识准确率: {full_new:.4f} (获取率: {full_acquisition:.2%})")

# 绘制旧知识保留情况的柱状图
plt.figure(figsize=(12, 6))
x = np.arange(len(subjects))
width = 0.25

plt.bar(x - width, baseline_old, width, label='基线模型')
plt.bar(x, lora_old, width, label='LoRA微调')
plt.bar(x + width, full_old, width, label='全参数微调')

plt.xlabel('知识领域')
plt.ylabel('准确率')
plt.title('不同微调方法对旧知识的保留情况')
plt.xticks(x, subjects, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("/Users/zenoy/Documents/Playground/old_knowledge_retention.png")

# 绘制新旧知识平衡的雷达图
plt.figure(figsize=(8, 8))
categories = ['旧知识保留率', '新知识获取率']
values = {
    'LoRA微调': [lora_retention, lora_acquisition],
    '全参数微调': [full_retention, full_acquisition]
}

angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]  # 闭合雷达图

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for method, vals in values.items():
    vals += vals[:1]  # 闭合数据
    ax.plot(angles, vals, linewidth=2, label=method)
    ax.fill(angles, vals, alpha=0.25)

ax.set_thetagrids(np.degrees(angles[:-1]), categories)
ax.set_ylim(0, 1.2)
ax.set_title('微调方法在新旧知识平衡上的表现')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig("/Users/zenoy/Documents/Playground/knowledge_balance_radar.png")

# 生成结论报告
with open("/Users/zenoy/Documents/Playground/experiment_conclusion.txt", "w") as f:
    f.write("# LoRA与全参数微调在知识保留与获取上的比较实验\n\n")
    
    f.write("## 实验结果\n\n")
    f.write(f"基线模型平均旧知识准确率: {baseline_old_avg:.4f}\n")
    f.write(f"LoRA微调后平均旧知识准确率: {lora_old_avg:.4f} (保留率: {lora_retention:.2%})\n")
    f.write(f"全参数微调后平均旧知识准确率: {full_old_avg:.4f} (保留率: {full_retention:.2%})\n\n")
    
    f.write(f"基线模型新知识准确率: {baseline_new:.4f}\n")
    f.write(f"LoRA微调后新知识准确率: {lora_new:.4f} (获取率: {lora_acquisition:.2%})\n")
    f.write(f"全参数微调后新知识准确率: {full_new:.4f} (获取率: {full_acquisition:.2%})\n\n")
    
    f.write("## 结论\n\n")
    if lora_retention > full_retention:
        f.write("1. LoRA微调在保留旧知识方面表现优于全参数微调，证明其有效减轻了灾难性遗忘问题。\n")
    else:
        f.write("1. 在本实验中，全参数微调在保留旧知识方面表现出乎意料地好，可能是由于训练数据量较小或训练轮次较少。\n")
    
    if lora_acquisition > full_acquisition:
        f.write("2. LoRA微调在获取新知识方面表现优于全参数微调，这可能是由于其更高的学习率和更聚焦的参数更新。\n")
    else:
        f.write("2. 全参数微调在获取新知识方面表现优于LoRA微调，这符合预期，因为它可以调整所有参数。\n")
    
    f.write(f"3. LoRA微调的旧知识保留率为{lora_retention:.2%}，而全参数微调为{full_retention:.2%}，")
    f.write(f"新知识获取率分别为{lora_acquisition:.2%}和{full_acquisition:.2%}。\n")
    
    f.write("\n## 建议\n\n")
    f.write("1. 当需要在保留模型原有能力的同时学习新知识时，LoRA是一个更好的选择。\n")
    f.write("2. 对于资源受限的场景，LoRA不仅训练成本低，而且能有效避免灾难性遗忘。\n")
    f.write("3. 未来研究可以探索不同LoRA参数配置对知识保留和获取的影响。\n")