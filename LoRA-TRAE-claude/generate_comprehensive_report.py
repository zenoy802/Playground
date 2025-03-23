import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 加载所有结果文件
with open("/Users/zenoy/Documents/Playground/baseline_results.json", "r") as f:
    baseline_results = json.load(f)

with open("/Users/zenoy/Documents/Playground/lora_results.json", "r") as f:
    lora_results = json.load(f)

with open("/Users/zenoy/Documents/Playground/full_finetune_results.json", "r") as f:
    full_results = json.load(f)

# 读取各分析报告的内容
with open("/Users/zenoy/Documents/Playground/experiment_conclusion.txt", "r") as f:
    experiment_conclusion = f.read()

with open("/Users/zenoy/Documents/Playground/knowledge_retention_report.txt", "r") as f:
    retention_report = f.read()

with open("/Users/zenoy/Documents/Playground/forgetting_pattern_report.txt", "r") as f:
    forgetting_report = f.read()

# 尝试读取参数分析报告（如果存在）
try:
    with open("/Users/zenoy/Documents/Playground/lora_parameter_analysis_report.txt", "r") as f:
        parameter_report = f.read()
    has_parameter_analysis = True
except FileNotFoundError:
    parameter_report = "参数敏感性分析未执行。"
    has_parameter_analysis = False

# 生成综合报告
report_date = datetime.now().strftime("%Y年%m月%d日")

with open("/Users/zenoy/Documents/Playground/comprehensive_experiment_report.md", "w") as f:
    f.write(f"# LoRA微调避免灾难性遗忘实验综合报告\n\n")
    f.write(f"**报告日期：{report_date}**\n\n")
    
    f.write("## 1. 实验概述\n\n")
    f.write("本实验旨在验证LoRA等参数高效微调方法(PEFT)能否在让大语言模型学习新知识的同时，避免对旧知识的灾难性遗忘。实验比较了LoRA微调和传统全参数微调在知识保留和获取方面的表现。\n\n")
    
    f.write("### 1.1 实验设置\n\n")
    f.write("- **基础模型**：Baichuan2-13B-Base\n")
    f.write("- **微调方法**：LoRA微调和全参数微调\n")
    f.write("- **评估数据**：旧知识评估数据集和新知识评估数据集\n")
    f.write("- **评估指标**：知识保留率和知识获取率\n\n")
    
    f.write("### 1.2 主要发现\n\n")
    
    # 提取主要结果
    subjects = list(baseline_results["old_knowledge"].keys())
    baseline_old = [baseline_results["old_knowledge"][subject] for subject in subjects]
    lora_old = [lora_results["old_knowledge"][subject] for subject in subjects]
    full_old = [full_results["old_knowledge"][subject] for subject in subjects]

    baseline_old_avg = sum(baseline_old) / len(baseline_old)
    lora_old_avg = sum(lora_old) / len(lora_old)
    full_old_avg = sum(full_old) / len(full_old)

    baseline_new = baseline_results["new_knowledge"]
    lora_new = lora_results["new_knowledge"]
    full_new = full_results["new_knowledge"]

    lora_retention = lora_old_avg / baseline_old_avg
    full_retention = full_old_avg / baseline_old_avg

    lora_acquisition = (lora_new - baseline_new) / (1 - baseline_new) if baseline_new < 1 else 1
    full_acquisition = (full_new - baseline_new) / (1 - baseline_new) if baseline_new < 1 else 1
    
    f.write(f"- LoRA微调的知识保留率为**{lora_retention:.2%}**，全参数微调为**{full_retention:.2%}**\n")
    f.write(f"- LoRA微调的知识获取率为**{lora_acquisition:.2%}**，全参数微调为**{full_acquisition:.2%}**\n")
    
    if lora_retention > full_retention:
        f.write("- **LoRA微调在避免灾难性遗忘方面表现优于全参数微调**\n")
    else:
        f.write("- 在本实验设置下，全参数微调在知识保留方面表现出乎意料地好\n")
    
    if lora_acquisition > full_acquisition:
        f.write("- **LoRA微调在获取新知识方面也表现优于全参数微调**\n")
    else:
        f.write("- 全参数微调在获取新知识方面表现优于LoRA微调\n")
    
    f.write("\n## 2. 详细实验结果\n\n")
    
    f.write("### 2.1 旧知识保留情况\n\n")
    f.write("| 知识领域 | 基线准确率 | LoRA准确率 | 全参数准确率 | LoRA变化 | 全参数变化 |\n")
    f.write("|----------|------------|------------|--------------|----------|------------|\n")
    
    for i, subject in enumerate(subjects):
        baseline_acc = baseline_old[i]
        lora_acc = lora_old[i]
        full_acc = full_old[i]
        lora_change = lora_acc - baseline_acc
        full_change = full_acc - baseline_acc
        
        f.write(f"| {subject} | {baseline_acc:.4f} | {lora_acc:.4f} | {full_acc:.4f} | ")
        f.write(f"{lora_change:+.4f} | {full_change:+.4f} |\n")
    
    f.write(f"| **平均** | {baseline_old_avg:.4f} | {lora_old_avg:.4f} | {full_old_avg:.4f} | ")
    avg_lora_change = lora_old_avg - baseline_old_avg
    avg_full_change = full_old_avg - baseline_old_avg
    f.write(f"{avg_lora_change:+.4f} | {avg_full_change:+.4f} |\n\n")
    
    f.write("### 2.2 新知识获取情况\n\n")
    f.write("| 模型 | 准确率 | 相对基线变化 | 知识获取率 |\n")
    f.write("|------|--------|--------------|------------|\n")
    f.write(f"| 基线模型 | {baseline_new:.4f} | - | - |\n")
    f.write(f"| LoRA微调 | {lora_new:.4f} | {lora_new-baseline_new:+.4f} | {lora_acquisition:.2%} |\n")
    f.write(f"| 全参数微调 | {full_new:.4f} | {full_new-baseline_new:+.4f} | {full_acquisition:.2%} |\n\n")
    
    f.write("## 3. 知识保留分析\n\n")
    # 提取知识保留报告的主要内容
    retention_summary = "\n".join(retention_report.split("\n\n")[2:4])  # 提取主要发现部分
    f.write(retention_summary + "\n\n")
    
    f.write("## 4. 灾难性遗忘模式分析\n\n")
    # 提取遗忘模式报告的主要内容
    forgetting_summary = "\n".join(forgetting_report.split("\n\n")[1:3])  # 提取主要发现部分
    f.write(forgetting_summary + "\n\n")
    
    if has_parameter_analysis:
        f.write("## 5. LoRA参数敏感性分析\n\n")
        # 提取参数分析报告的主要内容
        parameter_summary = "\n".join(parameter_report.split("\n\n")[1:3])  # 提取主要发现部分
        f.write(parameter_summary + "\n\n")
    
    f.write("## 6. 结论与建议\n\n")
    # 提取实验结论的主要内容
    conclusion_summary = "\n".join(experiment_conclusion.split("\n\n")[2:])  # 提取结论部分
    f.write(conclusion_summary + "\n\n")
    
    f.write("### 6.1 未来工作方向\n\n")
    f.write("1. **更细粒度的参数分析**：探索不同LoRA参数配置对知识保留和获取的影响\n")
    f.write("2. **混合策略研究**：研究结合LoRA与知识蒸馏等技术的混合策略\n")
    f.write("3. **长期知识保留**：研究连续学习场景下的长期知识保留能力\n")
    f.write("4. **跨领域知识迁移**：研究LoRA微调如何影响模型在不同领域间的知识迁移能力\n")
    f.write("5. **大规模实验验证**：在更多基础模型和更大规模数据集上验证结论\n\n")
    
    f.write("## 附录：实验图表\n\n")
    
    f.write("### 图1：旧知识保留情况\n\n")
    f.write("![旧知识保留情况](/Users/zenoy/Documents/Playground/old_knowledge_retention.png)\n\n")
    
    f.write("### 图2：知识平衡雷达图\n\n")
    f.write("![知识平衡雷达图](/Users/zenoy/Documents/Playground/knowledge_balance_radar.png)\n\n")
    
    f.write("### 图3：遗忘分布\n\n")
    f.write("![遗忘分布](/Users/zenoy/Documents/Playground/forgetting_distribution.png)\n\n")
    
    if has_parameter_analysis:
        f.write("### 图4：LoRA参数敏感性\n\n")
        f.write("![LoRA参数敏感性](/Users/zenoy/Documents/Playground/lora_parameter_sensitivity.png)\n\n")
        
        f.write("### 图5：参数权衡图\n\n")
        f.write("![参数权衡图](/Users/zenoy/Documents/Playground/lora_parameter_tradeoff.png)\n\n")
    
    f.write("---\n\n")
    f.write("**报告生成时间：** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

print("综合实验报告已生成：/Users/zenoy/Documents/Playground/comprehensive_experiment_report.md")