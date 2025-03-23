import os
import subprocess
import time

def run_command(command, description):
    print(f"\n{'='*50}")
    print(f"开始: {description}")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    process = subprocess.Popen(command, shell=True)
    process.wait()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*50}")
    print(f"完成: {description}")
    print(f"耗时: {duration:.2f}秒 ({duration/60:.2f}分钟)")
    print(f"{'='*50}\n")
    
    if process.returncode != 0:
        print(f"错误: {description}失败，返回代码 {process.returncode}")
        exit(1)

# 创建输出目录
os.makedirs("/Users/zenoy/Documents/Playground/lora_output", exist_ok=True)
os.makedirs("/Users/zenoy/Documents/Playground/full_finetune_output", exist_ok=True)
os.makedirs("/Users/zenoy/Documents/Playground/lora_parameter_results", exist_ok=True)

# 运行实验步骤
run_command("python /Users/zenoy/Documents/Playground/evaluate_baseline.py", 
           "评估基线模型性能")

run_command("python /Users/zenoy/Documents/Playground/lora_finetune.py", 
           "LoRA微调")

run_command("python /Users/zenoy/Documents/Playground/evaluate_finetuned.py", 
           "评估LoRA微调后的性能")

run_command("python /Users/zenoy/Documents/Playground/full_finetune.py", 
           "全参数微调")

run_command("python /Users/zenoy/Documents/Playground/evaluate_full_finetuned.py", 
           "评估全参数微调后的性能")

run_command("python /Users/zenoy/Documents/Playground/compare_results.py", 
           "比较结果并生成初步报告")

# 运行深入分析
run_command("python /Users/zenoy/Documents/Playground/knowledge_retention_analysis.py", 
           "知识保留详细分析")

run_command("python /Users/zenoy/Documents/Playground/forgetting_pattern_analysis.py", 
           "灾难性遗忘模式分析")

# 参数敏感性分析（可选，取决于计算资源）
run_parameter_analysis = input("是否运行LoRA参数敏感性分析？(y/n): ").strip().lower() == 'y'
if run_parameter_analysis:
    run_command("python /Users/zenoy/Documents/Playground/lora_parameter_analysis.py", 
               "LoRA参数敏感性分析")

# 生成综合报告
run_command("python /Users/zenoy/Documents/Playground/generate_comprehensive_report.py", 
           "生成综合实验报告")

print("\n实验完成！结果和分析已保存到以下文件:")
print("- /Users/zenoy/Documents/Playground/comprehensive_experiment_report.md")
print("- /Users/zenoy/Documents/Playground/old_knowledge_retention.png")
print("- /Users/zenoy/Documents/Playground/knowledge_balance_radar.png")
print("- /Users/zenoy/Documents/Playground/forgetting_distribution.png")
if run_parameter_analysis:
    print("- /Users/zenoy/Documents/Playground/lora_parameter_sensitivity.png")
    print("- /Users/zenoy/Documents/Playground/lora_parameter_tradeoff.png")