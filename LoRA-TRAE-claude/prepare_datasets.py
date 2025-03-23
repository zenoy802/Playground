import datasets
from datasets import load_dataset

# 加载旧知识评估数据集（例如使用MMLU的部分子集）
def load_old_knowledge_dataset():
    # 加载MMLU数据集作为旧知识评估
    mmlu = load_dataset("cais/mmlu", "all")
    # 选择几个代表性的学科作为旧知识测试
    old_subjects = ["high_school_mathematics", "high_school_physics", "high_school_biology"]
    old_knowledge_eval = {subject: mmlu["validation"].filter(lambda x: x["subject"] == subject) 
                         for subject in old_subjects}
    return old_knowledge_eval

# 准备新知识数据集（例如特定领域的新数据）
def prepare_new_knowledge_dataset():
    # 这里可以使用一个特定领域的数据集，例如医学或法律
    # 假设我们使用医学领域的数据
    medical_dataset = load_dataset("medical_qa_dataset")  # 替换为实际的医学数据集
    # 或者创建自定义数据集
    # medical_dataset = create_custom_medical_dataset()
    return medical_dataset

# 创建自定义医学数据集示例（如果没有合适的现成数据集）
def create_custom_medical_dataset():
    # 这里可以从医学论文、医学教科书或医学QA中提取数据
    # 返回格式化为instruction-following格式的数据
    medical_data = {
        "instruction": [...],  # 医学问题或指令
        "input": [...],        # 可选的输入上下文
        "output": [...]        # 期望的医学知识回答
    }
    return datasets.Dataset.from_dict(medical_data)