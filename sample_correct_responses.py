import json
import random
import os
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/sampling_log.txt'),
        logging.StreamHandler()
    ],
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# 确保输出目录存在
os.makedirs('data', exist_ok=True)

# 输入和输出文件路径
input_file = 'qwen3-4b-thinking/code_evaluation_results_20250914_001405/eval_aime25_FLASH_ATTN_20250915_182423.json'
output_file = 'data/sampled_aime25_responses.json'

def sample_correct_responses():
    logger.info(f"开始处理文件: {input_file}")
    
    # 读取原始JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 复制测试信息和配置
    result = {
        "test_info": data["test_info"],
        "vllm_config": data["vllm_config"],
        "sampling_params": data["sampling_params"],
        "results": []
    }
    
    # 记录没有正确回复的问题ID
    no_correct_answers = []
    
    # 处理每个问题
    for problem in data["results"]:
        problem_id = problem["problem_id"]
        logger.info(f"处理问题 ID: {problem_id}")
        
        # 找出所有正确的回复
        correct_runs = [run for run in problem["runs"] if run.get("is_correct", False)]
        
        # 创建新的问题对象
        new_problem = {
            "problem_id": problem["problem_id"],
            "dataset": problem["dataset"],
            "prompt": problem["prompt"],
            "ground_truth": problem["ground_truth"],
            "n_runs": problem["n_runs"]
        }
        
        # 如果有正确回复，随机选择一个
        if correct_runs:
            sampled_run = random.choice(correct_runs)
            new_problem["runs"] = [sampled_run]
            logger.info(f"问题 {problem_id}: 从 {len(correct_runs)} 个正确回复中采样了 run_index={sampled_run['run_index']}")
        else:
            new_problem["runs"] = []
            no_correct_answers.append(problem_id)
            logger.warning(f"问题 {problem_id}: 没有正确的回复!")
        
        result["results"].append(new_problem)
    
    # 保存结果到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 记录没有正确回复的问题ID
    if no_correct_answers:
        logger.warning(f"以下问题没有正确回复: {', '.join(no_correct_answers)}")
    else:
        logger.info("所有问题都至少有一个正确回复")
    
    logger.info(f"处理完成，结果已保存到: {output_file}")
    
    # 生成摘要报告
    with open('data/sampling_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"处理文件: {input_file}\n")
        f.write(f"总问题数: {len(data['results'])}\n")
        f.write(f"没有正确回复的问题数: {len(no_correct_answers)}\n")
        if no_correct_answers:
            f.write(f"没有正确回复的问题ID: {', '.join(no_correct_answers)}\n")

if __name__ == "__main__":
    sample_correct_responses()