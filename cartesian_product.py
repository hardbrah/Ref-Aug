import json
import os

# 定义文件路径
aime25_responses_path = r"data\sampled_aime25_responses.json"
aime24_questions_path = r"data\aime24\aime24.json"
output_path = r"cartesian_product_result.json"


prompt_template = "You have a QUESTION and a detailed THOUGHT PROCESS now, and you need to leverage the THOUGHT PROCESS to help you solve the QUESTION. \nQUESTION:{question} \nTHOUGHT PROCESS:{thought_process} \nReturn your final response within \\boxed{}."

# 读取aime24问题
with open(aime24_questions_path, "r", encoding="utf-8") as f:
    aime24_questions = json.load(f)

# 读取aime25响应
with open(aime25_responses_path, "r", encoding="utf-8") as f:
    aime25_responses = json.load(f)

# 创建结果列表
cartesian_product_results = []
cnt = 0
# 遍历aime25响应
for response_item in aime25_responses["results"]:
    problem_id = response_item["problem_id"]

    # 遍历每个run
    for run in response_item["runs"]:
        # 检查是否有生成的文本
        if "generated_text" in run and run["generated_text"].strip():
            thought_process = run["generated_text"]
            run_index = run["run_index"]

            # 遍历aime24问题，创建笛卡尔积
            for question_item in aime24_questions:
                question = question_item["prompt"]
                question_id = question_item["id"]

                # 填充提示词模板
                filled_prompt = prompt_template.replace("{question}", question).replace(
                    "{thought_process}", thought_process
                )

                # 创建结果项
                result_item = {
                    "prompt_id": cnt,
                    "prompt": filled_prompt,
                    "answer": question_item["answer"],
                    "question_source": question_item["source"],
                    "thought_process_source": response_item["dataset"],
                    "question_id": question_id,
                    "thought_process_id_runIndex": f"{problem_id}_{run_index}",
                }
                cnt += 1

                # 添加到结果列表
                cartesian_product_results.append(result_item)


# 保存结果到文件
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(cartesian_product_results, f, ensure_ascii=False, indent=2)

print(f"笛卡尔积已完成，共生成 {len(cartesian_product_results)} 个结果项")
print(f"结果已保存到: {output_path}")
