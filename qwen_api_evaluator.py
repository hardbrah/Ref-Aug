import os
import json
import time
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from openai import OpenAI, APIError

# 加载.env文件中的环境变量
load_dotenv()

# 导入extract_answer函数
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eval_vllm_config import extract_answer

# 配置API客户端
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("DASHSCOPE_API_KEY 环境变量未设置")


def call_qwen_api(prompt: str) -> str:
    """调用Qwen API获取响应"""

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    try:  # 发起流式请求
        completion = client.chat.completions.create(
            model="qwen3-4b",  # 可以根据需要更改模型
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=True,  # Qwen3商业版（思考模式）、Qwen3开源版、QwQ、QVQ只支持流式输出。
            temperature=0.7,
            # 关闭思考模式
            extra_body={"enable_thinking": False, "top_k": None, "top_p": 0.9},
            stream_options={  # 在最后一个chunk中获取本次请求的Token用量。
                "include_usage": True
            },
        )
        # 处理流式响应
        # 使用列表推导式和join()是处理大量文本片段时最高效的方式。
        content_parts = []
        for chunk in completion:
            # 最后一个chunk不包含choices，但包含usage信息。
            if chunk.choices:
                # 关键：delta.content可能为None，使用`or ""`避免拼接时出错。
                content = chunk.choices[0].delta.content or ""
                content_parts.append(content)
            elif chunk.usage:
                # 请求结束，打印Token用量。
                print("\n--- 请求用量 ---")
                print(f"输入 Tokens: {chunk.usage.prompt_tokens}")
                print(f"输出 Tokens: {chunk.usage.completion_tokens}")
                print(f"总计 Tokens: {chunk.usage.total_tokens}")
        full_response = "".join(content_parts)
        return (
            full_response,
            chunk.usage.prompt_tokens,
            chunk.usage.completion_tokens,
            chunk.usage.total_tokens,
        )
    except APIError as e:
        print(f"API 请求失败: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")


def process_data(input_file: str, output_file: str):
    """处理数据并评估结果"""
    # 读取输入文件
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    # 处理每个prompt
    for idx, item in enumerate(data):
        # for idx, item in enumerate(tqdm(data, desc="处理进度")):
        print(f"\n处理第 {idx + 1} 项...")

        prompt = item["prompt"]
        ground_truth = item["answer"]

        # 调用API获取响应
        print("调用API中...")
        response, prompt_tokens, completion_tokens, total_tokens = call_qwen_api(prompt)

        # 提取答案并评估正确性
        print("评估结果中...")
        # 假设使用math_verify方法，可以根据实际情况调整
        extraction_result = extract_answer(response, ground_truth, "math_verify")

        # 构建结果
        result = {
            "prompt_id": idx,
            "question_id": item.get("question_id"),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "thought_process_id_runIndex": item.get("thought_process_id_runIndex"),
            "prompt": prompt,
            "ground_truth": ground_truth,
            "generated_text": response,
            "extracted_answer": extraction_result.get("extracted_answer"),
            "is_correct": extraction_result.get("is_correct"),
            "extraction_error": extraction_result.get("extraction_error"),
        }
        results.append(result)

    # 保存结果
    with open(output_file, "a", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n处理完成，共处理 {len(results)} 项，结果已保存至 {output_file}")


if __name__ == "__main__":
    input_file = r"cartesian_product_result.json"
    output_file = r"data\qwen_api_results.json"

    process_data(input_file, output_file)
