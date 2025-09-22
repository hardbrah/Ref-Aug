#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于配置文件的vLLM统一评测脚本
支持通过配置文件灵活添加新数据集
"""

import os
import re
import csv
import json
import time
import argparse
import datetime
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from dataclasses import dataclass, asdict

# Math-Verify for AIME
try:
    from math_verify import parse, verify
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig

    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("Warning: math_verify not available. Math answer extraction disabled.")

# ================== Answer Extraction ==================


def extract_math_answer(generated_text: str, ground_truth: str) -> Dict[str, Any]:
    """使用Math-Verify提取数学答案"""
    if not MATH_VERIFY_AVAILABLE:
        return {
            "extracted_answer": None,
            "is_correct": None,
            "extraction_error": "math_verify not available",
        }

    latex_config = LatexExtractionConfig(
        boxed_match_priority=0, try_extract_without_anchor=True
    )
    expr_config = ExprExtractionConfig(try_extract_without_anchor=True)

    pred_parsed = parse(generated_text, extraction_config=[latex_config, expr_config])

    gold_parsed = (
        parse(str(ground_truth), extraction_config=[ExprExtractionConfig()])
        if ground_truth
        else None
    )

    is_correct = False
    extracted_answer = None

    if pred_parsed is not None:
        extracted_answer = str(pred_parsed)
        if gold_parsed is not None:
            is_correct = verify(gold_parsed, pred_parsed)

    return {
        "extracted_answer": extracted_answer,
        "is_correct": is_correct,
        "extraction_error": None,
    }


def extract_boxed_letter(generated_text: str, ground_truth: str) -> Dict[str, Any]:
    """从\\boxed{}中提取字母答案（MCQ），对大小写不敏感，使用Math-Verify"""
    letter = None
    error = None

    if not MATH_VERIFY_AVAILABLE:
        return {
            "extracted_answer": None,
            "is_correct": None,
            "extraction_error": "math_verify not available",
        }

    if generated_text:
        # 使用Math-Verify的parse函数提取boxed内容
        latex_config = LatexExtractionConfig(
            boxed_match_priority=0, try_extract_without_anchor=True  # 优先提取boxed内容
        )

        pred_parsed = parse(generated_text, extraction_config=[latex_config])

        # parse返回一个列表，取最后一个（如果有的话）
        if pred_parsed and len(pred_parsed) > 0:
            extracted = pred_parsed[-1]  # 取最后一个提取结果
            extracted_str = str(extracted)

            # 如果包含逗号（多个boxed的情况），取最后一个
            if "," in extracted_str:
                parts = extracted_str.split(",")
                # 从最后一个部分提取字母
                for part in reversed(parts):
                    part_upper = part.strip().upper()
                    for char in part_upper:
                        if char in "ABCD":
                            letter = char
                            break
                    if letter:
                        break
            else:
                # 单个boxed，优先查找独立的字母（被空格、标点等分隔）
                extracted_upper = extracted_str.upper()

                # 先尝试查找被分隔符包围的字母
                import string

                words = []
                current_word = ""
                for char in extracted_upper:
                    if char in string.ascii_letters or char in string.digits:
                        current_word += char
                    else:
                        if current_word:
                            words.append(current_word)
                            current_word = ""
                if current_word:
                    words.append(current_word)

                # 查找单独的字母词
                for word in words:
                    if word in ["A", "B", "C", "D"]:
                        letter = word
                        break

                # 如果没找到独立的字母，再从整个字符串查找
                if not letter:
                    for char in extracted_upper:
                        if char in "ABCD":
                            letter = char
                            break

            if not letter:
                error = f"No valid letter in extracted: {extracted}"
        else:
            # Math-Verify没有提取到任何内容
            error = "No answer extracted"

    is_correct = (
        letter is not None
        and ground_truth is not None
        and letter.upper() == ground_truth.upper()
    )

    return {
        "extracted_answer": letter,
        "is_correct": is_correct,
        "extraction_error": error,
    }


def extract_answer(
    generated_text: str, ground_truth: str, method: str
) -> Dict[str, Any]:
    """根据方法提取答案"""
    if method == "math_verify":
        return extract_math_answer(generated_text, ground_truth)
    elif method == "boxed_letter":
        return extract_boxed_letter(generated_text, ground_truth)
    elif method == "code_execution":
        # 对于代码题目，不进行答案提取，返回原始生成文本
        return {
            "extracted_answer": generated_text,
            "is_correct": None,
            "extraction_error": None,
        }
    else:
        return {
            "extracted_answer": None,
            "is_correct": None,
            "extraction_error": f"Unknown extraction method: {method}",
        }
