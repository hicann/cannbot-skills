# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
Exemplar test_cases.py for UNARY operators (e.g., FastGelu, ReLU, Sigmoid)

Structure:
1. TEST_CASES: dict with operator metadata and test case list
2. Distribution generators: functions that create input tensors

Usage:
    from test_cases import TEST_CASES, generate_input

    for case in TEST_CASES["cases"]:
        input_tensor = generate_input(case)
        output = operator(input_tensor)
"""

import logging
import sys

import torch

logger = logging.getLogger(__name__)

# ============================================================================
# Part 1: Test Cases Definition
# ============================================================================

TEST_CASES = {
    "operator": "FastGelu",
    "operator_type": "unary",  # single input
    "description": "Fast GELU activation function",
    "cases": [
        {
            "id": 1,
            "name": "basic_1d",
            "shape": [256],
            "distribution": "normal",
            "dtype": "float32",
            "description": "基础1D形状"
        },
        {
            "id": 2,
            "name": "basic_2d",
            "shape": [32, 128],
            "distribution": "normal",
            "dtype": "float32",
            "description": "基础2D形状"
        },
        {
            "id": 3,
            "name": "basic_3d",
            "shape": [4, 32, 256],
            "distribution": "normal",
            "dtype": "float32",
            "description": "基础3D形状"
        },
        {
            "id": 4,
            "name": "llm_small",
            "shape": [4, 128, 768],
            "distribution": "normal",
            "dtype": "float32",
            "description": "LLM: batch×seq×hidden"
        },
        {
            "id": 5,
            "name": "llm_large",
            "shape": [8, 512, 4096],
            "distribution": "normal",
            "dtype": "float32",
            "description": "大规模LLM"
        },
        {
            "id": 6,
            "name": "ffn_intermediate",
            "shape": [4, 128, 3072],
            "distribution": "normal",
            "dtype": "float32",
            "description": "FFN中间层典型维度"
        },
        {
            "id": 7,
            "name": "near_zero",
            "shape": [4, 128, 768],
            "distribution": "near_zero",
            "dtype": "float32",
            "description": "近零输入（std=0.1），近线性区"
        },
        {
            "id": 8,
            "name": "large_positive",
            "shape": [4, 128, 768],
            "distribution": "positive_large",
            "dtype": "float32",
            "description": "大正值区间，激活饱和区"
        },
        {
            "id": 9,
            "name": "large_negative",
            "shape": [4, 128, 768],
            "distribution": "negative_large",
            "dtype": "float32",
            "description": "大负值区间，激活衰减区"
        },
        {
            "id": 10,
            "name": "non_aligned",
            "shape": [3, 77, 1023],
            "distribution": "normal",
            "dtype": "float32",
            "description": "非2幂次维度"
        },
    ]
}


# ============================================================================
# Part 2: Distribution Generators
# ============================================================================

def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    return dtype_map.get(dtype_str, torch.float32)


def generate_input(test_case: dict, device: str = "cpu", seed: int = 42) -> torch.Tensor:
    """
    Generate input tensor based on test case specification.

    Args:
        test_case: Test case dict from TEST_CASES["cases"]
        device: Device to place tensor on ("cpu" or "npu:0")
        seed: Random seed for reproducibility

    Returns:
        Input tensor with specified shape, dtype, and distribution
    """
    torch.manual_seed(seed)

    shape = test_case["shape"]
    dtype = _get_torch_dtype(test_case["dtype"])
    distribution = test_case["distribution"]

    # Generate based on distribution type
    if distribution == "normal":
        # Standard normal distribution N(0, 1)
        tensor = torch.randn(shape, dtype=dtype, device=device)

    elif distribution == "near_zero":
        # Near-zero distribution: N(0, 0.01)
        # Tests linear region of activation functions
        tensor = torch.randn(shape, dtype=dtype, device=device) * 0.1

    elif distribution == "positive_large":
        # Large positive values: uniform [2, 8]
        # Tests saturation region (output ≈ input for GELU)
        tensor = torch.rand(shape, dtype=dtype, device=device) * 6 + 2

    elif distribution == "negative_large":
        # Large negative values: uniform [-8, -2]
        # Tests attenuation region (output ≈ 0 for GELU)
        tensor = torch.rand(shape, dtype=dtype, device=device) * 6 - 8

    else:
        raise ValueError(f"Unknown distribution type: {distribution}")

    return tensor


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logger.info(f"Operator: {TEST_CASES['operator']}")
    logger.info(f"Total test cases: {len(TEST_CASES['cases'])}")
    logger.info("")

    for case in TEST_CASES["cases"]:
        logger.info(f"Case {case['id']}: {case['name']}")
        logger.info(f"  Shape: {case['shape']}")
        logger.info(f"  Distribution: {case['distribution']}")

        # Generate sample input
        input_tensor = generate_input(case, device="cpu")
        logger.info(f"  Generated tensor: shape={input_tensor.shape}, "
                    f"dtype={input_tensor.dtype}, "
                    f"mean={input_tensor.mean():.4f}, "
                    f"std={input_tensor.std():.4f}")
        logger.info("")
