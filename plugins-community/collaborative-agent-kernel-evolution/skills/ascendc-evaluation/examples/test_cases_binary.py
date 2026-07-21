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
Exemplar test_cases.py for BINARY operators (e.g., Matmul, Add, Mul)

Structure:
1. TEST_CASES: dict with operator metadata and test case list
2. Distribution generators: functions that create input tensors

Usage:
    from test_cases import TEST_CASES, generate_inputs

    for case in TEST_CASES["cases"]:
        input1, input2 = generate_inputs(case)
        output = operator(input1, input2)
"""

import logging
import sys
from typing import Tuple

import torch

logger = logging.getLogger(__name__)

# ============================================================================
# Part 1: Test Cases Definition
# ============================================================================

TEST_CASES = {
    "operator": "Matmul",
    "operator_type": "binary",  # two inputs
    "description": "Matrix multiplication",
    "cases": [
        {
            "id": 1,
            "name": "small_2d",
            "shape1": [32, 64],
            "shape2": [64, 128],
            "distribution": "normal",
            "dtype": "float32",
            "description": "小规模2D矩阵乘"
        },
        {
            "id": 2,
            "name": "medium_2d",
            "shape1": [256, 512],
            "shape2": [512, 256],
            "distribution": "normal",
            "dtype": "float32",
            "description": "中规模2D矩阵乘"
        },
        {
            "id": 3,
            "name": "small_batched",
            "shape1": [4, 16, 32],
            "shape2": [4, 32, 64],
            "distribution": "normal",
            "dtype": "float32",
            "description": "小批量3D"
        },
        {
            "id": 4,
            "name": "medium_batched",
            "shape1": [4, 128, 768],
            "shape2": [4, 768, 512],
            "distribution": "normal",
            "dtype": "float32",
            "description": "典型LLM批量"
        },
        {
            "id": 5,
            "name": "large_batched",
            "shape1": [8, 512, 2048],
            "shape2": [8, 2048, 1024],
            "distribution": "normal",
            "dtype": "float32",
            "description": "大规模批量"
        },
        {
            "id": 6,
            "name": "linear_ffn",
            "shape1": [4, 512, 4096],
            "shape2": [4, 4096, 4096],
            "distribution": "normal",
            "dtype": "float32",
            "description": "FFN线性层"
        },
        {
            "id": 7,
            "name": "non_aligned",
            "shape1": [3, 127, 255],
            "shape2": [3, 255, 193],
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


def generate_inputs(
    test_case: dict,
    device: str = "cpu",
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate input tensors based on test case specification.

    Args:
        test_case: Test case dict from TEST_CASES["cases"]
        device: Device to place tensors on ("cpu" or "npu:0")
        seed: Random seed for reproducibility

    Returns:
        Tuple of (input1, input2) tensors
    """
    torch.manual_seed(seed)

    shape1 = test_case["shape1"]
    shape2 = test_case["shape2"]
    dtype = _get_torch_dtype(test_case["dtype"])
    distribution = test_case["distribution"]

    # Generate based on distribution type
    if distribution == "normal":
        # Standard normal distribution N(0, 1)
        input1 = torch.randn(shape1, dtype=dtype, device=device)
        input2 = torch.randn(shape2, dtype=dtype, device=device)

    elif distribution == "uniform":
        # Uniform distribution [0, 1]
        input1 = torch.rand(shape1, dtype=dtype, device=device)
        input2 = torch.rand(shape2, dtype=dtype, device=device)

    elif distribution == "ones":
        # All ones (stress test for numerical stability)
        input1 = torch.ones(shape1, dtype=dtype, device=device)
        input2 = torch.ones(shape2, dtype=dtype, device=device)

    elif distribution == "small_values":
        # Small values: uniform [0, 0.1]
        input1 = torch.rand(shape1, dtype=dtype, device=device) * 0.1
        input2 = torch.rand(shape2, dtype=dtype, device=device) * 0.1

    else:
        raise ValueError(f"Unknown distribution type: {distribution}")

    return input1, input2


# ============================================================================
# Additional generator for operators with more than 2 inputs
# ============================================================================

def generate_inputs_ternary(
    test_case: dict,
    device: str = "cpu",
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate input tensors for ternary operators (3 inputs).

    Example: FusedMultiplyAdd(a, b, c) = a * b + c

    Args:
        test_case: Test case dict with shape1, shape2, shape3
        device: Device to place tensors on
        seed: Random seed

    Returns:
        Tuple of (input1, input2, input3) tensors
    """
    torch.manual_seed(seed)

    shape1 = test_case["shape1"]
    shape2 = test_case["shape2"]
    shape3 = test_case["shape3"]
    dtype = _get_torch_dtype(test_case["dtype"])
    distribution = test_case["distribution"]

    if distribution == "normal":
        input1 = torch.randn(shape1, dtype=dtype, device=device)
        input2 = torch.randn(shape2, dtype=dtype, device=device)
        input3 = torch.randn(shape3, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unknown distribution type: {distribution}")

    return input1, input2, input3


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
        logger.info(f"  Shape1: {case['shape1']}")
        logger.info(f"  Shape2: {case['shape2']}")
        logger.info(f"  Distribution: {case['distribution']}")

        # Generate sample inputs
        input1, input2 = generate_inputs(case, device="cpu")
        logger.info(f"  Generated input1: shape={input1.shape}, "
                    f"dtype={input1.dtype}, "
                    f"mean={input1.mean():.4f}, "
                    f"std={input1.std():.4f}")
        logger.info(f"  Generated input2: shape={input2.shape}, "
                    f"dtype={input2.dtype}, "
                    f"mean={input2.mean():.4f}, "
                    f"std={input2.std():.4f}")
        logger.info("")
