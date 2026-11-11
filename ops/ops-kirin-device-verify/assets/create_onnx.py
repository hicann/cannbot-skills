#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build a minimal ONNX model with a single AddCustom node.

The generated model has two float16 inputs (``input1`` and ``input2``) and one
float16 output (``output``), all with shape ``[8, 2048]``. It uses opset 11 and
IR version 6 so ATC can convert it for Kirin device validation.
"""

import argparse

import onnx
from onnx import TensorProto, helper

DEFAULT_SHAPE = [8, 2048]
OPSET_VERSION = 11
IR_VERSION = 6


def build_model(shape=None):
    """Create the AddCustom ONNX model."""
    shape = list(shape or DEFAULT_SHAPE)
    inputs = [
        helper.make_tensor_value_info(f"input{index}", TensorProto.FLOAT16, shape)
        for index in (1, 2)
    ]
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT16, shape)
    node = helper.make_node(
        "AddCustom",
        inputs=["input1", "input2"],
        outputs=["output"],
        name="add",
        bias=1,
    )
    graph = helper.make_graph(
        [node],
        "custom_graph",
        inputs,
        [output],
    )
    model = helper.make_model(graph, producer_name="cannbot-kirin-template")
    model.opset_import[0].version = OPSET_VERSION
    model.ir_version = IR_VERSION
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="./add_custom.onnx",
        help="path of the generated ONNX file",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=DEFAULT_SHAPE,
        metavar=("M", "N"),
        help="tensor shape (default: 8 2048)",
    )
    args = parser.parse_args()

    model = build_model(args.shape)
    onnx.save(model, args.output)
    print(onnx.helper.printable_graph(model.graph))
    print(f"saved model to {args.output}")


if __name__ == "__main__":
    main()
