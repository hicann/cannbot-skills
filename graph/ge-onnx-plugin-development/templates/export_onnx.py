#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyTorch 自定义算子导出 ONNX 模板。

填充指引（使用前删除本段）：
- `__OP_NAME__`：算子名（symbolic 与插件 source 一致）；
- `example.domain`：自定义域（与插件 domain、custom_opsets 三处取值一致）；
- 属性后缀决定插件侧 node.attrs 取到的类型（_f=float/_i=int/_s=str；传列表值即导出同类型列表属性，后缀不变，无复数后缀）；
- 两侧一致约定见 skill references/export.md §3。
"""

import argparse
import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")


class MyOpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        del ctx
        return input_tensor * 2.0  # TODO(填充): 算子的真实前向计算

    @staticmethod
    def symbolic(graph, input_tensor):
        # TODO(填充): 域::算子名 与插件 source/domain 对齐；属性名与插件 node.attrs 取键一致
        return graph.op("example.domain::MyOp", input_tensor, alpha_f=1.0)


class ExampleModel(torch.nn.Module):
    def forward(self, input_tensor):
        return MyOpFunction.apply(input_tensor)


def main():
    parser = argparse.ArgumentParser(description="Export custom-op model to ONNX.")
    parser.add_argument("--output", required=True, help="Output ONNX file path.")
    args = parser.parse_args()

    sample_input = torch.randn(2, 3, dtype=torch.float32)  # TODO(填充): 真实输入形状
    torch.onnx.export(
        ExampleModel(),
        sample_input,
        args.output,
        opset_version=18,
        input_names=["x"],
        output_names=["y"],
        custom_opsets={
            "example.domain": 1
        },  # TODO(填充): 域与版本须与插件 domain/opsets 一致
    )
    logging.info("[Success] ONNX model exported to %s", args.output)


if __name__ == "__main__":
    main()
