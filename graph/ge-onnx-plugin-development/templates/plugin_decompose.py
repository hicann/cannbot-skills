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

"""一对多分解插件模板（parse_node + decompose 接力）。

适用：GE 没有现成算子，需用已有算子拼子图替换原节点（无需写 kernel）。
填充指引（使用前删除本段）：
- parse_node 负责属性中转 + 端口注册；decompose 读到的属性即 parse_node 写入的值；
- 子图输出要和注册的输出对应；子图输入不用手动指定 dtype/shape；
- 可用 ES 算子清单见 GE 仓文档 https://gitcode.com/cann/ge/blob/master/docs/zh/user_guides/es_graph/api/es_python.md。
"""

from ge.es import GraphBuilder
from ge.es.math import Mul
from ge.es.nn import Threshold
from ge.graph import Operator
from ge.onnx_plugin import OnnxNode, onnx_plugin

my_op = onnx_plugin(
    source="MyOp",  # TODO(填充): 与导出侧 symbolic 对齐
    domain="example.domain",  # TODO(填充): 与导出侧域一致
    opsets=(1,),
    target="PartitionedCall",  # 动态 IO target：decompose 场景常用中转
)


@my_op.parse_node
def parse_my_op(node: OnnxNode, target: Operator) -> None:
    """属性中转 + 端口注册（decompose 依赖这里的产出）。"""
    target.set_attr("alpha", node.attrs.get("alpha", 1.0))  # TODO(填充): 属性中转
    target.register_input("x")  # TODO(填充): 按 ONNX 输入顺序占满位置
    target.register_output("y")


@my_op.decompose
def decompose_my_op(source):
    """用已有算子拼子图替换原节点；返回值必须是 ge.graph.Graph。"""
    alpha = float(source.get_attr("alpha"))  # 读到的是 parse_node 写入的值
    builder = GraphBuilder("my_op_decomposition")
    x = builder.create_input(0)
    # TODO(填充): 用 ge.es.math / ge.es.nn 的算子拼出等价计算子图
    mask = Threshold(x, threshold=alpha)
    output = Mul(x, mask)
    return builder.build_and_reset([output])
