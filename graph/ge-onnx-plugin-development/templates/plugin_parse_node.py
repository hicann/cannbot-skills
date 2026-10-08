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

"""一对一映射插件模板（parse_node，最常用）。

适用：GE 已有现成算子可映射（target 静态 IR 时无需端口注册；动态 IO 时按注释注册）。
填充指引（使用前删除本段）：
- source/domain/opsets 与导出侧 symbolic 域、custom_opsets 版本两侧一致（references/export.md §3）；
- 属性键名 = 导出侧 symbolic 的关键字参数名；
- 端口注册规则见 references/plugin.md §3（顺序=ONNX 输入顺序，位置占满）。
"""

from ge.graph import Operator
from ge.onnx_plugin import OnnxNode, onnx_plugin

my_op = onnx_plugin(
    source="MyOp",  # TODO(填充): symbolic 里 域:: 后面的算子名
    domain="example.domain",  # TODO(填充): 与 symbolic 的域一致
    opsets=(1,),  # TODO(填充): 覆盖 custom_opsets 登记的版本
    target="Elu",  # TODO(填充): GE 目标算子类型（原型已安装注册）
)


@my_op.parse_node
def parse_my_op(node: OnnxNode, target: Operator) -> None:
    """按名读取 ONNX 属性写入目标算子；动态 IO target 时注册端口。"""
    # TODO(填充): 属性中转（键=导出侧属性名；值类型由导出后缀决定）
    target.set_attr("alpha", node.attrs.get("alpha", 1.0))

    # TODO(填充): 仅动态 IO target（如 PartitionedCall）需要以下端口注册；
    # 静态 IR target（如 Elu，原型已声明 INPUT(x)/OUTPUT(y)）必须删除以下两行
    target.register_input("x")
    target.register_output("y")
