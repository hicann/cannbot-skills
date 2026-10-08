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

"""JSON 整体解析插件模板（parse_operator）。

适用：节点带 tensor/子图属性（node.attrs 读到会报错），或需不预知属性名整体搬运。
填充指引（使用前删除本段）：
- source 只读；全部节点属性打包为 JSON 串存于 "attribute" 键；
- type 码：1=float 2=int 3=string 4=tensor 5=子图 6=float列表 7=int列表 8=string列表；
- 标量字段（f/i/s）的值是字符串形式，取用时转数值。
"""

import json

from ge.onnx_plugin import onnx_plugin

my_op = onnx_plugin(
    source="MyOp",  # TODO(填充): 与导出侧 symbolic 对齐
    domain="example.domain",  # TODO(填充): 与导出侧域一致
    opsets=(1,),
    target="Elu",  # TODO(填充): GE 目标算子类型
)


@my_op.parse_operator
def parse_my_op(source, target) -> None:
    """从 source 的 JSON 属性串解析所需属性，转写给目标算子。"""
    attrs = json.loads(source.get_attr("attribute"))
    alpha = 1.0
    # TODO(填充): 按属性名在 JSON 数组中查找；值字段按 type 码选择（f/i/s/floats/ints/strings）
    for attr in attrs.get("attribute", []):
        if attr.get("name") == "alpha":
            alpha = float(attr.get("f", 1.0))
    target.set_attr("alpha", alpha)
