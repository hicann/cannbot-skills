# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging
import os

import config
from clang.cindex import CursorKind, Index

logger = logging.getLogger(__name__)


class ClangParser:
    def __init__(self, project_root: str):
        """
        初始化 ClangParser

        project_root: 项目根目录，用于 -I include 搜索路径
        """
        self.index = Index.create()
        self.project_root = project_root

    def build_clang_args(self):
        """
        构造 clang 解析参数

        -x c++        指定语言为 C++
        -std=c++17    使用 C++17 标准
        -DACLNN_API=  定义空宏，避免 clang 解析失败
        -I            添加头文件搜索路径
        """
        return config.CLANG_LANG_FLAGS + [
            config.CLANG_STD,
            config.CLANG_ACLNN_DEFINE,
            "-I" + self.project_root,
        ]

    def make_symbol_id(self, name):
        """
        生成符号唯一 ID

        当前实现：直接返回函数名
        """
        return name

    def get_parent_function(self, node):
        """
        向上遍历 semantic_parent 找到所属函数

        用于在解析调用表达式时，稳定找到 caller，
        避免单纯依赖 current_function 导致的误归属。
        """
        parent = node.semantic_parent

        while parent is not None:
            if parent.kind == CursorKind.FUNCTION_DECL and parent.spelling:
                return parent
            parent = parent.semantic_parent

        return None

    def extract_callee_from_tokens(self, node):
        """
        fallback 方案：从 CALL_EXPR / CXX_MEMBER_CALL_EXPR 的 token 中提取被调用函数名

        处理形式：
        - self->Method()  -> Method
        - obj.method()    -> method
        - ns::Func()      -> Func

        思路：取第一个 "(" 之前最后一个有效标识符
        """
        try:
            tokens = list(node.get_tokens())

            before_paren = []
            for t in tokens:
                spelling = t.spelling
                if spelling == "(":
                    break
                before_paren.append(spelling)

            # 从右往左找第一个有效名字
            for spelling in reversed(before_paren):
                if spelling in ["(", ")", ",", "::", ".", "->", "*", "&"]:
                    continue
                if spelling in ["self", "this"]:
                    continue
                # namespace::func 取最后一级
                return spelling.split("::")[-1]

        except Exception:
            pass

        return None

    def parse_file(self, filepath: str) -> tuple[list, list]:
        """
        解析单个 C++ 文件

        返回:
            nodes: 节点列表
            edges: 关系列表

        解析失败时返回 ([], [])
        """
        nodes = []
        edges = []
        node_set = set()

        current_function = None
        call_order_map = {}

        # 使用 clang 解析源码生成 AST
        try:
            tu = self.index.parse(filepath, args=self.build_clang_args())
        except Exception as e:
            logger.error("Clang parse failed: %s", filepath)
            logger.error(str(e))
            return [], []

        def add_node(node_id, node_type, path=None):
            """添加节点（去重）"""
            key = (node_id, path)
            if key in node_set:
                return

            node = {"id": node_id, "type": node_type}
            if path:
                node["path"] = path

            nodes.append(node)
            node_set.add(key)

        # 当前文件绝对路径
        abs_path = os.path.abspath(filepath)

        # 遍历整个 AST 语法树
        for node in tu.cursor.walk_preorder():
            # 如果节点没有文件信息则跳过
            if node.location.file is None:
                continue

            node_file = os.path.abspath(node.location.file.name)

            # 只处理当前文件里的节点
            if node_file != abs_path:
                continue

            # --------------------------------------------------
            # Function 节点提取
            # --------------------------------------------------
            if node.kind == CursorKind.FUNCTION_DECL:
                func_name = node.spelling
                if not func_name:
                    continue

                symbol_id = self.make_symbol_id(func_name)
                add_node(symbol_id, "Function", filepath)

                # 函数定义
                if node.is_definition():
                    current_function = symbol_id

                    edges.append(
                        {
                            "type": "defines",
                            "source": filepath,
                            "target": symbol_id,
                            "sourcefile": filepath,
                            "targetfile": filepath,
                            "file": filepath,
                        }
                    )

                    # 解析函数内部 token，查找错误码
                    try:
                        tokens = list(node.get_tokens())
                        for t in tokens:
                            name = t.spelling
                            if name.startswith("ACLNN_ERR"):
                                add_node(name, "errorcode", None)
                                edges.append(
                                    {
                                        "type": "returns",
                                        "source": symbol_id,
                                        "target": name,
                                        "sourcefile": filepath,
                                        "targetfile": None,
                                        "file": filepath,
                                    }
                                )
                    except Exception:
                        pass

                # 函数声明
                else:
                    edges.append(
                        {
                            "type": "declares",
                            "source": filepath,
                            "target": symbol_id,
                            "sourcefile": filepath,
                            "targetfile": filepath,
                            "file": filepath,
                        }
                    )

                continue

            # --------------------------------------------------
            # Function Call 解析
            # --------------------------------------------------
            call_kinds = [CursorKind.CALL_EXPR]
            if hasattr(CursorKind, "CXX_MEMBER_CALL_EXPR"):
                call_kinds.append(CursorKind.CXX_MEMBER_CALL_EXPR)

            if node.kind in call_kinds:
                parent_func = self.get_parent_function(node)

                if parent_func is not None:
                    caller = self.make_symbol_id(parent_func.spelling)
                else:
                    caller = current_function

                if not caller:
                    continue

                callee = None
                callee_file = None

                # 优先使用 clang AST 获取被调用函数
                try:
                    ref = node.referenced
                    if ref and ref.spelling:
                        callee = ref.spelling
                        if ref.location.file:
                            callee_file = ref.location.file.name
                except Exception:
                    pass

                # fallback 方案
                if not callee:
                    callee = self.extract_callee_from_tokens(node)

                # 创建函数节点 + calls 关系
                if callee:
                    if callee_file is None and callee.isupper():
                        continue

                    add_node(callee, "Function", callee_file)

                    if caller not in call_order_map:
                        call_order_map[caller] = 0
                    call_order_map[caller] += 1

                    edges.append(
                        {
                            "type": "calls",
                            "source": caller,
                            "target": callee,
                            "sourcefile": filepath,
                            "targetfile": callee_file,
                            "file": filepath,
                            "line": node.location.line,
                            "column": node.location.column,
                            "order": call_order_map[caller],
                        }
                    )

                continue

            # --------------------------------------------------
            # Namespace 解析
            # --------------------------------------------------
            if node.kind == CursorKind.NAMESPACE:
                name = node.spelling
                if name:
                    add_node(name, "namespace", filepath)
                    edges.append(
                        {
                            "type": "defines",
                            "source": filepath,
                            "target": name,
                            "sourcefile": filepath,
                            "targetfile": filepath,
                            "file": filepath,
                        }
                    )
                continue

            # --------------------------------------------------
            # Class 解析
            # --------------------------------------------------
            if node.kind == CursorKind.CLASS_DECL:
                name = node.spelling
                if name:
                    add_node(name, "class", filepath)
                    edges.append(
                        {
                            "type": "defines",
                            "source": filepath,
                            "target": name,
                            "sourcefile": filepath,
                            "targetfile": filepath,
                            "file": filepath,
                        }
                    )
                continue

            # --------------------------------------------------
            # Struct 解析
            # --------------------------------------------------
            if node.kind == CursorKind.STRUCT_DECL:
                name = node.spelling
                if name:
                    add_node(name, "struct", filepath)
                    edges.append(
                        {
                            "type": "defines",
                            "source": filepath,
                            "target": name,
                            "sourcefile": filepath,
                            "targetfile": filepath,
                            "file": filepath,
                        }
                    )
                continue

        return nodes, edges
