# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""anchor 格式转换：将 source 数据转换为标准 anchor 格式（结构见 references/DESIGN-api-locator.md）。"""

import json
import os


class AnchorFormatter:
    """将 source 数据转换为标准 anchor 格式（结构见 references/DESIGN-api-locator.md）。"""

    def __init__(self, code_map_loader=None, tree_id_text="", repo_root=""):
        self._anchor_counter = 0
        self._ev_counter = 0
        self._code_map_loader = code_map_loader
        self._tree_id_text = tree_id_text
        self._repo_root = repo_root

    def _next_anchor_id(self) -> str:
        self._anchor_counter += 1
        return f"anchor_{self._anchor_counter:03d}"

    def _next_evidence_id(self) -> str:
        self._ev_counter += 1
        return f"ev_{self._ev_counter:03d}"

    def source_to_anchor(self, source: dict, sub_part: str = "function_proto") -> dict:
        anchor_id = self._next_anchor_id()
        ev_id = self._next_evidence_id()
        dimension = source.get("dimension")

        if dimension == "file":
            return self._fill_file_anchor(source, anchor_id, ev_id, sub_part)
        elif dimension == "function":
            return self._fill_function_anchor(source, anchor_id, ev_id, sub_part)
        elif dimension == "comment_block":
            return self._fill_note_anchor(source, anchor_id, ev_id, sub_part)
        elif dimension == "code_snippet":
            return self._fill_section_anchor(source, anchor_id, ev_id, sub_part)
        raise ValueError(f"source 缺少 dimension 字段或值非法: type={source.get('type', '?')}, dimension={dimension}")

    def _get_content(self, source: dict) -> str:
        """source 的内容字段可能是 content 或 snippet。"""
        return source.get("content", source.get("snippet", ""))

    def _read_file_content(self, file_path) -> str:
        """读取文件实际内容。file_path 可以是 str 或 list。"""
        if isinstance(file_path, list):
            return "\n".join(self._read_single_file(p) for p in file_path)
        return self._read_single_file(file_path)

    def _read_single_file(self, file_path: str) -> str:
        """读取单个文件内容，读取失败返回空字符串。"""
        if not file_path:
            return ""
        if os.path.isabs(file_path):
            full_path = file_path
        elif self._repo_root:
            full_path = os.path.join(self._repo_root, file_path)
        else:
            full_path = file_path
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except (FileNotFoundError, PermissionError, OSError):
            return ""

    def _resolve_file_path(self, source: dict) -> "str | list":
        """source 的 file 字段可能是 str 或 list，返回对应的 file_path。"""
        f = source.get("file", "")
        if isinstance(f, list):
            return f
        return f

    def _fill_file_anchor(self, source: dict, anchor_id: str, ev_id: str, sub_part: str = "function_proto") -> dict:
        """file 类型 anchor。"""
        content = self._get_content(source)
        file_path = self._resolve_file_path(source)
        file_name = (
            os.path.basename(file_path) if isinstance(file_path, str) else [os.path.basename(p) for p in file_path]
        )
        code_map_data = ""
        if self._code_map_loader and isinstance(file_path, str):
            code_map_data = self._code_map_loader.load_file_code_map(file_path)
        file_content = self._read_file_content(file_path)
        return {
            "anchor_id": anchor_id,
            "note": "file类型标准结构",
            "anchor_information": {
                "type": "file",
                "level1": {
                    "file_path": file_path,
                    "file_name": file_name,
                },
                "level2": {
                    "code_map": code_map_data,
                    "token_hash": file_content,
                    "file_功能": file_name,
                },
                "level3": {
                    "tree_id": self._tree_id_text,
                },
            },
            "evidence": [
                {
                    "evidence_id": ev_id,
                    "evidence_information": {
                        "file_path": file_path,
                        "file_name": file_name,
                        "file_功能": file_name,
                        "file_content": file_content,
                    },
                    "gene_rule": {
                        "target": [sub_part],
                    },
                }
            ],
        }

    def _fill_function_anchor(self, source: dict, anchor_id: str, ev_id: str, sub_part: str = "function_proto") -> dict:
        """function 类型 anchor。"""
        content = self._get_content(source)
        file_path = self._resolve_file_path(source)
        func_name = source.get("name", "")

        # 优先使用实际函数签名（full_signature 来自 return_type 提取，
        # snippet 来自 extract_snippets，content 是兜底的描述标签）
        full_func = source.get("full_signature", "") or source.get("snippet", "")
        if not full_func:
            full_func = content

        calls, called_by, returns = [], [], []
        if self._code_map_loader and func_name:
            info = self._code_map_loader.get_function_info(func_name)
            calls = info.get("calls", [])
            called_by = info.get("called_by", [])
            returns = info.get("returns", [])

        return {
            "anchor_id": anchor_id,
            "note": "function类型标准结构",
            "anchor_information": {
                "type": "function",
                "level1": {
                    "file_path": file_path,
                    "function_name": func_name,
                },
                "level2": {
                    "call": calls,
                    "callee": called_by,
                    "return": returns,
                },
                "level3": {
                    "tree_id": self._tree_id_text,
                    "function_sign": full_func,
                },
            },
            "evidence": [
                {
                    "evidence_id": ev_id,
                    "evidence_information": {
                        "file_path": file_path,
                        "content": [
                            {
                                "function_hash": full_func,
                                "full_function": full_func,
                            }
                        ],
                    },
                    "call": calls,
                    "callee": called_by,
                    "return": returns,
                    "gene_rule": {
                        "target": [sub_part],
                    },
                }
            ],
        }

    def _fill_note_anchor(self, source: dict, anchor_id: str, ev_id: str, sub_part: str = "function_proto") -> dict:
        """note 类型 anchor。

        level1.behind_code = 注释下方第一个完整代码块（函数签名）
        level2.token_hash = behind_code 的原始内容
        level2.note_hash = 注释本身的原始内容
        """
        content = self._get_content(source)
        note_content = source.get("note_content", content)
        behind_code = source.get("behind_code", "")
        file_path = self._resolve_file_path(source)
        return {
            "anchor_id": anchor_id,
            "note": "note类型标准结构",
            "anchor_information": {
                "type": "note",
                "level1": {
                    "file_path": file_path,
                    "behind_code": behind_code,
                },
                "level2": {
                    "token_hash": behind_code,
                    "note_hash": note_content,
                },
            },
            "evidence": [
                {
                    "evidence_id": ev_id,
                    "evidence_information": {
                        "file_path": file_path,
                        "content": content,
                        "behind_code": behind_code,
                    },
                    "gene_rule": {
                        "target": [sub_part],
                    },
                }
            ],
        }

    def _fill_section_anchor(self, source: dict, anchor_id: str, ev_id: str, sub_part: str = "function_proto") -> dict:
        """section 类型 anchor。"""
        content = self._get_content(source)
        file_path = self._resolve_file_path(source)
        return {
            "anchor_id": anchor_id,
            "note": "section类型标准结构",
            "anchor_information": {
                "type": "section",
                "level1": {
                    "file_path": file_path,
                    "token_hash": content,
                },
                "level2": {
                    "token_hash": content,
                },
            },
            "evidence": [
                {
                    "evidence_id": ev_id,
                    "evidence_information": {
                        "file_path": file_path,
                        "content": content,
                    },
                    "gene_rule": {
                        "target": [sub_part],
                    },
                }
            ],
        }

    def sources_to_anchors(self, sources: list, sub_part: str = "function_proto") -> list:
        anchors = []
        for source in sources:
            anchor = self.source_to_anchor(source, sub_part)
            anchors.append(anchor)
        return anchors

    def build_chapter_json(
        self,
        api_name: str,
        doc_path: str,
        chapter: str,
        anchors: list,
        sources: list = None,
        doc_content: str = "",
        function_proto: str = "",
    ) -> dict:
        """组装完整 JSON。

        Args:
            chapter: 章节名，如 "功能说明"
            sources: 原始 sources 列表（可选，用于产品支持情况注入 arch 字段）
            doc_content: 接口文档中该章节的原始内容
            function_proto: 函数原型（仅函数原型章节需要）
        """
        from doc_parser import get_sub_part

        sub_part = get_sub_part(chapter)
        # 动态顶层 key
        content_value = {"content": doc_content} if doc_content else {}
        if chapter == "函数原型" and function_proto:
            content_value = {"content": function_proto}

        doc = {
            "doc_path": doc_path,
            "doc_name": f"{api_name}.md",
            "doc_part": sub_part,
            "sub_part": content_value,
            "md_format": doc_content,
        }

        # 产品支持情况：从 chip_arch source 注入 arch 列表 + chip 列表
        if chapter == "产品支持情况" and sources:
            for s in sources:
                if s.get("type") == "chip_arch":
                    doc["arch"] = s.get("arch_list", [])
                    doc["arch_sources"] = s.get("arch_sources", {})
                    doc["chip"] = s.get("chip_list", [])
                    doc["chip_sources"] = s.get("chip_sources", {})
                    break

        return {
            "doc": doc,
            "anchor_tuple": anchors,
        }

    def write_chapter_json(
        self,
        export_dir: str,
        api_name: str,
        doc_path: str,
        chapter: str,
        anchors: list,
        sources: list = None,
        doc_content: str = "",
        function_proto: str = "",
    ) -> str:
        """写入 {sub_part}.json 到 export_dir。"""
        from doc_parser import get_sub_part

        data = self.build_chapter_json(api_name, doc_path, chapter, anchors, sources, doc_content, function_proto)
        sub_part = get_sub_part(chapter)
        filename = f"{sub_part}.json"
        out_path = os.path.join(export_dir, filename)
        os.makedirs(export_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return out_path
