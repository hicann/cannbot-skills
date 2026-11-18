# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""证据拉取核心：配置 + 代码提取 + API 定位 + 章节组装。"""

import glob
import json
import os
import re
from dataclasses import dataclass, field

from config import (
    ADV_API_CHECK_DIR,
    ADV_API_CHECK_FILE,
    BASE_API_LIST,
    CHAPTERS,
    COMMON_TEST_DIRS,
    IMPL_BASIC_DAV_GLOB,
    IMPL_CHECK_GLOB,
    OPERATOR_CATEGORIES,
    OPERATOR_TO_API,
    TEST_DIR_MAP,
)


def _search_in_files(*args, **kwargs):
    """惰性导入跨平台文件搜索，避免把 sys.path 引导写成模块级语句。"""
    import sys

    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from common.file_search import search_in_files

    return search_in_files(*args, **kwargs)


def _np(path: str) -> str:
    """路径分隔符归一化：统一使用正斜杠 /，兼容 Windows/Linux/macOS。"""
    return path.replace(os.sep, "/")


def _rp(full_path, base_path):
    """跨平台相对路径：等价于 _np(os.path.relpath(full_path, base_path))。"""
    return _np(os.path.relpath(full_path, base_path))


# ==================== 配置常量（从 config.py 导入，本模块自用） ====================



def _classify_operator(api_name: str) -> str | None:
    """根据 API 名判断运算符类别。

    Returns:
        "symbol_override" — 算术/比较运算符（有 OPERATOR_TO_API 映射）
        "subscript"       — operator[]（Tensor 偏移访问）
        "element_access"  — operator()（Tensor 元素值访问）
        "assignment"      — operator=（赋值）
        None              — 非运算符
    """
    if api_name in OPERATOR_TO_API:
        return "symbol_override"
    if api_name == "operator[]":
        return "subscript"
    if api_name == "operator()":
        return "element_access"
    if api_name == "operator=":
        return "assignment"
    # 其他 operator 开头的（不在映射表中但以 operator 开头）
    if api_name.startswith("operator"):
        return "symbol_override"  # 兜底：可能是不在表中的算术运算符
    return None


def _extract_operator_name_from_doc_title(doc_path: str) -> str:
    """从文档的第一个 # 标题提取运算符名。

    标题格式示例：
      # operator\\[\\]         → "operator[]"  （Markdown 转义自动去除）
      # operator\\(\\)         → "operator()"
      # operator=              → "operator="
      # operator =             → "operator="    （空格自动归一化）
      # 运算符重载              → "运算符重载"

    如果标题提取失败，fallback 到文件名清洗（仅去数字后缀，不去括号）。
    """
    try:
        with open(doc_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    title = re.sub(r"^#+\s+", "", line).strip()
                    # 去掉 HTML anchor 标签 <a name="..."></a>
                    title = re.sub(r'<a\s+name="[^"]*"></a>', "", title).strip()
                    if title:
                        # 去除 Markdown 反斜杠转义: \[ → [, \] → ], \( → (, \) → )
                        title = re.sub(r"\\([][()])", r"\1", title)
                        # 归一化运算符空格: "operator =" → "operator="
                        title = re.sub(r"^(operator)\s+(.+)$", r"\1\2", title)
                        return title
                elif line and not line.startswith("<"):
                    break  # 第一个非空非标题行 → 标题区结束
    except (FileNotFoundError, PermissionError):
        pass

    # fallback：文件名清洗（仅去数字后缀，不去括号）
    basename = os.path.splitext(os.path.basename(doc_path))[0]
    basename = re.sub(r"-\d+$", "", basename)
    return basename.strip()


# ==================== 章节提取器引用 ====================
# 延迟导入，避免循环依赖（chapters → evidence → chapters）


# ==================== Evidence ====================


@dataclass
class Evidence:
    """四维之一的一个证据项。"""

    dimension: str  # "file" | "function" | "code_snippet" | "comment_block"
    file: str
    line: int
    content: str
    metadata: dict = field(default_factory=dict)


def deduplicate_by_content(evidence: list[Evidence]) -> list[Evidence]:
    """同章节内同内容去重:仅当 (type, content, file) 完全相同时去重。

    不同 file 的相同 content 保留为独立 Evidence,确保每个 file 单独生成 anchor,
    避免 file_path 合并为 list 导致 anchor 与多文件绑定(下游重定向模块假定 str)。

    保留同 (type, content, file) 去重:同文件内重复 content(如同一文件多处相同调用行)
    合并为一条,避免冗余 anchor。
    """
    seen = set()
    result = []
    for ev in evidence:
        key = (ev.metadata.get("type", ""), ev.content, ev.file)
        if key not in seen:
            seen.add(key)
            result.append(ev)
    return result


# ==================== Extractor ====================


class Extractor:
    """从 C++ 源码提取各类代码段/注释。"""

    def __init__(self, repo: str):
        self.repo = repo

    @staticmethod
    def _extract_full_comment_block(lines: list, func_line: int) -> tuple:
        """从函数行往前找到完整 doxygen 注释块和函数签名。

        Returns:
            (comment_text, behind_code_text) — 注释原文 + 函数签名
        """
        j = func_line - 1
        while j >= 0 and (lines[j].strip() == "" or lines[j].strip().startswith("template")):
            j -= 1
        if j < 0 or "*/" not in lines[j]:
            return ("", "")
        end_comment = j
        while j >= 0 and "/*" not in lines[j]:
            j -= 1
        if j < 0:
            return ("", "")
        comment_block = "".join(lines[j: end_comment + 1])

        sig_lines = []
        k = func_line
        while k < len(lines):
            sig_lines.append(lines[k])
            if ";" in lines[k] or "{" in lines[k]:
                break
            k += 1
        behind_code = "".join(sig_lines).strip()

        return (comment_block.strip(), behind_code)

    def extract_doxygen_brief(self, api_name: str, decl_file: str) -> list:
        """提取 doxygen brief 注释。"""
        full_path = os.path.join(self.repo, decl_file)
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except (FileNotFoundError, PermissionError):
            return []

        results = []
        api_pattern = re.compile(rf"\b{re.escape(api_name)}\s*[<(]")

        for i, line in enumerate(lines):
            if api_pattern.search(line):
                comment_text, behind_code = self._extract_full_comment_block(lines, i)
                if comment_text:
                    brief_match = re.search(r"@brief\s+(.+)", comment_text)
                    if brief_match:
                        brief_text = brief_match.group(1).strip()
                        brief_text = re.sub(r"^\s*\*\s*", "", brief_text)
                        brief_text = re.sub(r"\s*\*/\s*$", "", brief_text)
                        results.append(
                            Evidence(
                                dimension="comment_block",
                                file=decl_file,
                                line=i + 1,
                                content=brief_text.strip(),
                                metadata={
                                    "type": "doxygen_brief",
                                    "note_content": comment_text,
                                    "behind_code": behind_code,
                                },
                            )
                        )
        return results

    def extract_doxygen_params(self, api_name: str, decl_file: str) -> list:
        """从声明文件的 Doxygen 注释中提取 @param。仅 basic_api 有。"""
        full_path = os.path.join(self.repo, decl_file)
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except (FileNotFoundError, PermissionError):
            return []

        results = []
        api_pattern = re.compile(rf"\b{re.escape(api_name)}\s*[<(]")

        for i, line in enumerate(lines):
            if api_pattern.search(line):
                comment_text, behind_code = self._extract_full_comment_block(lines, i)
                if comment_text:
                    param_matches = re.finditer(r"@param\s+\[([^]]+)\]\s+(\S+)\s+(.*)", comment_text)
                    found_any = False
                    for pm in param_matches:
                        found_any = True
                        direction = pm.group(1)
                        param_name = pm.group(2)
                        desc = pm.group(3).strip()
                        desc = re.sub(r"\s*\*/\s*$", "", desc)
                        results.append(
                            Evidence(
                                dimension="comment_block",
                                file=decl_file,
                                line=i + 1,
                                content=f"{param_name} ({direction}): {desc}",
                                metadata={
                                    "type": "doxygen_param",
                                    "direction": direction,
                                    "name": param_name,
                                    "note_content": comment_text,
                                    "behind_code": behind_code,
                                },
                            )
                        )
                    if not found_any and comment_text:
                        results.append(
                            Evidence(
                                dimension="comment_block",
                                file=decl_file,
                                line=i + 1,
                                content=comment_text,
                                metadata={
                                    "type": "doxygen_param",
                                    "note_content": comment_text,
                                    "behind_code": behind_code,
                                },
                            )
                        )
        return results

    def extract_return_type(self, api_name: str, decl_file: str) -> list:
        """从声明文件函数签名中提取返回类型。"""
        full_path = os.path.join(self.repo, decl_file)
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except (FileNotFoundError, PermissionError):
            return []

        def _extract_full_signature(lines, decl_start_idx):
            sig_lines = []
            start = decl_start_idx
            while start > 0 and lines[start - 1].strip().startswith("template"):
                start -= 1
            for i in range(start, len(lines)):
                sig_lines.append(lines[i])
                if ";" in lines[i] or "{" in lines[i]:
                    break
            return "".join(sig_lines).strip()

        results = []
        api_pattern = re.compile(rf"\b{re.escape(api_name)}\s*[<(]")

        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except (FileNotFoundError, PermissionError):
            return []

        for i, line in enumerate(lines):
            if api_pattern.search(line):
                sig = _extract_full_signature(lines, i)
                ret_match = re.search(
                    rf"(?:__aicore__\s+)?inline\s+(\S+)\s+{re.escape(api_name)}\s*[<(]",
                    line,
                )
                if not ret_match:
                    ret_match2 = re.search(
                        rf"inline\s+auto\s+{re.escape(api_name)}\s*[<(]",
                        line,
                    )
                    ret_type = "auto" if ret_match2 else ""
                else:
                    ret_type = ret_match.group(1)
                if ret_type:
                    results.append(
                        Evidence(
                            dimension="code_snippet",
                            file=decl_file,
                            line=i + 1,
                            content=ret_type,
                            metadata={
                                "type": "return_type",
                                "name": api_name,
                                "full_signature": sig,
                            },
                        )
                    )

        return results

    def extract_function_params(self, api_name: str, decl_file: str) -> list:
        """从函数签名中提取参数列表（无 @param 注释时使用）。"""
        if not decl_file:
            return []
        full_path = os.path.join(self.repo, decl_file)
        if os.path.isdir(full_path):
            return []
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except (FileNotFoundError, PermissionError):
            return []

        results = []
        api_pattern = re.compile(rf"\b{re.escape(api_name)}\s*[<(]")

        for i, line in enumerate(lines):
            if api_pattern.search(line):
                start = i
                while start > 0 and lines[start - 1].strip().startswith("template"):
                    start -= 1
                sig_lines = []
                end = start
                while end < len(lines):
                    sig_lines.append(lines[end])
                    if ";" in lines[end] or "{" in lines[end]:
                        break
                    end += 1
                sig = "".join(sig_lines)
                paren_start = sig.find("(")
                if paren_start >= 0:
                    depth = 0
                    paren_end = paren_start
                    for j in range(paren_start, len(sig)):
                        if sig[j] == "(":
                            depth += 1
                        elif sig[j] == ")":
                            depth -= 1
                            if depth == 0:
                                paren_end = j
                                break
                    params_str = sig[paren_start + 1: paren_end].strip()
                    if params_str:
                        results.append(
                            Evidence(
                                dimension="code_snippet",
                                file=decl_file,
                                line=i + 1,
                                content=params_str,
                                metadata={"type": "function_params"},
                            )
                        )
                break
        return results

    def extract_static_asserts(self, impl_files: list) -> list:
        """从 impl 文件中提取 static_assert（数据类型约束）。"""
        results = []
        for impl_file in impl_files:
            full_path = os.path.join(self.repo, impl_file)
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except (FileNotFoundError, PermissionError):
                continue

            for i, line in enumerate(lines):
                if "static_assert" in line and ("SupportType" in line or "support" in line.lower()):
                    full_line = line
                    end = i
                    while ")," not in full_line and end < len(lines) - 1:
                        end += 1
                        full_line += lines[end]
                    results.append(
                        Evidence(
                            dimension="code_snippet",
                            file=impl_file,
                            line=i + 1,
                            content=full_line.strip(),
                            metadata={"type": "static_assert"},
                        )
                    )
        return results

    def extract_arch_conditions(self, decl_file: str) -> list:
        """从声明文件中提取 __NPU_ARCH__ 条件编译。"""
        full_path = os.path.join(self.repo, decl_file)
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except (FileNotFoundError, PermissionError):
            return []

        results = []
        for i, line in enumerate(lines):
            if "__NPU_ARCH__" in line and ("#if" in line or "#elif" in line):
                cond_match = re.search(r"#(?:if|elif)\s+(.+)", line)
                if cond_match:
                    results.append(
                        Evidence(
                            dimension="code_snippet",
                            file=decl_file,
                            line=i + 1,
                            content=cond_match.group(1).strip(),
                            metadata={"type": "arch_condition"},
                        )
                    )
        return results

    def extract_check_functions(self, api_name: str, api_type: str, impl_files: list = None) -> list:
        """提取约束检查函数。"""
        results = []

        if api_type == "basic_api":
            check_pattern = os.path.join(self.repo, IMPL_CHECK_GLOB)
            check_files = sorted(glob.glob(check_pattern))
            for check_file in check_files:
                rel_path = _rp(check_file, self.repo)
                try:
                    with open(check_file, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except (FileNotFoundError, PermissionError):
                    continue
                if re.search(rf"\b{re.escape(api_name)}\b", content, re.IGNORECASE):
                    results.extend(self._extract_check_sigs(content, rel_path))

            if not results and impl_files:
                category = self._infer_check_category(impl_files)
                if category:
                    for check_file in check_files:
                        if category in os.path.basename(check_file):
                            rel_path = _rp(check_file, self.repo)
                            try:
                                with open(check_file, "r", encoding="utf-8", errors="replace") as f:
                                    content = f.read()
                            except (FileNotFoundError, PermissionError):
                                continue
                            results.extend(self._extract_check_sigs(content, rel_path))

        elif api_type == "adv_api":
            check_dirs = [
                ADV_API_CHECK_FILE,
                ADV_API_CHECK_DIR,
            ]
            for check_rel in check_dirs:
                check_path = os.path.join(self.repo, check_rel)
                if os.path.isfile(check_path):
                    results.append(
                        Evidence(
                            dimension="file",
                            file=check_rel,
                            line=1,
                            content="CHECK_FUNC_HIGHLEVEL_API 宏定义",
                            metadata={"type": "api_check"},
                        )
                    )
                elif os.path.isdir(check_path):
                    for cf in sorted(glob.glob(os.path.join(check_path, "*.h"))):
                        rel_cf = _rp(cf, self.repo)
                        results.append(
                            Evidence(
                                dimension="file",
                                file=rel_cf,
                                line=1,
                                content=os.path.basename(rel_cf),
                                metadata={"type": "api_check"},
                            )
                        )

        # #24 修复：为 simt_api、tensor_api、c_api、utils 添加 check 文件搜索
        elif api_type in ("simt_api", "tensor_api", "c_api", "utils"):
            # 对这些类型，递归 glob 所有 *check*.h 文件，做内容验证
            impl_dir = f"impl/{api_type}"
            check_pattern = os.path.join(self.repo, impl_dir, "**", "*check*.h")
            check_files = sorted(glob.glob(check_pattern, recursive=True))
            for check_file in check_files:
                rel_path = _rp(check_file, self.repo)
                try:
                    with open(check_file, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except (FileNotFoundError, PermissionError):
                    continue
                if re.search(rf"\b{re.escape(api_name)}\b", content, re.IGNORECASE):
                    results.extend(self._extract_check_sigs(content, rel_path))

        return results

    @staticmethod
    def _extract_check_sigs(content: str, rel_path: str) -> list:
        """从 check 文件内容中提取 Check 函数签名。"""
        results = []
        for m in re.finditer(r"(bool\s+\w*Check\w*\([^)]*\))", content):
            line_num = content[: m.start()].count("\n") + 1
            sig = m.group(1).strip()
            fn = re.search(r"\b(\w+)\s*\(", sig)
            meta = {"type": "check_function"}
            if fn:
                meta["name"] = fn.group(1)
            results.append(
                Evidence(
                    dimension="function",
                    file=rel_path,
                    line=line_num,
                    content=sig,
                    metadata=meta,
                )
            )
        return results

    @staticmethod
    def _infer_check_category(impl_files: list) -> "str | None":
        """从 impl 文件路径推断 check 文件类别。"""
        for impl in impl_files:
            base = os.path.basename(impl)
            for cat in ["vec_binary", "vec_unary", "vec_reduce", "vec_select", "data_copy"]:
                if cat in base:
                    return cat
            if base.startswith("kernel_operator") and "vec_" not in base:
                return "kernel_operator"
        return None

    def extract_impl_internal_calls(self, api_name: str, impl_files: list) -> list:
        """提取 impl 文件中调用的内部基础 API（用于实现原理章节）。"""
        results = []
        base_api_pattern = re.compile(rb"\b(" + rb"|".join(name.encode() for name in BASE_API_LIST) + rb")\s*[<(]")
        for impl_file in impl_files:
            full_path = os.path.join(self.repo, impl_file)
            try:
                with open(full_path, "rb") as f:
                    content = f.read()
            except (FileNotFoundError, PermissionError):
                continue

            seen = set()
            for m in base_api_pattern.finditer(content):
                call_name = m.group(1).decode()
                if call_name == api_name:
                    continue
                if call_name not in seen:
                    seen.add(call_name)
                    line_num = content[: m.start()].count(b"\n") + 1
                    line_start = content.rfind(b"\n", 0, m.start()) + 1
                    line_end = content.find(b"\n", m.end())
                    if line_end == -1:
                        line_end = len(content)
                    line_text = content[line_start:line_end].decode("utf-8", errors="replace").strip()
                    results.append(
                        Evidence(
                            dimension="code_snippet",
                            file=impl_file,
                            line=line_num,
                            content=line_text,
                            metadata={"type": "impl_internal_call", "called_api": call_name},
                        )
                    )
        return results

    def extract_test_calls(self, api_name: str, test_files: list) -> list:
        """从测试文件中提取 API 调用代码。"""
        results = []
        pattern = re.compile(rf"\b{re.escape(api_name)}\s*[<(]".encode())
        for test_file in test_files:
            full_path = os.path.join(self.repo, test_file)
            try:
                with open(full_path, "rb") as f:
                    lines = f.readlines()
            except (FileNotFoundError, PermissionError):
                continue

            for i, line_bytes in enumerate(lines):
                if pattern.search(line_bytes):
                    line_text = line_bytes.decode("utf-8", errors="replace").strip()
                    results.append(
                        Evidence(
                            dimension="code_snippet",
                            file=test_file,
                            line=i + 1,
                            content=line_text,
                            metadata={"type": "test_file_call"},
                        )
                    )
        return results

    def extract_snippets(self, file_path: str, search_name: str, is_declaration: bool) -> list:
        """从文件中提取目标函数的代码段。"""
        full_path = os.path.join(self.repo, file_path)
        try:
            with open(full_path, "rb") as f:
                raw_lines = f.readlines()
        except (FileNotFoundError, PermissionError):
            return []

        lines = [ln.decode("utf-8", errors="replace") for ln in raw_lines]
        pattern = re.compile(rf"\b{re.escape(search_name)}\s*[<(]".encode())
        snippets = []
        i = 0
        while i < len(lines):
            line_bytes = lines[i].encode("utf-8", errors="replace")
            if pattern.search(line_bytes):
                start = i
                while start > 0 and lines[start - 1].strip().startswith("template"):
                    start -= 1

                if is_declaration:
                    end = i
                    while end < len(lines) and ";" not in lines[end] and "{" not in lines[end]:
                        end += 1
                    if end < len(lines) and "{" in lines[end]:
                        brace_depth = 0
                        found_open = False
                        tmp = end
                        while tmp < len(lines):
                            for ch in lines[tmp]:
                                if ch == "{":
                                    brace_depth += 1
                                    found_open = True
                                elif ch == "}":
                                    brace_depth -= 1
                            if found_open and brace_depth <= 0:
                                end = tmp + 1
                                break
                            tmp += 1
                        else:
                            end = len(lines)
                    elif end < len(lines):
                        end += 1
                else:
                    end = i
                    brace_depth = 0
                    found_open = False
                    while end < len(lines):
                        line_bytes = lines[end].encode("utf-8", errors="replace")
                        for ch in line_bytes:
                            if ch == ord(b"{"):
                                brace_depth += 1
                                found_open = True
                            elif ch == ord(b"}"):
                                brace_depth -= 1
                        if found_open and brace_depth <= 0:
                            end += 1
                            break
                        end += 1

                snippet_text = "".join(lines[start:end])
                snippets.append(
                    Evidence(
                        dimension="code_snippet",
                        file=file_path,
                        line=start + 1,
                        content=snippet_text.rstrip(),
                        metadata={
                            "type": "declaration_snippet" if is_declaration else "impl_snippet",
                            "snippet": snippet_text.rstrip(),
                        },
                    )
                )
                i = end
            else:
                i += 1
        return snippets

    def extract_all_decl_snippets(self, api_name: str, decl_file: str) -> list:
        """提取声明文件中所有目标 API 的函数签名代码段。"""
        return self.extract_snippets(decl_file, api_name, is_declaration=True)

    def api_in_file(self, file_path: str, api_name: str, api_type: str) -> bool:
        """检查 impl 文件中是否包含目标 API（词边界匹配）。

        c_api 实现文件常以 {api_name}_impl 命名/定义标识符，
        需同时匹配 {api_name} 和 {api_name}_impl（对齐 api_change_detect 逻辑）。
        """
        if api_type == "basic_api":
            # 同时匹配 {api_name}Impl（Impl 类约定）与 {api_name} 本身（裸函数
            # 实现约定，如 kernel_operator_data_copy_intf_impl.h 里的
            # SetLoopModePara——该文件不含 Impl 类，仅按 Impl 后缀搜索会漏）
            search_names = [f"{api_name}Impl", api_name]
        elif api_type == "c_api" and not api_name.endswith("_impl"):
            search_names = [api_name, f"{api_name}_impl"]
        else:
            search_names = [api_name]
        patterns = [re.compile(rf"\b{re.escape(n)}\b".encode()) for n in search_names]
        full_path = os.path.join(self.repo, file_path)
        try:
            with open(full_path, "rb") as f:
                for line in f:
                    if any(p.search(line) for p in patterns):
                        return True
        except (FileNotFoundError, PermissionError):
            return False
        return False

    def _api_in_file_raw(self, file_path: str, api_name: str) -> bool:
        """检查 impl 文件中是否包含 API 名本身（不追加 Impl 后缀）。"""
        pattern = re.compile(rf"\b{re.escape(api_name)}\b".encode())
        full_path = os.path.join(self.repo, file_path)
        try:
            with open(full_path, "rb") as f:
                for line in f:
                    if pattern.search(line):
                        return True
        except (FileNotFoundError, PermissionError):
            return False
        return False

    def api_in_test_file(self, file_path: str, api_name: str) -> bool:
        """检查测试文件中是否包含目标 API 调用（分层匹配）。

        第一层：函数调用/模板形式 \\b{name}\\s*[<(]（精确，原有逻辑）
        第二层：纯词边界 \\b{name}\\b（覆盖宏参数形式，如 TEST_..(Abs, asc_abs, ..)）
        第三层：文件名含算子名（覆盖纯 #include 型测试，文件名即关联）
        """
        pattern_call = re.compile(rf"\b{re.escape(api_name)}\s*[<(]".encode())
        pattern_word = re.compile(rf"\b{re.escape(api_name)}\b".encode())
        full_path = os.path.join(self.repo, file_path)
        try:
            with open(full_path, "rb") as f:
                content = f.read()
            if pattern_call.search(content):
                return True
            if pattern_word.search(content):
                return True
        except (FileNotFoundError, PermissionError):
            return False
        # 第三层：文件名 fallback（覆盖内容无算子名但文件名含算子名的场景）
        basename = os.path.splitext(os.path.basename(file_path))[0].lower()
        if api_name.lower() in basename:
            return True
        return False


# ==================== APILocator ====================


class APILocator:
    """API 代码定位器"""

    def __init__(self, repo_root: str = "."):
        self.repo = os.path.abspath(repo_root)
        self.extractor = Extractor(self.repo)

    @staticmethod
    def extract_api_name_from_doc(doc_path: str) -> str:
        """从文档文件名提取 API 名。运算符文档走特殊清洗路径，保留运算符符号。"""
        doc_path = os.path.abspath(doc_path)
        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"Doc not found: {doc_path}")

        basename = os.path.splitext(os.path.basename(doc_path))[0]

        # 运算符文档：保留运算符符号，不从文件名去括号
        if basename.lower().startswith("operator") or basename == "运算符重载":
            return _extract_operator_name_from_doc_title(doc_path)

        # 非运算符文档：原有清洗逻辑
        basename = re.sub(r"[（(].*?[）)]", "", basename)
        basename = re.sub(r"-.*侧接口$", "", basename)
        basename = re.sub(r"接口$", "", basename)
        basename = re.sub(r"-\d+$", "", basename)
        return basename.strip()

    def determine_api_type(self, api_name: str) -> dict:
        """搜索 include/ 确定 API 类型和声明文件路径。运算符按类别分流处理。"""
        category = _classify_operator(api_name)

        if category == "symbol_override":
            return self._determine_operator_type(api_name)
        elif category in ("subscript", "element_access", "assignment"):
            return self._determine_special_operator_type(api_name, category)

        pattern = rf"\b{re.escape(api_name)}\s*[<(]"
        lines = _search_in_files(self.repo, [pattern], "include")
        if not lines:
            raise ValueError(f"API '{api_name}' not found in include/")
        best_match = self._pick_best_match(lines)
        if not best_match:
            raise ValueError(f"API '{api_name}' not found in include/ (no inline match)")

        file_path = best_match.split(":")[0]
        api_type = self._match_path_to_type(file_path)

        return {
            "api_type": api_type,
            "decl_file": file_path,
        }

    def _determine_operator_type(self, api_name: str) -> dict:
        """处理运算符重载的声明定位。"""
        op_symbol = api_name[len("operator"):]
        if not op_symbol:
            raise ValueError(f"Invalid operator name: {api_name}")

        op_pattern = rf"operator\s*{re.escape(op_symbol)}\s*\("
        matches = _search_in_files(self.repo, [op_pattern], "include")
        if not matches:
            raise ValueError(f"Operator '{api_name}' not found in include/basic_api/kernel_tensor.h")

        file_path = None
        for m in matches:
            if "kernel_tensor.h" in m:
                file_path = m.split(":")[0]
                break
        if file_path is None:
            file_path = matches[0].split(":")[0]

        return {
            "api_type": "basic_api",
            "decl_file": file_path,
        }

    def _determine_special_operator_type(self, api_name: str, category: str) -> dict:
        """处理 operator[]、operator()、operator= 等特殊运算符的声明定位。

        这类运算符不能走 grep 'ApiName(' 的通用搜索，
        因为它们的声明形式不同：
          operator[]  →  LocalTensor<T> operator[](const uint32_t offset) const;
          operator()  →  PrimType& operator()(const uint32_t offset) const;
          operator=   →  LocalTensor<T> operator=(const LocalTensor<T>& other);

        直接根据 OPERATOR_CATEGORIES 中的已知文件路径定位声明文件。
        """
        category_info = OPERATOR_CATEGORIES[category]

        # 特殊运算符的声明文件路径是已知的固定路径，直接定位
        parent_classes = category_info.get("parent_classes", {})
        if not parent_classes:
            raise ValueError(f"Operator '{api_name}' has no known declaration files")

        # 默认取第一个 parent_class 的声明文件
        first_class = next(iter(parent_classes.values()))
        decl_file = first_class["decl_file"]

        # 验证声明文件存在
        full_path = os.path.join(self.repo, decl_file)
        if not os.path.exists(full_path):
            raise ValueError(f"Declaration file not found: {full_path}")

        # 用专用正则在声明文件中搜索运算符声明
        op_patterns = {
            "subscript": r"operator\s*\[\s*\]",
            "element_access": r"operator\s*\(\s*\)",
            "assignment": r"operator\s*=\s*(?!=)\(",  # 排除 operator==
        }
        pattern = op_patterns.get(category)
        if pattern:
            found = False
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if re.search(pattern, line):
                        found = True
                        break
            if not found:
                raise ValueError(f"Operator '{api_name}' declaration not found in {decl_file}")

        # 推断 API 类型
        api_type = self._match_path_to_type(decl_file)

        return {
            "api_type": api_type,
            "decl_file": decl_file,
            "operator_category": category,
        }

    @staticmethod
    def _pick_best_match(lines: list) -> str:
        """从多行匹配中选出最佳结果，优先 basic_api 的 kernel_operator 文件。"""
        inline_lines = [ln for ln in lines if "inline" in ln or "__aicore__" in ln or "__SIMT_DEVICE" in ln]
        if not inline_lines:
            return lines[0] if lines else None

        for ln in inline_lines:
            if "include/basic_api/kernel_operator" in ln:
                return ln

        return inline_lines[0]

    @staticmethod
    def _match_path_to_type(file_path: str) -> str:
        if file_path.startswith("include/basic_api/"):
            return "basic_api"
        if file_path.startswith("include/adv_api/"):
            return "adv_api"
        if file_path.startswith("include/simt_api/"):
            return "simt_api"
        if file_path.startswith("include/tensor_api/"):
            return "tensor_api"
        if file_path.startswith("include/c_api/"):
            return "c_api"
        if file_path.startswith("include/aicpu_api/"):
            return "aicpu_api"
        if file_path.startswith("include/utils/"):
            return "utils"
        raise ValueError(f"Unknown API type from path: {file_path}")

    def find_impl_files(self, api_type: str, decl_file: str, api_name: str, operator_category: str = None) -> dict:
        """根据类型和声明文件定位所有 impl 文件。"""
        # 特殊运算符：直接使用 OPERATOR_CATEGORIES 中的已知 impl 路径
        if operator_category and operator_category in OPERATOR_CATEGORIES:
            category_info = OPERATOR_CATEGORIES[operator_category]
            parent_classes = category_info.get("parent_classes", {})
            impl_files = []
            for class_info in parent_classes.values():
                impl_path = class_info.get("impl_file")
                if impl_path:
                    full_impl = os.path.join(self.repo, impl_path)
                    if os.path.exists(full_impl):
                        impl_files.append(impl_path)
            if impl_files:
                chip_dirs = self._extract_chip_dirs(impl_files)
                total_chips = len(glob.glob(os.path.join(self.repo, IMPL_BASIC_DAV_GLOB)))
                has_chip_specific = len(chip_dirs) < total_chips and len(chip_dirs) > 0
                return {
                    "impl_files": impl_files,
                    "chip_dirs": chip_dirs,
                    "has_chip_specific": has_chip_specific,
                    "total_chips": total_chips,
                }

        impl_api_name = OPERATOR_TO_API.get(api_name, api_name)
        if api_type == "aicpu_api":
            return {"impl_files": [], "chip_dirs": [], "has_chip_specific": False, "total_chips": 0}

        # #11 修复：adv_api 不依赖目录镜像假设，用宽泛递归 glob + 部分路径匹配 + 内容验证
        # 参考 api_change_detect 的 _batch_find_all_impl_files() adv_api 分支。
        if api_type == "adv_api":
            return self._find_adv_api_impl_files(impl_api_name)

        patterns = self._build_impl_patterns(api_type, decl_file, impl_api_name)

        found_impls = []
        for pattern in patterns:
            full_pattern = os.path.join(self.repo, pattern)
            # 含 ** 的 pattern 需要递归搜索
            is_recursive = "**" in pattern
            impl_files = sorted(glob.glob(full_pattern, recursive=is_recursive))
            impl_files = [_rp(f, self.repo) for f in impl_files]
            for impl in impl_files:
                if impl not in found_impls and self.extractor.api_in_file(impl, impl_api_name, api_type):
                    found_impls.append(impl)

        if api_type == "basic_api" and not found_impls:
            for pattern in patterns:
                full_pattern = os.path.join(self.repo, pattern)
                impl_files = sorted(glob.glob(full_pattern))
                impl_files = [_rp(f, self.repo) for f in impl_files]
                for impl in impl_files:
                    if impl not in found_impls and self.extractor._api_in_file_raw(impl, impl_api_name):
                        found_impls.append(impl)

        # basic_api 同 dav 目录关联：已找到核心 impl 时，
        # 把同 dav 目录下所有含 API 名的文件也纳入
        if api_type == "basic_api" and found_impls:
            found_dav_dirs = set()
            for f in found_impls:
                for part in f.split("/"):
                    if part.startswith("dav_"):
                        found_dav_dirs.add(part)

            for dav_dir in sorted(found_dav_dirs):
                same_dir_pat = os.path.join(self.repo, "impl/basic_api", dav_dir, "*_impl.h")
                same_dir_files = sorted(glob.glob(same_dir_pat))
                same_dir_files = [_rp(f, self.repo) for f in same_dir_files]
                for impl in same_dir_files:
                    if impl not in found_impls and self.extractor._api_in_file_raw(impl, impl_api_name):
                        found_impls.append(impl)

        chip_dirs = self._extract_chip_dirs(found_impls)
        total_chips = len(glob.glob(os.path.join(self.repo, IMPL_BASIC_DAV_GLOB)))
        has_chip_specific = len(chip_dirs) < total_chips and len(chip_dirs) > 0

        return {
            "impl_files": found_impls,
            "chip_dirs": chip_dirs,
            "has_chip_specific": has_chip_specific,
            "total_chips": total_chips,
        }

    def _find_adv_api_impl_files(self, impl_api_name: str) -> dict:
        """adv_api impl 搜索：宽泛递归 glob + 分层路径匹配 + 内容验证。

        不依赖 include→impl 目录镜像假设。
        参考 api_change_detect 的 _batch_find_all_impl_files() adv_api 分支。

        搜索分层：
          第一层：核心 impl — 路径含 API 名 + 内容验证
          第二层：api_check 文件 — 路径含 API 名 + /api_check/ 或 /kernel_check/
          第三层：伴随文件 — 路径含 API 名 + 文件名含 _common/_check/_utils 等
          第四层：tiling 文件 — 路径含 API 名 + /tiling/ 子目录
          第五轮（兜底）：路径中包含 API 名（忽略大小写）的任意位置
        """
        # 一次性 glob 所有 adv_api impl 文件（含 api_check 和 tiling）
        detail_all_h = glob.glob(os.path.join(self.repo, "impl/adv_api/detail", "**", "*.h"), recursive=True)
        detail_all_cpp = glob.glob(os.path.join(self.repo, "impl/adv_api/detail", "**", "*.cpp"), recursive=True)
        tiling_all_h = glob.glob(os.path.join(self.repo, "impl/adv_api/tiling", "**", "*.h"), recursive=True)
        tiling_all_cpp = glob.glob(os.path.join(self.repo, "impl/adv_api/tiling", "**", "*.cpp"), recursive=True)
        all_adv_files = detail_all_h + detail_all_cpp + tiling_all_h + tiling_all_cpp

        # 批量读取文件内容到内存缓存
        adv_content_cache = {}
        for f in all_adv_files:
            try:
                with open(f, "rb") as fh:
                    adv_content_cache[f] = fh.read()
            except (FileNotFoundError, PermissionError):
                continue

        found_impls = []
        impl_name_lower = impl_api_name.lower()

        core_files = []
        check_files = []
        companion_files = []
        tiling_files = []

        for f in all_adv_files:
            if f not in adv_content_cache:
                continue
            rel = _rp(f, self.repo)

            # 路径必须含 /{API名小写}/ 才算候选
            if f"/{impl_name_lower}/" not in rel:
                continue

            # 第一层：内容验证
            if self.extractor._api_in_file_raw(rel, impl_api_name):
                if rel not in found_impls:
                    core_files.append(rel)
                continue

            # 第二层：api_check / kernel_check 文件
            if "/api_check/" in rel or "/kernel_check/" in rel:
                check_files.append(rel)
                continue

            # 第三层：伴随文件
            basename = os.path.basename(rel).lower()
            is_companion = any(
                basename.endswith(s) or basename.endswith(s + ".h") or basename.endswith(s + ".cpp")
                for s in [
                    "_common_impl",
                    "_common",
                    "_check",
                    "_check_common",
                    "_utils",
                    "_base_impl",
                    "_base",
                    "_func",
                    "_sub_func",
                    "_sub_api",
                    "_config",
                    "_tiling_impl",
                    "_tiling",
                ]
            )
            if is_companion:
                companion_files.append(rel)
                continue

            # 第四层：tiling 文件
            if "/tiling/" in rel:
                tiling_files.append(rel)

        # 合并四层结果
        all_valid = core_files + check_files + companion_files + tiling_files

        # 第五轮兜底：路径匹配无结果时，更广泛搜索
        if not all_valid:
            for f in all_adv_files:
                if f not in adv_content_cache:
                    continue
                rel = _rp(f, self.repo)
                if impl_name_lower in rel.lower():
                    if self.extractor._api_in_file_raw(rel, impl_api_name):
                        if rel not in all_valid:
                            all_valid.append(rel)

        # 去重
        found_impls = sorted(set(all_valid))

        chip_dirs = self._extract_chip_dirs(found_impls)
        total_chips = len(glob.glob(os.path.join(self.repo, IMPL_BASIC_DAV_GLOB)))
        has_chip_specific = len(chip_dirs) < total_chips and len(chip_dirs) > 0

        return {
            "impl_files": found_impls,
            "chip_dirs": chip_dirs,
            "has_chip_specific": has_chip_specific,
            "total_chips": total_chips,
        }

    def _build_impl_patterns(self, api_type: str, decl_file: str, api_name: str) -> list:
        """根据类型构建 impl 搜索路径列表。"""
        patterns = []

        if api_type == "basic_api":
            base = os.path.basename(decl_file)
            base_name = base.replace("_intf.h", "")
            patterns.append(f"impl/basic_api/dav_*/{base_name}_impl.h")
            # impl/basic_api 根目录一层：部分接口的实现不在 dav_ 架构子目录
            # （实测 PR 6620：SetLoopModePara/ResetLoopModePara 的实现在
            #   impl/basic_api/kernel_operator_data_copy_intf_impl.h，
            #   OUT_TO_L1 变化载体，漏抓导致变更检测误判「无变化」）
            patterns.append("impl/basic_api/*_impl.h")

            if "/reg_compute/" in decl_file or "/reg_base/" in decl_file:
                patterns.append("impl/basic_api/dav_*/kernel_operator_vec_binary_impl.h")
                patterns.append("impl/basic_api/dav_*/kernel_operator_vec_unary_impl.h")
                patterns.append("impl/basic_api/dav_*/kernel_operator_*.h")

            patterns.append("impl/basic_api/dav_*/*_impl.h")

        elif api_type == "simt_api":
            # 优化：递归 glob 所有 .h（不只是特定命名模式）
            patterns.append("impl/simt_api/**/*.h")

        elif api_type == "tensor_api":
            # 优化：递归 glob 所有 .h/.cpp（不只是镜像路径推导）
            patterns.append("impl/tensor_api/**/*.h")
            patterns.append("impl/tensor_api/**/*.cpp")

        elif api_type == "c_api":
            # 覆盖 instr_impl/reg_base_impl/memory_base_impl 等全部子目录
            # (实测 PR 6512:cache_ctrl 的实现在 reg_base_impl/ 下,仅
            #  instr_impl 模式会漏抓 impl,导致变更检测误判「无代码变更」)
            patterns.append("impl/c_api/**/*.h")
            patterns.append("impl/c_api/**/*.cpp")
            patterns.append("impl/c_api/**/*.cc")

        # adv_api 已在 find_impl_files() 中走 _find_adv_api_impl_files() 特殊分流，
        # 不再使用镜像路径模式，故此处无需返回 patterns。

        elif api_type == "utils":
            rel = _rp(decl_file, "include/utils/")
            subdir = os.path.dirname(rel)
            fname = os.path.splitext(os.path.basename(rel))[0]
            patterns.append(f"impl/utils/{subdir}/{fname}/*.h")

        return patterns

    @staticmethod
    def _extract_chip_dirs(impl_files: list) -> list:
        """从 impl 文件路径中提取芯片目录名。"""
        chip_dirs = set()
        for f in impl_files:
            parts = f.split("/")
            for p in parts:
                if p.startswith("dav_") or p.startswith("npu_arch_"):
                    chip_dirs.add(p)
        return sorted(chip_dirs)

    def find_test_files(self, api_type: str, api_name: str) -> list:
        """根据 API 类型和名称在 tests/api/ 中定位测试用例。

        #17 修复：搜索类型专属目录 + 公共目录（common/reg_compute_api）；
        若类型不在 TEST_DIR_MAP 中，fallback 到整个 tests/api/ 广度搜索。
        """
        found = []

        # 1. 搜索类型专属目录
        test_subdir = TEST_DIR_MAP.get(api_type)
        if test_subdir:
            full_test_path = os.path.join(self.repo, test_subdir)
            if os.path.isdir(full_test_path):
                pattern = os.path.join(full_test_path, "**", "*.cpp")
                all_cpp = sorted(glob.glob(pattern, recursive=True))
                all_cpp = [_rp(f, self.repo) for f in all_cpp]
                for cpp in all_cpp:
                    if cpp not in found and self.extractor.api_in_test_file(cpp, api_name):
                        found.append(cpp)

        # 2. #17 修复：搜索公共测试目录（common、reg_compute_api）
        for common_dir in COMMON_TEST_DIRS:
            full_path = os.path.join(self.repo, common_dir)
            if os.path.isdir(full_path):
                pattern = os.path.join(full_path, "**", "*.cpp")
                all_cpp = sorted(glob.glob(pattern, recursive=True))
                all_cpp = [_rp(f, self.repo) for f in all_cpp]
                for cpp in all_cpp:
                    if cpp not in found and self.extractor.api_in_test_file(cpp, api_name):
                        found.append(cpp)

        # 3. fallback: 若类型不在 TEST_DIR_MAP 且无结果，广度搜索整个 tests/api/
        if not test_subdir and not found:
            tests_root = os.path.join(self.repo, "tests/api")
            if os.path.isdir(tests_root):
                pattern = os.path.join(tests_root, "**", "*.cpp")
                all_cpp = sorted(glob.glob(pattern, recursive=True))
                all_cpp = [_rp(f, self.repo) for f in all_cpp]
                for cpp in all_cpp:
                    if cpp not in found and self.extractor.api_in_test_file(cpp, api_name):
                        found.append(cpp)

        return found

    @staticmethod
    def _evidence_to_sources(evidence: list) -> list:
        """Evidence 列表 → flat source 列表（供 formatter 使用）。"""
        sources = []
        for ev in evidence:
            source = {
                "dimension": ev.dimension,
                "type": ev.metadata.get("type", "unknown"),
                "file": ev.file,
                "line": ev.line,
                "content": ev.content,
                "snippet": ev.content,
            }
            source.update(ev.metadata)
            sources.append(source)
        return sources

    def _build_sections(
        self,
        api_name: str,
        api_type: str,
        decl_file: str,
        impl_result: dict,
        test_files: list,
        read_base: str = None,
        doc_chapters_json_path: str = None,
        selected_chapters: list = None,
    ) -> dict:
        """按文档章节组织所有代码来源。委托给 chapters/ 模块。

        Args:
            read_base: 读取代码的基准路径。为 None 时用 self.repo，
                       传入 export_root 时从重组文件夹读取。
            doc_chapters_json_path: 文档解析后的 JSON 路径。如果提供，只提取
                                    JSON 中存在的章节的证据，避免生成空章节。
            selected_chapters: 用户指定的章节名称列表。如果提供，只提取
                               列表中指定的章节，覆盖 doc_chapters_json_path 的过滤。
        """
        sections = {}
        impl_files = impl_result.get("impl_files", [])
        extractor = Extractor(read_base) if read_base else self.extractor

        doc_chapters = set()
        if doc_chapters_json_path and os.path.exists(doc_chapters_json_path):
            with open(doc_chapters_json_path, "r", encoding="utf-8") as f:
                doc_chapters = set(json.load(f).keys())

        from chapters import (
            extract_ch1,
            extract_ch2,
            extract_ch3,
            extract_ch4,
            extract_ch5,
            extract_ch6,
            extract_ch7,
            extract_ch8,
        )
        from chapters.ch1_chip_support import CHAPTER_NAME as CH1_NAME
        from chapters.ch2_function_desc import CHAPTER_NAME as CH2_NAME
        from chapters.ch3_impl_principle import CHAPTER_NAME as CH3_NAME
        from chapters.ch4_function_proto import CHAPTER_NAME as CH4_NAME
        from chapters.ch5_param_desc import CHAPTER_NAME as CH5_NAME
        from chapters.ch6_return_type import CHAPTER_NAME as CH6_NAME
        from chapters.ch7_constraints import CHAPTER_NAME as CH7_NAME
        from chapters.ch8_examples import CHAPTER_NAME as CH8_NAME

        chapter_extractors = {
            CH1_NAME: extract_ch1,
            CH2_NAME: extract_ch2,
            CH3_NAME: extract_ch3,
            CH4_NAME: extract_ch4,
            CH5_NAME: extract_ch5,
            CH6_NAME: extract_ch6,
            CH7_NAME: extract_ch7,
            CH8_NAME: extract_ch8,
        }

        for chapter in CHAPTERS:
            if chapter == CH3_NAME and api_type != "adv_api":
                continue
            if selected_chapters and chapter not in selected_chapters:
                continue
            if doc_chapters and chapter not in doc_chapters:
                continue
            ch_extract = chapter_extractors.get(chapter)
            if ch_extract is None:
                continue
            evidence = ch_extract(api_name, api_type, decl_file, impl_files, test_files, extractor, self.repo)
            evidence = deduplicate_by_content(evidence)
            sections[chapter] = {"sources": self._evidence_to_sources(evidence)}

        return sections
