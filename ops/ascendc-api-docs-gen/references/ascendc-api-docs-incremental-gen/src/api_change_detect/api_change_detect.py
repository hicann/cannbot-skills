#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
API 变更检测 — 独立工具

功能：
1. 构建反向索引：docs/zh/api/**/*.md → 所有关联文件
2. git diff 后查索引，输出受影响的 API

用法：
    python build_index.py              # 构建索引
    python detect.py                   # 默认比较 HEAD vs 上次提交
    python detect.py HEAD~3            # 比较 HEAD~3 vs HEAD
    python detect.py abc123 def456     # 比较两个 commit
"""

import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

# 跨平台文件搜索（替代 grep 子进程）
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from common.file_search import search_in_files

try:
    from config import DEFAULT_PATH_CONFIG, REPO_ROOT, PathConfig
except ImportError:
    REPO_ROOT = None
    DEFAULT_PATH_CONFIG = None
    PathConfig = None

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console(stderr=True)


def _np(path: str) -> str:
    """路径分隔符归一化：统一使用正斜杠 /，兼容 Windows/Linux/macOS。"""
    return path.replace(os.sep, "/")


def _rp(full_path, base_path):
    """跨平台相对路径：等价于 _np(os.path.relpath(full_path, base_path))。"""
    return _np(os.path.relpath(full_path, base_path))


def _get_default_path_config():
    """获取默认路径配置。"""
    if DEFAULT_PATH_CONFIG is not None:
        return DEFAULT_PATH_CONFIG
    # 回退：如果 config 模块不可用，使用硬编码默认值
    return PathConfig() if PathConfig else None


def _resolve_doc_dir(repo_root, path_config):
    """解析文档目录，支持自动回退。"""
    doc_dir = os.path.join(repo_root, path_config.doc_dir)
    if os.path.isdir(doc_dir):
        return doc_dir
    for fallback in ("docs/zh/api", "docs/api", "docs/en/api"):
        candidate = os.path.join(repo_root, fallback)
        if os.path.isdir(candidate):
            return candidate
    found = glob.glob(os.path.join(repo_root, "docs", "*", "api"))
    if found and os.path.isdir(found[0]):
        return found[0]
    return doc_dir


# ==================== 常量 ====================

API_NAME_PATTERN = re.compile(r"[一-龥a-zA-Z0-9_]+(?:接口|[-（(].*)?")

# 以下常量保留用于向后兼容，新代码应使用 PathConfig
# 文档目录以 asc-devkit 现结构为准：docs/zh/api/<API 族>/<分类>/<接口>_interface/<接口>.md
_DEFAULT_DOC_DIR = "docs/zh/api"
_DEFAULT_INCLUDE_BASIC = "include/basic_api"
_DEFAULT_IMPL_BASIC = "impl/basic_api"
_DEFAULT_IMPL_ADV = "impl/adv_api"
_DEFAULT_IMPL_SIMT = "impl/simt_api"
_DEFAULT_IMPL_TENSOR = "impl/tensor_api"
_DEFAULT_IMPL_CAPI = "impl/c_api"
_DEFAULT_IMPL_UTILS = "impl/utils"
_DEFAULT_TEST_DIR = "tests/api"
_DEFAULT_EXAMPLES_DIR = "examples"

# 运算符 → API 名映射
OPERATOR_TO_API = {
    "operator+": "Add",
    "operator-": "Sub",
    "operator*": "Mul",
    "operator/": "Div",
    "operator|": "Or",
    "operator&": "And",
    "operator<": "Less",
    "operator>": "Greater",
    "operator==": "Equal",
    "operator!=": "NotEqual",
    "operator<=": "LessEqual",
    "operator>=": "GreaterEqual",
}

# 运算符类型分类体系
OPERATOR_CATEGORIES = {
    "symbol_override": {
        "operators": [
            "operator+",
            "operator-",
            "operator*",
            "operator/",
            "operator|",
            "operator&",
            "operator<",
            "operator>",
            "operator==",
            "operator!=",
            "operator<=",
            "operator>=",
        ],
        "mapping": OPERATOR_TO_API,
    },
    "subscript": {
        "operators": ["operator[]"],
        "parent_classes": {
            "LocalTensor": {
                "decl_file": "include/basic_api/kernel_tensor.h",
                "impl_file": "impl/basic_api/kernel_tensor_impl.h",
            },
            "GlobalTensor": {
                "decl_file": "include/basic_api/kernel_tensor.h",
                "impl_file": "impl/basic_api/kernel_tensor_impl.h",
            },
        },
    },
    "element_access": {
        "operators": ["operator()"],
        "parent_classes": {
            "LocalTensor": {
                "decl_file": "include/basic_api/kernel_tensor.h",
                "impl_file": "impl/basic_api/kernel_tensor_impl.h",
            },
            "GlobalTensor": {
                "decl_file": "include/basic_api/kernel_tensor.h",
                "impl_file": "impl/basic_api/kernel_tensor_impl.h",
            },
        },
    },
    "assignment": {
        "operators": ["operator="],
        "parent_classes": {
            "LocalTensor": {
                "decl_file": "include/basic_api/kernel_tensor.h",
                "impl_file": "impl/basic_api/kernel_tensor_impl.h",
            },
            "OpMC2Def": {"decl_file": "include/utils/op_flag.h"},
            "OpAICoreConfig": {"decl_file": "include/utils/op_flag.h"},
        },
    },
}


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
                    break
    except (FileNotFoundError, PermissionError):
        pass

    basename = os.path.splitext(os.path.basename(doc_path))[0]
    basename = re.sub(r"-\d+$", "", basename)
    return basename.strip()


# ==================== 从文档名提取 API 名 ====================


def extract_api_name_from_doc(doc_path):
    """docs/zh/api/**/Add.md → Add；运算符文档保留符号。"""
    basename = os.path.splitext(os.path.basename(doc_path))[0]

    # 运算符文档：保留运算符符号，不从文件名去括号
    if basename.lower().startswith("operator") or basename == "运算符重载":
        return _extract_operator_name_from_doc_title(doc_path)

    # 非运算符文档：原有清洗逻辑
    basename = re.sub(r"[（(].*?[）)]", "", basename)
    basename = re.sub(r"-\d+$", "", basename)
    basename = re.sub(r"-.*侧接口$", "", basename)
    basename = re.sub(r"接口$", "", basename)
    return basename.strip()


# ==================== 文件发现逻辑（单 API，保留兼容） ====================


def _match_path_to_type(file_path, path_config=None):
    """从 include/ 路径推断 API 类型。"""
    if path_config is None:
        path_config = _get_default_path_config()
    for inc_dir in path_config.include_dirs:
        if file_path.startswith(inc_dir + "/"):
            return inc_dir.split("/")[-1]  # e.g. "include/basic_api" → "basic_api"
    return None


def _pick_best_match(lines):
    """从 grep 多行匹配中选出最佳声明文件，优先 kernel_operator + inline。"""
    inline_lines = [ln for ln in lines if "inline" in ln or "__aicore__" in ln or "__SIMT_DEVICE" in ln]
    if not inline_lines:
        return lines[0] if lines else None
    for ln in inline_lines:
        if "include/basic_api/kernel_operator" in ln:
            return ln
    return inline_lines[0]


def find_decl_files(api_name, repo_root):
    """找到 API 的声明头文件。"""
    decl_files = []

    # 搜索 include/ 下所有头文件中的 API 声明
    # #6 修复：匹配模板声明 ApiName<tparam>(args)，不只是函数调用形式 ApiName(args)
    pattern = rf"\b{re.escape(api_name)}\s*[<(]"
    lines = search_in_files(repo_root, [pattern], "include")
    for line in lines:
        file_path = line.split(":")[0]
        if file_path and file_path.startswith("include/"):
            decl_files.append(file_path)

    # 去重
    return sorted(set(decl_files))


def find_impl_files_for_api(api_name, decl_files, repo_root):
    """根据声明文件路径和 API 名找到实现文件。"""
    impl_files = []

    # 基本模式：从声明文件名推断 impl 文件名
    for decl in decl_files:
        base = os.path.basename(decl)
        base_name = base.replace("_intf.h", "")
        # 精确匹配: dav_*/{base_name}_impl.h
        pat = os.path.join(IMPL_BASIC, "dav_*", f"{base_name}_impl.h")
        impl_files.extend(glob.glob(os.path.join(repo_root, pat)))
        # reg_compute / reg_base 等也声明了 basic_api，需要用 kernel_operator 系列 impl
        if "/reg_compute/" in decl or "/reg_base/" in decl or "kernel_operator" in base:
            pat2 = os.path.join(IMPL_BASIC, "dav_*", "kernel_operator_*_impl.h")
            impl_files.extend(glob.glob(os.path.join(repo_root, pat2)))

    # adv_api: impl/adv_api/detail/
    pat_adv = os.path.join(IMPL_ADV, "detail", "**", f"*{api_name}*impl*.h")
    impl_files.extend(glob.glob(os.path.join(repo_root, pat_adv), recursive=True))

    # simt_api
    pat_simt = os.path.join(IMPL_SIMT, "**", f"*{api_name}*impl*.h")
    impl_files.extend(glob.glob(os.path.join(repo_root, pat_simt), recursive=True))

    # tensor_api
    pat_tensor = os.path.join(IMPL_TENSOR, "**", f"*{api_name}*impl*.h")
    impl_files.extend(glob.glob(os.path.join(repo_root, pat_tensor), recursive=True))

    # c_api
    pat_c = os.path.join(IMPL_CAPI, "instr_impl", "npu_arch_*", f"*{api_name}*")
    impl_files.extend(glob.glob(os.path.join(repo_root, pat_c)))

    # utils
    pat_utils = os.path.join(IMPL_UTILS, "**", f"*{api_name}*.h")
    impl_files.extend(glob.glob(os.path.join(repo_root, pat_utils), recursive=True))

    # 验证：impl 文件中确实包含 API 名（尝试 Impl 后缀 + 原名）
    valid = []
    search_names = [f"{api_name}Impl", api_name]
    for impl in impl_files:
        rel = os.path.relpath(impl, repo_root)
        if _api_in_file(impl, search_names):
            valid.append(rel)

    return sorted(set(valid))


def find_test_files_for_api(api_name, repo_root):
    """在 tests/api/ 下搜索调用 API 的测试文件。"""
    test_files = []
    pattern_bytes = rf"\b{re.escape(api_name)}\s*[<(]".encode()

    tests_path = os.path.join(repo_root, TEST_DIR)
    if not os.path.isdir(tests_path):
        return []

    for root, dirs, files in os.walk(tests_path):
        dirs.sort()
        for f in sorted(files):
            if f.endswith((".cpp", ".cc", ".py")):
                full_path = os.path.join(root, f)
                try:
                    with open(full_path, "rb") as fh:
                        for line in fh:
                            if re.search(pattern_bytes, line):
                                test_files.append(os.path.relpath(full_path, repo_root))
                                break
                except (FileNotFoundError, PermissionError):
                    continue

    return sorted(set(test_files))


def find_check_files_for_api(api_name, repo_root):
    """找到包含 API 的 kernel_check 文件。"""
    check_files = []
    pat = os.path.join(repo_root, IMPL_BASIC, "utils", "kernel_check_*.h")
    for f in glob.glob(pat):
        if _api_in_file(f, [api_name]):
            check_files.append(os.path.relpath(f, repo_root))
    return sorted(set(check_files))


def find_example_files_for_api(api_name, repo_root):
    """找到包含 API 的示例文件。"""
    example_files = []
    examples_path = os.path.join(repo_root, EXAMPLES_DIR)
    if not os.path.isdir(examples_path):
        return []

    api_lower = api_name.lower()
    for root, dirs, files in os.walk(examples_path):
        dirs.sort()
        dir_name = os.path.basename(root).lower()
        if dir_name == api_lower or dir_name.startswith(api_lower + "_") or dir_name.startswith(api_lower + "-"):
            for f in sorted(files):
                rel = os.path.relpath(os.path.join(root, f), repo_root)
                example_files.append(rel)
        else:
            for f in sorted(files):
                name_no_ext = os.path.splitext(f)[0].lower()
                if name_no_ext == api_lower or name_no_ext.startswith(api_lower + "_"):
                    rel = os.path.relpath(os.path.join(root, f), repo_root)
                    example_files.append(rel)

    return sorted(set(example_files))


def find_linked_docs(doc_path, repo_root):
    """递归找到文档中链接的样例文档。"""
    linked = set()
    try:
        with open(doc_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except (FileNotFoundError, PermissionError):
        return linked

    doc_dir = os.path.dirname(doc_path)
    links = re.findall(r"\[.*?\]\(([^)]+\.md)\)", content)
    for link in links:
        if "样例" in link or "示例" in link or "Sample" in link:
            full_link = os.path.normpath(os.path.join(doc_dir, link))
            if os.path.exists(full_link):
                rel = os.path.relpath(full_link, repo_root)
                linked.add(rel)
                linked.update(find_linked_docs(full_link, repo_root))

    return linked


def _api_in_file(file_path, names):
    """二进制模式搜索文件中是否包含任一名（大小写不敏感）。

    使用 re.IGNORECASE，适配 AscendC 仓库中 API 名大写（Gelu）
    但文件内容小写（geglu）的命名惯例。
    """
    for name in names:
        pattern = re.compile(rf"\b{re.escape(name)}\b".encode(), re.IGNORECASE)
        try:
            with open(file_path, "rb") as f:
                for line in f:
                    if pattern.search(line):
                        return True
        except (FileNotFoundError, PermissionError):
            continue
    return False


def _batch_read_files(file_paths):
    """批量读取文件内容到内存，返回 {abs_path: bytes_content} 缓存。

    一次性读取所有文件，避免逐文件 open/read/close 的 I/O 开销。
    读取失败的文件跳过（不放入缓存）。
    """
    cache = {}
    for f in file_paths:
        try:
            with open(f, "rb") as fh:
                cache[f] = fh.read()
        except (FileNotFoundError, PermissionError):
            continue
    return cache


def _api_in_content(content_bytes, names):
    """在已读取的 bytes 内容中搜索 API 名（大小写不敏感）。

    与 _api_in_file 功能相同，但输入是内存中的 bytes 而非文件路径，
    用于配合 _batch_read_files 批量缓存后的高速匹配。
    """
    for name in names:
        pattern = re.compile(rf"\b{re.escape(name)}\b".encode(), re.IGNORECASE)
        if pattern.search(content_bytes):
            return True
    return False


# ==================== 批量搜索函数（性能优化） ====================


def _batch_find_all_declarations(repo_root, api_names, path_config=None):
    """一次 grep 扫描所有 API 的声明文件。

    Returns:
        {api_name: [decl_file_paths]}  —— 每个 API 的声明文件列表（相对路径）
        {api_name: api_type}  —— 每个 API 的类型
    """
    if path_config is None:
        path_config = _get_default_path_config()

    decl_map = {}  # api_name → [rel_path]
    type_map = {}  # api_name → api_type

    # 分离特殊运算符（已知声明文件路径）和普通 API
    special_ops = {}  # category → [api_name]
    regular_api_names = []
    for name in api_names:
        cat = _classify_operator(name)
        if cat in ("subscript", "element_access", "assignment"):
            special_ops.setdefault(cat, []).append(name)
        else:
            regular_api_names.append(name)

    # 特殊运算符：直接使用已知声明文件
    for cat, names in special_ops.items():
        category_info = OPERATOR_CATEGORIES[cat]
        parent_classes = category_info.get("parent_classes", {})
        if not parent_classes:
            continue
        first_class = next(iter(parent_classes.values()))
        decl_file = first_class["decl_file"]
        api_type = _match_path_to_type(decl_file, path_config)
        for name in names:
            decl_map[name] = [decl_file]
            if api_type:
                type_map[name] = api_type

    # 普通 API：走 Python 原生搜索
    if not regular_api_names:
        return decl_map, type_map

    # 构建 include 搜索范围
    include_root = (
        os.path.commonpath([d.split("/")[0] + "/" for d in path_config.include_dirs])
        if path_config.include_dirs
        else "include"
    )
    if len(include_root) <= 1:
        include_root = "include"

    # 使用宽松模式匹配：API 名后面跟 ( 或 <（模板）
    all_patterns = "|".join(rf"\b{re.escape(name)}\s*[<(]" for name in regular_api_names)
    lines = search_in_files(repo_root, [all_patterns], include_root.rstrip("/"))
    if not lines:
        return decl_map, type_map

    # 解析输出：按 API 名分组
    api_patterns = {name: re.compile(rf"\b{re.escape(name)}\s*[<(]") for name in regular_api_names}

    grouped = {}  # api_name → [match_line]
    for line in lines:
        if not line.strip():
            continue
        file_path = _np(line.split(":")[0])
        if not any(file_path.startswith(d + "/") for d in path_config.include_dirs):
            continue
        for name, pat in api_patterns.items():
            # 搜索 grep-line 的内容部分（第3段）
            content = line.split(":", 2)[-1] if ":" in line else line
            if pat.search(content):
                grouped.setdefault(name, []).append(line)

    for name, lines in grouped.items():
        best = _pick_best_match(lines)
        if best:
            best_file = best.split(":")[0]
            decl_map[name] = [best_file]
            api_type = _match_path_to_type(best_file, path_config)
            if api_type:
                type_map[name] = api_type

    return decl_map, type_map


def _batch_find_all_impl_files(repo_root, api_names, decl_map, type_map, path_config=None):
    """按 API 类型各 glob 一次，在内存中匹配。

    Returns:
        {api_name: [impl_file_rel_paths]}
    """
    if path_config is None:
        path_config = _get_default_path_config()

    impl_result = {}

    # 特殊运算符 impl：已知固定路径
    for cat in ("subscript", "element_access", "assignment"):
        cat_names = [n for n in api_names if _classify_operator(n) == cat]
        if not cat_names:
            continue
        category_info = OPERATOR_CATEGORIES[cat]
        parent_classes = category_info.get("parent_classes", {})
        for name in cat_names:
            impls = []
            for class_info in parent_classes.values():
                impl_path = class_info.get("impl_file")
                if impl_path:
                    full_impl = os.path.join(repo_root, impl_path)
                    if os.path.exists(full_impl):
                        impls.append(_rp(full_impl, repo_root))
            if impls:
                impl_result[name] = sorted(set(impls))

    # 按类型分组 API
    by_type = {}  # api_type → [(api_name, decl_files)]
    for name in api_names:
        if name in impl_result:
            continue  # 已通过特殊运算符路径处理
        atype = type_map.get(name)
        if not atype:
            continue
        decls = decl_map.get(name, [])
        by_type.setdefault(atype, []).append((name, decls))

    # --- basic_api: 按 decl 文件名精确推导 + 同 dav 目录关联 ---
    if "basic_api" in by_type:
        # 一次 glob 所有 basic_api impl 文件，批量读取到内存
        all_basic_impls = glob.glob(os.path.join(repo_root, path_config.impl_basic, "**", "*.h"), recursive=True)
        basic_content_cache = _batch_read_files(all_basic_impls)

        for name, decls in by_type["basic_api"]:
            files = []
            for decl in decls:
                base_name = os.path.basename(decl).replace("_intf.h", "")
                # 精确匹配: dav_*/{base_name}_impl.h
                pat = os.path.join(repo_root, path_config.impl_basic, "dav_*", f"{base_name}_impl.h")
                candidates = glob.glob(pat)
                if "/reg_compute" in decl or "/reg_base" in decl or "kernel_operator" in decl:
                    pat2 = os.path.join(repo_root, path_config.impl_basic, "dav_*", "kernel_operator_*_impl.h")
                    candidates.extend(glob.glob(pat2))

            # 内容验证：核心 impl 文件
            valid = [
                _rp(f, repo_root)
                for f in candidates
                if f in basic_content_cache and _api_in_content(basic_content_cache[f], [f"{name}Impl"])
            ]

            # 同 dav 目录关联：如果找到了核心 impl 文件，
            # 把同 dav 目录下所有内容含 API 名的文件也纳入
            if valid:
                found_dav_dirs = set()
                for f in candidates:
                    if f in basic_content_cache:
                        # 提取 dav_xxx 目录名
                        for part in _np(f).split("/"):
                            if part.startswith("dav_"):
                                found_dav_dirs.add(part)

                for dav_dir in sorted(found_dav_dirs):
                    same_dir_pat = os.path.join(repo_root, path_config.impl_basic, dav_dir, "*_impl.h")
                    same_dir_files = glob.glob(same_dir_pat)
                    for f in same_dir_files:
                        if f in basic_content_cache:
                            rel = _rp(f, repo_root)
                            if rel not in valid and _api_in_content(basic_content_cache[f], [name, f"{name}Impl"]):
                                valid.append(rel)

            if valid:
                impl_result[name] = sorted(set(valid))

    # --- adv_api ---
    # 分层搜索策略：
    #   第一层：内容验证（精确匹配） — 路径含 API 名 + 文件内容含 API 名
    #   第二层：伴随文件 — 同目录下的 *_common*、*_check*、*_utils* 等配套文件
    #   第二层扩展：api_check 目录下的 check 文件（路径含 API 名子目录）
    #   第三层：tiling 文件 — 按路径中的 API 名归属
    #   上限：每个 API 的 impl 文件总数不超过 _adv_impl_limit
    _adv_impl_limit = 30

    if "adv_api" in by_type:
        # 扩展 glob：搜 detail/ 下所有 .h/.cpp（不限文件名模式），以及 tiling/ 下的文件
        detail_all_h = glob.glob(os.path.join(repo_root, path_config.impl_adv, "detail", "**", "*.h"), recursive=True)
        detail_all_cpp = glob.glob(
            os.path.join(repo_root, path_config.impl_adv, "detail", "**", "*.cpp"), recursive=True
        )
        tiling_all_h = glob.glob(os.path.join(repo_root, path_config.impl_adv, "tiling", "**", "*.h"), recursive=True)
        tiling_all_cpp = glob.glob(
            os.path.join(repo_root, path_config.impl_adv, "tiling", "**", "*.cpp"), recursive=True
        )
        all_adv_files = detail_all_h + detail_all_cpp + tiling_all_h + tiling_all_cpp

        # 批量读取所有 adv_api impl 文件到内存缓存
        adv_content_cache = _batch_read_files(all_adv_files)

        for name, _ in by_type["adv_api"]:
            name_lower = name.lower()
            core_files = []  # 第一层：内容验证通过的
            companion_files = []  # 第二层：伴随文件
            check_files = []  # 第二层扩展：api_check 文件
            tiling_files = []  # 第三层：tiling 文件

            for f in all_adv_files:
                if f not in adv_content_cache:
                    continue
                rel = _rp(f, repo_root)
                # 路径必须含 /{API名小写}/ 才算候选（排除 HCCL 等无关目录）
                if f"/{name_lower}/" not in rel:
                    continue

                # 第一层：内容验证
                if _api_in_content(adv_content_cache[f], [name]):
                    core_files.append(rel)
                    continue

                # 第二层扩展：api_check 目录下的 check 文件
                # 这些文件路径含 /api_check/kernel_check/{类别}/{api_lower}/
                # 文件名含 _check 或 _check_common 或 _aicore 等模式
                if "/api_check/" in rel or "/kernel_check/" in rel:
                    check_files.append(rel)
                    continue

                # 第二层：伴随文件 — 同目录下的配套文件
                # 模式：文件名含 _common、_check、_utils、_base、_func、_sub、_config 等
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

                # 第三层：tiling 目录下的文件（按路径归属）
                if "/tiling/" in rel:
                    tiling_files.append(rel)

            # 合并四层结果，按优先级排序，限制总数
            all_valid = core_files + check_files + companion_files + tiling_files
            if all_valid:
                impl_result[name] = sorted(set(all_valid[:_adv_impl_limit]))

    # --- simt_api ---
    # 优化：递归 glob 所有 .h（不只是 *impl*.h），批量读取做内容验证。
    # 搜索策略：路径含 API 名子目录 或 路径含 kernel_simt_{api_lower}，
    # 同时搜 api_name 和 api_name + "Impl"（basic_api 风格后缀）。
    if "simt_api" in by_type:
        all_impls = glob.glob(os.path.join(repo_root, path_config.impl_simt, "**", "*.h"), recursive=True)

        # 批量读取
        content_cache = _batch_read_files(all_impls)

        for name, _ in by_type["simt_api"]:
            name_lower = name.lower()
            search_names = [name, f"{name}Impl"]
            valid = []
            for f in all_impls:
                if f not in content_cache:
                    continue
                rel = _rp(f, repo_root)
                # 路径筛选：含 /{api_lower}/ 子目录 或含 kernel_simt_{api_lower} 文件名
                if (
                    f"/{name_lower}/" in rel.lower()
                    or f"kernel_simt_{name_lower}" in os.path.basename(f).lower()
                    or f"/{name_lower}" in rel.lower()
                ):
                    if _api_in_content(content_cache[f], search_names):
                        valid.append(rel)
            # 兜底：路径筛选无结果时，对所有文件做内容验证
            if not valid:
                for f in all_impls:
                    if f not in content_cache:
                        continue
                    if _api_in_content(content_cache[f], search_names):
                        valid.append(_rp(f, repo_root))
            if valid:
                impl_result[name] = sorted(set(valid))

    # --- tensor_api ---
    # 优化：递归 glob 所有 .h/.cpp（不只是 *impl*.h），批量读取做内容验证。
    # 搜索策略：路径含 API 名子目录（大小写不敏感），
    # 同时搜 api_name 和 api_name + "Impl"。
    if "tensor_api" in by_type:
        all_impls_h = glob.glob(os.path.join(repo_root, path_config.impl_tensor, "**", "*.h"), recursive=True)
        all_impls_cpp = glob.glob(os.path.join(repo_root, path_config.impl_tensor, "**", "*.cpp"), recursive=True)
        all_impls = all_impls_h + all_impls_cpp

        # 批量读取
        content_cache = _batch_read_files(all_impls)

        for name, _ in by_type["tensor_api"]:
            name_lower = name.lower()
            search_names = [name, f"{name}Impl"]
            valid = []
            for f in all_impls:
                if f not in content_cache:
                    continue
                rel = _rp(f, repo_root)
                # 路径筛选：路径含 /{api_lower}/ 子目录
                if f"/{name_lower}/" in rel.lower():
                    if _api_in_content(content_cache[f], search_names):
                        valid.append(rel)
            # 兜底：路径筛选无结果时，对所有文件做内容验证
            if not valid:
                for f in all_impls:
                    if f not in content_cache:
                        continue
                    if _api_in_content(content_cache[f], search_names):
                        valid.append(_rp(f, repo_root))
            if valid:
                impl_result[name] = sorted(set(valid))

    # --- c_api ---
    # 优化：递归 glob instr_impl/ 下所有 .h/.cpp（而非扁平一层 npu_arch_*/*），
    # 批量读取文件内容做高速正则匹配，并从文件名逆推 API 名做宽松匹配。
    if "c_api" in by_type:
        all_impls_h = glob.glob(
            os.path.join(repo_root, path_config.impl_capi, "instr_impl", "**", "*.h"), recursive=True
        )
        all_impls_cpp = glob.glob(
            os.path.join(repo_root, path_config.impl_capi, "instr_impl", "**", "*.cpp"), recursive=True
        )
        all_impls_cc = glob.glob(
            os.path.join(repo_root, path_config.impl_capi, "instr_impl", "**", "*.cc"), recursive=True
        )
        all_impls = all_impls_h + all_impls_cpp + all_impls_cc

        # 批量读取文件内容到内存缓存
        content_cache = _batch_read_files(all_impls)

        # 构建 c_api 文件名 → 可能的 API 名映射
        # 文件名模式：asc_set_atomic_add_float_impl.h → asc_set_atomic_add_float
        # 同时保留原名搜索（asc_xxx 直接在 include/c_api 中声明）
        c_api_name_patterns = {}
        for name, _ in by_type["c_api"]:
            # c_api 的 API 名如 asc_set_atomic_add_float，
            # impl 文件名如 asc_set_atomic_add_float_impl.h
            # 搜索策略：同时搜 api_name 和 api_name + "_impl"
            search_names = [name]
            if not name.endswith("_impl"):
                search_names.append(f"{name}_impl")
            # 宽松匹配：去掉 \b 后缀约束，允许 _impl 紧跟
            c_api_name_patterns[name] = [re.compile(rf"\b{re.escape(n)}".encode(), re.IGNORECASE) for n in search_names]

        for name, _ in by_type["c_api"]:
            valid = []
            for f in all_impls:
                if f not in content_cache:
                    continue
                content = content_cache[f]
                for pat in c_api_name_patterns[name]:
                    if pat.search(content):
                        valid.append(_rp(f, repo_root))
                        break
            if valid:
                impl_result[name] = sorted(set(valid))

    # --- utils ---
    if "utils" in by_type and path_config.impl_utils:
        pat = os.path.join(repo_root, path_config.impl_utils, "**", "*.h")
        all_impls = glob.glob(pat, recursive=True)
        for name, _ in by_type["utils"]:
            valid = []
            for f in all_impls:
                if _api_in_file(f, [name]):
                    valid.append(_rp(f, repo_root))
            if valid:
                impl_result[name] = sorted(set(valid))

    return impl_result


def _batch_find_all_test_files(repo_root, api_names, type_map, path_config=None):
    """按 API 类型各 walk 一次子目录，编译正则一次扫描。

    #17 修复：额外搜索公共测试目录（common、reg_compute_api）。

    Returns:
        {api_name: [test_file_rel_paths]}
    """
    if path_config is None:
        path_config = _get_default_path_config()

    test_result = {}

    ext_pattern = tuple(path_config.test_extensions)

    # 按类型分组 API
    by_type = {}
    for name in api_names:
        atype = type_map.get(name)
        if not atype or atype not in path_config.test_dir_map:
            continue
        by_type.setdefault(atype, []).append(name)

    # #17 修复：公共测试目录（适用于所有 API 类型）
    common_test_dirs = [
        "tests/api/common",
        "tests/api/reg_compute_api",
    ]

    for atype, names in by_type.items():
        test_subdir = path_config.test_dir_map[atype]
        tests_path = os.path.join(repo_root, test_subdir)
        if not os.path.isdir(tests_path):
            continue

        all_test_files = []
        for root, dirs, files in os.walk(tests_path):
            dirs.sort()
            for f in sorted(files):
                if f.endswith(ext_pattern):
                    all_test_files.append(os.path.join(root, f))

        api_pats = {name: re.compile(rf"\b{re.escape(name)}\s*[<(]".encode()) for name in names}

        for tf in all_test_files:
            try:
                with open(tf, "rb") as fh:
                    content = fh.read()
            except (FileNotFoundError, PermissionError):
                continue
            for name, pat in api_pats.items():
                if pat.search(content):
                    test_result.setdefault(name, []).append(_rp(tf, repo_root))

    # #17 修复：搜索公共测试目录
    for common_dir in common_test_dirs:
        common_path = os.path.join(repo_root, common_dir)
        if not os.path.isdir(common_path):
            continue
        common_files = []
        for root, dirs, files in os.walk(common_path):
            dirs.sort()
            for f in sorted(files):
                if f.endswith(ext_pattern):
                    common_files.append(os.path.join(root, f))
        api_pats_all = {name: re.compile(rf"\b{re.escape(name)}\s*[<(]".encode()) for name in api_names}
        for tf in common_files:
            try:
                with open(tf, "rb") as fh:
                    content = fh.read()
            except (FileNotFoundError, PermissionError):
                continue
            for name, pat in api_pats_all.items():
                if pat.search(content):
                    test_result.setdefault(name, []).append(_rp(tf, repo_root))

    for name in test_result:
        test_result[name] = sorted(set(test_result[name]))

    return test_result


def _batch_find_all_example_files(repo_root, api_names, path_config=None):
    """一次 walk examples/ 目录，按前缀+分隔符匹配。

    Returns:
        {api_name: [example_file_rel_paths]}
    """
    if path_config is None:
        path_config = _get_default_path_config()

    examples_path = os.path.join(repo_root, path_config.examples_dir)
    if not os.path.isdir(examples_path):
        return {}

    api_set = set(n.lower() for n in api_names)
    api_name_map = {n.lower(): n for n in api_names}

    # 缓存目录名到匹配的 API 名
    dir_api_cache = {}

    example_result = {}
    for root, dirs, files in os.walk(examples_path):
        dirs.sort()
        dir_name_lower = os.path.basename(root).lower()

        matched_api = dir_api_cache.get(dir_name_lower)
        if matched_api is None:
            matched_api = []
            for api_lower in api_set:
                api_orig = api_name_map[api_lower]
                # 检查目录是否匹配
                if dir_name_lower == api_lower:
                    matched_api.append(api_orig)
                elif dir_name_lower.startswith(api_lower):
                    rest = dir_name_lower[len(api_lower):]
                    if rest and not rest[0].isalnum():
                        matched_api.append(api_orig)
            dir_api_cache[dir_name_lower] = matched_api

        for fname in sorted(files):
            fname_lower = os.path.splitext(fname)[0].lower()
            for api_orig in matched_api:
                api_lower = api_orig.lower()
                rel = _rp(os.path.join(root, fname), repo_root)
                # 目录已匹配，加入所有文件
                if dir_name_lower == api_lower or (
                    dir_name_lower.startswith(api_lower)
                    and dir_name_lower[len(api_lower):]
                    and not dir_name_lower[len(api_lower):][0].isalnum()
                ):
                    example_result.setdefault(api_orig, []).append(rel)
                else:
                    # 检查文件名
                    if fname_lower == api_lower:
                        example_result.setdefault(api_orig, []).append(rel)
                    elif fname_lower.startswith(api_lower):
                        rest = fname_lower[len(api_lower):]
                        if rest and (rest[0] == "_" or not rest[0].isalnum()):
                            example_result.setdefault(api_orig, []).append(rel)

    for name in example_result:
        example_result[name] = sorted(set(example_result[name]))

    return example_result


def _batch_find_all_check_files(repo_root, api_names, path_config=None):
    """一次 glob kernel_check 文件，批量扫描。

    Returns:
        {api_name: [check_file_rel_paths]}
    """
    if path_config is None:
        path_config = _get_default_path_config()

    pat = os.path.join(repo_root, path_config.impl_basic, "utils", "kernel_check_*.h")
    check_files = glob.glob(pat)
    if not check_files:
        return {}

    api_pats = {name: re.compile(rf"\b{re.escape(name)}\b".encode()) for name in api_names}

    check_result = {}
    for cf in check_files:
        try:
            with open(cf, "rb") as f:
                content = f.read()
        except (FileNotFoundError, PermissionError):
            continue
        for name, pat in api_pats.items():
            if pat.search(content):
                check_result.setdefault(name, []).append(_rp(cf, repo_root))

    return check_result


# ==================== 指纹计算 ====================


def get_commit_id(repo_root):
    """获取代码仓库的当前 git commit ID 作为指纹。

    Returns:
        commit ID 字符串（如 "abc123def456..."），非 git 仓库返回 "unknown"
    """
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return "unknown"


# ==================== 增量更新 ====================


def _diff_dirs(old_dir, new_dir, path_config=None):
    """对比两个目录，返回 (changed_files, deleted_files)。"""
    if path_config is None:
        path_config = _get_default_path_config()

    exclude_dirs = tuple(path_config.walk_exclude_dirs)
    exclude_dir_suffixes = tuple(path_config.walk_exclude_dir_suffixes)
    exclude_file_suffixes = tuple(path_config.changed_file_exclude_suffixes)
    file_prefixes = tuple(path_config.changed_file_prefixes)

    def _should_exclude_dir(d):
        return d in exclude_dirs or (exclude_dir_suffixes and d.endswith(exclude_dir_suffixes))

    def _should_include_file(rel_path):
        return rel_path.startswith(file_prefixes) if file_prefixes else True

    def _scan_dir(d, result):
        for root, dirs, files in os.walk(d):
            dirs[:] = sorted(d for d in dirs if not _should_exclude_dir(d))
            for f in sorted(files):
                if f.endswith(exclude_file_suffixes):
                    continue
                full = os.path.join(root, f)
                rel = _rp(full, d)
                if not _should_include_file(rel):
                    continue
                try:
                    with open(full, "rb") as fh:
                        result[rel] = hashlib.md5(fh.read(), usedforsecurity=False).hexdigest()
                except (FileNotFoundError, PermissionError):
                    continue

    old_files = {}
    new_files = {}
    _scan_dir(old_dir, old_files)
    _scan_dir(new_dir, new_files)

    changed = set()
    deleted = set()
    for rel, h in new_files.items():
        if rel not in old_files or old_files[rel] != h:
            changed.add(rel)
    for rel in old_files:
        if rel not in new_files:
            deleted.add(rel)

    return changed, deleted


def incremental_update_index(old_code, new_code, old_index, old_api_file_map, path_config=None):
    """增量更新索引。"""
    if path_config is None:
        path_config = _get_default_path_config()

    t0 = time.time()
    console.print("[bold yellow]⏳ 正在增量更新索引...[/bold yellow]\n")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan][1/5] 对比目录变更...", total=None)

        # ---- 1. 目录 diff ----
        changed_files, deleted_files = _diff_dirs(old_code, new_code, path_config)

        # ---- 2. 反查受影响的 API ----
        progress.update(
            task, description=f"[cyan][2/5] 反查受影响 API (变更 {len(changed_files)}, 删除 {len(deleted_files)})..."
        )
        affected_apis = set()

        # 2.1 变更文件反查
        for f in changed_files:
            if f in old_index:
                affected_apis.update(old_index[f])

        # 2.2 删除文件反查（文件重命名场景：旧文件删除，API迁移到新文件）
        for f in deleted_files:
            if f in old_index:
                affected_apis.update(old_index[f])

        # 对不在索引中的变更文件，走 pattern 匹配
        orphan_changed = [f for f in changed_files if f not in old_index]
        pattern_affected = find_affected_by_pattern(orphan_changed, old_api_file_map)
        for api in pattern_affected:
            affected_apis.add(api)

        # ---- 3. 在 new_code 上重算 affected_apis ----
        progress.update(task, description=f"[cyan][3/5] 重算 {len(affected_apis)} 个 API: 搜索声明文件...")
        decl_map, type_map = _batch_find_all_declarations(new_code, affected_apis, path_config)

        progress.update(task, description=f"[cyan][3/5] 重算 {len(affected_apis)} 个 API: 搜索 impl 文件...")
        impl_result = _batch_find_all_impl_files(new_code, affected_apis, decl_map, type_map, path_config)

        progress.update(task, description=f"[cyan][3/5] 重算 {len(affected_apis)} 个 API: 搜索测试文件...")
        test_result = _batch_find_all_test_files(new_code, affected_apis, type_map, path_config)

        progress.update(task, description=f"[cyan][3/5] 重算 {len(affected_apis)} 个 API: 搜索示例文件...")
        example_result = _batch_find_all_example_files(new_code, affected_apis, path_config)

        progress.update(task, description=f"[cyan][3/5] 重算 {len(affected_apis)} 个 API: 搜索 check 文件...")
        check_result = _batch_find_all_check_files(new_code, affected_apis, path_config)

        # 构建 affected_apis 的新结果
        new_api_files = {}  # api → set(files)
        for api in affected_apis:
            files = set()
            for f in decl_map.get(api, []):
                files.add(f)
            for f in impl_result.get(api, []):
                files.add(f)
            for f in test_result.get(api, []):
                files.add(f)
            for f in check_result.get(api, []):
                files.add(f)
            for f in example_result.get(api, []):
                files.add(f)
            # 保留文档路径（从旧索引中提取，避免丢失）
            for f in old_api_file_map.get(api, []):
                if f.startswith("docs/") and f.endswith(".md"):
                    files.add(f)
            new_api_files[api] = files

        # ---- 4. 从旧索引中移除 affected_apis 旧数据 + 删除的文件 ----
        progress.update(task, description="[cyan][4/5] 更新索引...")
        new_index = {f: set(apis) for f, apis in old_index.items()}
        new_api_map = {api: set(files) for api, files in old_api_file_map.items()}

        # 移除 affected_apis 的旧 entry
        for api in affected_apis:
            if api in new_api_map:
                for f in new_api_map[api]:
                    if f in new_index:
                        new_index[f].discard(api)
                        if not new_index[f]:
                            del new_index[f]

        # 移除 deleted_files 的 entry
        for f in deleted_files:
            if f in new_index:
                for api in list(new_index[f]):
                    if api in new_api_map:
                        new_api_map[api].discard(f)
                del new_index[f]

        # ---- 5. 写入新数据 ----
        for api, files in new_api_files.items():
            new_api_map[api] = files
            for f in files:
                if f not in new_index:
                    new_index[f] = set()
                new_index[f].add(api)

        # ---- 6. 处理新增/删除的文档 ----
        progress.update(task, description="[cyan][5/5] 处理文档变更...")
        doc_ext = path_config.doc_extensions[0] if path_config.doc_extensions else ".md"
        old_doc_dir = _resolve_doc_dir(old_code, path_config)
        new_doc_dir = _resolve_doc_dir(new_code, path_config)
        import fnmatch

        exclude_patterns = path_config.doc_exclude_patterns

        def _list_docs(d, base):
            """列出文档，返回相对路径集合（相对于 base）。"""
            if not os.path.isdir(d):
                return set()
            if path_config.doc_recursive:
                all_docs = set(glob.glob(os.path.join(d, "**", "*" + doc_ext), recursive=True))
            else:
                all_docs = set(glob.glob(os.path.join(d, "*" + doc_ext)))
            # ★ 修复：转换为相对路径，避免绝对路径差集错误
            return {
                _rp(doc, base)
                for doc in all_docs
                if not any(fnmatch.fnmatch(os.path.basename(doc), pat) for pat in exclude_patterns)
            }

        old_docs = _list_docs(old_doc_dir, old_code)
        new_docs = _list_docs(new_doc_dir, new_code)

        # 新增文档 → 新 API
        added_doc_paths = new_docs - old_docs
        for rel_doc in added_doc_paths:
            # ★ 相对路径转绝对路径（extract_api_name_from_doc需要绝对路径）
            doc_abs = os.path.join(new_code, rel_doc)
            api_name = extract_api_name_from_doc(doc_abs)
            category = _classify_operator(api_name)
            if category == "symbol_override":
                canonical = OPERATOR_TO_API.get(api_name, api_name)
            else:
                canonical = api_name
            linked = find_linked_docs(doc_abs, new_code)
            files = set()
            files.add(rel_doc)
            files.update(linked)
            new_api_map[canonical] = files
            for f in files:
                if f not in new_index:
                    new_index[f] = set()
                new_index[f].add(canonical)

        # 删除文档 → 移除对应 API（仅移除纯文档型条目）
        removed_doc_paths = old_docs - new_docs
        removed_apis = set()
        for rel_doc in removed_doc_paths:
            # ★ 相对路径转绝对路径
            doc_abs = os.path.join(old_code, rel_doc)
            api_name = extract_api_name_from_doc(doc_abs)
            category = _classify_operator(api_name)
            if category == "symbol_override":
                canonical = OPERATOR_TO_API.get(api_name, api_name)
            else:
                canonical = api_name
            if canonical in new_api_map and canonical not in affected_apis:
                # ★ 方案B：校验是否是真正的 API（有代码文件，不只是纯文档）
                # 只有纯文档的条目才移除，有代码文件的保留
                has_code_file = any(
                    not f.startswith("docs/")  # 不是纯文档
                    for f in new_api_map[canonical]
                )
                if not has_code_file:  # 只有纯文档的才移除
                    for f in new_api_map[canonical]:
                        if f in new_index:
                            new_index[f].discard(canonical)
                            if not new_index[f]:
                                del new_index[f]
                    del new_api_map[canonical]
                    removed_apis.add(canonical)

    elapsed = time.time() - t0
    summary = {
        "mode": "incremental",
        "changed_files_count": len(changed_files),
        "deleted_files_count": len(deleted_files),
        "affected_apis": sorted(affected_apis),
        "new_apis": sorted(a for a in (set(new_api_map.keys()) - set(old_api_file_map.keys()))),
        "removed_apis": sorted(removed_apis),
        "elapsed": elapsed,
    }
    console.print(f"\n[bold green]✅ 增量更新完成: {elapsed:.1f}s[/bold green]")
    print(f"  增量更新耗时: {elapsed:.1f}s")

    # 序列化索引（set → list）
    new_index_serializable = {f: sorted(apis) for f, apis in sorted(new_index.items())}
    new_index_serializable["__fingerprint__"] = get_commit_id(new_code)

    return new_index_serializable, new_api_map, summary


# ==================== 内容扫描（文件级 API 调用识别）====================


def compile_api_name_pattern(api_names):
    """编译一次性正则：匹配文本中出现的任意已知 API 名（调用/模板形式）。

    按长度降序排列，避免短名抢先把长名前缀吃掉（如 Abs 先于 AbsSub）。
    返回 re.Pattern，可用 .findall(text) 得到匹配到的 API 名列表。
    """
    sorted_names = sorted((n for n in api_names if n), key=len, reverse=True)
    if not sorted_names:
        # 不可能匹配任何内容，但仍返回合法 pattern
        return re.compile(r"(?!x)x")
    escaped = [re.escape(n) for n in sorted_names]
    return re.compile(r"\b(" + "|".join(escaped) + r")\s*[(<]")


def extract_apis_from_content(content, compiled_pattern):
    """用预编译的正则从文本里提取出现的 API 名集合。"""
    return set(compiled_pattern.findall(content))


def path_matches_apis(file_rel_path, api_names):
    """从文件相对路径里提取匹配的 API 名（目录名/文件名前缀匹配）。

    与 _batch_find_all_example_files 同款逻辑：大小写不敏感、前缀 + 分隔符。
    用于处理"文件路径名暗示了 API 归属"的情况（如 impl/.../Abs.cpp → Abs）。
    """
    parts = file_rel_path.replace(os.sep, "/").split("/")
    file_stem = os.path.splitext(parts[-1])[0].lower() if parts else ""
    dir_names = [p.lower() for p in parts[:-1]]

    matched = set()
    for api in api_names:
        api_lower = api.lower()
        # 文件名匹配
        if file_stem == api_lower or (
            file_stem.startswith(api_lower)
            and len(file_stem) > len(api_lower)
            and not file_stem[len(api_lower):][0].isalnum()
        ):
            matched.add(api)
            continue
        # 目录名匹配（任一层级）
        for d in dir_names:
            if d == api_lower or (
                d.startswith(api_lower) and len(d) > len(api_lower) and not d[len(api_lower):][0].isalnum()
            ):
                matched.add(api)
                break
    return matched


# ==================== 主流程 ====================


def build_reverse_index(repo_root=None, path_config=None):
    """构建完整的 file → [APIs] 反向索引（批量搜索优化版）。"""
    if repo_root is None:
        repo_root = REPO_ROOT or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    if path_config is None:
        path_config = _get_default_path_config()

    doc_dir = _resolve_doc_dir(repo_root, path_config)
    if not os.path.isdir(doc_dir):
        raise FileNotFoundError(f"文档目录不存在: {doc_dir} (also tried docs/api, docs/en/api)")

    doc_ext = path_config.doc_extensions[0] if path_config.doc_extensions else ".md"
    if path_config.doc_recursive:
        docs = sorted(glob.glob(os.path.join(doc_dir, "**", "*" + doc_ext), recursive=True))
    else:
        docs = sorted(glob.glob(os.path.join(doc_dir, "*" + doc_ext)))

    # 过滤排除的文档
    import fnmatch

    exclude_patterns = path_config.doc_exclude_patterns
    docs = [d for d in docs if not any(fnmatch.fnmatch(os.path.basename(d), pat) for pat in exclude_patterns)]

    docs = sorted(set(docs))

    console.print(f"  > 找到 {len(docs)} 个文档")

    # 从文档提取所有 API 名（原始名 + 规范化名）
    doc_api_pairs = []  # [(doc_path, api_name_raw, canonical_name)]
    all_api_names = set()
    for doc in docs:
        api_name = extract_api_name_from_doc(doc)
        # 运算符按类别分流：特殊运算符保留原名，SymbolOverride 用映射表转换
        category = _classify_operator(api_name)
        if category == "symbol_override":
            canonical_name = OPERATOR_TO_API.get(api_name, api_name)
        else:
            canonical_name = api_name  # operator[]/operator()/operator= 保留原名
        doc_api_pairs.append((doc, api_name, canonical_name))
        all_api_names.add(api_name)

    console.print(f"  > 提取到 {len(all_api_names)} 个 API 名")
    console.print()
    console.print("[bold yellow]⏳ 正在构建中...(首次构建用时较长，请耐心等待)[/bold yellow]")
    console.print()

    # 批量搜索：声明 → impl → 测试 → 示例 → check → 合并
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        search_task = progress.add_task("[cyan][1/6] 批量搜索声明文件...", total=None)
        decl_map, type_map = _batch_find_all_declarations(repo_root, all_api_names, path_config)

        progress.update(search_task, description="[cyan][2/6] 批量搜索 impl 文件...")
        impl_result = _batch_find_all_impl_files(repo_root, all_api_names, decl_map, type_map, path_config)

        progress.update(search_task, description="[cyan][3/6] 批量扫描测试文件...")
        test_result = _batch_find_all_test_files(repo_root, all_api_names, type_map, path_config)

        progress.update(search_task, description="[cyan][4/6] 批量扫描示例文件...")
        example_result = _batch_find_all_example_files(repo_root, all_api_names, path_config)

        progress.update(search_task, description="[cyan][5/6] 批量扫描 check 文件...")
        check_result = _batch_find_all_check_files(repo_root, all_api_names, path_config)

        # 合并结果
        reverse_index = {}  # file → set of API names
        api_file_map = {}  # api_name → set of files

        progress.update(search_task, description="[cyan][6/6] 合并索引...", total=len(doc_api_pairs), completed=0)

        for i, (doc, api_name_raw, canonical_name) in enumerate(doc_api_pairs):
            rel_doc = _rp(doc, repo_root)
            files = set()
            files.add(rel_doc)

            # 关联文档
            linked = find_linked_docs(doc, repo_root)
            files.update(linked)

            # 声明文件
            for f in decl_map.get(api_name_raw, []):
                files.add(f)

            # impl 文件
            for f in impl_result.get(api_name_raw, []):
                files.add(f)

            # 测试文件
            for f in test_result.get(api_name_raw, []):
                files.add(f)

            # check 文件
            for f in check_result.get(api_name_raw, []):
                files.add(f)

            # 示例文件
            for f in example_result.get(api_name_raw, []):
                files.add(f)

            api_file_map[canonical_name] = files

            for f in files:
                if f not in reverse_index:
                    reverse_index[f] = set()
                reverse_index[f].add(canonical_name)

            progress.update(search_task, advance=1)

    console.print()
    console.print(
        "[bold green]✅ 构建完成: {} 个 API, {} 个文件[/bold green]".format(len(api_file_map), len(reverse_index))
    )

    # 序列化
    result = {}
    for f, apis in sorted(reverse_index.items()):
        result[f] = sorted(apis)
    result["__fingerprint__"] = get_commit_id(repo_root)

    return result, api_file_map


def save_index(index, output_dir=None):
    """保存索引到 JSON 文件。"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "api_reverse_index.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"\n索引已保存: {path}")
    print(f"索引大小: {os.path.getsize(path) / 1024:.1f} KB")
    return path


def load_index(index_path=None):
    """加载已保存的索引。"""
    if index_path is None:
        index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "api_reverse_index.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"索引文件不存在: {index_path}，请先运行 python pipeline.py 构建索引")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    # 转换 list → set（跳过 __fingerprint__）
    result = {}
    for k, v in index.items():
        if k == "__fingerprint__":
            result[k] = v
        else:
            result[k] = set(v)
    return result


def get_changed_files(diff_range="HEAD", repo_root=None):
    """获取 git diff 变更的文件列表。"""
    if repo_root is None:
        repo_root = REPO_ROOT or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

    if ".." in diff_range:
        cmd = ["git", "diff", "--name-only", diff_range]
    elif " " in diff_range:
        parts = diff_range.split()
        cmd = ["git", "diff", "--name-only"] + parts
    else:
        cmd = ["git", "diff", "--name-only", diff_range]

    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        print(f"git 命令失败: {' '.join(cmd)}")
        print(result.stderr)
        sys.exit(1)

    return [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]


def find_affected_apis(changed_files, index):
    """查索引找出受影响的 API。"""
    affected = {}  # api_name → [changed_files]

    for changed_file in changed_files:
        if changed_file in index:
            for api in index[changed_file]:
                if api not in affected:
                    affected[api] = []
                affected[api].append(changed_file)

    return affected


def find_affected_by_pattern(changed_files, api_file_map):
    """对于不在索引中的新文件，按文件名模式模糊匹配可能受影响的 API。"""
    affected = {}
    for changed_file in changed_files:
        base = os.path.basename(changed_file).lower()
        base_no_ext = os.path.splitext(base)[0]

        # 文件名以 API 名开头
        for api_name in sorted(api_file_map.keys()):
            api_lower = api_name.lower()
            if (
                base_no_ext == api_lower
                or base_no_ext.startswith(api_lower + "_")
                or base_no_ext.startswith(api_lower + "-")
                or api_lower.startswith(base_no_ext)
            ):
                if api_name not in affected:
                    affected[api_name] = []
                affected[api_name].append(changed_file)

    return affected


def main_detect(diff_range="HEAD", repo_root=None):
    """检测变更影响的 API。"""
    index = load_index()

    # 加载 api_file_map（如果有的话，从索引重建）
    api_file_map = {}
    for f, apis in index.items():
        for api in apis:
            if api not in api_file_map:
                api_file_map[api] = set()
            api_file_map[api].add(f)

    changed_files = get_changed_files(diff_range, repo_root)
    if not changed_files:
        print("没有检测到文件变更")
        return

    print(f"变更文件: {len(changed_files)} 个\n")

    # 精确匹配
    exact_affected = find_affected_apis(changed_files, index)

    # 模糊匹配（新文件可能不在索引中）
    pattern_affected = find_affected_by_pattern(changed_files, api_file_map)

    # 合并
    all_affected = {}
    for api in exact_affected:
        all_affected[api] = list(set(list(all_affected.get(api, [])) + list(exact_affected[api])))
    for api in pattern_affected:
        if api not in all_affected:
            all_affected[api] = list(pattern_affected[api])
        else:
            all_affected[api] = list(set(list(all_affected[api]) + list(pattern_affected[api])))

    if not all_affected:
        print("变更的文件不在索引中，未找到受影响的 API")
        print(f"\n变更文件:")
        for f in changed_files:
            print(f"  {f}")
        return

    # 按 API 名排序输出
    print(f"受影响的 API: {len(all_affected)} 个\n")
    for api_name in sorted(all_affected.keys()):
        files = sorted(all_affected[api_name])
        print(f"  {api_name}")
        for f in files:
            print(f"    ← {f}")

    print(f"\n共 {len(all_affected)} 个 API 受影响")

    return all_affected


# ==================== 入口 ====================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--detect":
        diff_range = sys.argv[2] if len(sys.argv) > 2 else "HEAD"
        main_detect(diff_range)
    elif len(sys.argv) > 1 and sys.argv[1] == "--detect-api":
        # 指定 commit range
        if len(sys.argv) < 3:
            print("用法: python detect.py --detect-api <commit_range>")
            sys.exit(1)
        main_detect(sys.argv[2])
    else:
        print("用法:")
        print("  python pipeline.py --old /old --new /new   # 对比两个目录")
        print("  python pipeline.py --git-range HEAD~3      # 对比两个 commit")
        print("  python pipeline.py --index idx.json        # 使用已有索引")
