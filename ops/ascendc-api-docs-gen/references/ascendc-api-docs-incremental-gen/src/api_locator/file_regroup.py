# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""文件重组：将 API 相关文件按原仓路径拷贝到 {API名}/ 目录。

使用方式：
    python file_regroup.py docs/zh/api/**/Add.md
"""

import argparse
import glob
import os
import re
import shutil
import sys

# 跨平台文件搜索（替代 grep 子进程）
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from common.file_search import search_in_files


def _np(path: str) -> str:
    """路径分隔符归一化：统一使用正斜杠 /，兼容 Windows/Linux/macOS。"""
    return path.replace(os.sep, "/")


def _rp(full_path, base_path):
    """跨平台相对路径。"""
    return _np(os.path.relpath(full_path, base_path))


def _name_matches_api(name: str, api_lower: str) -> bool:
    """判断目录/文件名是否与 API 名匹配。

    匹配规则（任一成立即匹配）：
    1. 精确相等：name == api_lower
    2. name 以 api_lower 为前缀，剩余部分以非字母数字开头（如 asc_abs_v2 匹配 asc_abs）
    3. api_lower 以 name 为前缀，剩余部分以非字母数字开头（如 asc_abs 匹配 asc）
    4. name 是 api_lower 的下划线/连字符分隔段之一（如 abs 匹配 asc_abs，add 匹配 asc_add）
    """
    if not name or not api_lower:
        return False
    if name == api_lower:
        return True
    if name.startswith(api_lower):
        rest = name[len(api_lower):]
        if rest and not rest[0].isalnum():
            return True
    if api_lower.startswith(name):
        rest = api_lower[len(name):]
        if rest and not rest[0].isalnum():
            return True
    # 分隔符分段匹配：asc_abs → ['asc', 'abs']，name='abs' 命中
    if re.search(r"[_\-]", api_lower):
        segments = re.split(r"[_\-]+", api_lower)
        if name in segments:
            return True
    return False


# 同目录模块
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
import config
from evidence import APILocator


class APIExporter:
    """把 API 相关的所有文件导出到以 {API名} 命名的目录。"""

    def __init__(self, repo: str):
        self.repo = repo
        self.locator = APILocator(repo)

    def export(self, api_name: str, output_dir: str, api_type: str = None, doc_path: str = None) -> str:
        """导出指定 API 的所有相关文件。

        Args:
            api_name: API 名称，如 "Add"
            output_dir: 输出根目录（绝对路径）
            api_type: API 类型，不传则自动识别
            doc_path: 接口文档路径，不传则按默认路径查找

        Returns:
            导出目录的绝对路径
        """
        if api_type is None:
            info = self.locator.determine_api_type(api_name)
            api_type = info["api_type"]
            decl_file = info["decl_file"]
        else:
            pattern = rf"\b{re.escape(api_name)}\s*[(]"
            lines = search_in_files(self.repo, [pattern], f"include/{api_type}")
            if lines:
                decl_file = lines[0].split(":")[0]
            else:
                raise ValueError(f"API '{api_name}' not found in include/{api_type}/")

        impl_result = self.locator.find_impl_files(api_type, decl_file, api_name)
        test_files = self.locator.find_test_files(api_type, api_name)

        file_paths = set()
        file_paths.add(decl_file)
        for impl in impl_result.get("impl_files", []):
            file_paths.add(impl)
        for test in test_files:
            file_paths.add(test)

        self._collect_extra_files(api_name, api_type, file_paths)
        self._collect_docs(doc_path, api_name, file_paths)

        os.makedirs(output_dir, exist_ok=True)
        regroup_dir = os.path.join(output_dir, f"{api_name}{config.OUTPUT_REGROUP_SUFFIX}")
        export_root = os.path.join(regroup_dir, config.OUTPUT_REFACTORED_DIR)
        if os.path.exists(regroup_dir):
            shutil.rmtree(regroup_dir)
        os.makedirs(export_root)

        for rel_path in sorted(file_paths):
            src = os.path.join(self.repo, rel_path)
            dst = os.path.join(export_root, rel_path)
            if not os.path.exists(src):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

        return export_root

    def _collect_extra_files(self, api_name: str, api_type: str, file_paths: set):
        """收集额外的关联文件：kernel_check、api_check、examples 等。"""
        for pattern in [config.IMPL_CHECK_GLOB]:
            for f in glob.glob(os.path.join(self.repo, pattern)):
                rel = _rp(f, self.repo)
                if self._api_in_file(rel, api_name, api_type):
                    file_paths.add(rel)

        if api_type == "adv_api":
            check_patterns = [
                "impl/adv_api/detail/api_check/**/*.h",
                "impl/adv_api/detail/**/*check*.h",
            ]
            for pat in check_patterns:
                for f in glob.glob(os.path.join(self.repo, pat), recursive=True):
                    rel = _rp(f, self.repo)
                    if self._api_in_file(rel, api_name, api_type):
                        file_paths.add(rel)

        examples_root = os.path.join(self.repo, config.OUTPUT_EXAMPLES_DIR)
        if os.path.isdir(examples_root):
            for root, dirs, files in os.walk(examples_root):
                # 排序确保遍历顺序跨平台一致
                dirs.sort()
                rel_dir = _rp(root, examples_root)
                dir_name = os.path.basename(root).lower()
                api_lower = api_name.lower()

                if _name_matches_api(dir_name, api_lower):
                    for fname in files:
                        rel = _rp(os.path.join(root, fname), self.repo)
                        file_paths.add(rel)
                else:
                    for fname in files:
                        name_no_ext = os.path.splitext(fname)[0].lower()
                        if _name_matches_api(name_no_ext, api_lower):
                            rel = _rp(os.path.join(root, fname), self.repo)
                            file_paths.add(rel)

    def _collect_docs(self, doc_path, api_name: str, file_paths: set):
        """收集接口文档。"""
        if doc_path:
            if os.path.isabs(doc_path):
                rel_doc = _rp(doc_path, self.repo)
            else:
                rel_doc = doc_path
            if os.path.exists(os.path.join(self.repo, rel_doc)):
                file_paths.add(rel_doc)
                self._collect_linked_samples(rel_doc, file_paths)
        else:
            main_doc = config.resolve_doc_path(self.repo, api_name)
            if main_doc and os.path.exists(os.path.join(self.repo, main_doc)):
                file_paths.add(main_doc)
                self._collect_linked_samples(main_doc, file_paths)

    def _collect_linked_samples(self, doc_path: str, file_paths: set):
        """递归收集文档中链接的更多样例文件。"""
        full_path = os.path.join(self.repo, doc_path)
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except (FileNotFoundError, PermissionError):
            return

        links = re.findall(r"\[.*?\]\((更多样例[^)]+\.md)\)", content)
        for link in links:
            sample_path = os.path.join(os.path.dirname(doc_path), link).replace(os.sep, "/")
            if os.path.exists(os.path.join(self.repo, sample_path)):
                file_paths.add(sample_path)
                self._collect_linked_samples(sample_path, file_paths)

    def _api_in_file(self, file_path: str, api_name: str, api_type: str) -> bool:
        """检查文件中是否包含目标 API 名。"""
        search_name = f"{api_name}Impl" if api_type == "basic_api" else api_name
        pattern = re.compile(rf"\b{re.escape(search_name)}\b")
        full_path = os.path.join(self.repo, file_path)
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                return bool(pattern.search(content))
        except (FileNotFoundError, PermissionError):
            return False


def export_regroup(doc_path: str, output_dir: str, api_type: str = None, repo_root: str = None) -> str:
    """完整流程：文档路径 → 提取 API 名 → 定位文件 → 打包重组。

    Args:
        doc_path: 接口文档路径（相对路径基于 repo_root，或绝对路径）
        output_dir: 输出根目录（绝对路径）
        api_type: API 类型，不传则自动识别
        repo_root: 仓库根目录，必须传入

    Returns:
        导出目录的绝对路径
    """
    if repo_root is None:
        raise ValueError("repo_root 必须指定")

    if not os.path.isabs(doc_path):
        doc_path = os.path.join(repo_root, doc_path)

    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"文档不存在: {doc_path}")

    api_name = APILocator.extract_api_name_from_doc(doc_path)

    exporter = APIExporter(repo=repo_root)
    export_root = exporter.export(api_name, output_dir=output_dir, api_type=api_type, doc_path=doc_path)

    return export_root


def main():
    parser = argparse.ArgumentParser(description="文件重组：给定接口文档路径，定位相关文件并按原仓路径拷贝到输出目录")
    parser.add_argument("doc", help="接口文档路径（相对路径基于 --repo，或绝对路径）")
    parser.add_argument("--repo", required=True, help="代码仓库根目录")
    parser.add_argument("--output", default="data/out", help="输出目录（相对路径基于运行目录，默认: data/out）")
    args = parser.parse_args()

    try:
        output_dir = os.path.abspath(args.output)
        path = export_regroup(args.doc, output_dir=output_dir, repo_root=args.repo)
        count = sum(len(f) for _, _, f in os.walk(path))
        print(f"打包完成: {path}")
        print(f"共 {count} 个文件")
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
