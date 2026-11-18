# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Directory tree builder for api_locator tree_id module."""

import os


def build_folder_tree(root_dir: str) -> str:
    """Build a directory tree string similar to the `tree` command output.

    Recursively traverses all files and subdirectories under root_dir,
    skipping __pycache__/ directories and .pyc files.

    Args:
        root_dir: Absolute or relative path to the root directory.

    Returns:
        A multi-line string representing the directory tree.
        Root node uses os.path.basename(root_dir), not the absolute path.
    """
    root_name = os.path.basename(os.path.normpath(root_dir))
    lines = [root_name + "/"]
    _build_tree(root_dir, "", lines)
    return "\n".join(lines)


def _build_tree(dir_path: str, prefix: str, lines: list[str]) -> None:
    """Recursively append tree lines for the contents of dir_path."""
    try:
        entries = sorted(os.listdir(dir_path))
    except PermissionError:
        return

    # Filter out __pycache__ directories and .pyc files
    filtered = [e for e in entries if e != "__pycache__" and not e.endswith(".pyc")]

    for i, entry in enumerate(filtered):
        is_last = i == len(filtered) - 1
        connector = "└── " if is_last else "├── "
        full_path = os.path.join(dir_path, entry)

        if os.path.isdir(full_path):
            lines.append(prefix + connector + entry + "/")
            child_prefix = prefix + ("    " if is_last else "│   ")
            _build_tree(full_path, child_prefix, lines)
        else:
            lines.append(prefix + connector + entry)
