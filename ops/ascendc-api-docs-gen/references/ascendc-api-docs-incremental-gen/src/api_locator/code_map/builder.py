# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Code map builder for API locator.

Traverses C++ source files under export_root, parses them with ClangParser,
and generates per-file code_map JSON files plus a functions.json summary.
"""

import json
import logging
import os
from pathlib import Path

import config
from config import CPP_EXTENSIONS

from .clang_parser import ClangParser

logger = logging.getLogger(__name__)


def _safe_name(rel_path: str) -> str:
    """Convert a relative path to a safe filename by replacing separators with underscores.

    Args:
        rel_path: Relative file path (e.g. 'include/test.h').

    Returns:
        Safe name with '/' and '\\' replaced by '_' (e.g. 'include_test.h').
    """
    return rel_path.replace("/", "_").replace("\\", "_")


def build_functions_summary(all_nodes: list, all_edges: list, output_dir: str) -> None:
    """Build functions.json from aggregated nodes and edges.

    Extracts all Function-type nodes and builds calls/called_by/returns mappings
    from the edges.

    Args:
        all_nodes: Aggregated list of node dicts from all parsed files.
        all_edges: Aggregated list of edge dicts from all parsed files.
        output_dir: Directory where functions.json will be written.
    """
    # Collect all Function nodes (deduplicated by id)
    function_ids = set()
    for node in all_nodes:
        if node.get("type") == "Function":
            function_ids.add(node.get("id"))

    # Build edge mappings: calls, called_by, returns
    calls_map = {}
    called_by_map = {}
    returns_map = {}

    for edge in all_edges:
        edge_type = edge.get("type")
        source = edge.get("source")
        target = edge.get("target")

        if edge_type == "calls":
            calls_map.setdefault(source, set()).add(target)
            called_by_map.setdefault(target, set()).add(source)
        elif edge_type == "returns":
            returns_map.setdefault(source, set()).add(target)

    # Build output list
    functions = []
    for func_id in sorted(function_ids):
        functions.append(
            {
                "id": func_id,
                "path": None,
                "calls": sorted(calls_map.get(func_id, set())),
                "called_by": sorted(called_by_map.get(func_id, set())),
                "returns": sorted(returns_map.get(func_id, set())),
            }
        )

    # Fill in path from nodes
    node_path_map = {}
    for node in all_nodes:
        if node.get("type") == "Function" and node.get("path"):
            node_path_map[node["id"]] = node["path"]

    for func in functions:
        func["path"] = node_path_map.get(func["id"])

    func_path = os.path.join(output_dir, config.OUTPUT_FUNCTIONS_JSON)
    with open(func_path, "w", encoding="utf-8") as f:
        json.dump(functions, f, indent=2, ensure_ascii=False)

    logger.info("Generated functions.json with %d functions", len(functions))


def build_code_map(export_root: str) -> str:
    """Build code_map for all C++ files under export_root.

    Traverses all C++ files (.cpp, .c, .cc, .h, .hpp) without skipping any
    directories (including tests/, examples/). For each file, generates a
    per-file code_map JSON. Also generates a summary functions.json.

    Args:
        export_root: Root directory to scan for C++ source files.

    Returns:
        Path to the generated code_map directory.
    """
    export_root = os.path.abspath(export_root)
    code_map_dir = os.path.join(export_root, config.OUTPUT_CODE_MAP_DIR)
    files_dir = os.path.join(code_map_dir, config.OUTPUT_CODE_MAP_FILES_DIR)
    os.makedirs(files_dir, exist_ok=True)

    parser = ClangParser(export_root)

    all_nodes = []
    all_edges = []

    # Collect all C++ files using pathlib
    root_path = Path(export_root)
    cpp_files = []
    for ext in CPP_EXTENSIONS:
        cpp_files.extend(root_path.rglob(f"*{ext}"))

    # Exclude files inside the code_map output directory itself
    code_map_path = Path(code_map_dir)
    cpp_files = [f for f in cpp_files if code_map_path not in f.parents and f.parent != code_map_path]

    logger.info("Found %d C++ files to parse", len(cpp_files))

    for filepath in sorted(cpp_files):
        filepath_str = str(filepath)
        rel_path = os.path.relpath(filepath_str, export_root)
        safe = _safe_name(rel_path)

        try:
            nodes, edges = parser.parse_file(filepath_str)
        except Exception as e:
            logger.error("Failed to parse %s: %s", rel_path, e)
            continue

        # Write per-file code_map
        file_map = {
            "file": rel_path,
            "nodes": nodes,
            "edges": edges,
        }

        map_path = os.path.join(files_dir, f"{safe}{config.OUTPUT_MAP_JSON_SUFFIX}")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(file_map, f, indent=2, ensure_ascii=False)

        # Aggregate for functions.json
        all_nodes.extend(nodes)
        all_edges.extend(edges)

    # Generate functions.json summary
    build_functions_summary(all_nodes, all_edges, code_map_dir)

    logger.info("Code map generated at %s", code_map_dir)
    return code_map_dir
