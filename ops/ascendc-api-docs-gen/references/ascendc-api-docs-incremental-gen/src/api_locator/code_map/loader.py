# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Code map data loader for API locator."""

import json
import os

import config
from config import CODE_MAP_EDGE_TYPES, CODE_MAP_NODE_TYPES


class CodeMapLoader:
    """Loader for code_map data (per-file maps and functions.json)."""

    def __init__(self, code_map_dir: str):
        self.code_map_dir = code_map_dir
        self._functions_map = None

    def load_file_code_map(self, file_path: str) -> dict:
        """Load and filter per-file code_map.

        Args:
            file_path: Source file path (e.g. 'include/test.h').

        Returns:
            Dict with filtered 'nodes' and 'edges' lists.
        """
        safe_name = file_path.replace("/", "_").replace("\\", "_")
        map_path = os.path.join(
            self.code_map_dir, config.OUTPUT_CODE_MAP_FILES_DIR, f"{safe_name}{config.OUTPUT_MAP_JSON_SUFFIX}"
        )

        if not os.path.isfile(map_path):
            return {"nodes": [], "edges": []}

        with open(map_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        filtered_nodes = [n for n in data.get("nodes", []) if n.get("type", "").lower() in CODE_MAP_NODE_TYPES]
        filtered_edges = [e for e in data.get("edges", []) if e.get("type", "").lower() in CODE_MAP_EDGE_TYPES]

        return {"nodes": filtered_nodes, "edges": filtered_edges}

    def load_functions_map(self) -> dict:
        """Lazily load functions.json and build func_id -> info mapping.

        Returns:
            Dict mapping func_id to {calls, called_by, returns}.
        """
        if self._functions_map is not None:
            return self._functions_map

        func_path = os.path.join(self.code_map_dir, config.OUTPUT_FUNCTIONS_JSON)

        if not os.path.isfile(func_path):
            self._functions_map = {}
            return self._functions_map

        with open(func_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        result = {}
        for item in data:
            func_id = item.get("id") or item.get("name", "")
            result[func_id] = {
                "calls": item.get("calls", []),
                "called_by": item.get("called_by", []),
                "returns": item.get("returns", []),
            }

        self._functions_map = result
        return self._functions_map

    def get_function_info(self, func_name: str) -> dict:
        """Query function call information.

        Args:
            func_name: Function name to look up.

        Returns:
            Dict with {calls, called_by, returns} or default empty values.
        """
        func_map = self.load_functions_map()
        return func_map.get(func_name, {"calls": [], "called_by": [], "returns": []})
