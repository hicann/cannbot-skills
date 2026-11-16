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
"""Report target-environment evidence without importing plugins or modifying it."""

import importlib.util
import json
import os
import platform
import shutil
from pathlib import Path


def collect_environment():
    home = os.environ.get("ASCEND_HOME_PATH")
    root = Path(home) if home else None
    headers = {}
    if root:
        for relative in (
            "include/graph/custom_op.h",
            "include/external/graph/custom_op.h",
            "include/graph_metadef/external/graph/custom_op.h",
        ):
            path = root / relative
            if path.is_file():
                text = path.read_text(errors="replace")
                headers[str(path)] = {
                    name: ("class " + name) in text
                    for name in (
                        "EagerExecuteOp",
                        "AnnotatedArgsOp",
                        "ResourceUsageReporter",
                    )
                }
    versions = {}
    if root:
        for relative in ("version.info", "compiler/version.info"):
            path = root / relative
            if path.is_file():
                versions[str(path)] = path.read_text(errors="replace")[:4096]
    try:
        ge = importlib.util.find_spec("ge")
    except (ImportError, ValueError):
        ge = None
    opp_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    paths = [p for p in opp_path.split(os.pathsep) if p]
    return {
        "host": {
            "os": platform.system(),
            "arch": platform.machine(),
            "python": platform.python_version(),
        },
        "ascend_home": home,
        "headers": headers,
        "version_files": versions,
        "ge_package": None if ge is None else ge.origin,
        "commands": {
            name: shutil.which(name)
            for name in (
                "cmake",
                "bisheng",
                "llvm-objcopy",
                "readelf",
                "atc",
                "npu-smi",
            )
        },
        "opp_paths": [{"path": p, "exists": Path(p).exists()} for p in paths],
        "runtime_validation": "not performed; headers/package discovery do not prove ABI or device availability",
    }


def main():
    report = collect_environment()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["headers"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
