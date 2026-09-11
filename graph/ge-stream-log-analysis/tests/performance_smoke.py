#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

LOGGER = logging.getLogger(__name__)
SKILL_ROOT = Path(__file__).resolve().parents[1]


def _source_logs(source_root):
    base = source_root / "examples/acl/2_sample_resnet50_imagenet_classification_dynamic_batch"
    return base / "model/atc_mainstream2.log", base / "resnet50_runtime.log"


def main():
    configured_repo = os.environ.get("GE_REPO_ROOT")
    source_root = Path(configured_repo).expanduser().resolve() if configured_repo else None
    if source_root is None:
        LOGGER.info("SKIP performance smoke: set GE_REPO_ROOT to an external GE checkout")
        return 0

    compile_log, runtime_log = _source_logs(source_root)
    if not compile_log.is_file() or not runtime_log.is_file():
        LOGGER.info("SKIP performance smoke: golden logs missing")
        return 0

    start = time.monotonic()
    result = subprocess.run(
        [
            sys.executable,
            str(SKILL_ROOT / "scripts/analyze_stream_logs.py"),
            "--compile",
            str(compile_log),
            "--runtime",
            str(runtime_log),
            "--repo-root",
            str(source_root),
            "--format",
            "json",
        ],
        stdout=subprocess.DEVNULL,
        check=False,
    )
    elapsed = time.monotonic() - start
    max_rss_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    LOGGER.info("returncode=%s elapsed=%.3fs maxrss_kb=%s", result.returncode, elapsed, max_rss_kb)
    return int(result.returncode != 0 or max_rss_kb > 131072)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
