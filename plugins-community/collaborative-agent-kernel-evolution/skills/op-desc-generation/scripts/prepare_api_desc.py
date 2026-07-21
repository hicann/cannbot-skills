#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Prepare api_description for downstream operator pipeline.

This helper copies a provided api_description document into the target work_dir
so downstream steps (especially evaluation) can rely on the conventional
work_dir/api_description.md lookup.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_source(api_desc: str | None) -> Path:
    if api_desc:
        src = Path(api_desc).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"api_description file not found: {src}")
        return src

    raise FileNotFoundError("Cannot resolve api_description source. Provide --api-desc.")


def copy_into_work_dir(src: Path, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    dst = work_dir / "api_description.md"
    shutil.copy2(src, dst)
    return dst


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Copy api_description into work_dir for downstream stages")
    parser.add_argument("--work-dir", required=True, help="Target work directory")
    parser.add_argument("--api-desc", default=None, help="Explicit source api_description(.md) path")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).expanduser().resolve()
    src = resolve_source(args.api_desc)
    dst = copy_into_work_dir(src, work_dir)
    logger.info(dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
