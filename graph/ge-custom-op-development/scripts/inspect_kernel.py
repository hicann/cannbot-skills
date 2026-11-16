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
"""Inspect the Ascend C device ELF route, not arbitrary kernel containers."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def inspect_elf(data):
    if (
        len(data) < 64
        or data[:4] != b"\x7fELF"
        or data[4] != 2
        or data[5] not in (1, 2)
    ):
        raise ValueError("expected a complete ELF64 header")
    order = "little" if data[5] == 1 else "big"
    kind = int.from_bytes(data[16:18], order)
    machine = int.from_bytes(data[18:20], order)
    if kind != 2 or machine != 0x1029:
        raise ValueError(
            "expected executable DaVinci ELF (type 2, machine 0x1029); host object or unsupported format"
        )
    return {
        "class": "ELF64",
        "byte_order": order,
        "type": kind,
        "machine": hex(machine),
        "bytes": len(data),
        "loading_validation": "not performed; inspect metadata and load on the target runtime",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument(
        "--symbol", help="Require an exact defined global FUNC symbol, using readelf"
    )
    args = parser.parse_args()
    try:
        report = inspect_elf(args.binary.read_bytes())
        if args.symbol:
            command = shutil.which("readelf")
            if not command:
                raise ValueError("readelf is required for --symbol")
            result = subprocess.run(
                [command, "-sW", str(args.binary)],
                check=True,
                capture_output=True,
                text=True,
            )
            symbols = []
            for line in result.stdout.splitlines():
                fields = line.split()
                if (
                    len(fields) >= 8
                    and fields[3:5] == ["FUNC", "GLOBAL"]
                    and fields[6] != "UND"
                ):
                    symbols.append(fields[7])
            if args.symbol not in symbols:
                raise ValueError("defined global FUNC symbol not found: " + args.symbol)
            report["defined_symbol"] = args.symbol
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
