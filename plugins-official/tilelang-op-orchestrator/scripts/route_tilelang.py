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
"""Route verified SoC/architecture pairs without loading a compiler or installing packages."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


def classify(evidence):
    """Unknown or conflicting information must never pick a default workflow."""
    if not isinstance(evidence, dict):
        raise ValueError("hardware evidence must be a JSON object")
    # The existing detector reports a runtime/INI disagreement in warnings.
    # Such evidence is insufficient to select a compiler workflow automatically.
    for warning in evidence.get("warnings") or []:
        if "NpuArch 不一致" in str(warning):
            raise ValueError(str(warning))
    soc = str(evidence.get("full_soc") or "")
    arch = str(evidence.get("npu_arch") or "")
    if re.fullmatch(
        r"Ascend910(?:B(?:[1-4]|2C)(?:[-_][A-Za-z0-9]+)?|_93[A-Za-z0-9_]*)", soc
    ):
        platform, expected = "Ascend910", "2201"
    elif re.fullmatch(r"Ascend950(?:PR|DT)(?:_[A-Za-z0-9]+)?", soc):
        platform, expected = "Ascend950", "3510"
    else:
        raise ValueError(f"unsupported or missing full SoC: {soc!r}")
    if arch != expected:
        raise ValueError(f"SoC {soc} requires NpuArch={expected}; received {arch!r}")
    return platform


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-json", type=Path, help="offline fixture, never a live device check"
    )
    parser.add_argument(
        "--probe-script",
        type=Path,
        help="get_npu_arch.py resolved from the installed ascendc-env-check Skill",
    )
    args = parser.parse_args(argv)
    evidence = {}
    mode = "offline" if args.probe_json else "local"
    try:
        if args.probe_json and args.probe_script:
            raise ValueError("--probe-json and --probe-script are mutually exclusive")
        if args.probe_json:
            evidence = json.loads(args.probe_json.read_text())
        else:
            if not args.probe_script:
                raise ValueError(
                    "--probe-script is required for live detection; resolve it from the "
                    "installed ascendc-env-check Skill"
                )
            probe = args.probe_script.expanduser().resolve()
            result = subprocess.run(
                [sys.executable, str(probe), "--json"],
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            evidence = json.loads(result.stdout)
            if result.returncode:
                raise ValueError(f"hardware probe failed: {result.stderr.strip()}")
        platform = classify(evidence)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(
            json.dumps(
                {
                    "platform": None,
                    "detection_mode": mode,
                    "error": str(error),
                    "evidence": evidence,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "platform": platform,
                "detection_mode": mode,
                "full_soc": evidence["full_soc"],
                "npu_arch": evidence["npu_arch"],
                "evidence": evidence,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
