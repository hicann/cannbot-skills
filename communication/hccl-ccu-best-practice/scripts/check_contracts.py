#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""One read-only invocation for launch, sync and arithmetic declaration checks."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True


def run_checks(launch, sync, hcomm_root, baseline=None, scripts=None):
    scripts = Path(scripts) if scripts else Path(__file__).resolve().parent
    checks = {}
    for name, filename, contract in (
        ("launch", "check_launch_contract.py", launch),
        ("sync", "check_sync_contract.py", sync),
        ("arith", "check_arith_ops.py", sync),
    ):
        command = [sys.executable, "-B", str(scripts / filename), str(contract)]
        if name != "launch":
            command += ["--hcomm-root", str(hcomm_root)]
            if baseline is not None:
                command += ["--baseline", str(baseline)]
        try:
            raw = Path(contract).read_bytes()
            if not (scripts / filename).is_file():
                raise ValueError(f"missing checker: {filename}")
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=30, check=False
            )
            if Path(contract).read_bytes() != raw:
                raise ValueError(
                    "contract changed while checker was running; freeze and rerun"
                )
            # Preserve child output exactly inside JSON; failures do not skip the other checks.
            checks[name] = {
                "ok": result.returncode == 0,
                "exit_code": result.returncode,
                "contract_sha256": hashlib.sha256(raw).hexdigest(),
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            checks[name] = {"ok": False, "exit_code": None, "error": str(exc)}
    sync_hashes = {
        row["contract_sha256"]
        for name, row in checks.items()
        if name in {"sync", "arith"} and "contract_sha256" in row
    }
    errors = (
        [
            "sync and arithmetic checks observed different contract bytes; freeze and rerun"
        ]
        if len(sync_hashes) > 1
        else []
    )
    return {
        "schema_version": 1,
        "ok": not errors and all(row["ok"] for row in checks.values()),
        "checks": checks,
        "errors": errors,
        "scope": (
            "declared contracts and source-backed limits only; no C++ completeness, "
            "liveness, device or deployment proof"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", required=True, type=Path)
    parser.add_argument("--sync", required=True, type=Path)
    parser.add_argument("--hcomm-root", required=True, type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="explicit reviewed knowledge baseline; same semantics as individual checkers",
    )
    args = parser.parse_args(argv)
    result = run_checks(args.launch, args.sync, args.hcomm_root, args.baseline)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
