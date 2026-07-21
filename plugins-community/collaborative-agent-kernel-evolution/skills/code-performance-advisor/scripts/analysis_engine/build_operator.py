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

"""AscendC operator build bridge: compile, install .run, and rebuild pybind.

This script bridges the gap between code-performance-advisor (which modifies
AscendC .cpp source) and ascendc_evaluation (which needs compiled .run +
pybind to run tests).

Pipeline:
  1. Copy modified source files to the build directory
  2. Execute build.sh to compile the operator
  3. Install the generated .run package
  4. Invoke generate_pybind.py to rebuild Python bindings

Usage:
    python build_operator.py --op fastgelu \
        --source-dir workspace/sessions/<session_id>/working_code/
    python build_operator.py --op fastgelu --rollback
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parents[2]
CAKE_ROOT = ROOT.parents[2]  # skills/code-performance-advisor -> CAKE2
OUTPUT_DIR = CAKE_ROOT / "output"

EVALUATION_SCRIPTS = (
    CAKE_ROOT / ".claude" / "skills" / "ascendc_evaluation" / "scripts"
)
GENERATE_PYBIND = EVALUATION_SCRIPTS / "generate_pybind.py"

WORKSPACE = ROOT / "workspace"
INPUTS = WORKSPACE / "inputs"
SESSIONS = WORKSPACE / "sessions"


def _capitalize_op(op: str) -> str:
    """Convert operator name to CamelCase for directory naming.

    Examples: fastgelu -> Fastgelu, addlayernorm -> Addlayernorm
    """
    return op[0].upper() + op[1:] if op else op


def _find_build_dir(op: str) -> Path | None:
    """Locate the build directory for operator.

    Search strategy (in order):
    1. output/{op}/{OpName}Custom/  (capitalized name + "Custom")
    2. output/{op}/*Custom/         (any *Custom subdirectory)
    3. output/{op}/                 (直接使用 op 目录,如果有 build.sh)
    4. output/{op}/*/               (任何包含 build.sh 的子目录)
    """
    op_output = OUTPUT_DIR / op
    if not op_output.exists():
        return None

    # Strategy 1: {OpName}Custom
    capitalized = _capitalize_op(op)
    candidate = op_output / f"{capitalized}Custom"
    if candidate.exists() and (candidate / "build.sh").exists():
        return candidate

    # Strategy 2: Any *Custom directory with build.sh
    for d in op_output.iterdir():
        if d.is_dir() and d.name.endswith("Custom") and (d / "build.sh").exists():
            return d

    # Strategy 3: Use op_output directly if it has build.sh
    if (op_output / "build.sh").exists():
        return op_output

    # Strategy 4: Search for any subdirectory with build.sh
    for d in op_output.iterdir():
        if d.is_dir() and (d / "build.sh").exists():
            return d

    return None


def _copy_source(source_dir: Path, build_dir: Path) -> int:
    """Copy modified source files to the build directory.

    Copies op_host/ and op_kernel/ from source_dir into build_dir,
    preserving directory structure.

    Returns count of files copied.
    """
    count = 0
    for sub in ("op_host", "op_kernel"):
        src = source_dir / sub
        dst = build_dir / sub
        if not src.exists():
            continue
        for item in src.rglob("*"):
            if item.is_file():
                rel = item.relative_to(src)
                target = dst / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
                count += 1
    return count


def _find_latest_session_working_code(op: str) -> Path | None:
    """Find latest session working_code dir for given operator."""
    if not SESSIONS.exists():
        return None

    candidates = [p for p in SESSIONS.glob(f"*_{op}_*") if p.is_dir()]
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    working = latest / "working_code"
    return working if working.exists() else None


def _run_build(build_dir: Path, timeout: int = 300) -> tuple[bool, str]:
    """Execute build.sh in the build directory.

    Returns (success, output_or_error).
    """
    build_script = build_dir / "build.sh"
    if not build_script.exists():
        return False, f"build.sh not found at {build_script}"

    try:
        os.chmod(
            build_script,
            os.stat(build_script).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
        )
        result = subprocess.run(
            [str(build_script)],
            cwd=str(build_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            return False, f"Build failed (exit code {result.returncode}):\n{output}"
        return True, output
    except subprocess.TimeoutExpired:
        return False, f"Build timed out after {timeout}s"
    except Exception as e:
        return False, f"Build error: {e}"


def _find_run_package(build_dir: Path) -> Path | None:
    """Find the generated .run package in build_out/."""
    build_out = build_dir / "build_out"
    if not build_out.exists():
        return None
    candidates = list(build_out.glob("*.run"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _install_run(run_package: Path) -> tuple[bool, str]:
    """Install the .run package."""
    try:
        os.chmod(
            run_package,
            os.stat(run_package).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
        )
        result = subprocess.run(
            [str(run_package), "--install"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            return False, f"Install failed (exit code {result.returncode}):\n{output}"
        return True, output
    except subprocess.TimeoutExpired:
        return False, "Install timed out"
    except Exception as e:
        return False, f"Install error: {e}"


def _rebuild_pybind(op: str) -> tuple[bool, str]:
    """Invoke generate_pybind.py to rebuild Python bindings."""
    if not GENERATE_PYBIND.exists():
        return False, f"generate_pybind.py not found at {GENERATE_PYBIND}"

    try:
        result = subprocess.run(
            [sys.executable, str(GENERATE_PYBIND), "--op", op],
            cwd=str(CAKE_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            # Pybind rebuild is non-fatal in some configurations
            return False, f"Pybind rebuild warning (exit {result.returncode}):\n{output}"
        return True, output
    except subprocess.TimeoutExpired:
        return False, "Pybind rebuild timed out"
    except Exception as e:
        return False, f"Pybind error: {e}"


def cmd_build(args: argparse.Namespace) -> int:
    op = args.op
    source_dir = Path(args.source_dir).resolve() if args.source_dir else None
    timeout = args.timeout

    if source_dir is None:
        source_dir = _find_latest_session_working_code(op)
        if source_dir:
            logger.info(f"[INFO] Auto-selected source dir from latest session: {source_dir}")

    build_dir = _find_build_dir(op)
    if build_dir is None:
        logger.info(f"[ERROR] Build directory not found for operator '{op}'")
        logger.info(f"  Expected: output/{op}/{_capitalize_op(op)}Custom/")
        return 1

    logger.info(f"[BUILD] Operator: {op}")
    logger.info(f"  Build dir: {build_dir}")

    # Step 1: Copy source files
    if source_dir:
        if not source_dir.exists():
            logger.info(f"[ERROR] Source directory not found: {source_dir}")
            return 1
        logger.info(f"\n[STEP 1] Copying source files from {source_dir}")
        count = _copy_source(source_dir, build_dir)
        logger.info(f"  Copied {count} files")
        if count == 0:
            logger.info(f"[WARN] No source files found in {source_dir}/op_host/ or op_kernel/")
    else:
        logger.info(f"\n[STEP 1] No source-dir specified, building with existing files")

    # Step 2: Build
    logger.info(f"\n[STEP 2] Running build.sh (timeout={timeout}s)")
    build_ok, build_msg = _run_build(build_dir, timeout=timeout)
    if not build_ok:
        logger.info(f"[FAIL] {build_msg}")
        return 2

    # Step 3: Install .run package
    run_pkg = _find_run_package(build_dir)
    if run_pkg:
        logger.info(f"\n[STEP 3] Installing {run_pkg.name}")
        install_ok, install_msg = _install_run(run_pkg)
        if not install_ok:
            logger.info(f"[FAIL] {install_msg}")
            return 3
        logger.info(f"  Install successful")
    else:
        logger.info(f"\n[STEP 3] No .run package found, skipping install")

    # Step 4: Rebuild pybind
    logger.info(f"\n[STEP 4] Rebuilding pybind")
    pybind_ok, pybind_msg = _rebuild_pybind(op)
    if not pybind_ok:
        logger.info(f"[WARN] {pybind_msg}")
        logger.info(f"  Pybind rebuild is not critical; evaluation may still work.")
    else:
        logger.info(f"  Pybind rebuild successful")

    logger.info(f"\n[DONE] Build pipeline completed for '{op}'")
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    """Rollback to baseline code and rebuild."""
    op = args.op
    baseline_code = INPUTS / op / "code"

    if not baseline_code.exists():
        logger.info(f"[ERROR] Baseline code not found: {baseline_code}")
        return 1

    build_dir = _find_build_dir(op)
    if build_dir is None:
        logger.info(f"[ERROR] Build directory not found for operator '{op}'")
        return 1

    logger.info(f"[ROLLBACK] Restoring baseline code for '{op}'")
    count = _copy_source(baseline_code, build_dir)
    logger.info(f"  Restored {count} files")

    logger.info(f"\n[ROLLBACK] Rebuilding with baseline code")
    build_ok, build_msg = _run_build(build_dir, timeout=args.timeout)
    if not build_ok:
        logger.info(f"[FAIL] Rollback build failed: {build_msg}")
        return 2

    run_pkg = _find_run_package(build_dir)
    if run_pkg:
        install_ok, install_msg = _install_run(run_pkg)
        if not install_ok:
            logger.info(f"[FAIL] Rollback install failed: {install_msg}")
            return 3

    pybind_ok, _ = _rebuild_pybind(op)

    logger.info(f"\n[DONE] Rollback completed for '{op}'")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AscendC operator build bridge: compile + install + pybind"
    )
    sub = parser.add_subparsers(dest="command")

    # The following arguments apply to the default (build) command.
    parser.add_argument("--op", required=True, help="Operator name (e.g., fastgelu)")
    parser.add_argument("--source-dir", default=None,
                        help="Directory with modified source code (op_host/, op_kernel/). Defaults to latest session working_code")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Build timeout in seconds (default: 300)")
    parser.add_argument("--rollback", action="store_true",
                        help="Rollback to baseline code and rebuild")

    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = build_parser()
    args = parser.parse_args()

    if args.rollback:
        return cmd_rollback(args)
    return cmd_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
