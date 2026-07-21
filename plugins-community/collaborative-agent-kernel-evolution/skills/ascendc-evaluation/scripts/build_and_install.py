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

"""
Build and install an AscendC custom operator.

Usage:
    python3 build_and_install.py <op_dir>

Example:
    python3 build_and_install.py output/FastGelu/FastGeluCustom
"""

import sys
import os
import re
import stat
import subprocess
import tempfile
import logging
import argparse
from pathlib import Path


def _ensure_executable(path: Path) -> None:
    """Grant execute permission (u/g/o) so the script can be run directly."""
    current_mode = os.stat(str(path)).st_mode
    os.chmod(
        str(path),
        current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )


def find_ascend_home() -> str:
    """Auto-detect ASCEND_HOME_PATH from environment or default locations."""
    for env_var in ("ASCEND_HOME_PATH", "ASCEND_AICPU_PATH", "BASE_LIBS_PATH"):
        val = os.environ.get(env_var)
        if val:
            return val

    default = "/usr/local/Ascend"
    if Path(default).exists():
        return default

    raise EnvironmentError(
        "Cannot find Ascend toolkit. Set ASCEND_HOME_PATH, ASCEND_AICPU_PATH, "
        "or BASE_LIBS_PATH environment variable."
    )


def _patch_makedirs_line(line: str) -> str:
    """Add exist_ok=True to an os.makedirs() call in a single source line."""
    marker = "os.makedirs("
    idx = line.find(marker)
    if idx < 0 or "exist_ok" in line:
        return line
    # Walk forward from the opening paren, counting depth, to find matching ')'
    start = idx + len(marker) - 1  # position of '('
    depth = 0
    for i in range(start, len(line)):
        if line[i] == "(":
            depth += 1
        elif line[i] == ")":
            depth -= 1
            if depth == 0:
                # Insert ', exist_ok=True' right before this closing paren
                return line[:i] + ", exist_ok=True" + line[i:]
    return line  # malformed / multi-line call — leave unchanged


def patch_compile_scripts(op_dir: Path) -> None:
    """
    WF-005: Patch ascendc_compile_kernel.py to add exist_ok=True to all
    os.makedirs() calls and guard os.symlink() against FileExistsError.

    msopgen generates these scripts without exist_ok=True; cmake configure
    creates the working directory first, then the build step recreates it
    and crashes with FileExistsError.
    """
    for script in op_dir.rglob("ascendc_compile_kernel.py"):
        lines = script.read_text(encoding="utf-8").splitlines(keepends=True)
        patched = []
        changed = False
        for line in lines:
            new_line = line

            # Fix 1: os.makedirs(path) → os.makedirs(path, exist_ok=True)
            if "os.makedirs(" in new_line and "exist_ok" not in new_line:
                new_line = _patch_makedirs_line(new_line.rstrip("\n")) + ("\n" if new_line.endswith("\n") else "")

            # Fix 2: os.symlink(src, dst) → guard against FileExistsError
            # Match lines like `        os.symlink(src, dst)`
            m = re.match(r"^(\s+)(os\.symlink\(.*\))\s*$", new_line.rstrip("\n"))
            if m and "FileExistsError" not in new_line:
                indent = m.group(1)
                call = m.group(2)
                eol = "\n" if new_line.endswith("\n") else ""
                new_line = (
                    f"{indent}try:{eol}"
                    f"{indent}    {call}{eol}"
                    f"{indent}except FileExistsError:{eol}"
                    f"{indent}    pass{eol}"
                )

            if new_line != line:
                changed = True
            patched.append(new_line)

        if changed:
            script.write_text("".join(patched), encoding="utf-8")
            logging.info(f"Patched {script} (WF-005: exist_ok + symlink guard)")


def build_operator(op_dir: Path) -> None:
    """Run build.sh inside the operator directory."""
    build_sh = op_dir / "build.sh"
    if not build_sh.exists():
        raise FileNotFoundError(f"build.sh not found in {op_dir}")

    # Apply WF-005 patch before building
    patch_compile_scripts(op_dir)

    ascend_home = find_ascend_home()
    env = os.environ.copy()
    env["ASCEND_HOME_PATH"] = ascend_home
    logging.info(f"Using ASCEND_HOME_PATH={ascend_home}")

    logging.info(f"Running build.sh in {op_dir}")
    result = subprocess.run(
        ["bash", "build.sh"],
        cwd=str(op_dir),
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"build.sh failed with exit code {result.returncode}")

    logging.info("Build completed successfully")


def install_operator(op_dir: Path) -> None:
    """Find and install the .run package into op_dir.parent/vendors/customize.

    Tries two installation strategies in order:
    1. Preferred (WF-006 two-step): bash *.run --noexec --extract=<tmp>,
       then bash install.sh --install-path=<work_dir>.
       Handles makeself packages where the outer header intercepts --install-path.
    2. Fallback (single-step): bash *.run --install-path=<work_dir>.
       Used for CPack/non-makeself packages that support --install-path directly.
    """
    build_out = op_dir / "build_out"
    if not build_out.exists():
        raise FileNotFoundError(f"build_out/ directory not found in {op_dir}")

    run_files = list(build_out.glob("*.run"))
    if not run_files:
        raise FileNotFoundError(f"No .run file found in {build_out}")

    run_file = run_files[0]
    run_file.chmod(run_file.stat().st_mode | 0o755)
    install_path = op_dir.parent.resolve()

    # ── Strategy 1: two-step --noexec extract ──
    try:
        logging.info(f"Installing {run_file.name} (strategy 1: --noexec extract)")
        with tempfile.TemporaryDirectory(prefix="opp_extract_") as extract_dir:
            _ensure_executable(run_file)
            result = subprocess.run(
                [str(run_file), "--noexec", f"--extract={extract_dir}"],
                cwd=str(build_out),
            )
            if result.returncode != 0:
                raise RuntimeError(f"--noexec extract failed (exit {result.returncode})")

            install_sh = Path(extract_dir) / "install.sh"
            if not install_sh.exists():
                sh_candidates = list(Path(extract_dir).glob("*.sh"))
                if not sh_candidates:
                    raise FileNotFoundError("install.sh not found in extracted package")
                install_sh = sh_candidates[0]

            install_sh.chmod(install_sh.stat().st_mode | 0o755)
            _ensure_executable(install_sh)
            result = subprocess.run(
                [str(install_sh), f"--install-path={install_path}"],
                cwd=extract_dir,
            )
            if result.returncode != 0:
                raise RuntimeError(f"install.sh failed (exit {result.returncode})")

        logging.info(f"Installation completed (strategy 1): {install_path}/vendors/customize")
        return

    except Exception as e:
        logging.warning(f"Strategy 1 (--noexec) failed: {e}. Falling back to --install-path.")

    # ── Strategy 2: single-step --install-path (CPack / non-makeself) ──
    logging.info(f"Installing {run_file.name} (strategy 2: --install-path)")
    _ensure_executable(run_file)
    result = subprocess.run(
        [str(run_file), f"--install-path={install_path}"],
        cwd=str(build_out),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Installation failed with both strategies. "
            f"Last exit code: {result.returncode}"
        )
    logging.info(f"Installation completed (strategy 2): {install_path}/vendors/customize")


def verify_installation(op_dir: Path) -> None:
    """Verify that the operator was installed into op_dir.parent/vendors/customize."""
    customize_dir = op_dir.parent / "vendors" / "customize"
    if customize_dir.exists():
        logging.info(f"Verified: customize directory exists at {customize_dir}")
        config_files = list(customize_dir.rglob("*.json"))
        if config_files:
            logging.info(f"Found {len(config_files)} config file(s)")
    else:
        # Fall back to checking system opp path (legacy single-install mode)
        opp_path = os.environ.get("ASCEND_OPP_PATH")
        if not opp_path:
            try:
                ascend_home = find_ascend_home()
                opp_path = os.path.join(ascend_home, "ascend-toolkit/latest/opp")
            except EnvironmentError:
                opp_path = None

        if opp_path:
            sys_customize = Path(opp_path) / "vendors" / "customize"
            if sys_customize.exists():
                logging.info(f"Verified (system path): {sys_customize}")
                return

        logging.warning(
            f"customize directory not found at {customize_dir}. "
            "Installation may have failed."
        )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Build and install an AscendC custom operator"
    )
    parser.add_argument(
        "op_dir",
        type=str,
        help="Absolute path to the operator directory (e.g., /path/to/FastGeluCustom)",
    )
    args = parser.parse_args()

    op_dir = Path(args.op_dir).resolve()
    if not op_dir.is_dir():
        logging.error(f"Operator directory not found: {op_dir}")
        sys.exit(1)

    try:
        build_operator(op_dir)
        install_operator(op_dir)
        verify_installation(op_dir)
        logging.info("Build and install completed successfully")
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
