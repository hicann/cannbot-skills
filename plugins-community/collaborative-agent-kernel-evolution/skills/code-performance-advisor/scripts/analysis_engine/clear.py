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

"""Workspace cleanup utility for session-based layout.

Current layout:
- workspace/inputs/<op>/              (baseline input, keep by default)
- workspace/sessions/<session_id>/    (workflow runs)
- workspace/cache/                    (tags and temporary artifacts)

This script focuses on cleaning session/cache artifacts and no longer manages
legacy InputMessages/OutputMessages paths.

Typical usage:
    python clear.py --sessions --dry-run
    python clear.py --sessions
    python clear.py --cache-tags --dry-run
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Iterable, List

logger = logging.getLogger(__name__)


def resolve_workspace_root(root_arg: str | None) -> Path:
    """Resolve workspace root directory."""

    if root_arg:
        return Path(root_arg).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "workspace"


def collect_session_dirs(workspace_root: Path) -> List[Path]:
    """Collect session directories under workspace/sessions/."""
    sessions_dir = workspace_root / "sessions"
    if not sessions_dir.exists():
        return []
    return [p for p in sessions_dir.iterdir() if p.is_dir()]


def collect_tag_files(workspace_root: Path) -> List[Path]:
    """Collect cached tag files under workspace/cache/tags/."""
    tags_dir = workspace_root / "cache" / "tags"
    if not tags_dir.exists():
        return []
    return [p for p in tags_dir.rglob("*.json") if p.is_file()]


def delete_files(files: Iterable[Path], dry_run: bool) -> int:
    """Delete files and return deleted count."""

    deleted_count = 0
    for file_path in files:
        if dry_run:
            logger.info(f"[DRY RUN] Would delete: {file_path}")
            continue

        try:
            file_path.unlink()
            deleted_count += 1
        except OSError as exc:
            logger.info(f"[WARN] Failed to delete {file_path}: {exc}")

    return deleted_count


def delete_dirs(dirs: Iterable[Path], dry_run: bool) -> int:
    """Delete directories recursively and return deleted count."""
    deleted_count = 0
    for dir_path in dirs:
        if dry_run:
            logger.info(f"[DRY RUN] Would delete dir: {dir_path}")
            continue
        try:
            shutil.rmtree(dir_path)
            deleted_count += 1
        except OSError as exc:
            logger.info(f"[WARN] Failed to delete {dir_path}: {exc}")
    return deleted_count


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Keeping this in a separate function makes it easy to add options later.
    """

    parser = argparse.ArgumentParser(
        description="Clear session/cache artifacts under workspace.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "Path to workspace directory. "
            "Defaults to skill_root/workspace."
        ),
    )
    parser.add_argument(
        "--sessions",
        action="store_true",
        help="Delete workspace/sessions/<session_id>/ directories.",
    )
    parser.add_argument(
        "--cache-tags",
        action="store_true",
        help="Delete workspace/cache/tags/*.json files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting anything.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for CLI usage."""

    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    workspace_root = resolve_workspace_root(args.root)
    if not workspace_root.exists() or not workspace_root.is_dir():
        logger.info(f"[ERROR] Workspace directory not found: {workspace_root}")
        return 1

    if not args.sessions and not args.cache_tags:
        logger.info("[ERROR] No cleanup target selected. Use --sessions and/or --cache-tags.")
        return 2

    if args.sessions:
        session_dirs = collect_session_dirs(workspace_root)
        if session_dirs:
            deleted_dirs = delete_dirs(session_dirs, dry_run=args.dry_run)
            if args.dry_run:
                logger.info(f"Dry run complete. {len(session_dirs)} session dir(s) would be deleted.")
            else:
                logger.info(f"Deleted {deleted_dirs} session dir(s) under: {workspace_root / 'sessions'}")
        else:
            logger.info(f"No session directories found under: {workspace_root / 'sessions'}")

    if args.cache_tags:
        tag_files = collect_tag_files(workspace_root)
        if tag_files:
            deleted_files = delete_files(tag_files, dry_run=args.dry_run)
            if args.dry_run:
                logger.info(f"Dry run complete. {len(tag_files)} tag file(s) would be deleted.")
            else:
                logger.info(f"Deleted {deleted_files} tag file(s) under: {workspace_root / 'cache' / 'tags'}")
        else:
            logger.info(f"No tag files found under: {workspace_root / 'cache' / 'tags'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
