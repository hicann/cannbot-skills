#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Resolve the bundled orchestrator engine without relying on the caller cwd."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PLUGIN_NAME = "ascendc-port-orchestrator"
_ENGINE_SENTINEL = Path("src/scripts/orchestrator/__main__.py")


def _manifest_candidates(manifest_path: Path) -> list[tuple[str, Path]]:
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, ValueError, TypeError):
        return []
    if not isinstance(payload, dict) or payload.get("plugin") != _PLUGIN_NAME:
        return []

    candidates: list[tuple[str, Path]] = []
    engine_root = payload.get("engine_root")
    if isinstance(engine_root, str) and engine_root:
        candidates.append(("manifest.engine_root", Path(engine_root)))

    hook_settings = payload.get("hooks_settings_engine")
    if isinstance(hook_settings, str) and hook_settings:
        settings_path = Path(hook_settings)
        if settings_path.name == "settings.json" and settings_path.parent.name == ".claude":
            candidates.append(("manifest.hooks_settings_engine", settings_path.parent.parent))

    plugin_root = payload.get("plugin_root")
    if isinstance(plugin_root, str) and plugin_root:
        candidates.append(("manifest.plugin_root", Path(plugin_root) / "engine"))
    return candidates


def resolve_engine(base_dir: Path) -> Path:
    """Resolve from the installed manifest, then from the real source/plugin tree."""
    if not base_dir.is_absolute():
        raise ValueError("skill Base directory must be an absolute path")

    # Installed skills live at <CONFIG_ROOT>/skills/<name>.  Do not resolve the
    # symlink before this step: its textual path is what identifies CONFIG_ROOT.
    config_root = base_dir.parent.parent
    candidates = _manifest_candidates(config_root / "cannbot-manifest.json")

    # A source-tree invocation, or a marketplace loader that bypasses init.sh,
    # still has <plugin>/skills/<name> as the real Skill location.
    real_base = base_dir.resolve()
    candidates.append(("source-tree fallback", real_base.parent.parent / "engine"))

    attempted: list[str] = []
    seen: set[Path] = set()
    for source, candidate in candidates:
        if not candidate.is_absolute():
            attempted.append(f"{source}=<non-absolute:{candidate}>")
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        attempted.append(f"{source}={resolved}")
        if resolved.is_dir() and (resolved / _ENGINE_SENTINEL).is_file():
            return resolved

    detail = "; ".join(attempted) if attempted else "no candidates"
    raise FileNotFoundError(f"bundled orchestrator engine not found ({detail})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="exact absolute Base directory printed by the Skill loader",
    )
    args = parser.parse_args()
    try:
        print(resolve_engine(args.base_dir))
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
