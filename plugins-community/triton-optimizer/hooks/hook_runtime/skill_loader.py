# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Iterator


def runtime_root() -> Path:
    return Path(__file__).resolve().parents[2]


def skills_root() -> Path:
    env_dir = os.environ.get("TRITON_AGENT_SKILLS_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    platform_skills_dirs = (".claude/skills", ".opencode/skills", ".codex/skills")

    cwd = Path.cwd()
    for rel_dir in platform_skills_dirs:
        candidate = cwd / rel_dir
        if candidate.is_dir():
            return candidate

    home = Path.home()
    for rel_dir in platform_skills_dirs:
        candidate = home / rel_dir
        if candidate.is_dir():
            return candidate

    return runtime_root() / "skills"


@lru_cache(maxsize=1)
def _skill_roots_by_name() -> dict[str, Path]:
    root = skills_root()
    if not root.is_dir():
        raise FileNotFoundError(f"skills root does not exist: {root}")

    discovered: dict[str, Path] = {}
    direct_skills = [path for path in root.iterdir() if path.is_dir()]
    if any((path / "SKILL.md").exists() for path in direct_skills):
        for skill_dir in direct_skills:
            if (skill_dir / "SKILL.md").exists():
                discovered[skill_dir.name] = skill_dir
        return discovered

    for group_dir in direct_skills:
        for skill_dir in group_dir.iterdir():
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                discovered[skill_dir.name] = skill_dir
    return discovered


def skill_script_root(skill_name: str) -> Path:
    roots = _skill_roots_by_name()
    if skill_name not in roots:
        raise FileNotFoundError(f"skill not found in runtime payload: {skill_name!r}")
    return roots[skill_name]


def skill_script_path(skill_name: str, script_name: str) -> Path:
    relative = Path(script_name + ".py")
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid skill script path: {script_name!r}")
    path = skill_script_root(skill_name) / "scripts" / relative
    if not path.exists():
        raise FileNotFoundError(f"Skill script does not exist: {path}")
    return path


def operator_eval_skill_root() -> Path:
    return skill_script_root("triton-npu-optimize")


def operator_eval_script_path(script_name: str) -> Path:
    return skill_script_path("triton-npu-optimize", script_name)


@contextmanager
def _script_import_path(path: Path) -> Iterator[None]:
    scripts_root = str(path)
    added = scripts_root not in sys.path
    if added:
        sys.path.insert(0, scripts_root)
    try:
        yield
    finally:
        if added:
            sys.path.remove(scripts_root)


def _execute_script_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load skill script: {path}")
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        with _script_import_path(path.parent):
            spec.loader.exec_module(module)
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
    return module


@lru_cache(maxsize=None)
def load_skill_script_module(skill_name: str, script_name: str) -> ModuleType:
    path = skill_script_path(skill_name, script_name)
    module_name = (
        f"skill_{skill_name.replace('-', '_')}_"
        f"{script_name.replace('/', '_').replace('-', '_')}"
    )
    return _execute_script_module(path, module_name)


def load_operator_eval_script_module(script_name: str) -> ModuleType:
    return load_skill_script_module("triton-npu-optimize", script_name)
