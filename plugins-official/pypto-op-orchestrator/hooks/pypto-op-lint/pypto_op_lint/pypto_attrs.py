#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""PyPTO 顶层属性查询、AST 提取与 Markdown 代码块抽取。

本模块为 OL55 (禁止使用 PyPTO 中不存在的 ``pypto.<attr>``) 提供基础能力:

1. ``get_pypto_attrs()``  — 通过子进程导入 pypto, 取 ``dir(pypto)`` 集合,
   并落地到带 TTL 的本地缓存文件, 避免每次 post-edit hook 都 fork 一个
   ``import pypto``  (PyPTO import 较重, 通常需要 1-3 秒)。
2. ``extract_pypto_attrs()`` — 在源代码 AST 中提取所有 ``pypto.<attr>``
   形式的属性访问 (含别名解析与链式属性最外层名称)。
3. ``extract_python_blocks()`` — 从 Markdown 抽取 fenced code block 中
   可被 ``ast.parse`` 接受的 Python 片段, 给 DESIGN.md 等文档使用。

设计取舍:

- 缓存方式: 文件缓存 + TTL=3600s。Mem-only cache 不能跨 hook 调用复用;
  hardcoded allowlist 又会让 PyPTO 升级失同步。
- 子进程失败处理: ``get_pypto_attrs`` 返回 ``None`` 而非抛出, 让调用方
  优雅地降级为 SKIP (例如本地未装 PyPTO 的 CI 环境)。
- 链式属性 (如 ``pypto.frontend.jit``): 仅校验最外层名 ``frontend``;
  深入 ``pypto.frontend.*`` 的校验属于另一层问题, 不在 OL55 范围内。
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import shutil
import subprocess
import time
from typing import Optional

from .core import PYTHON_BIN

logger = logging.getLogger(__name__)

# ── 缓存配置 ──
_cache_override = os.environ.get("PYPTO_OP_LINT_CACHE_DIR", "").strip()
_cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
if not _cache_home:
    _cache_home = os.path.expanduser("~/.cache")
_CACHE_DIR = os.path.abspath(os.path.expanduser(_cache_override)) if _cache_override else os.path.join(
    _cache_home, "pypto-gym", "pypto-op-lint"
)
_CACHE_FILE = os.path.join(_CACHE_DIR, "pypto_attrs.json")
_CACHE_TTL_SECONDS = 3600  # 1 小时
_SUBPROCESS_TIMEOUT_SECONDS = 15  # PyPTO import 通常 1-3 秒, 15 秒安全冗余

# ── Markdown 代码块 fence 正则 ──
# 匹配 ```python / ```py / ``` (无语言指定) 三种 fence 起始;
# 语言为 bash/shell/yaml/json 等明确非 Python 时排除。
# fence end 必须是行首的 ``` (后接换行或文件末), 否则 body 会被 ``re.MULTILINE``
# 模式下的 ``$`` 在空行处提前 anchored, 导致 body 被截断。
_FENCE_RE = re.compile(
    r"^```(?P<lang>[a-zA-Z0-9_+-]*)[^\n]*\n(?P<body>.*?)\n^```\s*(?:\n|\Z)",
    re.MULTILINE | re.DOTALL,
)
_PYTHON_FENCE_LANGS = {"", "python", "py", "python3"}


def _read_cache() -> Optional[set[str]]:
    """读取缓存文件; 过期或缺失返回 None。"""
    if not os.path.isfile(_CACHE_FILE):
        return None
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("[OL55] 缓存文件读取失败, 将重新探测: %s", e)
        return None
    ts = data.get("ts", 0)
    if time.time() - ts > _CACHE_TTL_SECONDS:
        return None
    attrs = data.get("attrs")
    if not isinstance(attrs, list):
        return None
    return set(attrs)


def _write_cache(attrs: set[str]) -> None:
    """将探测到的属性集合写入缓存文件。失败仅 debug 记录, 不影响主流程。"""
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        tmp_path = _CACHE_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "attrs": sorted(attrs)}, f)
        os.replace(tmp_path, _CACHE_FILE)
    except OSError as e:
        logger.debug("[OL55] 缓存写入失败 (非致命): %s", e)


def _append_unique(candidates: list[str], seen: set[str], path: str) -> None:
    if not path or path in seen:
        return
    seen.add(path)
    candidates.append(path)


def _conda_env_python_paths(conda_bin: str) -> list[str]:
    try:
        result = subprocess.run(
            [conda_bin, "info", "--envs", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        logger.debug("[OL55] conda info --envs --json 失败: %s", exc)
        return []
    return [
        os.path.join(path, "bin", "python")
        for path in data.get("envs", []) or []
        if isinstance(path, str)
    ]


def _enumerate_python_candidates() -> list[str]:
    """枚举可能存在 pypto 的 Python 解释器路径 (去重, 保留顺序)。

    探索顺序 (不依赖任何用户特定的环境变量, 仅使用 conda 通用机制):

    1. ``$CONDA_PREFIX/bin/python`` — 当前 shell 中已 activate 的 conda env。
       (`conda activate <env>` 后这个变量会被自动设置, 是最常见的 pypto env
       入口。)
    2. ``sys.executable`` (即 ``PYTHON_BIN``) — lint hook 自身的解释器,
       适用于 opencode 本身就是从 pypto env 启动的情况。
    3. ``conda info --envs --json`` 列出的所有 env — 即使 hook 在非 pypto env
       下启动, 也能发现机器上其它 env 中的 pypto。
    """
    candidates: list[str] = []
    seen: set[str] = set()
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        _append_unique(candidates, seen, os.path.join(conda_prefix, "bin", "python"))
    _append_unique(candidates, seen, PYTHON_BIN)
    conda_bin = shutil.which("conda")
    if conda_bin:
        for path in _conda_env_python_paths(conda_bin):
            _append_unique(candidates, seen, path)
    return candidates


def _probe_one_python(python_path: str) -> Optional[set[str]]:
    """用单个 Python 解释器尝试 ``import pypto``。

    成功返回 ``dir(pypto)`` 集合 (排除下划线开头); 失败返回 None。
    """
    if not os.path.isfile(python_path):
        return None
    probe_code = (
        "import json, sys\n"
        "try:\n"
        "    import pypto\n"
        "except Exception as e:\n"
        "    print(json.dumps({'error': str(e)}), file=sys.stderr)\n"
        "    sys.exit(2)\n"
        "attrs = [a for a in dir(pypto) if not a.startswith('_')]\n"
        "print(json.dumps(sorted(attrs)))\n"
    )
    try:
        result = subprocess.run(
            [python_path, "-c", probe_code],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        logger.debug("[OL55] %s pypto 探测超时", python_path)
        return None
    except OSError as e:
        logger.debug("[OL55] %s 启动失败: %s", python_path, e)
        return None

    if result.returncode != 0:
        logger.debug(
            "[OL55] %s pypto import 失败: %s",
            python_path, (result.stderr or "")[:200],
        )
        return None

    try:
        attrs = json.loads(result.stdout.strip())
    except json.JSONDecodeError as e:
        logger.debug("[OL55] %s 输出解析失败: %s", python_path, e)
        return None
    if not isinstance(attrs, list):
        return None
    return set(attrs)


def _probe_pypto_attrs_subprocess() -> Optional[set[str]]:
    """按 ``_enumerate_python_candidates`` 顺序尝试探测 pypto。

    任何一个候选成功即返回结果; 全部失败返回 None (调用方降级为 SKIP)。
    """
    for cand in _enumerate_python_candidates():
        attrs = _probe_one_python(cand)
        if attrs is not None:
            logger.debug("[OL55] pypto 在 %s 中找到 (%d attrs)", cand, len(attrs))
            return attrs
    return None


def get_pypto_attrs() -> Optional[set[str]]:
    """获取 ``dir(pypto)`` 集合 (带 TTL 文件缓存)。

    Returns:
        - ``set[str]``: 探测成功 (或缓存命中)。
        - ``None``: PyPTO 未安装 / 导入失败 / 探测超时。
                    调用方应将 OL55 标记为 SKIP, 不要 FAIL。
    """
    cached = _read_cache()
    if cached is not None:
        return cached
    fresh = _probe_pypto_attrs_subprocess()
    if fresh is not None:
        _write_cache(fresh)
    return fresh


# ──────────────────────────────────────────────────────────────────────────────
# AST 提取: pypto.<attr> 形式的属性访问
# ──────────────────────────────────────────────────────────────────────────────


def extract_pypto_attrs(source: str, aliases: Optional[set[str]] = None) -> set[str]:
    """从源代码字符串中提取所有 ``pypto.<attr>`` 形式的最外层属性名。

    Args:
        source: Python 源代码字符串。
        aliases: pypto 包的别名集合 (从 ``_resolve_pypto_aliases`` 获取)。
                 若为 None, 默认使用 ``{"pypto"}``。

    Returns:
        最外层 ``<attr>`` 集合。例如:

        - ``pypto.amax(x)``               → ``{"amax"}``
        - ``pypto.frontend.jit``          → ``{"frontend"}``  (仅最外层)
        - ``pypto.Tensor([], pypto.DT_FP32)`` → ``{"Tensor", "DT_FP32"}``
        - ``pt.zeros(...)`` (with ``import pypto as pt``) → ``{"zeros"}``
    """
    if aliases is None:
        aliases = {"pypto"}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # 源码无法解析时返回空集; 调用方决定如何处理 (SKIP / FAIL)。
        return set()

    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        # 找到最外层的 Name (即 ``pypto.X.Y.Z`` 中的 ``pypto``)。
        # 链式属性 ``pypto.frontend.jit`` 在 AST 中是嵌套结构,
        # 我们只关心位于链根的 Name 是否为 pypto 别名, 并取**链上第一层**的 attr。
        outer = node
        path: list[str] = [outer.attr]
        while isinstance(outer.value, ast.Attribute):
            outer = outer.value
            path.append(outer.attr)
        if isinstance(outer.value, ast.Name) and outer.value.id in aliases:
            # path 最后一个元素是最贴近 Name 的属性 (例: pypto.frontend.jit
            # 中是 ``frontend``); 这是 OL55 要校验的"最外层"属性名。
            found.add(path[-1])
    return found


# ──────────────────────────────────────────────────────────────────────────────
# Markdown fenced code block 提取
# ──────────────────────────────────────────────────────────────────────────────


def extract_python_blocks(md_source: str) -> list[str]:
    """从 Markdown 源文本提取可被 ``ast.parse`` 接受的 Python 代码块。

    匹配 fence 起始为 ``` ``` `` (无语言指定) 或 ``` ```python `` /
    ``` ```py `` / ``` ```python3 ``。明确非 Python 的语言 (如 ``bash``,
    ``shell``, ``yaml``, ``json``) 一律忽略。

    每个块单独尝试 ``ast.parse``; 解析失败的块 (含伪代码 / 不完整片段)
    会被静默丢弃, 因为 OL55 不应在 pseudo-code 上误报。
    """
    blocks: list[str] = []
    for match in _FENCE_RE.finditer(md_source):
        lang = match.group("lang") or ""
        if lang.lower() not in _PYTHON_FENCE_LANGS:
            continue
        body = match.group("body")
        if not body.strip():
            continue
        try:
            ast.parse(body)
        except SyntaxError:
            # 伪代码 / 片段, 跳过 (不视为 OL55 违规)。
            continue
        blocks.append(body)
    return blocks
