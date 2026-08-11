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

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
from typing import Any, Optional

from .core import SCRIPT_DIR, CheckContext, Finding

NPU_SMI_BIN = shutil.which("npu-smi") or "npu-smi"


def _check_npu_available() -> bool:
    """通过 npu-smi info 检测 NPU 环境是否可用"""
    try:
        result = subprocess.run(
            [NPU_SMI_BIN, "info"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


DTYPE_ALIASES: dict[str, str] = {
    "float32": "fp32", "fp32": "fp32", "dt_fp32": "fp32", "torch.float32": "fp32",
    "float16": "fp16", "fp16": "fp16", "dt_fp16": "fp16", "torch.float16": "fp16",
    "bfloat16": "bf16", "bf16": "bf16", "dt_bf16": "bf16", "torch.bfloat16": "bf16",
}


_UNPARSED = object()


def _parse_list_scalar(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        inner = value[1:-1].strip()
    if not inner:
        return []
    return [_parse_list_item(item.strip()) for item in inner.split(",")]


def _parse_list_item(item: str) -> Any:
    if item.startswith(("'", '"')) and item.endswith(("'", '"')):
        return item[1:-1]
    return _parse_scalar(item)


def _parse_mapping_scalar(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        inner = value[1:-1].strip()
    if not inner:
        return {}
    result: dict[str, Any] = {}
    for pair in inner.split(","):
        pair = pair.strip()
        if ":" not in pair:
            return _UNPARSED
        key, item = pair.split(":", 1)
        result[key.strip().strip("'\"")] = _parse_scalar(item)
    return result


def _parse_number(value: str) -> int | float | object:
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", value):
        return float(value)
    return _UNPARSED


def _parse_scalar(text: str) -> Any:
    value = text.strip()
    if value in ("", "null", "None"):
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        return _parse_list_scalar(value)
    if value.startswith("{") and value.endswith("}"):
        parsed = _parse_mapping_scalar(value)
        return value if parsed is _UNPARSED else parsed
    number = _parse_number(value)
    if number is not _UNPARSED:
        return number
    # Bare comma-separated values (no brackets) => parse as list
    if "," in value and not value.startswith(("[", "{")):
        return [_parse_scalar(v.strip()) for v in value.split(",") if v.strip()]
    return value


def _split_front_matter(content: str) -> tuple[str, str] | None:
    if not content.startswith("---\n"):
        return None
    end = content.find("\n---\n", 4)
    if end < 0:
        return None
    return content[4:end], content[end + 5:]


def _load_yaml_mapping(header: str) -> dict[str, Any] | None:
    try:
        import yaml  # noqa: PLC0415

        parsed = yaml.safe_load(header)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    for key, val in parsed.items():
        if isinstance(val, str) and val.strip().startswith(("{", "[")):
            parsed[key] = _parse_scalar(val)
    return parsed


def _parse_front_matter_lines(header: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for raw in header.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = _parse_scalar(value)
    return meta


def _parse_front_matter(content: str) -> tuple[dict[str, Any], str]:
    """解析 markdown front matter。

    优先使用 yaml.safe_load 解析（支持多行 YAML），若 YAML 解析失败或
    结果非 dict 则回退到逐行解析。对 YAML 无法识别的 Python 字面量值
    （如 ``{'key': 'val'}``）做后处理兼容。
    """
    sections = _split_front_matter(content)
    if sections is None:
        return {}, content
    header, body = sections
    parsed = _load_yaml_mapping(header)
    if parsed is not None:
        return parsed, body
    return _parse_front_matter_lines(header), body


def parse_front_matter(content: str) -> tuple[dict[str, Any], str]:
    """Public parser seam used by clients and parser contract tests."""
    return _parse_front_matter(content)


def _load_doc_meta(ctx: CheckContext, filename: str) -> dict[str, Any]:
    content = ctx.read_file(filename)
    if not content:
        return {}
    meta, _ = _parse_front_matter(content)
    return meta


_tolerance_schema_cache: list[dict[str, Any]] | None = None


def _load_tolerance_schema() -> list[dict[str, Any]]:
    """从 rules.json 加载 tolerance_schema.oneOf 定义（带模块级缓存）。"""
    global _tolerance_schema_cache
    if _tolerance_schema_cache is not None:
        return _tolerance_schema_cache
    rules_path = os.path.join(SCRIPT_DIR, "rules.json")
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        schema: list[dict[str, Any]] = []
    else:
        schema = data.get("tolerance_schema", {}).get("oneOf", [])
    _tolerance_schema_cache = schema
    return schema


def _validate_tolerance(tol: dict[str, Any]) -> list[str]:
    """基于 rules.json 中 tolerance_schema.oneOf 校验 tolerance dict。

    只要匹配任一模式即通过。
    """
    schemas = _load_tolerance_schema()
    if not schemas:
        # 无 schema 定义时回退到基础校验：只检查是否为非空 dict
        return []

    for schema in schemas:
        required = schema.get("required", [])
        if all(k in tol for k in required):
            return []  # 匹配成功

    # 所有模式都不匹配，列出可接受的模式
    modes = [f"{s.get('mode', '?')}({', '.join(s.get('required', []))})" for s in schemas]
    return [f"tolerance must match one of: {' | '.join(modes)}"]


def _validate_doc_schema(doc_type: str, meta: dict[str, Any]) -> list[str]:
    required: dict[str, list[str]] = {
        "SPEC": ["schema_version", "op_name", "supported_dtypes", "p0_shapes", "tolerance"],
        "DESIGN": ["schema_version", "op_name", "dynamic_axes"],
        "API_REPORT": ["schema_version", "op_name"],
    }
    errors: list[str] = []
    for key in required.get(doc_type, []):
        if key not in meta or meta[key] in (None, "", []):
            errors.append(f"missing field: {key}")
    if doc_type == "SPEC":
        if "supported_dtypes" in meta and not isinstance(meta.get("supported_dtypes"), list):
            errors.append("invalid type: supported_dtypes must be list")
        if "p0_shapes" in meta and not isinstance(meta.get("p0_shapes"), list):
            errors.append("invalid type: p0_shapes must be list")
        if "tolerance" in meta and not isinstance(meta.get("tolerance"), dict):
            errors.append("invalid type: tolerance must be dict")
        if isinstance(meta.get("tolerance"), dict):
            errors.extend(_validate_tolerance(meta["tolerance"]))
    if doc_type == "DESIGN":
        if "dynamic_axes" in meta and not isinstance(meta.get("dynamic_axes"), list):
            errors.append("invalid type: dynamic_axes must be list")
    return errors


def validate_doc_schema(doc_type: str, meta: dict[str, Any]) -> list[str]:
    """Public schema-validation seam used by clients and contract tests."""
    return _validate_doc_schema(doc_type, meta)


def _extract_spec_dtypes_from_meta(meta: dict[str, Any]) -> set[str]:
    raw = meta.get("supported_dtypes", [])
    if not isinstance(raw, list):
        return set()
    result: set[str] = set()
    for item in raw:
        key = str(item).strip().lower()
        canonical = DTYPE_ALIASES.get(key)
        if canonical:
            result.add(canonical)
    return result


def _extract_test_dtypes(source: str) -> set[str]:
    """从 test 文件源码中提取使用的 dtype"""
    dtypes: set[str] = set()
    lower = source.lower()
    for alias, canonical in DTYPE_ALIASES.items():
        if alias in lower:
            dtypes.add(canonical)
    return dtypes


def _extract_tolerance(content: str, key: str) -> list[float]:
    """从文本中提取所有 atol 或 rtol 数值"""
    values: list[float] = []
    for pat in [
        rf"{key}\s*[:=]\s*([0-9]+\.?[0-9]*(?:[eE][+\-]?\d+)?)",
        rf"\*\*{key}\*\*\s*[:=]?\s*([0-9]+\.?[0-9]*(?:[eE][+\-]?\d+)?)",
    ]:
        for m in re.finditer(pat, content, re.IGNORECASE):
            try:
                values.append(float(m.group(1)))
            except ValueError:
                continue
    return values


def _extract_shapes_from_text(content: str) -> set[tuple[int, ...]]:
    """从文本中提取 shape 元组，如 [1, 2048, 4096] 或 (1, 2048, 4096)"""
    shapes: set[tuple[int, ...]] = set()
    for m in re.finditer(r'[\[\(](\d+(?:\s*,\s*\d+)+)[\]\)]', content):
        try:
            dims = tuple(int(x.strip()) for x in m.group(1).split(","))
            if len(dims) >= 2:
                shapes.add(dims)
        except ValueError:
            continue
    return shapes


def _extract_markdown_headings(content: str) -> set[str]:
    _, body = _parse_front_matter(content)
    headings = re.findall(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", body, flags=re.MULTILINE)
    return {h.strip().lower() for h in headings}


def _has_heading_like(headings: set[str], *keywords: str) -> bool:
    keys = [k.strip().lower() for k in keywords if k.strip()]
    for h in headings:
        if any(k in h for k in keys):
            return True
    return False


def _extract_section_text(content: str, heading_keyword: str) -> str:
    _, body = _parse_front_matter(content)
    lines = body.splitlines()
    key = heading_keyword.strip().lower()
    start = -1
    start_level = 0
    for idx, line in enumerate(lines):
        m = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", line)
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip().lower()
        if key in title:
            start = idx + 1
            start_level = level
            break
    if start < 0:
        return ""

    buff: list[str] = []
    for line in lines[start:]:
        m = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", line)
        if m and len(m.group(1)) <= start_level:
            break
        buff.append(line)
    return "\n".join(buff).strip()


def _syntax_error_finding(ctx: CheckContext, rule_id: str, filename: str) -> Optional[Finding]:
    tree = ctx.parse_file(filename)
    if tree is not None:
        return None
    error = ctx.parse_error(filename)
    if not error:
        return None
    return ctx.make_finding(
        rule_id,
        "FAIL",
        f"{filename} 存在语法错误，无法解析: {error}",
        file=filename,
    )


PENDING_STUB_MARKER = "PYPTO_PENDING_STUB"


def _is_pending_stub(ctx: CheckContext, file_rel: str) -> bool:
    """True iff the file is marked as a "pending stub" by the designer.

    Convention: a stub created by :mod:`pypto-op-designer` for a not-yet-active
    phase contains a single comment line near the top:

        # PYPTO_PENDING_STUB: phase=M_k — implementation pending

    When the responsible coder fills the stub during Phase M_k dispatch, they
    remove this marker (or simply overwrite the file with the real
    implementation). Lint excludes any file still carrying the marker so:

    - post-edit hook on M1 won't complain about M2/M3 stubs
    - phase gate at complete_phase(M1) won't complain about M2/M3 stubs
    - complete_stage(5) won't complain if cleanup ran prematurely (the marker
      should be gone by then; if it's not, that's a real issue worth surfacing)

    We deliberately scan only the first 512 bytes so massive files don't pay a
    parse cost just for this check.
    """
    full = os.path.join(ctx.op_dir, file_rel)
    try:
        with open(full, "r", encoding="utf-8") as f:
            head = f.read(512)
    except (OSError, UnicodeDecodeError):
        return False
    return PENDING_STUB_MARKER in head


def _phase_to_module_suffix(phase: str) -> str:
    """``Mk`` → ``"1...k"`` (累积模块文件后缀)。

    例:
        ``M1`` → ``"1"``                (modules/<op>_module1_impl.py)
        ``M2`` → ``"12"``               (modules/<op>_module12_impl.py)
        ``M3`` → ``"123"``              (modules/<op>_module123_impl.py)
        ``M10`` → ``"12345678910"``    (单纯连接, 不分隔)

    ``ValueError`` 当 phase 不是 ``M`` 加正整数时抛出。
    """
    if not phase.startswith("M"):
        raise ValueError(f"unexpected phase format: {phase!r} (expected 'M<int>')")
    try:
        n = int(phase[1:])
    except ValueError as e:
        raise ValueError(f"unexpected phase format: {phase!r}") from e
    if n < 1:
        raise ValueError(f"phase index must be >= 1, got {phase!r}")
    return "".join(str(i) for i in range(1, n + 1))


def _impl_files_to_scan(ctx: CheckContext) -> list[str]:
    """返回当前算子所有 impl 文件的相对路径列表（用于 module-level lint 覆盖）。

    模式 A — file-scoped (``ctx.file_scope`` 非 None)：
        仅返回 ``ctx.file_scope`` 指定的单一文件。post-edit hook 在
        Coder Write/Edit 单文件后调用, 此时只想对刚写入的文件运行 lint,
        不应让其他 module 的 stub 文件触发误判 (例如 ``modules/<op>_module12_impl.py``
        的空 stub 会让 OL01/OL07 等 FAIL, 即使本次只写了 module1)。
        跨模块的整合性检查留给 phase / stage 网关。
        优先级最高: 与 phase_scope 同时设置时, file_scope 胜出。

    模式 B — phase-scoped (``ctx.phase_scope`` 非 None, ``file_scope`` 为 None)：
        仅返回 ``modules/<op>_module<suffix>_impl.py``,
        其中 ``suffix`` 由 :func:`_phase_to_module_suffix` 解析自
        ``ctx.phase_scope`` (如 ``M1``→``"1"``)。
        顶层 ``<op>_impl.py`` 与其他 phase 的 module impl 都排除在外。
        该模式由 ``--check-phase-gate`` 触发, 让 orchestrator 在
        ``complete_phase(M_k)`` 边界只验证当前 phase 的产出, 不要求
        Stage 5 cleanup 才会产生的整合 impl 已就绪。

    模式 C — 默认 (两个 scope 都为 None)：
        - 顶层集成 impl: ``<op>_impl.py``（如果存在）
        - 分阶段模块 impl: ``modules/<op>_module<k>_impl.py``（按文件名排序）

        Stage 1-4 阶段尚未生成任何 impl 时返回空列表。
        这是 D1/D3/D5 lint 规则的共享 helper, 使每条规则都能在模块开发
        阶段 (Stage 5 Phase M_k) 即时发现违规, 而不是等到 Stage 6
        集成时才暴露。
    """
    op = ctx.op_name
    if ctx.file_scope:
        return _active_impl_file(ctx, ctx.file_scope)
    if ctx.phase_scope:
        suffix = _phase_to_module_suffix(ctx.phase_scope)
        path = f"modules/{op}_module{suffix}_impl.py"
        return _active_impl_file(ctx, path)
    return _default_impl_files(ctx, op)


def _active_impl_file(ctx: CheckContext, path: str) -> list[str]:
    if not ctx.file_exists(path) or _is_pending_stub(ctx, path):
        return []
    return [path]


def _default_impl_files(ctx: CheckContext, op: str) -> list[str]:
    paths: list[str] = []
    paths.extend(_active_impl_file(ctx, f"{op}_impl.py"))
    modules_dir = ctx.file_path("modules")
    if not os.path.isdir(modules_dir):
        return paths
    for entry in sorted(os.listdir(modules_dir)):
        if not entry.startswith(f"{op}_module") or not entry.endswith("_impl.py"):
            continue
        paths.extend(_active_impl_file(ctx, f"modules/{entry}"))
    return paths


def _extract_design_identifiers(content: str) -> set[str]:
    # 仅从“代码相关上下文”提取变量名，减少自然语言文本噪声：
    # 1) markdown fenced code block
    # 2) inline code (`...`)
    code_blocks = re.findall(r"```[\w-]*\n(.*?)```", content, re.S)
    inline_codes = re.findall(r"`([^`\n]+)`", content)
    scoped_text = "\n".join(code_blocks + inline_codes)

    tokens = set(re.findall(r"\b[a-z_][a-z0-9_]{2,}\b", scoped_text))
    blacklist = {
        "input", "output", "tensor", "shape", "dtype", "dynamic", "algorithm",
        "default", "stage", "spec", "design", "report", "loop", "tiling",
        "step", "max", "min", "round", "clamp", "cast", "mul", "add", "sub", "div",
    }
    # 只保留更像“中间变量”的命名（snake_case 优先），减少误报
    return {t for t in tokens if t not in blacklist and ("_" in t or len(t) >= 6)}
