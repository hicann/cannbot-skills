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
validate_contract.py — algo_flow.json 合规性校验器

用法:
    python3 validate_contract.py <algo_flow.json>          # 校验
    python3 validate_contract.py <algo_flow.json> --fix    # 校验 + 尝试自动修复

退出码:
    0 — 通过
    1 — 失败（输出错误列表）

设计原则：
    契约优先 (contract-first)。gen_dashboard.py 渲染前先用本脚本校验，
    确保 Claude 生成的 algo_flow.json 符合 contracts/algo_flow.schema.json。
"""

import json
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_TOP = ["op_name", "shape_vars", "units"]
REQUIRED_UNIT = ["id", "label", "bg", "accent", "nodes"]
REQUIRED_NODE = ["api", "formula", "in_tmpl", "out_tmpl"]
# vector: 纯 AIV 算子；cube: 纯 AIC 算子
VALID_UNIT_IDS = {"vector", "cube"}

# 从 design_tokens.json 读取 unit 颜色/label（单一真源），fallback 到硬编码


def _load_unit_defaults() -> dict:
    tokens_path = Path(__file__).parent.parent / "design_tokens.json"
    try:
        tokens = json.loads(tokens_path.read_text())
        c = tokens.get("colors", {})
        return {
            "vector": {"label": "Vector Core  Compute",
                       "bg": c.get("unit_vector_bg", "#f5f0ff"),
                       "accent": c.get("unit_vector_accent", "#8250df")},
            "cube": {"label": "Cube Core  MAC",
                     "bg": c.get("unit_cube_bg", "#fffbf0"),
                     "accent": c.get("unit_cube_accent", "#bf8700")},
        }
    except Exception:
        return {
            "vector": {"label": "Vector Core  Compute", "bg": "#f5f0ff", "accent": "#8250df"},
            "cube": {"label": "Cube Core  MAC", "bg": "#fffbf0", "accent": "#bf8700"},
        }


UNIT_DEFAULTS = _load_unit_defaults()

_TMPL_VAR = re.compile(r'\{(\w+)\}')


def validate(data: dict) -> list:
    errors = []

    # 1. 顶层必填字段
    for f in REQUIRED_TOP:
        if f not in data:
            errors.append(f"[TOP] 缺少必填字段: `{f}`")
    if errors:
        return errors  # 无法继续

    shape_vars = data["shape_vars"]
    if not isinstance(shape_vars, list) or not all(isinstance(v, str) for v in shape_vars):
        errors.append("[TOP] `shape_vars` 必须是字符串数组，如 [\"N\", \"C\"]")
        shape_vars = []

    units = data.get("units", [])
    if not isinstance(units, list) or len(units) == 0:
        errors.append("[TOP] `units` 为空，至少需要一个硬件单元（vector 或 cube）")
        return errors

    for ui, unit in enumerate(units):
        pfx = f"[units[{ui}]]"

        # 2. Unit 必填字段
        for f in REQUIRED_UNIT:
            if f not in unit:
                errors.append(f"{pfx} 缺少字段 `{f}`")

        # 3. Unit id 合法性
        uid = unit.get("id", "")
        if uid not in VALID_UNIT_IDS:
            errors.append(f"{pfx} `id`='{uid}' 非法，必须是 {sorted(VALID_UNIT_IDS)} 之一")

        # 4. 颜色格式
        for cfield in ["bg", "accent"]:
            val = unit.get(cfield, "")
            if val and not re.fullmatch(r'#[0-9a-fA-F]{6}', val):
                errors.append(f"{pfx} `{cfield}`='{val}' 不是合法 hex 颜色（如 #8250df）")

        nodes = unit.get("nodes", [])
        if not isinstance(nodes, list) or len(nodes) == 0:
            errors.append(f"{pfx} `nodes` 为空，至少需要一个 API 节点")
            continue

        for ni, node in enumerate(nodes):
            npfx = f"[units[{ui}].nodes[{ni}]]"

            # 5. Node 必填字段
            for f in REQUIRED_NODE:
                if f not in node:
                    errors.append(f"{npfx} 缺少字段 `{f}`")

            # 6. 模板变量一致性：in_tmpl/out_tmpl 中的变量必须在 shape_vars 中
            if shape_vars:
                for tfield in ["in_tmpl", "out_tmpl"]:
                    tmpl = node.get(tfield, "")
                    used = set(_TMPL_VAR.findall(tmpl))
                    unknown = used - set(shape_vars)
                    if unknown:
                        errors.append(
                            f"{npfx} `{tfield}`='{tmpl}' 使用了未声明的模板变量 {sorted(unknown)}，"
                            f"请将其添加到顶层 `shape_vars`"
                        )

            # 7. API 不应包含 DataCopy（计算流图不含内存搬运）
            api = node.get("api", "")
            if api == "DataCopy":
                errors.append(
                    f"{npfx} `api`='DataCopy' 不允许出现在计算流图中（内存搬运不是计算节点）"
                )

    return errors


def try_fix(data: dict) -> tuple:
    """尝试自动修复常见问题，返回 (fixed_data, fix_log)"""
    fixes = []
    for unit in data.get("units", []):
        uid = unit.get("id", "")
        if uid in UNIT_DEFAULTS:
            for k, v in UNIT_DEFAULTS[uid].items():
                if k not in unit:
                    unit[k] = v
                    fixes.append(f"units[id={uid}]: 自动填充 `{k}`='{v}'")
    return data, fixes


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        logger.info(__doc__)
        sys.exit(0)

    path = Path(args[0])
    do_fix = "--fix" in args

    if not path.exists():
        logger.error(f"错误：文件不存在 {path}")
        sys.exit(1)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析错误：{e}")
        sys.exit(1)

    errors = validate(data)

    if not errors:
        n_units = len(data.get("units", []))
        n_nodes = sum(len(u.get("nodes", [])) for u in data.get("units", []))
        logger.info(f"✓ {path.name} — 校验通过（{n_units} 个单元，{n_nodes} 个节点）")
        sys.exit(0)

    logger.info(f"✗ {path.name} — 发现 {len(errors)} 个问题：")
    for i, e in enumerate(errors, 1):
        logger.info(f"  {i}. {e}")

    if do_fix:
        fixed, fix_log = try_fix(data)
        errors_after = validate(fixed)
        if fix_log:
            logger.info(f"\n自动修复（{len(fix_log)} 项）：")
            for f in fix_log:
                logger.info(f"  · {f}")
        if len(errors_after) < len(errors):
            fixed_path = path.with_stem(path.stem + "_fixed")
            fixed_path.write_text(json.dumps(fixed, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"\n修复后仍有 {len(errors_after)} 个问题，已保存到 {fixed_path}")
        else:
            logger.info("\n自动修复无法解决剩余问题，请手动修改。")

    sys.exit(1)


if __name__ == "__main__":
    main()
