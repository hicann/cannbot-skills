#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Greedy shape-to-template routing state machine.

Commands:
  init          initialize coverage state from shape_limits.json
  next          print the current worst uncovered shape
  validate      validate one LLM-proposed template (no state mutation)
  commit        validate tiling + add template + remove covered shapes
  freeze        validate full coverage and freeze the classification plan
  status        print coverage summary
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    ExprError,
    LOGGER,
    emit,
    case_variables,
    die,
    eval_derived,
    eval_expr,
    load_json,
    read_cases,
    sanitize_name,
    validate_template,
    write_json,
)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _case_vars(case: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    return case_variables(case, state.get("hardware") or {})


def _case_index(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(c["case_id"]): c for c in state.get("cases", [])}


def _current_worst(state: Dict[str, Any]) -> Any:
    rem = state.get("remaining") or []
    return rem[0] if rem else None


def _validate_identity(state, proposal):
    if not isinstance(proposal, dict):
        die("proposal 必须是 JSON 对象")
    worst = _current_worst(state)
    if worst is None:
        die("remaining 为空，没有可路由的 shape")
    worst_case_id = str(proposal.get("worst_case_id", ""))
    if worst_case_id != str(worst):
        die(
            f"最差 shape 必须是当前 remaining 的第一个: {worst}，proposal 给的是 {worst_case_id}"
        )
    raw_name = str(proposal.get("template_name", "")).strip()
    name = sanitize_name(raw_name)
    if not name or not raw_name:
        die("template_name 为空或非法")
    if name in {str(t.get("name")) for t in state.get("templates", [])}:
        die(f"模板名已存在: {name}")
    predicate = str(proposal.get("predicate", "")).strip()
    if not predicate:
        die("predicate 不能为空")
    derived = proposal.get("derived") or {}
    if not isinstance(derived, dict):
        die("derived 必须是对象")
    return worst, name, raw_name, predicate, derived


def _evaluate_cases(state, predicate, derived):
    index = _case_index(state)
    selected, checks = [], []
    for cid in state.get("remaining", []):
        case = index.get(str(cid))
        if case is None:
            continue
        try:
            namespace = eval_derived(derived, _case_vars(case, state))
            hit = eval_expr(predicate, namespace)
        except ExprError as exc:
            die(f"case {cid} 谓词求值失败: {exc}")
        if not isinstance(hit, bool):
            die(f"case {cid} 的 predicate 不是布尔值: {hit!r}")
        if hit:
            selected.append(str(cid))
        checks.append({"case_id": str(cid), "hit": hit})
    if not selected:
        die(f"predicate 没有命中任何 remaining shape: {predicate}")
    if len(set(selected)) != len(selected):
        die("selected case 存在重复")
    return selected, checks


def _validate_proposal(state, proposal):
    worst, name, raw_name, predicate, derived = _validate_identity(state, proposal)
    selected, checks = _evaluate_cases(state, predicate, derived)
    if str(worst) not in selected:
        die(f"predicate 没有命中当前最差 shape {worst}")
    return {
        "template_name": name,
        "raw_template_name": raw_name,
        "worst_case_id": str(worst),
        "predicate": predicate,
        "derived": derived,
        "feature_summary": proposal.get("feature_summary", ""),
        "tiling_intent": proposal.get("tiling_intent") or {},
        "tiling_plan": proposal.get("tiling_plan") or {},
        "expected_bottleneck": proposal.get("expected_bottleneck", ""),
        "evidence": proposal.get("evidence", ""),
        "kernel_file": str(proposal.get("kernel_file", "") or ""),
        "selected_cases": selected,
        "case_checks": checks,
    }


def cmd_init(args: argparse.Namespace) -> int:
    limits = load_json(args.limits)
    if not limits:
        die(f"shape limits 不存在: {args.limits}")
    hardware = (
        load_json(args.hardware) if args.hardware else (limits.get("hardware") or {})
    )
    if not hardware:
        die("缺少 hardware profile")
    operator = args.operator or limits.get("operator") or ""
    cases = read_cases(args.cases, operator)
    case_ids = {str(c["case_id"]) for c in cases}
    remaining, seen = [], set()
    for case in limits.get("cases", []):
        cid = str(case["case_id"])
        if cid in case_ids and cid not in seen:
            seen.add(cid)
            remaining.append(cid)
    if seen != case_ids:
        missing = sorted(case_ids - seen)
        die(f"shape_limits 未覆盖全部 case: {missing[:8]}")
    all_cases = []
    for case in cases:
        all_cases.append(str(case["case_id"]))
    state = {
        "schema_version": 1,
        "operator": operator,
        "created_at": _now(),
        "hardware": hardware,
        "cases": cases,
        "all_cases": all_cases,
        "remaining": remaining,
        "templates": [],
        "case_to_template": {},
        "next_tiling_key": int(args.tiling_key_base),
        "tiling_key_base": int(args.tiling_key_base),
        "stage": "classifying",
    }
    write_json(args.output, state)
    emit(
        f"[greedy_route] 初始化 {len(cases)} 个 shape，最差 {remaining[0] if remaining else '-'} → {args.output}"
    )
    return 0


def _load_state(path: str) -> Dict[str, Any]:
    state = load_json(path)
    if not isinstance(state, dict):
        die(f"state 不存在或非法: {path}")
    return state


def _write_state(path: str, state: Dict[str, Any]) -> None:
    state["updated_at"] = _now()
    write_json(path, state)


def cmd_next(args: argparse.Namespace) -> int:
    state = _load_state(args.state)
    worst = _current_worst(state)
    if worst is None:
        emit(json.dumps({"remaining": 0, "worst_case_id": None}, ensure_ascii=False))
        return 0
    idx = _case_index(state)
    emit(
        json.dumps(
            {
                "remaining": len(state.get("remaining", [])),
                "worst_case_id": str(worst),
                "case": idx.get(str(worst)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    state = _load_state(args.state)
    proposal = load_json(args.proposal)
    result = _validate_proposal(state, proposal)
    if args.output:
        write_json(args.output, result)
    else:
        emit(json.dumps(result, ensure_ascii=False, indent=2))
    emit(
        f"[greedy_route] 模板 {result['template_name']} 命中 {len(result['selected_cases'])} 个 shape"
    )
    return 0


def _tiling_validation(state, result):
    index = _case_index(state)
    selected = []
    for cid in result["selected_cases"]:
        key = str(cid)
        if key in index:
            selected.append(index[key])
    template = {
        "name": result["template_name"],
        "tiling_key": int(state.get("next_tiling_key", 1)),
        "derived": result["derived"],
        "tiling_intent": result["tiling_intent"],
        "tiling_plan": result["tiling_plan"],
    }
    return validate_template(template, selected, state.get("hardware") or {})


def _new_template(state, result, validation):
    cases = result["selected_cases"]
    return {
        "name": result["template_name"],
        "priority": len(state.get("templates", [])) + 1,
        "tiling_key": int(state.get("next_tiling_key", 1)),
        "feature_summary": result["feature_summary"],
        "predicate": result["predicate"],
        "derived": result["derived"],
        "tiling_intent": result["tiling_intent"],
        "tiling_plan": result.get("tiling_plan") or {},
        "expected_bottleneck": result["expected_bottleneck"],
        "evidence": result["evidence"],
        "kernel_file": result.get("kernel_file", ""),
        "cases": cases,
        "representative_case_id": result["worst_case_id"],
        "is_single_case_group": len(cases) == 1,
        "tiling_validation": {
            "ok": bool(validation.get("ok")),
            "n_cases": len(validation.get("cases", [])),
        },
        "committed_at": _now(),
    }


def _apply_commit(state, template, result):
    state.setdefault("templates", []).append(template)
    selected = set(result["selected_cases"])
    remaining = []
    for case_id in state.get("remaining", []):
        if str(case_id) not in selected:
            remaining.append(case_id)
    state["remaining"] = remaining
    state.setdefault("case_to_template", {})
    for cid in result["selected_cases"]:
        state["case_to_template"][str(cid)] = result["template_name"]
    state["next_tiling_key"] = int(template["tiling_key"]) + 1


def cmd_commit(args: argparse.Namespace) -> int:
    state = _load_state(args.state)
    result = _validate_proposal(state, load_json(args.proposal))
    validation = _tiling_validation(state, result)
    if not validation.get("ok"):
        LOGGER.error("[greedy_route][ERROR] 模板 tiling 校验未通过，拒绝 commit：")
        for problem in validation.get("errors", []):
            LOGGER.info("  - " + problem)
        return 2
    template = _new_template(state, result, validation)
    _apply_commit(state, template, result)
    _write_state(args.state, state)
    emit(
        f"[greedy_route] committed {template['name']} "
        f"({len(result['selected_cases'])} shapes)；"
        f"remaining={len(state['remaining'])}"
    )
    return 0


def _remaining_problems(state: Dict[str, Any]) -> List[str]:
    remaining = list(map(str, state.get("remaining", [])))
    if not remaining:
        return []
    return [f"remaining 非空，仍有 {len(remaining)} 个 shape 未归类: {remaining[:8]}"]


def _template_identity(template, problems):
    name = str(template.get("name", ""))
    key, priority = None, None
    try:
        key = int(template.get("tiling_key"))
    except Exception:
        problems.append(
            f"模板 {name!r} 的 tiling_key 非法: {template.get('tiling_key')!r}"
        )
    try:
        priority = int(template.get("priority"))
    except Exception:
        problems.append(f"模板 {name!r} 的 priority 非法: {template.get('priority')!r}")
    return name, key, priority


def _template_problems(template, name, cases) -> List[str]:
    problems = []
    if not cases:
        problems.append(f"模板 {name!r} 没有 cases")
    representative = str(template.get("representative_case_id") or "")
    if not representative:
        problems.append(f"模板 {name!r} 缺少 representative_case_id")
    elif representative not in cases:
        problems.append(
            f"模板 {name!r} 的 representative_case_id={representative!r} "
            "不属于本组 cases"
        )
    if bool(template.get("is_single_case_group")) != (len(cases) == 1):
        problems.append(f"模板 {name!r} 的 is_single_case_group 与 cases 数不一致")
    if not (template.get("tiling_validation") or {}).get("ok"):
        problems.append(f"模板 {name!r} 没有通过的 tiling_validation")
    return problems


def _membership_problems(name, cases, all_cases, assigned) -> List[str]:
    problems = []
    for cid in cases:
        if cid not in all_cases:
            problems.append(f"模板 {name!r} 包含未知 case: {cid}")
        if cid in assigned:
            problems.append(f"case {cid} 同时属于模板 {assigned[cid]!r} 和 {name!r}")
        assigned[cid] = name
    return problems


def _uniqueness_problems(names, keys, priorities) -> List[str]:
    problems = []
    if len(set(names)) != len(names):
        problems.append("模板名存在重复")
    if len(set(keys)) != len(keys):
        problems.append("tiling_key 存在重复")
    if len(set(priorities)) != len(priorities):
        problems.append("priority 存在重复")
    return problems


def _plan_problems(state: Dict[str, Any]) -> List[str]:
    problems = _remaining_problems(state)
    templates = state.get("templates") or []
    if not templates:
        problems.append("没有任何模板")
    names, keys, priorities = [], [], []
    all_cases = {str(x) for x in state.get("all_cases", [])}
    assigned: Dict[str, str] = {}
    for template in templates:
        name, key, priority = _template_identity(template, problems)
        cases = list(map(str, template.get("cases", [])))
        names.append(name)
        if key is not None:
            keys.append(key)
        if priority is not None:
            priorities.append(priority)
        problems += _template_problems(template, name, cases)
        problems += _membership_problems(name, cases, all_cases, assigned)
    problems += _uniqueness_problems(names, keys, priorities)
    missing = sorted(all_cases - set(assigned))
    if missing:
        problems.append(f"仍有 case 未归属: {missing[:8]}")
    return problems


def cmd_freeze(args: argparse.Namespace) -> int:
    state = _load_state(args.state)
    problems = _plan_problems(state)
    if problems:
        LOGGER.error("[greedy_route][ERROR] 归类计划未冻结：")
        for p in problems:
            LOGGER.info("  - " + p)
        return 2
    state["stage"] = "frozen"
    state["frozen_at"] = _now()
    out = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.state)), "plan.json"
    )
    state["plan_path"] = os.path.abspath(out)
    write_json(out, state)
    _write_state(args.state, state)
    emit(
        f"[greedy_route] frozen {len(state.get('templates', []))} templates, "
        f"{len(state.get('all_cases', []))} cases → {out}"
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    state = _load_state(args.state)
    template_rows = []
    for template in state.get("templates", []):
        template_rows.append(
            {
                "name": template.get("name"),
                "tiling_key": template.get("tiling_key"),
                "n_cases": len(template.get("cases", [])),
                "representative_case_id": template.get("representative_case_id"),
                "is_single_case_group": bool(template.get("is_single_case_group")),
                "predicate": template.get("predicate"),
            }
        )
    report = {
        "operator": state.get("operator"),
        "n_cases": len(state.get("all_cases", [])),
        "n_remaining": len(state.get("remaining", [])),
        "remaining": state.get("remaining", []),
        "templates": template_rows,
        "coverage": round(
            1.0
            - len(state.get("remaining", [])) / max(1, len(state.get("all_cases", []))),
            4,
        ),
    }
    emit(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _add_init_commands(sub):
    p = sub.add_parser("init")
    p.add_argument("--cases", required=True)
    p.add_argument("--limits", required=True)
    p.add_argument("--hardware", default="")
    p.add_argument("--operator", default="")
    p.add_argument(
        "--tiling-key-base",
        type=int,
        default=1,
        help="模板 tiling_key 起始值（默认 1）",
    )
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_init)
    p = sub.add_parser("next")
    p.add_argument("--state", required=True)
    p.set_defaults(func=cmd_next)
    p = sub.add_parser("validate")
    p.add_argument("--state", required=True)
    p.add_argument("--proposal", required=True)
    p.add_argument("-o", "--output", default="")
    p.set_defaults(func=cmd_validate)


def _add_commit_commands(sub):
    p = sub.add_parser("commit")
    p.add_argument("--state", required=True)
    p.add_argument("--proposal", required=True)
    p.set_defaults(func=cmd_commit)
    p = sub.add_parser("freeze")
    p.add_argument("--state", required=True)
    p.add_argument(
        "-o", "--output", default="", help="冻结计划快照（缺省 state 同目录 plan.json）"
    )
    p.set_defaults(func=cmd_freeze)


def _add_status_command(sub):
    p = sub.add_parser("status")
    p.add_argument("--state", required=True)
    p.set_defaults(func=cmd_status)


def _build_parser():
    ap = argparse.ArgumentParser(description="贪心 shape → tiling 模板路由状态机")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add_init_commands(sub)
    _add_commit_commands(sub)
    _add_status_command(sub)
    return ap


def main() -> int:
    args = _build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
