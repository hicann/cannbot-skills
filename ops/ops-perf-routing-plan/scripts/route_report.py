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
"""Resolve a frozen classification plan and write route tables plus a report."""

from __future__ import annotations

import argparse
import csv
import html
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    case_variables,
    die,
    emit,
    eval_derived,
    eval_expr,
    load_json,
    resolve_tiling,
    write_json,
)


def _matches(
    template: Dict[str, Any], case: Dict[str, Any], state: Dict[str, Any]
) -> bool:
    try:
        ns = eval_derived(
            template.get("derived") or {},
            case_variables(case, state.get("hardware") or {}),
        )
        return bool(eval_expr(template.get("predicate", "False"), ns))
    except Exception:
        return False


def _planned_template(case, templates, state):
    for template in templates:
        if _matches(template, case, state):
            return template
    return None


def _row(cid, planned, tiling=None, errors=None):
    errors = errors or []
    name = planned.get("name") if planned else "original"
    return {
        "case_id": cid,
        "planned_template": name,
        "effective_template": name,
        "tiling_key": (tiling or {}).get(
            "tiling_key", planned.get("tiling_key", 0) if planned else 0
        ),
        "num_cores": (tiling or {}).get("num_cores"),
        "tile_elems": (tiling or {}).get("tile_elems"),
        "tail_elems": (tiling or {}).get("tail_elems", "-"),
        "tail_policy": (tiling or {}).get("tail_policy", "-"),
        "unrouted": planned is None,
        "ok": not errors,
        "errors": errors,
    }


def _planned_row(cid, case, planned, state, errors):
    try:
        tiling = resolve_tiling(case, planned, state.get("hardware") or {})
        row_errors = list(tiling.get("errors") or [])
        if row_errors:
            errors.extend(f"{cid}/{planned.get('name')}: {e}" for e in row_errors)
        return _row(cid, planned, tiling, row_errors)
    except Exception as exc:
        errors.append(f"{cid}/{planned.get('name')}: {exc}")
        return _row(cid, planned, errors=[str(exc)])


def _build_routes(templates):
    routes = []
    for template in templates:
        routes.append(
            {
                "priority": template.get("priority"),
                "template": template.get("name"),
                "predicate": template.get("predicate"),
                "derived": template.get("derived") or {},
                "tiling_key": template.get("tiling_key"),
                "representative_case_id": template.get("representative_case_id"),
                "is_single_case_group": bool(template.get("is_single_case_group")),
                "cases": template.get("cases", []),
            }
        )
    return routes


def _build_table(state, templates, all_ids, by_id):
    table, counts, errors = [], {}, []
    for cid in all_ids:
        case = by_id.get(str(cid))
        if case is None:
            continue
        planned = _planned_template(case, templates, state)
        if planned is None:
            row = _row(cid, None)
            errors.append(f"{cid}: 冻结计划中没有任何模板命中")
        else:
            row = _planned_row(cid, case, planned, state, errors)
        counts[row["effective_template"]] = counts.get(row["effective_template"], 0) + 1
        table.append(row)
    return table, counts, errors


def _write_table(out_dir, state, routes, table):
    os.makedirs(out_dir, exist_ok=True)
    write_json(
        os.path.join(out_dir, "tiling_routes.json"),
        {
            "schema_version": 1,
            "operator": state.get("operator"),
            "stage": state.get("stage", ""),
            "routes": routes,
            "default": {"template": "original", "tiling_key": 0},
            "case_to_template": {r["case_id"]: r["effective_template"] for r in table},
        },
    )
    fields = [
        "case_id",
        "planned_template",
        "effective_template",
        "tiling_key",
        "num_cores",
        "tile_elems",
        "tail_elems",
        "tail_policy",
        "unrouted",
        "ok",
        "errors",
    ]
    with open(
        os.path.join(out_dir, "tiling_table.csv"), "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in table:
            record = dict(row)
            record["errors"] = "; ".join(row.get("errors") or [])
            writer.writerow(record)


def _build_coverage(state, out_dir, table, all_ids, counts, errors):
    unrouted = []
    for row in table:
        if row["unrouted"]:
            unrouted.append(row["case_id"])
    coverage = {
        "schema_version": 1,
        "operator": state.get("operator"),
        "stage": state.get("stage", ""),
        "n_cases": len(table),
        "planned_coverage": round(len(table) / max(1, len(all_ids)), 4),
        "effective_counts": counts,
        "unrouted_cases": unrouted,
        "n_unrouted": len(unrouted),
        "errors": errors,
        "ok": len(table) == len(all_ids) and not unrouted and not errors,
    }
    write_json(os.path.join(out_dir, "coverage_report.json"), coverage)
    return coverage


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _table(headers: List[str], rows: List[List[Any]]) -> List[str]:
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        out.append("| " + " | ".join("" if x is None else str(x) for x in row) + " |")
    return out


def _overview_lines(state, coverage, work):
    return [
        f"# {state.get('operator', '算子')} Shape 分类与 tiling 路由报告",
        "",
        "## 0. 概览",
        "",
        f"- 工作目录：`{work}`",
        f"- 分类阶段：`{state.get('stage', '-')}`",
        f"- shape 总数：{len(state.get('all_cases', []))}",
        f"- 模板数：{len(state.get('templates', []))}",
        f"- 未覆盖 shape：{len(state.get('remaining', []))}",
        f"- 路由覆盖：{coverage.get('planned_coverage', '-')}",
        f"- 未路由 shape：{coverage.get('n_unrouted', '-')}",
        "",
    ]


def _limits_lines(limits):
    rows = []
    for case in limits.get("cases", []):
        rows.append(
            [
                case.get("rank"),
                case.get("case_id"),
                case.get("headroom"),
                case.get("bound_kind"),
                case.get("t_candidate_us"),
                case.get("t_vector_us"),
                case.get("t_copy_us"),
                case.get("t_hw_us"),
            ]
        )
    lines = ["## 1. 理论排名", ""]
    lines += _table(
        [
            "排名",
            "case",
            "headroom",
            "bound",
            "实测 us",
            "vector us",
            "copy us",
            "理论 us",
        ],
        rows,
    )
    return lines + [""]


def _templates_lines(state):
    rows = []
    for template in state.get("templates", []):
        rows.append(
            [
                template.get("priority"),
                template.get("name"),
                template.get("tiling_key"),
                template.get("representative_case_id"),
                len(template.get("cases", [])),
                "yes" if template.get("is_single_case_group") else "no",
                template.get("predicate"),
            ]
        )
    lines = ["## 2. Shape 分组", ""]
    lines += _table(
        [
            "优先级",
            "模板",
            "tiling_key",
            "代表 shape",
            "case 数",
            "单 shape 组",
            "谓词",
        ],
        rows,
    )
    return lines + [""]


def _routes_lines(routes):
    lines = ["## 3. 逐 case 路由", ""]
    if routes.get("routes"):
        rows = []
        for cid, name in routes.get("case_to_template", {}).items():
            rows.append([cid, name])
        lines += _table(["case", "effective template"], rows)
    else:
        lines.append("> 尚未生成 `tiling_routes.json`。")
    return lines + [""]


def _grouping_lines(state):
    lines = ["## 4. 分组质量提示", ""]
    single = []
    for template in state.get("templates", []):
        if template.get("is_single_case_group"):
            single.append(template)
    if not single:
        lines.append("- 所有模板均覆盖至少 2 个 shape。")
    for template in single:
        lines.append(
            f"- `{template.get('name')}` 仅覆盖 "
            f"`{template.get('representative_case_id')}`，未形成有效分组；"
            "不得将该结果外推到不同特征的 shape。"
        )
    return lines + [""]


def _report_text(state, limits, routes, coverage, work):
    lines = _overview_lines(state, coverage, work)
    lines += _limits_lines(limits)
    lines += _templates_lines(state)
    lines += _routes_lines(routes)
    lines += _grouping_lines(state)
    return "\n".join(lines)


def _write_report(out_dir, state, limits, routes, coverage, work):
    text = _report_text(state, limits, routes, coverage, work)
    md = os.path.join(out_dir, "classification_report.md")
    with open(md, "w", encoding="utf-8") as handle:
        handle.write(text)
    html_path = os.path.join(out_dir, "classification_report.html")
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{html.escape(str(state.get('operator', 'shape classification')))}</title>"
            "<style>body{font-family:Menlo,Consolas,monospace;background:#1e1e1e;color:#d4d4d4;"
            "padding:24px}h1,h2{color:#4fc1ff}pre{white-space:pre-wrap;background:#252526;padding:12px;"
            "border-radius:6px}</style></head><body><pre>"
            + html.escape(text)
            + "</pre></body></html>"
        )
    return md, html_path


def _derive_paths(state_path: str):
    state_dir = os.path.dirname(os.path.abspath(state_path))
    if (
        os.path.basename(state_dir) == "coverage"
        and os.path.basename(os.path.dirname(state_dir)) == "routing"
    ):
        routing_dir = os.path.dirname(state_dir)
        return routing_dir, os.path.dirname(routing_dir)
    return state_dir, state_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="解析冻结计划并生成路由表/分类报告")
    ap.add_argument("--state", required=True, help="冻结后的 state.json 或 plan.json")
    ap.add_argument("--out-dir", default="", help="缺省为 plan.json 所在 routing/")
    ap.add_argument("--work", default="", help="缺省从 routing/coverage 路径推导")
    ap.add_argument("--limits", default="", help="theoretical/shape_limits.json")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()
    state = load_json(args.state)
    if not state:
        die(f"state 不存在: {args.state}")
    if state.get("stage") != "frozen":
        die("state 尚未冻结；请先执行 greedy_route.py freeze")
    derived_out, derived_work = _derive_paths(args.state)
    out_dir = os.path.abspath(args.out_dir or derived_out)
    work = os.path.abspath(args.work or derived_work)
    limits = (load_json(args.limits) if args.limits else {}) or {}
    by_id = {str(c["case_id"]): c for c in state.get("cases", [])}
    templates = state.get("templates", [])
    routes = _build_routes(templates)
    all_ids = state.get("all_cases") or list(by_id)
    table, counts, errors = _build_table(state, templates, all_ids, by_id)
    _write_table(out_dir, state, routes, table)
    coverage = _build_coverage(state, out_dir, table, all_ids, counts, errors)
    if not args.no_report:
        routes_doc = load_json(os.path.join(out_dir, "tiling_routes.json")) or {}
        _write_report(out_dir, state, limits, routes_doc, coverage, work)
    emit(f"[route_report] {len(table)} cases → {out_dir}")
    emit(
        f"  templates: {counts}; unrouted={coverage['n_unrouted']}; "
        f"errors={len(errors)}"
    )
    return 0 if coverage["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
