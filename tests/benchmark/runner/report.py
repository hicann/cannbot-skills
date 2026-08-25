#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""评测汇总报告生成 (markdown / html)。

汇总单次评测的通过率、耗时、token 消耗、成本等指标,
从 summary 结果列表生成 report_{run_id}.md 与 report_{run_id}.html。
无第三方依赖, 可被 run_eval.py 调用或独立运行:

  python runner/report.py results/summary.yaml
  python runner/report.py results/summary.yaml --run-id 20260725_120000
"""

import argparse
import datetime
import html
import logging
import os
import sys

import yaml

_log = logging.getLogger(__name__)

STATUS_ICONS = {"success": "✅", "failed": "❌", "timeout": "⏱️", "error": "💥"}


def _now() -> datetime.datetime:
    """带本地时区的当前时间 (G.PSL.02: 避免使用 naive datetime)。"""
    return datetime.datetime.now(datetime.timezone.utc).astimezone()


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds or 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _token_total(tokens) -> int:
    """从 serve info.tokens 字典中提取总 token 数 (字段名各版本不一, 防御式求和)。"""
    if isinstance(tokens, (int, float)):
        return int(tokens)
    if not isinstance(tokens, dict):
        return 0
    for key in ("total", "totalTokens"):
        if isinstance(tokens.get(key), (int, float)):
            return int(tokens[key])
    # 会话级 tokens 无 total 键 (input/output/reasoning + 嵌套 cache),
    # 与消息级 total 口径一致 (total 含 cache), 求和时一并计入
    cache = tokens.get("cache") or {}
    return int(sum(v for v in tokens.values() if isinstance(v, (int, float)))
               + sum(v for v in cache.values() if isinstance(v, (int, float))))


def _safe_float(value) -> float:
    """防御式 float 转换: 脏数据 (非数值等) 归 0, 避免单个算子拖垮整份报告。"""
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _delivery_cell(r: dict) -> str:
    """交付完整性列: True=✅ / False=⚠️ 缺whl / 无该字段 (旧 summary) = -。"""
    ok = r.get("delivery_ok")
    if ok is None:
        return "-"
    return "✅" if ok else "⚠️ 缺whl"


def collect_stats(results: list[dict]) -> dict:
    """汇总统计: 状态计数/通过率/交付完整数/总耗时/总 token/总成本。"""
    stats = {
        "total": len(results),
        "success": 0, "failed": 0, "timeout": 0, "error": 0,
        "delivery_ok": 0, "duration_s": 0.0, "tokens": 0, "cost": 0.0,
    }
    for r in results:
        status = r.get("status", "error")
        stats[status] = stats.get(status, 0) + 1
        stats["duration_s"] += _safe_float(r.get("duration_s"))
        stats["tokens"] += _token_total(r.get("tokens"))
        stats["cost"] += _safe_float(r.get("cost"))
        if r.get("delivery_ok", True):
            stats["delivery_ok"] += 1
    stats["pass_rate"] = (stats["success"] / stats["total"]
                          if stats["total"] else 0.0)
    return stats


def _result_row(r: dict) -> list[str]:
    return [
        r.get("op_name", "?"),
        f"{STATUS_ICONS.get(r.get('status'), '❓')} {r.get('status', '?')}",
        _fmt_duration(_safe_float(r.get("duration_s"))),
        # 算子级尝试次数 (OP_RETRY); 旧结果无该字段时回退 serve 层 attempts
        str(r.get("op_attempt", r.get("attempts", 1))),
        _delivery_cell(r),
        str(_token_total(r.get("tokens"))),
        f"{_safe_float(r.get('cost')):.4f}",
        r.get("model_actual") or r.get("model", "?"),
    ]


_HEADERS = ["算子", "状态", "耗时", "尝试", "交付", "Tokens", "Cost", "实际模型"]


def render_markdown(results: list[dict], meta: dict) -> str:
    stats = collect_stats(results)
    lines = [
        f"# 评测报告 {meta.get('run_id', '')}",
        "",
        f"- **Run ID**: {meta.get('run_id', '?')}",
        f"- **时间**: {meta.get('date', '?')}",
        f"- **模型**: {meta.get('model', '?')}",
        f"- **工作流**: {meta.get('workflow', '?')}",
        "",
        "## 汇总",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 算子总数 | {stats['total']} |",
        f"| 通过 | {stats['success']} |",
        f"| 失败 | {stats.get('failed', 0)} |",
        f"| 超时 | {stats.get('timeout', 0)} |",
        f"| 异常 | {stats.get('error', 0)} |",
        f"| 通过率 | {stats['pass_rate']:.1%} |",
        f"| 交付完整 | {stats['delivery_ok']}/{stats['total']} |",
        f"| 总耗时 | {_fmt_duration(stats['duration_s'])} |",
        f"| 总 Tokens | {stats['tokens']} |",
        f"| 总 Cost | {stats['cost']:.4f} |",
        "",
        "## 明细",
        "",
        "| " + " | ".join(_HEADERS) + " |",
        "|" + "---|" * len(_HEADERS),
    ]
    for r in results:
        lines.append("| " + " | ".join(_result_row(r)) + " |")
    lines.append("")
    return "\n".join(lines)


def _detail_row_html(r: dict, esc) -> str:
    """渲染明细行 <tr> (抽出避免 render_html 内嵌套推导式过深)。"""
    cells = "</td><td>".join(esc(c) for c in _result_row(r))
    status = esc(r.get("status", "error"))
    return f"<tr class='st-{status}'><td>{cells}</td></tr>"


def render_html(results: list[dict], meta: dict) -> str:
    stats = collect_stats(results)
    esc = html.escape
    summary_rows = "".join(
        f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in [
            ("算子总数", stats["total"]),
            ("通过", stats["success"]),
            ("失败", stats.get("failed", 0)),
            ("超时", stats.get("timeout", 0)),
            ("异常", stats.get("error", 0)),
            ("通过率", f"{stats['pass_rate']:.1%}"),
            ("交付完整", f"{stats['delivery_ok']}/{stats['total']}"),
            ("总耗时", _fmt_duration(stats["duration_s"])),
            ("总 Tokens", stats["tokens"]),
            ("总 Cost", f"{stats['cost']:.4f}"),
        ])
    detail_rows = "".join(_detail_row_html(r, esc) for r in results)
    headers = "".join(f"<th>{esc(h)}</th>" for h in _HEADERS)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>评测报告 {esc(meta.get('run_id', ''))}</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", "PingFang SC", sans-serif; margin: 2em; color: #222; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
th {{ background: #f0f0f0; }}
tr.st-success td {{ background: #f0faf0; }}
tr.st-failed td, tr.st-error td {{ background: #fdf0f0; }}
tr.st-timeout td {{ background: #fffbe8; }}
.meta {{ color: #666; }}
</style>
</head>
<body>
<h1>评测报告 {esc(meta.get('run_id', ''))}</h1>
<p class="meta">时间: {esc(meta.get('date', '?'))} |
模型: {esc(meta.get('model', '?'))} |
工作流: {esc(meta.get('workflow', '?'))}</p>
<h2>汇总</h2>
<table>{summary_rows}</table>
<h2>明细</h2>
<table><tr>{headers}</tr>{detail_rows}</table>
</body>
</html>
"""


def generate_report(results: list[dict], run_id: str, model: str,
                    workflow: str, output_dir: str) -> dict:
    """生成 md/html 报告到 output_dir, 返回 {md, html} 路径。"""
    meta = {
        "run_id": run_id,
        "date": _now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "workflow": workflow,
    }
    os.makedirs(output_dir, exist_ok=True)
    md_path = os.path.join(output_dir, f"report_{run_id}.md")
    html_path = os.path.join(output_dir, f"report_{run_id}.html")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(results, meta))
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render_html(results, meta))
    return {"md": md_path, "html": html_path}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="从 summary.yaml 生成评测报告")
    parser.add_argument("summary", help="summary.yaml 路径")
    parser.add_argument("--run-id", default=None,
                        help="报告文件名后缀 (默认: 当前时间)")
    parser.add_argument("--model", default="?")
    parser.add_argument("--workflow", default="?")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="报告输出目录 (默认: summary 所在目录)")
    args = parser.parse_args()

    with open(args.summary) as f:
        results = yaml.safe_load(f)
    if not isinstance(results, list):
        _log.error("错误: %s 不是结果列表", args.summary)
        return 1

    run_id = args.run_id or _now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.summary))
    paths = generate_report(results, run_id, args.model, args.workflow,
                            output_dir)
    _log.info("报告已生成:\n  %s\n  %s", paths["md"], paths["html"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
