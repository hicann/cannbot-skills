#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FINISH-phase report generator — produces .ar_state/report.md with summary
tables and an inline SVG optimization curve.

Stdlib only — no matplotlib, no numpy. The SVG is embedded directly so the
report is a self-contained Markdown file (renders in VS Code / GitHub).

Usage:
    python report.py <task_dir>          # write .ar_state/report.md
    python report.py <task_dir> --print  # dump to stdout (debug)
"""

import argparse
import os
import sys
from dataclasses import dataclass
from html import escape as _h
from typing import Optional

from op_autoresearch.utils.console import emit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase_machine import history_path, load_progress
from task_config import load_task_config
from utils.json_io import load_jsonl

REPORT_FILE = "report.md"


def report_path(task_dir: str) -> str:
    return os.path.join(task_dir, ".ar_state", REPORT_FILE)


def _load_history(task_dir: str) -> list[dict]:
    return load_jsonl(history_path(task_dir))


def _escape_md_cell(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _fmt_num(v: float) -> str:
    av = abs(v)
    if av >= 1000:
        return f"{v:.0f}"
    if av >= 10:
        return f"{v:.1f}"
    if av >= 1:
        return f"{v:.2f}"
    return f"{v:.3g}"


@dataclass(frozen=True)
class SvgRequest:
    history: list[dict]
    primary: str
    lower_is_better: bool
    ref_val: Optional[float]
    ref_label: str
    task_name: str


@dataclass
class PlotSeries:
    keep_rounds: list[int]
    keep_values: list[float]
    discard_rounds: list[int]
    discard_values: list[float]
    fail_rounds: list[int]
    fail_values: list[Optional[float]]
    best_rounds: list[int]
    best_values: list[float]
    speedups: dict[int, float]

    @property
    def all_values(self) -> list[float]:
        return (
            self.keep_values
            + self.discard_values
            + [value for value in self.fail_values if value is not None]
        )

    @property
    def all_rounds(self) -> list[int]:
        return self.keep_rounds + self.discard_rounds + self.fail_rounds


@dataclass(frozen=True)
class ChartBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    width: int = 900
    height: int = 420
    left: int = 70
    right: int = 170
    top: int = 40
    bottom: int = 60

    @property
    def plot_width(self) -> int:
        return self.width - self.left - self.right

    @property
    def plot_height(self) -> int:
        return self.height - self.top - self.bottom

    def sx(self, value: float) -> float:
        return (
            self.left
            + (value - self.x_min)
            / (self.x_max - self.x_min)
            * self.plot_width
        )

    def sy(self, value: float) -> float:
        return (
            self.top
            + (1 - (value - self.y_min) / (self.y_max - self.y_min))
            * self.plot_height
        )


def _generate_svg(request: SvgRequest) -> str:
    """Render the optimization curve as inline SVG."""
    series = _collect_plot_series(request)
    if not series.all_values and not series.fail_rounds:
        return ""
    bounds = _chart_bounds(series, request.ref_val)
    parts: list[str] = []
    _render_svg_axes(parts, request, bounds)
    _render_reference_line(parts, request, bounds)
    _render_best_line(parts, series, bounds)
    _render_measurements(parts, series, bounds)
    _render_speedup_annotations(parts, series, bounds, request.ref_val)
    _render_failure_markers(parts, series, bounds)
    _render_legend(parts, request, series, bounds)
    parts.append("</svg>")
    return "\n".join(parts)


def _collect_plot_series(request: SvgRequest) -> PlotSeries:
    series = PlotSeries([], [], [], [], [], [], [], [], {})
    current_best: Optional[float] = None
    for record in request.history:
        round_id = record.get("round")
        if round_id is None:
            continue
        metrics = record.get("metrics", {})
        value = metrics.get(request.primary)
        speedup = metrics.get("speedup_vs_ref")
        if isinstance(speedup, (int, float)) and speedup > 0:
            series.speedups[int(round_id)] = float(speedup)
        decision = record.get("decision", "")
        if decision == "FAIL":
            series.fail_rounds.append(round_id)
            series.fail_values.append(value)
            continue
        if value is None:
            continue
        current_best = _record_measurement(
            series,
            (round_id, float(value), decision),
            current_best,
            request.lower_is_better,
        )
    return series


def _record_measurement(
    series: PlotSeries,
    measurement: tuple[int, float, str],
    current_best: Optional[float],
    lower_is_better: bool,
) -> Optional[float]:
    round_id, value, decision = measurement
    if decision in ("KEEP", "SEED"):
        series.keep_rounds.append(round_id)
        series.keep_values.append(value)
        if current_best is None:
            current_best = value
        elif lower_is_better:
            current_best = min(current_best, value)
        else:
            current_best = max(current_best, value)
    elif decision == "DISCARD":
        series.discard_rounds.append(round_id)
        series.discard_values.append(value)
    if current_best is not None and decision in ("KEEP", "SEED", "DISCARD"):
        series.best_rounds.append(round_id)
        series.best_values.append(current_best)
    return current_best


def _chart_bounds(
    series: PlotSeries,
    reference: Optional[float],
) -> ChartBounds:
    x_max = max(series.all_rounds) if series.all_rounds else 1
    x_max = max(1, x_max)
    values = list(series.all_values)
    if reference is not None:
        values.append(reference)
    if not values:
        return ChartBounds(0, x_max, 0.0, 1.0)
    y_min, y_max = min(values), max(values)
    span = y_max - y_min if y_max > y_min else max(abs(y_max) * 0.1, 1.0)
    return ChartBounds(
        0,
        x_max,
        y_min - span * 0.05,
        y_max + span * 0.10,
    )


def _render_svg_axes(
    parts: list[str],
    request: SvgRequest,
    bounds: ChartBounds,
) -> None:
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{bounds.width}" '
        f'height="{bounds.height}" viewBox="0 0 {bounds.width} {bounds.height}" '
        'font-family="sans-serif" font-size="11">'
    )
    direction = "lower is better" if request.lower_is_better else "higher is better"
    parts.append(
        f'<text x="{bounds.width / 2:.1f}" y="22" text-anchor="middle" '
        f'font-size="13" font-weight="bold">'
        f'{_h(f"{request.task_name} — {request.primary} ({direction})")}</text>'
    )
    parts.append(
        f'<rect x="{bounds.left}" y="{bounds.top}" '
        f'width="{bounds.plot_width}" height="{bounds.plot_height}" '
        'fill="white" stroke="#888" stroke-width="0.5"/>'
    )
    _render_y_ticks(parts, bounds)
    _render_x_ticks(parts, bounds)
    parts.append(
        f'<text x="{bounds.left + bounds.plot_width / 2:.1f}" '
        f'y="{bounds.height - 12}" text-anchor="middle" '
        'font-size="12">Round</text>'
    )
    center_y = bounds.top + bounds.plot_height / 2
    parts.append(
        f'<text x="16" y="{center_y:.1f}" text-anchor="middle" font-size="12" '
        f'transform="rotate(-90 16,{center_y:.1f})">'
        f'{_h(request.primary)}</text>'
    )


def _render_y_ticks(parts: list[str], bounds: ChartBounds) -> None:
    for index in range(6):
        value = bounds.y_min + (bounds.y_max - bounds.y_min) * index / 5
        y = bounds.sy(value)
        parts.append(
            f'<line x1="{bounds.left}" y1="{y:.1f}" '
            f'x2="{bounds.left + bounds.plot_width}" y2="{y:.1f}" '
            'stroke="#e8e8e8" stroke-dasharray="2,2"/>'
        )
        parts.append(
            f'<text x="{bounds.left - 6}" y="{y + 3:.1f}" '
            f'text-anchor="end">{_fmt_num(value)}</text>'
        )


def _render_x_ticks(parts: list[str], bounds: ChartBounds) -> None:
    span = max(1, int(bounds.x_max - bounds.x_min))
    tick_count = min(span + 1, 11)
    step = max(1, int(round(span / max(1, tick_count - 1))))
    for round_id in range(int(bounds.x_min), int(bounds.x_max) + 1, step):
        x = bounds.sx(round_id)
        parts.append(
            f'<line x1="{x:.1f}" y1="{bounds.top + bounds.plot_height}" '
            f'x2="{x:.1f}" y2="{bounds.top + bounds.plot_height + 4}" '
            'stroke="#444"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{bounds.top + bounds.plot_height + 18}" '
            f'text-anchor="middle">R{round_id}</text>'
        )


def _render_reference_line(
    parts: list[str],
    request: SvgRequest,
    bounds: ChartBounds,
) -> None:
    if (
        request.ref_val is None
        or not bounds.y_min <= request.ref_val <= bounds.y_max
    ):
        return
    y = bounds.sy(request.ref_val)
    parts.append(
        f'<line x1="{bounds.left}" y1="{y:.1f}" '
        f'x2="{bounds.left + bounds.plot_width}" y2="{y:.1f}" '
        'stroke="#ff8c00" stroke-width="1.5" '
        'stroke-dasharray="6,3" opacity="0.75"/>'
    )


def _render_best_line(
    parts: list[str],
    series: PlotSeries,
    bounds: ChartBounds,
) -> None:
    if not series.best_rounds:
        return
    path = []
    for index, (round_id, value) in enumerate(
        zip(series.best_rounds, series.best_values)
    ):
        x = bounds.sx(round_id)
        y = bounds.sy(value)
        if index == 0:
            path.append(f"M {x:.1f} {y:.1f}")
        else:
            previous_y = bounds.sy(series.best_values[index - 1])
            path.append(f"L {x:.1f} {previous_y:.1f} L {x:.1f} {y:.1f}")
    parts.append(
        f'<path d="{" ".join(path)}" fill="none" stroke="#1f77b4" '
        'stroke-width="2" opacity="0.85"/>'
    )


def _render_measurements(
    parts: list[str],
    series: PlotSeries,
    bounds: ChartBounds,
) -> None:
    for round_id, value in zip(series.discard_rounds, series.discard_values):
        parts.append(
            f'<circle cx="{bounds.sx(round_id):.1f}" '
            f'cy="{bounds.sy(value):.1f}" r="4.5" '
            'fill="salmon" stroke="red" stroke-width="0.5" opacity="0.7"/>'
        )
    for round_id, value in zip(series.keep_rounds, series.keep_values):
        parts.append(
            f'<circle cx="{bounds.sx(round_id):.1f}" '
            f'cy="{bounds.sy(value):.1f}" r="5.5" '
            'fill="#2ca02c" stroke="darkgreen" stroke-width="0.6"/>'
        )


def _render_speedup_annotations(
    parts: list[str],
    series: PlotSeries,
    bounds: ChartBounds,
    reference: Optional[float],
) -> None:
    if reference is None:
        return
    annotations = [
        (round_id, value)
        for round_id, value in zip(series.keep_rounds, series.keep_values)
        if round_id != 0 and value > 0
    ]
    if not annotations:
        return
    filtered = _spaced_annotations(annotations, series.all_values)
    for round_id, value in filtered:
        speedup = series.speedups.get(int(round_id))
        if speedup is not None:
            parts.append(
                f'<text x="{bounds.sx(round_id):.1f}" '
                f'y="{bounds.sy(value) - 9:.1f}" text-anchor="middle" '
                f'font-size="9" fill="darkgreen">{speedup:.1f}x</text>'
            )


def _spaced_annotations(
    annotations: list[tuple[int, float]],
    all_values: list[float],
) -> list[tuple[int, float]]:
    value_span = (
        max(all_values) - min(all_values)
        if len(all_values) > 1
        else 1
    )
    minimum_gap = value_span * 0.06
    filtered = [annotations[-1]]
    for annotation in reversed(annotations[:-1]):
        if abs(annotation[1] - filtered[-1][1]) >= minimum_gap:
            filtered.append(annotation)
    return list(reversed(filtered))


def _render_failure_markers(
    parts: list[str],
    series: PlotSeries,
    bounds: ChartBounds,
) -> None:
    default_y = bounds.top + 10
    for round_id, value in zip(series.fail_rounds, series.fail_values):
        x = bounds.sx(round_id)
        y = bounds.sy(value) if value is not None else default_y
        parts.append(
            f'<path d="M {x - 4:.1f} {y - 4:.1f} '
            f'L {x + 4:.1f} {y + 4:.1f} '
            f'M {x + 4:.1f} {y - 4:.1f} '
            f'L {x - 4:.1f} {y + 4:.1f}" '
            'stroke="black" stroke-width="1.5"/>'
        )


def _render_legend(
    parts: list[str],
    request: SvgRequest,
    series: PlotSeries,
    bounds: ChartBounds,
) -> None:
    items = _legend_items(request, series)
    legend_x = bounds.left + bounds.plot_width + 14
    legend_y = bounds.top + 14
    for index, item in enumerate(items):
        kind, fill, stroke, label = item
        y = legend_y + index * 18
        _render_legend_symbol(parts, (kind, fill, stroke), legend_x, y)
        parts.append(
            f'<text x="{legend_x + 18}" y="{y + 3:.1f}" '
            f'font-size="10">{_h(label)}</text>'
        )


def _legend_items(
    request: SvgRequest,
    series: PlotSeries,
) -> list[tuple[str, str, str, str]]:
    items = []
    if series.keep_values:
        items.append(("circle", "#2ca02c", "darkgreen", f"keep ({len(series.keep_rounds)})"))
    if series.discard_values:
        items.append(("circle", "salmon", "red", f"discard ({len(series.discard_rounds)})"))
    if series.fail_rounds:
        items.append(("x", "black", "black", f"fail ({len(series.fail_rounds)})"))
    if series.best_rounds:
        items.append(("line", "#1f77b4", "#1f77b4", "best so far"))
    if request.ref_val is not None:
        label = f"{request.ref_label} ({request.ref_val:.1f})"
        items.append(("dashline", "#ff8c00", "#ff8c00", label))
    return items


def _render_legend_symbol(
    parts: list[str],
    symbol: tuple[str, str, str],
    x: float,
    y: float,
) -> None:
    kind, fill, stroke = symbol
    if kind == "circle":
        parts.append(
            f'<circle cx="{x + 6}" cy="{y:.1f}" r="4" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="0.5"/>'
        )
    elif kind == "x":
        parts.append(
            f'<path d="M {x + 1} {y - 3:.1f} L {x + 11} {y + 3:.1f} '
            f'M {x + 11} {y - 3:.1f} L {x + 1} {y + 3:.1f}" '
            f'stroke="{stroke}" stroke-width="1.5"/>'
        )
    else:
        dash = ' stroke-dasharray="4,2"' if kind == "dashline" else ""
        parts.append(
            f'<line x1="{x}" y1="{y:.1f}" x2="{x + 12}" y2="{y:.1f}" '
            f'stroke="{stroke}" stroke-width="2"{dash}/>'
        )


@dataclass(frozen=True)
class ReportData:
    history: list[dict]
    primary: str
    lower_is_better: bool
    progress: dict
    task_name: str
    ref_val: Optional[float]
    ref_label: str


def render_report(task_dir: str) -> str:
    """Build the full markdown report."""
    data = _load_report_data(task_dir)
    if data is None:
        return ""
    lines = _overview_lines(data)
    lines.extend(_shape_lines(data))
    svg = _generate_svg(SvgRequest(
        data.history,
        data.primary,
        data.lower_is_better,
        data.ref_val,
        data.ref_label,
        data.task_name,
    ))
    if svg:
        lines.extend(["## Optimization Curve", "", svg, ""])
    lines.extend(_improvement_lines(data))
    lines.extend(_all_round_lines(data))
    return "\n".join(lines)


def _load_report_data(task_dir: str) -> Optional[ReportData]:
    config = load_task_config(task_dir)
    history = _load_history(task_dir)
    if config is None or not history:
        return None
    progress = load_progress(task_dir) or {}
    raw_reference = progress.get("baseline_metric")
    reference = (
        float(raw_reference)
        if isinstance(raw_reference, (int, float)) and raw_reference > 0
        else None
    )
    ref_label = (
        "PyTorch ref"
        if progress.get("baseline_source") == "ref"
        else "baseline"
    )
    task_name = (
        progress.get("task")
        or os.path.basename(os.path.normpath(task_dir))
    )
    return ReportData(
        history,
        config.primary_metric,
        config.lower_is_better,
        progress,
        task_name,
        reference,
        ref_label,
    )


def _overview_lines(data: ReportData) -> list[str]:
    decisions = [record.get("decision") for record in data.history]
    kept = sum(decision in ("KEEP", "SEED") for decision in decisions)
    discarded = decisions.count("DISCARD")
    failed = decisions.count("FAIL")
    seed = data.progress.get("seed_metric")
    best = data.progress.get("best_metric")
    best_round = _best_round(data, best)
    direction = "越低越好" if data.lower_is_better else "越高越好"
    lines = [
        f"# {data.task_name} — 优化报告",
        "",
        "## 总览",
        "",
        "| 项目 | 值 |",
        "|------|---|",
        f"| 任务 | {_escape_md_cell(data.task_name)} |",
        f"| 总轮次 | {len(data.history)} |",
        f"| 接受 / 失败 / 丢弃 | {kept} / {failed} / {discarded} |",
        f"| 主指标 | {data.primary} ({direction}) |",
    ]
    if data.ref_val is not None:
        lines.append(f"| **{data.ref_label}** | **{data.ref_val:.2f}** |")
    if seed is not None:
        lines.append(f"| Seed kernel | {seed} |")
    lines.append(f"| **最优结果** | **{best} (Round {best_round})** |")
    lines.append(f"| 总改进 (vs seed) | {_improvement_label(data, seed, best)} |")
    speedup = data.progress.get("best_speedup")
    if isinstance(speedup, (int, float)) and speedup > 0:
        lines.append(
            f"| **最优加速比 (vs {data.ref_label})** | **{speedup:.2f}x** |"
        )
    lines.append("")
    return lines


def _best_round(data: ReportData, best: object) -> Optional[int]:
    if not isinstance(best, (int, float)):
        return None
    for record in data.history:
        if record.get("decision") not in ("KEEP", "SEED"):
            continue
        value = record.get("metrics", {}).get(data.primary)
        if isinstance(value, (int, float)) and abs(value - best) < 1e-9:
            return record.get("round")
    return None


def _improvement_label(data: ReportData, seed: object, best: object) -> str:
    if not (
        isinstance(seed, (int, float))
        and isinstance(best, (int, float))
        and seed
    ):
        return "N/A"
    delta = seed - best if data.lower_is_better else best - seed
    suffix = "reduction" if data.lower_is_better else "increase"
    return f"{delta / seed * 100:.1f}% {suffix}"


def _shape_lines(data: ReportData) -> list[str]:
    descriptions = []
    for record in data.history:
        candidate = (record.get("metrics", {}) or {}).get("per_shape_descs")
        if isinstance(candidate, list) and candidate:
            descriptions = candidate
            break
    if len(descriptions) <= 1:
        return []
    lines = [f"## 测试形状 ({len(descriptions)})", ""]
    lines.extend(
        f"{index}. {description}"
        for index, description in enumerate(descriptions)
    )
    lines.append("")
    return lines


def _key_improvements(data: ReportData) -> list[dict]:
    improvements = []
    previous_best: Optional[float] = None
    for record in data.history:
        if record.get("decision") not in ("KEEP", "SEED"):
            continue
        value = record.get("metrics", {}).get(data.primary)
        if not isinstance(value, (int, float)):
            continue
        if previous_best is not None:
            delta = (
                previous_best - value
                if data.lower_is_better
                else value - previous_best
            )
            if delta > 0:
                improvements.append({
                    "round": record.get("round"),
                    "desc": record.get("description", ""),
                    "from": previous_best,
                    "to": value,
                    "delta": delta,
                })
        is_better = (
            previous_best is None
            or (value < previous_best if data.lower_is_better else value > previous_best)
        )
        if is_better:
            previous_best = value
    return improvements


def _improvement_lines(data: ReportData) -> list[str]:
    improvements = _key_improvements(data)
    if not improvements:
        return []
    lines = [
        "## Key Improvements",
        "",
        f"| Round | Description | {data.primary} | Improvement |",
        f"|-------|-------------|{'---' * 4}|-------------|",
    ]
    for improvement in improvements:
        before = _table_number(improvement["from"])
        after = _table_number(improvement["to"])
        delta = _table_number(improvement["delta"])
        description = _escape_md_cell(improvement["desc"])
        lines.append(
            f"| R{improvement['round']} | {description} | "
            f"{before} → {after} | -{delta} |"
        )
    lines.append("")
    return lines


def _table_number(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _all_round_lines(data: ReportData) -> list[str]:
    lines = [
        "## All Rounds",
        "",
        f"| Round | Description | Decision | {data.primary} |",
        f"|-------|-------------|----------|{'---' * 4}|",
    ]
    for record in data.history:
        value = record.get("metrics", {}).get(data.primary, "—")
        if isinstance(value, float):
            value = f"{value:.4f}"
        description = _escape_md_cell(
            (record.get("description") or "")[:80]
        )
        lines.append(
            f"| R{record.get('round', '?')} | {description} | "
            f"{record.get('decision', '?')} | {value} |"
        )
    lines.append("")
    return lines


def write_report(task_dir: str) -> Optional[str]:
    """Write the report to .ar_state/report.md. Returns path or None."""
    md = render_report(task_dir)
    if not md:
        return None
    out = report_path(task_dir)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate FINISH-phase report.md")
    ap.add_argument("task_dir", help="Path to autoresearch task directory")
    ap.add_argument("--print", dest="to_stdout", action="store_true",
                    help="Print report to stdout instead of writing")
    args = ap.parse_args()

    task_dir = os.path.abspath(args.task_dir)
    if args.to_stdout:
        emit(render_report(task_dir), end="")
        return
    p = write_report(task_dir)
    if p:
        emit(p)
    else:
        emit("(no plottable data — empty history)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
