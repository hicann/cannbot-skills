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

"""Compact formatter for pipeline.py's eval result print.

Replaces ``print(f"[PIPELINE] Eval: correctness={c}, metrics={m}")`` —
that line dumped the entire metrics dict (per_shape_descs ~4KB, etc.).

Two functions:
  - ``summary_line(metrics, correctness)``: one-liner with the headline numbers
  - ``per_shape_table(metrics)``: aligned table of per-shape latencies + shapes

Caller prints summary_line first, then per_shape_table (if num_cases > 0).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from op_autoresearch.utils.console import emit

# ---------------------------------------------------------------------------
# Shape-desc cleaning
# ---------------------------------------------------------------------------

# Reference scaffolds describe the data tensor first, followed by normalized
# placeholders for scalar arguments. Only the first segment carries useful
# shape information, so the formatter removes the scalar placeholders.
_SHAPE_NONE_RE = re.compile(r",\s*shape=None\s+dtype=\w+")
_TORCH_DTYPE_RE = re.compile(r"\btorch\.")
_INPUTS_PREFIX_RE = re.compile(r"^inputs\[\d+\]:\s*")


def _clean_shape_desc(desc: str) -> str:
    """Strip scaffold boilerplate so the table column stays narrow."""
    if not desc:
        return ""
    s = _INPUTS_PREFIX_RE.sub("", desc)
    s = _SHAPE_NONE_RE.sub("", s)
    s = _TORCH_DTYPE_RE.sub("", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summary_line(metrics: Dict[str, Any], correctness: bool,
                 tag: str = "PIPELINE") -> str:
    """One-liner. Falls back to ``correctness=...`` only when metrics is
    empty (eval crashed before producing numbers).
    """
    if not metrics:
        return f"[{tag}] Eval: correctness={correctness} (no metrics)"

    parts = [f"correctness={correctness}"]
    lat = metrics.get("latency_us")
    if isinstance(lat, (int, float)):
        parts.append(f"latency={lat:.2f}us")
    spd = metrics.get("speedup_vs_ref")
    if isinstance(spd, (int, float)):
        parts.append(f"speedup={spd:.2f}x vs ref")
    nc = metrics.get("num_cases")
    if isinstance(nc, int) and nc > 0:
        parts.append(f"num_cases={nc}")
    return f"[{tag}] Eval: " + " | ".join(parts)


def _numeric_value(values: List[Any], index: int):
    if index >= len(values):
        return None
    value = values[index]
    return value if isinstance(value, (int, float)) else None


def _per_shape_header(
    show_status: bool, show_gen: bool, show_base: bool
) -> List[str]:
    header = ["#"]
    if show_status:
        header.append("status")
    if show_gen:
        header.append("gen_us")
    if show_base:
        header.extend(["base_us", "speedup"])
    header.append("shape")
    return header


def _per_shape_row(
    index: int,
    values: tuple[List[str], List[Any], List[Any], List[str]],
    columns: tuple[bool, bool, bool],
) -> List[str]:
    status, gen, base, descs = values
    show_status, show_gen, show_base = columns
    row = [f"#{index}"]
    if show_status:
        row.append(status[index] if index < len(status) else "—")
    gen_time = _numeric_value(gen, index)
    if show_gen:
        row.append(
            f"{gen_time:.2f}" if gen_time is not None else "—"
        )
    if show_base:
        base_time = _numeric_value(base, index)
        row.append(
            f"{base_time:.2f}" if base_time is not None else "—"
        )
        row.append(
            f"{base_time / gen_time:.2f}x"
            if base_time and gen_time
            else "—"
        )
    description = (
        _clean_shape_desc(descs[index])
        if index < len(descs)
        else ""
    )
    row.append(description)
    return row


def per_shape_table(metrics: Dict[str, Any]) -> str:
    """Render the canonical aligned per-shape result table."""
    status = list(metrics.get("per_shape_status") or [])
    gen = list(metrics.get("per_shape_gen_us") or [])
    base = list(metrics.get("per_shape_base_us") or [])
    descs = list(metrics.get("per_shape_descs") or [])
    row_count = max(len(status), len(gen), len(descs))
    if row_count == 0:
        return ""
    show_status = bool(status)
    show_gen = any(
        _numeric_value(gen, index) is not None
        for index in range(row_count)
    )
    show_base = show_gen and any(
        _numeric_value(base, index) is not None
        for index in range(row_count)
    )
    columns = show_status, show_gen, show_base
    values = status, gen, base, descs
    rows = [_per_shape_header(*columns)]
    rows.extend(
        _per_shape_row(index, values, columns)
        for index in range(row_count)
    )
    return _render_table(rows, indent="  ")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_numeric_cell(value: str) -> bool:
    value = value.strip().rstrip("x").rstrip("us").rstrip()
    if not value or value in ("—", "#"):
        return False
    if value.startswith("#") and value[1:].isdigit():
        return True
    try:
        float(value)
        return True
    except ValueError:
        return False


def _render_table_cell(value: str, width: int, *, numeric: bool,
                       last_column: bool) -> str:
    if last_column:
        return value
    if numeric:
        return value.rjust(width)
    return value.ljust(width)


def _render_table(rows: List[List[str]], indent: str = "") -> str:
    """Left-align text columns, right-align numeric columns. First row is
    header, gets a thin separator underneath.
    """
    if not rows:
        return ""
    ncols = max(len(r) for r in rows)
    widths = [0] * ncols
    for r in rows:
        for j, cell in enumerate(r):
            widths[j] = max(widths[j], len(str(cell)))

    out: List[str] = []
    for ri, r in enumerate(rows):
        cells = []
        for j in range(ncols):
            value = str(r[j]) if j < len(r) else ""
            cells.append(_render_table_cell(
                value,
                widths[j],
                numeric=ri > 0 and _is_numeric_cell(value),
                last_column=j == ncols - 1,
            ))
        out.append(indent + "  ".join(cells).rstrip())
        if ri == 0:
            # underline header — same width as columns separated by '  '
            sep_parts = ["-" * widths[j] for j in range(ncols)]
            out.append(indent + "  ".join(sep_parts).rstrip())
    return "\n".join(out)


# Shared eval-round plumbing for engine/baseline.py + engine/pipeline.py.

def write_artifact(path, data: Dict[str, Any]) -> str:
    """Dump ``data`` as an indented JSON artifact file; return its path. Single
    owner of the "write a JSON result file" step shared by batch verify.py
    (verify_results.json) and the FAIL report — neither hand-rolls its own
    write + json.dumps.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, default=str, ensure_ascii=False),
                 encoding="utf-8")
    return str(p)


def eval_result_to_dict(result) -> Dict[str, Any]:
    """EvalResult -> the dict record_baseline / record_round consume."""
    d = {
        "outcome": result.outcome.value,
        "correctness": result.correctness,
        "metrics": result.metrics or {},
        "error": result.error,
        "error_source": result.error_source,
    }
    if not result.correctness or result.error:
        # signals already parsed by eval_bridge from the full log — don't re-parse
        # the truncated tail; raw_output is already the tail.
        d["failure_signals"] = result.failure_signals or {}
        d["raw_output_tail"] = result.raw_output
        if result.fail_report:
            d["fail_report"] = result.fail_report
    return d


def print_eval_metrics(eval_data: Dict[str, Any], tag: str = "PIPELINE") -> None:
    """Summary line + per-shape table. On a multi-case FAIL the same table shows
    each shape's status + (verify-wall) gen/base/speedup — same path as a pass.
    """
    metrics = eval_data.get("metrics", {})
    emit(summary_line(metrics, eval_data.get("correctness", False), tag), flush=True)
    table = per_shape_table(metrics)
    if table:
        emit(table, flush=True)


def print_failure_signals(eval_data: Dict[str, Any], tag: str = "PIPELINE") -> None:
    """On failure: just point to the FAIL report file (full per-case + tracebacks
    + complete log + structured signals) for the agent to open with its file
    reader. The per-shape table is printed by ``print_eval_metrics``; the error
    line / signals / raw log are NOT echoed to stdout — they all live in the
    report. The raw log tail prints inline only as a last resort (no report).
    """
    if eval_data.get("correctness", False) and not eval_data.get("error"):
        return
    report = eval_data.get("fail_report")
    if report:
        emit(f"[{tag}] Full failure detail (per-shape status + tracebacks + "
              f"complete log + signals) written to: {report}", flush=True)
        emit(f"[{tag}] ^ open it with your file-reading tool (Read), not "
              f"bash/cat — it is the full, untruncated record.", flush=True)
    elif eval_data.get("raw_output_tail"):
        emit(f"[{tag}] Eval log tail (no report written):", flush=True)
        emit(eval_data["raw_output_tail"], flush=True)
    elif eval_data.get("error"):
        emit(f"[{tag}] Error: {eval_data['error']}", flush=True)
