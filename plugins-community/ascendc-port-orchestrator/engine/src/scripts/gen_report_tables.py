# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Generate REPORT.md core tables from per-kernel verification.json files.

Usage:
    python3 src/scripts/gen_report_tables.py <project_dir> [--output <md_file>]

Input:  reads <project_dir>/src/kernels/*/verification.json (schema per OUTPUT_PROJECT_LAYOUT.md §3).
Output: markdown-formatted tables for REPORT.md §核心成果, §确定性, §性能.

Does NOT write narrative sections (precision details, architecture decisions,
methodology, pipeline findings) — those stay human-written. Goal: eliminate
hand-maintained tables that drift out of sync with verification.json.

Behaviour:
- If --output is given and the file exists, the tables are INJECTED between
  `<!-- BEGIN-GEN:<section> -->` and `<!-- END-GEN:<section> -->` comment
  markers, leaving human prose around them alone. If markers are absent, the
  tables are printed to stdout instead.
- If --output is absent, always prints to stdout.
- Each kernel's table row is generated from its verification.json; ops missing
  a verification.json (e.g. planned-but-not-done) are listed with `—` dashes.

Canonical table format is defined in `src/skills/references/shared/OUTPUT_PROJECT_LAYOUT.md`
§4.2 / §4.4. Any column-order change there must be mirrored here.
"""
from __future__ import annotations
import logging

import argparse
import json
import pathlib
import re
import sys
from typing import Any


def _fmt_num(v: Any, fmt: str = ".3f", dash: str = "—") -> str:
    if v is None:
        return dash
    try:
        return format(float(v), fmt)
    except Exception:
        return dash


def _fmt_precision(vj: dict) -> str:
    p = vj.get("precision", {})
    n = p.get("pass", 0)
    m = p.get("total", 0)
    # Fallback 1: some verification.json use `results: "50/50"` string instead of pass/total ints
    if not m:
        results = p.get("results", "")
        mo = re.match(r"(\d+)\s*/\s*(\d+)", str(results))
        if mo:
            n = int(mo.group(1))
            m = int(mo.group(2))
    # Fallback 2: pass_benchmark / total_benchmark (op#11 schema variant)
    if not m:
        n = p.get("pass_benchmark", 0)
        m = p.get("total_benchmark", 0)
    # Fallback 3: nested pass_a (op#1_GELU schema variant)
    if not m:
        pa = p.get("pass_a", {})
        if isinstance(pa, dict):
            n = pa.get("pass", 0)
            m = pa.get("total", 0)
            if not p.get("status"):
                p = {**p, "status": pa.get("status", "")}
    status = p.get("status", "")
    if status == "PASS":
        return f"**{n}/{m} bit-exact**"
    if status == "PASS_WITHIN_TOLERANCE":
        tol = p.get("tolerance", {})
        rtol = tol.get("rtol")
        return f"**{n}/{m} within-tol**" + (f"（rtol={rtol:g}）" if rtol else "")
    if status in ("PARTIAL", "PARTIAL_PASS"):
        return f"{n}/{m} PARTIAL"
    if status == "FAIL":
        return f"{n}/{m} FAIL"
    return f"{n}/{m} {status or '—'}"


def _precision_verdict(vj: dict) -> str:
    """Status column value — PRECISION-ONLY axis.

    Rule (2026-04-22, per user directive): Status reflects precision pass/fail ONLY.
    Perf below target is NOT a reason to flag PARTIAL. Perf information belongs in the
    Notes column or a separate Perf column.

    Mapping:
      precision.status == "PASS" / "PASS_WITHIN_TOLERANCE"  → ✅ DONE
      precision.status == "PARTIAL" / "PARTIAL_PASS"          → ⚠️ PARTIAL
      precision.status == "FAIL"                              → ❌ FAIL
      (missing / unknown)                                     → —
    """
    status = vj.get("precision", {}).get("status", "")
    if status in ("PASS", "PASS_WITHIN_TOLERANCE"):
        return "✅ DONE"
    if status in ("PARTIAL", "PARTIAL_PASS"):
        return "⚠️ PARTIAL"
    if status == "FAIL":
        return "❌ FAIL"
    return "—"


def _fmt_det(vj: dict) -> str:
    d = vj.get("determinism", {})
    pol = d.get("policy", "—")
    if pol == "n/a":
        return "N/A"
    if d.get("policy_satisfied"):
        return f"✅ {pol}"
    return f"❌ {pol}"


def _sha_short(s: str, width: int = 12) -> str:
    if not s:
        return "—"
    return s[:width] + "…" if len(s) > width else s


def _core_results_row(op: str, vj: dict) -> str:
    perf = vj.get("performance", {})
    source = perf.get("source") or perf.get("a3") or {}
    target = perf.get("target") or perf.get("a5") or {}
    source_ms = _fmt_num(source.get("median_ms"))
    target_ms = _fmt_num(target.get("median_ms"))
    source_bw = _fmt_num(source.get("effective_bandwidth_gb_s"), ".1f")
    target_bw = _fmt_num(target.get("effective_bandwidth_gb_s"), ".1f")
    ratio = perf.get("target_over_source_bandwidth_ratio")
    time_ratio = (target.get("median_ms") / source.get("median_ms")
                  ) if (target.get("median_ms") and source.get("median_ms")) else None
    bw_ratio_s = f"**{ratio:.3f}×**" if ratio is not None else "—"
    time_ratio_s = f"{time_ratio:.3f}×" if time_ratio is not None else "—"
    cov = vj.get("precision", {}).get("precision_coverage", {}).get("tier", "—")
    approach = vj.get("notes", {}).get("primitive_choice", "—")
    if len(approach) > 80:
        approach = approach[:77] + "…"
    return (f"| {op} | {_fmt_precision(vj)} | `{cov}` | {_fmt_det(vj)} | "
            f"{source_ms} | {target_ms} | {time_ratio_s} | {target_bw} | {source_bw} | {bw_ratio_s} | "
            f"{approach} |")


# ── V3.4 (DEBT-035): a3/a2 project class — reference vs ascendc table ──
# Used when project_dir.name ends in `-a3` or `-a2`. Reads flat schema
# (`performance.{ratio_mean, ratio_median, reference_ms_mean, reference_ms_median,
# ascendc_ms_mean, ascendc_ms_median}`) instead of the source/target nested schema.

def _op_link(project_dir: pathlib.Path, op: str) -> str:
    """Make op name a markdown link to its archive directory.

    REPORT.md lives at <project>/docs/REPORT.md; archive is at <project>/src/kernels/<op>/.
    Relative path from REPORT.md to archive dir is `../src/kernels/<op>/`.
    """
    return f"[{op}](../src/kernels/{op}/)"


def _artifact_links(project_dir: pathlib.Path, op: str) -> str:
    """Inline links to common per-op artifacts when present on disk.

    Returns markdown like ` [[probe](path)] [[critic](path)]` (leading space,
    so it can be appended to a notes column). Empty string if none.
    """
    base = project_dir / "src" / "kernels" / op
    out = []
    for label, fname in [("kernel.h", None),  # placeholder; resolved below
                         ("probe", "probe_report.md"),
                         ("critic", "self_critic_report.md"),
                         ("knowledge", "knowledge_update.md"),
                         ("perf", "perf_report.md")]:
        if fname is None:
            # kernel.h: there can be one .h under kernel/ — try a glob
            kdir = base / "kernel"
            if kdir.is_dir():
                hs = sorted(p.name for p in kdir.glob("*_kernel.h"))
                if hs:
                    out.append(f"[{label}](../src/kernels/{op}/kernel/{hs[0]})")
            continue
        if (base / fname).exists():
            out.append(f"[{label}](../src/kernels/{op}/{fname})")
    return (" " + " ".join(out)) if out else ""


def _core_results_row_v220(op: str, vj: dict, project_dir: pathlib.Path) -> str:
    p = vj.get("performance", {})
    cov = vj.get("precision", {}).get("precision_coverage", {}).get("tier", "—")
    rm = _fmt_num(p.get("reference_ms_median"))
    am = _fmt_num(p.get("ascendc_ms_median"))
    mr = p.get("ratio_median")
    nr = p.get("ratio_mean")
    # Fallback: ops that only stored an aggregate ratio (op#11/op#30 variants)
    if mr is None:
        mr = p.get("overall_speedup_ratio") or p.get("overall_speedup")
    if nr is None:
        nr = p.get("overall_speedup_ratio") or p.get("overall_speedup")
    mr_s = f"**{mr:.2f}x**" if mr is not None else "—"
    nr_s = f"{nr:.2f}x" if nr is not None else "—"
    notes = vj.get("a3_porting_notes", {})
    note_str = "; ".join(filter(None, [notes.get("fix_applied") or notes.get("fix_applied_1"),
                                       notes.get("nblk_status") or notes.get("nblk_experiment"),
                                       notes.get("open_issue_F3")]))
    if len(note_str) > 80:
        note_str = note_str[:77] + "…"
    note_str = (note_str or "—") + _artifact_links(project_dir, op)
    return (f"| {_op_link(project_dir, op)} | {_fmt_precision(vj)} | `{cov}` | {_fmt_det(vj)} | "
            f"{rm} | {am} | {mr_s} | {nr_s} | {note_str} |")


def _perf_row_v220(op: str, vj: dict, project_dir: pathlib.Path) -> str:
    p = vj.get("performance", {})
    rm = _fmt_num(p.get("reference_ms_mean"))
    am = _fmt_num(p.get("ascendc_ms_mean"))
    rmed = _fmt_num(p.get("reference_ms_median"))
    amed = _fmt_num(p.get("ascendc_ms_median"))
    mr = p.get("ratio_mean")
    mr_s = f"{mr:.2f}x" if mr is not None else "—"
    md = p.get("ratio_median")
    md_s = f"**{md:.2f}x**" if md is not None else "—"
    return f"| {_op_link(project_dir, op)} | {rm} | {am} | {rmed} | {amed} | {mr_s} | {md_s} |"


def _is_v220_project(project_dir: pathlib.Path) -> bool:
    """Detect a3/a2 project class. Heuristic: name ends in `-a3` or `-a2`,
    OR a `.project_meta.json` exists with `target_class: v220`."""
    name = project_dir.name.lower()
    if name.endswith("-a3") or name.endswith("-a2"):
        return True
    meta = project_dir / ".project_meta.json"
    if meta.is_file():
        try:
            return json.loads(meta.read_text()).get("target_class") == "v220"
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
    return False


def _det_row(op: str, vj: dict) -> str:
    d = vj.get("determinism", {})
    sha = _sha_short(d.get("run_sha256", ""), 64)
    runs = d.get("n_runs", 0)
    cases = d.get("n_cases_checked", 0)
    return f"| {op} | {runs} | {cases} | `{sha}` |"


def _perf_row(op: str, vj: dict) -> str:
    perf = vj.get("performance", {})
    if perf.get("status") != "MEASURED":
        return f"| {op} | — | — | — | — | — | — |"
    source = perf.get("source") or perf.get("a3") or {}
    target = perf.get("target") or perf.get("a5") or {}
    n = perf.get("n", 0)
    return (f"| {op} | {n:,} | "
            f"{_fmt_num(source.get('median_ms'))} | {_fmt_num(target.get('median_ms'))} | "
            f"{_fmt_num(source.get('effective_bandwidth_gb_s'), '.1f')} | "
            f"{_fmt_num(target.get('effective_bandwidth_gb_s'), '.1f')} | "
            f"**{perf.get('target_over_source_bandwidth_ratio', 0):.3f}×** |")


def gen_tables(project_dir: pathlib.Path) -> dict[str, str]:
    kernels_dir = project_dir / "src" / "kernels"
    if not kernels_dir.is_dir():
        sys.exit(f"ERROR: {kernels_dir} not found — is {project_dir} a valid project (needs src/kernels/)?")

    # Collect (op_name, verification.json) pairs, sorted by dir name.
    # Skip dirs starting with `_` (pre-merge test backups, e.g. `_1_GELU_pre_merged_test_*`).
    ops: list[tuple[str, dict]] = []
    for sub in sorted(kernels_dir.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.startswith("_"):
            continue
        vj_path = sub / "verification.json"
        if vj_path.exists():
            try:
                ops.append((sub.name, json.loads(vj_path.read_text())))
            except json.JSONDecodeError as e:
                print(f"WARN: {vj_path} not parseable: {e}", file=sys.stderr)

    if not ops:
        sys.exit(f"ERROR: no verification.json under {kernels_dir}")

    if _is_v220_project(project_dir):
        # V3.4 a3/a2 project class — reference-vs-ascendc tables (DEBT-035)
        core = [
            "| 算子 | 精度 (Pass A) | coverage_tier | 确定性 | reference median (ms) "
            "| ascendc median (ms) | median ratio | mean ratio | 备注 |",
            "|------|:---:|:---:|:---:|---:|---:|:---:|:---:|------|",
        ]
        core += [_core_results_row_v220(op, vj, project_dir) for op, vj in ops]

        det = [
            "| 算子 | DET_POLICY | observed | by-construction 论证 |",
            "|------|:---:|:---:|------|",
        ]
        for op, vj in ops:
            d = vj.get("determinism", {})
            policy = d.get("policy", "—")
            obs = "by-construction" if d.get("by_construction_reasoning") else _fmt_det(vj)
            reason = (d.get("by_construction_reasoning", "") or "").strip()
            if len(reason) > 80:
                reason = reason[:77] + "…"
            det.append(f"| {_op_link(project_dir, op)} | {policy} | {obs} | {reason or '—'} |")

        perf = [
            "| 算子 | reference mean (ms) | ascendc mean (ms) | reference median (ms) "
            "| ascendc median (ms) | mean ratio | median ratio |",
            "|------|---:|---:|---:|---:|:---:|:---:|",
        ]
        perf += [_perf_row_v220(op, vj, project_dir) for op, vj in ops]

        return {
            "core_results": "\n".join(core),
            "determinism": "\n".join(det),
            "performance": "\n".join(perf),
        }

    # Default — source / target comparison tables
    # §核心成果
    core = [
        "| 算子 | 精度 | coverage_tier | 确定性 | 来源 median (ms) | 目标 median (ms) "
        "| time(目标/来源) | 目标带宽 (GB/s) | 来源带宽 (GB/s) | **BW(目标/来源)** | 实现方案 |",
        "|------|:---:|:---:|:---:|---:|---:|:---:|---:|---:|:---:|------|",
    ]
    core += [_core_results_row(op, vj) for op, vj in ops]

    # §确定性
    det = [
        "| 算子 | n_runs | n_cases | run_sha256 |",
        "|------|:---:|:---:|---|",
    ]
    det += [_det_row(op, vj) for op, vj in ops]

    # §性能
    perf = [
        "| 算子 | n | 来源 median (ms) | 目标 median (ms) | 来源 BW (GB/s) | 目标 BW (GB/s) | **BW(目标/来源)** |",
        "|------|---:|---:|---:|---:|---:|:---:|",
    ]
    perf += [_perf_row(op, vj) for op, vj in ops]

    return {
        "core_results": "\n".join(core),
        "determinism": "\n".join(det),
        "performance": "\n".join(perf),
    }


def _inject(md_text: str, section: str, table: str) -> tuple[str, bool]:
    begin = re.compile(rf"<!--\s*BEGIN-GEN:{re.escape(section)}\s*-->")
    end = re.compile(rf"<!--\s*END-GEN:{re.escape(section)}\s*-->")
    mb = begin.search(md_text)
    me = end.search(md_text)
    if not mb or not me or me.start() < mb.end():
        return md_text, False
    return (md_text[:mb.end()] + "\n" + table + "\n" + md_text[me.start():], True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_dir", help="e.g. output/BabelStream")
    ap.add_argument("--output", help="REPORT.md to inject tables into (uses <!-- BEGIN-GEN:<section> --> markers)")
    args = ap.parse_args()

    project = pathlib.Path(args.project_dir).resolve()
    tables = gen_tables(project)

    if not args.output:
        for name, t in tables.items():
            print(f"## GEN:{name}\n\n{t}\n")
        return

    md_path = pathlib.Path(args.output).resolve()
    if not md_path.exists():
        print(f"{md_path} does not exist — printing tables to stdout instead", file=sys.stderr)
        for name, t in tables.items():
            print(f"## GEN:{name}\n\n{t}\n")
        return

    md = md_path.read_text()
    n_injected = 0
    for name, t in tables.items():
        md, ok = _inject(md, name, t)
        if ok:
            n_injected += 1
        else:
            print(f"warn: no <!-- BEGIN-GEN:{name} --> / <!-- END-GEN:{name} --> in {md_path}", file=sys.stderr)
    md_path.write_text(md)
    print(f"injected {n_injected}/{len(tables)} sections into {md_path}")


if __name__ == "__main__":
    main()
