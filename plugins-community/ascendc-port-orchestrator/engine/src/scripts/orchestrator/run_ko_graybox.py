#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""run_ko_graybox.py — hermetic standalone runner for the KO (kernel-optimizer) GRAYBOX.

The perf-side counterpart of run_kw_graybox.py. Where the kw-graybox asks "can the
worker REPRODUCE a correct kernel from KB + arch22 alone?", the ko-graybox asks the
next pipeline question:

  *given the kw's CORRECT-but-SLOW kernel + its verification.json + the perf
  measurement (device-time vs baseline showing it's slow) — the REAL outputs of the
  preceding pipeline stages — plus the codified KB perf patterns + msprof profiling,
  with external undeclared tuned trees unavailable, does
  the ko OPTIMIZE it?*

This is the owner's "graybox is not only kw — do we need graybox for the rest of the
agents?" applied to ko (2026-06-09). Per the owner's real-input principle, the ko's
graybox input is the PRECEDING stage's real output (the kw kernel + perf), NOT a
hand-crafted or pre-optimized artifact.

ISOLATED INPUT-PROVENANCE CONTRACT (load-bearing):
  - The ko's "answer" is the hand-tuned / whole-port perf-optimized version of THIS
    op's kernel (multi-core banking, vectorization, dual-AIV, prefetch pipelines —
    the perf-grind results). Those MUST be absent so the ko re-derives optimizations
    from KB + profiling, not by copying.
  - Excluded from the arch22 source copy before traversal: every target arch35
    directory. Perf-grind artifacts and every other op's output are also absent.
  - PROVIDED (the real input): the kw's correct-but-slow kernel seeded into
    workspace/kernel/ (the ko tunes it IN-PLACE per the ko contract), verification.json
    (precision PASS + perf showing the gap), model.py spec, op_classification.json.
  - Verdict (OUTSIDE this runner, by an independent evaluator): re-measure the ko's kernel with
    perf_util.measure_perf (device+wall vs vendor). Did device-ratio improve from
    KB+profiling from the declared inputs? For a host_bound op (e.g. FA) the valid outcome is
    "ko correctly identifies kernel is near-vendor, gap is host (DEBT-147), no kernel
    headroom" — that demonstrates ko's host-vs-kernel discrimination. For a
    kernel_slow op (verdict=kernel_slow, picked via measure_perf) the demo is real
    kernel speedup.

USAGE
  python3 src/scripts/orchestrator/run_ko_graybox.py \
      --workspace workspace/<op>_ko_graybox \
      --kw-kernel <dir of the correct-but-slow kernel> \
      --verification <kw verification.json with precision PASS + perf gap> \
      [--op <op>] [--arch22 <V220 src>] [--spec <model.py>] [--lane 1] [--timeout 7200]
"""
import argparse
import fnmatch
import json
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# The ko's ANSWER = a perf-optimized version of this op's kernel. Any of these in the
# workspace would invalidate this isolated probe by supplying undeclared tuning.
_FORBIDDEN_NAMES = ["generated_ascendc_kernel"]
_FORBIDDEN_GLOBS = [
    "produced_*", "*_multicore*", "*_vec2*", "*_vec3*", "*_optimized*",
    "*_tuned*", "*.cce",
]


def _strip_optimized_artifacts(dst_root: Path) -> list[str]:
    """Delete perf-optimized or hand-tuned artifacts from a generated kernel input."""
    removed = []
    for pat in _FORBIDDEN_GLOBS:
        for p in list(dst_root.rglob(pat)):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
            removed.append(str(p.relative_to(dst_root)))
    return removed


def _copy_arch22_tree(src_root: Path, dst_root: Path) -> list[str]:
    """Copy allowed source files while pruning target dirs before traversal."""
    copied: list[str] = []
    pending = [(src_root, dst_root)]
    while pending:
        source, target = pending.pop()
        target.mkdir(parents=True, exist_ok=True)
        for item in sorted(source.iterdir()):
            if item.name.lower() == "arch35":
                continue
            if item.is_symlink():
                raise SystemExit(f"INPUT-PROVENANCE VIOLATION — source symlink is not allowed: {item}")
            if any(fnmatch.fnmatch(item.name, pattern) for pattern in _FORBIDDEN_GLOBS):
                continue
            destination = target / item.name
            if item.is_dir():
                pending.append((item, destination))
            elif item.is_file():
                shutil.copy2(item, destination)
                copied.append(str(destination.relative_to(dst_root)))
    return sorted(copied)


def _assert_hermetic(ws: Path) -> None:
    """Refuse to spawn if a perf-optimized answer artifact leaked into the workspace.
    The kw kernel under workspace/kernel/ is the LEGAL input (correct-but-slow) and is
    exempt — we only forbid optimized versions and cross-op output."""
    leaked = []
    for name in _FORBIDDEN_NAMES:
        leaked += [str(p.relative_to(ws)) for p in ws.rglob(name)]
    for pat in _FORBIDDEN_GLOBS:
        leaked += [str(p.relative_to(ws)) for p in ws.rglob(pat)]
    if leaked:
        raise SystemExit(
            f"INPUT-PROVENANCE VIOLATION — undeclared optimized artifacts in workspace: "
            f"{sorted(set(leaked))}. The ko must optimize from KB + profiling on the "
            f"kw's correct-but-slow kernel ALONE."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="hermetic standalone ko-graybox runner")
    ap.add_argument("--workspace", required=True, help="FRESH workspace dir (refuses if non-empty)")
    ap.add_argument("--op", default="flash_attention_score")
    ap.add_argument("--kw-kernel", required=True,
                    help="dir of the kw's CORRECT-but-SLOW kernel (the ko's real input; "
                         "seeded into workspace/kernel/, tuned in-place)")
    ap.add_argument("--verification", required=True,
                    help="kw verification.json (precision PASS + perf gap = the real perf signal)")
    ap.add_argument("--arch22", default=None, help="optional V220 algorithm source")
    ap.add_argument("--spec", required=True, help="user-provided model.py specification")
    ap.add_argument("--lane", type=int, default=1)
    ap.add_argument("--target", default="a5")
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()

    kw_kernel = Path(args.kw_kernel).expanduser().resolve()
    if not kw_kernel.is_dir():
        raise SystemExit(f"--kw-kernel not found: {kw_kernel}")
    verif = Path(args.verification).expanduser().resolve()
    if not verif.is_file():
        raise SystemExit(f"--verification not found: {verif}")
    spec = Path(args.spec).expanduser().resolve()

    ws = Path(args.workspace).resolve()
    if ws.exists() and any(ws.iterdir()):
        raise SystemExit(f"--workspace {ws} non-empty; a hermetic run needs a FRESH dir.")
    ws.mkdir(parents=True, exist_ok=True)

    # 1) spec (immutable reference)
    if spec.is_file():
        shutil.copy2(spec, ws / "model.py")

    # 2) seed the kw's correct-but-slow kernel as workspace/kernel/ (the REAL input the
    #    ko tunes in-place), then strip any perf-optimized artifact that rode along.
    kdir = ws / "kernel"
    shutil.copytree(kw_kernel, kdir)
    removed = _strip_optimized_artifacts(kdir)
    print(f"[ko-graybox] seeded kw kernel -> kernel/ ; stripped perf-answer artifacts: {removed}", flush=True)

    # 3) the kw's verification.json (precision PASS + perf gap = the real perf signal
    #    the ko reads in phase B to classify the bottleneck). This IS pipeline-produced.
    vj = json.loads(verif.read_text())
    (ws / "verification.json").write_text(json.dumps(vj, indent=2))
    perf = vj.get("precision", {}).get("pass_b", {}) or vj.get("performance", {})
    print(f"[ko-graybox] kw verification.json staged (perf signal present={bool(vj.get('performance'))})", flush=True)

    # 4) optional arch22 algorithm source; target dirs are pruned before traversal
    if args.arch22:
        src = Path(args.arch22).expanduser().resolve()
        if src.is_dir():
            a22 = ws / "arch22_src"
            a22.mkdir()
            for sub in ("op_host", "op_kernel"):
                s = src / sub
                if s.is_dir():
                    _copy_arch22_tree(s, a22 / sub)
            print("[ko-graybox] arch22-only source copied", flush=True)

    # 5) init durable state (port_a3 mode) + op_classification
    import phase_o05
    rep = phase_o05.init_durable_state(
        ws, args.op, lane=args.lane, target=args.target, opgen_mode="port_a3_to_a5"
    )
    print(f"[init] {rep.summary}", flush=True)
    cls = {
        "op": args.op,
        "op_class_tags": ["a3_to_a5_port", "FUSED_SOFTMAX", "ATTENTION", "fa_class", "CUBE_MIX"],
        "op_complexity": "L4",
        "kb_recommendations": [],
        "source": "ko_graybox_runner",
        "schema_version": 1,
    }
    (ws / "op_classification.json").write_text(json.dumps(cls, indent=2))

    # 6) hermetic pre-assert (no perf-optimized answer leaked)
    _assert_hermetic(ws)
    print(f"[hermetic] OK — workspace top: {sorted(p.name for p in ws.iterdir())}. Spawning ko...", flush=True)

    # 7) spawn the ko (kernel-optimizer). await_optimizer → aog-kernel-optimizer.
    import agent_dispatch
    result = agent_dispatch.spawn_for_state(
        args.op, ws, "await_optimizer", lane=args.lane, spawn_index=1, timeout_sec=args.timeout,
    )
    print(f"[ko] spawn returned (verdict={getattr(result, 'verdict', '?')}).", flush=True)

    # 8) what did the ko change? (graybox VERDICT — re-measure perf OUTSIDE,
    #    via perf_util.measure_perf: did device-ratio improve from KB+profiling ALONE?)
    olog = ws / "optimization_log.md"
    print(f"[ko-graybox] optimization_log.md present: {olog.exists()}", flush=True)
    print(
        "[ko-graybox] ko ran from isolated declared inputs. NEXT (graybox "
        "VERDICT, by independent evaluator): perf_util.measure_perf on the ko kernel — device+wall vs "
        "vendor. Did ko improve device-ratio from KB+profiling alone? (host_bound op → valid "
        "outcome is 'ko correctly finds no kernel headroom, gap is host'.)",
        flush=True,
    )


if __name__ == "__main__":
    main()
