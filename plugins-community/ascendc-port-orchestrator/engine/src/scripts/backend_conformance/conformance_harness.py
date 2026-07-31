#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Harness-backend conformance harness (HARNESS_BACKEND_ABSTRACTION_DESIGN.md §Conformance).

Proves the CCBackend refactor did NOT weaken any canonical gate: freeze an artifact,
run the CANONICAL gate entrypoints against it, capture {gate: verdict+reason-code}, and
assert BIT-IDENTICAL pre-refactor vs post-refactor (later: cross-backend).

**Never touches gate logic** — it REUSES `finalize_pipeline._pass_branch_gate_specs()`
(the canonical registry) so it stays faithful + in-sync as gates evolve. This file only
*invokes* gates and snapshots verdicts.

Usage:
  capture:  python3 conformance_harness.py capture <workspace> --out before.json
  diff:     python3 conformance_harness.py diff before.json after.json   # exit 1 if any divergence
"""
from __future__ import annotations
import argparse
import json
import sys
import traceback
from pathlib import Path

_ORCH = Path(__file__).resolve().parents[1] / "orchestrator"
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))


def capture_verdicts(workspace: Path) -> dict:
    """Run every canonical gate against `workspace`, return {gate_id: verdict}.
    verdict = None (pass) | str (reason-code) | '__EXC__:...' (gate errored on this artifact)
    | '__SKIP__:...' (gate needs an input this frozen artifact lacks).
    """
    import finalize_pipeline as fp  # canonical gates live here; import = reuse, not reimplement

    vj = {}
    vjp = workspace / "verification.json"
    if vjp.exists():
        try:
            vj = json.loads(vjp.read_text())
        except Exception as e:
            vj = {"__vj_parse_error__": str(e)}
    prec = vj.get("precision", {}) if isinstance(vj, dict) else {}

    verdicts: dict[str, object] = {}

    def _run(name, fn):
        try:
            verdicts[name] = fn()
        except FileNotFoundError as e:
            verdicts[name] = f"__SKIP__: missing input: {e}"
        except Exception as e:
            verdicts[name] = f"__EXC__: {type(e).__name__}: {e}"

    # --- inline pre-checks (not in the pass-branch registry) ---
    _run("STALE_ORCHESTRATOR", lambda: getattr(fp, "_check_stale_orchestrator")())
    _run("MODEL_PY_SHAPE", lambda: getattr(fp, "_check_model_py_shape")(workspace))
    _run("PASS_A_COVERAGE", lambda: getattr(fp, "_check_pass_a_coverage")(workspace, prec))

    # --- canonical pass-branch gate registry (authoritative, reused) [stage: finalize] ---
    for gate_id, pred in getattr(fp, "_pass_branch_gate_specs")():
        name = getattr(gate_id, "value", str(gate_id))
        if name == getattr(fp, "_PLUGIN_EXTRAS_SENTINEL", "plugin_extras"):
            # plugin-supplied extras return a dict, not a reason string — snapshot as-is
            _run("finalize:PLUGIN_EXTRAS", lambda p=pred: p(workspace, vj))
            continue
        _run("finalize:" + name, lambda p=pred: p(workspace, vj))

    # --- stage: O2.5 reference-provider contract (E) ---
    try:
        import phase_o25_a3_ref as o25  # canonical provider validators (main merged this round)
        _run("o25:model_contract", lambda: getattr(o25.validate_model_contract(workspace), "reason_code", "?"))
    except Exception as e:
        verdicts["o25:model_contract"] = f"__EXC__: import phase_o25_a3_ref: {type(e).__name__}: {e}"

    # --- stage: workflow_critic CHECK-LOGIC (B) — capture rejections, NOT the hook wiring ---
    try:
        _wf = str(Path(__file__).resolve().parents[1] / "workflow")
        if _wf not in sys.path:
            sys.path.insert(0, _wf)
        import workflow_critic as wc
        for nm, fn in [("wc:target_simt_compat", wc.check_target_simt_compat),
                       ("wc:global_invariants", wc.check_global_invariants),
                       ("wc:taxonomy_coverage", wc.check_taxonomy_coverage)]:
            def _cap(f=fn):
                rej: list = []
                f(workspace, rej)  # check-logic appends Rejection(s); empty = pass
                return None if not rej else "; ".join(sorted(getattr(r, "rule_id", str(r)) for r in rej))
            _run(nm, _cap)
    except Exception as e:
        verdicts["wc:__import__"] = f"__EXC__: import workflow_critic: {type(e).__name__}: {e}"

    # --- stage: safety guard CHECK-LOGIC (C) — FIXED probes (per-tool-call, workspace-independent);
    #     the key acceptance for the dispatch/hook-wiring refactor. Incl the bash-bypass airtight case. ---
    try:
        import output_read_guard as org  # workflow path already on sys.path (B section)
        it_mode = next(iter(getattr(org, "_ITERATE_MODES", {"optimize"})), "optimize")
        own = "output/myproj/kernels/opA"
        _probes = {
            # fresh/cold gen: ANY output/ read denied
            "guard:fresh_output_read_deny": lambda: getattr(org, "_should_deny")(
                "output/proj/kernels/opA/verification.json", "default", ""
            ),
            # iterate on OWN archive: allowed
            "guard:iterate_own_archive_allow": lambda: getattr(org, "_should_deny")(own + "/kernel/x.so", it_mode, own),
            # iterate but CROSS archive (other op / other SoC): denied (the cheat rule)
            "guard:iterate_cross_archive_deny": lambda: getattr(org, "_should_deny")("output/other/kernels/opA/x.so", it_mode, own),
            # non-output path: not our concern
            "guard:non_output_ignored": lambda: getattr(org, "_should_deny")("/home/w/workspace/opA/kernel.h", "default", ""),
            # BASH-BYPASS airtight case (DS PoC's key finding): a bash read-cmd's extracted output frag
            # hits the SAME deny path (guard isn't tool-name-gated in a bypassable way for reads).
            "guard:bash_read_bypass_denied": lambda: (
                bool(getattr(org, "_output_paths")("cat output/other/kernels/opA/verification.json"))
                and all(
                    getattr(org, "_should_deny")(file_path, "default", "")
                    for file_path in getattr(org, "_output_paths")("cat output/other/kernels/opA/verification.json")
                )
            ),
        }
        for nm, fn in _probes.items():
            _run(nm, fn)
    except Exception as e:
        verdicts["guard:__import__"] = f"__EXC__: {type(e).__name__}: {e}"

    # --- NEXT INCREMENT: ---
    #   D precision: precision_eval_port_a3_two_tier (calls compare.py) — needs tensor inputs
    #     (a3_capture/cpu_truth .pt); SKIP-record when a frozen archive lacks them. Low priority
    #     (grades artifact; dispatch/wiring refactor can't touch precision judgement).
    # EXCLUDED from hermetic conformance (external-state-dependent, not pure check-logic):
    #   - ship_claim_audit._verify_sha_on_main  (hits git → non-hermetic)
    #   - _maybe_port_a3_perf_remeasure          (NPU device measurement, not gate-replay)

    return dict(sorted(verdicts.items()))


def diff_snapshots(before: dict, after: dict) -> list[str]:
    """Return per-gate divergences (empty = bit-identical = conformant)."""
    keys = sorted(set(before) | set(after))
    out = []
    for k in keys:
        b, a = before.get(k, "<absent>"), after.get(k, "<absent>")
        if b != a:
            out.append(f"  {k}: before={b!r}  after={a!r}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("workspace")
    c.add_argument("--out", default=None)
    d = sub.add_parser("diff")
    d.add_argument("before")
    d.add_argument("after")
    a = ap.parse_args()

    if a.cmd == "capture":
        snap = capture_verdicts(Path(a.workspace).resolve())
        payload = {"workspace": str(Path(a.workspace).resolve()), "verdicts": snap}
        txt = json.dumps(payload, ensure_ascii=False, indent=2)
        if a.out:
            Path(a.out).write_text(txt)
            print(f"captured {len(snap)} gate verdicts -> {a.out}")
        else:
            print(txt)
        # summary
        npass = sum(1 for v in snap.values() if v is None)
        nfail = sum(1 for v in snap.values() if isinstance(v, str) and not v.startswith("__"))
        nskip = sum(1 for v in snap.values() if isinstance(v, str) and v.startswith("__SKIP__"))
        nexc = sum(1 for v in snap.values() if isinstance(v, str) and v.startswith("__EXC__"))
        print(f"# pass={npass} fail/reason={nfail} skip={nskip} exc={nexc} total={len(snap)}", file=sys.stderr)
        sys.exit(0)

    if a.cmd == "diff":
        before = json.loads(Path(a.before).read_text()).get("verdicts", {})
        after = json.loads(Path(a.after).read_text()).get("verdicts", {})
        div = diff_snapshots(before, after)
        if div:
            print("CONFORMANCE FAIL — canonical gate verdict divergence (refactor weakened/changed a gate):")
            print("\n".join(div))
            sys.exit(1)
        print(f"CONFORMANCE PASS — all {len(set(before)|set(after))} gate verdicts bit-identical")
        sys.exit(0)
