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
"""run_kw_wire_graybox.py — hermetic standalone runner for the KW-WIRE (feature-wiring) GRAYBOX.

Third sibling to run_kw_graybox / run_ko_graybox. The precision-side pipeline question:

  *given a kw kernel that FAILED precision on a REAL case (the masked-row-sentinel
  mismatch on sparse 2/3/4) + its verification showing the FAIL + the case
  inputs/golden — the preceding stages' real outputs — plus the codified KB, and
  NOTHING ELSE (no pre-fixed kernel, no answer), does the pp ROOT-CAUSE and FIX it?*

Owner 2026-06-09: graybox is not only kw — validate the rest of the agents. pp was
never graybox-proven for FA (the kw self-fixed precision internally). The genuine pp
input (DS-confirmed): the masked-row-sentinel convention-mismatch — for fully-masked
rows (sparse 2/3/4 compressed-causal mask, edge_inputs_feature case 3/4/5) the kernel
computes a wrong FINITE value instead of vendor's FP_MAX (3.4e38) sentinel; the kw
could not resolve it. The pp must derive the sentinel convention from the failure +
KB, under isolated input provenance.

ISOLATED INPUT-PROVENANCE CONTRACT (the answer for pp = the precision FIX):
  - No pre-fixed / correct-sentinel version of the kernel in the workspace. The seeded
    kernel is the UN-fixed failing one; the pp edits it in place.
  - External undeclared target trees are unavailable in this probe. Production
    may supply provenance-tracked advisory artifacts, but they cannot replace
    arch22 semantics, failure evidence, or independent verification.
  - PROVIDED (real input): failing kernel + case inputs (edge_inputs) + golden
    (edge_dataset a3_outputs, incl the FP_MAX sentinel rows) + pass_b runners +
    verification.json framing sparse 2/3/4 as the precision FAIL + a probe handoff note.
  - Verdict (outside this runner, by an independent evaluator): did the pp root-cause the masked-row
    sentinel + fix it so 2/3/4 pass vs faithful golden, WITHOUT regressing the 7
    in-scope cases — from the declared KB + failure evidence?

USAGE
  python3 src/scripts/orchestrator/run_kw_wire_graybox.py \
      --workspace workspace/fa_pp_graybox \
      --kw-kernel workspace/fa_kw_graybox_feature/kernel \
      --feature-ws workspace/fa_kw_graybox_feature \
      --inputs  output/a3_to_a5_port/src/kernels/flash_attention_score/edge_inputs_feature.pt \
      --golden  output/a3_to_a5_port/src/kernels/flash_attention_score/edge_dataset_feature_captured.pt \
      [--fail-cases 2,3,4] [--op flash_attention_score] [--lane 1] [--timeout 7200]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_FORBIDDEN_GLOBS = ["*_fixed*", "*_correct*", "*sentinel_fix*", "*.cce"]

# Per-feature WIRE templates. The directive states WHAT to achieve (the unwired
# feature + the WIRE target), NOT the HOW — the kw figures out the wiring itself
# (isolated-input graybox), as it did for pse (empirically pinned semantics + surgical
# edits). {fc} = fail case ids, {isc} = in-scope case ids (filled at runtime).
_FEATURES = {
    "pse": {
        "fail_cases": [10],
        "in_scope": [0, 1, 5, 8, 9, 11],
        "title": "pse (positional bias) feature UNWIRED",
        "fail_reason": (
            "pse UNWIRED: the case ships a pse (positional bias) tensor but the kernel runs "
            "pseMode=PSE_NONE (wp_pse.h present but disabled) → output ignores the bias ≠ pse-golden. "
            "= the kw-WIRE target: wire wp_pse (enable pseMode + thread pse pybind→host→kernel + apply "
            "the bias to QK^T before softmax per arch22 algorithm semantics) so the pse case matches a3 golden, "
            "WITHOUT regressing the in-scope cases."
        ),
        "directive": (
            "Case(s) {fc}: a pse (positional bias) tensor is supplied, but the kernel runs "
            "`pseMode=PSE_NONE` (the `wp_pse.h` block is present but disabled) so the bias is ignored "
            "→ output ≠ pse-golden. WIRE it (debug-on-top, do NOT rewrite the kernel): enable the pse "
            "path (pseMode != PSE_NONE), thread the pse tensor pybind→host→kernel, apply the bias to "
            "QK^T before softmax per wp_pse and arch22 semantics. Constraint: in-scope {isc} must "
            "STAY passing. Do not read or reuse any arch35 implementation source."
        ),
        "truth": "a3_capture (edge_dataset golden, with pse applied for the pse case)",
    },
    "dropout": {
        "fail_cases": [6, 7],
        "in_scope": [0, 1, 5, 8, 9, 10, 11],
        "title": "dropout (Philox drop mask) feature UNWIRED",
        "fail_reason": (
            "dropout UNWIRED: cases ship keep_prob<1.0 but the kernel runs needDropMaskOp=0 / "
            "dropMaskOuter=0 / seed=offset=0 (wp_dropmask.h present but disabled) → no drop mask applied "
            "≠ dropout-golden. dropout is STOCHASTIC (Philox): the a3 golden was captured with a RECORDED "
            "seed/offset (a3_outputs.seed/offset per case) — the kernel must BIT-REPRODUCE the same mask "
            "from that seed/offset. = the kw-WIRE target: enable wp_dropmask + thread keep_prob + the "
            "recorded seed/offset pybind→host→kernel + faithfully port the arch22 drop_mask_adapter "
            "Philox so the dropout cases match a3 golden, WITHOUT regressing the in-scope cases."
        ),
        "directive": (
            "Case(s) {fc}: keep_prob<1.0 (dropout) is supplied, but the kernel runs "
            "`needDropMaskOp=0 / dropMaskOuter=0 / seed=offset=0` (the `wp_dropmask.h` block is present "
            "but disabled) so NO drop mask is applied → output ≠ dropout-golden. dropout is STOCHASTIC "
            "(Philox): the golden carries a RECORDED seed/offset (`a3_outputs.seed/offset` per case) — "
            "your kernel MUST bit-reproduce the SAME mask from that seed/offset (faithfully port the "
            "arch22 `flash_attention_score_drop_mask_adapter.h` Philox / the #38 scheme; do NOT invent a "
            "different RNG). WIRE it (debug-on-top, do NOT rewrite): enable the drop path "
            "(needDropMaskOp/dropMaskOuter), thread keep_prob + the recorded seed/offset "
            "pybind→host→kernel, apply the Philox drop mask to the softmax probabilities (surviving "
            "entries scaled by 1/keep_prob) per wp_dropmask / arch22. Constraint: in-scope {isc} must "
            "STAY passing. Do not read or reuse any arch35 implementation source."
        ),
        "truth": "a3_capture (edge_dataset golden, dropout applied with recorded seed/offset)",
    },
}


def _strip_answer(root: Path) -> list[str]:
    removed = []
    for pat in _FORBIDDEN_GLOBS:
        for p in list(root.rglob(pat)):
            if p.is_file():
                p.unlink()
                removed.append(str(p.relative_to(root)))
    return removed


def main() -> None:
    ap = argparse.ArgumentParser(description="hermetic standalone pp-graybox runner")
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--op", default="flash_attention_score")
    ap.add_argument("--kw-kernel", required=True,
                    help="dir of the kw kernel that FAILS the masked-row cases (pp edits in place)")
    ap.add_argument("--feature-ws", required=True, help="source of pass_b runners + model.py")
    ap.add_argument("--inputs", required=True, help="edge_inputs_feature.pt (case tensors incl masked-row)")
    ap.add_argument("--golden", required=True,
                    help="edge_dataset_*captured.pt (a3_outputs golden incl FP_MAX sentinel rows)")
    ap.add_argument("--feature", default="pse", choices=sorted(_FEATURES),
                    help="which gated feature to WIRE (selects directive/fail_reason template)")
    ap.add_argument("--fail-cases", default=None,
                    help="comma case_ids that FAIL (the unwired-feature cases); default = feature template")
    ap.add_argument("--in-scope-cases", default=None,
                    help="comma case_ids that MUST stay passing; default = feature template")
    ap.add_argument("--lane", type=int, default=1)
    ap.add_argument("--target", default="a5")
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()

    spec = _FEATURES[args.feature]

    kw_kernel = Path(args.kw_kernel).expanduser().resolve()
    feat = Path(args.feature_ws).expanduser().resolve()
    inputs = Path(args.inputs).expanduser().resolve()
    golden = Path(args.golden).expanduser().resolve()
    for p, n in [(kw_kernel, "--kw-kernel"), (feat, "--feature-ws"), (inputs, "--inputs"), (golden, "--golden")]:
        if not p.exists():
            raise SystemExit(f"{n} not found: {p}")
    fail_cases = ([int(x) for x in args.fail_cases.split(",") if x.strip()]
                  if args.fail_cases else list(spec["fail_cases"]))
    in_scope = ([int(x) for x in args.in_scope_cases.split(",") if x.strip()]
                if args.in_scope_cases else list(spec["in_scope"]))

    ws = Path(args.workspace).resolve()
    if ws.exists() and any(ws.iterdir()):
        raise SystemExit(f"--workspace {ws} non-empty; hermetic run needs a FRESH dir.")
    ws.mkdir(parents=True, exist_ok=True)

    # 1) seed the FAILING kw kernel (pp edits in place)
    kdir = ws / "kernel"
    shutil.copytree(kw_kernel, kdir)
    removed = _strip_answer(kdir)
    print(f"[pp-graybox] seeded failing kw kernel -> kernel/ ; stripped any fix-answer: {removed}", flush=True)

    # 2) case inputs + golden (the pp debugs against these)
    shutil.copy2(inputs, ws / "edge_inputs.pt")
    shutil.copy2(golden, ws / "edge_dataset.pt")
    # 3) pass_b runners + model.py spec (the pp re-verifies with these)
    for fn in ("pass_b_runner.py", "run_pass_b.py", "pass_a_runner.py", "model.py"):
        s = feat / fn
        if s.is_file():
            shutil.copy2(s, ws / fn)
    print("[pp-graybox] seeded inputs/golden/runners/model.py (self-contained)", flush=True)

    # 4) verification.json framing the UNWIRED feature case(s) as the FAIL (the kw-wire signal)
    vj = {
        "precision": {
            "status": "FAIL",
            "pass_b": {
                "status": "FAIL", "tier1_pass": len(in_scope), "total": len(in_scope) + len(fail_cases),
                "in_scope_cases": in_scope,
                "fail_cases": fail_cases,
                "fail_reason": spec["fail_reason"],
            },
        },
        "truth_source": spec["truth"],
        "schema_version": 1,
    }
    (ws / "verification.json").write_text(json.dumps(vj, indent=2))

    # 5) wire directive (the @aog-kernel-worker trigger context) — states WHAT, not HOW
    directive = spec["directive"].format(fc=fail_cases, isc=in_scope)
    (ws / "wire_directive.md").write_text(
        f"# @aog-kernel-worker — {spec['title']}\n\n{directive}\n"
    )
    print(
        f"[kw-wire-graybox] feature={args.feature} verification.json "
        f"(FAIL on {fail_cases}, in-scope {in_scope}) + wire_directive.md written",
        flush=True,
    )

    # 6) durable state + classification
    import phase_o05
    rep = phase_o05.init_durable_state(ws, args.op, lane=args.lane, target=args.target, opgen_mode="port_a3_to_a5")
    print(f"[init] {rep.summary}", flush=True)
    (ws / "op_classification.json").write_text(json.dumps({
        "op": args.op,
        "op_class_tags": ["a3_to_a5_port", "FUSED_SOFTMAX", "ATTENTION", "fa_class", "CUBE_MIX"],
        "op_complexity": "L4", "kb_recommendations": [], "source": "pp_graybox_runner", "schema_version": 1,
    }, indent=2))

    print(f"[hermetic] OK — workspace top: {sorted(p.name for p in ws.iterdir())}. Spawning pp...", flush=True)

    # 7) spawn the pp (precision-probe). await_worker → aog-kernel-worker.
    import agent_dispatch
    result = agent_dispatch.spawn_for_state(args.op, ws, "await_worker",
                                            lane=args.lane, spawn_index=1, timeout_sec=args.timeout)
    print(f"[pp] spawn returned (verdict={getattr(result, 'verdict', '?')}).", flush=True)

    pr = ws / "analysis.md"
    print(f"[pp-graybox] analysis.md present: {pr.exists()}", flush=True)
    print(
        "[pp-graybox] pp ran from isolated declared inputs. NEXT (independent verdict): did pp "
        f"root-cause the masked-row sentinel + fix cases {fail_cases} vs faithful golden WITHOUT "
        "regressing the 7 in-scope cases — from KB + the failure alone?",
        flush=True,
    )


if __name__ == "__main__":
    main()
