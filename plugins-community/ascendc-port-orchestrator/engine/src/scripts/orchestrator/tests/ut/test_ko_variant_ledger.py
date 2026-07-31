# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Unit tests for ko_variant_ledger — the deterministic kernel-identity gate.

LAST-RECORD SEMANTICS under test (main hard gate). The write side records
`kernel_md5` AFTER each keep/revert decision, so the LAST Opt entry equals the
current kernel by construction. The verdict compares only that endpoint:

  - endpoint == current  -> whole history already-explored -> all skippable
  - endpoint != current  -> kernel rewritten outside the log -> all must_rerun
  - no entries / no md5 / unresolvable kernel -> nothing skippable (safe re-run)

The load-bearing test is `test_multi_variant_real_keep_revert_all_skippable`:
an N=2 session with a REAL revert (Opt0) then keep (Opt1) — the earlier entry
legitimately holds a *different* md5 (the reverted baseline), yet the log is
faithful to the current kernel and every variant must be skippable. The old
per-entry model returned `baseline_changed=True` + `must_rerun=[Opt0,...]` here
(livelock intact); this test pins the corrected behaviour. It drives the WRITE
recipe through the real CLI (`--print-kernel-md5`) at each recording point, so a
write/read md5 drift also fails it.
"""
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
# src/scripts/orchestrator on sys.path so `import ko_variant_ledger` resolves.
sys.path.insert(0, str(_HERE.parents[2]))

import ko_variant_ledger as L  # noqa: E402

_LEDGER = _HERE.parents[2] / "ko_variant_ledger.py"


def _mk_workspace(tmp_path, kernel_files: dict, log: str | None):
    ws = tmp_path / "op"
    (ws / "kernel").mkdir(parents=True)
    for name, body in kernel_files.items():
        (ws / "kernel" / name).write_text(body)
    if log is not None:
        (ws / "optimization_log.md").write_text(log)
    return ws


def _write_kernel(ws, kernel_files: dict):
    """Overwrite the kernel sources in place (simulates an optimizer edit)."""
    for name, body in kernel_files.items():
        (ws / "kernel" / name).write_text(body)


def _print_md5_via_cli(ws) -> str:
    """The exact WRITE-side recipe the brief instructs the agent to run.

    Records go through the real CLI, not compute_kernel_md5 directly, so a
    write/read hash divergence surfaces as a test failure (not a silent inert
    gate).
    """
    out = subprocess.run(
        [sys.executable, str(_LEDGER), "--workspace", str(ws), "--print-kernel-md5"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert len(out) == 32 and all(c in "0123456789abcdef" for c in out), out
    return out


# --------------------------------------------------------------------------
# The load-bearing N>=2 test: real keep + revert, kernel unchanged externally.
# --------------------------------------------------------------------------

def test_multi_variant_real_keep_revert_all_skippable(tmp_path):
    """N=2 session, Opt0 REVERT then Opt1 KEEP, no external kernel change.

    Simulates the true optimizer loop with per-decision md5 recording:
      - start: kernel = baseline v0
      - Opt0: try variant A (edit -> vA), REVERT (kernel back to v0),
              record md5(v0)   <- recorded AFTER the decision
      - Opt1: try variant B (edit -> vB), KEEP (kernel stays vB),
              record md5(vB)   <- recorded AFTER the decision, == current

    The two recorded md5s DIFFER (v0 vs vB) — exactly the shape that broke the
    per-entry model. Under last-record semantics the log endpoint (Opt1 = vB) ==
    current kernel, so BOTH variants are already-explored -> skippable, and there
    is NO spurious baseline_changed.
    """
    ws = _mk_workspace(tmp_path, {"k.h": "baseline-v0"}, log="")

    # Opt0: edit to variant A, then REVERT back to baseline; record post-decision.
    _write_kernel(ws, {"k.h": "variant-A-body"})
    _write_kernel(ws, {"k.h": "baseline-v0"})          # revert
    md5_opt0 = _print_md5_via_cli(ws)                   # == md5(baseline-v0)

    # Opt1: edit to variant B and KEEP; record post-decision (== current kernel).
    _write_kernel(ws, {"k.h": "variant-B-body"})       # keep
    md5_opt1 = _print_md5_via_cli(ws)                   # == md5(variant-B-body)

    assert md5_opt0 != md5_opt1, "fixture must exercise differing per-entry md5s"

    (ws / "optimization_log.md").write_text(
        f"## Opt0 — try A (reverted)\nkernel_md5: {md5_opt0}\nDecision: REVERT\n\n"
        f"## Opt1 — try B (kept)\nkernel_md5: {md5_opt1}\nDecision: KEEP\n"
    )

    v = L.build_verdict(ws)
    assert v["last_recorded_md5"] == md5_opt1
    assert set(v["skippable"]) == {"Opt0", "Opt1"}, (
        "log endpoint == current kernel -> all variants already explored; got " + repr(v)
    )
    assert v["must_rerun"] == []
    assert v["baseline_changed"] is False, (
        "differing earlier-entry md5 (reverted baseline) must NOT trip "
        "baseline_changed under last-record semantics; got " + repr(v)
    )


def test_multi_variant_external_kernel_change_all_rerun(tmp_path):
    """N=2 session then the WORKER rewrites the kernel outside the log.

    After Opt0/Opt1 (endpoint recorded == vB), a Kind-2 worker directive rewrites
    the kernel to vC. Now current (vC) != endpoint (vB) -> the whole optimization
    history was measured on a superseded kernel -> ALL must_rerun + baseline_changed.
    """
    ws = _mk_workspace(tmp_path, {"k.h": "baseline-v0"}, log="")
    _write_kernel(ws, {"k.h": "baseline-v0"})
    md5_opt0 = _print_md5_via_cli(ws)
    _write_kernel(ws, {"k.h": "variant-B-body"})       # kept
    md5_opt1 = _print_md5_via_cli(ws)
    (ws / "optimization_log.md").write_text(
        f"## Opt0 — a\nkernel_md5: {md5_opt0}\nDecision: REVERT\n\n"
        f"## Opt1 — b\nkernel_md5: {md5_opt1}\nDecision: KEEP\n"
    )

    # Worker rewrites the kernel (external change, not logged as an Opt).
    _write_kernel(ws, {"k.h": "worker-rewrite-vC"})

    v = L.build_verdict(ws)
    assert v["last_recorded_md5"] == md5_opt1
    assert v["current_kernel_md5"] != md5_opt1
    assert v["skippable"] == []
    assert set(v["must_rerun"]) == {"Opt0", "Opt1"}
    assert v["baseline_changed"] is True


def test_single_variant_kept_is_skippable(tmp_path):
    """N=1, kept: endpoint == current -> skippable (round-trips via CLI)."""
    ws = _mk_workspace(tmp_path, {"k.h": "kernel-v1"}, log="")
    md5 = _print_md5_via_cli(ws)
    (ws / "optimization_log.md").write_text(
        f"## Opt0 — first variant kept\nkernel_md5: {md5}\nDecision: KEEP\n"
    )
    v = L.build_verdict(ws)
    assert v["skippable"] == ["Opt0"]
    assert v["must_rerun"] == []
    assert v["baseline_changed"] is False


def test_fresh_op_no_log_nothing_skippable(tmp_path):
    ws = _mk_workspace(tmp_path, {"k.h": "body"}, log=None)
    v = L.build_verdict(ws)
    assert v["log_present"] is False
    assert v["n_entries"] == 0
    assert v["skippable"] == []
    assert v["must_rerun"] == []
    assert v["baseline_changed"] is False
    assert v["kernel_resolved"] is True


def test_empty_log_nothing_skippable(tmp_path):
    ws = _mk_workspace(tmp_path, {"k.h": "body"}, log="")
    v = L.build_verdict(ws)
    assert v["n_entries"] == 0
    assert v["skippable"] == []
    assert v["must_rerun"] == []
    assert v["baseline_changed"] is False


def test_last_entry_missing_md5_uses_prior_recorded(tmp_path):
    """A trailing legacy entry without md5 must not blind the endpoint check —
    the last entry that DID record one is the endpoint.
    """
    ws = _mk_workspace(tmp_path, {"k.h": "kernel-v1"}, log="")
    md5 = _print_md5_via_cli(ws)
    (ws / "optimization_log.md").write_text(
        f"## Opt0 — recorded\nkernel_md5: {md5}\nDecision: KEEP\n\n"
        f"## Opt1 — legacy note, no md5\nDecision: NOTE\n"
    )
    v = L.build_verdict(ws)
    assert v["last_recorded_md5"] == md5
    assert set(v["skippable"]) == {"Opt0", "Opt1"}
    assert v["baseline_changed"] is False


def test_no_entry_records_md5_all_rerun_not_baseline_changed(tmp_path):
    """Old-format log, no md5 anywhere -> can't establish endpoint -> rerun all,
    but missing != different, so NOT baseline_changed.
    """
    ws = _mk_workspace(tmp_path, {"k.h": "body"}, log="")
    (ws / "optimization_log.md").write_text(
        "## Opt0 — legacy\nBaseline: 0.5x\nDecision: REVERT\n\n"
        "## Opt1 — legacy\nBaseline: 0.7x\nDecision: KEEP\n"
    )
    v = L.build_verdict(ws)
    assert v["last_recorded_md5"] is None
    assert v["skippable"] == []
    assert set(v["must_rerun"]) == {"Opt0", "Opt1"}
    assert v["baseline_changed"] is False


def test_unresolvable_kernel_never_skips(tmp_path):
    """No kernel dir -> identity unknown -> never a false skip, never baseline_changed."""
    ws = tmp_path / "op"
    ws.mkdir()
    (ws / "optimization_log.md").write_text(
        f"## Opt0 — x\nkernel_md5: {'c' * 32}\nDecision: KEEP\n"
    )
    v = L.build_verdict(ws)
    assert v["kernel_resolved"] is False
    assert v["current_kernel_md5"] is None
    assert v["skippable"] == []
    assert v["must_rerun"] == ["Opt0"]
    assert v["baseline_changed"] is False


def test_lenient_md5_field_parsing(tmp_path):
    """Endpoint md5 recorded with bullet + backticks + odd label casing parses."""
    ws = _mk_workspace(tmp_path, {"k.h": "bodyP"}, log="")
    cur = L.compute_kernel_md5(ws)
    (ws / "optimization_log.md").write_text(
        f"## Opt0 — a\n- Kernel-MD5: `{cur}`\nDecision: KEEP\n"
    )
    v = L.build_verdict(ws)
    assert v["skippable"] == ["Opt0"]
    assert v["baseline_changed"] is False


def test_kernel_md5_changes_when_source_changes(tmp_path):
    ws = _mk_workspace(tmp_path, {"k.h": "v1"}, log=None)
    m1 = L.compute_kernel_md5(ws)
    (ws / "kernel" / "k.h").write_text("v2")
    m2 = L.compute_kernel_md5(ws)
    assert m1 != m2
    # rename-swap sensitivity: name is part of identity
    ws2 = _mk_workspace(tmp_path / "b", {"a.h": "X", "b.h": "Y"}, log=None)
    ma = L.compute_kernel_md5(ws2)
    (ws2 / "kernel" / "a.h").write_text("Y")
    (ws2 / "kernel" / "b.h").write_text("X")
    mb = L.compute_kernel_md5(ws2)
    assert ma != mb


def test_print_kernel_md5_matches_compute(tmp_path):
    """--print-kernel-md5 (write side) == compute_kernel_md5 (read side): the
    by-construction alignment that makes recorded md5s comparable.
    """
    ws = _mk_workspace(tmp_path, {"k.h": "abc", "m.cpp": "def"}, log=None)
    assert _print_md5_via_cli(ws) == L.compute_kernel_md5(ws)
