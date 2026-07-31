# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""UT for phase_o5_helpers.py — the pure leaf helpers relocated out of
phase_o5_runner.py in the DEBT-201 god-file decomposition (2026-07-06).

Two responsibilities:
  1. Fill the direct-coverage gaps for helpers that had no dedicated UT before
     the split (`_resolve_ssh_key_opts`, `_shell_quote`) plus a behaviour anchor
     for each other relocated helper.
  2. Lock the split as BEHAVIOR-NEUTRAL: every relocated name must remain
     reachable via `phase_o5_runner.<name>` AND be the SAME object as the
     definition in phase_o5_helpers (the bottom re-import shim, not a copy),
     so that `monkeypatch.setattr(phase_o5_runner, ...)` semantics for the
     NON-relocated (patched) functions are unaffected.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # orchestrator/

import phase_o5_helpers as h  # noqa: E402
import phase_o5_runner as por  # noqa: E402
import phase_o5_verify as pv  # noqa: E402


_RELOCATED = (
    "_find_verifier",
    "_lane_aware_benchmark_root",
    "_normalize_canonical_pass_a",
    "_normalize_port_a3_two_tier_pass_a",
    "_normalize_verifier_output",
    "_resolve_extra_ld",
    "_resolve_npu_python_bin",
    "_resolve_ssh_key_opts",
    "_shell_quote",
    "_try_fetch_remote_result_json",
    "_try_parse_json_tail",
)


# ── split-neutrality locks ───────────────────────────────────────────────────
def test_relocated_names_are_reexported_same_object() -> None:
    """phase_o5_runner.<name> must BE phase_o5_helpers.<name> (shim, not copy)."""
    for name in _RELOCATED:
        assert hasattr(por, name), f"phase_o5_runner lost attribute {name}"
        assert getattr(por, name) is getattr(h, name), (
            f"{name} in phase_o5_runner is a different object than in "
            "phase_o5_helpers — the split copied instead of re-importing"
        )


def test_patched_functions_reachable_via_parent_module() -> None:
    """The monkeypatched functions MUST stay reachable as
    `phase_o5_runner.<name>` so `monkeypatch.setattr(phase_o5_runner, ...)`
    keeps biting. DEBT-201 batch5 (2026-07-06) moved the verifier-execution
    cluster to phase_o5_verify.py; those names are re-imported (bottom shim)
    into phase_o5_runner, so the patch surface `phase_o5_runner.<name>` is
    unchanged. The bite is preserved because (a) this module's parent-resident
    callers use the bare re-imported name (resolves to the patched attribute),
    and (b) the moved functions call back into patched functions via the
    qualified `phase_o5_runner.<name>` form.

    STAY in phase_o5_runner (parent-resident, patched here):
    """
    stay_in_parent = (
        "ssh_runner", "_resync_workspace_to_container", "_read_ascendc_env",
        "_port_a3_claims_pass_a", "_is_port_a3_mode",
    )
    for name in stay_in_parent:
        fn = getattr(por, name)
        assert fn.__module__ == "phase_o5_runner", (
            f"{name} unexpectedly moved out of phase_o5_runner "
            f"(__module__={fn.__module__})"
        )
    # MOVED to phase_o5_verify but re-exported (same object) so the patch
    # surface phase_o5_runner.<name> is preserved.
    moved_to_verify = (
        "_run_verifier", "_run_verifier_local", "_run_canonical_pass_a",
        "_run_canonical_pass_a_local", "_verify_runner_independence",
        "_gate_port_a3_two_tier",
    )
    for name in moved_to_verify:
        fn = getattr(por, name)
        assert fn.__module__ == "phase_o5_verify", (
            f"{name} expected in phase_o5_verify (__module__={fn.__module__})"
        )
        assert fn is getattr(pv, name), (
            f"phase_o5_runner.{name} is not the same object as "
            f"phase_o5_verify.{name} — the re-export drifted, patch surface broken"
        )


# ── _resolve_ssh_key_opts (previously no direct UT) ──────────────────────────
def test_resolve_ssh_key_opts_target_specific_wins() -> None:
    env = {"A5_SSH_KEY": "/keys/a5", "SSH_KEY": "/keys/generic"}
    assert getattr(h, '_resolve_ssh_key_opts')(env, "A5") == ["-i", "/keys/a5"]


def test_resolve_ssh_key_opts_generic_fallback() -> None:
    env = {"SSH_KEY": "/keys/generic"}
    assert getattr(h, '_resolve_ssh_key_opts')(env, "A5") == ["-i", "/keys/generic"]


def test_resolve_ssh_key_opts_empty_when_unset() -> None:
    assert getattr(h, '_resolve_ssh_key_opts')({}, "A5") == []
    # whitespace-only is treated as unset
    assert getattr(h, '_resolve_ssh_key_opts')({"SSH_KEY": "   "}, "A3") == []


# ── _shell_quote (previously no direct UT) ───────────────────────────────────
def test_shell_quote_plain() -> None:
    assert getattr(h, '_shell_quote')("abc") == "'abc'"


def test_shell_quote_embedded_single_quote() -> None:
    # POSIX-safe: close quote, escaped literal quote, reopen quote.
    assert getattr(h, '_shell_quote')("a'b") == "'a'\\''b'"


def test_shell_quote_empty() -> None:
    assert getattr(h, '_shell_quote')("") == "''"


# ── behaviour anchors for the other relocated helpers ────────────────────────
def test_resolve_npu_python_bin_prefers_target() -> None:
    env = {"A3_NPU_PYTHON_BIN": "/py/a3/", "NPU_PYTHON_BIN": "/py/generic"}
    assert getattr(h, '_resolve_npu_python_bin')(env, "A3") == "/py/a3"  # rstrip('/')
    assert getattr(h, '_resolve_npu_python_bin')({}, "A5") == ""


def test_resolve_extra_ld_prefers_target_and_rstrips_colon() -> None:
    env = {"A5_EXTRA_LD_LIBRARY_PATH": "/lib/a5:", "EXTRA_LD_LIBRARY_PATH": "/lib/gen"}
    assert getattr(h, '_resolve_extra_ld')(env, "A5") == "/lib/a5"
    assert getattr(h, '_resolve_extra_ld')({}, "A5") == ""


def test_find_verifier_returns_first_existing(tmp_path: Path) -> None:
    (tmp_path / "pass_b_runner.py").write_text("x")
    assert getattr(h, '_find_verifier')(tmp_path, ["run_pass_b.py", "pass_b_runner.py"]) == "pass_b_runner.py"
    assert getattr(h, '_find_verifier')(tmp_path, ["nope.py"]) is None


def test_try_parse_json_tail_handles_trailing_noise() -> None:
    stdout = 'log line\n{"tier1_pass": 3, "total": 3}\n[Warning]: tiling struct conflict'
    parsed = getattr(h, '_try_parse_json_tail')(stdout)
    assert parsed == {"tier1_pass": 3, "total": 3}
    assert getattr(h, '_try_parse_json_tail')("") is None
    assert getattr(h, '_try_parse_json_tail')("no json here") is None


def test_normalize_verifier_output_routes_two_tier_on_tier2_status() -> None:
    parsed = {"tier2_status": "PASS", "total": 5, "tier1_pass": 4,
              "tier2_pass": 1, "status": "PASS"}
    out = getattr(h, '_normalize_verifier_output')(parsed, "pass_a")
    assert out["method"] == "canonical_precision_eval_port_a3_two_tier"
    assert out["tier1_pass"] == 4 and out["tier2_pass"] == 1


def test_normalize_verifier_output_legacy_npass_mapping() -> None:
    out = getattr(h, '_normalize_verifier_output')({"n_pass": 7, "n_total": 8}, "pass_b")
    assert out["tier1_pass"] == 7 and out["total"] == 8


def test_normalize_canonical_pass_a_t1_t2_axis() -> None:
    parsed = {"n_total": 10, "n_pass_t1_inclusive": 6, "n_pass_t2": 4,
              "n_fail": 0, "n_err": 0, "tier_axis": "T1_T2"}
    out = getattr(h, '_normalize_canonical_pass_a')(parsed)
    assert out["tier1_pass"] == 10 and out["status"] == "PASS"


def test_normalize_canonical_pass_a_t3_axis_and_fail() -> None:
    parsed = {"n_total": 4, "n_pass_t3": 3, "n_fail": 1, "n_err": 0, "tier_axis": "T3"}
    out = getattr(h, '_normalize_canonical_pass_a')(parsed)
    assert out["tier1_pass"] == 3 and out["status"] == "FAIL"


def test_normalize_port_a3_two_tier_selects_strict_and_inclusive() -> None:
    parsed = {"total": 9, "tier1_pass": 5, "tier2_pass": 3,
              "tier1_pass_inclusive": 8, "status": "PASS", "tier2_status": "PASS"}
    out = getattr(h, '_normalize_port_a3_two_tier_pass_a')(parsed)
    assert out["tier1_pass"] == 5              # STRICT
    assert out["tier1_pass_inclusive"] == 8    # INCLUSIVE
    assert out["tier2_pass"] == 3


def test_try_fetch_remote_result_json_parses_stdout() -> None:
    class _R:
        returncode = 0
        stdout = '{"ok": true}'

    def _fake_run(cmd, **kw):
        return _R()

    out = getattr(h, '_try_fetch_remote_result_json')(
        "host", "user", "pw", "cont", "/root", "pass_b", [],
        _subprocess_run=_fake_run,
    )
    assert out == {"ok": True}


def test_try_fetch_remote_result_json_none_on_nonzero_rc() -> None:
    class _R:
        returncode = 1
        stdout = ""

    out = getattr(h, '_try_fetch_remote_result_json')(
        "host", "user", "pw", "cont", "/root", "pass_b", [],
        _subprocess_run=lambda cmd, **kw: _R(),
    )
    assert out is None


# ── phase_o5_verify (DEBT-201 batch5) coverage + patch-bite proofs ───────────
def test_verify_gate_port_a3_two_tier_inert_when_not_port_a3(tmp_path, monkeypatch):
    """_gate_port_a3_two_tier (moved to phase_o5_verify) is inert for non-port_a3.
    It calls _is_port_a3_mode via the qualified phase_o5_runner.<name> form, so
    patching por._is_port_a3_mode MUST bite the moved function.
    """
    monkeypatch.setattr(por, "_is_port_a3_mode", lambda ws: False)
    assert getattr(pv, '_gate_port_a3_two_tier')(tmp_path, {"tier1_pass": 5}) is None


def test_verify_gate_port_a3_two_tier_trips_on_missing_tier2(tmp_path, monkeypatch):
    """port_a3 + dict pass_a WITHOUT tier2_status → masquerade error string.
    Proves the por._is_port_a3_mode patch reaches the moved gate (qualified call).
    """
    monkeypatch.setattr(por, "_is_port_a3_mode", lambda ws: True)
    err = getattr(pv, '_gate_port_a3_two_tier')(tmp_path, {"tier1_pass": 5, "total": 5})
    assert err is not None and "two-tier" in err


def test_verify_runner_independence_moved_and_bites(tmp_path):
    """_verify_runner_independence moved to phase_o5_verify; still reachable via
    por.<name> (same object) and functions correctly on a self-citing script.
    """
    (tmp_path / "pass_a_runner.py").write_text(
        "import json\nd = json.load(open('verification.json'))\nprint(d)\n"
    )
    err = getattr(pv, '_verify_runner_independence')(tmp_path, "pass_a_runner.py")
    assert err is not None and "verification.json" in err
    assert getattr(por, '_verify_runner_independence') is getattr(pv, '_verify_runner_independence')


def test_verify_run_canonical_pass_a_local_qualified_ispa3(tmp_path, monkeypatch):
    """_run_canonical_pass_a_local (moved) reads _is_port_a3_mode via the
    qualified phase_o5_runner form. Patch por._is_port_a3_mode → the moved fn
    sees it; with model.py missing it returns the missing-model error string.
    """
    calls = {"n": 0}

    def _fake(ws):
        calls["n"] += 1
        return True

    monkeypatch.setattr(por, "_is_port_a3_mode", _fake)
    out = getattr(pv, '_run_canonical_pass_a_local')(tmp_path, "op", {}, lane=0)
    assert calls["n"] >= 1, "patch por._is_port_a3_mode did NOT bite moved fn"
    assert isinstance(out, str) and "model.py missing" in out
