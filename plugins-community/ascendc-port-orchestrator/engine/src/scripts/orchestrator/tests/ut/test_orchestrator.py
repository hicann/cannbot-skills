# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Unit tests for orchestrator.py top-level helpers.

Run: python3 -m pytest src/scripts/orchestrator/tests/test_orchestrator.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import orchestrator as orch  # noqa: E402


def test_extract_canonical_handoff_done_at_end():
    """Worker emits markdown summary then `→ orchestrator: done` on final line."""
    stdout = (
        "All artifacts in place.\n\n"
        "## Final Status: DONE\n\n"
        "| Gate | Result |\n|---|---|\n| Build | PASS |\n\n"
        "→ orchestrator: done — 17_AdamW PASS (Pass A N/A; Pass B 9/9)"
    )
    assert orch.extract_canonical_handoff(stdout).startswith("→ orchestrator: done — 17_AdamW PASS")


def test_extract_canonical_handoff_done_with_trailing_blank_line():
    stdout = "Stuff.\n→ orchestrator: done — summary\n\n"
    assert orch.extract_canonical_handoff(stdout) == "→ orchestrator: done — summary"


def test_extract_canonical_handoff_partial_persist():
    stdout = "...\n→ orchestrator: PARTIAL_PERSIST — Tier-2 evidence in pass_b\n"
    assert orch.extract_canonical_handoff(stdout).startswith("→ orchestrator: PARTIAL_PERSIST")


def test_extract_canonical_handoff_at_probe():
    stdout = "kernel stuck\n@aog-precision-probe iter4 same signature\n"
    assert orch.extract_canonical_handoff(stdout).startswith("@aog-precision-probe")


def test_extract_canonical_handoff_at_optimizer():
    stdout = "perf 0.4x\n@aog-kernel-optimizer escalate\n"
    assert orch.extract_canonical_handoff(stdout).startswith("@aog-kernel-optimizer")


def test_extract_canonical_handoff_last_match_wins():
    """Multiple canonical lines — last one is the worker's final decision."""
    stdout = (
        "First attempt:\n"
        "→ orchestrator: done — partial early\n"
        "But then escalating because perf low.\n"
        "@aog-kernel-optimizer recover perf\n"
    )
    assert orch.extract_canonical_handoff(stdout).startswith("@aog-kernel-optimizer")


def test_extract_canonical_handoff_blocked_infra_falls_through():
    """`→ @orchestrator: BLOCKED_INFRA` is NOT a canonical handoff prefix.

    The `→ @orchestrator:` form is a worker-improvised state name (not in
    EXIT HANDOFF OPTIONS). Must NOT match `→ orchestrator:` (notice the
    `→ @` vs `→ ` distinction in the actual run-1 stdout 2026-05-04).
    Falling through to the full-text return is correct — the state machine
    will then hit its abort catch-all and the orchestrator routes to abort
    with a clear contract-violation message.
    """
    stdout = (
        "What was attempted...\n"
        "→ @orchestrator: BLOCKED_INFRA — sandbox denies bash invocation\n"
    )
    extracted = orch.extract_canonical_handoff(stdout)
    # Should NOT match — BLOCKED_INFRA is non-canonical
    assert not extracted.startswith("→ orchestrator: done")
    # Falls through to full text
    assert "BLOCKED_INFRA" in extracted


def test_extract_canonical_handoff_empty_input():
    assert orch.extract_canonical_handoff("") == ""


def test_extract_canonical_handoff_no_canonical_line_returns_full():
    stdout = "Random worker output\nWith no canonical handoff\nLine three\n"
    # Falls through to stripped full text
    assert orch.extract_canonical_handoff(stdout) == stdout.strip()


# ---------------------------------------------------------------------------
# P0h (Day 4 op#10 finding): markdown-wrapped handoff lines
# ---------------------------------------------------------------------------
def test_extract_canonical_handoff_markdown_bold_prefix_with_backticks():
    """op#10 kw-2 emitted: **Exit handoff**: `→ orchestrator: await_user_decision`
    — markdown bold prefix + backtick wrap. Was missed by V1 startswith().
    """
    stdout = (
        "Lots of analysis text...\n"
        "Final summary table here.\n"
        "**Exit handoff**: `→ orchestrator: await_user_decision`\n"
    )
    extracted = orch.extract_canonical_handoff(stdout)
    assert extracted.startswith("→ orchestrator: await_user_decision")
    # Trailing backtick must be stripped
    assert not extracted.endswith("`")


def test_extract_canonical_handoff_only_backticks():
    stdout = "stuff\n`→ orchestrator: done — summary`\n"
    assert orch.extract_canonical_handoff(stdout).startswith("→ orchestrator: done")


def test_extract_canonical_handoff_list_bullet_prefix():
    stdout = "Options:\n- → orchestrator: done — pass\n"
    assert orch.extract_canonical_handoff(stdout).startswith("→ orchestrator: done")


def test_extract_canonical_handoff_md_quoted_prefix():
    stdout = "> Final: → orchestrator: PARTIAL_PERSIST — evidence\n"
    extracted = orch.extract_canonical_handoff(stdout)
    assert extracted.startswith("→ orchestrator: PARTIAL_PERSIST")


def test_extract_canonical_handoff_substring_last_match_still_wins():
    """Multiple canonical lines (some markdown-wrapped) — last wins."""
    stdout = (
        "Tried `→ orchestrator: done` first, but perf low.\n"
        "**Exit handoff**: `@aog-kernel-optimizer escalate to ko`\n"
    )
    extracted = orch.extract_canonical_handoff(stdout)
    assert extracted.startswith("@aog-kernel-optimizer")


def test_extract_canonical_handoff_blocked_infra_still_falls_through():
    """Sanity: P0h substring search must NOT match `→ @orchestrator:`
    (which is a non-canonical worker improvisation, not a valid prefix).
    The `@` between `→` and `orchestrator:` is the discriminator.
    """
    stdout = "→ @orchestrator: BLOCKED_INFRA — stuff\n"
    extracted = orch.extract_canonical_handoff(stdout)
    # Must NOT have been promoted to canonical `→ orchestrator: ...`
    assert not extracted.startswith("→ orchestrator: done")
    # `@orchestrator:` IS a canonical prefix, however — extract from there
    assert extracted.startswith("@orchestrator:")


# ---------------------------------------------------------------------------
# P0s (2026-05-05 op#10 kw-2 finding): improvised wrapper handoff
# ---------------------------------------------------------------------------
def test_p0s_arrow_wrapper_with_inner_aog_returns_inner():
    """Worker mashed forms: `→ orchestrator: handoff to @aog-kernel-optimizer ...`
    The arrow wrapper is not a canonical arrow form (not done/PARTIAL_PERSIST/etc),
    but the inner @aog-X reference IS canonical — return it.
    """
    text = """\
some output...
→ orchestrator: handoff to @aog-kernel-optimizer per V3.8.4 routing (precision PASS + det PASS + perf 0.19× < 0.6× threshold).
"""
    h = orch.extract_canonical_handoff(text)
    assert h.startswith("@aog-kernel-optimizer"), \
        f"Expected inner @aog-kernel-optimizer extracted, got {h!r}"


def test_p0s_arrow_wrapper_with_inner_probe():
    text = "→ orchestrator: handoff to @aog-precision-probe with signature=fp32-tanh"
    h = orch.extract_canonical_handoff(text)
    assert h.startswith("@aog-precision-probe")


def test_p0s_valid_arrow_done_unchanged():
    """Canonical `→ orchestrator: done — ...` is NOT wrapped — return as-is."""
    text = "→ orchestrator: done — Pass A 60/60, perf 0.65×"
    h = orch.extract_canonical_handoff(text)
    assert h == "→ orchestrator: done — Pass A 60/60, perf 0.65×"


def test_p0s_valid_arrow_partial_persist_unchanged():
    text = "→ orchestrator: PARTIAL_PERSIST — OL-110 evidence cited"
    h = orch.extract_canonical_handoff(text)
    assert h.startswith("→ orchestrator: PARTIAL_PERSIST")


def test_p0s_valid_arrow_user_decision_unchanged():
    text = "→ orchestrator: await_user_decision — research path needed"
    h = orch.extract_canonical_handoff(text)
    assert h.startswith("→ orchestrator: await_user_decision")


def test_p0s_arrow_wrapper_no_inner_agent_falls_through():
    """`→ orchestrator: <gibberish>` with no inner @aog-X — return the line.
    State machine will then route to abort (correct behavior for malformed).
    """
    text = "→ orchestrator: please do something nice"
    h = orch.extract_canonical_handoff(text)
    # Returns the line (will fail downstream startswith for any valid form)
    assert h == "→ orchestrator: please do something nice"


def test_p0s_research_done_arrow_form_unchanged():
    """`→ orchestrator: research_done` is a valid researcher exit form."""
    text = "→ orchestrator: research_done — algorithm = X, directive at .../optimization_directive.md"
    h = orch.extract_canonical_handoff(text)
    assert h.startswith("→ orchestrator: research_done")


def test_agent_timeout_dispatch():
    """P0aau: DS/V4 backends get 10800s; A5/default get 9000s.
    (A5 was bumped 5400->9000 on 2026-06-11 for the backward input_gen 90-case
    sweep — orchestrator.py:_DEFAULT_AGENT_TIMEOUT_SEC_A5; will revert to 5400
    once input_gen gets a per-op case-budget cap. Test tracks the code intent.)
    """
    from orchestrator import _agent_timeout_for_target
    assert _agent_timeout_for_target("a5") == 9000
    assert _agent_timeout_for_target("a3-ds") == 10800
    assert _agent_timeout_for_target("a3") == 9000
    assert _agent_timeout_for_target("v4-something") == 10800
    assert _agent_timeout_for_target("") == 9000
    # _resolve_env returns None on error → empty string → 9000 (safe fail-closed)


# ---------------------------------------------------------------------------
# P0aaz: dynamic lane detection via npu-smi
# ---------------------------------------------------------------------------
_NPU_SMI_A2_5NPU = """+------------------------------------------------------------------------------------------------+
| npu-smi 24.1.rc2                 Version: 24.1.rc2                                             |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B2C              | Alarm         | 92.6        50                0    / 0             |
| 0                         | 0000:5A:00.0  | 0           0    / 0          3521 / 65536         |
+===========================+===============+====================================================+
| 1     910B2C              | OK            | 103.0       53                0    / 0             |
| 1                         | 0000:19:00.0  | 0           0    / 0          3395 / 65536         |
+===========================+===============+====================================================+
| 2     910B2C              | OK            | 96.7        53                0    / 0             |
| 2                         | 0000:49:00.0  | 0           0    / 0          3389 / 65536         |
+===========================+===============+====================================================+
| 3     910B2C              | OK            | 99.5        52                0    / 0             |
| 3                         | 0000:39:00.0  | 0           0    / 0          3410 / 65536         |
+===========================+===============+====================================================+
| 4     910B2C              | OK            | 90.5        53                0    / 0             |
| 4                         | 0000:DA:00.0  | 0           0    / 0          3392 / 65536         |
+===========================+===============+====================================================+"""

_NPU_SMI_A5_3NPU = """+------------------------------------------------------------------------------------------------+
| npu-smi 24.1.rc2                 Version: 24.1.rc2                                             |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     Ascend950PR         | OK            | 150.0       60                0    / 0             |
| 0                         | 0000:01:00.0  | 0           0    / 0          1000 / 65536         |
+===========================+===============+====================================================+
| 1     Ascend950PR         | OK            | 150.0       60                0    / 0             |
| 1                         | 0000:02:00.0  | 0           0    / 0          1000 / 65536         |
+===========================+===============+====================================================+
| 2     Ascend950PR         | OK            | 150.0       60                0    / 0             |
| 2                         | 0000:03:00.0  | 0           0    / 0          1000 / 65536         |
+===========================+===============+====================================================+"""

_NPU_SMI_SINGLE = """+------------------------------------------------------------------------------------------------+
| npu-smi 24.1.rc2                 Version: 24.1.rc2                                             |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B2C              | OK            | 90.5        53                0    / 0             |
| 0                         | 0000:DA:00.0  | 0           0    / 0          3392 / 65536         |
+===========================+===============+====================================================+"""


# npu-smi 25.7.rc1 (independent review_iouv2 on 203.0.113.141): pipe-SEPARATED columns
# (`| 0      | Ascend950PR |`) + a process table whose 2nd column is a PID.
# The pre-fix regex `^\|\s*(\d+)\s+\w+` matched 0 device rows here (the char
# after the ID is `|`, not a word char) → "parse yielded 0 NPUs" → fallback
# cap=4 → lane 6 wrongly rejected. 7 NPUs (0-6) → max lane MUST be 6.
_NPU_SMI_25RC1_PIPE_7NPU = """\
+-------------------------------------------------------------------------------------------------+
| npu-smi 25.7.rc1                                 Version: 25.7.rc1                              |
+--------+------------------+---------------+-----------------------------------------------------+
| NPU ID | Name             | Health        | Power(W)    Temp(C)           Hugepages-Usage(page) |
|        |                  | Bus-Id        | NPU Util(%) Memory-Usage(MB)  HBM-Usage(MB)         |
+========+==================+===============+=====================================================+
| 0      | Ascend950PR      | OK            | 187.1       46                0     / 0             |
|        |                  | 0000:01:00.0  | 0           0    / 0          6620  / 131072        |
+========+==================+===============+=====================================================+
| 1      | Ascend950PR      | OK            | 197.9       48                0     / 0             |
|        |                  | 0000:11:00.0  | 0           0    / 0          6056  / 131072        |
+========+==================+===============+=====================================================+
| 2      | Ascend950PR      | OK            | 188.7       46                0     / 0             |
|        |                  | 0000:61:00.0  | 0           0    / 0          5413  / 131072        |
+========+==================+===============+=====================================================+
| 3      | Ascend950PR      | OK            | 191.3       43                0     / 0             |
|        |                  | 0000:71:00.0  | 0           0    / 0          5414  / 131072        |
+========+==================+===============+=====================================================+
| 4      | Ascend950PR      | OK            | 196.6       50                0     / 0             |
|        |                  | 0000:81:00.0  | 0           0    / 0          5415  / 131072        |
+========+==================+===============+=====================================================+
| 5      | Ascend950PR      | OK            | 195.2       44                0     / 0             |
|        |                  | 0000:91:00.0  | 0           0    / 0          10569 / 131072        |
+========+==================+===============+=====================================================+
| 6      | Ascend950PR      | OK            | 195.8       46                0     / 0             |
|        |                  | 0000:F1:00.0  | 0           0    / 0          5411  / 131072        |
+========+==================+===============+=====================================================+
+---------------------------+---------------+-----------------------------------------------------+
| NPU ID                    | Process id    | Process name             | Process memory(MB)       |
+===========================+===============+=====================================================+
| 0                         | 846718        |                          | 332                     |
+===========================+===============+=====================================================+
| 1                         | 2430352       |                          | 544                     |
+===========================+===============+=====================================================+
| No running processes found in NPU 2                                                             |
+===========================+===============+=====================================================+"""


def _patch_fake_env(monkeypatch, target="a5"):
    """Make `_detect_max_lane` env-INDEPENDENT for the npu-smi PARSE-path tests.

    `_detect_max_lane` starts with `try: load_env() except: return 2`. In a
    clean checkout / CI / the pre-commit hook the gitignored
    `workspace/.ascendc_env` is ABSENT, so `load_env()` raises and the
    function early-returns 2 BEFORE it ever reaches the (faked) npu-smi parse
    — making the parse-path assertions fail with `2 != <expected>`. These
    tests are about the PARSE logic, not env loading, so we inject a fake env
    (target only; `host` defaults to '' → local path, and subprocess.run is
    faked anyway). Mirrors the p135lf fallback tests' load_env monkeypatch."""
    import importlib
    fake_env = type("Env", (), {})()
    fake_env.target = target
    briefs_common = importlib.import_module("briefs._common")
    monkeypatch.setattr(briefs_common, "load_env", lambda: fake_env)


def test_detect_max_lane_25rc1_pipe_7npu(monkeypatch):
    """Parses npu-smi 25.7.rc1 pipe-separated output (7 NPUs) → max lane 6.

    Regression for the lane-detection harness gap: the pre-fix regex yielded
    0 NPUs on this layout → fallback cap=4 → a build-root lane (6) was
    rejected by `--lane must be 0-N` validation. The process table (PID 2nd
    column) must NOT be counted as devices.
    """
    import subprocess
    _patch_fake_env(monkeypatch)
    fake_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=_NPU_SMI_25RC1_PIPE_7NPU)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: fake_result)
    from orchestrator import _detect_max_lane
    assert _detect_max_lane() == 6


def test_detect_max_lane_a2_5npu(monkeypatch):
    """Parses A2 npu-smi output → max lane 4."""
    import subprocess
    _patch_fake_env(monkeypatch, target="a2")
    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout=_NPU_SMI_A2_5NPU)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: fake_result)
    from orchestrator import _detect_max_lane
    assert _detect_max_lane() == 4


def test_detect_max_lane_a5_3npu(monkeypatch):
    """Parses A5 npu-smi output → max lane 2."""
    import subprocess
    _patch_fake_env(monkeypatch)
    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout=_NPU_SMI_A5_3NPU)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: fake_result)
    from orchestrator import _detect_max_lane
    assert _detect_max_lane() == 2


def test_detect_max_lane_single_npu(monkeypatch):
    """Single NPU → max lane 0."""
    import subprocess
    _patch_fake_env(monkeypatch)
    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout=_NPU_SMI_SINGLE)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: fake_result)
    from orchestrator import _detect_max_lane
    assert _detect_max_lane() == 0


def test_detect_max_lane_npu_smi_fails_fallback(monkeypatch):
    """When npu-smi fails, fall back to per-target default.
    P135.LF (2026-05-18 task #16): A5 fallback bumped 2→4 (5 NPUs: 0-4).
    """
    import subprocess
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("npu-smi not found")),
    )
    from orchestrator import _detect_max_lane
    # Falls back based on detected env target; env mock not imported so we
    # exercise the exception path which falls through to env loading. If env
    # loading also fails, default is 2.
    result = _detect_max_lane()
    assert result in (2, 4)  # both are valid fallbacks (env-dependent)


def test_detect_max_lane_empty_output_fallback(monkeypatch):
    """npu-smi returns empty → fall back to defaults."""
    import subprocess
    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: fake_result)
    from orchestrator import _detect_max_lane
    result = _detect_max_lane()
    assert result in (2, 4)


# ── P135.LF (2026-05-18 task #16): A5 fallback bumped 2→4 ──


def test_p135lf_a5_fallback_is_4_after_bump(monkeypatch):
    """P135.LF: when env target=a5 + npu-smi probe fails, fallback returns 4
    (max lane index for 5 NPUs: 0,1,2,3,4). Old fallback was 2 (assumed
    3 NPUs). Caught 2026-05-18: erfinv cold-start hit cap=2 because
    A5 host had no working npu-smi (libc_sec.so missing from shell env),
    even though hardware has 5 NPUs. After bump: 4 = correct max.
    """
    import subprocess
    # Force probe failure → exception path
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("npu-smi not found")),
    )
    # Force env target to a5
    fake_env = type("Env", (), {})()
    fake_env.target = "a5"

    def _fake_load_env():
        return fake_env

    # Patch the briefs._common.load_env path used by _detect_max_lane
    import importlib
    briefs_common = importlib.import_module("briefs._common")
    monkeypatch.setattr(briefs_common, "load_env", _fake_load_env)

    from orchestrator import _detect_max_lane
    result = _detect_max_lane()
    assert result == 4, f"P135.LF: A5 fallback must be 4 (5 NPUs), got {result}"


def test_p135lf_a3_fallback_unchanged(monkeypatch):
    """P135.LF: A3 fallback stays at 2 (the fix is A5-specific)."""
    import subprocess
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("npu-smi not found")),
    )
    fake_env = type("Env", (), {})()
    fake_env.target = "a3"

    import importlib
    briefs_common = importlib.import_module("briefs._common")
    monkeypatch.setattr(briefs_common, "load_env", lambda: fake_env)

    from orchestrator import _detect_max_lane
    result = _detect_max_lane()
    assert result == 2, f"P135.LF: A3 fallback must remain 2, got {result}"


def test_p135lf_a2_fallback_unchanged(monkeypatch):
    """P135.LF: A2 fallback stays at 4 (already correct for 5 NPUs)."""
    import subprocess
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("npu-smi not found")),
    )
    fake_env = type("Env", (), {})()
    fake_env.target = "a2"

    import importlib
    briefs_common = importlib.import_module("briefs._common")
    monkeypatch.setattr(briefs_common, "load_env", lambda: fake_env)

    from orchestrator import _detect_max_lane
    result = _detect_max_lane()
    assert result == 4, f"P135.LF: A2 fallback must remain 4, got {result}"
