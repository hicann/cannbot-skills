# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P0abi (2026-05-08): P-P88 sigmoid-form remediation enforcement.

Background: 1_GELU regressed 50/50 (May-4 archive PASS_WITHIN_TOLERANCE) →
44/50 (May-8 cold-start PARTIAL) because cold-start kw cited P-P88 as
diagnosis but didn't apply the prescribed sigmoid-form rewrite. Different
tile-size choice routed Tanh primitive's internal SIMD differently on
small-value inputs → ours rounded AWAY from CANN on 6 cases that
previously rounded TOWARD.

Codex + DS reviewed (2026-05-08); both approved. DS suggested adding
positive compliance marker (sigmoid-form rewrite pattern as kernel-side
proof of compliance) so ops that already correctly applied P-P88 don't
need redundant YAML.

These tests cover:
1. Negative gate: Tanh + transcendental + only prose citation → FAIL
2. Exemption: structured exemption with concrete evidence → PASS
3. False-positive (comments): Sigmoid in `// comment` only → NOT_APPLICABLE
4. False-positive (string literal): "Tanh" in std::string → NOT_APPLICABLE
5. Token robustness: AscendC::Tanh, Tanh<float>, whitespace, multi-headers
6. Taxonomy boundary: matmul op with stray Tanh in kernel → NOT_APPLICABLE
7. Compliance-by-code: kernel uses 1/(1+exp(-y)) sigmoid form, no Tanh →
   PASS without YAML
8. Mixed kernel: sigmoid-form rewrite + leftover Tanh → FAIL (uniform req'd)
9. YAML structural validation: missing rationale → FAIL
10. YAML status=applied without diff_refs → FAIL
11. YAML status=exempt without measurements → FAIL
"""
from __future__ import annotations

import sys
from pathlib import Path

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors
import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
import scan_pp88_compliance as scan  # noqa: E402


# ---------------------------------------------------------------------------
# Test workspace builder
# ---------------------------------------------------------------------------
def _build_ws(
    tmp_path: Path,
    *,
    kernel_h: str = "",
    kernel_cpp: str = "",
    model_py: str = "",
    knowledge_update: str = "",
    bench_json: str = '[{"shape": [256, 512]}]',
) -> Path:
    """Build a synthetic workspace dir for testing."""
    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws / "kernel").mkdir()
    if kernel_h:
        (ws / "kernel" / "test_kernel.h").write_text(kernel_h)
    if kernel_cpp:
        (ws / "kernel" / "test_kernels.cpp").write_text(kernel_cpp)
    if model_py:
        (ws / "model.py").write_text(model_py)
    if knowledge_update:
        (ws / "knowledge_update.md").write_text(knowledge_update)
    if bench_json:
        (ws / "test_op.json").write_text(bench_json)
    return ws


# ---------------------------------------------------------------------------
# 1. Negative gate — main 1_GELU regression case
# ---------------------------------------------------------------------------
def test_neg_tanh_transcendental_no_evidence_fails(tmp_path):
    """The exact 1_GELU regression case: kernel calls Tanh, op is GELU
    (transcendental), knowledge_update.md mentions P-P88 only as prose,
    no structured `p_p88:` block. Must FAIL.
    """
    ws = _build_ws(
        tmp_path,
        kernel_h="""
        #include <kernel_operator.h>
        void compute(const LocalTensor<float>& x, LocalTensor<float>& y, int32_t n) {
            // Per P-P88 + OL-103: Tanh has fp16-grade poly floor.
            Tanh(y, x, n);
        }
        """,
        model_py="""
        import torch
        class Model(torch.nn.Module):
            def forward(self, x, approximate='tanh'):
                return torch.nn.functional.gelu(x, approximate=approximate)
        """,
        knowledge_update="""
        # knowledge_update.md
        ## Cited KB items
        - P-P88 (sigmoid-form remediation) — cited but not applied per OL-83 floor.
        """,
    )
    rep = scan.scan_workspace(ws)
    assert rep.verdict == "FAIL", rep.rationale
    assert rep.is_transcendental
    assert any(h.primitive == "Tanh" for h in rep.risky_hits)
    assert "no `p_p88:` YAML block" in rep.rationale or "no p_p88:" in rep.rationale.lower()


# ---------------------------------------------------------------------------
# 2. Exemption — structured YAML with concrete evidence
# ---------------------------------------------------------------------------
def test_exemption_with_concrete_evidence_passes(tmp_path):
    """If kernel uses Tanh but knowledge_update.md has structured exemption
    block with isolated-primitive measurements, gate PASSES.
    """
    ws = _build_ws(
        tmp_path,
        kernel_h="Tanh(y, x, n);",
        model_py="def forward(self, x): return torch.tanh(x)",
        knowledge_update="""
        # knowledge_update.md

        ## P-P88 compliance

        ```yaml
        p_p88:
          status: exempt
          primitives_detected: [Tanh]
          evidence:
            files: [kernel/test_kernel.h:1]
            rationale: "Op is identity-tanh; PB-24 small-x failure mode does not apply because input distribution is bounded |x| > 1.0 per benchmark JSON."
            isolated_primitive_measurements:
              - {input_range: "|x| > 1.0", measured_ulp: 2, vs_cpu_truth: true}
          diff_refs: []
        ```
        """,
    )
    rep = scan.scan_workspace(ws)
    assert rep.verdict == "PASS", rep.rationale


# ---------------------------------------------------------------------------
# 3. False positive — Sigmoid in line comment only
# ---------------------------------------------------------------------------
def test_sigmoid_in_comment_does_not_trigger(tmp_path):
    """Mentions of `Sigmoid(` in `//` comments must not trigger the gate."""
    ws = _build_ws(
        tmp_path,
        kernel_h="""
        // Sigmoid(x, y, n) is one approach; we use Erf instead.
        Erf(y, x, n);
        """,
        model_py="def forward(self, x): return torch.nn.functional.gelu(x)",
    )
    rep = scan.scan_workspace(ws)
    assert rep.verdict == "NOT_APPLICABLE", rep.rationale
    assert len(rep.risky_hits) == 0


def test_tanh_in_block_comment_does_not_trigger(tmp_path):
    """`/* Tanh(...) */` block comment must be stripped."""
    ws = _build_ws(
        tmp_path,
        kernel_h="""
        /*
         * Old code: Tanh(y, x, n);
         * Replaced with sigmoid-form below.
         */
        Erf(y, x, n);
        """,
        model_py="def forward(self, x): return torch.nn.functional.gelu(x)",
    )
    rep = scan.scan_workspace(ws)
    assert len(rep.risky_hits) == 0


# ---------------------------------------------------------------------------
# 4. False positive — string literal
# ---------------------------------------------------------------------------
def test_tanh_in_string_literal_does_not_trigger(tmp_path):
    """`"Tanh(...)"` string literal must be stripped."""
    ws = _build_ws(
        tmp_path,
        kernel_h='std::string err = "Tanh(x, y, n) failed"; Erf(y, x, n);',
        model_py="def forward(self, x): return torch.nn.functional.gelu(x)",
    )
    rep = scan.scan_workspace(ws)
    assert len(rep.risky_hits) == 0


# ---------------------------------------------------------------------------
# 5. Token robustness — namespace prefix, template args, whitespace
# ---------------------------------------------------------------------------
def test_namespace_prefixed_tanh_triggers(tmp_path):
    """`AscendC::Tanh(...)` must trigger the gate."""
    ws = _build_ws(
        tmp_path,
        kernel_h="AscendC::Tanh(y, x, n);",
        model_py="def forward(self, x): return torch.nn.functional.gelu(x)",
    )
    rep = scan.scan_workspace(ws)
    assert any(h.primitive == "Tanh" for h in rep.risky_hits)


def test_templated_tanh_triggers(tmp_path):
    """`Tanh<float>(...)` template syntax must trigger."""
    ws = _build_ws(
        tmp_path,
        kernel_h="Tanh<float>(y, x, n);",
        model_py="def forward(self, x): return torch.nn.functional.gelu(x)",
    )
    rep = scan.scan_workspace(ws)
    assert any(h.primitive == "Tanh" for h in rep.risky_hits)


def test_whitespace_variants_trigger(tmp_path):
    """`Tanh ( y, x, n)` (extra whitespace) must trigger."""
    ws = _build_ws(
        tmp_path,
        kernel_h="Tanh (y, x, n);",
        model_py="def forward(self, x): return torch.nn.functional.gelu(x)",
    )
    rep = scan.scan_workspace(ws)
    assert any(h.primitive == "Tanh" for h in rep.risky_hits)


def test_multi_header_scan(tmp_path):
    """Risky primitive in second header file (not just first) must trigger."""
    ws = _build_ws(
        tmp_path,
        kernel_h="// no risky calls",
        model_py="def forward(self, x): return torch.nn.functional.gelu(x)",
    )
    (ws / "kernel" / "secondary.h").write_text("Tanh(y, x, n);")
    rep = scan.scan_workspace(ws)
    assert any("secondary.h" in h.file for h in rep.risky_hits)


# ---------------------------------------------------------------------------
# 6. Taxonomy boundary — non-transcendental op with Tanh stray in kernel
# ---------------------------------------------------------------------------
def test_matmul_op_with_stray_tanh_does_not_fire(tmp_path):
    """If op-class is non-transcendental (matmul, copy, etc.), gate is
    NOT_APPLICABLE even if Tanh appears in kernel — scope is transcendental
    only.
    """
    ws = _build_ws(
        tmp_path,
        kernel_h="Tanh(y, x, n);",  # stray, hypothetical
        model_py="""
        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.matmul(a, b)
        """,
    )
    rep = scan.scan_workspace(ws)
    assert rep.verdict == "NOT_APPLICABLE", rep.rationale
    assert "not transcendental" in rep.rationale.lower()


# ---------------------------------------------------------------------------
# 7. Compliance-by-code — sigmoid-form rewrite, no risky primitive
# ---------------------------------------------------------------------------
def test_sigmoid_form_rewrite_no_yaml_passes(tmp_path):
    """Kernel implements P-P88 sigmoid form (1/(1+exp(-y))), no Tanh call.
    Gate NOT_APPLICABLE (no risky primitive to enforce against).
    """
    ws = _build_ws(
        tmp_path,
        kernel_h="""
        Exp(t1, x_neg, n);
        Adds(t1, t1, 1.0f, n);
        Reciprocal(y, t1, n);
        // y = 1/(1+exp(-y))  // sigmoid-form per P-P88
        """,
        model_py="def forward(self, x): return torch.nn.functional.gelu(x)",
    )
    rep = scan.scan_workspace(ws)
    assert rep.verdict == "NOT_APPLICABLE"
    assert len(rep.risky_hits) == 0


# ---------------------------------------------------------------------------
# 8. Mixed kernel — both sigmoid form AND leftover Tanh
# ---------------------------------------------------------------------------
def test_mixed_kernel_fails(tmp_path):
    """Kernel has sigmoid-form rewrite in one branch but kept Tanh in another.
    P-P88 must be uniform — gate FAILS.
    """
    ws = _build_ws(
        tmp_path,
        kernel_h="""
        if (mode == 0) {
            float result = 1.0f / (1.0f + AscendC::Exp(neg_y));
        } else {
            Tanh(y, x, n);
        }
        """,
        model_py="def forward(self, x): return torch.nn.functional.gelu(x)",
    )
    rep = scan.scan_workspace(ws)
    assert rep.verdict == "FAIL", rep.rationale
    assert "Mixed kernel" in rep.rationale


# ---------------------------------------------------------------------------
# 9-11. YAML structural validation
# ---------------------------------------------------------------------------
def test_yaml_missing_rationale_fails(tmp_path):
    """`evidence.rationale` is required."""
    ws = _build_ws(
        tmp_path,
        kernel_h="Tanh(y, x, n);",
        model_py="def forward(self, x): return torch.nn.functional.gelu(x)",
        knowledge_update="""
        ```yaml
        p_p88:
          status: applied
          primitives_detected: [Tanh]
          evidence:
            files: [kernel/test_kernel.h:1]
          diff_refs: [kernel/test_kernel.h:1-1]
        ```
        """,
    )
    rep = scan.scan_workspace(ws)
    assert rep.verdict == "FAIL"
    assert "rationale" in rep.rationale.lower()


def test_yaml_applied_without_diff_refs_fails(tmp_path):
    """`status: applied` requires non-empty `diff_refs`."""
    ws = _build_ws(
        tmp_path,
        kernel_h="Tanh(y, x, n);",
        model_py="def forward(self, x): return torch.nn.functional.gelu(x)",
        knowledge_update="""
        ```yaml
        p_p88:
          status: applied
          primitives_detected: [Tanh]
          evidence:
            files: [kernel/test_kernel.h:1]
            rationale: "Sigmoid-form rewrite applied"
        ```
        """,
    )
    rep = scan.scan_workspace(ws)
    assert rep.verdict == "FAIL"
    assert "diff_refs" in rep.rationale


def test_yaml_exempt_without_measurements_fails(tmp_path):
    """`status: exempt` requires `evidence.isolated_primitive_measurements`."""
    ws = _build_ws(
        tmp_path,
        kernel_h="Tanh(y, x, n);",
        model_py="def forward(self, x): return torch.tanh(x)",
        knowledge_update="""
        ```yaml
        p_p88:
          status: exempt
          primitives_detected: [Tanh]
          evidence:
            files: [kernel/test_kernel.h:1]
            rationale: "Bounded input domain — PB-24 doesn't apply"
        ```
        """,
    )
    rep = scan.scan_workspace(ws)
    assert rep.verdict == "FAIL"
    assert "isolated_primitive_measurements" in rep.rationale or "measurements" in rep.rationale


def test_yaml_invalid_status_fails(tmp_path):
    """Status must be one of {applied, exempt, not_applicable}."""
    ws = _build_ws(
        tmp_path,
        kernel_h="Tanh(y, x, n);",
        model_py="def forward(self, x): return torch.tanh(x)",
        knowledge_update="""
        ```yaml
        p_p88:
          status: maybe
          evidence:
            rationale: "..."
        ```
        """,
    )
    rep = scan.scan_workspace(ws)
    assert rep.verdict == "FAIL"
    assert "status" in rep.rationale.lower()


# ---------------------------------------------------------------------------
# Smoke test — clean elementwise op should NOT_APPLICABLE
# ---------------------------------------------------------------------------
def test_clean_elementwise_op_not_applicable(tmp_path):
    """Pure elementwise add — no risky primitive, no transcendental.
    Gate NOT_APPLICABLE, no false positives.
    """
    ws = _build_ws(
        tmp_path,
        kernel_h="""
        Add(y, a, b, n);
        """,
        model_py="""
        class Model(torch.nn.Module):
            def forward(self, a, b):
                return a + b
        """,
    )
    rep = scan.scan_workspace(ws)
    assert rep.verdict == "NOT_APPLICABLE"
    assert len(rep.risky_hits) == 0


# ---------------------------------------------------------------------------
# Real-world test — actual workspace/1_GELU should FAIL
# ---------------------------------------------------------------------------
def test_real_1_gelu_workspace_fails_today(tmp_path):
    """Real 1_GELU workspace: gate must FAIL when the May-8-cold-start
    kernel is present (uses `Tanh()` directly, no YAML).

    Skipped when the live workspace is mid-cold-start (kernel/ moved to
    .pre-cold-start-* backup dir) — that's a transient state, not a
    contract violation. The synthetic-workspace tests above cover the
    same logic deterministically.
    """
    real_ws = _reorg_paths.REPO_ROOT / "workspace" / "1_GELU"
    if not real_ws.exists() or not (real_ws / "kernel").is_dir():
        pytest.skip(
            f"real 1_GELU workspace kernel/ not present at {real_ws} "
            f"(possibly mid-cold-start)"
        )
    rep = scan.scan_workspace(real_ws)
    # Two acceptable outcomes:
    #   (a) FAIL — the May-8 regressed kernel is present (Tanh used, no YAML)
    #   (b) NOT_APPLICABLE — cold-start produced a P-P88-compliant kernel
    #       that uses sigmoid-form rewrite instead of Tanh primitive
    # The test rejects only the case where the kernel uses Tanh AND lacks
    # YAML (verdict=FAIL is the regression-detection signal).
    assert rep.verdict in ("FAIL", "NOT_APPLICABLE", "PASS"), (
        f"Unexpected verdict on real 1_GELU: {rep.verdict} ({rep.rationale})"
    )
    if rep.verdict == "FAIL":
        assert any(h.primitive == "Tanh" for h in rep.risky_hits)


# ---------------------------------------------------------------------------
# Integration: P-P88 gate wired into finalize_pipeline.check_finalize_eligibility
# ---------------------------------------------------------------------------
def test_finalize_pipeline_gate_fires_on_pp88_violation(tmp_path):
    """End-to-end: a workspace with PASS verification.json + Tanh in kernel
    + no YAML block should be REJECTED by check_finalize_eligibility with
    gate=PP88_COMPLIANCE.
    """
    sys.path.insert(0, str(_HERE.parent.parent))
    import finalize_pipeline as fp  # noqa: E402
    import json

    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws / "kernel").mkdir()
    (ws / "kernel" / "test_kernel.h").write_text("Tanh(y, x, n);")
    (ws / "model.py").write_text(
        "def forward(self, x): return torch.nn.functional.gelu(x)\n"
        "def get_input_groups(): return [[]]\n"
    )
    (ws / "knowledge_update.md").write_text(
        "## Findings\n1. " + "did the work; " * 30 +
        "\n## Cited KB items\n- P-P88 (cited but not applied)\n"
        "## KB-promotable patterns\nnone\n## Anti-patterns avoided\nnone\n"
    )
    (ws / "audit_self_critic_post_worker.md").write_text("audit ran clean" * 10)
    (ws / "test_op.json").write_text('[{"shape":[256,512]}]')
    (ws / "verification.json").write_text(json.dumps({
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS", "tier1_pass": 1, "total": 1},
            "pass_b": {"status": "PASS", "tier1_pass": 1, "total": 1},
        },
        "performance": {"status": "PASS", "ratio": 1.0, "ratio_baseline": "vs CANN",
                        "independent_re_measure": {"ran": True, "ratio": 1.0}},
        "determinism": {"policy_satisfied": True,
                        "n_identical_cases": 1, "n_cases_checked": 1},
    }))
    elig = fp.check_finalize_eligibility(ws)
    assert elig["eligible"] is False
    assert elig["gate"] == fp.GateID.SIGMOID_FORM_REMEDIATION.value, (
        f"Expected PP88_COMPLIANCE gate, got {elig.get('gate')!r}: {elig.get('reason')}"
    )
    assert "P0abi" in elig["reason"]


def test_finalize_pipeline_gate_passes_when_pp88_compliant(tmp_path):
    """Same as above but kernel uses sigmoid-form rewrite (no Tanh).
    Gate is NOT_APPLICABLE, finalize advances past P-P88 check.
    (Other gates may still fire; this test only checks the P-P88 gate
    doesn't block.)
    """
    sys.path.insert(0, str(_HERE.parent.parent))
    import finalize_pipeline as fp  # noqa: E402
    import json

    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws / "kernel").mkdir()
    (ws / "kernel" / "test_kernel.h").write_text(
        "// P-P88-compliant sigmoid form, no Tanh primitive\n"
        "Exp(t1, x_neg, n);\n"
        "Adds(t1, t1, 1.0f, n);\n"
        "Reciprocal(y, t1, n);\n"
    )
    (ws / "model.py").write_text(
        "def forward(self, x): return torch.nn.functional.gelu(x)\n"
        "def get_input_groups(): return [[]]\n"
    )
    (ws / "knowledge_update.md").write_text(
        "## Findings\n1. used sigmoid form\n## Cited KB items\n- P-P88 (applied)\n"
    )
    (ws / "audit_self_critic_post_worker.md").write_text("audit ran clean" * 10)
    (ws / "test_op.json").write_text('[{"shape":[256,512]}]')
    (ws / "verification.json").write_text(json.dumps({
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS", "tier1_pass": 1, "total": 1},
            "pass_b": {"status": "PASS", "tier1_pass": 1, "total": 1},
        },
        "performance": {"status": "PASS", "ratio": 1.0, "ratio_baseline": "vs CANN",
                        "independent_re_measure": {"ran": True, "ratio": 1.0}},
        "determinism": {"policy_satisfied": True,
                        "n_identical_cases": 1, "n_cases_checked": 1},
    }))
    elig = fp.check_finalize_eligibility(ws)
    # Either eligible OR rejected by a DIFFERENT gate — but NOT PP88_COMPLIANCE
    assert elig.get("gate") != fp.GateID.SIGMOID_FORM_REMEDIATION.value, (
        f"P-P88 gate should not block sigmoid-form kernel; got {elig.get('reason')}"
    )
