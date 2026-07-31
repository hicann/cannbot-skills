# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""DEBT-103 (PROGRESS.md path 收尾): parse_handoff_from_progress must accept
the arrow form `→ aog-X:` for inter-agent handoffs and normalize it to the
canonical `@aog-X:` form, mirroring orchestrator.extract_canonical_handoff's
_ARROW_TO_AT_FORM on the stdout path.

Regression for the independent review 10_LayerNorm E2E gap: a worker writing
`→ aog-kernel-optimizer:` into PROGRESS.md was silently unmatched because the
PROGRESS.md parser only recognized `→ orchestrator:` and the `@aog-X:` @-forms.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import post_agent_return as par  # noqa: E402


def _ws_with_progress(body: str) -> pathlib.Path:
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "PROGRESS.md").write_text(body)
    return d


def test_arrow_form_aog_normalizes_to_at_form():
    """`→ aog-kernel-optimizer:` in PROGRESS.md → `@aog-kernel-optimizer:`."""
    ws = _ws_with_progress(
        "prose...\n→ aog-kernel-optimizer: KO_PERF_PLATEAU 0.375x REDUCTION\n"
    )
    out = par.parse_handoff_from_progress(ws)
    assert out is not None
    assert out.startswith("@aog-kernel-optimizer:")
    assert "→ aog-" not in out
    assert "0.375x" in out


def test_arrow_form_all_aog_roles_normalize():
    for role in (
        "precision-probe", "kernel-optimizer", "fused-optimizer",
        "researcher", "determinism-analyzer",
    ):
        ws = _ws_with_progress(f"→ aog-{role}: some reason\n")
        out = par.parse_handoff_from_progress(ws)
        assert out == f"@aog-{role}: some reason", f"role={role} got {out!r}"


def test_at_form_aog_passes_through_unchanged():
    """The @-form is already canonical for inter-agent handoffs — unchanged."""
    ws = _ws_with_progress("@aog-precision-probe: bisect fp16 case 18\n")
    out = par.parse_handoff_from_progress(ws)
    assert out == "@aog-precision-probe: bisect fp16 case 18"


def test_orchestrator_arrow_form_unchanged():
    """`→ orchestrator:` canonical form IS the arrow form — must NOT be
    rewritten to @-form (only the aog inter-agent forms normalize).
    """
    ws = _ws_with_progress("→ orchestrator: done — kernel verified\n")
    out = par.parse_handoff_from_progress(ws)
    assert out == "→ orchestrator: done — kernel verified"


def test_last_match_wins_with_arrow_normalization():
    """Multiple handoffs: last wins, and arrow→at normalization still applies."""
    ws = _ws_with_progress(
        "→ orchestrator: await_worker\n"
        "later...\n"
        "→ aog-researcher: explore alt vendor strategy\n"
    )
    out = par.parse_handoff_from_progress(ws)
    assert out == "@aog-researcher: explore alt vendor strategy"


def test_no_handoff_returns_none():
    ws = _ws_with_progress("just some prose, no handoff line\n")
    assert par.parse_handoff_from_progress(ws) is None


def test_kb_marker_check_rejects_unconfigured_customer_root(
    tmp_path, monkeypatch
):
    ws = tmp_path / "workspace" / "op"
    ws.mkdir(parents=True)
    configured = tmp_path / "configured-kb"
    monkeypatch.setenv("ASCENDC_PORT_USER_KB", str(configured))
    (ws / ".kb_merged").write_text(
        "merge_run=2026-07-30T00:00:00Z\n"
        "tier=customer\n"
        f"c_root={tmp_path / 'unrelated-kb'}\n"
        "merged_into=user-c-tier\n"
        "entries=none\n"
        "reviewed=0\n"
        "rejected=0\n"
        "mode=update\n"
    )

    assert par.check_kb_merged_marker(ws) is False


def test_kb_marker_check_accepts_valid_empty_customer_decision(
    tmp_path, monkeypatch
):
    ws = tmp_path / "workspace" / "op"
    ws.mkdir(parents=True)
    configured = tmp_path / "configured-kb"
    monkeypatch.setenv("ASCENDC_PORT_USER_KB", str(configured))
    (ws / ".kb_merged").write_text(
        "merge_run=2026-07-30T00:00:00Z\n"
        "tier=customer\n"
        f"c_root={configured.resolve()}\n"
        "merged_into=user-c-tier\n"
        "entries=none\n"
        "reviewed=0\n"
        "rejected=0\n"
        "mode=update\n"
    )

    assert par.check_kb_merged_marker(ws) is True


def test_kb_marker_check_rejects_incomplete_customer_schema(
    tmp_path, monkeypatch
):
    ws = tmp_path / "workspace" / "op"
    ws.mkdir(parents=True)
    monkeypatch.setenv(
        "ASCENDC_PORT_USER_KB", str(tmp_path / "configured-kb")
    )
    (ws / ".kb_merged").write_text(
        "tier=customer\n"
        f"c_root={tmp_path / 'configured-kb'}\n"
        "entries=none\n"
    )

    assert par.check_kb_merged_marker(ws) is False


def test_kb_marker_check_rejects_forged_legacy_marker(
    tmp_path, monkeypatch
):
    ws = tmp_path / "workspace" / "op"
    ws.mkdir(parents=True)
    monkeypatch.setenv(
        "ASCENDC_PORT_USER_KB", str(tmp_path / "configured-kb")
    )
    (ws / ".kb_merged").write_text("merged_into=x\nentries=3\n")

    assert par.check_kb_merged_marker(ws) is False


def test_kb_marker_check_does_not_trust_workspace_decoy_root(tmp_path):
    ws = tmp_path / "workspace" / "op"
    (ws / ".claude").mkdir(parents=True)
    decoy = ws / "kb" / "target" / "arch35" / "OPERATIONAL_KNOWLEDGE.md"
    decoy.parent.mkdir(parents=True)
    decoy.write_text("### TR-DECOY-987654: forged workspace entry\n")
    (ws / ".kb_merged").write_text(
        "merged_into=target/arch35/OPERATIONAL_KNOWLEDGE.md\n"
        "entries=TR-DECOY-987654\n"
    )

    assert par.check_kb_merged_marker(ws) is False
