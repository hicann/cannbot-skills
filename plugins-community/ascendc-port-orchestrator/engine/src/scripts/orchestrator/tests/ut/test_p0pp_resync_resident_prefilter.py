# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""P0pp regression: O5 resync resident-file pre-filter (2026-06-11).

FA-A5 blackbox kw-3 escalation: `_resync_workspace_to_container` aborted O5
VERIFY->finalize for ALL FA-class / large-dataset port_a3 ops because the
P135.TS oversized-.pt guard fired on edge_dataset.pt (223 MiB) + edge_inputs.pt
(153 MiB) — even though those are GENUINE 68-case FA tensors, already
container-resident from the deploy step, and the untar is `tar --skip-old-files`
(would skip them anyway). Fix: pre-filter push_files to only those ABSENT in
current_task before the oversized guard, so already-resident large datasets are
never re-tarred (and never false-trip the guard); only the small worker-written
verifier scripts (pass_a_runner.py) get pushed.

These tests mock subprocess.run so no live ssh/scp happens.
"""
import sys
import types
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import phase_o5_runner  # noqa: E402


def _mk_ws(tmp_path):
    ws = tmp_path / "flash_attention_score"
    ws.mkdir()
    # large genuine dataset (sparse — no real disk use) > 100 MiB threshold
    for big in ("edge_dataset.pt", "edge_inputs.pt"):
        with open(ws / big, "wb") as f:
            f.truncate(105 * 1024 * 1024)
    # small worker-written verifier script (the thing resync actually needs to push)
    # P0pp-followup (2026-06-12): runner REFERENCES edge_dataset.pt by name → the
    # oversized fixtures are genuinely "needed on the container" (port_a3 truth-
    # source case), so the oversized guard's REFERENCED-abort path is exercised.
    # The unreferenced case (drop, not abort) has its own test.
    (ws / "pass_a_runner.py").write_text(
        "# verifier\nimport torch\nd = torch.load('edge_dataset.pt')\n"
        "e = torch.load('edge_inputs.pt')\n")
    (ws / "model.py").write_text("# model\n")
    return ws


_ENV = {
    "A5_HOST": "1.2.3.4", "A5_USER": "root", "A5_PASSWORD": "",
    "A5_CONTAINER": "ctr", "BENCHMARK_ROOT": "/data/AscendOpGenAgent",
}


def _run_with_resident(ws, resident_names):
    """Invoke _resync with subprocess.run mocked: ls returns resident_names,
    scp/untar succeed. Returns the function's result (None == success)."""
    def fake_run(cmd, *a, **k):
        joined = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        rc = 0
        out = ""
        if "ls -1" in joined:
            out = "\n".join(resident_names) + "\n"
        return types.SimpleNamespace(returncode=rc, stdout=out, stderr="")
    with mock.patch.object(phase_o5_runner.subprocess, "run", side_effect=fake_run):
        return getattr(phase_o5_runner, '_resync_workspace_to_container')(ws, dict(_ENV), lane=0)


def test_resident_large_pt_filtered_no_oversized_abort(tmp_path):
    """Large .pt already resident => filtered out => NO oversized-abort; success."""
    ws = _mk_ws(tmp_path)
    result = _run_with_resident(
        ws, ["edge_dataset.pt", "edge_inputs.pt", "model.py"])  # big ones resident
    # Must NOT be the oversized-abort string; None == clean resync.
    assert result is None, f"expected success, got: {result!r}"


def test_oversized_guard_still_fires_when_large_pt_absent(tmp_path):
    """If the large .pt are NOT resident (genuine first push), the guard still
    fires — the fix must not disable the guard wholesale, only skip resident.
    """
    ws = _mk_ws(tmp_path)
    result = _run_with_resident(ws, ["pass_a_runner.py"])  # big ones NOT resident
    assert result is not None and "oversized" in result.lower(), \
        f"expected oversized abort, got: {result!r}"


def test_probe_failure_falls_back_to_guard(tmp_path):
    """If the resident probe ssh fails (rc!=0), resident stays empty and the
    oversized guard still applies (safe fallback, no silent bypass).
    """
    ws = _mk_ws(tmp_path)

    def fake_run(cmd, *a, **k):
        joined = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        if "ls -1" in joined:
            return types.SimpleNamespace(returncode=255, stdout="", stderr="ssh fail")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    with mock.patch.object(phase_o5_runner.subprocess, "run", side_effect=fake_run):
        result = getattr(phase_o5_runner, '_resync_workspace_to_container')(ws, dict(_ENV), lane=0)
    assert result is not None and "oversized" in result.lower(), \
        f"expected oversized abort on probe-failure fallback, got: {result!r}"


def test_unreferenced_oversized_pt_dropped_not_aborted(tmp_path):
    """P0pp-followup (2026-06-12, FA cohort Phase 0a).

    Here the golden + input fixtures (edge_dataset.pt/edge_inputs.pt)
    are input-gen scratch the O5 verifiers NEVER read (pass_a_runner.py re-runs
    Model.forward live), AND they are generated locally / scp'd back so they are
    never container-resident → the resident-prefilter can't remove them. Before
    the fix the unconditional oversized guard false-ABORTED O5 finalize on a clean
    48/48 kernel (3_FusionAttention, edge_dataset.pt 338 MiB). The fix: an
    oversized .pt that NO pushed runner references is DROPPED (not aborted).
    """
    ws = tmp_path / "3_FusionAttention"
    ws.mkdir()
    for big in ("edge_dataset.pt", "edge_inputs.pt"):
        with open(ws / big, "wb") as f:
            f.truncate(105 * 1024 * 1024)
    # This runner does NOT reference the .pt (re-runs Model.forward live).
    (ws / "pass_a_runner.py").write_text("# verifier — re-runs model.forward, no .pt load\n")
    (ws / "model.py").write_text("# model\n")
    # big ones NOT resident (generated locally / scp'd back)
    result = _run_with_resident(ws, ["pass_a_runner.py"])
    assert result is None, \
        f"unreferenced oversized .pt should be DROPPED (success), got: {result!r}"


def test_stale_force_update_script_pre_deleted_before_additive_untar(tmp_path):
    """P0aba.O5 fix (2026-06-12, FA cohort Phase 0a gap-3): a stale resident
    verifier script (pass_b_runner.py left in the shared current_task by a prior
    op) must be REFRESHED. It is exempted from the resident pre-filter (stays in
    the tar), but the additive `tar --skip-old-files` untar would SKIP it → O5
    re-verify keeps running the stale script (ERR99999 loop). Fix: the remote
    command must `rm -f` the force-update scripts BEFORE the untar; build
    artifacts (NOT in FORCE_UPDATE_SCRIPTS) must NOT be rm'd.
    """
    ws = tmp_path / "3_FusionAttention"
    ws.mkdir()
    (ws / "pass_a_runner.py").write_text("# fresh pass_a\n")
    (ws / "pass_b_runner.py").write_text("# fresh pass_b\n")
    (ws / "model.py").write_text("# model wrapper (build artifact)\n")

    captured = []

    def fake_run(cmd, *a, **k):
        joined = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        captured.append(joined)
        out = ""
        if "ls -1" in joined:
            # pass_b_runner.py + model.py already resident (pass_b stale)
            out = "pass_b_runner.py\nmodel.py\n"
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    with mock.patch.object(phase_o5_runner.subprocess, "run", side_effect=fake_run):
        result = getattr(phase_o5_runner, '_resync_workspace_to_container')(ws, dict(_ENV), lane=0)
    assert result is None, f"expected success, got {result!r}"

    untar_cmds = [c for c in captured if "tar --skip-old-files" in c]
    assert untar_cmds, "no additive untar command issued"
    u = untar_cmds[0]
    assert "rm -f " in u, f"force-update scripts not pre-deleted: {u!r}"
    rm_seg = u[u.index("rm -f "):u.index("tar --skip-old-files")]
    # the stale verifier script IS refreshed (rm'd so the fresh copy lands)
    assert "pass_b_runner.py" in rm_seg, f"stale pass_b_runner.py not refreshed: {rm_seg!r}"
    # build artifact model.py must NOT be rm'd (preserves P0nn --skip-old-files guarantee)
    assert "model.py" not in rm_seg, f"build artifact model.py wrongly rm'd: {rm_seg!r}"


def test_resident_op_json_force_refreshed_not_dropped(tmp_path):
    """O5-RESYNC-JSON-FRESH (2026-07-20): the op benchmark case json (<op>.json,
    e.g. gated_delta_rule.json) is a RESYNC INPUT that model.py.get_input_groups()
    reads at O5 time. A STALE resident <op>.json in the shared current_task was
    (a) DROPPED by the resident pre-filter and (b) never overwritten by the
    additive `tar --skip-old-files` untar → get_input_groups() reads stale/missing
    shapes → spurious O5 failure. Fix: force-refresh the pushed op jsons exactly
    like FORCE_UPDATE_SCRIPTS (exempt from the pre-filter drop + rm-f before untar).

    Discriminator: a NON-forced resident build artifact (model.py) must STILL be
    dropped by the pre-filter — proving the fix did not blanket-disable it.
    """
    import tarfile

    ws = tmp_path / "gated_delta_rule"
    ws.mkdir()
    (ws / "gated_delta_rule.json").write_text('{"input_groups": []}')  # op case json
    (ws / "pass_a_runner.py").write_text("# verifier\n")
    (ws / "model.py").write_text("# model wrapper (build artifact)\n")

    tar_names: list = []
    captured: list = []

    class _FakeTar:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def add(self, name, arcname=None, **_kwargs):
            tar_names.append(arcname if arcname is not None else str(name))

    def fake_run(cmd, *a, **k):
        joined = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        captured.append(joined)
        out = ""
        if "ls -1" in joined:
            # op json + model.py + pass_a_runner.py all already resident;
            # the json is STALE, model.py is a resident build artifact.
            out = "gated_delta_rule.json\nmodel.py\npass_a_runner.py\n"
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    with mock.patch.object(phase_o5_runner.subprocess, "run", side_effect=fake_run), \
            mock.patch.object(tarfile, "open", return_value=_FakeTar()):
        result = getattr(phase_o5_runner, '_resync_workspace_to_container')(ws, dict(_ENV), lane=0)
    assert result is None, f"expected success, got {result!r}"

    # (a) op json survives the resident pre-filter => present in the pushed tar.
    assert "gated_delta_rule.json" in tar_names, \
        f"op json wrongly dropped from push: {tar_names!r}"
    # (b) op json is force-refreshed: rm-f'd before the additive untar.
    untar = next(c for c in captured if "tar --skip-old-files" in c)
    rm_seg = untar[untar.index("rm -f "):untar.index("tar --skip-old-files")]
    assert "gated_delta_rule.json" in rm_seg, \
        f"op json not force-refreshed (missing from rm -f): {rm_seg!r}"
    # DISCRIMINATOR: a NON-forced resident build artifact (model.py) is STILL
    # dropped by the pre-filter (not re-tarred) and NOT force-refreshed — proving
    # the fix didn't blanket-disable the pre-filter.
    assert "model.py" not in tar_names, \
        f"resident non-forced model.py should be pre-filtered out: {tar_names!r}"
    assert "model.py" not in rm_seg, \
        f"resident build artifact model.py wrongly force-refreshed: {rm_seg!r}"


def test_full_instrument_closure_force_refreshed_before_untar(tmp_path):
    """O5-INSTRUMENT-FRESH (2026-07-20, anti-cheat / TRUST): the WHOLE
    measurement-instrument closure — the canonical grader front-end scripts
    (precision_eval_two_tier / precision_eval_port_a3_two_tier / precision_tier1 /
    precision_tier2) PLUS their staged import closure (cannbench_grader package +
    reference_provider/verify.py) — computes pass/fail at O5 re-measure. Each is
    tar.add()'d, bypasses push_files, and is subject to the additive
    `tar --skip-old-files` untar → a STALE or TAMPERED resident copy of ANY of them
    would SURVIVE across O5 runs and regrade against a non-canonical instrument.
    Fix: precisely-scoped removal of the EXACT current_task/<arcname> path BEFORE
    the untar (FILE → rm -f, DIR → rm -rf) → resident copy removed → the fresh
    canonical copy lands.

    Asserts, for EVERY instrument member: (a) it is staged into the pushed tar, and
    (b) the untar command removes the EXACT current_task/<arcname> path (dir via
    rm -rf, file via rm -f) before `tar --skip-old-files`. Precise scoping: no
    wildcard, no bare current_task wipe.
    """
    import tarfile

    ws = tmp_path / "some_op"
    ws.mkdir()
    (ws / "pass_a_runner.py").write_text("# verifier\n")
    (ws / "model.py").write_text("# model\n")

    tar_names: list = []
    captured: list = []

    class _FakeTar:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def add(self, name, arcname=None, **_kwargs):
            tar_names.append(arcname if arcname is not None else str(name))

    def fake_run(cmd, *a, **k):
        joined = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        captured.append(joined)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    with mock.patch.object(phase_o5_runner.subprocess, "run", side_effect=fake_run), \
            mock.patch.object(tarfile, "open", return_value=_FakeTar()):
        result = getattr(phase_o5_runner, '_resync_workspace_to_container')(ws, dict(_ENV), lane=0)
    assert result is None, f"expected success, got {result!r}"

    untar = next(c for c in captured if "tar --skip-old-files" in c)
    pre_untar = untar[:untar.index("tar --skip-old-files")]
    _ct = f"{_ENV['BENCHMARK_ROOT']}/current_task"

    # DIR instrument: cannbench_grader package → rm -rf exact path.
    _grader_arc = "orchestrator/precision/cannbench_grader"
    assert _grader_arc in tar_names, f"grader dir not staged: {tar_names!r}"
    assert f"rm -rf {_ct}/{_grader_arc} " in pre_untar, \
        f"grader dir not force-refreshed with precise rm -rf: {pre_untar!r}"

    # FILE instruments: grader front-end scripts + reference_provider/verify.py →
    # rm -f exact path. verify.py is the same-class closure member the coordinator
    # flagged — red-before/green-after is anchored on it (and its siblings).
    _instrument_files = [
        "precision_eval_two_tier.py",
        "precision_eval_port_a3_two_tier.py",
        "precision_tier1.py",
        "precision_tier2.py",
        "reference_provider/verify.py",
    ]
    for _arc in _instrument_files:
        assert _arc in tar_names, f"instrument file {_arc!r} not staged: {tar_names!r}"
        assert f"rm -f {_ct}/{_arc} " in pre_untar, \
            f"instrument file {_arc!r} not force-refreshed with precise rm -f: {pre_untar!r}"

    # Precise scoping: NOT a blunt wildcard / bare current_task wipe (would nuke
    # build artifacts the P0nn --skip-old-files guard keeps container-authoritative).
    assert f"rm -rf {_ct} " not in pre_untar
    assert f"rm -f {_ct} " not in pre_untar
    assert "current_task/*" not in pre_untar


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
