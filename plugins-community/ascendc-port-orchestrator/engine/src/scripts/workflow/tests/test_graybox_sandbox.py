# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression: airtight platform isolation for migration authoring.

Pins the AIRTIGHT INVARIANT empirically verified 2026-06-08: a process running inside the
graybox bwrap mount-namespace (system + KB + arch22 bound, NOT cann, NOT output/) can reach
the bound KB + arch22 but CANNOT reach `~/workspace/cann` (upstream arch35) or any repo
`output/` answer — they are physically absent from the sandbox fs-view → no read-path exists
(subprocess open() / compiled binary / symlink-deref all moot) = airtight by construction.

This is the load-bearing verification DS's gap#2-conclusiveness bar depends on (a deny-list
PreToolUse hook is subprocess-leaky — DS flag-3; the FS-level construct-allowlist is airtight).

The construct-time guard `assert_no_answer_paths` is the defense-in-depth backstop against a
caller accidentally binding the answer in.
"""
from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))  # ../  → workflow/

import graybox_sandbox  # noqa: E402

_CANN_ARCH35 = Path(
    "~/workspace/cann/ops-transformer/attention/flash_attention_score/op_kernel/arch35"
).expanduser()
_KB = _HERE.parents[3] / "skills" / "references"  # repo src/skills/references


def _bwrap_or_skip():
    if not graybox_sandbox.bwrap_available():
        pytest.skip("bwrap (bubblewrap) not installed — a-fs airtight isolation unavailable")


# Use SYSTEM python (under bound /usr) for the in-sandbox probe — sys.executable may be a
# conda/venv interpreter under $HOME, which is intentionally NOT bound (the real graybox
# integration must explicitly bind its toolchain dir, e.g. ~/miniconda3, separately; binding
# a specific tool dir is fine — it is not the answer).
_SYS_PY = "/usr/bin/python3"


def _sys_py_or_skip():
    if not Path(_SYS_PY).exists():
        pytest.skip(f"{_SYS_PY} absent")


# ---------------------------------------------------------------------------
# AIRTIGHT INVARIANT — the empirical seal (the probe, formalized)
# ---------------------------------------------------------------------------
_PROBE = r"""
import os
print("kb reachable:", os.path.isdir("/kb"))
print("ws reachable:", os.path.isdir("/ws"))
print("cann tree reachable:", os.path.exists(os.path.expanduser("~/workspace/cann")))
try:
    os.listdir("/kb")  # KB must be usable inside the sandbox
    print("kb listable: True")
except Exception as e:
    print("kb listable: False", type(e).__name__)
try:
    os.listdir(os.path.expanduser(
        "~/workspace/cann/ops-transformer/attention/flash_attention_score/op_kernel/arch35"))
    print("LEAK: listed cann arch35")
except Exception as e:
    print("AIRTIGHT cann:", type(e).__name__)
"""


def test_sandbox_denies_cann_allows_kb(tmp_path):
    """Bound KB reachable; ~/workspace/cann (upstream arch35) physically absent → unreachable."""
    _bwrap_or_skip()
    _sys_py_or_skip()
    if not _KB.is_dir():
        pytest.skip(f"KB dir absent: {_KB}")
    ws = tmp_path / "ws"
    ws.mkdir()
    cmd = graybox_sandbox.build_bwrap_cmd(
        [_SYS_PY, "-c", _PROBE],
        allow_ro=[(_KB, "/kb")],
        allow_rw=[(ws, "/ws")],
    )
    out = subprocess.run(cmd, capture_output=True, text=True)
    assert "kb reachable: True" in out.stdout, out.stdout + out.stderr
    assert "ws reachable: True" in out.stdout, out.stdout + out.stderr
    assert "kb listable: True" in out.stdout, out.stdout + out.stderr
    # The seal: cann tree absent + arch35 listdir fails inside the sandbox
    assert "cann tree reachable: False" in out.stdout, out.stdout + out.stderr
    assert "AIRTIGHT cann:" in out.stdout, out.stdout + out.stderr
    assert "LEAK:" not in out.stdout, out.stdout


def test_sandbox_denies_repo_output(tmp_path):
    """A repo output/ answer dir, when NOT bound, is unreachable inside the sandbox."""
    _bwrap_or_skip()
    _sys_py_or_skip()
    repo_output = _HERE.parents[4] / "output"  # repo-root/output
    if not repo_output.is_dir():
        pytest.skip("repo output/ absent")
    ws = tmp_path / "ws"
    ws.mkdir()
    probe = (
        "import os;"
        f"print('output reachable:', os.path.exists({str(repo_output)!r}))"
    )
    cmd = graybox_sandbox.build_bwrap_cmd(
        [_SYS_PY, "-c", probe], allow_rw=[(ws, "/ws")]
    )
    out = subprocess.run(cmd, capture_output=True, text=True)
    assert "output reachable: False" in out.stdout, out.stdout + out.stderr


def test_bwrap_runtime_provides_null_and_writable_tmp():
    """The Linux runtime contract must support launcher redirection and scratch files.

    ``--dev /dev`` supplies the standard null device while ``--tmpfs /tmp`` gives
    Claude a private writable scratch area.  Keep this as an executable probe,
    because an argv-only assertion would not catch a broken mount namespace.
    """
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux bwrap runtime probe")
    _bwrap_or_skip()
    probe = [
        "/bin/sh",
        "-eu",
        "-c",
        (
            "test -r /dev/null; test -w /dev/null; "
            "printf ready >/tmp/graybox-runtime-probe; "
            "test \"$(cat /tmp/graybox-runtime-probe)\" = ready; "
            "echo BWRAP_RUNTIME_READY"
        ),
    ]
    out = subprocess.run(
        graybox_sandbox.build_bwrap_cmd(probe, share_net=True),
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "BWRAP_RUNTIME_READY" in out.stdout, out.stdout + out.stderr


# ---------------------------------------------------------------------------
# Construct-time backstop — assert_no_answer_paths
# ---------------------------------------------------------------------------
def test_assert_no_answer_paths_rejects_cann():
    with pytest.raises(ValueError, match="cann"):
        graybox_sandbox.assert_no_answer_paths([(str(_CANN_ARCH35), "/x")])


def test_assert_no_answer_paths_rejects_output(tmp_path):
    bad = "output/a3_to_a5_port/src/kernels/flash_attention_score/op_kernel/wholeport"
    with pytest.raises(ValueError, match="output/"):
        graybox_sandbox.assert_no_answer_paths([(bad, "/x")])


def test_assert_no_answer_paths_allows_kb_and_arch22(tmp_path):
    """KB + a copied-in arch22 dir (not under cann, not output/) pass the guard."""
    arch22 = tmp_path / "arch22_copy"
    arch22.mkdir()
    graybox_sandbox.assert_no_answer_paths([(str(_KB), "/kb"), (str(arch22), "/arch22")])


def test_build_bwrap_cmd_never_binds_home(tmp_path):
    """Structural check: the constructed argv binds only system + explicit allows — never
    $HOME or any parent of cann — so the answer trees can't sneak in via a parent mount.
    """
    if not graybox_sandbox.bwrap_available():
        pytest.skip("bwrap absent")
    ws = tmp_path / "ws"
    ws.mkdir()
    argv = graybox_sandbox.build_bwrap_cmd(
        ["true"], allow_ro=[(_KB, "/kb")], allow_rw=[(ws, "/ws")]
    )
    home = os.path.expanduser("~")
    # No bind target equals $HOME or ~/workspace (which would expose cann).
    bind_pairs = [a for a in argv]
    assert home not in bind_pairs
    assert f"{home}/workspace" not in bind_pairs
    assert str(_CANN_ARCH35.parent.parent) not in bind_pairs


# ---------------------------------------------------------------------------
# Bind-set builder + construction manifest (the dispatch-wiring helpers)
# ---------------------------------------------------------------------------
def test_default_toolchain_dirs_excludes_answer(tmp_path):
    """Toolchain dirs are real infra under $HOME but NEVER under cann/output."""
    tc = graybox_sandbox.default_toolchain_dirs()
    assert tc, "expected at least one toolchain dir on this host"
    # none is the cann tree or a repo output dir
    graybox_sandbox.assert_no_answer_paths([(d, d) for d in tc])


def test_graybox_allow_set_binds_kb_and_workspace(tmp_path):
    if not _KB.is_dir():
        pytest.skip(f"KB dir absent: {_KB}")
    ws = tmp_path / "ws"
    ws.mkdir()
    ro, rw = graybox_sandbox.graybox_allow_set(ws, kb_dir=_KB)
    ro_srcs = [s for s, _ in ro]
    # The compatibility path may be a symlink after the plugin relocation;
    # graybox_allow_set intentionally binds its canonical target.
    assert str(_KB.resolve()) in ro_srcs             # KB bound ro
    assert rw == [(str(ws.resolve()), str(ws.resolve()))]  # workspace bound rw
    # identity binds (in-sandbox path == host path) so #include roots resolve unchanged
    assert all(s == d for s, d in ro + rw)


def test_graybox_allow_set_rejects_cann_as_arch22(tmp_path):
    """If a caller mistakenly passes a cann path as the arch22 input → refuse (backstop)."""
    if not _KB.is_dir():
        pytest.skip("KB absent")
    ws = tmp_path / "ws"
    ws.mkdir()
    with pytest.raises(ValueError, match="cann"):
        graybox_sandbox.graybox_allow_set(ws, kb_dir=_KB, arch22_dir=str(_CANN_ARCH35.parent))


def test_graybox_allow_set_mounts_only_plugin_runtime_subtrees(tmp_path):
    """Claude workers get --plugin-dir content without the plugin workspace/output tree."""
    ws = tmp_path / "ws"
    ws.mkdir()
    kb = tmp_path / "kb"
    kb.mkdir()
    plugin = tmp_path / "plugin"
    for relative in (
        ".claude-plugin", "agents", "skills", "kb", "hooks", "scripts",
        "workflows", "engine/src/scripts",
    ):
        (plugin / relative).mkdir(parents=True)
    (plugin / "AGENTS.md").write_text("# test plugin\n")
    (plugin / "engine/workspace").mkdir(parents=True)
    (plugin / "engine/output").mkdir(parents=True)

    ro, rw = graybox_sandbox.graybox_allow_set(
        ws,
        kb_dir=kb,
        toolchain_dirs=[],
        plugin_dir=plugin,
    )
    mount = graybox_sandbox.DEFAULT_PLUGIN_MOUNT
    plugin_binds = [
        (src, dst) for src, dst in ro
        if dst.startswith(f"{mount}/")
    ]
    assert plugin_binds
    assert {dst for _src, dst in plugin_binds} == {
        f"{mount}/.claude-plugin",
        f"{mount}/AGENTS.md",
        f"{mount}/agents",
        f"{mount}/skills",
        f"{mount}/kb",
        f"{mount}/hooks",
        f"{mount}/scripts",
        f"{mount}/workflows",
        f"{mount}/engine/src/scripts",
        f"{mount}/.graybox_dependency_manifest.json",
    }
    assert all(src != str(plugin.resolve()) for src, _dst in ro)
    # The worker can write its entire workspace, so the per-spawn plugin
    # snapshot must live outside it and cannot be reused after tampering.
    assert all(
        not Path(src).resolve().is_relative_to(ws.resolve())
        for src, _dst in plugin_binds
    )
    assert all("engine/workspace" not in dst and "engine/output" not in dst for _src, dst in ro)
    graybox_sandbox.assert_no_answer_paths(ro + rw)


def test_graybox_allow_set_cleans_snapshot_when_answer_guard_fails(
    tmp_path, monkeypatch
):
    ws = tmp_path / "workspace"
    ws.mkdir()
    plugin = tmp_path / "plugin"
    for relative in (
        ".claude-plugin", "agents", "skills", "kb", "hooks", "engine/src/scripts",
    ):
        (plugin / relative).mkdir(parents=True)
    (plugin / ".claude-plugin" / "plugin.json").write_text('{"name": "port"}\n')

    staged_roots = []
    original_stage = graybox_sandbox.stage_plugin_runtime

    def _stage(*args, **kwargs):
        staged = original_stage(*args, **kwargs)
        staged_roots.append(staged)
        return staged

    def _reject(_allow):
        raise ValueError("answer path")

    monkeypatch.setattr(graybox_sandbox, "stage_plugin_runtime", _stage)
    monkeypatch.setattr(graybox_sandbox, "assert_no_answer_paths", _reject)

    with pytest.raises(ValueError, match="answer path"):
        graybox_sandbox.graybox_allow_set(
            ws, kb_dir=tmp_path / "kb", toolchain_dirs=[], plugin_dir=plugin
        )

    assert staged_roots
    assert all(not root.exists() for root in staged_roots)


def test_graybox_plugin_runtime_is_fresh_and_outside_workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    plugin = tmp_path / "plugin"
    for relative in (
        ".claude-plugin", "agents", "skills", "kb", "hooks", "engine/src/scripts",
    ):
        (plugin / relative).mkdir(parents=True)
    (plugin / ".claude-plugin" / "plugin.json").write_text('{"name": "test"}\n')
    first = graybox_sandbox.stage_plugin_runtime(plugin, ws)
    (first / "agents" / "tampered.md").write_text("bad\n")
    second = graybox_sandbox.stage_plugin_runtime(plugin, ws)
    assert first != second
    assert not first.is_relative_to(ws.resolve())
    assert not second.is_relative_to(ws.resolve())
    assert not (second / "agents" / "tampered.md").exists()
    # Tests own these externally staged temporary snapshots.
    import shutil
    shutil.rmtree(first)
    shutil.rmtree(second)


def _make_marketplace_plugin(tmp_path: Path, primary_skill: str):
    """Build the marketplace plugin (+ declared dependency) tree shared by staging tests."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    marketplace = tmp_path / "cache" / "cannbot"
    plugin = marketplace / "port" / "1.0.0"
    dependency = marketplace / "shared" / "2.0.0"
    for relative in (
        ".claude-plugin", "agents", "skills", "kb", "hooks", "engine/src/scripts",
    ):
        (plugin / relative).mkdir(parents=True)
    (plugin / ".claude-plugin" / "plugin.json").write_text(
        '{"name": "port", "dependencies": ["shared"]}\n'
    )
    (plugin / "skills" / primary_skill / "SKILL.md").parent.mkdir(parents=True)
    (plugin / "skills" / primary_skill / "SKILL.md").write_text("# primary\n")
    return ws, plugin, dependency


def test_graybox_plugin_runtime_materializes_marketplace_dependency_skills(tmp_path):
    """Graybox keeps marketplace dependency skills visible to Claude workers."""
    ws, plugin, dependency = _make_marketplace_plugin(tmp_path, "primary")
    (dependency / "ops-precision-standard").mkdir(parents=True)
    (dependency / "ops-precision-standard" / "SKILL.md").write_text("# precision\n")
    (dependency / "unrelated-runtime").mkdir()
    (dependency / "unrelated-runtime" / "entrypoint.py").write_text("# not a skill\n")

    staged = graybox_sandbox.stage_plugin_runtime(plugin, ws)
    assert (staged / "skills" / "primary" / "SKILL.md").is_file()
    assert (staged / "skills" / "ops-precision-standard" / "SKILL.md").read_text() == "# precision\n"
    assert not (staged / "skills" / "unrelated-runtime").exists()
    dependency_manifest = json.loads(
        (staged / ".graybox_dependency_manifest.json").read_text()
    )
    assert dependency_manifest["unresolved"] == []
    assert dependency_manifest["records"][0]["status"] == "resolved"
    assert dependency_manifest["staged_tree_sha256"]

    import shutil
    shutil.rmtree(staged)


def test_graybox_plugin_runtime_materializes_direct_checkout_dependency_skills(tmp_path):
    """A repository checkout must expose declared shared skills in graybox."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    checkout = tmp_path / "checkout"
    plugin = checkout / "plugins-community" / "ascendc-port-orchestrator"
    ops = checkout / "ops" / "ops-precision-standard"
    knowledge = checkout / "plugins-community" / "cannbot-knowledge" / "skills" / "knowledge-query"
    for relative in (
        ".claude-plugin", "agents", "skills", "kb", "hooks", "engine/src/scripts",
    ):
        (plugin / relative).mkdir(parents=True)
    (plugin / ".claude-plugin" / "plugin.json").write_text(
        '{"name": "port", "dependencies": ['
        '"ascendc-port-orchestrator-shared-skills", '
        '"cannbot-knowledge-consumer-skills"]}\n'
    )
    ops.mkdir(parents=True)
    (ops / "SKILL.md").write_text("# precision\n")
    knowledge.mkdir(parents=True)
    (knowledge / "SKILL.md").write_text("# knowledge\n")

    staged = graybox_sandbox.stage_plugin_runtime(plugin, ws)
    assert (staged / "skills" / "ops-precision-standard" / "SKILL.md").read_text() == "# precision\n"
    assert (staged / "skills" / "knowledge-query" / "SKILL.md").read_text() == "# knowledge\n"
    dependency_manifest = json.loads(
        (staged / ".graybox_dependency_manifest.json").read_text()
    )
    assert [record["status"] for record in dependency_manifest["records"]] == [
        "resolved",
        "resolved",
    ]

    import shutil
    shutil.rmtree(staged)


def test_graybox_plugin_runtime_rejects_unresolved_declared_dependency(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    plugin = tmp_path / "plugin"
    for relative in (
        ".claude-plugin", "agents", "skills", "kb", "hooks", "engine/src/scripts",
    ):
        (plugin / relative).mkdir(parents=True)
    (plugin / ".claude-plugin" / "plugin.json").write_text(
        '{"name": "port", "dependencies": ["missing-shared-skills"]}\n'
    )

    with pytest.raises(RuntimeError, match="unresolved"):
        graybox_sandbox.stage_plugin_runtime(plugin, ws)


def test_construction_manifest_binds_dependency_resolution_proof(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    kb = tmp_path / "kb"
    kb.mkdir()
    plugin = tmp_path / "plugin"
    for relative in (
        ".claude-plugin", "agents", "skills", "kb", "hooks", "engine/src/scripts",
    ):
        (plugin / relative).mkdir(parents=True)
    (plugin / ".claude-plugin" / "plugin.json").write_text(
        '{"name": "port", "dependencies": []}\n'
    )

    allow_ro, allow_rw = graybox_sandbox.graybox_allow_set(
        ws, kb_dir=kb, toolchain_dirs=[], plugin_dir=plugin
    )
    manifest_path = graybox_sandbox.write_construction_manifest(
        ws, allow_ro, allow_rw, inner_cmd=["claude", "--plugin-dir"]
    )
    manifest = json.loads(manifest_path.read_text())

    proof = manifest["plugin_runtime"]["dependency_manifests"]
    assert len(proof) == 1
    assert proof[0]["schema"] == "graybox_plugin_dependencies/v1"
    assert proof[0]["unresolved"] == []
    graybox_sandbox.cleanup_staged_plugin_runtimes(allow_ro)


def test_graybox_plugin_runtime_cleanup_only_removes_owned_snapshot(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    plugin = tmp_path / "plugin"
    for relative in (
        ".claude-plugin", "agents", "skills", "kb", "hooks", "engine/src/scripts",
    ):
        (plugin / relative).mkdir(parents=True)
    (plugin / ".claude-plugin" / "plugin.json").write_text('{"name": "port"}\n')

    staged = graybox_sandbox.stage_plugin_runtime(plugin, ws)
    allow_ro = [
        (str(staged / ".claude-plugin"), f"{graybox_sandbox.DEFAULT_PLUGIN_MOUNT}/.claude-plugin"),
        (str(staged / "skills"), f"{graybox_sandbox.DEFAULT_PLUGIN_MOUNT}/skills"),
        (str(plugin), "/unrelated-plugin"),
    ]
    assert graybox_sandbox.staged_plugin_runtime_roots(allow_ro) == (staged.resolve(),)
    graybox_sandbox.cleanup_staged_plugin_runtimes(allow_ro)
    assert not staged.exists()
    assert plugin.exists()


def test_graybox_plugin_runtime_preserves_primary_skill_on_dependency_collision(tmp_path):
    ws, plugin, dependency = _make_marketplace_plugin(tmp_path, "same")
    (dependency / "same").mkdir(parents=True)
    (dependency / "same" / "SKILL.md").write_text("# dependency\n")

    staged = graybox_sandbox.stage_plugin_runtime(plugin, ws)
    assert (staged / "skills" / "same" / "SKILL.md").read_text() == "# primary\n"

    import shutil
    shutil.rmtree(staged)


def test_plugin_dir_uses_real_staged_source_for_sandbox_exec(tmp_path):
    staged = tmp_path / "runtime"
    manifest = staged / ".claude-plugin"
    manifest.mkdir(parents=True)
    allow_ro = [(str(manifest), f"{graybox_sandbox.DEFAULT_PLUGIN_MOUNT}/.claude-plugin")]
    assert graybox_sandbox.plugin_dir_for_isolation_backend(
        allow_ro, backend="sandbox-exec"
    ) == str(staged.resolve())
    assert graybox_sandbox.plugin_dir_for_isolation_backend(
        allow_ro, backend="bwrap"
    ) == graybox_sandbox.DEFAULT_PLUGIN_MOUNT


def test_bwrap_creates_dedicated_plugin_mount_root(monkeypatch, tmp_path):
    monkeypatch.setattr(graybox_sandbox, "_BWRAP", "/usr/bin/bwrap")
    ws = tmp_path / "ws"
    ws.mkdir()
    plugin_root = graybox_sandbox.DEFAULT_PLUGIN_MOUNT
    argv = graybox_sandbox.build_bwrap_cmd(
        [],
        allow_ro=[(str(tmp_path), f"{plugin_root}/agents")],
        allow_rw=[(str(ws), str(ws))],
    )
    assert argv[argv.index("--dir") + 1] == "/usr"
    assert ["--dir", plugin_root] in [argv[i:i + 2] for i in range(len(argv) - 1)]


def test_curated_scan_allows_target_named_advisory_knowledge(tmp_path):
    root = tmp_path / "arch22"
    target = root / "arch35"
    target.mkdir(parents=True)
    (target / "README.md").write_text("# advisory target notes\n")
    (root / "allowed.h").write_text("// arch22\n")

    target_sources, answer_files = getattr(graybox_sandbox, '_scan_curated_tree')(root)
    assert target_sources == 0
    assert answer_files == 0


def test_curated_scan_rejects_source_inside_target_tree(tmp_path):
    root = tmp_path / "arch22"
    target = root / "arch35"
    target.mkdir(parents=True)
    (target / "answer.h").write_text("// target implementation\n")

    target_sources, answer_files = getattr(graybox_sandbox, '_scan_curated_tree')(root)
    assert target_sources == 1
    assert answer_files == 0


def test_curated_scan_allows_pybind_only_in_immutable_direct_source_stage(tmp_path):
    source = tmp_path / ".port_source"
    source.mkdir()
    (source / "3_Add.cpp").write_text("// source kernel\n")
    (source / "pybind11.cpp").write_text("// source binding\n")

    target_sources, answer_files = getattr(graybox_sandbox, "_scan_curated_tree")(
        source, allow_source_pybind=True
    )
    assert target_sources == 0
    assert answer_files == 0

    # The same file remains answer-bearing under the default candidate scan.
    _, rejected_answers = getattr(graybox_sandbox, "_scan_curated_tree")(source)
    assert rejected_answers == 1


def test_workspace_scan_only_exempts_exact_port_source_tree(tmp_path):
    ws = tmp_path / "workspace"
    source = ws / ".port_source"
    candidate = ws / "candidate" / "kernel"
    ordinary = ws / "scratch"
    source.mkdir(parents=True)
    candidate.mkdir(parents=True)
    ordinary.mkdir(parents=True)
    (source / "pybind11.cpp").write_text("// frozen source binding\n")
    (candidate / "pybind11.cpp").write_text("// candidate binding\n")
    (ordinary / "pybind11.cpp").write_text("// workspace binding\n")

    _, answer_files = getattr(graybox_sandbox, "_scan_curated_tree")(
        ws, source_stage_root=source
    )
    assert answer_files == 2


def test_workspace_scan_does_not_exempt_fake_port_source_tree(tmp_path):
    ws = tmp_path / "workspace"
    real_source = ws / ".port_source"
    fake_source = ws / "nested" / ".port_source"
    real_source.mkdir(parents=True)
    fake_source.mkdir(parents=True)
    (real_source / "pybind11.cpp").write_text("// frozen source binding\n")
    (fake_source / "pybind11.cpp").write_text("// untrusted binding\n")

    _, answer_files = getattr(graybox_sandbox, "_scan_curated_tree")(
        ws, source_stage_root=real_source
    )
    assert answer_files == 1


def test_assert_no_answer_paths_allows_pure_target_named_knowledge(tmp_path):
    knowledge = tmp_path / "arch35"
    knowledge.mkdir()
    (knowledge / "README.md").write_text("# advisory target notes\n")

    graybox_sandbox.assert_no_answer_paths([(str(knowledge), "/kb/arch35")])


def test_assert_no_answer_paths_rejects_source_in_target_tree(tmp_path):
    target = tmp_path / "arch35"
    target.mkdir()
    (target / "answer.h").write_text("// target implementation\n")

    with pytest.raises(ValueError, match="target implementation/answer"):
        graybox_sandbox.assert_no_answer_paths([(str(target), "/arch35")])


def test_assert_no_answer_paths_rejects_direct_target_source_file(tmp_path):
    target = tmp_path / "arch35"
    target.mkdir()
    answer = target / "answer.h"
    answer.write_text("// target implementation\n")

    with pytest.raises(ValueError, match="target implementation/answer"):
        graybox_sandbox.assert_no_answer_paths([(str(answer), "/kb/answer.h")])


def test_assert_no_answer_paths_allows_direct_target_advisory_file(tmp_path):
    target = tmp_path / "arch35"
    target.mkdir()
    notes = target / "notes.json"
    notes.write_text('{"kind": "advisory"}\n')

    graybox_sandbox.assert_no_answer_paths([(str(notes), "/kb/notes.json")])


def test_manifest_asserts_airtight_and_scopes_scan(tmp_path):
    if not _KB.is_dir():
        pytest.skip("KB absent")
    ws = tmp_path / "ws"
    ws.mkdir()
    ro, rw = graybox_sandbox.graybox_allow_set(ws, kb_dir=_KB)
    mpath = graybox_sandbox.write_construction_manifest(
        ws, ro, rw, inner_cmd=["claude", "--agent", "aog-kernel-worker"]
    )
    m = json.loads(Path(mpath).read_text())
    assert m["assertions"]["airtight"] is True
    assert m["assertions"]["arch35_dirs_reachable"] == 0
    assert m["assertions"]["assembled_answer_cpp_reachable"] == 0
    # toolchain binds recorded but NOT deep-scanned (huge infra); KB+ws ARE scanned
    by_src = {e["src"]: e for e in m["bound"]}
    assert by_src[str(_KB.resolve())]["deep_scanned"] is True
    assert by_src[str(ws.resolve())]["deep_scanned"] is True
    tc = graybox_sandbox.default_toolchain_dirs()
    if tc:
        assert by_src[tc[0]]["deep_scanned"] is False


def test_construction_manifest_seal_rejects_worker_mutation(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    kb = tmp_path / "kb"
    kb.mkdir()
    ro, rw = graybox_sandbox.graybox_allow_set(ws, kb_dir=kb)
    mpath = graybox_sandbox.write_construction_manifest(
        ws, ro, rw, inner_cmd=["claude", "--agent", "aog-kernel-worker"]
    )
    digest = graybox_sandbox.construction_manifest_sha256(mpath)
    graybox_sandbox.verify_construction_manifest(mpath, digest)
    mpath.write_text(mpath.read_text() + "\nworker mutation\n")

    with pytest.raises(RuntimeError, match="changed during worker dispatch"):
        graybox_sandbox.verify_construction_manifest(mpath, digest)


def test_dispatch_prefix_pattern_is_airtight(tmp_path):
    """The EXACT pattern agent_dispatch uses: build_bwrap_cmd([], …) yields a bwrap arg-list
    ending in '--'; agent_transport prepends it to the real claude argv. Simulate with a
    real command and confirm the composed argv is airtight (cann unreachable, KB reachable).
    """
    _bwrap_or_skip()
    _sys_py_or_skip()
    if not _KB.is_dir():
        pytest.skip("KB absent")
    ws = tmp_path / "ws"
    ws.mkdir()
    ro, rw = graybox_sandbox.graybox_allow_set(ws, kb_dir=_KB)
    prefix = graybox_sandbox.build_bwrap_cmd(
        [], allow_ro=ro, allow_rw=rw, workdir=str(ws), share_net=False
    )
    assert prefix[-1] == "--"  # ends at the command separator
    # graybox_allow_set uses IDENTITY binds (in-sandbox path == host path) so the agent's
    # KB #include roots + scripts resolve unchanged — probe the REAL host paths, not /kb.
    identity_probe = (
        "import os;"
        f"print('kb reachable:', os.path.isdir({str(_KB)!r}));"
        f"print('ws reachable:', os.path.isdir({str(ws.resolve())!r}));"
        "print('cann reachable:', os.path.exists(os.path.expanduser('~/workspace/cann')))"
    )
    composed = prefix + [_SYS_PY, "-c", identity_probe]  # == sandbox_prefix + claude-argv
    out = subprocess.run(composed, capture_output=True, text=True)
    assert "kb reachable: True" in out.stdout, out.stdout + out.stderr
    assert "ws reachable: True" in out.stdout, out.stdout + out.stderr
    assert "cann reachable: False" in out.stdout, out.stdout + out.stderr


def test_platform_backend_selection_is_fail_closed(monkeypatch):
    monkeypatch.setattr(graybox_sandbox, "_BWRAP", "/usr/bin/bwrap")
    monkeypatch.setattr(
        graybox_sandbox, "_SANDBOX_EXEC", "/usr/bin/sandbox-exec"
    )
    assert graybox_sandbox.isolation_backend("linux") == "bwrap"
    assert graybox_sandbox.isolation_backend("darwin") == "sandbox-exec"
    assert graybox_sandbox.isolation_backend("win32") is None

    monkeypatch.setattr(graybox_sandbox, "_BWRAP", None)
    monkeypatch.setattr(graybox_sandbox, "_SANDBOX_EXEC", None)
    assert graybox_sandbox.isolation_backend("linux") is None
    assert graybox_sandbox.isolation_backend("darwin") is None
    with pytest.raises(RuntimeError, match="no supported strict isolation"):
        graybox_sandbox.build_isolated_cmd(["true"])


def test_bwrap_is_narrow_no_network_and_source_overlay_last(tmp_path, monkeypatch):
    monkeypatch.setattr(graybox_sandbox, "_BWRAP", "/usr/bin/bwrap")
    ws = tmp_path / "ws"
    stage = ws / ".source_arch22"
    stage.mkdir(parents=True)
    argv = graybox_sandbox.build_bwrap_cmd(
        ["true"],
        allow_ro=[(str(stage), str(stage))],
        allow_rw=[(str(ws), str(ws))],
    )
    assert "--unshare-net" in argv
    pairs = list(zip(argv, argv[1:]))
    assert ("--ro-bind", "/usr") not in pairs
    assert ("--ro-bind", "/opt") not in pairs
    ws_bind = argv.index(str(ws), argv.index("--bind"))
    stage_bind = argv.index(str(stage), argv.index("--ro-bind"))
    assert ws_bind < stage_bind
    shared = graybox_sandbox.build_bwrap_cmd(["true"], share_net=True)
    assert "--unshare-net" not in shared


def test_bwrap_exposes_local_claude_runtime_but_not_cann(monkeypatch):
    """A /usr/local/bin/claude symlink must resolve in the graybox namespace."""
    monkeypatch.setattr(graybox_sandbox, "_BWRAP", "/usr/bin/bwrap")
    argv = graybox_sandbox.build_bwrap_cmd(["true"])
    ro_bind_sources = {
        argv[i + 1]
        for i, token in enumerate(argv[:-1])
        if token == "--ro-bind"
    }
    assert "/usr/local/lib" in ro_bind_sources or not Path("/usr/local/lib").is_dir()
    assert "/usr/local/Ascend" not in ro_bind_sources


def test_bwrap_binds_resolver_config_when_present(monkeypatch):
    """Network-enabled Kimi workers need DNS without broad /etc exposure."""
    monkeypatch.setattr(graybox_sandbox, "_BWRAP", "/usr/bin/bwrap")
    argv = graybox_sandbox.build_bwrap_cmd(["true"], share_net=True)
    pairs = [argv[i:i + 3] for i in range(len(argv) - 2)]
    # The implementation deliberately skips a top-level symlink: its target
    # might not be present in the isolated mount namespace.  Containers such
    # as cjm_cann2 provide a regular file, which is the supported DNS path.
    if Path("/etc/resolv.conf").is_file() and not Path("/etc/resolv.conf").is_symlink():
        assert ["--ro-bind", "/etc/resolv.conf", "/etc/resolv.conf"] in pairs
    else:
        assert ["--ro-bind", "/etc/resolv.conf", "/etc/resolv.conf"] not in pairs


def test_macos_profile_denies_network_history_and_stage_writes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        graybox_sandbox, "_SANDBOX_EXEC", "/usr/bin/sandbox-exec"
    )
    ws = tmp_path / "ws"
    stage = ws / ".source_arch22"
    stage.mkdir(parents=True)
    argv = graybox_sandbox.build_sandbox_exec_cmd(
        [],
        allow_ro=[(str(stage), str(stage))],
        allow_rw=[(str(ws), str(ws))],
        workdir=str(ws),
    )
    profile = argv[2]
    assert "(deny default)" in profile
    assert "(deny network*)" in profile
    assert os.path.expanduser("~/.claude") not in profile
    assert f'(allow file-write* (subpath "{ws}"))' in profile
    assert f'(deny file-write* (subpath "{stage}"))' in profile


def test_macos_profile_allows_explicit_model_network_sharing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        graybox_sandbox, "_SANDBOX_EXEC", "/usr/bin/sandbox-exec"
    )
    ws = tmp_path / "ws"
    ws.mkdir()
    argv = graybox_sandbox.build_sandbox_exec_cmd(
        ["true"], allow_rw=[(str(ws), str(ws))], share_net=True
    )
    profile = argv[argv.index("-p") + 1]
    assert "(allow network*)" in profile
    assert "(deny network*)" not in profile


@pytest.mark.parametrize(
    ("share_net", "network", "enforcement"),
    [
        (False, "denied", "(deny network*)"),
        (True, "shared", "(allow network*)"),
    ],
)
def test_construction_manifest_records_effective_macos_network_policy(
    tmp_path, monkeypatch, share_net, network, enforcement
):
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(graybox_sandbox, "_BWRAP", None)
    monkeypatch.setattr(graybox_sandbox, "_SANDBOX_EXEC", "/usr/bin/sandbox-exec")
    monkeypatch.setattr(
        graybox_sandbox, "isolation_backend", lambda platform=None: "sandbox-exec"
    )

    manifest = graybox_sandbox.write_construction_manifest(
        ws,
        [],
        [(str(ws), str(ws))],
        inner_cmd=["true"],
        share_net=share_net,
    )
    payload = json.loads(manifest.read_text())
    assert payload["network"] == network
    assert payload["network_enforcement"] == enforcement


def test_macos_profile_allows_runtime_tmp_alias_and_null_sink(tmp_path, monkeypatch):
    monkeypatch.setattr(
        graybox_sandbox, "_SANDBOX_EXEC", "/usr/bin/sandbox-exec"
    )
    ws = tmp_path / "ws"
    ws.mkdir()
    argv = graybox_sandbox.build_sandbox_exec_cmd(
        ["true"], allow_rw=[(str(ws), str(ws))]
    )
    profile = argv[argv.index("-p") + 1]
    assert '(allow file-read* (subpath "/private/tmp"))' in profile
    assert '(allow file-read* (subpath "/tmp"))' in profile
    assert '(allow file-write* (subpath "/private/tmp"))' in profile
    assert '(allow file-write* (subpath "/tmp"))' in profile
    assert '(allow file-read* (literal "/dev/null"))' in profile
    assert '(allow file-write* (literal "/dev/null"))' in profile


def test_macos_profile_explicitly_denies_staged_plugin_runtime_writes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        graybox_sandbox, "_SANDBOX_EXEC", "/usr/bin/sandbox-exec"
    )
    ws = tmp_path / "ws"
    ws.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    argv = graybox_sandbox.build_sandbox_exec_cmd(
        ["true"],
        allow_ro=[
            (
                str(runtime),
                f"{graybox_sandbox.DEFAULT_PLUGIN_MOUNT}/agents",
            )
        ],
        allow_rw=[(str(ws), str(ws))],
    )
    profile = argv[argv.index("-p") + 1]
    assert f'(allow file-read* (subpath "{runtime}"))' in profile
    assert f'(deny file-write* (subpath "{runtime}"))' in profile


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox-exec only")
def test_macos_sandbox_exec_enforces_stage_read_only(tmp_path):
    if not getattr(graybox_sandbox, '_SANDBOX_EXEC'):
        pytest.skip("sandbox-exec absent")
    ws = tmp_path / "ws"
    stage = ws / ".source_arch22"
    stage.mkdir(parents=True)
    protected = stage / "source.h"
    protected.write_text("source\n")
    cmd = graybox_sandbox.build_sandbox_exec_cmd(
        [
            "/bin/sh",
            "-c",
            "printf ok > own-output; ! printf leak >> .source_arch22/source.h",
        ],
        allow_ro=[(str(stage), str(stage))],
        allow_rw=[(str(ws), str(ws))],
        workdir=str(ws),
    )
    out = subprocess.run(cmd, cwd=ws, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert (ws / "own-output").read_text() == "ok"
    assert protected.read_text() == "source\n"


def test_installed_cann_env_root_is_rejected(tmp_path, monkeypatch):
    cann_root = tmp_path / "toolkit"
    cann_root.mkdir()
    monkeypatch.setenv("ASCEND_HOME_PATH", str(cann_root))
    with pytest.raises(ValueError, match="CANN"):
        graybox_sandbox.assert_no_answer_paths(
            [(str(cann_root / "include"), "/include")]
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
