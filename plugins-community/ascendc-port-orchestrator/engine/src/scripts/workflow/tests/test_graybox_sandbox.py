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
    import json
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
    with pytest.raises(ValueError, match="forbids network"):
        graybox_sandbox.build_bwrap_cmd(["true"], share_net=True)


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
