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
"""graybox_sandbox — FS-level input isolation for the FA-A5 gap#2 graybox.

The graybox agent assembles an arch35 FA op from the declared arch22 source and
codified KB, with external undeclared target trees absent:
  - upstream CANN arch35 (`~/workspace/cann/.../op_kernel/arch35/`, `op_host/arch35/`), and
  - our assembled deliverable (`output/.../wholeport/*.cpp`, `.so`).

A deny-list PreToolUse hook (output_read_guard) is necessary but NOT airtight: a subprocess
`open()` / compiled-binary / symlink-deref bypasses it (DS flag-3, 2026-06-07). This module is
the AIRTIGHT PRIMARY: run the graybox agent inside a `bwrap` mount-namespace that bind-mounts
ONLY the declared inputs — essential system dirs + codified-KB + copied-in arch22 + the agent's
own workspace. `~/workspace/cann` and `output/` are NEVER bound, so the answer is PHYSICALLY
ABSENT from the sandbox fs-view → no read-path exists (subprocess / binary / symlink-deref all
moot). Production may stage provenance-tracked advisory artifacts inside its workspace; this
standalone probe intentionally does not. The construction-manifest (what was bound in) is DS's
conclusive primary audit-artifact.

Empirically verified 2026-06-08 (and pinned by test_graybox_sandbox.py): with only system +
KB + arch22 bound, a sandboxed process reaches KB+arch22 but CANNOT reach cann arch35 or
output/ (FileNotFoundError).

Design: docs/design/HERMETIC_REPRODUCTION_PROBE_DESIGN.md §"gap#2 graybox isolation mode".
The isolation contract is fail-closed: graybox must not reach the external source tree.
"""
from __future__ import annotations
import logging

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence

_BWRAP = shutil.which("bwrap")
_SANDBOX_EXEC = shutil.which("sandbox-exec")

# Essential read-only system dirs the agent's python / claude-CLI / compiler need. Only the
# ones that exist are bound (lib64 / sbin absent on some hosts). NONE of these expose the
# answer: cann + output/ live under $HOME which is NOT bound.
_SYSTEM_PARENTS = ("/usr", "/usr/local", "/etc", "/var", "/run")
_SYSTEM_RO = (
    "/usr/bin",
    "/usr/lib",
    "/usr/lib64",
    "/usr/libexec",
    "/usr/share",
    "/usr/local/bin",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/lib32",
    "/etc/alternatives",
    "/etc/ca-certificates",
    "/etc/pki",
    "/etc/ssl",
)
_SYSTEM_RO_FILES = (
    "/etc/group",
    "/etc/hosts",
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/nsswitch.conf",
    "/etc/passwd",
)
_TARGET_NAME_MARKERS = ("arch35", "ascend950", "dav_c310", "v351")
_TARGET_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
_CANN_ENV_ROOTS = (
    "ASCEND_AICPU_PATH",
    "ASCEND_HOME",
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    "ASCEND_TOOLKIT_HOME",
    "CANN_HOME",
)


def _scan_curated_tree(
    root: Path, *, assume_target_tree: bool = False
) -> tuple[int, int]:
    """Count target implementation sources and assembled-answer artifacts.

    Target/prior-art KB names are advisory metadata, so a name such as
    ``arch35`` is not itself evidence that an implementation is reachable.
    Source files *inside* a target-named tree remain forbidden, as do known
    assembled-answer and binary artifacts anywhere in a curated input.
    """
    target_sources = 0
    answer_files = 0
    root_is_target = any(marker in root.name.lower() for marker in _TARGET_NAME_MARKERS)
    pending = [(root, assume_target_tree or root_is_target)]
    while pending:
        current, in_target_tree = pending.pop()
        for item in sorted(current.iterdir()):
            if item.is_symlink():
                raise RuntimeError(f"graybox curated input contains symlink: {item}")
            if item.is_dir():
                item_is_target = any(
                    marker in item.name.lower() for marker in _TARGET_NAME_MARKERS
                )
                pending.append((item, in_target_tree or item_is_target))
            elif item.is_file():
                is_answer = (
                    item.name.endswith("_wp.cpp")
                    or (item.name.startswith("pybind11") and item.name.endswith(".cpp"))
                    or item.suffix in {".o", ".so"}
                )
                if is_answer:
                    answer_files += 1
                elif in_target_tree and item.suffix.lower() in _TARGET_SOURCE_SUFFIXES:
                    target_sources += 1
    return target_sources, answer_files


def bwrap_available() -> bool:
    """True iff bubblewrap is installed (a-fs airtight isolation requires it)."""
    return _BWRAP is not None


def isolation_backend(platform: str | None = None) -> str | None:
    """Return the supported strict isolation backend for this platform."""
    platform = platform or sys.platform
    if platform.startswith("linux"):
        return "bwrap" if _BWRAP else None
    if platform == "darwin":
        return "sandbox-exec" if _SANDBOX_EXEC else None
    return None


def isolation_available(platform: str | None = None) -> bool:
    return isolation_backend(platform) is not None


def _sandbox_quote(path: str | Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def build_sandbox_exec_cmd(
    inner_cmd: Sequence[str],
    *,
    allow_ro: Iterable[tuple[str, str]] = (),
    allow_rw: Iterable[tuple[str, str]] = (),
    workdir: str | None = None,
) -> list[str]:
    """Build a deny-default macOS Seatbelt profile for migration authoring."""
    if not _SANDBOX_EXEC:
        raise RuntimeError(
            "sandbox-exec is unavailable — refusing an unsandboxed migration spawn"
        )
    ro = list(allow_ro)
    rw = list(allow_rw)
    assert_no_answer_paths([*ro, *rw])
    system_read = (
        "/System",
        "/Library/Apple",
        "/Library/Preferences",
        "/bin",
        "/dev",
        "/etc",
        "/private/etc",
        "/private/var/db",
        "/private/var/select",
        "/sbin",
        "/usr/bin",
        "/usr/lib",
        "/usr/share",
        "/var/db",
        "/var/select",
    )
    rules = [
        "(version 1)",
        "(deny default)",
        "(deny network*)",
        "(allow process*)",
        "(allow signal (target self))",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        # Rosetta/dyld opens the filesystem root itself before resolving its
        # narrowly allowed runtime paths.  This does not grant subtree access.
        '(allow file-read* (literal "/"))',
    ]
    for path in system_read:
        if Path(path).exists():
            rules.append(f'(allow file-read* (subpath "{_sandbox_quote(path)}"))')

    def _lexical_and_real(path: str | Path) -> set[str]:
        raw = os.path.abspath(os.path.expanduser(str(path)))
        return {raw, str(Path(raw).resolve())}

    ancestor_literals: set[str] = set()
    for src, _dst in [*rw, *ro]:
        for candidate_text in _lexical_and_real(src):
            candidate = Path(candidate_text)
            ancestor_literals.update(
                str(parent) for parent in candidate.parents if str(parent) != "/"
            )
    for ancestor in sorted(ancestor_literals):
        rules.append(
            f'(allow file-read* (literal "{_sandbox_quote(ancestor)}"))'
        )
    for src, _dst in [*rw, *ro]:
        for candidate in sorted(_lexical_and_real(src)):
            rules.append(
                f'(allow file-read* (subpath "{_sandbox_quote(candidate)}"))'
            )
    # Claude/Node and compiler subprocesses need an ephemeral scratch area.
    # No persistent host directory or answer tree is exposed there.
    rules.append('(allow file-write* (subpath "/private/tmp"))')
    for src, _dst in rw:
        for candidate in sorted(_lexical_and_real(src)):
            rules.append(
                f'(allow file-write* (subpath "{_sandbox_quote(candidate)}"))'
            )
    # A source stage nested under the writable workspace stays immutable.
    for src, _dst in ro:
        for candidate in sorted(_lexical_and_real(src)):
            rules.append(
                f'(deny file-write* (subpath "{_sandbox_quote(candidate)}"))'
            )
    if workdir:
        resolved_workdir = str(Path(workdir).resolve())
        if not any(
            resolved_workdir == str(Path(src).resolve())
            or resolved_workdir.startswith(str(Path(src).resolve()) + os.sep)
            for src, _dst in rw
        ):
            raise ValueError("sandbox workdir must be under an allowed writable root")
    profile = "\n".join(rules)
    argv = [_SANDBOX_EXEC, "-p", profile]
    if workdir:
        argv.extend(
            [
                "/bin/sh",
                "-c",
                'cd "$1" && shift && exec "$@"',
                "aog-migration-sandbox",
                str(workdir),
            ]
        )
    argv.extend(inner_cmd)
    return argv


def build_isolated_cmd(
    inner_cmd: Sequence[str],
    *,
    allow_ro: Iterable[tuple[str, str]] = (),
    allow_rw: Iterable[tuple[str, str]] = (),
    workdir: str | None = None,
) -> list[str]:
    """Build the platform-native strict sandbox or fail closed."""
    backend = isolation_backend()
    if backend == "bwrap":
        return build_bwrap_cmd(
            inner_cmd,
            allow_ro=allow_ro,
            allow_rw=allow_rw,
            workdir=workdir,
            share_net=False,
        )
    if backend == "sandbox-exec":
        return build_sandbox_exec_cmd(
            inner_cmd,
            allow_ro=allow_ro,
            allow_rw=allow_rw,
            workdir=workdir,
        )
    raise RuntimeError(
        "no supported strict isolation backend (Linux requires bwrap; "
        "macOS requires sandbox-exec); refusing unsandboxed migration spawn"
    )


def build_bwrap_cmd(
    inner_cmd: Sequence[str],
    *,
    allow_ro: Iterable[tuple[str, str]] = (),
    allow_rw: Iterable[tuple[str, str]] = (),
    workdir: str | None = None,
    share_net: bool = False,
) -> list[str]:
    """Wrap `inner_cmd` in a bwrap mount-namespace exposing ONLY the explicitly-allowed paths
    plus essential read-only system dirs. Nothing else of the host filesystem is visible.

    AIRTIGHT INVARIANT (the gap#2 seal): NEVER pass a path under `~/workspace/cann` or any
    repo `output/` dir in allow_ro/allow_rw. Because $HOME is not bound, those answer-bearing
    trees are physically absent from the sandbox → unreachable by any means (subprocess open(),
    compiled binary, symlink-deref). Callers MUST only pass: codified-KB (src/skills/references),
    the copied-in arch22 input, and the agent's own workspace dir.

    Args:
      inner_cmd: the command to run inside the sandbox (e.g. the claude-CLI agent spawn argv).
      allow_ro: [(host_src, sandbox_dst)] read-only binds (KB, copied-in arch22, proto, config).
      allow_rw: [(host_src, sandbox_dst)] read-write binds (the agent's workspace dir only).
      workdir: chdir target inside the sandbox.
      share_net: retained for call compatibility; True is rejected.

    Returns the full argv list to exec.
    """
    if not _BWRAP:
        raise RuntimeError(
            "bwrap (bubblewrap) not available — a-fs airtight graybox isolation requires it. "
            "Install bubblewrap or fall back to a-trace (syscall-level open() trace)."
        )
    if share_net:
        raise ValueError(
            "migration isolation forbids network sharing; no bypass is supported"
        )
    assert_no_answer_paths([*allow_ro, *allow_rw])
    argv: list[str] = [_BWRAP]
    for parent in _SYSTEM_PARENTS:
        argv += ["--dir", parent]
    for d in _SYSTEM_RO:
        if os.path.isdir(d) and not os.path.islink(d):
            argv += ["--ro-bind", d, d]
    for path in _SYSTEM_RO_FILES:
        if os.path.isfile(path) and not os.path.islink(path):
            argv += ["--ro-bind", path, path]
    # symlinked top-levels (e.g. /lib -> usr/lib) — preserve the symlink so paths resolve.
    for d in ("/lib", "/lib64", "/bin", "/sbin"):
        if os.path.islink(d):
            argv += ["--symlink", os.readlink(d), d]
    argv += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    # Bind the mutable workspace first, then overlay the immutable source stage
    # and KB.  Reversing this order lets a parent workspace bind turn a nested
    # source snapshot writable again.
    for src, dst in allow_rw:
        argv += ["--bind", str(src), str(dst)]
    for src, dst in allow_ro:
        argv += ["--ro-bind", str(src), str(dst)]
    argv += ["--unshare-user", "--unshare-pid", "--die-with-parent"]
    argv += ["--unshare-net"]
    if workdir:
        argv += ["--chdir", str(workdir)]
    argv += ["--"]
    argv += list(inner_cmd)
    return argv


def assert_no_answer_paths(allow: Iterable[tuple[str, str]]) -> None:
    """Guard: raise if any allow-bind resolves under ~/workspace/cann or a repo output/ dir.
    Defense-in-depth against a caller accidentally binding the answer into the sandbox.
    """
    cann_roots = {
        os.path.abspath(os.path.expanduser("~/workspace/cann")),
        "/opt/Ascend",
        "/usr/local/Ascend",
    }
    for name in _CANN_ENV_ROOTS:
        value = os.environ.get(name)
        if value:
            cann_roots.add(os.path.abspath(os.path.expanduser(value)))
    for src, dst in allow:
        rp = os.path.abspath(os.path.expanduser(str(src)))
        dp = os.path.abspath(os.path.expanduser(str(dst)))
        for cann in cann_roots:
            if rp == cann or rp.startswith(cann + os.sep):
                raise ValueError(
                    "airtight violation: refusing to bind installed/source CANN "
                    f"tree into graybox sandbox: {src}"
                )
        src_target_named = any(
            marker in part.lower()
            for part in Path(rp).parts
            for marker in _TARGET_NAME_MARKERS
        )
        dst_target_named = any(
            marker in part.lower()
            for part in Path(dp).parts
            for marker in _TARGET_NAME_MARKERS
        )
        if (src_target_named or dst_target_named) and Path(rp).is_dir():
            target_sources, answer_files = _scan_curated_tree(
                Path(rp), assume_target_tree=True
            )
            if target_sources or answer_files:
                raise ValueError(
                    "airtight violation: refusing target implementation/answer "
                    f"artifacts in graybox bind: {src}"
                )
        elif (src_target_named or dst_target_named) and Path(rp).is_file():
            source = Path(rp)
            is_answer = (
                source.name.endswith("_wp.cpp")
                or (
                    source.name.startswith("pybind11")
                    and source.name.endswith(".cpp")
                )
                or source.suffix in {".o", ".so"}
            )
            if source.suffix.lower() in _TARGET_SOURCE_SUFFIXES or is_answer:
                raise ValueError(
                    "airtight violation: refusing target implementation/answer "
                    f"artifact in graybox bind: {src}"
                )
        # repo output/ trees (the assembled answer)
        parts = Path(rp).parts
        if "output" in parts and any(p in ("a3_to_a5_port", "backward_ops") for p in parts):
            raise ValueError(f"airtight violation: refusing to bind output/ answer dir into graybox sandbox: {src}")


# ---------------------------------------------------------------------------
# Toolchain bind-set — the dirs the claude-CLI graybox agent needs to RUN
# (interpreter / launcher / agent-config), discovered empirically 2026-06-08.
# NONE of these contain the answer: cann lives at ~/workspace/cann, the
# assembled answer at <repo>/output — neither is under any toolchain dir.
# Bind these SPECIFIC dirs (not the whole $HOME, which would re-expose them).
# The real bind-set is Part-2-validated against a live under-sandbox spawn;
# this is the grounded best-effort default + the existence-filter keeps it portable.
# ---------------------------------------------------------------------------
def default_toolchain_dirs() -> list[str]:
    """Existence-filtered list of host dirs the claude-CLI agent's runtime needs."""
    home = Path("~").expanduser()
    cand = [
        home / ".local" / "bin",            # claude launcher
        home / ".local" / "share" / "claude",  # claude versions
        home / ".nvm",                      # node (claude runtime)
    ]
    # A caller may provision a purpose-built auth directory containing no
    # projects, transcripts, memories, or learned target implementation.  The
    # ordinary CLAUDE_CONFIG_DIR/.claude tree is deliberately never exposed.
    sandbox_auth = os.environ.get("AOG_SANDBOX_AUTH_DIR")
    if sandbox_auth:
        cand.append(Path(sandbox_auth))
    # de-dup, keep only existing real dirs, never any answer tree
    seen: set[str] = set()
    out: list[str] = []
    for p in cand:
        skip_current_item = False
        try:
            rp = str(p.resolve())
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            skip_current_item = True
        if skip_current_item:
            continue
        if rp in seen or not Path(rp).is_dir():
            continue
        seen.add(rp)
        out.append(rp)
    return out


def graybox_allow_set(
    workspace: str | Path,
    *,
    kb_dir: str | Path,
    arch22_dir: str | Path | None = None,
    extra_ro: Iterable[str | Path] = (),
    toolchain_dirs: Iterable[str | Path] | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Build the (allow_ro, allow_rw) bind-set for a graybox agent spawn.

    Reads (ro): the codified KB (with the template-asset), the copied-in arch22 spec,
    any extra legal inputs, and the agent's runtime toolchain dirs. Writes (rw): only
    the agent's own isolated workspace. cann + output/ are NEVER listed → absent from
    the sandbox = airtight. Binds are dst==src (identity) so in-sandbox paths match host
    paths (the agent's KB-relative #include roots, scripts, configs all resolve unchanged).

    Raises (via assert_no_answer_paths) if any input accidentally resolves under the answer.
    """
    tc = list(toolchain_dirs) if toolchain_dirs is not None else default_toolchain_dirs()
    ro_srcs: list[str] = [str(Path(kb_dir).resolve())]
    if arch22_dir is not None:
        ro_srcs.append(str(Path(arch22_dir).resolve()))
    ro_srcs += [str(Path(p).resolve()) for p in extra_ro]
    ro_srcs += [str(Path(p).resolve()) for p in tc]

    ws = str(Path(workspace).resolve())
    allow_ro = [(s, s) for s in ro_srcs]
    allow_rw = [(ws, ws)]
    # airtight backstop: refuse if anything resolves into cann/output
    assert_no_answer_paths(allow_ro + allow_rw)
    return allow_ro, allow_rw


def write_construction_manifest(
    workspace: str | Path,
    allow_ro: list[tuple[str, str]],
    allow_rw: list[tuple[str, str]],
    *,
    inner_cmd: Sequence[str],
    scan_srcs: Iterable[str | Path] | None = None,
    out_name: str = "construction_manifest.json",
) -> Path:
    """Emit the construction-manifest — DS's PRIMARY gap#2 audit artifact.

    Records exactly what was bound into the sandbox + a hard self-assertion that ZERO
    target implementation sources and ZERO assembled-answer .cpp/.so were reachable. DS
    reads this (not agent self-report) to certify the run was airtight. The bwrap
    mount-namespace is the actual enforcement; the manifest is the declarative proof of
    what enforcement was applied.

    The structural seal is twofold: (1) assert_no_answer_paths — none of the binds is under
    cann/output; (2) deep-scan of the INPUT binds (`scan_srcs`, default = KB+arch22+workspace,
    i.e. allow_rw + the first allow_ro entries) confirming the curated inputs don't smuggle an
    target implementation source or assembled answer .cpp. Advisory target/prior-art KB
    names alone are not implementation artifacts. Toolchain binds (python/node/claude
    infra) are recorded but NOT deep-scanned — they are known-safe infra and huge.
    """
    ws = Path(workspace)
    # Defense-in-depth: re-run the answer-path guard at manifest time.
    assert_no_answer_paths(allow_ro + allow_rw)

    # Deep-scan set: explicit scan_srcs, else the curated INPUT binds = workspace (always)
    # + any allow_ro src that is NOT a known toolchain dir (i.e. KB + arch22 + extra inputs).
    # Toolchain binds (python/node/claude infra) are skipped — known-safe + huge.
    if scan_srcs is None:
        toolchain = set(default_toolchain_dirs())
        scan_set = {s for s, _ in allow_rw} | {s for s, _ in allow_ro if s not in toolchain}
    else:
        scan_set = {str(Path(s).resolve()) for s in scan_srcs}

    arch35_reachable = 0
    answer_cpp_reachable = 0
    bound = []
    for src, dst in list(allow_ro) + list(allow_rw):
        sp = Path(src)
        scanned = src in scan_set
        entry = {
            "src": src, "dst": dst,
            "mode": "ro" if (src, dst) in allow_ro else "rw",
            "deep_scanned": scanned,
        }
        if scanned:
            if sp.is_dir():
                target_dirs, answer_files = _scan_curated_tree(sp)
                arch35_reachable += target_dirs
                answer_cpp_reachable += answer_files
        bound.append(entry)

    backend = isolation_backend()
    if backend is None:
        raise RuntimeError(
            "cannot certify construction without a supported strict isolation backend"
        )
    manifest = {
        "schema": "graybox_construction_manifest/v1",
        "mechanism": (
            "bwrap-mount-namespace"
            if backend == "bwrap"
            else "macos-sandbox-exec-deny-default"
        ),
        "isolation_backend": backend,
        "backend_path": _BWRAP if backend == "bwrap" else _SANDBOX_EXEC,
        "network": "denied",
        "inner_cmd": list(inner_cmd),
        "bound": bound,
        "assertions": {
            # Kept for construction-manifest/v1 compatibility; this now counts
            # implementation sources inside target-named trees, not directory names.
            "arch35_dirs_reachable": arch35_reachable,
            "assembled_answer_cpp_reachable": answer_cpp_reachable,
            "airtight": (arch35_reachable == 0 and answer_cpp_reachable == 0),
        },
    }
    out = ws / out_name
    out.write_text(json.dumps(manifest, indent=2))
    if not manifest["assertions"]["airtight"]:
        raise RuntimeError(
            "graybox construction rejected target/answer-bearing curated input; "
            f"see {out}"
        )
    return out
