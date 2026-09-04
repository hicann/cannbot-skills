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
import hashlib
import logging

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

_BWRAP = shutil.which("bwrap")
_SANDBOX_EXEC = shutil.which("sandbox-exec")

# How the graybox sandbox provides /proc (2026-08-21, restricted-container fix).
# Some nested/restricted containers (e.g. DSH-style per-command sandboxes) allow
# devtmpfs/tmpfs/bind mounts but deny mounting a FRESH proc instance inside a
# child user+pid namespace (mount -t proc → EPERM), and bwrap treats that as
# fatal for the whole spawn.  Probe once per process: keep the strict fresh
# --proc mount when the kernel permits it, otherwise degrade to a read-only
# bind of the outer /proc.  The airtight FILE allow-set (KB + source stage +
# workspace, everything else absent) is identical in both modes; only the proc
# view is inherited read-only instead of being freshly mounted.
_BWRAP_PROC_MODE: str | None = None


def _bwrap_proc_mode() -> str:
    global _BWRAP_PROC_MODE
    if _BWRAP_PROC_MODE is not None:
        return _BWRAP_PROC_MODE
    override = os.environ.get("AOG_GRAYBOX_BWRAP_PROC_MODE")
    if override in ("fresh", "robind"):
        _BWRAP_PROC_MODE = override
        return _BWRAP_PROC_MODE
    if not _BWRAP:
        _BWRAP_PROC_MODE = "robind"
        return _BWRAP_PROC_MODE
    probe = [
        _BWRAP, "--unshare-user", "--unshare-pid", "--proc", "/proc",
        "--dev-bind", "/", "/", "--", "true",
    ]
    try:
        res = subprocess.run(
            probe, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=15, check=False,
        )
        _BWRAP_PROC_MODE = "fresh" if res.returncode == 0 else "robind"
    except Exception:
        _BWRAP_PROC_MODE = "robind"
    if _BWRAP_PROC_MODE == "robind":
        logging.getLogger(__name__).warning(
            "graybox: fresh proc mount denied in this container (bwrap probe failed); "
            "degrading to read-only /proc bind — file isolation unchanged"
        )
    return _BWRAP_PROC_MODE

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
    # On Linux containers Claude Code is commonly installed under
    # /usr/local/lib/node_modules and exposed through /usr/local/bin/claude.
    # The latter is a symlink, so binding only /usr/local/bin leaves the
    # executable target absent inside bwrap.  Keep this narrow runtime tree
    # read-only; /usr/local/Ascend is deliberately not included.
    "/usr/local/lib",
    "/usr/local/lib64",
    "/usr/local/share",
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
    # A Kimi/Anthropic-compatible worker needs DNS resolution when the
    # deliberately opt-in model network namespace is shared.  `/etc/hosts`
    # alone cannot resolve ordinary API hostnames; bind this resolver config
    # read-only without exposing the rest of /etc.
    "/etc/resolv.conf",
)
_TARGET_NAME_MARKERS = ("arch35", "ascend950", "dav_c310", "v351")

# Harness-generated trees excluded from the answer-bearing deep scan (see
# _scan_curated_tree): they hold the agent's OWN built artifacts, never
# curated reference implementations.
_HARNESS_ARTIFACT_DIRS = {".npubench_candidate", ".npubench_exec", "npubench_evidence"}
_TARGET_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
_CANN_ENV_ROOTS = (
    "ASCEND_AICPU_PATH",
    "ASCEND_HOME",
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    "ASCEND_TOOLKIT_HOME",
    "CANN_HOME",
)
# Alternate in-sandbox mount for the declarative plugin runtime.  The host
# marketplace cache path is intentionally not exposed to the worker; Claude is
# given this path through ``--plugin-dir`` after only safe plugin subtrees are
# bound there.
DEFAULT_PLUGIN_MOUNT = "/usr/local/cannbot-port-plugin"


def _scan_curated_tree(
    root: Path, *, assume_target_tree: bool = False,
    allow_source_pybind: bool = False,
    source_stage_root: str | Path | None = None,
) -> tuple[int, int]:
    """Count target implementation sources and assembled-answer artifacts.

    Target/prior-art KB names are advisory metadata, so a name such as
    ``arch35`` is not itself evidence that an implementation is reachable.
    Source files *inside* a target-named tree remain forbidden, as do known
    assembled-answer and binary artifacts anywhere in a curated input.
    """
    target_sources = 0
    answer_files = 0
    # ``allow_source_pybind`` is retained for the small direct-stage probe API.
    # Production workspace scans pass the exact immutable stage root instead of
    # granting a basename-wide exemption to arbitrary workspace directories.
    allowed_stage = (
        Path(source_stage_root).resolve()
        if source_stage_root is not None
        else (root.resolve() if allow_source_pybind else None)
    )

    def _in_source_stage(path: Path) -> bool:
        if allowed_stage is None:
            return False
        try:
            path.resolve().relative_to(allowed_stage)
        except ValueError:
            return False
        return True

    root_is_target = any(marker in root.name.lower() for marker in _TARGET_NAME_MARKERS)
    root_resolved = root.resolve()
    pending = [(root, assume_target_tree or root_is_target)]
    while pending:
        current, in_target_tree = pending.pop()
        source_stage_tree = _in_source_stage(current)
        for item in sorted(current.iterdir()):
            if item.is_symlink():
                # 2026-08-29 (76_BatchMLAPagedAttention_evo kw-6 spawn rc=3):
                # harness-owned runtime copies under workspace/.opencode-runtime
                # acquire npm/bun node_modules/.bin INTRA-TREE symlinks at
                # runtime (something runs an install into the config copy
                # mid-session); hard-failing the seal on them parked the line.
                # The airtight concern is content ESCAPING the curated tree,
                # so tolerate symlinks that resolve inside the scanned root
                # (their target content is already scanned in place) and keep
                # rejecting anything pointing outside it.
                resolved = item.resolve()
                if resolved.is_relative_to(root_resolved):
                    continue
                raise RuntimeError(f"graybox curated input contains symlink: {item}")
            if item.is_dir():
                if item.name in _HARNESS_ARTIFACT_DIRS:
                    # Harness-owned build/eval trees (content-addressed candidate
                    # snapshots, exec scratch, evidence) are generated by the
                    # orchestrator from the agent's OWN sources — they are not
                    # curated/answer-bearing input.  The agent is expected to
                    # read its own built artifacts while debugging (probe),
                    # and a respawn must not be rejected because a previous
                    # O5 left kernel/build/*.so inside .npubench_candidate
                    # (2026-08-22 BAM/SDPA await_probe spawn rejections).
                    continue
                item_is_target = any(
                    marker in item.name.lower() for marker in _TARGET_NAME_MARKERS
                )
                pending.append((item, in_target_tree or item_is_target))
            elif item.is_file():
                is_answer = (
                    item.name.endswith("_wp.cpp")
                    or (
                        item.name.startswith("pybind11")
                        and item.name.endswith(".cpp")
                        and not source_stage_tree
                    )
                    or item.suffix in {".o", ".so"}
                )
                if is_answer:
                    answer_files += 1
                elif (
                    in_target_tree
                    and not source_stage_tree
                    and item.suffix.lower() in _TARGET_SOURCE_SUFFIXES
                ):
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


def _macos_network_rule(share_net: bool) -> str:
    """Return the explicit Seatbelt network rule for the requested policy."""
    # ``(deny default)`` does not make an opt-in network policy work: an
    # explicit allow is required for every operation class that is otherwise
    # denied.  Keep the deny rule explicit too, so the construction manifest
    # and generated profile describe the same effective policy.
    return "(allow network*)" if share_net else "(deny network*)"


def build_sandbox_exec_cmd(
    inner_cmd: Sequence[str],
    *,
    allow_ro: Iterable[tuple[str, str]] = (),
    allow_rw: Iterable[tuple[str, str]] = (),
    workdir: str | None = None,
    share_net: bool = False,
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
        _macos_network_rule(share_net),
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
    # Claude Code creates a short-lived per-user runtime directory through the
    # lexical /tmp path on macOS (/tmp is a symlink to /private/tmp).  Seatbelt
    # does not consistently normalize that alias, so grant both spellings and
    # the standard null sink needed by launcher stderr redirection.
    rules.extend(
        (
            '(allow file-read* (subpath "/private/tmp"))',
            '(allow file-read* (subpath "/tmp"))',
            '(allow file-write* (subpath "/private/tmp"))',
            '(allow file-write* (subpath "/tmp"))',
            '(allow file-read* (literal "/dev/null"))',
            '(allow file-write* (literal "/dev/null"))',
        )
    )
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
    share_net: bool = False,
) -> list[str]:
    """Build the platform-native strict sandbox or fail closed."""
    backend = isolation_backend()
    if backend == "bwrap":
        return build_bwrap_cmd(
            inner_cmd,
            allow_ro=allow_ro,
            allow_rw=allow_rw,
            workdir=workdir,
            share_net=share_net,
        )
    if backend == "sandbox-exec":
        return build_sandbox_exec_cmd(
            inner_cmd,
            allow_ro=allow_ro,
            allow_rw=allow_rw,
            workdir=workdir,
            share_net=share_net,
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
      share_net: when true, preserve model endpoint network access while keeping
        the filesystem allow-set unchanged.

    Returns the full argv list to exec.
    """
    if not _BWRAP:
        raise RuntimeError(
            "bwrap (bubblewrap) not available — a-fs airtight graybox isolation requires it. "
            "Install bubblewrap or fall back to a-trace (syscall-level open() trace)."
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
    if _bwrap_proc_mode() == "fresh":
        argv += ["--proc", "/proc"]
    else:
        # Degraded mode (restricted container): fresh proc mounts are denied,
        # bind the outer /proc read-only.  The file allow-set is unchanged.
        argv += ["--ro-bind", "/proc", "/proc"]
    argv += ["--dev", "/dev", "--tmpfs", "/tmp"]
    # bwrap requires an existing destination parent for nested plugin binds.
    # /usr/local itself is created above (and its lib/share children are
    # system binds), so the dedicated plugin root is safe to create here.
    plugin_dest = Path(DEFAULT_PLUGIN_MOUNT)
    if any(
        str(dst) == str(plugin_dest)
        or str(dst).startswith(str(plugin_dest) + os.sep)
        for _src, dst in [*allow_ro, *allow_rw]
    ):
        argv += ["--dir", str(plugin_dest)]
    # Bind the mutable workspace first, then overlay the immutable source stage
    # and KB.  Reversing this order lets a parent workspace bind turn a nested
    # source snapshot writable again.
    for src, dst in allow_rw:
        argv += ["--bind", str(src), str(dst)]
    for src, dst in allow_ro:
        argv += ["--ro-bind", str(src), str(dst)]
    argv += ["--unshare-user", "--unshare-pid", "--die-with-parent"]
    if not share_net:
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


_PLUGIN_BIND_PARTS = (
    ".claude-plugin",
    "AGENTS.md",
    "agents",
    "skills",
    "kb",
    "hooks",
    "scripts",
    "workflows",
    "engine/src/scripts",
)
_PLUGIN_BIND_OPTIONAL_PARTS = frozenset({"AGENTS.md", "scripts", "workflows"})


def _validated_plugin_mount(plugin_mount: str) -> Path:
    """Return the in-sandbox plugin mount root, refusing a non-absolute path."""
    mount_root = Path(plugin_mount)
    if not mount_root.is_absolute() or mount_root == Path("/"):
        raise ValueError("plugin_mount must be an absolute non-root path")
    return mount_root


def _is_absent_optional_plugin_bind(source: Path, relative: str) -> bool:
    """Report an optional plugin subtree that this snapshot simply does not have."""
    return relative in _PLUGIN_BIND_OPTIONAL_PARTS and not source.exists()


def _plugin_runtime_ro_binds(plugin_root: Path, mount_root: Path) -> list[tuple[str, str]]:
    """Return the read-only binds for a staged plugin runtime.

    The bound subtrees are kept explicit.  In particular, engine/ is never
    bound as a whole: engine/workspace and engine/output are mutable
    answer-bearing trees.
    """
    binds: list[tuple[str, str]] = []
    for relative in (*_PLUGIN_BIND_PARTS, _PLUGIN_DEPENDENCY_MANIFEST):
        source = plugin_root / relative
        if _is_absent_optional_plugin_bind(source, relative):
            continue
        if not source.exists() or source.is_symlink():
            raise ValueError(f"graybox plugin subtree missing or symlinked: {source}")
        binds.append((str(source), str(mount_root / relative)))
    return binds


def graybox_allow_set(
    workspace: str | Path,
    *,
    kb_dir: str | Path,
    arch22_dir: str | Path | None = None,
    extra_ro: Iterable[str | Path] = (),
    toolchain_dirs: Iterable[str | Path] | None = None,
    plugin_dir: str | Path | None = None,
    plugin_mount: str = DEFAULT_PLUGIN_MOUNT,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Build the (allow_ro, allow_rw) bind-set for a graybox agent spawn.

    Reads (ro): the codified KB (with the template-asset), the copied-in arch22 spec,
    any extra legal inputs, the agent's runtime toolchain dirs, and (when requested) the
    minimal declarative/runtime subtrees of the plugin used with Claude's ``--plugin-dir``.
    Writes (rw): only the agent's own isolated workspace. cann + output/ are NEVER listed
    → absent from the sandbox = airtight. Ordinary binds are dst==src (identity); the
    plugin is mounted at a dedicated alternate path so its host checkout is never exposed.

    Raises (via assert_no_answer_paths) if any input accidentally resolves under the answer.
    """
    plugin_root: Path | None = None
    tc = list(toolchain_dirs) if toolchain_dirs is not None else default_toolchain_dirs()
    try:
        ro_srcs: list[tuple[str, str]] = []
        kb_resolved = str(Path(kb_dir).resolve())
        ro_srcs.append((kb_resolved, kb_resolved))
        if arch22_dir is not None:
            resolved = str(Path(arch22_dir).resolve())
            ro_srcs.append((resolved, resolved))
        ro_srcs += [
            (str(Path(p).resolve()), str(Path(p).resolve())) for p in extra_ro
        ]
        ro_srcs += [(str(Path(p).resolve()), str(Path(p).resolve())) for p in tc]

        if plugin_dir is not None:
            mount_root = _validated_plugin_mount(plugin_mount)
            # Materialize marketplace symlinks inside the task workspace before
            # binding.  A direct bind of the installed cache would leave shared
            # skills pointing at the unmounted ordinary CLAUDE_CONFIG_DIR.
            plugin_root = stage_plugin_runtime(plugin_dir, workspace)
            ro_srcs += _plugin_runtime_ro_binds(plugin_root, mount_root)

        ws = str(Path(workspace).resolve())
        allow_ro = ro_srcs
        allow_rw = [(ws, ws)]
        # airtight backstop: refuse if anything resolves into cann/output
        assert_no_answer_paths(allow_ro + allow_rw)
        return allow_ro, allow_rw
    except BaseException:
        if plugin_root is not None:
            _cleanup_owned_plugin_runtime(plugin_root)
        raise


def graybox_plugin_runtime_dir(plugin_root: str | Path) -> Path:
    """Return the installed plugin root after validating its boundary.

    This helper is intentionally read-only.  Use :func:`stage_plugin_runtime`
    before a graybox spawn so the worker never receives the ordinary Claude
    config tree or the installed plugin's mutable ``engine/workspace``.
    """
    root = Path(plugin_root).resolve()
    if not root.is_dir():
        raise ValueError(f"plugin root is not a directory: {plugin_root}")
    return root


_PLUGIN_RUNTIME_DIRNAME = ".graybox_plugin_runtime"
_PLUGIN_RUNTIME_PARTS = (
    ".claude-plugin",
    "agents",
    "hooks",
    "kb",
    "skills",
    "engine/src/scripts",
)
_PLUGIN_RUNTIME_OPTIONAL_PARTS = ("AGENTS.md", "scripts", "workflows")
_PLUGIN_RUNTIME_MARKER = "graybox-plugin-runtime/v1\n"
_PLUGIN_DEPENDENCY_MANIFEST = ".graybox_dependency_manifest.json"


def _new_plugin_runtime_staging(ws: Path) -> Path:
    """Create the private staging root for one plugin snapshot.

    ``tempfile`` intentionally uses the host temporary root rather than the
    task workspace.  Do not add a marker-based reuse path: any file below a
    writable workspace bind is attacker-controlled by the worker.
    """
    staging = Path(tempfile.mkdtemp(prefix=f"{_PLUGIN_RUNTIME_DIRNAME}-"))
    try:
        staging.resolve().relative_to(ws)
    except ValueError:
        return staging
    # ``TMPDIR`` is process-configurable.  Refuse a launcher environment that
    # redirects temporary files into the worker workspace rather than
    # accidentally placing a supposedly immutable runtime below its RW bind.
    shutil.rmtree(staging, ignore_errors=True)
    raise RuntimeError(
        "graybox plugin runtime temporary directory is inside workspace"
    )


def _declared_plugin_dependencies(root: Path) -> list:
    """Return the raw ``dependencies`` list declared by a plugin manifest.

    Claude's marketplace cache installs dependencies as sibling plugin
    directories.  The normal config root makes those skills visible, but
    graybox deliberately does not mount that root, so the declarations are read
    here and merged into the private runtime instead.
    """
    source_manifest = root / ".claude-plugin" / "plugin.json"
    try:
        source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        source_payload = {}
    dependencies = (
        source_payload.get("dependencies", [])
        if isinstance(source_payload, dict)
        else []
    )
    if not isinstance(dependencies, list):
        raise RuntimeError("graybox plugin manifest dependencies must be a list")
    return dependencies


def _direct_checkout_dependency_roots(root: Path) -> dict[str, Path]:
    """Map declared dependency names to their repository-checkout roots.

    A direct repository checkout has the shared packages beside the plugin
    rather than in a Claude marketplace cache.  The normal installer exposes
    those packages through CLAUDE_CONFIG_DIR, but graybox deliberately does not
    mount that mutable config tree.  Resolve only the two declared,
    repository-owned dependency layouts; never broaden this into an arbitrary
    parent-directory skill search.
    """
    if root.parent.name != "plugins-community":
        return {}
    checkout_root = root.parent.parent
    return {
        "ascendc-port-orchestrator-shared-skills": checkout_root / "ops",
        "cannbot-knowledge-consumer-skills": root.parent / "cannbot-knowledge",
    }


def _resolve_one_dependency(
    dependency: object,
    direct_roots: dict[str, Path],
    cache_root: Path,
) -> tuple[dict[str, str], Path | None]:
    """Resolve one declared dependency to a record and its root (or None)."""
    if not isinstance(dependency, str) or not dependency.strip():
        return {"requested": str(dependency), "status": "invalid"}, None
    name = dependency.split("@", 1)[0].strip()
    if not name:
        return {"requested": dependency, "status": "invalid"}, None
    direct_root = direct_roots.get(name)
    if direct_root is not None and direct_root.is_dir():
        return {
            "requested": dependency,
            "name": name,
            "status": "resolved",
            "source": "direct_checkout",
            "root_name": direct_root.name,
        }, direct_root
    family = cache_root / name
    versions = _sorted_cache_versions(family)
    if not versions:
        return {
            "requested": dependency,
            "name": name,
            "status": "missing",
            "source": "marketplace_cache",
        }, None
    return {
        "requested": dependency,
        "name": name,
        "status": "resolved",
        "source": "marketplace_cache",
        "root_name": versions[-1].name,
    }, versions[-1]


def _sorted_cache_versions(family: Path) -> list[Path]:
    """Return the marketplace-cache version dirs of one dependency family."""
    if not family.is_dir():
        return []
    versions: list[Path] = []
    for candidate in family.iterdir():
        if candidate.is_dir():
            versions.append(candidate)
    versions.sort(key=lambda candidate: candidate.name)
    return versions


def _resolve_plugin_dependencies(
    root: Path,
    dependencies: list,
) -> tuple[list[dict[str, str]], list[str], list[tuple[Path, dict[str, str]]]]:
    """Resolve declared dependencies to (records, unresolved, roots)."""
    records: list[dict[str, str]] = []
    unresolved: list[str] = []
    roots: list[tuple[Path, dict[str, str]]] = []
    direct_roots = _direct_checkout_dependency_roots(root)
    cache_root = root.parent.parent if root.parent.name else root.parent
    for dependency in dependencies:
        record, resolved_root = _resolve_one_dependency(
            dependency, direct_roots, cache_root
        )
        records.append(record)
        if resolved_root is None:
            unresolved.append(str(dependency))
        else:
            roots.append((resolved_root, record))
    return records, unresolved, roots


def _assert_dependencies_resolved(unresolved: list[str]) -> None:
    """Fail closed unless the operator opted into an unresolved dependency."""
    if not unresolved:
        return
    override = os.environ.get(
        "CANNBOT_GRAYBOX_ALLOW_UNRESOLVED_DEPENDENCIES", ""
    ).strip().lower()
    if override in {"1", "true", "yes"}:
        return
    raise RuntimeError(
        "graybox declared plugin dependencies are unresolved: "
        + ", ".join(unresolved)
        + "; install/materialize them before spawning a worker"
    )


def _copy_plugin_runtime_parts(root: Path, staging: Path) -> None:
    """Materialize the declared plugin subtrees into the staging root."""
    for relative in (*_PLUGIN_RUNTIME_PARTS, *_PLUGIN_RUNTIME_OPTIONAL_PARTS):
        source = root / relative
        if not source.exists() and relative in _PLUGIN_RUNTIME_OPTIONAL_PARTS:
            continue
        if not source.exists():
            raise RuntimeError(f"required plugin runtime path is missing: {source}")
        target = staging / relative
        if source.is_dir():
            _copy_runtime_tree(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target, follow_symlinks=True)


def _copy_runtime_tree(source: Path, target: Path) -> None:
    """Copy one runtime subtree, dereferencing symlinks and dropping caches."""
    shutil.copytree(
        source,
        target,
        symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".git"),
    )


def _is_mergeable_skill_dir(skill: Path, staged_skills: Path) -> bool:
    """Accept only a real skill directory that is not already staged."""
    if skill.name.startswith(".") or not skill.is_dir():
        return False
    if not (skill / "SKILL.md").is_file():
        return False
    return not (staged_skills / skill.name).exists()


def _merge_one_dependency_skills(staged_skills: Path, dependency_root: Path) -> None:
    """Merge one dependency package's skills into the staged skills tree."""
    dependency_skills = dependency_root / "skills"
    if not dependency_skills.is_dir():
        # Some shared-skill marketplace packages expose skill folders directly
        # at their root rather than under ``skills/``.
        dependency_skills = dependency_root
    for skill in sorted(dependency_skills.iterdir(), key=lambda path: path.name):
        if not _is_mergeable_skill_dir(skill, staged_skills):
            continue
        _copy_runtime_tree(skill, staged_skills / skill.name)


def _merge_dependency_skills(
    staging: Path,
    dependency_roots: list[tuple[Path, dict[str, str]]],
) -> None:
    """Merge dependency skills into the staged runtime.

    Runs after the main plugin so a product-owned skill always wins a name
    collision.  A missing dependency is left for the normal install/health
    checks to report; this staging helper must never invent or broaden a
    dependency search path.  A root-style dependency package may contain
    infrastructure directories alongside its skills, so accept only
    directories that are actual skills.
    """
    staged_skills = staging / "skills"
    for dependency_root, _record in dependency_roots:
        _merge_one_dependency_skills(staged_skills, dependency_root)


def _strip_manifest_dependencies(staging: Path) -> None:
    """Remove the install-time dependency declaration from the snapshot.

    The full marketplace dependency graph lives in CLAUDE_CONFIG_DIR, which is
    intentionally absent from graybox.  Shared dependency skills were
    materialized through the plugin's symlinks, so only the declaration itself
    is dropped here.
    """
    manifest = staging / ".claude-plugin" / "plugin.json"
    if not manifest.exists():
        # Minimal synthetic plugin roots used by unit tests and probes need no
        # manifest; Claude itself will reject such a root if somebody attempts
        # to launch it, but the bind-set remains testable.
        manifest.write_text(json.dumps({"name": "graybox-plugin", "version": "0"}) + "\n")
    try:
        payload = json.loads(manifest.read_text())
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f"invalid plugin manifest in graybox snapshot: {manifest}") from error
    if isinstance(payload, dict) and payload.get("dependencies"):
        payload.pop("dependencies", None)
        manifest.write_text(json.dumps(payload, indent=2) + "\n")


def _write_dependency_manifest(
    staging: Path,
    declared: list[str],
    records: list[dict[str, str]],
    unresolved: list[str],
) -> None:
    """Write the staged runtime's dependency audit manifest."""
    staged_skills = staging / "skills"
    staged_skill_names: list[str] = []
    for path in staged_skills.iterdir():
        if path.is_dir() and (path / "SKILL.md").is_file():
            staged_skill_names.append(path.name)
    dependency_manifest = {
        "schema": "graybox_plugin_dependencies/v1",
        "declared": declared,
        "records": records,
        "unresolved": unresolved,
        "staged_skill_names": sorted(staged_skill_names),
        "staged_tree_sha256": _staged_runtime_tree_sha256(staging),
    }
    (staging / _PLUGIN_DEPENDENCY_MANIFEST).write_text(
        json.dumps(
            dependency_manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _discard_plugin_runtime_staging(staging: Path) -> None:
    """Remove a half-built snapshot without masking the original failure.

    The snapshot has just been created by its constructor, so it is safe to
    remove even before the ownership marker exists.  Keep this cleanup local to
    the constructor: the public cleanup helpers remain marker-gated and
    fail-closed for caller-supplied paths.
    """
    try:
        shutil.rmtree(staging, ignore_errors=True)
    except BaseException as error:
        # Deliberate suppression: the caller is already propagating a
        # BaseException and cleanup must never replace it.
        logging.getLogger(__name__).debug(
            "Recoverable operation failed.", exc_info=error
        )


def stage_plugin_runtime(plugin_root: str | Path, workspace: str | Path) -> Path:
    """Create a self-contained, answer-free plugin snapshot for graybox.

    The installed marketplace cache contains symlinks back into the ordinary
    Claude config tree.  That tree is intentionally not mounted in graybox,
    so materialize the declared plugin subtrees into a fresh private directory
    outside the task workspace and bind them read-only at a stable alternate
    path.  Generated engine/workspace and unrelated examples are omitted.

    The snapshot must not live below ``workspace``.  The workspace is the
    worker's sole writable bind and therefore a worker could otherwise mutate
    (or replace) a marker-validated snapshot and affect a later spawn.  A new
    snapshot is deliberately created for every call; the caller/dispatcher
    owns its lifetime for the duration of the spawn.
    """
    root = graybox_plugin_runtime_dir(plugin_root)
    ws = Path(workspace).resolve()
    if not ws.is_dir():
        raise ValueError(f"workspace is not a directory: {workspace}")
    staging = _new_plugin_runtime_staging(ws)
    try:
        dependencies = _declared_plugin_dependencies(root)
        declared_dependencies = [str(item) for item in dependencies]
        records, unresolved, dependency_roots = _resolve_plugin_dependencies(
            root, dependencies
        )
        _assert_dependencies_resolved(unresolved)
        _copy_plugin_runtime_parts(root, staging)
        _merge_dependency_skills(staging, dependency_roots)
        _strip_manifest_dependencies(staging)
        _write_dependency_manifest(
            staging, declared_dependencies, records, unresolved
        )
        (staging / ".runtime_complete").write_text(_PLUGIN_RUNTIME_MARKER)
    except BaseException:
        _discard_plugin_runtime_staging(staging)
        raise
    return staging


def _staged_runtime_tree_sha256(root: Path) -> str:
    """Hash staged runtime bytes excluding mutable audit metadata."""
    entries: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {
            _PLUGIN_DEPENDENCY_MANIFEST,
            ".runtime_complete",
        }:
            continue
        entries.append(
            (
                path.relative_to(root).as_posix(),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    return hashlib.sha256(
        json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def staged_plugin_runtime_roots(
    allow_ro: Iterable[tuple[str, str]],
    *,
    plugin_mount: str = DEFAULT_PLUGIN_MOUNT,
) -> tuple[Path, ...]:
    """Identify runtime snapshots owned by a graybox allow-set.

    This accepts only fresh ``tempfile`` roots bearing our exact completion
    marker.  The narrow predicate makes cleanup idempotent and prevents a
    caller-supplied plugin source from becoming a deletion target.
    """
    mount = Path(plugin_mount)
    roots: set[Path] = set()
    for source, destination in allow_ro:
        try:
            relative = Path(destination).relative_to(mount)
        except ValueError:
            continue
        root = Path(source)
        for _ in relative.parts:
            root = root.parent
        owned = _owned_plugin_runtime_root(root)
        if owned is not None:
            roots.add(owned)
    return tuple(sorted(roots))


def _owned_plugin_runtime_root(root: str | Path) -> Path | None:
    """Return a validated staging root, or ``None`` for an untrusted path."""
    raw = Path(root)
    if raw.is_symlink():
        return None
    candidate = raw.resolve()
    try:
        candidate.relative_to(Path(tempfile.gettempdir()).resolve())
    except ValueError:
        return None
    if not candidate.name.startswith(f"{_PLUGIN_RUNTIME_DIRNAME}-"):
        return None
    marker = candidate / ".runtime_complete"
    try:
        if not marker.is_file() or marker.read_text(encoding="utf-8") != _PLUGIN_RUNTIME_MARKER:
            return None
    except OSError:
        return None
    return candidate


def _cleanup_owned_plugin_runtime(root: str | Path) -> None:
    """Remove one staging root only after validating its ownership marker."""
    owned = _owned_plugin_runtime_root(root)
    if owned is not None:
        shutil.rmtree(owned, ignore_errors=True)


def cleanup_staged_plugin_runtimes(
    allow_ro: Iterable[tuple[str, str]],
    *,
    plugin_mount: str = DEFAULT_PLUGIN_MOUNT,
) -> None:
    """Remove only runtime snapshots created for this graybox dispatch."""
    for root in staged_plugin_runtime_roots(allow_ro, plugin_mount=plugin_mount):
        _cleanup_owned_plugin_runtime(root)


def plugin_dir_for_isolation_backend(
    allow_ro: Iterable[tuple[str, str]],
    *,
    backend: str | None = None,
    plugin_mount: str = DEFAULT_PLUGIN_MOUNT,
) -> str:
    """Return the plugin directory visible to a strict isolation backend.

    Linux ``bwrap`` mounts the staged runtime at ``plugin_mount``.  macOS
    ``sandbox-exec`` has no mount namespace, so the worker must receive the
    actual staged host source path that is present in ``allow_ro``.  Failing
    closed when the declarative plugin bind is absent avoids passing Claude a
    path that cannot exist in the selected sandbox.
    """
    selected = backend or isolation_backend()
    if not isinstance(plugin_mount, str) or not Path(plugin_mount).is_absolute() or Path(plugin_mount) == Path("/"):
        raise ValueError("plugin_mount must be an absolute non-root path")
    if selected != "sandbox-exec":
        if selected == "bwrap":
            return plugin_mount
        raise ValueError(f"unsupported isolation backend: {selected!r}")
    expected = str(Path(plugin_mount) / ".claude-plugin")
    matches = [str(Path(src).resolve().parent) for src, dst in allow_ro if str(dst) == expected]
    if len(matches) != 1:
        raise ValueError(
            "sandbox-exec plugin runtime bind is missing or ambiguous; "
            f"expected exactly one read-only bind to {expected}"
        )
    return matches[0]


def _deep_scan_set(
    allow_ro: list[tuple[str, str]],
    allow_rw: list[tuple[str, str]],
    scan_srcs: Iterable[str | Path] | None,
) -> set[str]:
    """Return the bind sources that get the curated-input deep scan.

    Explicit ``scan_srcs`` wins; otherwise the curated INPUT binds = workspace
    (always) + any allow_ro src that is NOT a known toolchain dir (i.e. KB +
    arch22 + extra inputs).  Toolchain binds (python/node/claude infra) are
    skipped — known-safe and huge.
    """
    if scan_srcs is not None:
        return {str(Path(s).resolve()) for s in scan_srcs}
    toolchain = set(default_toolchain_dirs())
    return {s for s, _ in allow_rw} | {s for s, _ in allow_ro if s not in toolchain}


def _bound_entries(
    allow_ro: list[tuple[str, str]],
    allow_rw: list[tuple[str, str]],
    scan_set: set[str],
    stage_root: Path,
) -> tuple[list[dict], int, int]:
    """Describe every bind and count the answer-bearing hits in scanned trees."""
    arch35_reachable = 0
    answer_cpp_reachable = 0
    bound: list[dict] = []
    for src, dst in list(allow_ro) + list(allow_rw):
        scanned = src in scan_set
        bound.append({
            "src": src, "dst": dst,
            "mode": "ro" if (src, dst) in allow_ro else "rw",
            "deep_scanned": scanned,
        })
        if not scanned or not Path(src).is_dir():
            continue
        # Only the exact immutable direct-launch source stage may contain its
        # original pybind11.cpp. Candidate/workspace trees retain the
        # answer-artifact gate, including arbitrary .port_source names.
        target_dirs, answer_files = _scan_curated_tree(
            Path(src), source_stage_root=stage_root
        )
        arch35_reachable += target_dirs
        answer_cpp_reachable += answer_files
    return bound, arch35_reachable, answer_cpp_reachable


def _network_enforcement(backend: str, share_net: bool) -> str:
    """Name the network rule the selected isolation backend actually applies."""
    if backend != "bwrap":
        return _macos_network_rule(share_net)
    return "shared-network-namespace" if share_net else "unshared-network-namespace"


def _staged_dependency_manifest_entry(runtime_root: Path) -> dict:
    """Summarize one staged runtime's dependency manifest for the audit record."""
    dependency_manifest = runtime_root / _PLUGIN_DEPENDENCY_MANIFEST
    if not dependency_manifest.is_file() or dependency_manifest.is_symlink():
        raise RuntimeError(
            "graybox staged plugin runtime is missing its dependency manifest"
        )
    try:
        dependency_payload = json.loads(
            dependency_manifest.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        # UnicodeError is a ValueError, so decoding failures are covered here.
        raise RuntimeError(
            "graybox staged plugin dependency manifest is unreadable"
        ) from error
    if not isinstance(dependency_payload, dict):
        raise RuntimeError(
            "graybox staged plugin dependency manifest is not an object"
        )
    return {
        "sha256": hashlib.sha256(dependency_manifest.read_bytes()).hexdigest(),
        "schema": dependency_payload.get("schema"),
        "declared": dependency_payload.get("declared", []),
        "records": dependency_payload.get("records", []),
        "unresolved": dependency_payload.get("unresolved", []),
        "staged_skill_names": dependency_payload.get("staged_skill_names", []),
        "staged_tree_sha256": dependency_payload.get("staged_tree_sha256"),
    }


def _staged_dependency_manifests(allow_ro: list[tuple[str, str]]) -> list[dict]:
    """Collect the dependency-manifest records of every staged plugin runtime."""
    entries: list[dict] = []
    for runtime_root in staged_plugin_runtime_roots(allow_ro):
        entries.append(_staged_dependency_manifest_entry(runtime_root))
    return entries


def write_construction_manifest(
    workspace: str | Path,
    allow_ro: list[tuple[str, str]],
    allow_rw: list[tuple[str, str]],
    *,
    inner_cmd: Sequence[str],
    scan_srcs: Iterable[str | Path] | None = None,
    source_stage_root: str | Path | None = None,
    share_net: bool = False,
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

    scan_set = _deep_scan_set(allow_ro, allow_rw, scan_srcs)
    stage_root = (
        Path(source_stage_root).resolve()
        if source_stage_root is not None
        else (ws / ".port_source")
    )
    bound, arch35_reachable, answer_cpp_reachable = _bound_entries(
        allow_ro, allow_rw, scan_set, stage_root
    )

    backend = isolation_backend()
    if backend is None:
        raise RuntimeError(
            "cannot certify construction without a supported strict isolation backend"
        )
    network_enforcement = _network_enforcement(backend, share_net)
    dependency_manifests = _staged_dependency_manifests(allow_ro)
    manifest = {
        "schema": "graybox_construction_manifest/v1",
        "mechanism": (
            "bwrap-mount-namespace"
            if backend == "bwrap"
            else "macos-sandbox-exec-deny-default"
        ),
        "isolation_backend": backend,
        "backend_path": _BWRAP if backend == "bwrap" else _SANDBOX_EXEC,
        "network": "shared" if share_net else "denied",
        "network_enforcement": network_enforcement,
        "inner_cmd": list(inner_cmd),
        "plugin_runtime": {
            "dependency_manifests": dependency_manifests,
        },
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


def construction_manifest_sha256(path: str | Path) -> str:
    """Return the digest the dispatcher uses to seal a construction manifest."""
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError(f"construction manifest is not a regular file: {candidate}")
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


def verify_construction_manifest(path: str | Path, expected_sha256: str) -> None:
    """Fail closed if a worker changed the pre-spawn construction proof."""
    actual = construction_manifest_sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            "graybox construction manifest changed during worker dispatch: "
            f"expected {expected_sha256}, observed {actual}"
        )
