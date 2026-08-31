#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""phase_o5_verify — Phase O5 verifier-execution cluster (SSH/local verifier
runners + canonical Pass A + independence/two-tier gates).

Behavior-neutral extraction from phase_o5_runner.py (DEBT-201 god-file
decomposition, 2026-07-06). Function bodies are byte-identical to their former
inline definitions EXCEPT that calls to MONKEYPATCHED phase_o5_runner functions
are QUALIFIED as `phase_o5_runner.<name>(...)` (they were bare names before).
This is the sanctioned patch-preservation step: a bare re-imported name would
resolve in THIS module's namespace, so `monkeypatch.setattr(phase_o5_runner,
"<name>", ...)` would not bite it; qualifying through the parent module object
makes the existing `phase_o5_runner`-targeted patches keep biting (the parent
re-exports every moved name, so the qualified lookup and the patch target are
the same attribute). The qualified calls are: `_is_port_a3_mode`,
`_verify_runner_independence`, `_run_canonical_pass_a_local` (all patched).

phase_o5_runner re-imports the names defined here (its bottom shim) so
`phase_o5_runner.<name>` / `por.<name>` / `o5.<name>` attribute access + `from
phase_o5_runner import <name>` stay valid for callers (orchestrator / tests).

Cycle-safety: this module top-imports the non-patched build/config helpers +
_PROJECT_ROOT from phase_o5_runner (all defined at parent top, before parent's
bottom import of this module) and `import phase_o5_runner` for the qualified
patched calls (resolved at CALL time). No import-time cycle.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import phase_o5_runner  # qualified access to monkeypatched fns (patch-bite preservation)
from phase_o5 import MeasuredResult
from phase_o5_runner import (  # non-patched build/config helpers + project root
    _a5_build_host, _a5_build_container, _a5_build_cann_path,
    _PROJECT_ROOT,
)
from phase_o5_helpers import (
    _find_verifier, _lane_aware_benchmark_root, _normalize_canonical_pass_a,
    _normalize_port_a3_two_tier_pass_a,
    _normalize_verifier_output,
    _resolve_extra_ld, _resolve_extra_pythonpath, _resolve_npu_python_bin,
    _resolve_ssh_key_opts,
    _shell_quote, _docker_sudo_enabled, _maybe_sudo_wrap_remote,
    _try_fetch_remote_result_json, _try_parse_json_tail,
)


@dataclass(frozen=True)
class _RemoteTarget:
    """Resolved target settings shared by remote verifier invocations."""

    name: str
    host: str
    user: str
    password: str
    container: str
    cann_path: str


def _is_lower_sha256(value: object) -> bool:
    """Return whether ``value`` is a canonical SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _regular_file_sha256(path: Path, label: str) -> str:
    """Hash a local regular file without following a replacement symlink.

    The controller must never unpickle a target-produced candidate capture:
    target NPU tensor storage may be unavailable locally, and finalization only
    needs an exact byte binding.  A no-follow descriptor makes that binding
    robust against a local symlink replacement while the transfer is in flight.
    """
    try:
        initial_mode = os.lstat(path).st_mode
    except OSError as error:
        raise OSError(f"{label} is missing or unreadable: {error}") from error
    if not stat.S_ISREG(initial_mode):
        raise OSError(f"{label} must be a regular non-symlink file")
    # Read-only descriptor: ``os.open`` takes no meaningful permission ``mode``
    # here because a mode only applies when the open creates the file
    # (O_CREAT/O_TMPFILE).  O_NOFOLLOW and O_CLOEXEC are security properties of
    # this binding and must be kept.
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise OSError(f"cannot open {label} without following links: {error}") from error
    # The descriptor is owned here, so close it on every path including the
    # regular-file rejection and any read error.
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"{label} must be a regular non-symlink file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _reject_nonregular_local_destination(path: Path, label: str) -> None:
    """Reject a stale link/special file before atomically replacing it."""
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISREG(mode):
        raise OSError(f"{label} must be a regular non-symlink file when present")


def _scp_command(
    target: _RemoteTarget, ssh_opts: list[str], remote_path: str, local_path: Path,
) -> list[str]:
    """Build the password- or key-auth SCP command used for an O5 receipt."""
    remote_spec = f"{target.user}@{target.host}:{remote_path}"
    if target.password:
        return [
            "sshpass", "-p", target.password, "scp", *ssh_opts,
            remote_spec, str(local_path),
        ]
    return ["scp", *ssh_opts, remote_spec, str(local_path)]


def _normalized_target(env: dict) -> str:
    """Return the target name used by target-scoped O5 configuration keys."""
    target_raw = (env.get("TARGET") or "a5").lower()
    if target_raw.endswith("-ds"):
        target_raw = target_raw[:-3]
    return target_raw.upper()


def _resolve_remote_target(
    workspace: Path,
    env: dict,
    *,
    target: str | None = None,
    cann_path: str | None = None,
) -> _RemoteTarget:
    """Resolve the target-specific host, container, and CANN settings."""
    target = target or _normalized_target(env)
    return _RemoteTarget(
        name=target,
        host=_a5_build_host(env, workspace, target),
        user=env.get(f"{target}_USER") or env.get("A5_USER", "root"),
        password=env.get(f"{target}_PASSWORD") or env.get("A5_PASSWORD", ""),
        container=_a5_build_container(env, workspace, target),
        cann_path=cann_path or _a5_build_cann_path(env, workspace, target),
    )


def _ssh_options(env: dict, target: str) -> list[str]:
    """Return the common SSH options used by verifier and canonical runs."""
    return [
        *_resolve_ssh_key_opts(env, target),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=60",
        "-o", "LogLevel=ERROR",
    ]


def _ssh_command(target: _RemoteTarget, ssh_opts: list[str], remote_cmd: str) -> list[str]:
    """Build the password- or key-auth SSH command for a remote O5 action."""
    login = f"{target.user}@{target.host}"
    if target.password:
        return ["sshpass", "-p", target.password, "ssh", *ssh_opts, login, remote_cmd]
    return ["ssh", *ssh_opts, login, remote_cmd]


def _resolve_visible_device(env: dict, workspace: Path, lane: int) -> int:
    """Resolve the physical NPU for ASCEND_RT_VISIBLE_DEVICES, health-aware.

    Keeps `lane` for BENCHMARK_ROOT_lane{N} (the .so deploy location) while
    routing execution around a wedged physical device when lane health can be
    queried. Fail-open: return `lane` when the host is unknown or no healthy
    alternative is available.
    """
    target = _normalized_target(env)
    override = str(env.get(f"{target}_O5_VISIBLE_DEVICE") or env.get("O5_VISIBLE_DEVICE") or "").strip()
    if override:
        try:
            return int(override)
        except ValueError:
            import logging
            logging.getLogger(__name__).warning(
                "[lane-health] ignoring invalid %s_O5_VISIBLE_DEVICE/O5_VISIBLE_DEVICE=%r",
                target, override,
            )
    a5_host = _a5_build_host(env, workspace, target)
    if not a5_host:
        return lane
    a5_user = env.get(f"{target}_USER") or env.get("A5_USER", "root")
    dev, reason = phase_o5_runner.resolve_healthy_device(lane, a5_host, a5_user)
    if dev != lane:
        import logging
        logging.getLogger(__name__).warning(
            "[lane-health] phase_o5 visible-device reroute (lane %s deploy kept): %s",
            lane, reason,
        )
    return dev


def _container_npu_python_setup(cann_path: str, npu_py_bin: str, extra_ld: str,
                                extra_pythonpath: str = "") -> tuple[str, str]:
    """Build the container-side Python/runtime setup used by O5 verifiers.

    Multi-CANN A5 hosts need the compile/runtime toolkit first, then the
    PyTorch/torch_npu libraries, then any older compatibility runtime that
    supplies missing side libraries such as libacl_dvpp.so. Putting the extra
    runtime first can load an older libascendcl and fail kernels that require
    newer symbols.

    `extra_pythonpath` (default "" for back-compat) is prepended onto the CANN
    site-packages on the PYTHONPATH export — the symmetric analog of `extra_ld`
    on LD_LIBRARY_PATH. Empty -> the export is byte-identical to the historical
    form (no stray colon).
    """
    if not npu_py_bin:
        return "python3", ""

    python3 = _shell_quote(f"{npu_py_bin}/python3")
    python311 = _shell_quote(f"{npu_py_bin}/python3.11")
    cann_python = f"{cann_path}/python/site-packages"
    cann_ld_prefix = (
        f"{cann_path}/x86_64-linux/lib64:"
        f"{cann_path}/lib64:"
        f"/usr/local/Ascend/driver/lib64/driver:"
        f"/usr/local/Ascend/driver/lib64/common"
    )
    extra_ld_tail = f":{extra_ld}" if extra_ld else ""
    extra_pp_seg = f"{extra_pythonpath}:" if extra_pythonpath else ""
    setup = (
        f"PYBIN={python3}; "
        f"[ -x \"$PYBIN\" ] || PYBIN={python311}; "
        f"export PYROOT=$(dirname $(dirname \"$PYBIN\")); "
        f"export PYTHONPATH={extra_pp_seg}{cann_python}:${{PYTHONPATH:-}}; "
        f"for SP in $PYROOT/lib/python*/site-packages; do "
        f"export LD_LIBRARY_PATH={cann_ld_prefix}:"
        f"$SP/torch/lib:$SP/torch_npu/lib:$PYROOT/lib"
        f"{extra_ld_tail}:${{LD_LIBRARY_PATH:-}}; "
        f"done; "
    )
    return '"$PYBIN"', setup


def _local_plugin_pass_b_skip(workspace: Path) -> Optional[dict]:
    """Return the local plugin Pass B skip result when it is explicitly declared."""
    from plugins import detect_plugin as _detect_plugin_local

    active_plugin = _detect_plugin_local(workspace)
    if active_plugin is None or active_plugin.pass_b_required():
        return None

    verification_path = workspace / "verification.json"
    if not verification_path.is_file():
        return None
    try:
        verification = json.loads(verification_path.read_text())
        pass_b_status = (
            (verification.get("precision") or {}).get("pass_b", {}).get("status")
        )
        if pass_b_status == "N/A":
            return {
                "status": "SKIPPED",
                "reason": (
                    f"plugin '{active_plugin.name}' "
                    f"pass_b_required=False; "
                    f"verification.json.precision.pass_b.status=N/A "
                    f"— pass_b script not invoked (local-container path)"
                ),
            }
    except Exception as error:
        logging.getLogger(__name__).debug("Recoverable operation failed.", exc_info=error)
    return None


def _run_local_pass_b(workspace: Path, script_name: str, lane: int) -> dict | str:
    """Run and normalize the local Pass B verifier, preserving O5 failure text."""
    try:
        local_env = os.environ.copy()
        local_env["ASCEND_RT_VISIBLE_DEVICES"] = str(lane)
        result = subprocess.run(
            ["python3", str(workspace / script_name)],
            capture_output=True, text=True, timeout=600,
            cwd=str(workspace), env=local_env,
        )
        if result.returncode != 0:
            return (
                f"local Pass B verifier exited {result.returncode}: "
                f"{result.stderr[:300]}"
            )
        parsed = _try_parse_json_tail(result.stdout)
        if parsed:
            return _normalize_verifier_output(parsed, "pass_b_local")
        return {"raw_stdout": result.stdout[:500], "label": "pass_b_local"}
    except subprocess.TimeoutExpired:
        return "local Pass B verifier timed out after 600s"
    except Exception as error:
        return f"local Pass B verifier exception: {error!r}"


def _run_local_pass_a_fallback(
    workspace: Path, script_name: str, lane: int,
) -> dict | str | None:
    """Run the worker Pass A fallback when the canonical local path is unavailable.

    ``None`` means the worker did not return a usable fallback and the caller must
    retain the canonical result that triggered the fallback attempt.
    """
    cycle_error = getattr(phase_o5_runner, "_verify_runner_independence")(workspace, script_name)
    if cycle_error:
        return cycle_error
    try:
        local_env = os.environ.copy()
        local_env["ASCEND_RT_VISIBLE_DEVICES"] = str(lane)
        result = subprocess.run(
            ["python3", str(workspace / script_name)],
            capture_output=True, text=True, timeout=600,
            cwd=str(workspace), env=local_env,
        )
        if result.returncode == 0:
            parsed = _try_parse_json_tail(result.stdout)
            if parsed:
                # Use ``pass_a`` so port_a3 receives the two-tier normalization.
                return _normalize_verifier_output(parsed, "pass_a")
    except Exception as error:
        logging.getLogger(__name__).debug("Recoverable operation failed.", exc_info=error)
    return None


def _local_measured_result(pass_a, pass_b, perf=None) -> MeasuredResult:
    """Build the local verifier result without changing its normalization rules."""
    return MeasuredResult(
        pass_a=pass_a if isinstance(pass_a, dict) else None,
        pass_b=pass_b if isinstance(pass_b, dict) else None,
        determinism=None,
        perf=perf if isinstance(perf, dict) else None,
    )


def _run_verifier_local(workspace: Path, op: str, env: dict, *, lane: int = 0) -> MeasuredResult:
    """P0aay (2026-05-11): Run O5 verifiers directly on local host.

    When CONTAINER=local (Ascend910B2C direct, no Docker), the verifier
    scripts are at their workspace paths on the local filesystem. Invoke
    them via subprocess without SSH/scp/docker.
    """
    # DEBT-172 (2026-06-24, top_k_top_p_sample kw-2): P96 plugin-skip for the
    # LOCAL-container path. Mirror of ssh_runner lines 244-281. Without this,
    # port_a3 ops on CONTAINER=local fail with "no Pass B verifier found" even
    # when verification.json correctly declares pass_b.status=N/A — per the
    # port_a3 D.6 kw_brief which explicitly forbids writing run_pass_b.py (it
    # would trip the P94 anti-cycle gate). The SSH path has this check; the
    # local path did not → harness gap, not worker error. CLAUDE.md "Fix Harness
    # for Next Customer" rule applies: this fix benefits every future port_a3
    # op on a local container, not just this one.
    plugin_skip_pass_b = _local_plugin_pass_b_skip(workspace)
    if plugin_skip_pass_b is not None:
        pass_b = plugin_skip_pass_b
    else:
        pass_b_script = _find_verifier(
            workspace, ["run_pass_b.py", "pass_b_runner.py", "run_edge_pass_b.py"]
        )
        if not pass_b_script:
            return MeasuredResult(
                runner_error=(
                    f"workspace/{op}: no Pass B verifier found locally "
                    f"(tried run_pass_b.py, pass_b_runner.py, run_edge_pass_b.py)"
                ),
            )
        cycle_error = getattr(phase_o5_runner, "_verify_runner_independence")(workspace, pass_b_script)
        if cycle_error:
            return MeasuredResult(runner_error=cycle_error)
        pass_b = _run_local_pass_b(workspace, pass_b_script, lane)
        if isinstance(pass_b, str):
            return MeasuredResult(runner_error=pass_b)
    # Canonical Pass A via precision_eval_two_tier.py — run locally
    # (P0aay extension: local mode for pass_a, fixing the gap where
    # _run_canonical_pass_a unconditionally used SSH+docker).
    pass_a = getattr(phase_o5_runner, "_run_canonical_pass_a_local")(workspace, op, env, lane=lane)
    if isinstance(pass_a, str) or pass_a is None:
        pass_a_script = _find_verifier(
            workspace, ["pass_a_runner.py", "edge_verify.py", "verify_edge.py"]
        )
        if pass_a_script:
            fallback_pass_a = _run_local_pass_a_fallback(workspace, pass_a_script, lane)
            if fallback_pass_a is not None:
                pass_a = fallback_pass_a

    return _local_measured_result(pass_a, pass_b, None)


def _run_local_canonical_script(
    workspace: Path,
    op: str,
    canonical_script: Path,
    lane: int,
    is_port_a3: bool,
) -> dict | str:
    """Run a local canonical evaluator and return its normalized JSON result."""
    json_path = f"/tmp/p2t_{op}_local.json"
    try:
        local_env = os.environ.copy()
        local_env["ASCEND_RT_VISIBLE_DEVICES"] = str(lane)
        result = subprocess.run(
            [sys.executable, str(canonical_script), str(workspace), "--json", json_path, "--quiet"],
            capture_output=True, text=True, timeout=900,
            cwd=str(workspace), env=local_env,
        )
        try:
            json_data = json.loads(Path(json_path).read_text())
        except Exception as error:
            logging.getLogger(__name__).debug("Recoverable operation failed.", exc_info=error)
            json_data = None
        try:
            Path(json_path).unlink()
        except Exception as error:
            logging.getLogger(__name__).debug("Recoverable operation failed.", exc_info=error)

        if json_data is None:
            return (
                f"canonical pass_a (local): no JSON output at {json_path} "
                f"(rc={result.returncode}); stderr_tail={result.stderr[-200:]!r}"
            )
        if is_port_a3:
            return _normalize_port_a3_two_tier_pass_a(json_data)
        return _normalize_canonical_pass_a(json_data)
    except subprocess.TimeoutExpired:
        return "canonical pass_a (local): timeout after 900s"
    except FileNotFoundError as error:
        return f"canonical pass_a (local): tool missing ({error})"
    except Exception as error:
        return f"canonical pass_a (local): exception: {error!r}"


def _run_canonical_pass_a_local(
    workspace: Path, op: str, env: dict, *, lane: int = 0,
) -> dict | str | None:
    """P0aay extension (2026-05-13): run canonical precision_eval_two_tier.py
    locally when CONTAINER=local. Mirrors the SSH path in
    _run_canonical_pass_a but without SSH/docker scaffolding.

    Returns dict on success, str on error, None if canonical not applicable.
    """
    # ① HARNESS-AUTHORITATIVE port_a3 pass_a (owner-directed 2026-06-30) — local-container path,
    # mirrors the SSH path. SUPERSEDES the prior "skip → worker pass_a_runner.py": the HARNESS grades
    # the worker-EMITTED tensors with compare.py (生态) via precision_eval_port_a3_two_tier.
    # load_and_classify (AUTHORITATIVE). This local path reads .pt files only (no NPU) so it is fully
    # exercised by the WSL unit test; the SSH variant's exec is VALIDATED-BY-③.
    is_port_a3 = getattr(phase_o5_runner, "_is_port_a3_mode")(workspace)
    if not is_port_a3:
        # non-port_a3: a plugin may produce its native verdict elsewhere → skip generic file evaluator.
        from plugins import detect_plugin as _detect_plugin_ca_local
        _active_plugin_ca_local = _detect_plugin_ca_local(workspace)
        if _active_plugin_ca_local is not None:
            _skip_reason_ca_local = _active_plugin_ca_local.canonical_pass_a_skip_reason(workspace)
            if _skip_reason_ca_local:
                import logging as _logging_ca_local
                _logging_ca_local.getLogger(__name__).info(
                    "[phase_o5] canonical pass_a (local) skipped per plugin "
                    "'%s': %s", _active_plugin_ca_local.name, _skip_reason_ca_local,
                )
                return _skip_reason_ca_local

    if not (workspace / "model.py").exists():
        return f"canonical pass_a (local): workspace/{op}/model.py missing"
    if not (workspace / "model_new_ascendc.py").exists():
        return f"canonical pass_a (local): workspace/{op}/model_new_ascendc.py missing"

    canonical_basename = (
        "precision_eval_port_a3_two_tier.py"
        if is_port_a3
        else "precision_eval_two_tier.py"
    )
    canonical_script = _PROJECT_ROOT / "src" / "scripts" / canonical_basename
    if not canonical_script.exists():
        return f"canonical pass_a (local): canonical script not found at {canonical_script}"
    return _run_local_canonical_script(
        workspace,
        op,
        canonical_script,
        lane,
        is_port_a3,
    )


def _gate_port_a3_two_tier(workspace: Path, pass_a) -> Optional[str]:
    """task#82 (2026-06-14): port_a3 two-tier ENGAGEMENT gate.

    In port_a3 mode the worker pass_a_runner.py is the native two-tier verdict
    producer (it runs ours on NPU, reads a3 from edge_dataset, runs cpu_truth on
    CPU, and delegates each case to precision_eval_port_a3_two_tier.
    classify_port_a3_case). Its summary JSON MUST therefore carry `tier2_status`
    (and tier2_pass). If pass_a is a dict (the runner ran + produced parseable
    output) but has NO tier2_status, the runner is single-tier — FAIL LOUD with a
    RUNNER_FAILED so phase_o5 routes back to the worker to fix the runner, instead
    of silently accepting a single-tier verdict as a passed two-tier verify.

    Returns None when:
      - not port_a3 mode (gate inert), OR
      - pass_a is None / an error string (a DIFFERENT failure path already
        surfaces it — don't double-report), OR
      - pass_a is a dict carrying tier2_status (engagement confirmed).
    Returns an error string only for the masquerade case (port_a3 + dict +
    missing tier2_status).
    """
    if not getattr(phase_o5_runner, "_is_port_a3_mode")(workspace):
        return None
    from reference_source import uses_live_a3_reference

    if not uses_live_a3_reference(workspace):
        return None
    if not isinstance(pass_a, dict):
        return None  # None or error string — surfaced elsewhere; not a masquerade
    if pass_a.get("tier2_status") is None:
        import logging as _logging
        pass_a_status = {}
        for key in ("tier1_pass", "total", "status"):
            pass_a_status[key] = pass_a.get(key)
        _logging.getLogger(__name__).error(
            "[phase_o5 task#82] port_a3 two-tier engagement gate TRIPPED: "
            "pass_a_runner emitted %s (no tier2_status). Single-tier runner "
            "masquerading as two-tier.",
            pass_a_status,
        )
        return (
            "port_a3 two-tier NOT engaged: pass_a_runner.py emitted a single-tier "
            "verdict (no tier2_status field). task#82 requires the native two-tier "
            "judge — pass_a_runner.py MUST import precision_eval_port_a3_two_tier "
            "and emit tier2_pass + tier2_status (per-case classify_port_a3_case: "
            "ours vs cpu_truth = T1, ours vs a3_outputs = T2). See kw_brief D.3. "
            "Refusing to accept a single-tier verdict as a passed two-tier verify."
        )
    import logging as _logging
    _logging.getLogger(__name__).info(
        "[phase_o5 task#82] port_a3 two-tier ENGAGED: tier1_pass=%s tier2_pass=%s "
        "tier2_status=%s total=%s",
        pass_a.get("tier1_pass"), pass_a.get("tier2_pass"),
        pass_a.get("tier2_status"), pass_a.get("total"),
    )
    return None


def _verify_runner_independence(workspace: Path, script_name: str) -> Optional[str]:
    """P94 attack-id WORKER-SELF-CITING-VERIFIER (2026-05-15T08:54Z):
    detect verifier scripts that read verification.json — the very file
    they're supposed to verify — creating a cycle where the "independent"
    measurement re-emits the worker's own claim.

    foreach_abs pass_a_runner.py docstring literally admitted:
    "This script does NOT execute on NPU. Instead, it reads the persisted
    pass_a block from verification.json which was populated from the
    source-identity argument."

    Returns None if script is independence-clean, error string if it
    cites verification.json in its body.
    """
    script_path = workspace / script_name
    if not script_path.is_file():
        return None  # no script to check; caller handles missing
    try:
        body = script_path.read_text(errors="ignore")
    except Exception:
        return None  # don't block on read error
    # Strip docstring/comments to focus on actual code references
    # Cheap heuristic: count non-comment lines with 'verification.json'
    # If ≥ 1 such reference, the verifier reads the claim file — cycle.
    cycle_lines = []
    in_docstring = False
    for lineno, raw in enumerate(body.splitlines(), 1):
        triple = raw.count('"""') + raw.count("'''")
        if triple >= 2:
            # Single-line docstring/string-literal — skip entirely (any
            # `verification.json` in it is documentation, not code).
            continue
        if triple == 1:
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        stripped = raw.strip()
        if stripped.startswith("#"):
            continue
        # Strip inline `#` comments before substring check
        code_part = stripped.split("#", 1)[0]
        if "verification.json" in code_part:
            cycle_lines.append(lineno)
    if cycle_lines:
        return (
            f"O5 verifier '{script_name}' contains {len(cycle_lines)} "
            f"code reference(s) to 'verification.json' (lines: "
            f"{cycle_lines[:5]}). This creates a CYCLE: the verifier "
            f"reads the same file whose claims it's supposed to "
            f"independently re-derive. P94 attack-id "
            f"WORKER-SELF-CITING-VERIFIER. Foreach_abs case: verifier "
            f"docstring admitted 'This script does NOT execute on NPU. "
            f"Instead, it reads the persisted pass_a block from "
            f"verification.json'. Verifier must be data-flow-independent "
            f"of the file it verifies."
        )
    return None


def _build_remote_verifier_command(
    workspace: Path,
    env: dict,
    target: _RemoteTarget,
    benchmark_root: str,
    script_name: str,
    lane: int,
    script_args: tuple[str, ...] = (),
) -> tuple[str, bool]:
    """Build the verifier command for either container or host-direct mode."""
    cann_setenv = f"source {target.cann_path}/set_env.sh"
    npu_python = _resolve_npu_python_bin(env, target.name)
    extra_ld = _resolve_extra_ld(env, target.name)
    extra_pythonpath = _resolve_extra_pythonpath(env, target.name)
    container_python, container_setup = _container_npu_python_setup(
        target.cann_path, npu_python, extra_ld, extra_pythonpath)
    visible_device = _resolve_visible_device(env, workspace, lane)
    argument_suffix = "".join(f" {_shell_quote(argument)}" for argument in script_args)
    docker_cmd = (
        f"cd {benchmark_root}/current_task && "
        f"export ASCEND_RT_VISIBLE_DEVICES={visible_device} && "
        f"{cann_setenv} && {container_setup}{container_python} {script_name}{argument_suffix}"
    )

    host_mode = str(env.get("A5_HOST_MODE", "")).strip() in ("1", "true", "yes")
    if not host_mode:
        remote_cmd = f"docker exec {target.container} bash -c {_shell_quote(docker_cmd)}"
        return _maybe_sudo_wrap_remote(remote_cmd, env, target.name), host_mode

    host_python = env.get("A5_HOST_PYTHON") or "/home/miniconda3/envs/msmodelslim-wr/bin/python"
    host_python_root = os.path.dirname(os.path.dirname(host_python))
    host_extra_ld = (
        f"{extra_ld + ':' if extra_ld else ''}"
        f"{host_python_root}/lib/python3.11/site-packages/torch/lib:"
        f"{host_python_root}/lib/python3.11/site-packages/torch_npu/lib:"
        f"{target.cann_path}/lib64:/usr/local/Ascend/driver/lib64/common:"
        f"/usr/local/Ascend/driver/lib64/driver"
    )
    host_cmd = (
        f"set +eu; set +o pipefail; "
        f"export LD_LIBRARY_PATH=\"${{LD_LIBRARY_PATH:-}}\" "
        f"PYTHONPATH=\"{extra_pythonpath + ':' if extra_pythonpath else ''}${{PYTHONPATH:-}}\"; "
        f"if [ -f {target.cann_path}/set_env.sh ]; then source {target.cann_path}/set_env.sh || true; fi; "
        f"export LD_LIBRARY_PATH={host_extra_ld}:${{LD_LIBRARY_PATH:-}}; "
        f"export PATH={os.path.dirname(host_python)}:$PATH; "
        f"export ASCEND_RT_VISIBLE_DEVICES={lane}; "
        f"cd {benchmark_root}/current_task && {host_python} {script_name}{argument_suffix}"
    )
    return host_cmd, host_mode


def _parse_remote_verifier_result(
    result: subprocess.CompletedProcess,
    target: _RemoteTarget,
    env: dict,
    benchmark_root: str,
    label: str,
    ssh_opts: list[str],
    host_mode: bool,
    raw: bool,
) -> dict | str:
    """Parse remote verifier stdout and retain the on-disk JSON fallback."""
    parsed = _try_parse_json_tail(result.stdout)
    if not parsed:
        fetch_kwargs = {"host_mode": host_mode}
        if result.returncode != 0:
            fetch_kwargs["sudo_wrap"] = _docker_sudo_enabled(env, target.name)
        parsed = _try_fetch_remote_result_json(
            target.host, target.user, target.password, target.container,
            benchmark_root, label, ssh_opts, **fetch_kwargs,
        )
    if parsed:
        return parsed if raw else _normalize_verifier_output(parsed, label)
    if result.returncode != 0:
        return (
            f"{label}: verifier exit {result.returncode}; "
            f"stderr={result.stderr[:200]!r}; stdout_tail={result.stdout[-200:]!r}"
        )
    return f"{label}: verifier stdout had no parseable JSON; tail={result.stdout[-300:]!r}"


def _run_verifier(
    workspace: Path,
    op: str,
    env: dict,
    script_name: str,
    label: str,
    *,
    lane: int = 0,
    raw: bool = False,
    script_args: tuple[str, ...] = (),
) -> dict | str:
    """Run a verifier script (run_pass_b.py / run_det_check.py / edge_verify.py)
    on A5 via SSH+docker exec. Captures stdout, parses last JSON object.

    Returns dict on success, error string on failure.

    raw=False (default): fold parsed output into the normalized
    {tier1_pass, total, status} precision shape (what pass_a/pass_b callers
    need). raw=True: return the parsed JSON verbatim — REQUIRED for perf
    (perf_runner.py emits {ratio, ratio_min, ...}; normalization would strip
    the ratio -> DEBT-192 port_a3 perf re-measure would false-N/A a measurable
    op). Only the transport (SSH+docker exec + last-JSON parse) is shared.
    """
    # P0vv (2026-05-06): target-aware host resolution. Reads TARGET and uses
    # <TARGET>_HOST / _USER / _PASSWORD / _CONTAINER. Falls back to A5_*
    # for legacy configs. P0aba also normalizes *-ds isolation targets.
    target = _normalized_target(env)
    cann_path = _a5_build_cann_path(env, workspace, target)
    remote_target = _resolve_remote_target(
        workspace, env, target=target, cann_path=cann_path)
    benchmark_root = _lane_aware_benchmark_root(env, lane)
    remote_cmd, host_mode = _build_remote_verifier_command(
        workspace, env, remote_target, benchmark_root, script_name, lane, script_args)
    ssh_opts = _ssh_options(env, remote_target.name)
    ssh_cmd = _ssh_command(remote_target, ssh_opts, remote_cmd)

    try:
        result = subprocess.run(
            ssh_cmd, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return f"{label}: SSH+verifier timeout after 600s"
    except FileNotFoundError as e:
        return f"{label}: tool missing ({e})"

    return _parse_remote_verifier_result(
        result, remote_target, env, benchmark_root, label, ssh_opts, host_mode, raw)


def _precision_standard_cli_suffix(is_pa3: bool) -> str:
    """Owner-directed run-control (controllable-harness #4): translate the
    orchestrator's AOG_PRECISION_STANDARD_CLI transport env var into a
    ` --precision-standard <v> ` argv fragment for the port_a3 grader.

    The orchestrator CLI (orchestrator_cli.main) sets AOG_PRECISION_STANDARD_CLI
    (a DISTINCT name from PRECISION_STANDARD, on purpose — see the transport note
    there) after normalizing/validating the value. Here we append it ONLY for the
    port_a3 canonical grader (precision_eval_port_a3_two_tier.py, which accepts
    `--precision-standard`); the benchmark grader (precision_eval_two_tier.py) is
    生态-only and does NOT accept the flag, so the suffix is empty when not pa3.

    Returns a fragment with a trailing space (so it drops cleanly between
    `--quiet ` and the `>/dev/null` tail), or "" when not applicable.
    """
    _ps = os.environ.get("AOG_PRECISION_STANDARD_CLI")
    if _ps and is_pa3:
        return f"--precision-standard {_ps} "
    return ""


def _canonical_pass_a_config(workspace: Path) -> tuple[bool, str, str] | str:
    """Choose the canonical grader or return a plugin-declared skip reason."""
    is_port_a3 = getattr(phase_o5_runner, "_is_port_a3_mode")(workspace)
    if is_port_a3:
        logging.getLogger(__name__).info(
            "[phase_o5 ①] port_a3 canonical pass_a = HARNESS load_and_classify (compare.py 生态), "
            "AUTHORITATIVE — supersedes worker pass_a_runner.py self-verdict")
        return is_port_a3, "precision_eval_port_a3_two_tier.py", _precision_standard_cli_suffix(True)

    from plugins import detect_plugin as _detect_plugin

    active_plugin = _detect_plugin(workspace)
    if active_plugin is not None:
        skip_reason = active_plugin.canonical_pass_a_skip_reason(workspace)
        if skip_reason:
            logging.getLogger(__name__).info(
                f"[phase_o5 task#82] canonical pass_a skipped per plugin "
                f"'{active_plugin.name}': {skip_reason}")
            return skip_reason
    return False, "precision_eval_two_tier.py", ""


def _build_remote_canonical_command(
    workspace: Path,
    env: dict,
    target: _RemoteTarget,
    benchmark_root: str,
    op: str,
    script_name: str,
    precision_suffix: str,
    lane: int,
) -> tuple[str, bool]:
    """Build the remote canonical Pass A command for container or host mode."""
    json_path = f"/tmp/p2t_{op}.json"
    npu_python = _resolve_npu_python_bin(env, target.name)
    extra_ld = _resolve_extra_ld(env, target.name)
    extra_pythonpath = _resolve_extra_pythonpath(env, target.name)
    container_python, container_setup = _container_npu_python_setup(
        target.cann_path, npu_python, extra_ld, extra_pythonpath)
    visible_device = _resolve_visible_device(env, workspace, lane)
    docker_cmd = (
        f"cd {benchmark_root}/current_task && "
        f"export ASCEND_RT_VISIBLE_DEVICES={visible_device} && "
        f"source {target.cann_path}/set_env.sh; {container_setup}"
        f"{container_python} {script_name} . --json {json_path} --quiet {precision_suffix}"
        f">/dev/null 2>&1; cat {json_path} 2>/dev/null; rm -f {json_path}"
    )

    host_mode = str(env.get("A5_HOST_MODE", "")).strip() in ("1", "true", "yes")
    if not host_mode:
        remote_cmd = f"docker exec {target.container} bash -c {_shell_quote(docker_cmd)}"
        return _maybe_sudo_wrap_remote(remote_cmd, env, target.name), host_mode

    host_python = env.get("A5_HOST_PYTHON") or "/home/miniconda3/envs/msmodelslim-wr/bin/python"
    host_python_root = os.path.dirname(os.path.dirname(host_python))
    host_extra_ld = (
        f"{extra_ld + ':' if extra_ld else ''}"
        f"{host_python_root}/lib/python3.11/site-packages/torch/lib:"
        f"{host_python_root}/lib/python3.11/site-packages/torch_npu/lib:"
        f"{target.cann_path}/lib64:/usr/local/Ascend/driver/lib64/common:"
        f"/usr/local/Ascend/driver/lib64/driver"
    )
    host_cmd = (
        f"set +eu; set +o pipefail; "
        f"export LD_LIBRARY_PATH=\"${{LD_LIBRARY_PATH:-}}\" "
        f"PYTHONPATH=\"{extra_pythonpath + ':' if extra_pythonpath else ''}${{PYTHONPATH:-}}\"; "
        f"if [ -f {target.cann_path}/set_env.sh ]; then source {target.cann_path}/set_env.sh || true; fi; "
        f"export LD_LIBRARY_PATH={host_extra_ld}:${{LD_LIBRARY_PATH:-}}; "
        f"export PATH={os.path.dirname(host_python)}:$PATH; "
        f"export ASCEND_RT_VISIBLE_DEVICES={lane}; cd {benchmark_root}/current_task && "
        f"{host_python} {script_name} . --json {json_path} --quiet {precision_suffix}>/dev/null 2>&1; "
        f"cat {json_path} 2>/dev/null; rm -f {json_path}"
    )
    return host_cmd, host_mode


def _normalize_remote_canonical_result(
    result: subprocess.CompletedProcess,
    is_port_a3: bool,
) -> dict | str:
    """Return the canonical Pass A result while retaining its failure text."""
    parsed = _try_parse_json_tail(result.stdout)
    if not parsed:
        return (
            f"canonical pass_a: no JSON in stdout (rc={result.returncode}); "
            f"stderr_tail={result.stderr[-200:]!r}"
        )
    if is_port_a3:
        return _normalize_port_a3_two_tier_pass_a(parsed)
    return _normalize_canonical_pass_a(parsed)


def _run_canonical_pass_a(
    workspace: Path, op: str, env: dict,
    *,
    lane: int = 0,
) -> dict | str | None:
    """P0abh (2026-05-08): authoritative Pass A measurement via canonical
    precision_eval_two_tier.py. Vendor 昇腾算子精度标准 2.1 §4.5.3 small-value
    rule baked into classify_output() — no agent compliance dependency.

    Returns:
        dict: MeasuredResult.pass_a-shape {tier1_pass, total, status, ...}
              with §4.5.3 promotions reflected in tier1_pass count.
        str:  error message (caller falls back to worker pass_a_runner.py).
        None: canonical not applicable (e.g. T3-axis op — caller falls back).
    """
    canonical_config = _canonical_pass_a_config(workspace)
    if isinstance(canonical_config, str):
        return canonical_config
    is_port_a3, canonical_script_name, precision_suffix = canonical_config
    if not (workspace / "model.py").exists():
        return f"canonical pass_a skipped: workspace/{op}/model.py missing"
    if not (workspace / "model_new_ascendc.py").exists():
        return f"canonical pass_a skipped: workspace/{op}/model_new_ascendc.py missing"

    target = _normalized_target(env)
    cann_path = _a5_build_cann_path(env, workspace, target)
    remote_target = _resolve_remote_target(
        workspace, env, target=target, cann_path=cann_path)
    if not remote_target.host:
        return f"canonical pass_a: missing {remote_target.name}_HOST"
    benchmark_root = _lane_aware_benchmark_root(env, lane)
    remote_cmd, _ = _build_remote_canonical_command(
        workspace, env, remote_target, benchmark_root, op, canonical_script_name,
        precision_suffix, lane)
    ssh_opts = _ssh_options(env, remote_target.name)
    ssh_cmd = _ssh_command(remote_target, ssh_opts, remote_cmd)

    try:
        result = subprocess.run(
            ssh_cmd, capture_output=True, text=True, timeout=900,
        )
    except subprocess.TimeoutExpired:
        return "canonical pass_a: SSH+canonical timeout after 900s"
    except FileNotFoundError as e:
        return f"canonical pass_a: tool missing ({e})"

    normalized = _normalize_remote_canonical_result(
        result,
        is_port_a3,
    )
    return normalized
