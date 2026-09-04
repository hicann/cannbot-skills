# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""A.8 compile-only contract gate — orchestrator-side minimal piece.

Origin (2_FFN_evo close-out, 2026-08-30): a graybox worker cannot compile
its own candidate (no CANN inside the sandbox), so compile errors were
discovered only by a full O5 round.  This module is the controlled channel:

  1. The worker drops a compile REQUEST at ``<workspace>/.compile_requests/<id>.json``
     (schema ``aog.compile_request/v1``: op, request id, and the
     candidate-relative paths it wants diagnostics for).
  2. The orchestrator (outside the sandbox) validates the request — the
     path whitelist admits ONLY the candidate's own sources, never absolute
     paths, ``..`` escapes, symlinks, or evaluator-owned runtime files —
     then runs the SAME controlled-build compile step O5 uses
     (``npubench_target.build_*_on_target``); no second compile path exists.
  3. The candidate's compile diagnostic lines (filtered: only lines that
     reference the whitelisted candidate files — CANN headers and reference
     stage paths never leak back) are written to
     ``<workspace>/.compile_responses/<id>.json``.

Worker-side brief wiring is deliberately OUT of this phase's scope (v3.1
review): nothing invokes the gate from the FSM yet; the executor is driven
by the CLI below or by future wiring.

NOTE for the digest-scheme owner: ``.compile_requests`` / ``.compile_responses``
are runtime channels, not candidate source — they belong in
``npubench_core._CANDIDATE_RUNTIME_TOP_LEVEL`` alongside the other runtime
markers.

CLI:
  python3 -m npubench.npubench_compile_gate --workspace <ws> --lane <n>
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from npubench.npubench_core import _candidate_excluded  # noqa: E402

log = logging.getLogger(__name__)

COMPILE_REQUEST_SCHEMA = "aog.compile_request/v1"
COMPILE_RESPONSE_SCHEMA = "aog.compile_response/v1"
COMPILE_REQUEST_DIRNAME = ".compile_requests"
COMPILE_RESPONSE_DIRNAME = ".compile_responses"

_REQUEST_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_DIAGNOSTIC_LINE_CAP = 200
_DIAGNOSTIC_LINE_WIDTH = 500


class CompileGateError(RuntimeError):
    """A compile request failed validation or could not be executed."""


def request_dir(workspace: Path) -> Path:
    return Path(workspace) / COMPILE_REQUEST_DIRNAME


def response_dir(workspace: Path) -> Path:
    return Path(workspace) / COMPILE_RESPONSE_DIRNAME


def _validate_request_path(workspace: Path, request_path: Path) -> Path:
    workspace = Path(workspace).resolve()
    request_path = Path(request_path)
    if request_path.is_symlink() or not request_path.is_file():
        raise CompileGateError(f"compile request is not a regular file: {request_path}")
    resolved = request_path.resolve()
    if resolved.parent != request_dir(workspace).resolve():
        raise CompileGateError(
            f"compile request must live directly under {COMPILE_REQUEST_DIRNAME}/: {request_path}"
        )
    return resolved


def _validate_candidate_paths(workspace: Path, paths: Any) -> list[str]:
    """Whitelist: candidate-relative, existing, regular, evaluator-owned-excluded."""
    if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
        raise CompileGateError("compile request 'paths' must be a non-empty list of strings")
    workspace = Path(workspace).resolve()
    validated: list[str] = []
    for raw in paths:
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise CompileGateError(f"compile request path escapes the candidate: {raw!r}")
        resolved = (workspace / relative).resolve()
        if resolved.parent != workspace and workspace not in resolved.parents:
            raise CompileGateError(f"compile request path resolves outside the workspace: {raw!r}")
        candidate_file = workspace / relative
        if candidate_file.is_symlink() or not candidate_file.is_file():
            raise CompileGateError(f"compile request path is not an existing regular file: {raw!r}")
        if _candidate_excluded(relative):
            raise CompileGateError(
                f"compile request path is evaluator-owned, not candidate source: {raw!r}"
            )
        validated.append(relative.as_posix())
    return validated


def validate_compile_request(workspace: Path, request_path: Path) -> dict[str, Any]:
    """Validate one compile request file; return the normalized request dict."""
    resolved = _validate_request_path(workspace, request_path)
    try:
        request = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompileGateError(f"compile request is unreadable JSON: {exc}") from exc
    if not isinstance(request, dict):
        raise CompileGateError("compile request must be a JSON object")
    if request.get("schema") != COMPILE_REQUEST_SCHEMA:
        raise CompileGateError(
            f"compile request schema must be {COMPILE_REQUEST_SCHEMA!r}, "
            f"got {request.get('schema')!r}"
        )
    request_id = request.get("request_id")
    if not isinstance(request_id, str) or not _REQUEST_ID_RE.fullmatch(request_id):
        raise CompileGateError(f"invalid compile request id: {request_id!r}")
    if resolved.name != f"{request_id}.json":
        raise CompileGateError(
            f"compile request id {request_id!r} does not match filename {resolved.name!r}"
        )
    op = request.get("op")
    if not isinstance(op, str) or not op.strip():
        raise CompileGateError("compile request 'op' must be a non-empty string")
    paths = _validate_candidate_paths(workspace, request.get("paths"))
    return {
        "schema": COMPILE_REQUEST_SCHEMA,
        "request_id": request_id,
        "op": op.strip(),
        "paths": paths,
        "request_path": resolved,
    }


def _default_build_fn(workspace: Path) -> Callable[..., dict[str, Any]]:
    """Pick the SAME controlled-build entry point O5 uses for this workspace.

    No second compile path: the source-kind mapping mirrors
    ``npubench_o5_runner._CONTROLLED_TARGET_BUILDS`` and falls back to the
    generic authored kernel project build.
    """
    from npubench import npubench_target
    from npubench.npubench_candidate_contract import TILELANG2ASCENDC_SOURCE_KIND

    try:
        state = json.loads((Path(workspace) / ".opgen_state.json").read_text())
    except (OSError, json.JSONDecodeError):
        state = {}
    source = state.get("port_source") if isinstance(state, dict) else None
    if isinstance(source, dict) and source.get("kind") == TILELANG2ASCENDC_SOURCE_KIND:
        return npubench_target.build_tilelang2ascendc_candidate_on_target
    return npubench_target.build_generic_kernel_project_on_target


def _filter_candidate_diagnostics(text: str, candidate_paths: list[str]) -> list[str]:
    """Keep ONLY diagnostic lines that reference a whitelisted candidate file.

    Compiler output routinely inlines CANN header paths and reference-stage
    snippets; none of that may flow back into the workspace, so the filter is
    a positive match on the candidate's own relative paths (or basenames),
    never a denylist of external prefixes.
    """
    tokens = set(candidate_paths)
    tokens.update(Path(p).name for p in candidate_paths)
    kept: list[str] = []
    for line in (text or "").splitlines():
        if len(kept) >= _DIAGNOSTIC_LINE_CAP:
            break
        if any(token and token in line for token in tokens):
            kept.append(line[:_DIAGNOSTIC_LINE_WIDTH])
    return kept


def execute_compile_request(
    workspace: Path,
    lane: int,
    request_path: Path,
    *,
    build_fn: Optional[Callable[..., dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Run one validated compile request and write the filtered response.

    ``build_fn`` is injectable for tests; production uses the O5 controlled
    build for the workspace's source kind.
    """
    workspace = Path(workspace)
    request = validate_compile_request(workspace, request_path)
    build = build_fn or _default_build_fn(workspace)
    result = build(workspace, lane, build_attempt_id=secrets.token_hex(16))
    if not isinstance(result, dict):
        raise CompileGateError(
            f"controlled build returned a non-dict result: {type(result).__name__}"
        )
    diagnostics = _filter_candidate_diagnostics(
        "\n".join(
            str(result.get(key) or "")
            for key in ("reason", "stdout_tail", "stderr_tail")
        ),
        request["paths"],
    )
    response = {
        "schema": COMPILE_RESPONSE_SCHEMA,
        "request_id": request["request_id"],
        "op": request["op"],
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": result.get("status"),
        "returncode": result.get("returncode"),
        "failure_kind": result.get("failure_kind"),
        "paths": request["paths"],
        "diagnostics_filtered": True,
        "diagnostics": diagnostics,
    }
    out_dir = response_dir(workspace)
    out_dir.mkdir(parents=True, exist_ok=True)
    response_path = out_dir / f"{request['request_id']}.json"
    if response_path.is_symlink():
        raise CompileGateError(f"refusing to write over symlinked response: {response_path}")
    response_path.write_text(json.dumps(response, indent=1, sort_keys=True) + "\n")
    log.info(
        "compile gate %s: status=%s, %d diagnostic lines -> %s",
        request["request_id"], response["status"], len(diagnostics), response_path,
    )
    return response


def pending_compile_requests(workspace: Path) -> list[Path]:
    """Request files with no response newer than the request itself."""
    directory = request_dir(workspace)
    if not directory.is_dir():
        return []
    pending = []
    for path in sorted(directory.glob("*.json")):
        response = response_dir(workspace) / path.name
        if not response.is_file() or response.stat().st_mtime < path.stat().st_mtime:
            pending.append(path)
    return pending


def run_pending_compile_requests(
    workspace: Path,
    lane: int,
    *,
    build_fn: Optional[Callable[..., dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Execute every pending request; invalid ones fail closed per file."""
    responses = []
    for request_path in pending_compile_requests(workspace):
        try:
            responses.append(execute_compile_request(workspace, lane, request_path,
                                                     build_fn=build_fn))
        except CompileGateError as exc:
            log.warning("compile gate rejected %s: %s", request_path.name, exc)
            responses.append({
                "schema": COMPILE_RESPONSE_SCHEMA,
                "request_id": request_path.stem,
                "status": "REJECTED",
                "reason": str(exc),
            })
    return responses


def main() -> int:
    parser = argparse.ArgumentParser(description="A.8 compile-only contract gate executor")
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--lane", required=True, type=int)
    args = parser.parse_args()
    responses = run_pending_compile_requests(args.workspace, args.lane)
    # Emitted as a log record, not written to stdout: nothing consumes this as a
    # data channel. The only in-tree caller of the gate is
    # tests/ut/test_npubench_compile_gate.py, which invokes
    # run_pending_compile_requests() directly rather than running main().
    log.info("compile gate responses: %s", json.dumps(responses, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
