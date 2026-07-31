# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Phase O1.7 — LLM-driven op classification (P0aaj/P0aak, 2026-05-07).

Replaces the bench-name-keyed `op_taxonomy.OP_TAGS` map (P0aai regression).
Per user direction: classification should be done by an LLM agent reading the
source, not a hardcoded Python dict + regex catalog. This module wraps the
`claude --print --skill aog-op-classify` headless call, parses the resulting
JSON, returns a structured dataclass for brief construction.

Architecture:
- Phase O1.7 runs after O1.5 (DET_POLICY classification) and before O2 (benchmark sync)
- Per-op cost: ~$0.05 + ~30s, cached after via source content SHA256
- Failure mode graceful: subprocess error → falls back to None, brief uses default-only KB
- No state-machine integration (classification is a Python pipeline step, not a state)

Subprocess layer is testable via monkeypatch — unit tests inject fake `claude` exec.
"""
from __future__ import annotations
import logging

import hashlib
import json
import os
import subprocess  # noqa: F401 (retained for test monkeypatch compat + TimeoutExpired symbol)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from backends import get_backend

# Funnel (backend-decoupling step 3): the claude skill invocation goes through the pluggable
# Backend, not a hardcoded `claude --print`. phase_o17's divergent shape (acceptEdits + plain-text
# output + prompt-via-stdin) is expressed via dispatch's permission_mode/output_format/stdin_prompt.
_backend = get_backend()


def _strip_backend_suffix(name: str) -> str:
    """Compatibility name retained; scoped workspaces use flat names."""
    return name


@dataclass
class OpClassification:
    op: str
    op_class_tags: list[str]
    kb_recommendations: list[dict]  # [{"path": str, "reason": str}, ...]
    rationale: str
    source_signatures_observed: list[str]
    source_sha256: str
    schema_version: int = 2
    error: Optional[str] = None  # populated on subprocess failure / parse error


_SKILL_NAME = "aog-op-classify"
_CACHE_FILENAME = "op_classification.json"
_ERROR_FILENAME = "op_classification.error"
# P135.O17 (2026-05-18): sentinel written when claude CLI reports the skill
# is missing in this checkout. Caches the "skill unavailable" outcome so
# subsequent orchestrator restarts (e.g. --resume after pause) skip the
# 20-30s claude subprocess + $0.05-0.5 cost instead of re-discovering the
# same env limitation. Cleared by the next source-content change (via the
# same sha256 path the normal cache uses).
_SKILL_UNAVAILABLE_SENTINEL = "op_classification.skill_unavailable"


def _compute_source_sha256(workspace: Path) -> str:
    """SHA256 of concatenated source files (in the order SKILL.md §Step 1 lists)."""
    parts: list[bytes] = []
    candidates = [
        workspace / "model.py",
        workspace / "model_new_ascendc.py",
        workspace / "analysis.md",
    ]
    for sub in ("kernel", "."):
        d = workspace / sub
        if not d.exists() or not d.is_dir():
            continue
        for ext in ("*.h", "*.cpp", "*.py"):
            candidates.extend(sorted(d.glob(ext)))
    seen: set[Path] = set()
    for p in candidates:
        if p in seen or not p.exists() or not p.is_file():
            continue
        seen.add(p)
        skip_current_item = False
        try:
            parts.append(p.read_bytes()[:200_000])
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            skip_current_item = True
        if skip_current_item:
            continue
    return hashlib.sha256(b"\n".join(parts)).hexdigest()


def _read_cached(workspace: Path) -> Optional[OpClassification]:
    """Return cached classification if `op_classification.json` exists AND its
    `source_sha256` matches current source content.

    W15-followup (2026-05-12): when classification.json has source=cli_flag_port_a3
    (written by `_cmd_port_a3` from the orchestrator CLI flag), skip the SHA256
    check — the CLI flag IS the authoritative classification, no source-scan
    needed. Without this, phase_o17 re-runs the slow LLM classify skill even
    though _cmd_port_a3 already wrote the right answer.
    """
    cache = workspace / _CACHE_FILENAME
    if not cache.exists():
        return None
    try:
        data = json.loads(cache.read_text())
    except json.JSONDecodeError:
        return None
    # CLI-flag-authoritative shortcut
    if data.get("source") == "cli_flag_port_a3":
        return OpClassification(
            op=data.get("op", _strip_backend_suffix(workspace.name)),
            op_class_tags=data.get("op_class_tags", []),
            kb_recommendations=data.get("kb_recommendations", []),
            rationale=data.get("rationale", "CLI flag --port-a3 classification"),
            source_signatures_observed=data.get("source_signatures_observed", []),
            source_sha256=data.get("source_sha256", "cli_flag_port_a3"),
            schema_version=data.get("schema_version", 2),
        )
    cached_sha = data.get("source_sha256", "")
    current_sha = _compute_source_sha256(workspace)
    if cached_sha != current_sha:
        return None
    return OpClassification(
        op=data.get("op", _strip_backend_suffix(workspace.name)),
        op_class_tags=data.get("op_class_tags", []),
        kb_recommendations=data.get("kb_recommendations", []),
        rationale=data.get("rationale", ""),
        source_signatures_observed=data.get("source_signatures_observed", []),
        source_sha256=cached_sha,
        schema_version=data.get("schema_version", 2),
    )


def _write_unclassified_fallback(workspace: Path, cur_sha: str) -> OpClassification:
    """Deterministic sha-matched fallback for the O1.7 "exit-0-without-file" glitch.

    scan 2026-07-24 (main-assigned harness-gap #4; OL-282 vehicle-id family): when
    `aog-op-classify` exits 0 but writes no op_classification.json — because there is
    no benchmark source in scope to classify (for example an `--optimize` re-entry op
    lacking a <op>.py/<op>.json; surfaced on selective_scan_fwd_simd re-optimize) — the
    classifier narrates / asks a clarifying question instead of writing. The old code
    returned a bare error WITHOUT caching, so (a) every orchestrator resume re-invoked
    the flaky 120s LLM subprocess, and (b) downstream saw no file (mode/routing signal
    lost). This writes a deterministic sha-matched fallback (empty tags → brief uses
    default KB sections, the documented graceful mode; see module docstring §Failure
    mode) so the file always exists post-O1.7, re-runs on the SAME source read it cached
    (source_sha256 match → no re-invoke), and a real source change re-triggers classify.
    """
    rationale = (
        "O1.7 fallback: aog-op-classify exited 0 without writing "
        "op_classification.json (no benchmark source in scope to classify, e.g. "
        "an --optimize re-entry). Using default KB sections; a source "
        "change re-triggers classification."
    )
    try:
        (workspace / _CACHE_FILENAME).write_text(
            json.dumps(
                {
                    "op": _strip_backend_suffix(workspace.name),
                    "op_class_tags": [],
                    "kb_recommendations": [],
                    "rationale": rationale,
                    "source_signatures_observed": [],
                    "source_sha256": cur_sha,
                    "schema_version": 2,
                    "source": "o17_fallback_unclassified",
                },
                indent=2,
            )
        )
    except OSError:
        pass
    return OpClassification(
        op=_strip_backend_suffix(workspace.name),
        op_class_tags=[],
        kb_recommendations=[],
        rationale=rationale,
        source_signatures_observed=[],
        source_sha256=cur_sha,
    )


def _invoke_claude_skill(workspace: Path, timeout: int = 300) -> tuple[bool, str, str]:
    """Run `/aog-op-classify <workspace>` via headless `claude --print`.

    Note: claude CLI does NOT support `--skill <name>` flag. Skills resolve via
    `/skill-name` syntax in the prompt body (per `claude --help` "Skills still
    resolve via /skill-name"). So we pipe `/<skill> <args>` as the prompt.

    Returns (ok, stdout, stderr). ok=False indicates subprocess error or non-zero exit.
    """
    prompt = f"/{_SKILL_NAME} {workspace}"
    # Funnel through the pluggable Backend. Divergent shape preserved verbatim: acceptEdits +
    # plain-text output (output_format=None → no --output-format json) + prompt via stdin.
    env = _backend.dispatch(_SKILL_NAME, prompt, kind="skill", timeout=timeout,
                            permission_mode="acceptEdits", output_format=None, stdin_prompt=True)
    raw = env.raw_envelope
    if raw.get("timed_out"):
        return False, "", f"timeout after {timeout}s"
    if raw.get("not_found"):
        return False, "", f"claude CLI not found: {raw.get('stderr')}"
    stdout = env.output_text or ""
    stderr = raw.get("stderr") or ""
    if env.is_error:  # returncode != 0
        return False, stdout, stderr or f"exit {raw.get('returncode')}"
    return True, stdout, stderr


def _try_existing_classification(
    workspace: Path, cur_sha: str, error_msg: str
) -> Optional[OpClassification]:
    """Helper: respect intrinsic op_class taxonomy across cache events.

    Per P135.O17c (2026-05-23 review): when the
    classify skill is unavailable (cached-sentinel) OR cached-error pins
    current sha, the cached short-circuit was returning empty op_class_tags.
    But `op_class` is the operator's INTRINSIC taxonomy — stable across
    kernel iterations of the SAME op. source_sha tracks regen-relevant
    changes, NOT taxonomy validity.

    This helper checks for an existing `op_classification.json` with
    non-empty tags BEFORE returning empty. If found, returns the stale-but-
    -valid classification with the cache-event message captured in `error`
    (transparency: downstream sees the fallback happened).

    Returns None if no usable classification exists (caller should return
    the original error OpClassification).
    """
    import json as _json
    cls_path = workspace / _CACHE_FILENAME
    if not cls_path.is_file():
        return None
    try:
        d = _json.loads(cls_path.read_text())
    except (OSError, _json.JSONDecodeError):
        return None
    tags = d.get("op_class_tags") or []
    if not tags:
        return None
    json_sha = (d.get("source_sha256") or "")[:8]
    return OpClassification(
        op=_strip_backend_suffix(workspace.name),
        op_class_tags=list(tags),
        kb_recommendations=list(d.get("kb_recommendations") or []),
        rationale=d.get("rationale", "") or "",
        source_signatures_observed=list(d.get("source_signatures_observed") or []),
        source_sha256=d.get("source_sha256", "") or "",
        error=(
            f"{error_msg} — reused intrinsic taxonomy from prior valid "
            f"classification (json sha={json_sha}.., current sha="
            f"{cur_sha[:8]}..); op_class is stable across kernel iters of "
            f"the same op"
        ),
    )


def classify(workspace: Path, *, force: bool = False, timeout: int = 300) -> OpClassification:
    """Run Phase O1.7 classification on a workspace.

    Args:
        workspace: path to workspace dir
        force: if True, ignore cache and re-run skill
        timeout: subprocess timeout in seconds

    Returns:
        OpClassification (with error field populated on failure — never raises)
    """
    cur_sha = _compute_source_sha256(workspace)
    if not force:
        cached = _read_cached(workspace)
        if cached is not None:
            return cached
        # P135.O17 (2026-05-18): skill-unavailable sentinel — if a prior
        # invocation already determined this checkout's claude CLI doesn't
        # know /aog-op-classify, skip the subprocess. Sentinel is sha-pinned
        # so a source-content change still re-tries (in case the skill was
        # installed in between).
        sentinel = workspace / _SKILL_UNAVAILABLE_SENTINEL
        if sentinel.exists():
            try:
                pinned_sha = sentinel.read_text().strip()
            except OSError:
                pinned_sha = ""
            if pinned_sha == cur_sha:
                # P135.O17c (2026-05-23 review):
                # mirror the cached-error fallback — check for existing valid
                # op_classification.json BEFORE returning empty. Same intrinsic-
                # taxonomy reasoning as the cached-error path below.
                stale = _try_existing_classification(
                    workspace, cur_sha,
                    "skill unavailable (cached sentinel — /aog-op-classify not installed in this checkout)",
                )
                if stale is not None:
                    return stale
                return OpClassification(
                    op=_strip_backend_suffix(workspace.name),
                    op_class_tags=[], kb_recommendations=[], rationale="",
                    source_signatures_observed=[], source_sha256=cur_sha,
                    error="skill unavailable (cached sentinel; brief will use default-only KB)",
                )

    # P135.O17b (2026-05-18 DS): reap cached error from prior timeout.
    # If op_classification.error exists with matching sha256, the prior
    # run timed out — return the cached error immediately instead of
    # re-invoking claude --print (another 120s + $0.05-0.50).
    error_file = workspace / _ERROR_FILENAME
    if error_file.exists():
        try:
            cached_err = error_file.read_text().strip()
        except OSError:
            cached_err = ""
        # sha-pin the error cache: if source files haven't changed,
        # the skill would time out again — skip it.
        error_sha_file = workspace / (_ERROR_FILENAME + ".sha256")
        try:
            pinned = error_sha_file.read_text().strip() if error_sha_file.exists() else ""
        except OSError:
            pinned = ""
        if pinned == cur_sha:
            # P135.O17c (independent review 2026-05-23T19:25Z): when
            # cached error pins current sha, FALL BACK to existing
            # op_classification.json (stale-but-valid) before returning
            # empty tags. Op_class is the operator's INTRINSIC taxonomy
            # (FA op is FA regardless of kernel iter); source_sha tracks
            # regen-relevant changes, NOT taxonomy validity. Returning
            # empty tags here was breaking all op_class-aware paths (PR
            # #127 FA-class gates, structural-rewrite routing, DEBT-080
            # escalation, and the op-class-aware escalation-threshold and
            # reference-truth strategy selectors).
            #
            # Empirical: workspace/3_FusionAttention had valid FA tags in
            # op_classification.json from 2026-05-19 (sha f368fa85...),
            # current sha aeae25c7... after kernel iter, error.sha256
            # matched current → pre-fix returned tags=[] → is_fa_class()
            # → False → 0/7 gates fire on the FA op the gates were built
            # for.
            stale = _try_existing_classification(
                workspace, cur_sha,
                f"skill unavailable (cached error: {cached_err[:120]})",
            )
            if stale is not None:
                return stale
            # No stale classification available — return error-as-before
            return OpClassification(
                op=_strip_backend_suffix(workspace.name),
                op_class_tags=[], kb_recommendations=[], rationale="",
                source_signatures_observed=[], source_sha256=cur_sha,
                error=f"skill unavailable (cached error: {cached_err[:200]})",
            )

    ok, stdout, stderr = _invoke_claude_skill(workspace, timeout=timeout)
    if not ok:
        # DEBT-134 (2026-05-26): on a first-invocation skill failure/timeout,
        # prefer a prior valid op_classification.json (non-empty tags) over an
        # empty-tags error — op_class is the operator's INTRINSIC taxonomy,
        # stable across kernel iters of the same op (same reasoning as the
        # P135.O17c L3b stale-fallback, just fired on the FIRST timeout instead
        # of waiting for a second invocation to sha-pin the error). Still write
        # the error file + sha pin so subsequent invocations short-circuit via
        # the cheaper L3a cached-error path.
        stale = _try_existing_classification(
            workspace, cur_sha,
            f"skill subprocess failed (first invocation): {stderr[:120]}",
        )
        if stale is not None:
            try:
                error_file.write_text(f"skill subprocess failed: {stderr[:400]}")
                (workspace / (_ERROR_FILENAME + ".sha256")).write_text(cur_sha)
            except OSError:
                pass
            return stale
        err = OpClassification(
            op=_strip_backend_suffix(workspace.name),
            op_class_tags=[],
            kb_recommendations=[],
            rationale="",
            source_signatures_observed=[],
            source_sha256=cur_sha,
            error=f"skill subprocess failed: {stderr[:400]}",
        )
        try:
            error_file.write_text(err.error or "")
            # P135.O17b: pin sha256 so next resume skips the 120s timeout
            (workspace / (_ERROR_FILENAME + ".sha256")).write_text(cur_sha)
        except OSError:
            pass
        return err

    # Skill writes the JSON to workspace/op_classification.json directly; we just
    # re-read it. Sanity-check that the skill emitted the expected file.
    cached = _read_cached(workspace)
    if cached is not None:
        return cached
    # P135.O17 (2026-05-18): claude CLI exited 0 but the skill produced no
    # output file. The deterministic cause is "Unknown command: /<skill>" —
    # the skill is not installed/registered in this checkout. Cache the
    # outcome so the next orchestrator restart short-circuits.
    if "Unknown command" in stdout:
        try:
            (workspace / _SKILL_UNAVAILABLE_SENTINEL).write_text(cur_sha)
        except OSError:
            pass
        return OpClassification(
            op=_strip_backend_suffix(workspace.name),
            op_class_tags=[],
            kb_recommendations=[],
            rationale="",
            source_signatures_observed=[],
            source_sha256=cur_sha,
            error=(
                "skill exited 0 but op_classification.json not written or sha "
                f"mismatch; stdout_tail={stdout[-200:]!r}"
            ),
        )
    # scan 2026-07-24 (O1.7-glitch fix, main-assigned #4; OL-282 vehicle-id family):
    # non-"Unknown command" exit-0-without-file = the classifier narrated / asked a
    # clarifying question instead of writing (no benchmark source in scope to classify
    # — for example an --optimize re-entry). Write a deterministic sha-matched fallback
    # (see _write_unclassified_fallback) so re-runs on the same source skip the flaky
    # 120s re-invoke + downstream always has a file; a source change re-triggers classify.
    return _write_unclassified_fallback(workspace, cur_sha)
