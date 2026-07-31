# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""KB merge invocation — wraps aog-knowledge-maintain Skill.

The Skill is an LLM-skill (semantic generalization quality requires LLM
judgment per `aog-knowledge-maintain/SKILL.md`). Orchestrator decides WHEN
to invoke; LLM inside the skill decides CONTENT.

After the orchestrator's independent finalize gates accept a non-trivial
workspace/<op>/knowledge_update.md, this module:
1. Runs `aog-knowledge-maintain` for semantic review into a workspace intake
2. Admits reviewed entries through Arbiter into user-local c-tier only
3. Drops `.kb_merged` only after deterministic persistence, then logs the run

The plugin-bundled b-tier is release-owned and read-only at runtime. A before /
after fingerprint makes accidental semantic-agent edits a visible failure.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sys as _sys
from pathlib import Path

# Harness-decoupling: the claude invocation is owned by the Backend, not hardcoded here.
_sys.path.insert(0, str(Path(__file__).resolve().parent))  # orchestrator/ for `backends` package
from backends import get_backend

_backend = get_backend()

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SCRIPTS_ROOT))

from kb_tiering.adapters.cannbot_c import make_cannbot_c, resolve_c_root  # noqa: E402
from kb_tiering.interface import Arbiter, Entry, hard_key  # noqa: E402
from workflow.kb_marker_verifier import verify_marker  # noqa: E402

_CANDIDATE_FILENAME = ".kb_c_tier_candidates.json"
_BUNDLED_KB_ROOT = Path(__file__).resolve().parents[4] / "kb"
_CANDIDATE_FIELDS = {
    "kind", "claim", "scope", "key", "evidence", "provenance", "meta",
}


def _bundled_kb_fingerprint() -> str:
    """Content fingerprint for the release-owned b-tier.

    The semantic reviewer may read this tree, but a runtime merge must never
    mutate it.  Comparing before/after dispatch turns an accidental edit into
    a visible merge failure instead of accepting a lying marker.
    """
    digest = hashlib.sha256()
    if not _BUNDLED_KB_ROOT.is_dir():
        return digest.hexdigest()
    for path in sorted(_BUNDLED_KB_ROOT.rglob("*")):
        if not path.is_file() and not path.is_symlink():
            continue
        rel = path.relative_to(_BUNDLED_KB_ROOT).as_posix().encode()
        digest.update(len(rel).to_bytes(4, "big"))
        digest.update(rel)
        if path.is_symlink():
            digest.update(b"L")
            digest.update(path.readlink().as_posix().encode())
        else:
            digest.update(b"F")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _candidate_path(workspace: Path) -> Path:
    return workspace / _CANDIDATE_FILENAME


def _build_c_tier_arbiter() -> tuple[Arbiter, Path]:
    """Build the canonical write path, with c as the only writable tier."""
    c_root = resolve_c_root()
    providers = []
    c_provider = make_cannbot_c(str(c_root), providers=providers)
    providers.append(c_provider)

    # Include bundled b for cross-tier tombstone/admission context when the
    # aggregate plugin is installed.  Its adapter rejects direct writes, and
    # this function always targets ``customer`` below.
    b_index = _BUNDLED_KB_ROOT / "KB_INDEX.md"
    if b_index.is_file():
        from kb_tiering.adapters.cannbot_b import make_cannbot_b
        providers.append(make_cannbot_b(str(_BUNDLED_KB_ROOT)))
    return Arbiter(providers), c_root


def _load_candidate_payload(workspace: Path) -> list[dict]:
    path = _candidate_path(workspace)
    if not path.is_file():
        raise ValueError(
            f"semantic review did not emit required c-tier intake: {path}"
        )
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid c-tier intake {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("c-tier intake must be an object with schema_version=1")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("c-tier intake entries must be a list")
    if len(entries) > 10:
        raise ValueError("c-tier intake exceeds the 10-entry per-invocation limit")
    for idx, raw in enumerate(entries):
        if not isinstance(raw, dict):
            raise ValueError(f"c-tier entry {idx} must be an object")
        unknown = set(raw) - _CANDIDATE_FIELDS
        if unknown:
            raise ValueError(
                f"c-tier entry {idx} has unsupported fields: {sorted(unknown)}"
            )
        if raw.get("kind") not in {"positive_pattern", "anti_pattern", "experience"}:
            raise ValueError(f"c-tier entry {idx} has invalid kind")
        if not isinstance(raw.get("claim"), str) or not raw["claim"].strip():
            raise ValueError(f"c-tier entry {idx} needs a non-empty claim")
        for field in ("scope", "evidence", "provenance", "meta"):
            if field in raw and not isinstance(raw[field], dict):
                raise ValueError(f"c-tier entry {idx}.{field} must be an object")
        if "key" in raw and not isinstance(raw["key"], str):
            raise ValueError(f"c-tier entry {idx}.key must be a string")
    return entries


def _entry_from_candidate(raw: dict, workspace: Path) -> Entry:
    claim = raw["claim"].strip()
    candidate_key = (raw.get("key") or "").strip()
    provenance = dict(raw.get("provenance") or {})
    provenance.update({
        "source": str(workspace / "knowledge_update.md"),
        "workspace": str(workspace),
        "ingested_by": "aog-knowledge-maintain",
    })
    return Entry(
        id="",
        tier="customer",
        role="user-local",
        kind=raw["kind"],
        scope=dict(raw.get("scope") or {}),
        key=candidate_key or hard_key(claim),
        claim=claim,
        evidence=dict(raw.get("evidence") or {}),
        # Runtime semantic review is not a release promotion authority.
        trust="unverified",
        provenance=provenance,
        meta=dict(raw.get("meta") or {}),
    )


def _persist_c_tier(workspace: Path, entries: list[dict], *, mode: str) -> dict:
    arbiter, c_root = _build_c_tier_arbiter()
    written: list[str] = []
    rejected: list[dict] = []
    for idx, raw in enumerate(entries):
        entry = _entry_from_candidate(raw, workspace)
        status, detail = arbiter.write(entry, "customer")
        if status == "WRITTEN":
            written.append(f"customer:{entry.content_hash}")
        else:
            rejected.append({"index": idx, "status": status, "reason": detail})

    # A reviewed empty/rejected intake is still a completed semantic decision.
    # The marker is written only here, after deterministic c-tier admission.
    marker = workspace / ".kb_merged"
    marker.write_text(
        f"merge_run={_dt.datetime.now(_dt.timezone.utc).isoformat().replace('+00:00', 'Z')}\n"
        "tier=customer\n"
        f"c_root={c_root.resolve()}\n"
        "merged_into=user-c-tier\n"
        f"entries={','.join(written) if written else 'none'}\n"
        f"reviewed={len(entries)}\n"
        f"rejected={len(rejected)}\n"
        f"mode={mode}\n"
    )
    return {
        "success": True,
        "marker_path": str(marker),
        "marker_exists_after": True,
        "c_root": str(c_root),
        "written": written,
        "rejected": rejected,
    }


def _runtime_prompt(workspace: Path) -> str:
    intake = _candidate_path(workspace)
    return (
        f"Run the aog-knowledge-maintain skill in Mode 1 for workspace `{workspace}`. "
        f"Read `{workspace}/knowledge_update.md` and perform the skill's semantic "
        f"generalization, evidence, scope, conflict, and dedup review. The bundled "
        f"plugin KB is release-owned b-tier and MUST remain byte-for-byte read-only. "
        f"Do not edit bundled KB markdown, KB_INDEX.md, promotion markers, or "
        f"`.kb_merged`; do not write directly under the user KB root. Emit only "
        f"`{intake}` as JSON: {{\"schema_version\":1,\"entries\":[...]}}. Each "
        f"entry may contain only kind, claim, scope, key, evidence, provenance, and "
        f"meta. kind is positive_pattern, anti_pattern, or experience. Use an empty "
        f"entries list when review decides nothing should be admitted. The "
        f"orchestrator will persist accepted entries through "
        f"Arbiter.write(..., \"customer\") to ASCENDC_PORT_USER_KB (or its default) "
        f"and will create the completion marker. Report the semantic decisions."
    )


def _log_merge(workspace: Path, log_entry: dict) -> None:
    with open(workspace / ".kb_merge_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def _quarantine_marker(workspace: Path) -> Path | None:
    """Move a non-canonical completion marker aside for later inspection."""
    marker = workspace / ".kb_merged"
    if not marker.exists() and not marker.is_symlink():
        return None
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    quarantined = workspace / f".kb_merged.invalid-{stamp}"
    marker.rename(quarantined)
    return quarantined


def _prepare_existing_marker(workspace: Path) -> bool:
    """Return True only for a marker bound to durable KB state.

    Unverified markers are quarantined before semantic review so legacy
    `entries=<anything>` text cannot bypass c-tier admission.
    """
    marker = workspace / ".kb_merged"
    if not marker.exists() and not marker.is_symlink():
        return False
    try:
        report = verify_marker(
            workspace,
            _BUNDLED_KB_ROOT.parent,
            expected_c_root=resolve_c_root().resolve(),
        )
    except (OSError, ValueError):
        report = None
    if report is not None and report.verdict == "OK":
        if report.tier == "customer":
            fields = report.marker_fields
            required = {
                "merge_run", "tier", "c_root", "merged_into", "entries",
                "reviewed", "rejected", "mode",
            }
            try:
                reviewed = int(fields.get("reviewed", ""))
                rejected = int(fields.get("rejected", ""))
            except ValueError:
                reviewed = rejected = -1
            listed = len(report.new_entry_ids)
            entries_shape_ok = (
                fields.get("entries") == "none"
                if listed == 0
                else "none" not in fields.get("entries", "").split(",")
            )
            if (
                required.issubset(fields)
                and report.merged_into == ["user-c-tier"]
                and not report.refinement_ids
                and fields.get("mode") in {"update", "batch"}
                and reviewed >= 0
                and 0 <= rejected <= reviewed
                and reviewed - rejected == listed
                and entries_shape_ok
            ):
                return True
        elif (
            report.merged_into
            and report.new_entry_ids
            and not report.refinement_ids
            and set(report.found) == set(report.new_entry_ids)
        ):
            return True
    _quarantine_marker(workspace)
    return False


def merge_one(workspace: Path, *, timeout_sec: int = 1200) -> dict:
    """Semantically review one update, then admit it to user c-tier.

    Returns:
        dict with keys: success, marker_path, log_entry
    """
    marker = workspace / ".kb_merged"
    if _prepare_existing_marker(workspace):
        return {"success": True, "skipped": f"already merged: {marker}"}

    ku_path = workspace / "knowledge_update.md"
    if not ku_path.exists() or ku_path.stat().st_size < 100:
        return {"success": True, "skipped": "no knowledge_update.md or < 100 bytes"}

    intake = _candidate_path(workspace)
    intake.unlink(missing_ok=True)
    prompt = _runtime_prompt(workspace)
    bundled_before = _bundled_kb_fingerprint()

    # bypassPermissions per Day 4 finding (P0f). Invocation via Backend (harness-decoupling):
    # was a hardcoded `claude --print` cmd → now CCBackend.dispatch(kind="skill"). Behavior faithful.
    env = _backend.dispatch("aog-knowledge-maintain", prompt, kind="skill", timeout=timeout_sec)
    if env.raw_envelope.get("timed_out"):
        bundled_changed = _bundled_kb_fingerprint() != bundled_before
        _quarantine_marker(workspace)
        return {
            "success": False, "timed_out": True,
            "stdout_tail": "", "stderr_tail": "(kb_invoke timed out)",
            "bundled_b_tier_changed": bundled_changed,
        }
    success = not env.is_error
    error = ""
    persisted: dict = {}
    if _bundled_kb_fingerprint() != bundled_before:
        success = False
        error = "aog-knowledge-maintain modified release-owned bundled b-tier"
    if success and _quarantine_marker(workspace) is not None:
        success = False
        error = "semantic reviewer wrote the orchestrator-owned completion marker"
    if success:
        try:
            persisted = _persist_c_tier(
                workspace, _load_candidate_payload(workspace), mode="update"
            )
        except (OSError, ValueError, KeyError) as exc:
            success = False
            error = str(exc)
    if not success:
        _quarantine_marker(workspace)

    log_entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "workspace": str(workspace),
        "exit_code": env.raw_envelope.get("returncode"),
        "success": success,
        "stdout_tail": (env.output_text or "")[-2000:],
        "stderr_tail": (env.raw_envelope.get("stderr") or "")[-1000:],
        "error": error,
        "c_tier_written": persisted.get("written", []),
        "c_tier_rejected": persisted.get("rejected", []),
    }
    _log_merge(workspace, log_entry)

    return {
        **persisted,
        "success": success,
        "marker_path": str(marker),
        "marker_exists_after": marker.exists(),
        "log_entry": log_entry,
        **({"error": error} if error else {}),
    }


def merge_batch(workspaces: list[Path], *, timeout_sec: int = 1800) -> dict:
    """Review a batch once, then admit each workspace's entries to c-tier.

    Used by parallel batch runs — single sequential merge after all per-op
    knowledge_updates are written.
    """
    pending: list[Path] = []
    for ws in workspaces:
        if _prepare_existing_marker(ws):
            continue
        update = ws / "knowledge_update.md"
        if not update.exists() or update.stat().st_size <= 100:
            continue
        pending.append(ws)
    if not pending:
        return {"success": True, "skipped": "no pending workspaces", "n_pending": 0}

    workspace_root = pending[0].parent
    for ws in pending:
        _candidate_path(ws).unlink(missing_ok=True)
    intake_contract = [str(_candidate_path(ws)) for ws in pending]
    prompt = (
        f"Run aog-knowledge-maintain in Mode 1-batch (--scan-roots {workspace_root}). "
        f"Semantically review all pending knowledge_update.md files and apply "
        f"cross-batch dedup. The bundled plugin KB is release-owned b-tier and "
        f"MUST remain byte-for-byte read-only. Do not edit bundled KB files, "
        f"KB_INDEX.md, promotion markers, `.kb_merged`, or the user KB directly. "
        f"For each pending workspace emit its exact intake path from "
        f"{intake_contract!r} using the Mode 1 schema "
        f"{{\"schema_version\":1,\"entries\":[...]}}; emit an empty list when "
        f"nothing is admissible. The orchestrator alone writes c-tier via "
        f"Arbiter.write(..., \"customer\") and creates markers. "
        f"Pending workspaces: {[ws.name for ws in pending]}"
    )

    # bypassPermissions (P0f). Invocation via Backend (harness-decoupling), behavior faithful.
    bundled_before = _bundled_kb_fingerprint()
    env = _backend.dispatch("aog-knowledge-maintain", prompt, kind="skill", timeout=timeout_sec)
    if env.raw_envelope.get("timed_out"):
        bundled_changed = _bundled_kb_fingerprint() != bundled_before
        for ws in pending:
            _quarantine_marker(ws)
        return {
            "success": False, "timed_out": True,
            "stdout_tail": "", "stderr_tail": "(kb_invoke timed out)",
            "bundled_b_tier_changed": bundled_changed,
        }
    if _bundled_kb_fingerprint() != bundled_before:
        for ws in pending:
            _quarantine_marker(ws)
        return {
            "success": False,
            "n_pending": len(pending),
            "error": "aog-knowledge-maintain modified release-owned bundled b-tier",
        }
    if env.is_error:
        for ws in pending:
            _quarantine_marker(ws)
        return {
            "success": False,
            "n_pending": len(pending),
            "exit_code": env.raw_envelope.get("returncode"),
            "stdout_tail": (env.output_text or "")[-2000:],
        }

    results: dict[str, dict] = {}
    try:
        reviewer_markers = [
            ws for ws in pending if _quarantine_marker(ws) is not None
        ]
        if reviewer_markers:
            raise ValueError(
                "semantic reviewer wrote orchestrator-owned completion markers: "
                + ", ".join(str(ws) for ws in reviewer_markers)
            )
        # Validate every intake before the first durable c-tier write so a
        # malformed lane cannot produce a misleading partially-complete batch.
        payloads = {ws: _load_candidate_payload(ws) for ws in pending}
        for ws, entries in payloads.items():
            results[str(ws)] = _persist_c_tier(ws, entries, mode="batch")
    except (OSError, ValueError, KeyError) as exc:
        for ws in pending:
            _quarantine_marker(ws)
        return {
            "success": False,
            "n_pending": len(pending),
            "error": str(exc),
            "results": results,
        }
    return {
        "success": True,
        "n_pending": len(pending),
        "exit_code": env.raw_envelope.get("returncode"),
        "stdout_tail": (env.output_text or "")[-2000:],
        "results": results,
    }


# CLI for smoke
if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("merge-one")
    p1.add_argument("--workspace", required=True, type=Path)
    p2 = sub.add_parser("merge-batch")
    p2.add_argument("--workspaces", nargs="+", required=True, type=Path)
    args = ap.parse_args()

    if args.cmd == "merge-one":
        result = merge_one(args.workspace)
    else:
        result = merge_batch(args.workspaces)
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result.get("success") else 1)
