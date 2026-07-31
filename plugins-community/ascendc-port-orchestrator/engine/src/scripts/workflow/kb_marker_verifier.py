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
"""Verify a `.kb_merged` marker matches the actual KB file state on disk.

Background
==========

`/aog-knowledge-maintain` is a markdown-defined semantic reviewer. Runtime
entries are persisted by the orchestrator to user-local c-tier; release-owned
bundled KB files are read-only. The completion marker records the c-tier entry
IDs and resolved c-root. Historical bundled-file markers remain supported for
existing workspaces.

Empirically (2026-05-17 audit of 4 workspaces), the LLM in 3 out of 4
cases produces a marker with claimed `entries=TR-OL-NN,...` but does NOT
actually edit the KB files. The marker is then treated by downstream
gates (`workflow_critic SC2`, `finalize_pipeline`) as proof the merge
ran — even though no entry was actually added.

This verifier closes the gap. Given a workspace directory, it:

1. Reads `<workspace>/.kb_merged`.
2. Parses `entries=` and `merged_into=` fields.
3. For each NEW-entry ID (matching `TR-[A-Z]+-\\d+` exactly), greps
   for a matching `^### TR-...` header in EACH file listed under
   `merged_into=`. The entry must appear in at least one of those
   files.
4. Refinement / evidence-append IDs (those carrying a `(refinement)`,
   `-evidence-append`, or `-refinement` suffix) are NOT checked — they
   modify an existing entry's body without adding a new header line,
   so the strict grep would always miss them. A future revision can
   tighten this by also comparing mtimes; for v1 we accept refinements
   as unverifiable (still better than the v0 "no check at all").

The verifier exits 0 when every checkable entry is found, 2 otherwise.
Output is one line per missing entry on stderr.

CLI
===

    python3 src/scripts/workflow/kb_marker_verifier.py <workspace_dir>

Or programmatic use:

    from kb_marker_verifier import verify_marker
    rep = verify_marker(workspace_dir, project_root=...)
    if rep.verdict != "OK":
        ...

Wire-in
-------

The CLI form is wired into the existing `workflow_critic` SC2 gate (at
commit time) and the `finalize_pipeline` post-merge audit. A non-zero
exit blocks `git commit` of the workspace archive — same model as the
existing "marker absent" rejection — but now also catches "marker
present but lying".

This is a code-level check, not an LLM-judgment check. The marker
either matches disk or it does not. No prompt engineering involved.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import stat
import sys


# Match a new-entry ID exactly: "TR-EC-12", "TR-OL-22", "TR-PB-2".
_NEW_ENTRY_RE = re.compile(r"^TR-[A-Z]+-\d+$")
_C_ENTRY_RE = re.compile(r"^customer:[0-9a-f]{12}$")
_REFINEMENT_RE = re.compile(
    r"^(TR-[A-Z]+-\d+)\s*(?:\((?:refinement|evidence-append)\)"
    r"|[-_: ]+(?:refinement|evidence-append))$",
    re.IGNORECASE,
)
# Match a header in a KB file body: `### TR-OL-22:` or `### TR-OL-22 (...)`.
_HEADER_RE_TEMPLATE = r"^###\s+{eid}\b"


@dataclasses.dataclass
class VerifyReport:
    verdict: str                    # "OK" | "NO_MARKER" | "MISSING_ENTRIES"
    marker_path: str
    new_entry_ids: list[str] = dataclasses.field(default_factory=list)
    refinement_ids: list[str] = dataclasses.field(default_factory=list)
    merged_into: list[str] = dataclasses.field(default_factory=list)
    found: dict[str, str] = dataclasses.field(default_factory=dict)
    missing: list[str] = dataclasses.field(default_factory=list)
    diagnostics: list[str] = dataclasses.field(default_factory=list)
    tier: str = ""
    c_root: str = ""
    marker_fields: dict[str, str] = dataclasses.field(default_factory=dict)


def _parse_marker(marker_path: pathlib.Path) -> dict[str, str]:
    """Return marker fields while preserving spaces in line-based values.

    Accepts both line-per-key and historical space-per-key formats.
    """
    text = marker_path.read_text()
    flat: dict[str, str] = {}
    key_re = re.compile(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_-]*)=")
    for line in text.splitlines():
        matches = list(key_re.finditer(line))
        for idx, match in enumerate(matches):
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
            flat[match.group(1)] = line[match.end():end].strip()
    return flat


def _verify_c_tier_entries(rep: VerifyReport, entries: list[str]) -> VerifyReport:
    """Bind each customer:<hash> marker claim to its provider-owned JSON."""
    c_root = pathlib.Path(
        rep.c_root
        or os.environ.get("ASCENDC_PORT_USER_KB", "")
        or pathlib.Path.home() / ".ascendc-port" / "user_kb"
    ).expanduser()
    rep.c_root = str(c_root)
    for eid in entries:
        entry_file = c_root / "entries" / f"{eid.partition(':')[2]}.json"
        try:
            payload = json.loads(entry_file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            rep.missing.append(eid)
            rep.diagnostics.append(f"c-tier entry unreadable: {entry_file}: {exc}")
            continue
        scope = payload.get("scope")
        claim = payload.get("claim")
        key = payload.get("key")
        if not isinstance(scope, dict) or not isinstance(claim, str) or not isinstance(key, str):
            rep.missing.append(eid)
            rep.diagnostics.append(f"c-tier entry content fields invalid: {entry_file}")
            continue
        normalized = f"{claim.strip().lower()}|{key}|{sorted(scope.items())}"
        computed_hash = hashlib.sha1(normalized.encode()).hexdigest()[:12]
        claimed_hash = eid.partition(":")[2]
        if (
            payload.get("id") != eid
            or payload.get("tier") != "customer"
            or payload.get("role") != "user-local"
            or computed_hash != claimed_hash
            or entry_file.name != f"{computed_hash}.json"
        ):
            rep.missing.append(eid)
            rep.diagnostics.append(f"c-tier entry envelope mismatch: {entry_file}")
            continue
        rep.found[eid] = str(entry_file)
    if rep.missing:
        rep.verdict = "MISSING_ENTRIES"
    return rep


def _is_refinement(entry_id: str) -> bool:
    """True if the entry-id token denotes an evidence-append, not a new entry."""
    return _REFINEMENT_RE.fullmatch(entry_id) is not None


def _is_within(path: pathlib.Path, root: pathlib.Path) -> bool:
    """Return whether a resolved path is contained by an allowed KB root."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def _resolve_kb_files(claimed: str, project_root: pathlib.Path) -> list[pathlib.Path]:
    """Resolve a `merged_into=` token to one or more absolute KB-file paths.

    Historical skill output uses several conventions:
      - `src/skills/references/target/ascendc/OPERATIONAL_KNOWLEDGE.md`
      - `target/ascendc/OPERATIONAL_KNOWLEDGE.md` (no legacy-root prefix)
      - `OPERATIONAL_KNOWLEDGE.md` (bare filename — ambiguous)

    For bare filenames, rglob may find multiple candidates. Return ALL of them
    and let the caller grep each; finding the entry in any allowed KB root is
    sufficient.
    """
    # Candidate KB roots, in priority order. project_root-relative roots win over
    # the real relocated plugin kb/ so that (a) test fixtures resolve to their own
    # tmp KB rather than the real one, and (b) a workspace-local KB shadows the
    # global one. The real <plugin_root>/kb/ (parents[4] == plugin_root after the
    # 2026-07-05 relocation) is the final runtime fallback.
    plugin_kb = pathlib.Path(__file__).resolve().parents[4] / "kb"
    candidate_roots = [
        project_root / "src" / "skills" / "references",  # legacy layout + test fixtures
        project_root / "kb",                             # KB relocated inside the tree
        plugin_kb,                                        # real relocated plugin kb/ (runtime)
    ]
    # Normalize claimed: strip a leading known KB-root prefix if present, so a token
    # like `src/skills/references/target/...` or `kb/target/...` resolves under any root.
    norm = claimed
    for prefix in ("src/skills/references/", "kb/"):
        if norm.startswith(prefix):
            norm = norm[len(prefix):]
            break

    p = pathlib.Path(claimed)
    if p.is_absolute():
        resolved = p.resolve()
        if resolved.exists() and any(
            _is_within(resolved, root) for root in candidate_roots
        ):
            return [resolved]
        return []
    # Try the normalized path under each candidate root (first existing wins)
    for root in candidate_roots:
        cand = (root / norm).resolve()
        if cand.exists() and _is_within(cand, root):
            return [cand]
    # Bare filename: return ALL recursive matches under the first existing root
    if "/" not in norm:
        for root in candidate_roots:
            if root.exists():
                matches = sorted(
                    match for match in root.rglob(norm)
                    if _is_within(match, root)
                )
                if matches:
                    return matches
    return []  # caller will surface as missing-file diagnostic


def verify_marker(workspace_dir: pathlib.Path,
                  project_root: pathlib.Path | None = None,
                  expected_c_root: pathlib.Path | None = None) -> VerifyReport:
    workspace_dir = workspace_dir.resolve()
    project_root = (project_root or _infer_project_root(workspace_dir)).resolve()

    marker_path = workspace_dir / ".kb_merged"
    try:
        marker_mode = marker_path.lstat().st_mode
    except FileNotFoundError:
        return VerifyReport(
            verdict="NO_MARKER",
            marker_path=str(marker_path),
            diagnostics=[f"no .kb_merged at {marker_path}"],
        )
    except OSError as exc:
        return VerifyReport(
            verdict="MISSING_ENTRIES",
            marker_path=str(marker_path),
            diagnostics=[f"cannot inspect .kb_merged: {exc}"],
        )
    if not stat.S_ISREG(marker_mode):
        return VerifyReport(
            verdict="MISSING_ENTRIES",
            marker_path=str(marker_path),
            diagnostics=[".kb_merged must be a regular file, not a link or device"],
        )

    fields = _parse_marker(marker_path)
    entries = [e.strip() for e in fields.get("entries", "").split(",") if e.strip()]
    merged_into = [m.strip() for m in fields.get("merged_into", "").split(",") if m.strip()]
    tier = fields.get("tier", "")
    if tier == "customer":
        new_ids = [e for e in entries if _C_ENTRY_RE.match(e)]
        refinement_ids = [e for e in entries if e not in new_ids and e != "none"]
        invalid_ids: list[str] = []
    else:
        new_ids = [e for e in entries if _NEW_ENTRY_RE.match(e)]
        refinement_ids = [e for e in entries if _is_refinement(e)]
        invalid_ids = [
            e for e in entries
            if e not in new_ids and e not in refinement_ids
        ]

    rep = VerifyReport(
        verdict="OK",
        marker_path=str(marker_path),
        new_entry_ids=new_ids,
        refinement_ids=refinement_ids,
        merged_into=merged_into,
        tier=tier,
        c_root=fields.get("c_root", ""),
        marker_fields=fields,
    )

    if tier == "customer" and refinement_ids:
        rep.verdict = "MISSING_ENTRIES"
        rep.missing.extend(refinement_ids)
        rep.diagnostics.append(
            "customer marker contains invalid entry IDs: "
            f"{refinement_ids!r}"
        )

    if tier != "customer":
        if invalid_ids:
            rep.verdict = "MISSING_ENTRIES"
            rep.missing.extend(invalid_ids)
            rep.diagnostics.append(
                f"legacy marker contains invalid entry IDs: {invalid_ids!r}"
            )
        if not entries:
            rep.verdict = "MISSING_ENTRIES"
            rep.diagnostics.append("legacy marker has no entries= ids")
        if not merged_into:
            rep.verdict = "MISSING_ENTRIES"

    if not merged_into:
        rep.diagnostics.append("marker has empty merged_into= field")

    if tier == "customer":
        required = {
            "merge_run", "tier", "c_root", "merged_into", "entries",
            "reviewed", "rejected", "mode",
        }
        missing_fields = sorted(required - set(fields))
        try:
            reviewed = int(fields.get("reviewed", ""))
            rejected = int(fields.get("rejected", ""))
        except ValueError:
            reviewed = rejected = -1
        entries_shape_ok = (
            fields.get("entries") == "none"
            if not new_ids
            else "none" not in fields.get("entries", "").split(",")
        )
        if (
            missing_fields
            or merged_into != ["user-c-tier"]
            or fields.get("mode") not in {"update", "batch"}
            or reviewed < 0
            or not 0 <= rejected <= reviewed
            or reviewed - rejected != len(new_ids)
            or not entries_shape_ok
        ):
            rep.verdict = "MISSING_ENTRIES"
            rep.diagnostics.append(
                "customer marker does not match the orchestrator completion schema"
                + (f"; missing fields: {missing_fields!r}" if missing_fields else "")
            )
            return rep
        if expected_c_root is not None:
            try:
                claimed_c_root = pathlib.Path(rep.c_root).expanduser().resolve()
            except (OSError, RuntimeError):
                claimed_c_root = pathlib.Path()
            if not rep.c_root or claimed_c_root != expected_c_root.expanduser().resolve():
                rep.verdict = "MISSING_ENTRIES"
                rep.diagnostics.append(
                    "customer marker c_root does not match the configured provider root"
                )
                return rep
        if not new_ids and not refinement_ids:
            rep.diagnostics.append("marker has no durable entry ids; reviewed intake was empty")
            return rep
        return _verify_c_tier_entries(rep, new_ids)

    if rep.verdict != "OK":
        return rep

    # Build candidate file paths for grepping (one token may resolve to
    # multiple files when bare-named — search all of them).
    kb_files: list[pathlib.Path] = []
    for token in merged_into:
        resolved = _resolve_kb_files(token, project_root)
        if resolved:
            kb_files.extend(resolved)
        else:
            rep.diagnostics.append(
                f"merged_into= references unresolvable file: {token!r}"
            )
    if not kb_files:
        rep.verdict = "MISSING_ENTRIES"
        rep.diagnostics.append("legacy marker resolves no allowed KB files")
        return rep

    # For each NEW entry id, grep for `^### <eid>` in each KB file.
    for eid in new_ids:
        header_re = re.compile(_HEADER_RE_TEMPLATE.format(eid=re.escape(eid)),
                               re.MULTILINE)
        for f in kb_files:
            try:
                body = f.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            if header_re.search(body):
                try:
                    rep.found[eid] = str(f.relative_to(project_root))
                except ValueError:
                    # Resolved KB file lives outside project_root (e.g. the real
                    # relocated <plugin_root>/kb/ when project_root == engine/).
                    rep.found[eid] = str(f)
                break
        else:
            rep.missing.append(eid)

    # Refinement/evidence-append markers cannot prove the appended prose, but
    # they must at least bind to the existing canonical entry being refined.
    for refinement in refinement_ids:
        match = _REFINEMENT_RE.fullmatch(refinement)
        base_id = match.group(1).upper() if match is not None else ""
        header_re = re.compile(
            _HEADER_RE_TEMPLATE.format(eid=re.escape(base_id)),
            re.MULTILINE,
        )
        for file_path in kb_files:
            try:
                body = file_path.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            if header_re.search(body):
                rep.found[refinement] = str(file_path)
                break
        else:
            rep.missing.append(refinement)

    if rep.missing:
        rep.verdict = "MISSING_ENTRIES"

    return rep


def _infer_project_root(workspace_dir: pathlib.Path) -> pathlib.Path:
    """Walk up to find the repo root (contains `.claude/` or `src/skills/`)."""
    for parent in [workspace_dir, *workspace_dir.parents]:
        if (parent / ".claude").is_dir() or (parent / "src" / "skills").is_dir():
            return parent
    # Fallback: workspace_dir.parent.parent (workspace/<op>/.kb_merged → repo)
    return workspace_dir.parent.parent


def format_report(rep: VerifyReport) -> str:
    lines = [
        f"[kb-marker-verifier] verdict: {rep.verdict}",
        f"[kb-marker-verifier] marker: {rep.marker_path}",
    ]
    if rep.new_entry_ids:
        lines.append(
            f"[kb-marker-verifier] new entries claimed: {rep.new_entry_ids!r}"
        )
    if rep.refinement_ids:
        lines.append(
            f"[kb-marker-verifier] refinements/evidence-appends "
            f"(unchecked): {rep.refinement_ids!r}"
        )
    if rep.merged_into:
        lines.append(
            f"[kb-marker-verifier] merged_into: {rep.merged_into!r}"
        )
    if rep.tier == "customer":
        lines.append(f"[kb-marker-verifier] c-tier root: {rep.c_root}")
    if rep.found:
        lines.append("[kb-marker-verifier] found entries:")
        for eid, where in rep.found.items():
            lines.append(f"  ✓ {eid} → {where}")
    if rep.missing:
        lines.append("[kb-marker-verifier] MISSING entries (BUG):")
        for eid in rep.missing:
            if rep.tier == "customer":
                lines.append(f"  ✗ {eid} (no matching c-tier provider entry)")
            else:
                lines.append(
                    f"  ✗ {eid} (marker claims merged but no `### {eid}` "
                    f"header in any merged_into= file)"
                )
        lines.append("  The marker does not match durable KB state. Re-run the merge.")
    if rep.diagnostics:
        lines.append("[kb-marker-verifier] diagnostics:")
        for d in rep.diagnostics:
            lines.append(f"  - {d}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workspace", help="path to workspace/<op>/ containing .kb_merged")
    ap.add_argument("--project-root",
                    help="path to repo root (auto-inferred if omitted)")
    ap.add_argument("--no-marker-ok", action="store_true",
                    help="treat NO_MARKER verdict as exit 0 (default: exit 0)")
    args = ap.parse_args(argv)

    workspace_dir = pathlib.Path(args.workspace)
    project_root = pathlib.Path(args.project_root) if args.project_root else None
    rep = verify_marker(workspace_dir, project_root=project_root)
    print(format_report(rep), file=sys.stderr)
    if rep.verdict == "OK":
        return 0
    if rep.verdict == "NO_MARKER":
        return 0  # not an error; SC2 gate handles "marker missing"
    return 2  # MISSING_ENTRIES — actionable bug


if __name__ == "__main__":
    sys.exit(main())
