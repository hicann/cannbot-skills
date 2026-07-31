# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Production spawn adapter — wires aog-cann-learner agent into Mode 5.

Provides `spawn_cann_learner_agent` matching the contract Mode 5 expects:

    callable(op, workspace, module_path, sealed_dir, run_id, kb_root,
             api_catalog_path) → dict with keys:
      - sealed_files: list[Path]
      - summary_path: Path
      - candidate_paths: list[Path]
      - cann_files_read: list[Path]

Builds the brief, invokes the configured harness backend for
aog-cann-learner, parses returned envelope
+ workspace artifacts, returns the dict Mode 5 will independently re-validate.

Non-trivial structural choices:

1. **Brief is mostly cited path lists** (per agent's SKILL.md scope) —
   not free-form prompt. The agent reads `module_path`, writes sealed
   notes + public summary.json + appends to `patterns/unverified/candidates.md`.

2. **The agent reports candidate filenames in handoff line** so we can
   collect them; we also do a directory diff (kb_root/patterns/unverified/
   listing before/after) as defense in depth.

3. **cann_files_read is reconstructed** from the agent's `cann_learn_summary.json`
   `cann_files_read` field (per schema). If absent, falls back to
   listing module_path's files (over-conservative for re-scan purposes —
   prefer false-positive C34a/C34c hits over false-negatives).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
# Add src/scripts/orchestrator to path so agent_transport imports work.
sys.path.insert(0, str(_HERE.parents[1] / "orchestrator"))

import agent_transport  # noqa: E402
# Harness-decoupling: spawn via the Backend (CC plugin), not agent_transport directly.
from backends import get_backend  # noqa: E402

_backend = get_backend()


# Default per-spawn wall-clock cap. The agent does Read-only scanning
# + 4 self-checks; should comfortably finish under 30 min on any reasonable
# module. Bump if observed runs need more.
DEFAULT_TIMEOUT_SEC = 2400  # 40 min


def build_cann_learner_brief(
    *,
    op: str,
    workspace: Path,
    module_path: Path,
    sealed_dir: Path,
    run_id: str,
    kb_root: Path,
    api_catalog_path: Path,
    extraction_mode: str = "kernel_structural",
) -> str:
    """Construct the brief that aog-cann-learner agent receives.

    Per agent's SKILL.md the brief should be terse — agent already knows
    its own protocol from the system prompt. The brief just provides
    op-specific paths + the run-id token.
    """
    slug = f"{op.lower().replace('_', '')}-cl-1"

    _sealed_rel = sealed_dir.relative_to(workspace.parent.parent) if sealed_dir.is_absolute() else sealed_dir
    workspace_abs = workspace.resolve()
    module_abs = module_path.resolve()
    kb_root_abs = kb_root.resolve()
    api_catalog_abs = api_catalog_path.resolve()

    cann_strategy_path = workspace_abs / "cann_strategy_inference.md"
    summary_path = workspace_abs / "cann_learn_summary.json"
    candidates_path = kb_root_abs / "patterns" / "unverified" / "candidates.md"

    return f"""{slug} — cann_learn spawn

OP: {op}
RUN_ID: {run_id}
WORKSPACE (absolute): {workspace_abs}
MODULE_PATH (CANN source root, read-only — your scope): {module_abs}
KB_ROOT (existing KB for cross-reference, READ): {kb_root_abs}
API_CATALOG (public AscendC API allowlist, READ): {api_catalog_abs}

# Required reading (mandatory, before Phase A)
- src/skills/references/shared/ANTI_PRESSURE_PROTOCOLS.md   # P1-P8 catalog
- {cann_strategy_path}   # researcher's strategy_inference for this op
  (this is the "what KB already inferred" baseline — use to seed Phase A
   pre-scan + C35 reason-code matcher)

# Output paths (write here, ONLY here)
- Sealed (your private working dir, agent caller will archive on exit):
    {sealed_dir.resolve() / 'source_notes.md'}
    {sealed_dir.resolve() / 'extraction_drafts.md'}
- Public sanitized summary (JSON-only schema, validated by skill caller):
    {summary_path}
- Candidate KB additions (append-only):
    {candidates_path}

# Phases
A. KB pre-scan against {kb_root_abs} for op-class keywords from
   cann_strategy_inference.md. Build "what KB already knows" list.
   Output to sealed/source_notes.md §pre_scan.
B. Read 2-5 files within {module_abs} (header + impl + tiling).
   Capture algorithm structure + public-API equivalents + internal-only
   primitives (flag explicitly).
C. Pattern extraction: rewrite each insight in PUBLIC-API surface terms.
   Layered shape: title (principle) + body (abstract) + concrete-anchor
   (3-5 line snippet, public AscendC API only) + evidence + other-instances.
   Append candidates to {candidates_path} with `derived-from: cann-source`
   metadata. Drafts in sealed/extraction_drafts.md.
D. Self-review: run C34a / C34b / C34c / C35 on your own output. Drop
   leaking candidates BEFORE writing summary.json. Verdict + scores in
   summary.json `self_review_verdict` + `checks` block.
E. Exit handoff (last line of stdout):
   `→ orchestrator: cann_learn_done — kept N candidates, M metadata-fix proposals, leak_score=X`
   OR
   `→ orchestrator: cann_learn_blocked — <reason>`

{_mode_specific_section(op, extraction_mode, kb_root_abs, candidates_path)}

**Anti-patterns (REJECTED)**
- Verbatim CANN code (even renamed identifiers — C34c catches token n-gram)
- CANN-internal namespace / class / template / macro names in candidate body (C34a)
- Hand-prose inside summary.json outside schema (validator rejects)
- Writing to kernel/ or canonical KB files (G11 hook rejects)
- Patterns with NO public-API equivalent (drop, don't ship infeasible)
- Self-reporting verdict=PASS while leaving leaks (skill caller re-scans)

# Required summary.json field — extraction_mode

You MUST include `"extraction_mode": "{extraction_mode}"` as a top-level field
in `cann_learn_summary.json`. This tells the skill caller which scanner
profile + KB destination to apply. Mode 5 caller validates the value.

ITER BUDGET: single shot. Self-review failure → candidate dropped, no respawn.
"""


def _mode_specific_section(op: str, extraction_mode: str, kb_root_abs: Path, candidates_path: Path) -> str:
    """Build the mode-specific Phase B/C scope + extraction-question section.

    Mode 5 (kernel_structural) — DEFAULT, the historical scope. Read 2-5
    kernel files (header + impl + tiling). Extract algorithm structure +
    public-API equivalents. Candidates → patterns/unverified/candidates.md
    with prefix CAND-* / promotion path to P-P canonical.

    Mode 6 (build_system) NEW (2026-05-21) — read CMakeLists.txt + register
    + op_proto + apt.cpp files. Extract build-system-level recipes
    (per-source-file compile flag isolation; multi-target binary registration;
    launch macro routing). Candidates → target/ascendc/build_system/candidates.md
    with prefix CAND-BSP-* / promotion path to BSP-N canonical.

    Origin: FA Pattern A iter 1-5 (~$53 spend) empirically falsified all
    worker-scope hypotheses for V220 MIX_AIC_1_2 cube-internal sync. The
    remaining root-cause candidates live in build-system glue (per-source
    flag isolation / FFTSCNT mailbox / register attribute metadata), NOT in
    kernel.h. Mode 5 brief explicitly restricted scope to kernel files; the
    info needed wasn't reachable. Mode 6 extends scope to address this.
    """
    if extraction_mode == "build_system":
        bs_candidates = kb_root_abs / "target" / "ascendc" / "build_system" / "candidates.md"
        return f"""# Specific extraction questions for this op (Mode 6: build_system)

{op} extraction in Mode 6 is BUILD-SYSTEM-FOCUSED — the kernel structural
patterns are out of scope (Mode 5 covers those). Focus Phase B/C on the
HOST-SIDE GLUE that defines how the kernel binary is built + launched:

- **Per-source-file compile flag isolation**: which `target_compile_definitions`
  / `add_compile_definitions` apply to which source files. Especially
  `-DASCENDC_MATMUL_AICORE` — does it apply uniformly or only to cube cpp?
- **Multi-target binary registration**: how the same op registers different
  arch variants (V220 vs V351) via `register_*.cpp` / `op_proto*.cpp`.
- **Launch macro routing**: what `KERNEL_TASK_TYPE_DEFAULT` arguments appear
  in which `*_apt.cpp` files, gated by what `__NPU_ARCH__` macros.
- **CMake build dependency chains**: `add_dependencies` revealing build-time
  ordering constraints (e.g. tiling lib must build before kernel cpp).
- **Binary attribute metadata**: how `aclrtLaunchKernel` is called with
  arch-specific attributes — does the op's host glue inject FFTSCNT mailbox
  init metadata that a simpler standalone launch path doesn't replicate?
- **CMake-time API surface gates**: `if(ASCENDC_ENABLE_*)` blocks revealing
  what subsystems are conditionally compiled.

## Scope reminder for Mode 6 file reads (Phase B)

Within `module_path`, READ:
- `CMakeLists.txt` and any `*.cmake` files (top-level + per-subdir)
- `register_*.cpp` (host-side op registration)
- `op_proto*.{{cpp,h}}` (host-side op definition)
- `op_kernel/*_apt.cpp` (kernel adapter / launch entry — host-facing)
- `BUILD` / `BUILD.bazel` if present
- Top-level kernel.h (FOR CONTEXT ONLY — the launch macro it references;
  do NOT extract kernel-structural patterns in Mode 6, those are Mode 5's
  scope)

DO NOT read internal `common/*` headers, shared utility headers, or recurse.

## Output path (Mode 6 specific)

Candidates → `{bs_candidates}` (NOT the patterns/unverified/candidates.md
that Mode 5 uses — that's reserved for kernel-structural).

Use prefix `CAND-BSP-*` for candidates (CANN Build-System Pattern), promotable
to canonical `BSP-N` entries in `target/ascendc/build_system/PRINCIPLES.md`.

## Generality requirement (Mode 6)

Each BSP candidate must be applicable to AT LEAST 2 distinct op-classes
(e.g. fused-attention + fused-quant-matmul) — the build-system patterns
are higher-level than algorithm patterns and should generalize broadly.

If a pattern is op-class-specific (e.g. only flash_attention_score uses it),
flag it as `narrow_scope: true` in the candidate metadata and drop unless
the body explains WHY no generalization is possible."""

    # default: Mode 5 (kernel_structural) — preserves historical behavior
    return f"""# Specific extraction questions for this op (Mode 5: kernel_structural)

{op} is a fused operator. The skill caller wants generalizable patterns
applicable to multiple fused ops (FA / MoE / fused-norm / KV-cache / RoPE).
Focus your Phase B/C on:
- How CANN structures multi-stage tile orchestration (Q·K^T → softmax → ·V chains)
- Public-API expression of online softmax (max-reduction + exp + accumulate)
- Multi-output API contract conventions (main return + auxiliary tensors
  like softmax_max / softmax_sum, tile-coord-tagged outputs)
- UB budget partitioning for fused ops (Q/K/V cache + scores + attn + out)
- dtype templating decisions (when to fp32-promote vs stay fp16)

These questions seed Phase A pre-scan keywords AND Phase C `other-instances-predicted`
generality (so candidates aren't FA-specific).

## Scope reminder for Mode 5 file reads (Phase B)

Read 2-5 files (header + impl + tiling) within `module_path`. DO NOT recurse
beyond passed scope. Keep notes on:
- Algorithm structure (loops, dispatch, reduction tree, branch shape)
- Public-API equivalents (which AscendC API would express the same idea)
- Internal-only primitives — flag explicitly as non-portable

## Output path (Mode 5 default)

Candidates → `{candidates_path}` (patterns/unverified/candidates.md)"""


def _list_unverified_candidates(kb_root: Path) -> set[Path]:
    """Snapshot of candidate files in patterns/unverified/ (excluding markers)."""
    return _list_unverified_candidates_at(kb_root / "patterns" / "unverified")


def _list_unverified_candidates_at(dir_path: Path) -> set[Path]:
    """Snapshot of candidate files in `dir_path` (excluding markers).

    Mode 6 (build_system) uses `target/ascendc/build_system/` as the
    candidates dir; Mode 5 uses `patterns/unverified/`. Shared listing
    logic factored here.
    """
    if not dir_path.exists():
        return set()
    candidates: set[Path] = set()
    for candidate in dir_path.iterdir():
        if (
            candidate.is_file()
            and not candidate.name.startswith(".")
            and candidate.suffix == ".md"
            and candidate.name != "candidates.md"
        ):
            candidates.add(candidate)
    return candidates


def _extract_cann_files_read(summary_path: Path, module_path: Path) -> list[Path]:
    """Read summary.json's cann_files_read field; fallback to module_path enumeration."""
    if summary_path.exists():
        try:
            d = json.loads(summary_path.read_text())
            files = d.get("cann_files_read", [])
            if isinstance(files, list) and files:
                return [Path(f) for f in files]
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback: enumerate module_path files (over-conservative — re-scan
    # may flag patterns that didn't actually read those files, but better
    # than missing real leaks).
    if module_path.is_dir():
        return list(module_path.rglob("*.h")) + list(module_path.rglob("*.cpp"))
    elif module_path.is_file():
        return [module_path]
    return []


def spawn_cann_learner_agent(
    *,
    op: str,
    workspace: Path,
    module_path: Path,
    sealed_dir: Path,
    run_id: str,
    kb_root: Path,
    api_catalog_path: Path,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    extraction_mode: str = "kernel_structural",
) -> dict:
    """Spawn aog-cann-learner agent and collect output artifacts.

    Returns dict matching Mode 5's spawn_agent_func contract.

    Raises agent_transport.AgentTransportError if claude CLI fails.

    extraction_mode: "kernel_structural" (default, historical Mode 5 behavior)
    or "build_system" (Mode 6, 2026-05-21, extends scope to CMakeLists.txt +
    register + op_proto + apt.cpp for build-system pattern extraction).
    """
    brief = build_cann_learner_brief(
        op=op, workspace=workspace, module_path=module_path,
        sealed_dir=sealed_dir, run_id=run_id, kb_root=kb_root,
        api_catalog_path=api_catalog_path,
        extraction_mode=extraction_mode,
    )

    # Snapshot candidates dir before spawn (for diff after).
    # Mode 6 routes candidates to target/ascendc/build_system/candidates.md
    # instead of patterns/unverified/candidates.md.
    if extraction_mode == "build_system":
        candidates_dir = kb_root / "target" / "ascendc" / "build_system"
        candidates_md = candidates_dir / "candidates.md"
        candidates_dir.mkdir(parents=True, exist_ok=True)  # may not exist yet
        cands_before = _list_unverified_candidates_at(candidates_dir)
    else:
        candidates_dir = kb_root / "patterns" / "unverified"
        candidates_md = candidates_dir / "candidates.md"
        cands_before = _list_unverified_candidates(kb_root)
    cands_md_size_before = candidates_md.stat().st_size if candidates_md.exists() else 0

    spawn_started_at = time.time()

    # Spawn — STREAMING mode for diagnostic visibility (P0aau-c35.d, 2026-05-09).
    # First M2 attempt with --output-format json hit 40min timeout with zero
    # output and no diagnostic trace. Switching to stream-json + verbose so
    # tool_use events tee to disk and we can see what the agent did even
    # if it ultimately stalls. Performance penalty is acceptable (~10% per
    # tool call); diagnostic value is critical for first real run.
    extra_dirs: list[str] = []
    cwd = Path.cwd().resolve()
    for must_read in (workspace.resolve(), module_path.resolve(), kb_root.resolve()):
        if not str(must_read).startswith(str(cwd)):
            extra_dirs.extend(["--add-dir", str(must_read)])

    # Tee stream events into sealed dir for forensics.
    tee_path = sealed_dir / "spawn_stream.jsonl"

    result = _backend.dispatch(
        "aog-cann-learner", brief, kind="agent", mode="streaming",
        tee_path=tee_path, timeout=timeout_sec, cwd=cwd,
        extra_args=extra_dirs or None,
    )
    # (permission_mode="bypassPermissions" was explicit; it's spawn_agent_streaming's default → faithful.)

    spawn_duration = time.time() - spawn_started_at

    # Collect output artifacts.
    summary_path = workspace / "cann_learn_summary.json"

    # Sealed dir contents.
    sealed_files: list[Path] = []
    if sealed_dir.exists():
        sealed_files = [f for f in sealed_dir.rglob("*") if f.is_file()]

    # Candidate files: anything new in candidates_dir + grow in candidates.md.
    if extraction_mode == "build_system":
        cands_after = _list_unverified_candidates_at(candidates_dir)
    else:
        cands_after = _list_unverified_candidates(kb_root)
    new_cand_files = sorted(cands_after - cands_before)

    # If candidates.md grew, treat its entire current content as a "candidate path"
    # for re-scan purposes — Mode 5's revalidate runs C34a / C34c against EACH
    # candidate path's contents.
    candidate_paths: list[Path] = list(new_cand_files)
    if candidates_md.exists() and candidates_md.stat().st_size > cands_md_size_before:
        candidate_paths.append(candidates_md)

    cann_files_read = _extract_cann_files_read(summary_path, module_path)

    # Write a small spawn audit record into sealed_dir for forensics.
    if sealed_dir.exists():
        (sealed_dir / "spawn_audit.json").write_text(json.dumps({
            "run_id": run_id,
            "op": op,
            "spawn_duration_sec": round(spawn_duration, 2),
            "agent_output_text": result.output_text,
            "agent_is_error": result.is_error,
            "agent_cost_usd": result.cost_usd,
            "n_sealed_files": len(sealed_files),
            "n_new_cand_files": len(new_cand_files),
            "candidates_md_size_delta": (
                candidates_md.stat().st_size - cands_md_size_before
                if candidates_md.exists() else 0
            ),
            "n_cann_files_read": len(cann_files_read),
        }, indent=2))

    return {
        "sealed_files": sealed_files,
        "summary_path": summary_path if summary_path.exists() else None,
        "candidate_paths": candidate_paths,
        "cann_files_read": cann_files_read,
        "_agent_result": result,  # extra; Mode 5 ignores; useful for debugging
    }
