# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Phase-specific guidance - what the LLM should do next.

`get_guidance(task_dir)` is the only public API; it reads phase + progress
+ task config + plan, then returns the `[AR Phase: …]` message that hooks
inject into Claude's context after every state-changing event.

The XML schema example for plan creation (`_PLAN_XML_EXAMPLE`) and the
field-rules tail (`_PLAN_FIELD_RULES`) live here - they're prompt content
shared between PLAN, DIAGNOSE, and REPLAN guidance.
"""
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

from .state_store import (
    BASELINE,
    DIAGNOSE,
    DIAGNOSE_ATTEMPTS_CAP,
    EDIT,
    FINISH,
    INIT,
    PLAN,
    PLAN_ITEMS_FILE,
    REPLAN,
    diagnose_artifact_path,
    diagnose_marker,
    history_path,
    load_progress,
    read_phase,
    state_path,
)
from .validators import (
    DIAGNOSE_MANUAL_FALLBACK,
    DIAGNOSE_READY,
    diagnose_state,
    get_active_item,
)

logger = logging.getLogger(__name__)


def _format_fail_record(
    record: dict,
    progress: Optional[dict] = None,
) -> str:
    """Render one compact failure block for diagnosis."""
    round_id = record.get("round", "?")
    description = (record.get("description") or "")[:80]
    lines = [f"  R{round_id}: {description}"]
    signals = record.get("failure_signals") or {}
    structured = signals.get("signals") or []
    python_error = signals.get("python_error")
    lines.extend(_format_structured_signals(structured, python_error))
    if not structured and not python_error:
        lines.extend(_format_unstructured_failure(record, signals))
    shape_block = _failed_shapes_block(record.get("metrics"), progress)
    if shape_block:
        lines.append(shape_block)
    return "\n".join(lines)


def _format_structured_signals(
    signals: list[dict],
    python_error: Optional[str],
) -> list[str]:
    lines = []
    for signal in signals[:2]:
        params = ", ".join(
            f"{key}={value}"
            for key, value in signal.items()
            if key not in ("kind", "excerpt", "hint") and value is not None
        )
        marker = f"      → {signal.get('kind', '?')}"
        lines.append(f"{marker}  [{params}]" if params else marker)
        if signal.get("hint"):
            lines.append(f"        hint: {signal['hint']}")
    if python_error:
        lines.append(f"      python_error: {python_error[:200]}")
    return lines


def _format_unstructured_failure(record: dict, signals: dict) -> list[str]:
    lines = []
    error = (record.get("error") or "").strip()
    if error and "verify failed (kernel broken)" not in error:
        lines.append(f"      error: {error[:160]}")
    tail = (
        signals.get("tail_excerpt")
        or record.get("raw_output_tail")
        or ""
    ).strip()
    kept = [
        line
        for line in tail.splitlines()[-8:]
        if line.strip()
    ]
    if kept:
        lines.append("      tail:")
        lines.extend(f"        {line[:160]}" for line in kept)
    return lines


# Shared plan-item scaffolding shown in PLAN / DIAGNOSE / REPLAN guidance.
#
# Design notes:
#
# 1. THREE concrete items in the example, not one + "repeat" hint. Agents
#    consistently copy-as-shown - a single-item example produces single-
#    item submissions that fail the ">=3 items" check immediately.
#
# 2. Schema rules live in a plain bullet block ABOVE the example, not as
#    inline <!-- XML comments -->. Comments inside the structure get
#    treated as part of the shape and either leak into the agent's output
#    or train the agent to think the schema is "this with prose".
#
# 3. Wrong-vs-right pairs cover the most common drifts (attributes,
#    snake_case desc, all-parameter-tuning plans). Negative-only rules
#    underperform - pair each "don't" with a concrete "do" alternative.
#
# 4. Example items deliberately avoid every word in create_plan.py's
#    `_PARAM_WORDS` / `_PARAM_PHRASES` (block, tile, num_warps, etc.) so
#    the diversity check passes when the agent generalises the shape to
#    their own task. The three items represent: kernel fusion / memory
#    layout / data alignment - structural changes, not parameter sweeps.
#
# 5. XML stays the required format (tag-delimited beats JSON for LLMs -
#    no commas to forget, no brace balance).
_PLAN_XML_RULES = (
    "Plan item schema (each rule below maps to a create_plan.py check):\n"
    "  - Root <items> has NO attributes.\n"
    "  - At least 3 <item> children. NO attributes on <item> (pid is auto-assigned).\n"
    "  - Each <item> has EXACTLY two children: <desc>, <rationale>.\n"
    "    NO <id>, <pid>, <keywords>, <priority>, or any other tag.\n"
    "  - <desc>: short prose sentence, >=12 chars, MUST contain spaces.\n"
    "  - <rationale>: 30-400 chars, explains WHY the change should help.\n"
    "  - At most ONE item may be pure parameter tuning (block size / num_warps /\n"
    "    num_stages / autotune sweep). The rest must be structural changes:\n"
    "    algorithmic / fusion / memory layout / data movement.\n"
    "\n"
    "Common drifts (these get rejected):\n"
    "  WRONG: <item id=\"p1\">...</item>          -> <item>...</item>     (no attributes)\n"
    "  WRONG: <desc>fuse_swiglu_epilogue</desc> -> <desc>Fuse the SwiGLU epilogue</desc>\n"
    "         (snake_case label fails the 'must contain spaces' check)\n"
    "  WRONG: 3 items all named 'tune block size to N' -> mix in a fusion or\n"
    "         layout change (diversity check rejects param-only plans)\n"
    "  WRONG: <keywords>fuse,matmul</keywords>  -> drop it, _check_diversity\n"
    "         tokenises <desc> directly, no separate keyword tag exists\n"
    "  Escape special chars in text: '&'->'&amp;', '<'->'&lt;', '>'->'&gt;'\n"
    "  (or wrap the field body in <![CDATA[...]]>)."
)
_PLAN_XML_EXAMPLE = (
    '<items>\n'
    '  <item>\n'
    '    <desc>Fuse the activation into the matmul epilogue to avoid a second '
    'kernel launch</desc>\n'
    '    <rationale>The separate activation kernel re-reads the matmul output '
    'from DRAM; folding it into the epilogue removes one round-trip and one '
    'launch overhead.</rationale>\n'
    '  </item>\n'
    '  <item>\n'
    '    <desc>Transpose the input layout so the reduction axis is contiguous '
    'in memory</desc>\n'
    '    <rationale>Current reduction stride is 16380 bytes, which traps the '
    'vector core because it needs 256-byte-aligned access. Making the reduce '
    'axis contiguous gives aligned vectorised loads.</rationale>\n'
    '  </item>\n'
    '  <item>\n'
    '    <desc>Pad the inner dimension to a multiple of 64 elements</desc>\n'
    '    <rationale>The current inner dim is 4095, one short of the 4096 '
    'alignment the vector unit needs; padding to the next multiple lets the '
    'main loop drop its tail-handling branch entirely.</rationale>\n'
    '  </item>\n'
    '</items>'
)
# One-line schema reminder for re-plans (REPLAN / DIAGNOSE / repeat PLAN).
# The agent already authored plan.md once and has it on disk as a live
# example, so the full _PLAN_XML_RULES + _PLAN_XML_EXAMPLE blocks are pure
# repetition — create_plan.py re-validates everything anyway.
_PLAN_SCHEMA_ONELINE = (
    "Schema (same as your existing plan.md, create_plan.py re-checks it): "
    ">= 3 <item>, each with EXACTLY <desc> (prose, has spaces) / <rationale> "
    "(30-400 chars); no attributes on any "
    "tag; at most ONE pure parameter-tuning item, the rest structural."
)
# Single-item shape shown on re-plans — enough to anchor the tag nesting
# without the full 3-item block. No XML comments inside the structure
# (design note 2: the agent treats them as part of the shape and leaks
# them) — the "repeat" instruction lives in the prose around this block.
_PLAN_XML_EXAMPLE_ONE = (
    '<items>\n'
    '  <item>\n'
    '    <desc>Reuse the second calc buffer for the sigmoid output instead '
    'of a third allocation</desc>\n'
    '    <rationale>Freeing one TILE*4 fp32 slot lets BUFFER_NUM go 2->3 '
    'within the 192KB UB, enabling MTE2/V overlap.</rationale>\n'
    '  </item>\n'
    '</items>'
)
# Kept as a named constant because callers (create_plan.py docstring,
# tests) reference the rules block by attribute. The rules text is the
# primary content; _PLAN_FIELD_RULES is now an alias to keep the public
# name stable.
_PLAN_FIELD_RULES = _PLAN_XML_RULES


def _create_plan_instruction(task_dir: str, *, first_plan: bool = True) -> str:
    """'How to invoke create_plan.py' block for PLAN / DIAGNOSE / REPLAN.

    Emits the canonical two-step flow (write XML to the FIXED path, then
    run create_plan.py with just <task_dir>). The fixed path eliminates
    the LLM-drift class where the model wrote to one path then passed a
    different `@<path>` to create_plan.

    first_plan controls the schema teaching:
      - first_plan=True (the very first PLAN, plan_version 0): print the
        full _PLAN_XML_RULES + the worked 3-item _PLAN_XML_EXAMPLE so the
        agent learns the format once.
      - first_plan=False (every re-plan: REPLAN, DIAGNOSE, or a PLAN
        revisited with plan_version >= 1): the agent already wrote plan.md
        and has it on disk as a live example. Re-printing the rules +
        example is pure repetition (create_plan.py re-validates anyway),
        so collapse to a one-line schema reminder.
    """
    xml_path = state_path(task_dir, PLAN_ITEMS_FILE)
    invoke = (
        f"To create the plan, do EXACTLY these two steps:\n"
        f"  1. Use the Write tool to write your <items>...</items> XML to:\n"
        f"       {xml_path}\n"
        f"     (Path is fixed - do NOT invent a different path, do NOT use "
        f"/tmp/, do NOT pass it as a CLI arg later.)\n"
        f"  2. Run: python scripts/engine/create_plan.py \"{task_dir}\"\n"
        f"     (No second argument. The script reads .ar_state/"
        f"{PLAN_ITEMS_FILE} automatically.)\n"
    )
    if not first_plan:
        return (
            f"{invoke}\n"
            f"{_PLAN_SCHEMA_ONELINE}\n"
            f"\n"
            f"Shape of a single <item> (repeat per change; count + "
            f"constraints are in the schema line above):\n"
            f"{_PLAN_XML_EXAMPLE_ONE}\n"
        )
    return (
        f"{invoke}"
        f"\n"
        f"{_PLAN_XML_RULES}\n"
        f"\n"
        f"Canonical example (copy the SHAPE - three items, two children "
        f"each;\nreplace the contents with items that fit YOUR task):\n"
        f"{_PLAN_XML_EXAMPLE}\n"
    )


def _load_config_safe(task_dir: str):
    """Load TaskConfig, return None on any failure.

    task_config lives in scripts/ root (one level up from this package);
    insert the parent dir into sys.path so the import resolves no matter
    who's importing us.
    """
    try:
        _scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        from task_config import load_task_config
        return load_task_config(task_dir)
    except Exception:
        return None


def _target_dsl_safe() -> tuple[str, str]:
    """Return (KernelVerifier DSL, backend) for prompts."""
    try:
        from utils.settings import (
            target_backend as _target_backend,
        )
        from utils.settings import (
            target_dsl as _target_dsl,
        )
        return _target_dsl(), _target_backend()
    except Exception:
        return "<configured-dsl>", "<backend>"


def _editable_paths(task_dir: str, editable: list[str]) -> list[str]:
    return [f"{task_dir}/{name}" for name in editable]


def _editable_scope_text(editable: list[str]) -> str:
    if editable:
        return ", ".join(editable)
    return "task.yaml editable_files"


def _multi_shape_plan_note(progress: Optional[dict],
                           task_dir: Optional[str] = None) -> str:
    """One-line note for the PLAN phase: say the op is multi-shape and point
    at the actual file(s) holding the shape spec. Deliberately does NOT
    list individual shapes - plan items are coarse-grained decisions, a
    30-line case dump in the planning prompt makes the agent over-engineer
    for shape generality at the expense of writing good plan items.

    NPUKernelBench-style refs read shapes from a sidecar JSON (the ref's
    `get_input_groups()` opens a same-directory `<basename>.json`). When
    that JSON is present in `task_dir`, this note names it explicitly -
    pointing at reference.py alone is not enough because the .py file is
    just a loader; the actual shape list lives in the JSON.

    Returns "" for single-shape ops (progress.num_cases <= 1) and when
    progress isn't initialized yet (pre-BASELINE).
    """
    if not progress:
        return ""
    n = progress.get("num_cases")
    if not isinstance(n, int) or n <= 1:
        return ""

    where = _shape_location(task_dir, _shape_sidecars(task_dir))

    return (
        f"Note: multi-shape op - reference exposes {n} input groups "
        f"via get_input_groups(). {where}\n"
        f"Plan items must hold across all shapes; rely on shape-aware "
        f"logic (read shape at runtime, dispatch on dtype/rank, adapt "
        f"tile size) rather than constants pinned to one shape."
    )


def _shape_sidecars(task_dir: Optional[str]) -> list[str]:
    if not task_dir or not os.path.isdir(task_dir):
        return []
    try:
        sidecars = []
        for name in sorted(os.listdir(task_dir)):
            is_visible_json = name.endswith(".json") and not name.startswith(".")
            if is_visible_json and os.path.isfile(os.path.join(task_dir, name)):
                sidecars.append(name)
        return sidecars
    except OSError:
        logger.debug("Could not enumerate verification sidecars", exc_info=True)
        return []


def _shape_location(task_dir: Optional[str], sidecars: list[str]) -> str:
    paths = [f"{task_dir}/{name}" for name in sidecars]
    if len(paths) == 1:
        return f"shape list: {paths[0]}"
    if paths:
        return "shape lists:\n  - " + "\n  - ".join(paths)
    return (
        f"shape list: {task_dir}/reference.py "
        "(in the get_input_groups() body)"
    )


def _last_failure_metrics(task_dir: str) -> Optional[dict]:
    """Return the metrics dict of the most recent FAIL/SEED record in
    history.jsonl whose `correctness` is False. Returns None when no
    failed record exists or history is missing.
    """
    failed = [
        record
        for record in _read_history_records(task_dir)
        if record.get("correctness") is False
    ]
    if not failed:
        return None
    metrics = failed[-1].get("metrics")
    return metrics if isinstance(metrics, dict) else None


def _parse_history_lines(history_file) -> list[dict]:
    records = []
    for line in history_file:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.debug("Skipping malformed history record", exc_info=True)
    return records


def _read_history_records(task_dir: str) -> list[dict]:
    path = history_path(task_dir)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as history_file:
            return _parse_history_lines(history_file)
    except OSError:
        logger.debug("Could not read task history %s", path, exc_info=True)
        return []


def _failed_shapes_block(metrics: Optional[dict],
                         progress: Optional[dict],
                         *, max_listed: int = 5) -> str:
    """Render the per-shape failure detail used in DIAGNOSE's FAIL records.

    Pulls from the metrics block of a single FAIL history record (which
    eval_client populates from the verify subprocess's verify_json):
      - correctness_failed_cases: list of failing case indices
      - correctness_total_cases:  total case count at FAIL time
      - correctness_worst_case:   index with the largest max_abs_diff
      - correctness_worst_max_abs: that diff

    Resolves indices via progress.per_shape_descs
    so the agent sees both the index and the actual shape it fouled up.
    Returns "" when none of those fields are present (e.g. compile-error
    failure with no per-case detail, or single-shape op where the failure
    block is redundant).
    """
    if not metrics:
        return ""
    failed = metrics.get("correctness_failed_cases")
    total = metrics.get("correctness_total_cases")
    if not isinstance(failed, list) or not failed:
        return ""
    if not isinstance(total, int) or total <= 1:
        # Single-shape FAIL: the message that says "kernel broken"
        # already conveys this; an extra "1/1 shapes failed" block is noise.
        return ""

    descs = (progress or {}).get("per_shape_descs") or []
    parts = [f"      failed shapes: {len(failed)}/{total}"]
    for idx in failed[:max_listed]:
        if isinstance(idx, int) and 0 <= idx < len(descs):
            parts.append(f"        [{idx}] {descs[idx]}")
        else:
            parts.append(f"        [{idx}] (desc unavailable)")
    if len(failed) > max_listed:
        parts.append(f"        ... ({len(failed)} failures total)")
    worst_idx = metrics.get("correctness_worst_case")
    worst_max = metrics.get("correctness_worst_max_abs")
    if isinstance(worst_idx, int) and isinstance(worst_max, (int, float)):
        parts.append(
            f"      worst: case [{worst_idx}] max_abs={worst_max:.3e}"
        )
    return "\n".join(parts)


def _diagnose_plan_next_step(task_dir: str, *,
                             artifact_path: Optional[str] = None,
                             fallback: bool = False) -> str:
    """Guidance text for the post-DIAGNOSE create_plan step.

    Two callers in get_guidance: action == DIAGNOSE_READY passes the
    artifact path; action == DIAGNOSE_MANUAL_FALLBACK passes nothing
    (artifact_path is unused in fallback mode - the diagnosis context
    is history.jsonl + plan.md).
    """
    if fallback:
        header = "[AR Phase: DIAGNOSE - manual planning fallback]"
        source = "history.jsonl + plan.md (subagent route exhausted)"
    else:
        header = "[AR Phase: DIAGNOSE - diagnosis ready]"
        source = artifact_path or "(diagnosis artifact)"
    return (
        f"{header}\n"
        f"Create a NEW plan with >= 3 diverse items using {source}.\n"
        f"Max 1 parameter-tuning item; the rest must be structural changes "
        f"(algorithmic / fusion / memory layout / data movement).\n\n"
        f"{_create_plan_instruction(task_dir, first_plan=False)}"
        f"\nAfter create_plan.py validates, the hook advances phase to EDIT "
        f"and emits the plan-mirror payload."
    )


def _trace_analysis_block(task_dir: str, backend: str) -> str:
    """Tell the agent about existing --trace captures + how to read them;
    empty when none / non-Ascend. Used by PLAN/REPLAN/DIAGNOSE.
    """
    if backend != "ascend":
        return ""
    import glob
    dirs = glob.glob(os.path.join(
        task_dir, ".ar_state", "op_autoresearch_verify", "**",
        "prof_generation_output_case_*"), recursive=True)
    if not dirs:
        return ""
    return (
        f"\n[trace] {len(dirs)} prior --trace capture(s) under op_autoresearch_verify/<op>/"
        "Iteration*_Step<round>_verify/prof_generation_output_case_<N>/ "
        "(case_<N> = per-shape #N). Read kernel_details.csv / op_statistic.csv "
        "(per-op time) to find the hotspot op / cube-vs-vector bound before "
        "planning; trace_view.json (same dir) is the full timeline — open in "
        "perfetto / chrome://tracing.\n")


@dataclass(frozen=True)
class GuidanceContext:
    task_dir: str
    phase: str
    active: Optional[dict]
    progress: Optional[dict]
    config: object
    editable: list[str]
    primary_metric: str
    target_dsl: str
    backend: str


@dataclass(frozen=True)
class DiagnosisBrief:
    context: GuidanceContext
    plan_version: int
    attempts: int
    artifact_path: str
    recent_summary: str
    fail_details: str
    metric_line: str


def _guidance_context(task_dir: str) -> GuidanceContext:
    config = _load_config_safe(task_dir)
    target_dsl, backend = _target_dsl_safe()
    return GuidanceContext(
        task_dir=task_dir,
        phase=read_phase(task_dir),
        active=get_active_item(task_dir),
        progress=load_progress(task_dir),
        config=config,
        editable=config.editable_files if config else [],
        primary_metric=config.primary_metric if config else "score",
        target_dsl=target_dsl,
        backend=backend,
    )


def _init_guidance(context: GuidanceContext) -> str:
    return f'[AR Phase: INIT] Run: export AR_TASK_DIR="{context.task_dir}"'


def _baseline_guidance(context: GuidanceContext) -> str:
    return (
        "[AR Phase: BASELINE] Run: "
        f'python scripts/engine/baseline.py "{context.task_dir}"'
    )


def _plan_guidance(context: GuidanceContext) -> str:
    progress = context.progress or {}
    baseline = progress.get("baseline_metric")
    metric_hint = (
        f" Baseline {context.primary_metric}: {baseline}."
        if baseline is not None
        else ""
    )
    plan_note = _multi_shape_plan_note(
        context.progress,
        task_dir=context.task_dir,
    )
    note_section = f"\n\n{plan_note}" if plan_note else ""
    first_plan = progress.get("plan_version", 0) == 0
    return (
        "[AR Phase: PLAN] "
        f"Target DSL: {context.target_dsl} "
        f"(backend: {context.backend}). "
        "Read task.yaml, reference.py, and editable files "
        f"({_editable_scope_text(context.editable)}). Directory-backed DSLs "
        "may expose multiple editable project files; plan only changes "
        "inside that editable surface."
        f"{metric_hint}{note_section}{_seed_failure_section(context)}"
        f"{_trace_analysis_block(context.task_dir, context.backend)}\n\n"
        f"{_create_plan_instruction(context.task_dir, first_plan=first_plan)}\n"
        "The script writes plan.md in the correct format. Hook validates "
        "and advances to EDIT.\n"
        "(After validation the hook emits the plan-mirror payload; mirror "
        "it verbatim, do not pre-emptively craft one here.)"
    )


def _seed_failure_section(context: GuidanceContext) -> str:
    progress = context.progress
    if not progress:
        return ""
    outcome = progress.get("baseline_outcome")
    seed_missing = progress.get("seed_metric") is None
    if not seed_missing and outcome != "kernel_fail":
        return ""
    reason = (
        "seed kernel produced no timing (compile/profile failed)"
        if seed_missing
        else "seed kernel ran but failed correctness vs reference"
    )
    shape_detail = _seed_failure_shapes(context, outcome)
    editable_paths = ", ".join(
        _editable_paths(context.task_dir, context.editable)
    )
    return (
        f"\n\nSEED FAILED: {reason}.\n"
        "Plan items must focus on FIXING / REWRITING the editable seed "
        f"surface ({_editable_scope_text(context.editable)}) so the next "
        "round passes baseline.\n"
        "Read these editable files to see what failed: "
        f"{editable_paths or context.task_dir}. baseline.py printed "
        "structured failure signals (UB overflow / aivec trap / OOM / "
        "correctness mismatch) above — use those as primary evidence. "
        "Each plan item is a structural change attempt; incremental fixes "
        f"converge faster than rewrites from scratch.{shape_detail}"
    )


def _seed_failure_shapes(context: GuidanceContext, outcome: object) -> str:
    if outcome != "kernel_fail":
        return ""
    metrics = _last_failure_metrics(context.task_dir)
    block = _failed_shapes_block(metrics, context.progress)
    if not block:
        return ""
    return (
        "\nThe BASELINE correctness failure had per-shape detail "
        f"(round-0 SEED record):\n{block}\n"
    )


def _edit_guidance(context: GuidanceContext) -> str:
    active = context.active
    description = active["description"] if active else "(no active item)"
    item_id = active["id"] if active else "?"
    files_hint = (
        f" (files: {', '.join(context.editable)})"
        if context.editable
        else ""
    )
    return (
        f"[AR Phase: EDIT] ACTIVE item: **{item_id}** - {description}\n"
        f"{files_hint}\n"
        f"CRITICAL: Implement ONLY {item_id}'s idea. Do NOT implement "
        "other plan items.\n"
        f"Target DSL: {context.target_dsl}; edit only task.yaml "
        "editable_files. For directory-backed DSLs, this may include "
        "wrapper and project source/build files, not just kernel.py.\n"
        f"The pipeline will settle {item_id} with this round's metric.\n"
        f'Make your edit(s), then: python scripts/engine/pipeline.py '
        f'"{context.task_dir}"\n'
        f"{_edit_trace_hint(context)}"
        "(The hook delivers the plan-mirror payload after each settle / "
        "create_plan; mirror it verbatim, do not synthesize one from this hint.)"
    )


def _edit_trace_hint(context: GuidanceContext) -> str:
    if context.backend != "ascend":
        return ""
    return (
        "PROFILE (only when evidence is needed): run python "
        f'scripts/engine/pipeline.py "{context.task_dir}" --trace to keep '
        "trace_view.json plus kernel_details / op_statistic CSVs. Later "
        "planning or diagnosis guidance automatically surfaces them; "
        "read the CSVs first for hotspot / bound analysis.\n"
    )


def _diagnose_guidance(context: GuidanceContext) -> str:
    progress = context.progress
    diagnose = (
        diagnose_state(context.task_dir, progress=progress)
        if progress
        else None
    )
    plan_version = diagnose.plan_version if diagnose else 0
    artifact_path = diagnose_artifact_path(context.task_dir, plan_version)
    if diagnose and diagnose.action == DIAGNOSE_READY:
        return _diagnose_plan_next_step(
            context.task_dir,
            artifact_path=artifact_path,
        )
    if diagnose and diagnose.action == DIAGNOSE_MANUAL_FALLBACK:
        return _diagnose_plan_next_step(context.task_dir, fallback=True)
    brief = _build_diagnosis_brief(
        context,
        plan_version,
        diagnose.attempts if diagnose else 0,
        artifact_path,
    )
    return _diagnosis_action(brief, _diagnosis_subagent_prompt(brief))


def _build_diagnosis_brief(
    context: GuidanceContext,
    plan_version: int,
    attempts: int,
    artifact_path: str,
) -> DiagnosisBrief:
    records = _read_history_records(context.task_dir)
    recent = "".join(
        f"  R{record.get('round', '?')}: "
        f"{record.get('decision', '?')} - "
        f"{record.get('description', '')[:60]}\n"
        for record in records[-5:]
    )
    failures = []
    for record in records:
        if record.get("decision") == "FAIL" and record.get("round") is not None:
            failures.append(record)
    failures = failures[-3:]
    fail_details = "\n".join(
        _format_fail_record(record, context.progress)
        for record in failures
    )
    return DiagnosisBrief(
        context=context,
        plan_version=plan_version,
        attempts=attempts,
        artifact_path=artifact_path,
        recent_summary=recent,
        fail_details=fail_details,
        metric_line=_diagnosis_metric_line(context),
    )


def _diagnosis_metric_line(context: GuidanceContext) -> str:
    progress = context.progress or {}
    values = (
        progress.get("seed_metric"),
        progress.get("baseline_metric"),
        progress.get("best_metric"),
    )
    if not any(value is not None for value in values):
        return ""
    return (
        f"\nMetrics ({context.primary_metric}): seed={values[0]} | "
        f"ref_baseline={values[1]} | current_best={values[2]}"
    )


def _diagnosis_subagent_prompt(brief: DiagnosisBrief) -> str:
    context = brief.context
    config = context.config
    arch = config.arch if config and config.arch else "<unknown>"
    editable = context.editable or ["<task editable file>"]
    editable_paths = "\n".join(
        f"  - {context.task_dir}/{name}"
        for name in editable
    )
    fail_block = _diagnosis_fail_block(brief)
    marker = diagnose_marker(brief.plan_version)
    return (
        "Diagnose why the current optimization rounds are failing, then "
        "Write a structured report to a fixed path.\n\n"
        f"Target: dsl={context.target_dsl}, backend={context.backend}, "
        f"arch={arch}{brief.metric_line}\n"
        f"plan_version={brief.plan_version}\n\n"
        "Recent rounds (last 5 from history.jsonl):\n"
        f"{brief.recent_summary or '  (none settled yet)'}\n"
        f"{fail_block}Read these task files for context:\n"
        f"  - {context.task_dir}/reference.py\n{editable_paths}\n"
        f"  - {context.task_dir}/.ar_state/plan.md\n"
        f"  - {context.task_dir}/.ar_state/history.jsonl "
        "(focus on the last ~10 rounds; older entries are usually stale)\n\n"
        f"{_trace_analysis_block(context.task_dir, context.backend)}"
        "Hard constraints:\n"
        "  - Do NOT search git history (git log / git show / git grep) "
        "- per-round commits carry no keyword signal and burn tool calls.\n"
        "  - Do NOT Glob / Grep the wider codebase. The listed task files "
        "are the entire scope.\n"
        "  - Stop after at most 12 tool uses.\n"
        "  - Write tool may ONLY target the artifact path below. Do NOT "
        "Write editable source files, reference.py, plan.md, or anywhere else.\n\n"
        "REQUIRED OUTPUT - your final action MUST be a Write call to this "
        f"exact path:\n  {brief.artifact_path}\n\n"
        "The file body must contain ALL of:\n"
        "  - heading section 'Root cause' (one paragraph grounded in the "
        "FAIL summary / history)\n"
        "  - heading section 'Fix directions' (≤3 STRUCTURALLY different "
        "approaches: algorithmic / fusion / memory layout / data movement; "
        "NOT parameter tuning.)\n"
        "  - heading section 'What to avoid' (≤3 patterns to NOT repeat)\n"
        "  - the magic marker line on its own line at the end:\n"
        f"      {marker}\n"
        "Total ≤ 300 words across the three sections. The host validates "
        "path + marker + the three section names after this Task call returns; "
        "missing any element will force a retry."
    )


def _diagnosis_fail_block(brief: DiagnosisBrief) -> str:
    if not brief.fail_details:
        return "Last 3 FAILs: (none yet - use history.jsonl if needed)\n\n"
    return (
        "Last 3 FAILs (use these as the primary evidence):\n"
        f"{brief.fail_details}\n\n"
    )


def _diagnosis_action(brief: DiagnosisBrief, prompt: str) -> str:
    retry_note = ""
    if brief.attempts > 0:
        retry_note = (
            f"\nThis is DIAGNOSE attempt {brief.attempts + 1}/"
            f"{DIAGNOSE_ATTEMPTS_CAP}. The previous artifact was missing "
            "or malformed - re-issue Task and ensure the subagent ends its "
            "work with a Write of the marker line."
        )
    return (
        "[AR Phase: DIAGNOSE] consecutive_failures >= 3.\n"
        "Required action: call the Task tool with "
        "subagent_type='ar-diagnosis' and this EXACT prompt. Do not "
        "paraphrase. Do not add or remove constraints. Do not Edit, Write, "
        "or Bash before this Task call.\n"
        f"---BEGIN SUBAGENT PROMPT---\n{prompt}\n"
        "---END SUBAGENT PROMPT---\n"
        "Artifact contract: the host gates plan creation on a valid "
        f"{os.path.basename(brief.artifact_path)} (path + marker + 3 "
        f"sections). Up to {DIAGNOSE_ATTEMPTS_CAP} Task attempts are "
        "allowed; after that the gate is relaxed and you must write "
        "plan_items.xml directly (manual-planning fallback) before running "
        "create_plan.py - the DIAGNOSE phase still requires a new plan, "
        f"just without subagent help.{retry_note}"
    )


def _replan_guidance(context: GuidanceContext) -> str:
    progress = context.progress or {}
    remaining = progress.get("max_rounds", 0) - progress.get("eval_rounds", 0)
    plan_version = progress.get("plan_version", 0)
    retry_hint = ""
    if plan_version >= 2:
        retry_hint = (
            f"\nNote: plan_version is already {plan_version}. Before "
            "inventing entirely new ideas, scan history.jsonl for DISCARD "
            "items whose metric was close to best (within ~20%) - those "
            "ideas may compose differently now that the kernel's structural "
            "baseline has shifted. To revisit one, include it as a new item "
            "with a fresh pid and reference the prior pid in <desc>."
        )
    return (
        f"[AR Phase: REPLAN] All items settled. Budget: {remaining} rounds "
        f"left. Read .ar_state/history.jsonl. Analyze what worked/failed."
        f"{_trace_analysis_block(context.task_dir, context.backend)}\n\n"
        f"{_create_plan_instruction(context.task_dir, first_plan=False)}"
        f"{retry_hint}"
    )


def _finish_guidance(context: GuidanceContext) -> str:
    progress = context.progress or {}
    return (
        f"[AR Phase: FINISH] Done. Best {context.primary_metric}: "
        f"{progress.get('best_metric', '?')} "
        f"(baseline: {progress.get('baseline_metric', '?')}). Report "
        "auto-generated at .ar_state/report.md. Summarize for user; "
        "do not write any files."
    )


_GUIDANCE_HANDLERS = {
    INIT: _init_guidance,
    BASELINE: _baseline_guidance,
    PLAN: _plan_guidance,
    EDIT: _edit_guidance,
    DIAGNOSE: _diagnose_guidance,
    REPLAN: _replan_guidance,
    FINISH: _finish_guidance,
}


def get_guidance(task_dir: str) -> str:
    """Return the instruction for the task's current state-machine phase."""
    context = _guidance_context(task_dir)
    handler = _GUIDANCE_HANDLERS.get(context.phase)
    if handler is None:
        return f"[AR Phase: {context.phase}] Unknown phase."
    return handler(context)
