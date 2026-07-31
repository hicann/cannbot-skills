# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""aog-kernel-worker brief construction.

Worker brief is the largest of the agent briefs. Per codex review B,
agent_dispatch.py was identified as the biggest underestimation in the
2-day plan — this implementation aims to cover all the cases in
ascendc-op-gen/SKILL.md §"WORKER_BRIEF (first invocation)".

Sections:
- env_block (lane / target / host / paths) — from _common.py
- hard_floors_block (preserve baseline) — from _common.py
- kb_manifest_block (op-class taxonomy lookup) — from _common.py
- safety_block (CLAUDE.md rules) — from _common.py
- schema_contract_block (V3.8.x canonical) — from _common.py
- worker-specific: G1 marker handling, analysis.md pre-seed, exit handoff
  options, iter cap remaining

DEBT-201 (2026-07-06): the FA-class cluster, the forced-architecture shared
leaf, and the port_a3 Phase A-E builders were extracted into sibling modules to
keep this file <1000 lines with functional cohesion:
  - kw_brief_fa.py          — FA-class predicates + template-assembly + backward stitch
  - kw_brief_shared.py      — _forced_architecture_block (shared leaf, no cycle)
  - kw_brief_pa3_phases.py  — port_a3 Phase A/B/C body builders + context (leaf)
  - kw_brief_port_a3.py     — port_a3 orchestrator + Phase D/E/budget bodies
The four FA backward symbols + the port_a3 orchestrator are re-imported below so
`from briefs.kw_brief import ...` stays stable for external callers
(BackwardPlugin, the golden test). Behavior is byte-identical (prompt-template
refactor; golden-locked).
"""
from __future__ import annotations
import logging

from pathlib import Path
from typing import Optional

from briefs._common import (
    env_quirks_block,
    AscendCEnv, load_env,
    env_block, hard_floors_block, kb_manifest_block,
    schema_contract_block, fixed_layout_block, safety_block, g7_slug,
    self_introspection_block, rollback_context_block,
    _detect_forced_architecture, _FORCED_ARCH_TAGS,
)

# DEBT-201 sibling-module re-imports. These keep the public `briefs.kw_brief`
# surface stable after the god-file decomposition:
#   - the 4 FA-class symbols are imported by BackwardPlugin
#     (plugins/backward/__init__.py) via `from briefs.kw_brief import ...`
#   - `_forced_architecture_block` is used by `_phase_instructions_block` below
#   - `_fa_class_template_assembly_block` is used by `_phase_instructions_block`
#   - `_port_a3_phase_instructions_block` is used by `_phase_instructions_block`
#     and directly by test_kw_brief_port_a3_golden.py
from briefs.kw_brief_shared import _forced_architecture_block  # noqa: F401
from briefs.kw_brief_fa import (  # noqa: F401
    _is_fa_class_backward,
    _fused_fa_backward_requested,
    _fa_class_template_assembly_block,
    _fa_assembly_intro_block,
    _fa_assembly_recipe_block,
    _fa_assembly_compile_block,
    _fa_assembly_verify_hard_block,
    _fa_class_backward_stitch_block,
    _fa_class_backward_multilaunch_block,
    _fa_ge_host_gen_block,
)
from briefs.kw_brief_port_a3 import _port_a3_phase_instructions_block  # noqa: F401
# `_port_a3_cube_class_mix_block` is imported directly by
# test_port_a3_cube_mix_brief.py — keep it on the public `briefs.kw_brief` surface.
from briefs.kw_brief_pa3_phases import _port_a3_cube_class_mix_block  # noqa: F401


def _branched_from_addendum(workspace: Path) -> str:
    """Describe a provenance-tracked prior implementation used as a branch base.

    The seed is advisory migration context.  It never replaces the current
    arch22 source-NPU truth or the target-NPU verification required for this op.
    """
    import json as _json

    marker = workspace / ".branched_from.json"
    if not marker.is_file():
        return ""
    try:
        metadata = _json.loads(marker.read_text())
    except Exception:
        return ""
    if not isinstance(metadata, dict):
        return ""

    parent = metadata.get("parent_op", "?")
    seed_dir = metadata.get("seed_dir", "branched_from_kernel")
    similarity = metadata.get("similarity")
    return (
        "# BRANCH BASE (DEBT-203 — advisory migration context)\n\n"
        f"A provenance-tracked implementation for **{parent}** "
        f"(similarity {similarity}) is available at `{seed_dir}/`. It is a "
        "starting point for research and adaptation: it is NOT a submission and "
        "never replaces truth.\n"
        "- Record every file read from it in `reference_manifest.jsonl`.\n"
        "- ADAPT shapes, dtypes, semantics, scheduling, and tiling to THIS op.\n"
        "- Author the result into `kernel/` and RE-VERIFY against THIS op's fresh "
        "arch22 source-NPU truth and on the target NPU.\n"
        f"- Do not submit `{seed_dir}/` unchanged; ordinary precision, determinism, "
        "provenance, and finalization gates still apply.\n"
    )


def build_worker_brief(
    op: str,
    workspace: Path,
    *,
    lane: int,
    spawn_index: int,
    iter_cap_remaining: int,
    env: Optional[AscendCEnv] = None,
    directive_text: Optional[str] = None,
    handoff_from_prior_agent: Optional[str] = None,
    backend: str = "ascendc",
    plugin: Optional[object] = None,
) -> str:
    """Build a complete WORKER_BRIEF for aog-kernel-worker spawn.

    Args:
        op: workspace dir name (e.g. "22_Nonzero" or "9_topktopp")
        workspace: absolute path to workspace dir
        lane: 0/1/2 (real NPU IDs on A5)
        spawn_index: 1-based counter for this op's worker spawns
        iter_cap_remaining: from state_executor.iter_cap - state_executor.iter_count
        env: parsed AscendCEnv; if None, loads from workspace/.ascendc_env
        directive_text: if respawn from probe/optimizer, the directive content
        handoff_from_prior_agent: handoff text from prior agent's exit

    Returns:
        complete prompt body (str) ready to pass to agent_transport
    """
    if env is None:
        env = load_env()

    # P131: when backend arg defaults, inherit from env.backend (set from
    # .ascendc_env BACKEND= or by orchestrator main() before brief build)
    from briefs._common import resolve_backend_from_env
    backend = resolve_backend_from_env(backend, env)

    # Phase 2 plugin-as-param (Q3 main agent): auto-resolve if caller didn't.
    if plugin is None:
        from briefs._common import resolve_plugin_for_brief
        plugin = resolve_plugin_for_brief(env, workspace=workspace, backend_override=backend)

    slug = g7_slug(op, "aog-kernel-worker", spawn_index)

    # P0abe (2026-05-07): if the prior spawn was rolled back by the finalize
    # gate, surface the rollback reason at the TOP of the brief — before any
    # other section the worker might fixate on. Empty when no rollback history
    # (cold start path).
    rb_ctx = rollback_context_block(workspace)
    rb_block = rb_ctx + "\n" if rb_ctx else ""

    # C2 backward-perf (OL-200) — op_class-driven. Surfaces the
    # MIX_AIC cube/vec software-pipelining whitebox-check for gradient ops
    # in backward generation and arch22 -> arch35 migration.
    # Empty for forward ops so their briefs stay byte-identical.
    bw_c2_block_str = _backward_perf_c2_block(workspace)

    branch_base = _branched_from_addendum(workspace)
    sections = [
        f"{slug} — kernel-worker spawn",
        "",
        *([branch_base, ""] if branch_base else []),
        rb_block,
        env_block(env, lane=lane, op=op, workspace=workspace),
        "",
        env_quirks_block(env.target),
        "",
        hard_floors_block(workspace),
        "",
        _reference_provenance_block(op),
        "",
        kb_manifest_block(op, workspace=workspace, target=env.target),
        "",
        _phase_instructions_block(op, workspace, iter_cap_remaining, directive_text,
                                  handoff_from_prior_agent, env=env, backend=backend, plugin=plugin),
        "",
        # C2 backward-perf block — spread iff non-empty so forward-op briefs
        # stay byte-identical (matches phase_e / gate_spec convention).
        *([bw_c2_block_str, ""] if bw_c2_block_str else []),
        schema_contract_block(),
        fixed_layout_block(),
        "",
        self_introspection_block(),
        "",
        safety_block(env),
        "",
        _exit_handoff_block(),
        "",
        "# G1 MARKER\n\n"
        "Write `.kernel_worker_active` marker at workspace start; remove on exit.\n"
        "This marker prevents concurrent kernel-edit conflicts when probe/optimizer "
        "also spawn.\n",
        "",
        f"# ITER BUDGET\n\niter_cap_remaining = {iter_cap_remaining}. "
        "If you exhaust this, exit with handoff to orchestrator "
        "(NOT keep iterating).",
    ]
    return "\n".join(sections)


def _reference_provenance_block(op: str) -> str:
    """Require an auditable manifest of every generation input."""
    return (
        "# REFERENCE PROVENANCE — MANDATORY (record every file you read)\n"
        "\n"
        "For EVERY file you open as reference during generation (Read / cat / grep /\n"
        "head / sed on any source, spec, KB pattern, header, or archive), IMMEDIATELY\n"
        f"append ONE JSON line to `workspace/{op}/reference_manifest.jsonl`:\n"
        "```\n"
        '{"path": "<repo-rel or abs path>", "category": "arch22_source|kb_template|'
        'kb_doc|public_api_doc|prior_archive|target_source|prestage_manifest|other", '
        '"reason": "<why, 1 phrase>", '
        '"phase": "A|B|C|D|E"}\n'
        "```\n"
        "- This is a GENERATION-PROVENANCE record, read by the orchestrator + owner to\n"
        "  verify how the arch22 -> arch35 implementation was produced.\n"
        "- The detected arch22 source and a fresh source-NPU capture remain the truth.\n"
        "  Target archives, target-source prior art, and DEBT-203 branch bases are\n"
        "  allowed only as advisory migration context; they never replace truth or\n"
        "  target-NPU verification.\n"
        "- A pre-staged target file is trusted as tracked context only when its current\n"
        "  SHA256 matches `.upstream_prestaged.json`. Log both the manifest and the\n"
        "  staged file. Raw, untracked byte-copy into the deliverable is forbidden.\n"
        "- Do NOT fabricate or omit entries: the agent stream-log records your actual\n"
        "  tool-calls; a manifest that disagrees with it is a fraud signal.\n"
        f"- Self-check before EXIT: `workspace/{op}/reference_manifest.jsonl` exists and\n"
        "  is non-empty.\n"
    )


def _backward_perf_c2_block(workspace: Path) -> str:
    """C2 backward-perf guidance (KB OL-200), injected when the op is a
    backward/gradient op per `plugins.base.is_backward_class`.

    Gated only on op_class (via `schema_norm.detect_op_class`), so it fires for
    backward generation and backward-class arch22 -> arch35 migration. Returns "" for forward
    ops so their briefs stay byte-identical.

    Companion to the #273 contract-completeness gate (which uses the same
    op_class signal for the backward-completeness verdict in B2/B3). This is
    why `is_backward_class` is NOT a pure label — it drives real brief
    content here, mirroring the `is_fa_class` block above.
    """
    if workspace is None:
        return ""
    try:
        from schema_norm import detect_op_class
        from plugins.base import is_backward_class
    except Exception:
        return ""
    op_class = detect_op_class(workspace, {})
    if not is_backward_class(op_class):
        return ""
    return (
        "# BACKWARD-PERF (C2 — KB OL-200, MIX_AIC pipelining)\n"
        "\n"
        f"op_class={op_class!r} 是反向（gradient）算子（判定见 "
        "`plugins.base.is_backward_class`）。在 MIX_AIC 单元上，精度正确并不等于性能正常"
        "（KB **OL-200**）：\n"
        "\n"
        "- `while(sched.HasNext()){ cube(); vec(); }` 这种 lockstep 循环精度正确，"
        "但让 cube(AIC) 与 vector(AIV) 各自约一半时间空闲。\n"
        "- 性能正常的形态会把 cube/vec 按 taskId 错位做软件流水：例如 Vec 在 `i%K`、"
        "Cube 在 `(i-1)%K`（LIG V220 用 4 级 `i%4` ring）。\n"
        "- **调 tile 之前先做 whitebox 检查**：把生成的 `Process()` 与参考的错位调度对比——"
        "cube 与 vec 之间没有 `i%K` 偏移即为流水塌缩（OL-200 反模式）。先怀疑这一点，"
        "再去怀疑 tile 尺寸。\n"
        "- 如果是移植已知良好的调度，参考配方见 "
        "`docs/handovers/LIG_V220_PIPELINE_SCHEDULE_SPEC.md`。\n"
        "\n"
        "对 MIX 反向算子，精度通过是必要而非充分条件：一个 34/34 正确的 kernel 若流水塌缩，"
        "性能仍可能差约 2 倍（实例：LIG backward，a5_ops `8d8c5538`）。"
    )


def _general_simd_regbase_block(
    workspace: Optional[Path], env: Optional["AscendCEnv"]
) -> str:
    """OL-245 (2026-06-23): general arch35-SIMD regbase steer for the NON-FA
    cold-start path. Returns "" unless target is a5/arch35 AND the op is not
    FA-class (FA already gets its own regbase steer via the wholeport
    template-assembly block, which short-circuits above this path).

    Empty for non-a5 targets (a3/a2 use a different intrinsic surface) so those
    briefs stay byte-identical. Empty for FA-class ops (defensive; FA never
    reaches the cold-start path). When it fires, instructs kw to CONSIDER regbase
    (MicroAPI RegTensor compute chain) over Membase/LocalTensor for a vector/SIMD
    compute chain — carrying the OL-245 amortization DECISION RULE (regbase wins
    for large per-call work, loses for many-small-shallow VF calls via the
    ~0.42µs/call __VEC_SCOPE__ overhead → measure/judge by granularity), citing
    SIMD_DEVELOPMENT_REFERENCE §0 + the migration guides. Fixes the bias (kw no
    longer defaults to Membase blind) WITHOUT over-correcting (no blind regbase
    on fine-grained small-call ops).

    Root cause (lead-verified 2026-06-23): regbase guidance was SILOED in the
    FA-class wholeport templates + migration/ guides; the general SIMD-gen path
    (SIMD_DEVELOPMENT_REFERENCE + this brief's general recipe) was Membase-only,
    so kw defaulted to Membase for every general SIMD op — selective_scan
    fwd+bwd were 100% Membase → ~15× redundant vector work, ~33× off the
    roofline floor (2490µs vs ~75µs).
    """
    # Target gate: arch35/A5 only. a3/a2 have a different (low-level intrinsic)
    # vector surface — the arch35 MicroAPI RegTensor regbase API is A5-specific.
    if env is None or getattr(env, "target", "") != "a5":
        return ""

    # FA-class defensive exclusion: FA ops get the wholeport-template regbase
    # steer and short-circuit above the cold-start path; if an FA op somehow
    # reaches here, do not double-steer.
    op_class = ""
    op_name = ""
    if workspace is not None:
        op_name = workspace.name
        try:
            import json as _json
            cls_p = workspace / "op_classification.json"
            if cls_p.is_file():
                cls = _json.loads(cls_p.read_text())
                tags = cls.get("op_class_tags") or []
                op_class = " ".join(tags) if tags else ""
        except Exception:
            op_class = ""
    try:
        from plugins.base import is_attention_named as _is_fa_named
        from plugins.base import is_fa_class as _is_fa_tag
        if _is_fa_named(op_name) or _is_fa_tag(op_class):
            return ""
    except Exception as error:
        logging.getLogger(__name__).debug(
            "Recoverable operation failed.", exc_info=error
        )

    return (
        "# arch35 (A5) SIMD: CONSIDER Regbase over Membase — decision rule (OL-245)\n"
        "\n"
        "If this op is a **vector / SIMD compute op** (an elementwise / fused-vector / "
        "scan / normalization chain — i.e. ≥2 dependent vector ops like Cast→Mul→Add), "
        "**CONSIDER authoring the compute chain in regbase (MicroAPI)** instead of "
        "Membase/LocalTensor — do NOT default to Membase without weighing regbase. "
        "**But regbase is NOT an unconditional win — apply the OL-245 amortization rule:**\n"
        "- **Regbase WINS** when each `__VEC_SCOPE__` call does LARGE, amortized per-call "
        "work (a long/deep reg-resident chain over a big contiguous span) — the saved UB "
        "round-trips dominate. (Isolated 8-op fp32 chain measured 1.26× faster.)\n"
        "- **Regbase LOSES** for **many small/shallow VF calls** — the `__VEC_SCOPE__` "
        "entry/exit overhead (~0.42µs/call) then exceeds the round-trip saving. "
        "(selective_scan fwd build = ~30720 calls on 64-tile 3-op chunks → regbase 1.27× "
        "SLOWER; loop iter-2, OL-245.) If your hot chain is per-tiny-chunk / per-row "
        "fine-grained, Membase can win — **measure the hot chain BOTH ways, or judge by "
        "per-call granularity, before committing.**\n"
        "\n"
        "- **Membase cost (why):** the LocalTensor `Add(dst,a,b,count)` / `Mul` / `Cast` "
        "style READS operands from UB, WRITES each intermediate back to UB, and the next "
        "op RELOADS it from UB — a redundant UB store+reload **plus a `PipeBarrier<PIPE_V>` "
        "per dependent step**. For an N-op chain that is ~N redundant UB round-trips + N "
        "barriers of pure overhead (selective_scan fwd+bwd were 100% Membase → ~15× "
        "redundant vector work, ~33× off the roofline floor).\n"
        "- **Regbase (do this):** `__VEC_SCOPE__` + `AscendC::MicroAPI::RegTensor<float>` — "
        "`LoadAlign` UB→reg ONCE at the chain head, compute reg→reg (`Mul`/`Add`/...), "
        "`StoreAlign` reg→UB ONCE at the tail. The intermediate round-trips and per-step "
        "barriers disappear.\n"
        "- **Membase is fine for** trivial single-op (no dependent successor), "
        "bitwise/type-punning cases where the LocalTensor high-level API is the documented "
        "A5 path, AND (per the rule above) **fine-grained many-small-shallow-VF-call chains** "
        "where the ~0.42µs/call regbase overhead would dominate.\n"
        "\n"
        "Read FIRST: `kb/target/ascendc/SIMD_DEVELOPMENT_REFERENCE.md` "
        "**§0** (the MicroAPI shape + the Membase-vs-Regbase rationale) and the full guides "
        "`target/ascendc/migration/reg-base-vector/Reg矢量计算编程.md`, "
        "`target/ascendc/migration/l2-register-based-guide.md`, "
        "`target/ascendc/migration/l5-register-based-guide.md`.\n"
        "\n"
        "(NOT an architecture override — this is the SIMD *authoring style* within the SIMD "
        "architecture, not a SIMT-vs-SIMD decision. If the architecture is FORCED above, "
        "honor it; this steer only shapes HOW you write the SIMD compute chain.)"
    )


def _phase_instructions_block(
    op: str,
    workspace: Path,
    iter_cap_remaining: int,
    directive_text: Optional[str],
    handoff_from_prior: Optional[str],
    env: Optional["AscendCEnv"] = None,
    backend: str = "ascendc",
    *,
    plugin: Optional[object] = None,
) -> str:
    """Build instructions for the two explicit AscendC customer modes."""
    declared_mode = getattr(env, "opgen_mode", "") if env is not None else ""
    if backend != "ascendc" or declared_mode not in {"port_a3_to_a5", "backward"}:
        raise RuntimeError(
            "unsupported worker route: expected AscendC port_a3_to_a5 or "
            f"backward, got backend={backend!r}, mode={declared_mode!r}"
        )

    # FA-class +
    # no directive = inject the template-assembly recipe. FA-class ops route to
    # the standard kw worker, which assembles the arch35 op from arch22 + KB
    # templates. FA is an op-class concern shared by both supported modes.
    if not directive_text:
        # DEBT-208: pass the build target so the FA MIX cross-core-sync block
        # carries only the KB cards whose own `applies_to: soc=` covers it (PB-34
        # is V220-only and INVERTS the KB's advice on A5). `env` may be None in
        # unit contexts — fall back to the `a5` default the composer declares.
        _fa_emit = _fa_class_template_assembly_block(
            op, workspace, target=getattr(env, "target", None) or "a5"
        )
        if _fa_emit is not None:
            return _fa_emit

    # NOTE: directive_text wins over opgen_mode. When a probe / optimizer
    # respawns the worker with a specific directive ("fix BF16 cast in line
    # 234"), the directive is the tactical command — applies regardless of
    # cold-start vs. port mode. Port-mode prose is only for the first spawn
    # (no prior directive). Order checked by W5 test
    # test_directive_text_overrides_port_mode.
    if directive_text:
        return f"""# DIRECTIVE FROM PRIOR AGENT

{directive_text}

# PHASES

A. KB Manifest LOAD (per section above)
B. Read prior PROGRESS.md / verification.json / probe_report.md (if any)
C. Apply the directive above; build via deploy_to_npu_lane.sh; verify
   precision floors; measure perf
D. If directive succeeds → exit `→ orchestrator: done — <one-line summary>`
   If directive infeasible → exit with REJECTED reason + actionable handoff
   If precision regresses → REVERT to prior baseline + log reason"""

    # W5 (2026-05-12, ROADMAP §1.5): arch22→arch35 port mode brief. Reads
    # workspace/a3_reference_runnable.json (emitted by phase_o25_a3_ref W4)
    # to surface aclnn entry path + peer_op_dependencies. The brief shape
    # diverges meaningfully from the generic cold-start path — different sources,
    # writing instructions, different archive layout, cross-op router check.
    if env is not None and getattr(env, "opgen_mode", "") == "port_a3_to_a5":
        common_perf = _general_simd_regbase_block(workspace, env)
        port_brief = _port_a3_phase_instructions_block(
            op, workspace, iter_cap_remaining, env
        )
        return f"{common_perf}\n\n{port_brief}" if common_perf else port_brief

    # Backward owns its complete phase block through the explicit plugin.
    if plugin is not None:
        override = plugin.kw_brief_phase_block(
            op=op, workspace=workspace,
            iter_cap_remaining=iter_cap_remaining,
            directive_text=directive_text,
            handoff_from_prior_agent=handoff_from_prior,
            env=env,
        )
        if override is not None:
            return override

    raise RuntimeError(
        "backward plugin did not provide its required worker phase block"
    )


def _exit_handoff_block() -> str:
    return """# EXIT HANDOFF OPTIONS

Write your handoff line to **PROGRESS.md tail ONLY**. Do NOT write to
`state_transitions.jsonl` — the orchestrator records that file from your
final stdout text. Double-writing causes routing ambiguity (codex review
2026-05-04 finding #8).

## PRE-DONE FILE-EXISTENCE CHECKLIST + HANDOFF CATALOG

**FULL CATALOG: `kb/shared/GATE_CONTRACT.md` §kw-exit-handoff** (DEBT-114).

TL;DR pre-done check (BEFORE `→ orchestrator: done`):
```bash
ls workspace/{op}/pass_a_runner.py workspace/{op}/pass_b_runner.py
python3 src/scripts/orchestrator/check_verification_schema.py workspace/{op}/verification.json
```
Schema needs `precision.pass_a.tier1_pass` + `total`; P0cc needs
`tier1_pass_inclusive` for PASS_WITHIN_TOLERANCE/PARTIAL_PASS*.
`verify.py` is NOT a substitute for `*_runner.py` (different consumers).
Incident 2026-05-21 7_Sum: missing runners → rollback (~5-10min lost).

**Runner invocation (P0aba)**: `pass_a/b_runner.py` bare-invocable (`kernel_dir` `nargs="?"`) + canonical JSON BOTH last-stdout-line AND `--json` — full in GATE_CONTRACT.md §MANDATORY-artifacts item 5.

Allowed (TL;DR — see GATE_CONTRACT.md):
- `→ orchestrator: done` — precision PASS + det PASS
- `→ orchestrator: PARTIAL_PERSIST` — Tier-2 per OL-109
- `→ orchestrator: structural_rewrite_needed — <reason>` (§4.3): scope spans
  **≥2 of** {algorithm design, tile structure decision, primitive selection,
  cross-core sync discipline} + ≥1 signal (pass_count baseline / ≥2 kernel
  files / ≥2 kernel phases / new tiling). E.g. FA fused-attention=yes;
  foreach_sqrt single-axis=NO stay PARTIAL_PERSIST.
- `@aog-precision-probe` / `@aog-kernel-optimizer` / `@aog-fused-optimizer` / `@aog-determinism-analyzer`
- `→ orchestrator: await_user_decision`

DO NOT write:
- `→ orchestrator: PARTIAL_PERF_STRUCTURAL_CEILING` — RESERVED ko/fo
- `→ orchestrator: done` if perf < 0.6× (unless verification.json perf N/A)
- Non-YAML states / state_transitions.jsonl entries

**HANDOFF-LINE FORMAT (REQUIRED)**: the verdict line MUST be the LAST non-empty line, carry EXACTLY ONE
handoff token, and contain NO inline `@aog-X` mention (the orchestrator reverse-scans last-match-wins → a
`→ orchestrator: <verdict>` line trailing a `@aog-X` recommendation is mis-extracted as the `@aog-X` →
spurious `await_worker → abort`). Routing reasoning goes EARLIER, never on/after the verdict line. See
GATE_CONTRACT.md §kw-exit-handoff for the worked example."""
