# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""aog-researcher brief construction (V3.7.11 vendor-strategy escalation).

Spawned when:
- await_optimizer/await_fused_optimizer plateau triggers researcher escalation
  (V3.8.8 + V3.7.11) — researcher investigates alternate vendor strategies that
  optimizer/probe couldn't see (private aclnn dlsym, fp64 internal compute,
  alternate adv_api primitives, magnitude-aware techniques).
- await_user_decision routes to await_researcher per user's user_decision.md
  (V3.8.5 #59 path).

Researcher mandate: produce `cann_strategy_inference.md` (analysis) AND
optionally `optimization_directive.md` (Kind-2 directive for kw respawn). If
the directive is written, state machine routes await_researcher → await_worker
automatically (V3.3.4 B-fix path).

Output handoff (one of):
- `→ orchestrator: research_done — algorithm = X, directive at workspace/{op}/optimization_directive.md, ready for kw-N`
- `→ orchestrator: research_partial — Trend: Y, gap: Z`
- `→ orchestrator: research_blocked — Blocker: <gap>`

V3.8.8 / 2026-05-05: also emit `→ orchestrator: PARTIAL_PERSIST — <evidence>`
when researcher exhausts the alternate-strategy search and confirms structural
ceiling — routes to finalize PARTIAL.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from briefs._common import (
    env_quirks_block,
    AscendCEnv, load_env,
    env_block, hard_floors_block, kb_manifest_block,
    schema_contract_block, fixed_layout_block, safety_block, g7_slug,
    self_introspection_block,
)


def build_researcher_brief(
    op: str,
    workspace: Path,
    *,
    lane: int,
    spawn_index: int,
    iter_cap_remaining: int,
    env: Optional[AscendCEnv] = None,
    handoff_from_prior_agent: Optional[str] = None,
    directive_text: Optional[str] = None,  # accepted for API symmetry, unused
    backend: str = "ascendc",
    plugin: Optional[object] = None,
) -> str:
    if env is None:
        env = load_env()
    from briefs._common import resolve_backend_from_env
    backend = resolve_backend_from_env(backend, env)
    if plugin is None:
        from briefs._common import resolve_plugin_for_brief
        plugin = resolve_plugin_for_brief(env, workspace=workspace, backend_override=backend)
    slug = g7_slug(op, "aog-researcher", spawn_index)

    sections = [
        f"{slug} — researcher spawn",
        "",
        env_block(env, lane=lane, op=op, workspace=workspace),
        "",
        env_quirks_block(env.target),
        "",
        hard_floors_block(workspace),
        "",
        kb_manifest_block(op, workspace=workspace, target=env.target),
        "",
        _researcher_phase_block(op, handoff_from_prior_agent, iter_cap_remaining, plugin=plugin),
        "",
        _researcher_output_block(op),
        "",
        schema_contract_block(),
        fixed_layout_block(),
        "",
        self_introspection_block(),
        "",
        safety_block(env),
        "",
        f"# ITER BUDGET\n\niter_cap_remaining = {iter_cap_remaining}. "
        "Researcher caps are tight (default 2). If you cannot identify a Kind-2 "
        "directive within budget, emit `research_partial` (don't force "
        "`research_done` without evidence).",
    ]
    return "\n".join(sections)


def _researcher_phase_block(
    op: str,
    handoff: Optional[str],
    iter_cap: int,
    *,
    plugin: Optional[object] = None,
) -> str:
    handoff_str = f"\n\n## Handoff from prior agent\n\n{handoff}" if handoff else ""
    user_decision_str = (
        f"\n\n## User decision (may be present)\n\nIf `workspace/{op}/user_decision.md` "
        "exists, read it for explicit research mandate from user "
        "(alternate-strategy hints, specific APIs to investigate)."
    )
    return f"""# PHASES (aog-researcher){handoff_str}{user_decision_str}

## Phase R-A: KB inventory + grep coverage map

Before drafting any new pattern number (P-P-XXX, OL-XX, EC-XX), inventory KB:
1. Glob `kb/**/*.md` and Read top-level files (KB_INDEX,
   ALWAYS_LOADED_RULES, SIMT_VS_SIMD_DECISION, PLATFORM_BUGS, ASCENDC_API_CATALOG,
   patterns/PATTERN_INDEX) plus relevant patterns/domains/*.md
2. Grep across full KB for each concept your candidates touch
3. Verify proposed pattern slot is unused — find current highest, +1
4. Check if equivalent pattern exists under different name → EXTEND existing entry

## Phase R-B: Vendor strategy investigation (this session's focus)

When invoked from V3.8.8 (probe verdict=requirement) or V3.7.11 (optimizer
plateau) or user_decision.md, the load-bearing question is "what does CANN do
strategically that we don't?". Investigate via:

1. **msprof on reference op** — run `msprof` on the CANN reference call,
   identify the kernel name + BlockDim + AIV/AIC ratio + SIMD/SIMT mode.
   Mandatory before declaring "vendor strategy unknown".
2. **Public adv_api search** — grep `find /data/cann_b103/cann-9.0.0/include
   -name "*.h" | xargs grep -l <relevant_keyword>` for missed primitives.
3. **dlsym dispatch inspection** — for ops where adv_api is missing, can we
   reach the vendor's binary impl via `dlsym(libascendc.so, "aclnnX*")`?
4. **hiascend.com docs** — search official docs for alternate API paths
   (e.g. for op#10 LayerNorm: LayerNormV2/V3/NormalizeV2).
5. **Public numerical-algorithm literature** (P0aac, 2026-05-06).
   When the gap is at the **numerical algorithm** level — transcendentals
   (tanh/sigmoid/exp/log), reductions, sorts, range reduction, polynomial
   approximation — the answer is rarely vendor-specific. Established math
   library implementations (Cephes, fdlibm, libm, ARM Compute Library,
   RLIBM, MKL) document the public algorithms vendors USE INTERNALLY.
   Concrete steps:

   a. Identify the numerical-algorithm class of the failure. Examples:
      - "AscendC `Tanh` 1-ULP-fp32 ceiling, fails on (1+tanh) cancellation"
        → range-reduction class
      - "fp32 sum-of-many regresses on long reductions"
        → Kahan / pairwise reduction class
      - "transcendental in saturation band fails sub-ULP"
        → Cephes-form reformulation class
   b. WebSearch the algorithm class. Keywords: `tanh fp32 range
      reduction polynomial`, `Cephes tanh implementation`, `Kahan summation
      compensated`, `IEEE 754 correctly rounded transcendental`. Look for
      arxiv.org / netlib.org / apple.com/library / GitHub references.
   c. Cross-check whether the algorithm can be implemented with the
      AscendC primitives whose precision is known fp32-grade (per OL-103
      §Refined-statement: `Exp`, `Reciprocal`, `Add`, `Mul` are fp32-grade;
      `Tanh` is fp32-grade-but-1-ULP-ceiling; `Sigmoid` is fp16-grade —
      do NOT use Sigmoid for sub-ULP).
   d. If a public algorithm exists AND maps to fp32-grade primitives,
      emit a Kind-2 directive citing the algorithm + its source.

   This step exists because R-B §1-4 frame the question as "what does
   CANN do?" — but for numerical-algorithm gaps, vendors USE the public
   literature. The Cephes-form `tanh(y) = 1 - 2/(exp(2y) + 1)` for large
   |y| eliminates `(1+tanh)` cancellation; that pattern is widely
   published. If we can't beat CANN at a transcendental, R-B §5 is more
   likely to find the answer than §1-4.

Source-isolation rule: do NOT read CANN op_impl source. msprof,
public headers, hiascend.com, dlsym, public math libraries (Cephes,
fdlibm, libm, ARM CL), and arxiv.org are all allowed.

## Phase R-C: Output

Write BOTH files (if directive feasible):
- `workspace/{op}/cann_strategy_inference.md` — analysis (mandatory)
- `workspace/{op}/optimization_directive.md` — Kind-2 directive for kw
  respawn IF an actionable strategy was identified (optional but preferred)

If no actionable strategy found AND probe verdict was `requirement`,
emit `→ orchestrator: PARTIAL_PERSIST — <evidence citing exhausted research>`
to ship at structural ceiling with full pipeline-exhaustion evidence."""


def _researcher_output_block(op: str) -> str:
    return f"""# REQUIRED OUTPUT

Mandatory: `workspace/{op}/cann_strategy_inference.md` with:
- §Vendor strategy hypothesis
- §msprof evidence (kernel name, BlockDim, ratio, SIMT/SIMD)
- §Public-API gap analysis (what we have vs what vendor uses)
- §Recommendation (Kind-2 directive OR PARTIAL_PERSIST verdict)

Optional: `workspace/{op}/optimization_directive.md` with:
- Mandatory KB reads (Phase A) for kw-N+1
- Algorithm sketch (pseudocode + UB layout)
- Primitive list (every API verified in ASCENDC_API_CATALOG.md)
- Vectorization plan
- Expected perf range
- Anti-cheating gates the verifier must enforce
- Determinism policy + rollback condition

## EXIT HANDOFF (one of)

- `→ orchestrator: research_done — algorithm = <name>, directive at workspace/{op}/optimization_directive.md, ready for kw-<N>`
  (Use when you wrote optimization_directive.md with a viable Kind-2 strategy.)

- `→ orchestrator: research_partial — Trend: <Y>, gap: <Z>, would benefit from extended budget for: <W>`
  (Use when you found leads but exhausted iter_cap before producing a directive.)

- `→ orchestrator: research_blocked — Blocker: <gap>`
  (Use when infra/access blocked investigation, e.g. msprof unavailable.)

- `→ orchestrator: PARTIAL_PERSIST — <evidence>`
  (V3.8.8: full pipeline exhausted — probe + researcher both unable to find
  alternate strategy. Routes to finalize PARTIAL with structural-ceiling
  evidence. Cite probe_report.md + cann_strategy_inference.md as the
  evidence chain.)"""
