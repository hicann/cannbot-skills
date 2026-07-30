---
name: aog-knowledge-maintain
description: >
  Review AscendC runtime findings and stage user-local c-tier entries; audit the
  release-owned bundled knowledge base without mutating it.
  Four modes: "update" (fast), "scan" (thorough), "validate" (single entry), "learn" (web scraping).
  Use when an operator run produces new knowledge or the AscendC KB needs maintenance.
argument-hint: >
  Mode 1 (default): knowledge_update_path=workspace/{op}/knowledge_update.md
    → Review + stage c-tier entries. Fast, runs after each op.
  Mode 2: --scan
    → Full KB scan: dedup all files, validate known issues against A5,
      check for obsolete entries, propose release promotions. Slow, run explicitly.
  Mode 3: --validate PB-9
    → Test a specific known issue (EC/PB/OL) against current A5 environment.
  Mode 4: --learn [url]
    → Scrape AscendC best practices from hiascend.com, extract optimization patterns,
      compare with existing KB, stage c-tier findings. Default URL: SIMD optimization overview.
  Options: mode=auto|review (default auto)
context: inline
---

# AscendC Knowledge Base Maintenance


> **Anti-pressure protocols (MANDATORY, load first)**: read `${CLAUDE_PLUGIN_ROOT}/kb/shared/ANTI_PRESSURE_PROTOCOLS.md` before any decision. P1–P8 are LLM-pressure drift modes (user-watching, context-filling, batch-throughput, simple-op assumption, failure discomfort, infrastructure friction, closure desire, tool-path-of-least-resistance) that override technical rules under load. Cite the relevant Px at every high-leverage decision point (spawn / done / PARTIAL / skip-verify).

## Scope

The bundled files below are the **read-only b-tier baseline** used for semantic
review and dedup. They are not runtime write targets:

| File | Entry format | ID prefix |
|------|-------------|-----------|
| `ERROR_CORRECTIONS.md` | `### EC-N: title` | EC- |
| `OPERATIONAL_KNOWLEDGE.md` | `## OL-N: title` | OL- |
| `PLATFORM_BUGS.md` | `### PB-N: title` | PB- |
| `patterns/PATTERN_INDEX.md` | Table row with ID | P-P |
| `patterns/unverified/candidates.md` | Candidate patterns | P-CAT-N etc |

Do NOT modify any file under `${CLAUDE_PLUGIN_ROOT}/kb/` in an installed
workflow, including KB_INDEX.md, promotion markers, or the files above.

### Runtime persistence contract (mandatory; overrides historical examples below)

- Mode 1 and Mode 1-batch perform semantic review only. Emit the exact
  workspace intake path supplied by the orchestrator as JSON:
  `{"schema_version":1,"entries":[...]}`.
- Each entry may contain only `kind`, `claim`, `scope`, `key`, `evidence`,
  `provenance`, and `meta`. `kind` is one of `positive_pattern`,
  `anti_pattern`, or `experience`. Emit `entries: []` when nothing passes.
- Do **not** write directly to `ASCENDC_PORT_USER_KB`, its default root, or
  `.kb_merged`. The orchestrator validates the intake, calls
  `Arbiter.write(entry, "customer")` through `CannbotCProvider`, and creates
  the marker only after durable admission.
- All operator-run knowledge remains user-local c-tier
  (`ASCENDC_PORT_USER_KB`, default `~/.ascendc-port/user_kb`). Promotion into
  bundled b-tier is a separate release-review operation and is never run by
  operator finalize.
- Historical prose or pseudocode below that says append/edit/promote a
  canonical bundled file describes the semantic classification destination,
  not permission to mutate b-tier. In an installed workflow, represent that
  decision as a c-tier intake entry or a workspace audit recommendation.

## Safety Rules

1. **Max 10 c-tier candidate entries per invocation**. Report remainder.
2. **NEVER delete or edit bundled b-tier content**.
3. **One intake file per workspace**; never write provider storage directly.
4. **NEVER change entry IDs** — they're permanent references.
5. **NEVER auto-resolve contradictions** — flag for user.
6. **KB language policy (2026-04-23): English only for code-facing KB.** All new c-tier claims **MUST be written in English prose**. Chinese keyword-in-quotes for user-facing strings is fine ("错误码 507035" as a verbatim symptom). When running Mode 1 / Mode 2 / Mode 4, verify proposed content is English before emitting it.

---

## Mode 1: Update (default — after each op)

Triggered by: `knowledge_update_path=workspace/{op}/knowledge_update.md`

**P0ff (2026-05-23, owner directive 20:48Z customer-side reproducibility)**:
This skill MUST ALSO read `workspace/{op}/kb_draft_from_user_decision.md`
when present. It is auto-extracted by the orchestrator (`_extract_kb_draft_from_user_decision`
in `src/scripts/orchestrator/orchestrator.py`) from a session's user_decision.md
god-mode-researcher directive. Treat each candidate entry in that file as a
PROVISIONAL c-tier candidate from owner-agent's white-box intervention. Apply
the same semantic admission gates (mechanical scanners C34a/C34b/C34c/C35 +
codex hook + cross-op transferability + applies_to scope tag per step 2.5).

Provenance rule (P0ff): every entry staged from kb_draft_from_user_decision.md
MUST carry the source anchor `derived from {op}/user_decision.md session
{YYYY-MM-DD}` in its provenance — so future readers + audits can trace the
strategic intervention.

Cross-source validation (P0ff): when both `knowledge_update.md` (worker
emitted) AND `kb_draft_from_user_decision.md` (orchestrator extracted from
owner directive) are present, the worker's `## Session-directive cross-check`
section in knowledge_update.md tells you whether to:
  - **CONFIRM** (`rule_confirmed: yes`): stage the kb_draft entry as-is
  - **EXTEND** (`rule_confirmed: extended`): merge worker's `additional_applies_to_found`
    and `rule_caveats` into the candidate before staging
  - **REFUTE** (`rule_confirmed: no`): DO NOT stage it; instead record a
    `KB-correction-pending` item in the workspace intake audit for owner review

If kb_draft_from_user_decision.md is in HEURISTIC mode (no structured
`kb_distillation:` block in original user_decision.md — the file body will
say `mode=heuristic`), DO NOT stage it automatically. Record its path and
digest in the workspace intake audit as a `P0ff heuristic-mode candidate
(needs owner re-write as structured block)` and omit it from the admission
payload. Mode 1 should NOT silently lose heuristic-mode candidates; they stay
in the workspace for owner re-distillation.

### Steps:
1. Read knowledge_update.md AND kb_draft_from_user_decision.md (P0ff).
   If kb_draft present + structured-mode: merge candidate entries into the
   step-2 generalization-check pipeline. If heuristic-mode: record it in the
   workspace audit per above + proceed with knowledge_update.md only.
1a. **Read `workspace/{op}/failures_ledger.md` if present** (new 2026-04-17).
    It has up to 3 sections: `## Build failures`, `## Precision failures`,
    `## Performance hypotheses`. Entry format:
    `- iter/opt N | {category} | {pattern_id or "novel"} | {one-line summary}`
    Process each entry:
      - **pattern_id matches existing KB entry** (EC-/PB-/OL-/P-P-N):
        - Read the entry, then emit a c-tier evidence refinement whose
          provenance names that b-tier ID; never edit the bundled entry.
        - Purpose: accumulate validation evidence for pattern reliability
      - **"novel" entries**:
        - Classify as an unverified c-tier candidate, or as EC/PB/OL material
          for the semantic checks in step 2. Provider IDs are allocated by the
          orchestrator; do not allocate bundled serial IDs.
        - **C-tier admission pipeline triggers**: kb_manager runs steps 2-5,
          emits a reviewed intake candidate, and lets the orchestrator's
          deterministic gate admit it to user-local c-tier. This is not
          permission to promote or mutate bundled b-tier.
      - **Performance REVERT entries**: still record evidence, but flag with
        `(REVERT: hypothesis invalidated)` so future optimizer sees anti-pattern
2. **Generalization check** — for each new entry, assess specificity. **This is the
   primary KB-quality gate** (replaces the dropped script-based SC3 from 2026-04-28
   per user direction "use a skill to update and maintain quality is way better than
   scripts"). The skill agent IS the semantic reviewer; if you skip step 2 the gate
   has nothing.

   **Design rule**: generalize the principle, KEEP concrete op/code pieces as
   illustrative anchors. Pure-abstract entries get misread by downstream weaker
   models; pure-specific entries don't transfer to the next op. Combine: abstract
   title + abstract principle in body + small concrete pieces (3–5 line snippets,
   one or two named-op references) that anchor the abstraction. **Anchor pieces,
   not full code listings** — do NOT paste a whole kernel.h or pybind.cpp file
   into an OL/P-P body. Record a release-review recommendation for full
   templates rather than moving them into bundled `patterns/domains/<domain>.md`.

   Layered shape for an OL/P-P entry:
   - **Title** — principle-first. No op-class colon-scoping (`<Op>:`), no
     "<Op>-unit primer" / "<Op> playbook" framing.
   - **Body — abstract principle**: the rule / technique / anti-pattern stated
     in terms of API surfaces / hardware behaviors / numerical properties /
     pipeline stages — words that name the *class of situations* the lesson
     applies to.
   - **Body — concrete anchor (small piece)**: 3–5 line code snippet OR a
     specific API signature OR a named-op evidence reference. Just enough that
     a weaker model reading the abstract principle can pin it to something
     real. NOT a full kernel listing.
   - **Body — `Evidence` sub-bullet**: where the principle was first observed
     + dates + concrete results ("op#X 2026-MM-DD: 50/50 PASS, 1.45× median").
   - **Body — `Other instances (predicted)` sub-bullet**: forward-looking
     generality — what other ops/regimes this applies to. Forces the author to
     check the abstraction is useful beyond the case that surfaced it.

   **CRITICAL — generic-principle placement (V3.4.5, 2026-04-29)**:
   When an OL/PP entry already has multiple Evidence rows AND a NEW evidence
   adds a generalizable decision rule (e.g., "Path A vs Path B" choice
   criterion, "trigger condition X vs Y" classifier, "applicable when Z"
   scope test), the rule MUST be lifted into a top-of-entry section
   (typically immediately ABOVE the Evidence list, with its own bold
   sub-heading like `**Mitigation paths (decision rule)**` or `**Trigger
   classifier**` or `**Scope condition**`). DO NOT bury a generic decision
   rule inside one Evidence row's body — future agents scanning by
   keyword see only the Evidence anchors and miss the rule entirely.

   **BAD pattern** (what triggered this rule, 2026-04-29):
   - Evidence #5 of OL-89 had a `**Generalized lesson**:` sub-paragraph at
     the bottom of an op#30-NMS-anchored evidence block. The lesson
     ("Path B is suitable when reference's non-determinism comes from a
     single torch.<sort>/argsort/argmin/argmax/topk call") was generic but
     visually scoped to NMS. User caught it: "you mentioned op#30 NMS, is
     this just a named example? will it limit this knowledge from being
     generically used?"

   **GOOD pattern**: lift the lesson into a `**Mitigation paths (decision
   rule)** — applies to ANY op...` sub-section above the Evidence list.
   Each evidence then tags which path/branch was taken. Future agents
   read the rule first, find their own op's path, then optionally browse
   evidence for analogues. Op-anchored evidence is fine; op-anchored
   *rules* are the trap.

   **Apply this rule whenever**:
   - The new evidence introduces a fork in the existing entry's logic
     (path A vs path B, sub-case X vs sub-case Y)
   - The new evidence adds a scope-narrowing or scope-broadening clause
     ("only when ...", "also applies when ...")
   - The new evidence adds a mitigation step or decision criterion that
     wasn't in the entry's main body

   When in doubt: ask "if a future agent grep'd for the keyword without
   reading the full evidence list, would they find the rule?" If no, lift
   it.

   - **Known issues (EC/PB)**: OK to be case-specific (they describe exact error
     patterns). EC titles MAY lead with op-class names (e.g. "EC-39: Cube
     `MatmulImpl<MM_CFG=CFG_NORM>` rejected") because they document a specific
     compile/runtime symptom that scopes to that API surface. **No generalization
     required for EC/PB**.

   - **Operational knowledge (OL)**: MUST follow the layered shape above.

     **Reject pattern (apply to every proposed OL title)**:
       - Title leads with `<OpClass>: ...` (colon-scoping) — REJECT, regen with a
         principle-first title. Example to reject:
           BAD:  "OL-N: GroupedMatmul: per-AIC dispatch on group axis"
           GOOD: "OL-N: Multi-instance partition-dispatch when output rows partition
                  by a host-knowable boundary"
                  (then in body: "**Cube-unit instance** (current evidence base):
                  GroupedMatmul — `blockDim = G`, per-AIC `MatmulImpl`, host
                  pre-computes `cum_out[G+1]`...")
       - Title leads with `<OpClass>-<word> primer` / `<OpClass> playbook` — REJECT.
           BAD:  "OL-N: Cube-unit primer — MatmulImpl with constexpr static tiling"
           GOOD: "OL-N: Hardware-accelerator typed-config — use the typed config, not
                  the generic policy"
                  (then in body: "**Cube-unit instance**: `MatmulApiStaticTiling`
                  wraps `MatmulConfig`. Code template + cube-specific gotchas live in
                  `patterns/domains/platform_compat.md §P-P68`.")
       - Title contains a specific op-name (`Cat`, `BatchMatmul`, `SwigluQuant`, etc.)
         in a way that scopes the lesson to that op — REJECT. (Containing the word in
         a general sentence is fine — e.g. "Algorithm selection: when Sort vs Reduce
         is the right primitive" uses `Sort` as a noun, not as a scope.)

     **Accept pattern**: title describes the underlying rule, principle, or
     anti-pattern abstractly; body has the principle in prose plus small concrete
     pieces (snippets/signatures/named-op evidence) that pin it down. Full code
     template placement is a release-review proposal for
     `patterns/domains/<domain>.md`, not a runtime edit.

     E.g.:
       Original case (op#X surfaced this): "Cat V2 fixed alignment bug by overlapping tail write"
       → Generalized OL title: "Non-aligned strided DataCopy can be fixed by
         overlapping tail write"
       → Body principle: prose explaining the alignment-class mismatch + the
         tail-overlap technique
       → Concrete anchor (small piece): 3-line `DataCopy(dst, src, ALIGN8(count))`
         + `DataCopy(dst + ALIGN8(count) - 8, src + count - 8, 8)` snippet
       → Evidence: "Cat V2 (2026-MM-DD): first instance — fp16 [4096,8192] dim=1.
         Cross-validated on op#Y (date)."
       → Other instances (predicted): strided gather + scatter + segmented copy

   - **Patterns (P-)**: MUST be general — describe the technique, not the specific op.
     Same reject pattern as OL applies. A release review may place op-specific
     code templates in `patterns/domains/<domain>.md`; runtime Mode 1 only
     records that proposal. Op-specific
     entries in the top-level `patterns/PATTERN_INDEX.md` are NOT allowed.

     Example to reject:
       BAD:  "P-PN: For torch.cat on fp16 [4096,8192] dim=1, use Adds(dst,src,0.0f)"
       GOOD: "P-PN: Bridge VECIN→VECOUT in pure data-movement kernels with a no-op
              VEC op (Adds with 0.0)"

   - **Workflow when an entry is too specific**:
     1. Rewrite the title to lead with the principle (the rule, technique, or
        anti-pattern), not the op class.
     2. Propose moving full code templates / extensive tiling field maps /
        4-corner-lattice tables to `patterns/domains/<domain>.md` in the
        workspace audit; do not perform the bundled-file move.
     3. Keep the op-specific instance under `## Evidence` (op name + date + concrete
        result) AND embed a small 3–5 line snippet in the body as concrete anchor —
        not the whole kernel.
     4. Add a `## Other instances (predicted)` bullet listing other ops/regimes the
        principle applies to — forces forward-looking generality.

   - **Use judgment**: some OL entries are inherently case-specific (e.g., OL-11 about
     SOC version mismatch — there's only ONE SOC version mismatch issue, the title
     can't be more general without becoming useless). Those are fine. The reject
     pattern catches the common case where the entry is one-op-scoped because the
     author never tried to generalize, not because no generalization exists.

   - **History (why this step matters)**: 2026-04-28 incident — orchestrator wrote
     OL-91 ("Cube-unit primer"), OL-92 ("ISTRANS template flag is NOT the *cube*
     transpose driver"), OL-93 ("GroupedMatmul: per-AIC dispatch") via direct Edit
     calls bypassing this skill. All three shipped with op-class-prefixed titles —
     exactly the antipattern this step exists to prevent. User caught post-merge:
     "for all 1-4 in op kb are all not generic enough. how did we allow those item
     into kb?" Fix landed as: this expanded step 2 (the semantic review IS the
     quality gate, layered title + principle + concrete-anchor pieces structure),
     regenerated entries (OL-91/92/93 retitled with cube-unit-instance evidence
     kept as anchors), regression test on op#5 MatmulTransB.

     A short-lived hook-based enforcement (G9) was tried then walked back same day
     per user direction: "I don't like the idea to have a global hook to prevent
     certain file changing. It cannot be part of skills that installed from a
     public marketplace". A project-level `.claude/settings.json` hook doesn't ship
     with marketplace-distributed skills, and forcing one mutates user settings on
     install. **Quality lives in this skill's review prompt — discipline, not
     barrier**. When this skill runs it self-applies the check; when bypassed via
     direct Edit, there's no enforcement — that's an orchestrator-discipline
     question, not a skill-packaging one.

     User clarification on layered shape: "you can generalize the topic/content.
     but meanwhile you can definitely use certain op/code as example to explain,
     just in case some weaker model misunderstands the statement. in this way we
     link what we met to generic ideas and also not losing details" + "but not
     code by code, just some pieces". See
     `docs/postmortem/kb_generalization_gap_2026_04_28.md`.
2.5. **Mandatory applicability scope** (P0aax, 2026-05-07).

   Every c-tier candidate MUST encode the equivalent of the historical
   `applies_to:` line in its JSON `scope` object. Do not emit a bundled
   markdown header or serial ID:

   `"scope": {"soc": "<chip>", "cann": "<version>",
   "bisheng": "<version>", "op_class": "<class>"}`

   Tag values:
   - `scope.soc`: REQUIRED. `Ascend950PR` / `Ascend910_V220` / `all` (last only if
     the lesson was empirically validated on ≥2 SoCs in the Evidence section).
   - `scope.cann`: REQUIRED. Specific version (e.g., `9.0.0`) or `all` (same rule
     as soc — only after multi-version evidence).
   - `scope.bisheng`: REQUIRED for compiler-dependent lessons (vector/cube codegen,
     register allocation, calling conventions); `n/a` if purely API-surface.
   - `scope.op_class`: OPTIONAL. If the lesson is op-class scoped (e.g.
     `quant-matmul`, `reduction`, `attention-fwd`), say so. `all` if generic.
   - Additional scope keys allowed: `dtype`, `arch`, `kernel_type`.

   **Why mandatory** (P0aax fix): cross-arch drift between A5/A3 (and
   driver/CANN-version drift over time) means a rule validated on one
   environment may not hold on another. Scope-less entries silently
   over-generalize. The `scope` object lets future agents — and the
   `--scan` mode — narrow KB load by environment, and lets the tag-aware
   regression gate (step 3.5 below) detect "this new rule contradicts an
   existing rule scoped to the same SoC/CANN".

   When the proposed entry lacks `scope`, add it before emitting the intake.
   Default values when unstated: `{"soc":"Ascend950PR","cann":"9.0.0",
   "bisheng":"n/a","op_class":"all"}` for an A5-discovered finding.
   Do NOT use `all` for soc/cann unless evidence is multi-environment.

   **Encoding partial certainty (P0aax follow-up, 2026-05-07)**: when the
   kb_manager isn't sure whether a lesson applies beyond its discovery
   environment, encode the uncertainty rather than erase it. Put verification
   boundaries in the candidate's `meta` object:

   ```
   "scope": {"soc":"Ascend950PR","cann":"9.0.0","op_class":"quant-matmul"},
   "meta": {
     "verified_on": [{"soc":"Ascend950PR","cann":"9.0.0"}],
     "unverified_on": [{"soc":"Ascend910_V220","reason":"evidence does not transfer automatically"}]
   }
   ```

   Rules for the verified/unverified metadata:
   - Omit `meta.verified_on` if it duplicates `scope` exactly (i.e.
     everything claimed is also tested) — this is the common case.
   - Add `meta.unverified_on` whenever the agent has ANY hesitation that the
     rule may NOT apply on a known sibling environment. Explicit
     "unverified on A3" beats silent "applies to all".
   - The `unverified_on` metadata is not a permission to ignore — future
     agents reading the entry on the unverified environment should
     test the lesson before relying on it (and either lift to
     `verified_on` or document a counter-instance).
   - When in doubt, prefer narrower `scope` + explicit `unverified_on` over a
     broad `scope.soc=all` or `scope.cann=all`.

   Why this matters: the harness runs on A5 (Ascend950PR) and A3
   (Ascend910_V220 / V100 family) with shared KB. Empirical evidence
   on A5 does NOT automatically transfer — primitive precision (Tanh
   bimodal floor, Sigmoid uniform 2-ULP) is chip-family-wide for some
   ops but quantitatively different for others. Encoding "verified=A5,
   unknown=A3" gives the A3-side agent a chance to either confirm
   transfer or document a counter-pattern, rather than silently
   inheriting an A5-only conclusion.

   Existing c-tier entries without scope may be proposed as refinements when
   already relevant. For bundled entries, record any backfill as a release
   audit recommendation; never edit them in this skill invocation.

2.6. **Evidence cross-check**. Origin incident: OL-158 misled the optimizer.

   When a proposed entry cites a specific op in its Evidence section
   (e.g. `5_Cumsum (51/51 PASS)`), verify that the cited op's actual
   implementation MATCHES the pattern being described:

   ```
   For each evidence-cited op:
     1. Find the archive: output/<project>/src/kernels/<op>/ or workspace/<op>/
     2. Grep the kernel source for the key pattern described (e.g. "fp32
        accumulation" vs "per-step fp16")
     3. If the cited op's code CONTRADICTS the entry's claim:
        → REMOVE that op from evidence AND flag as EVIDENCE_MISMATCH
        → If removing evidence leaves the entry with NO evidence → omit
          it from the intake and record it as unverified in the workspace audit
     4. If no archive/workspace exists to check: add `[evidence unverified]`
        annotation after the cited op
   ```

   **Why mandatory** (2026-05-15 incident): OL-158 claimed `5_Cumsum (51/51)` as
   evidence for "fp32 accumulation is correct for scan", but the actual 51/51
   archive uses per-step fp16 — the OPPOSITE pattern. The entry was merged
   without verification, causing a downstream optimizer to apply the wrong
   pattern and regress precision from 51/51 to 49/51. Evidence cross-check
   would have caught: cited code != claimed pattern → demote or fix.

   This is a semantic check — LLM judgment required. Cite mismatch findings
   in the merge report.

2.6. **Skill-shape WARNING** (owner-directed 2026-07-02; WARNING, not a block).
   For each proposed KB entry, judge whether it reads like a **procedure/workflow**
   rather than a **principle/pattern**. Procedures belong in a SKILL (actively
   invoked, so it's actually used), NOT in KB (passively grepped, so it may never
   be recalled at the moment it's needed). See `docs/design/KB_VS_SKILLS_BOUNDARY.md`.

   Signals (any **≥2** → emit the warning):
   - the body contains a **command sequence** (≥2 shell/CLI/tool invocations in order)
   - **ordinal step language** ("first … then … finally", "step 1/2/3", "run X then parse Y")
   - it describes **producing / measuring / rendering an artifact** (a report, chart,
     profile, dataset) as its payload
   - the payload is **which files/tables/flags to parse** (a how-to), not a rule about
     *behavior* / *what-causes-what*

   On a match, PRINT (do not block, do not drop the entry):
   ```
   [kb-manager] NOTE: entry '<id>' reads like a PROCEDURE (command sequence / step-by-step /
   artifact-production). Procedures belong in a SKILL (actively invoked), not KB (passively
   grepped). Consider: create/extend a skill and reference it from ko/fo; keep only the
   PRINCIPLE in KB. See docs/design/KB_VS_SKILLS_BOUNDARY.md. Proceed if you disagree.
   ```
   Then continue processing the entry normally (the author keeps control — this only makes the
   KB-vs-skill choice conscious). If the entry is a MIX (a principle wrapped around a procedure),
   suggest splitting: principle → KB, procedure → skill.

3. For each entry (after generalization): check if it duplicates existing
   - If duplicate in c-tier: emit one consolidated c-tier candidate.
   - If duplicate in b-tier with no new evidence: omit it from the intake.
   - If it adds evidence to b-tier: emit a c-tier refinement with the b-tier
     ID in provenance; never append to the bundled file.
   - If new: emit a c-tier candidate without a canonical serial ID.

3.5. **Regression-risk gate** (P0aax, 2026-05-07).

   Before merging a new entry (or appending evidence to an existing one),
   check whether the proposed change would WEAKEN or REGRESS against a
   currently-validated rule. This is distinct from "logic conflict" (Mode 2
   step 2) — logic-conflict detects A-says-X vs B-says-not-X; regression-risk
   detects "the new entry, if accepted, would lower the bar an older entry
   established with multi-evidence proof".

   For each proposed entry, scan existing OL/EC/PB/P-P with overlapping
   applicability scope (soc + cann + op_class) for the following risk
   patterns:

   1. **Tolerance-loosening**: new entry recommends a wider numeric tolerance
      (e.g., `T2=1e-3`) than an existing entry with the same scope (e.g.,
      OL-104 mandates `T2=1e-5` for fp16 reductions). RISK: blanket-loosens
      tolerance; future ops will under-validate.
   2. **Re-introducing an anti-pattern**: new entry's body contains a
      construct the existing KB explicitly marks as anti-pattern (search
      `## Anti-patterns avoided` sections + `❌` markers in P-P entries).
      RISK: undoes a hard-won lesson.
   3. **Contradicting Evidence-anchored guidance**: new entry's principle
      is "use X" but an existing entry has ≥2 Evidence rows showing X
      caused regression. RISK: ignores prior validation.
   4. **Scope-broadening without evidence**: new entry promotes an existing
      narrow rule to a broader scope (e.g., `op_class=reduction` →
      `op_class=all`) without adding multi-class Evidence. RISK: false
      generalization.

   Action when risk pattern matches:
   - Write `workspace/kb_scan/regression_risk_<ts>.md` with the proposed
     entry, the existing entry it would regress against, and the risk
     classification.
   - DO NOT silently accept the new entry. Instead, choose one of:
     a. **Keep unadmitted** — omit the proposed entry from the c-tier intake,
        cite the risk file in the workspace audit, and wait for user review.
     b. **Stage an evidence refinement** — if the new entry's content is
        actually an instance of the existing rule (not a contradiction), emit
        a c-tier refinement per step 3.
     c. **Block + flag** — if neither (a) nor (b) fits, BLOCK the merge
        for that one entry, accept the rest, and report `NEEDS_USER_INPUT`
        in the merge report.
   - The regression-risk gate is per-entry, not per-batch. Other entries
     in the same batch proceed normally.

   This is a quality gate, not a syntactic check — semantic LLM judgment
   applies. Cite the risk classification in the merge report so user can
   audit later.

4. Quick dedup: scan the proposed intake entries and their matched KB entries
   for obvious duplicates.
4.5. **Index sync**.

   Do not edit bundled `KB_INDEX.md`. `CannbotCProvider.reindex()` rebuilds the
   user-local `INDEX.md` after each admitted c-tier entry.

5. **Do not drop `.kb_merged`.** The orchestrator writes a `tier=customer`
   marker only after the Arbiter/CannbotCProvider write succeeds (or after a
   reviewed intake validly contains no admissible entries).
6. Print report: what was staged/rejected and what was generalized. The
   orchestrator reports the durable c-tier IDs and marker location.

This is fast (< 2 min). Runs automatically after every bundled-orchestrator workflow.

---

## Mode 1-batch: Update with multi-folder scan (V2, 2026-04-29)

Triggered by: `/aog-knowledge-maintain --batch [--scan-roots workspace/ output/<project>/src/kernels/]`

For parallel-lane execution by the bundled orchestrator, the
orchestrator runs N agents concurrently; each writes its own
`workspace/<op>/knowledge_update.md`. After all complete, the
orchestrator calls THIS mode ONCE to review the entire batch in one
serial pass — not N separate Mode 1 invocations.

### Why batch (vs. N × Mode 1)

Two reasons. First, **cross-batch dedup**: if lane 1 and lane 2 both
discover the same pattern, batch-mode dedups across the entire intake before
admission. Second, **single deterministic admission pass**: gather all proposed
entries, sort by intake order, and let `CannbotCProvider` allocate content-hash
IDs. This avoids duplicate per-lane semantic decisions and provider writes.

### Idempotency contract

This mode is **idempotent**: the orchestrator skips only a `.kb_merged`
marker that the deterministic verifier has bound to durable provider state.
Text presence, a non-empty marker, or an `entries=` token alone is never
trusted. Crash recovery quarantines an invalid marker and reviews that
workspace again. The skill itself performs no provider writes or locking.

### Steps

1. **Use orchestrator-verified pending sources** — for each scan root, process
   the `**/knowledge_update.md` paths supplied by the orchestrator after its
   durable-marker verification:
   - The knowledge_update.md is non-trivial (>50 chars after stripping
     comment lines and section headers)

2. **Load all proposed entries into one batch list**:
   `[(source_workspace, entry_proposal), ...]` — preserving source
   attribution for the merge log later.

3. **Cross-batch dedup** (extends existing Mode 1 step 3):
   - Group entries by semantic class (EC / PB / OL / P-P / CAND-PP)
   - Within each group, scan for >70 % content overlap pairs across
     ALL sources (not per-source). Consolidate near-duplicates into a single
     entry with both sources cited in the Evidence section.
   - Treat already-existing KB entries as the baseline — proposed
     entries that duplicate something in the current KB get demoted
     to a c-tier evidence-refinement form (Mode 1 behavior).

4. **Provider ID allocation**: do not allocate EC-/OL-/PB-/P-P serial IDs.
   Emit ID-free c-tier candidates; `CannbotCProvider` assigns
   `customer:<content_hash>` after admission.

5. **Logic-conflict detection** (extends existing Mode 2 step 2):
   - Cross-entry contradiction scan over the batch + current KB.
   - "Entry A says use X" vs "entry B says avoid X" → BLOCK those
     conflicting entries from admission.
   - Write `workspace/kb_scan/conflicts_batch_<ts>.md` with both
     entries quoted + proposed resolution + `NEEDS_USER_INPUT`.
   - Omit conflicting entries from the intake; emit the non-conflicting ones,
     and tell the user explicitly.

6. **Emit intake files** — one exact orchestrator-supplied JSON path per
   workspace, max 10 entries per invocation. Do not apply provider or bundled
   KB changes from the skill.

7. **Do not drop markers** — the orchestrator validates every intake before
   the first durable c-tier write, then writes one `tier=customer` marker per
   processed workspace.

8. **Report**:
   ```
   Knowledge Base Mode 1-batch Report
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     Sources scanned:   N workspaces
     Sources processed: M (skipped K already-merged)
     Proposed entries:  P → after dedup: D
     Entries staged:    N c-tier candidates (provider IDs pending)
     Logic conflicts:   C (BLOCKED if >0 — see workspace/kb_scan/conflicts_batch_*.md)
     Intake files:      <list>
   ```

### Failure modes

- **Logic conflict in batch**: conflicting entries are omitted with a
  user-facing message; only the validated non-conflicting intake proceeds.
- **Mid-batch crash**: re-run is safe (idempotent — skips processed sources).
- **Same op archive has an orchestrator-verified durable marker**: the
  orchestrator omits it from the supplied pending set and reports it as
  "already merged: src=<path>".
- **No pending sources found**: skill exits with "nothing to merge",
  exit 0, no error.

### Cross-CC-instance concurrency

Provider content-hash IDs make repeated writes idempotent. The current
`CannbotCProvider` index rebuild has no cross-process lock, so operators should
avoid concurrent writers to the same `ASCENDC_PORT_USER_KB`; this skill never
uses git or bundled-ID renumbering as a runtime coordination mechanism.

---

## Mode 2: Full Scan (explicitly triggered with --scan)

Triggered by: `/aog-knowledge-maintain --scan`

This is slow (5-10 min). Run periodically (e.g., after every 5 ops, or when user asks).

Mode 2 is an audit in an installed plugin. It may write reports under
`workspace/kb_scan/` and may stage c-tier intake entries, but it MUST NOT apply
edits, archives, renames, or promotions to bundled b-tier. Any "apply" wording
below means "record the proposed release change in the audit report".

### Step 1: Dedup Scan (all files)

**Required output**: Write `workspace/kb_scan/dedup_results.md` with findings.

For each KB file, scan ALL entries for:
- **Exact duplicates**: same lesson described with different IDs → propose archiving newer one
- **Near duplicates**: >70% content overlap → propose merging into older and archiving newer
- **Semantic duplicates**: same underlying issue described differently (search by keywords
  like TQue, bf16, DataCopy, atomicAdd — if 2+ entries discuss the same API/feature,
  check if they're saying the same thing)
- **ID conflicts**: duplicate IDs (e.g., two PB-7 entries) → propose archiving
  the later one and creating a release-owned entry with the next available ID.
  (NEVER rename IDs — safety rule. Record archive/create proposals only.)

### Step 2: Cross-File Conflict Detection

**Required output**: Write `workspace/kb_scan/conflicts_found.md` with findings.

Conflicts happen ACROSS files, not just within one file. Explicitly check:

**Status conflicts** — for each PB entry with Status: OPEN, search OL entries for
the same keywords (e.g., "TQue", "bf16 cast", "typed kernel"). If an OL entry says
"RESOLVED" or "fixed" for the same issue, that's a conflict.

```
For each PB-N with Status: OPEN:
  keywords = extract key terms (TQue, DataCopy, bf16, etc.)
  for each OL-M:
    if OL-M mentions same keywords AND says "resolved"/"fixed"/"preferred":
      CONFLICT: PB-N says OPEN, OL-M says resolved

For each EC-N:
  for each PB-M:
    if both describe the same error pattern:
      NEAR-DUPLICATE (cross-file): add cross-references, keep both
```

**Scope conflicts** — check if an entry lacks scoping:
```
For entries mentioning "CANN source" or "PyTorch":
  Does it specify "NPUKernelBench only" or "always"?
  If ambiguous, flag for clarification.
```

**Advice conflicts** — opposite recommendations:
```
Scan for pairs where one says "use X" and another says "avoid X".
E.g., OL-26 says "research CANN source" vs OL-36 says "CANN is prohibited"
Check if scoping resolves it (different contexts).
```

For each conflict found:
1. Read BOTH entries fully (not just titles)
2. Check evidence dates — newer evidence may supersede older
3. **NEVER auto-resolve** — write a PROPOSED resolution with reasoning
4. Mark as `NEEDS_USER_INPUT` in the conflicts file
5. If the conflict is clearly a labeling issue (e.g., "RESOLVED" means
   "workaround adopted"), propose a label clarification edit

### Step 3: Validate Known Issues

**Required output**: Write `workspace/kb_scan/validation_results.md` with per-entry results.

For each entry with a testable claim, verify against current A5:

**ERROR_CORRECTIONS** — reproduce the compile error:
```
1. Write minimal .cpp triggering the error pattern described in the entry
2. Deploy to A5: write to current_task/kernel/, build via build_ascendc.py
3. Check output:
   - Error still occurs → VALIDATED
   - Error message changed → record a proposed entry update with the new message
   - Error no longer occurs → mark POSSIBLY_FIXED, flag for user
```

**PLATFORM_BUGS** — reproduce the bug:
```
1. If repro test exists (tests/repro/): run on A5
2. If no repro but symptom is described: try to reproduce
3. Results: VALIDATED / POSSIBLY_FIXED / UNTESTABLE
```

**OPERATIONAL_KNOWLEDGE** — check references still exist:
```
1. If entry references a file/function: grep/glob for it
2. If entry references a tool/process: check it exists
3. Results: VALIDATED / REFERENCE_MOVED / OBSOLETE
```

**PATTERN_INDEX** — check domain files present:
```
1. For each pattern: check referenced domain file exists
2. Check trigger conditions still match current codebase
```

### Step 4: Candidate Promotion Audit

**Required output**: Add promotion decisions to `workspace/kb_scan/validation_results.md`.

Check `patterns/unverified/candidates.md`:
- **Recommend promotion** if ALL of:
  - Validated on **2+ operators** with independent evidence (different ops, not same session)
  - Does NOT duplicate an existing pattern in PATTERN_INDEX (check by trigger + technique)
  - Validation evidence includes actual test data (not just "should work")
  → Propose a release-owned move to `patterns/domains/*.md` and a future
  PATTERN_INDEX ID; do not perform either edit
- **Recommend keeping as candidate** if only 1 op validated
- **Recommend archive** if contradicted by evidence or superseded by a verified pattern

### Step 5: External Review (if codex available)

**Required output**: Write `workspace/kb_scan/review_feedback.md` with reviewer responses.

Before accepting proposals into the audit report, check for an external reviewer:
```bash
which codex 2>/dev/null && echo "codex available"
```

If codex is available (preferred — stronger model), send proposed changes for review.
Use `/codex-expert` skill with this structured prompt:

```
Review these proposed changes to an AscendC knowledge base.
For each change, I provide: action, entry ID, file, full BEFORE text, full AFTER text,
reason, validation evidence, and (for patterns/OL) the ORIGINAL case-specific version
vs the GENERALIZED version I'm proposing to store.

[For each proposed change:]
---
CHANGE N: [ADD|EDIT|ARCHIVE]
File: [path]
Entry: [ID]: [title]
BEFORE: [full entry text, or "new entry" for ADD]
AFTER: [full new text, or "archived with reason: ..." for ARCHIVE]
Reason: [why this change]
Evidence: [validation test result, or "dedup scan", or "conflict resolution"]
---

Review checklist (respond for each change):
1. APPROVE or REJECT
2. If REJECT: what's wrong and what should be done instead
3. Risk assessment: LOW (cosmetic) / MEDIUM (semantic) / HIGH (could break KB consumers)
4. For patterns/OL: is the generalization correct? Too broad? Too narrow?
   (Check both the original case-specific input and the generalized version)

Also check:
- Did I miss any duplicates or conflicts in the KB?
- Are any proposed archives premature?
- Are any patterns still too case-specific to be useful for other ops?
```

- **Codex REJECT** → omit that proposal and flag it for the user
- **Codex APPROVE** → retain the proposal in the audit report
- **No codex available** → retain proposals as "unreviewed" in the report
- Optionally also run `/opencode-expert` for supplementary review (lower weight)
- Log ALL reviewer feedback in `review_feedback.md`

### Step 6: Record Proposed Release Changes

Record approved proposals in the workspace audit (max 10 per invocation).
Do not apply them to bundled b-tier from this skill.
For each change, print:
```
CHANGE N/10: [ADD|EDIT|ARCHIVE] in [file]
  Entry: [ID]: [title]
  Reason: [why]
  Reviewer: [codex approved / opencode flagged X / no reviewer]
```

### Step 7: Report

```
Knowledge Base Full Scan Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total entries scanned: N

  Duplicates: N found (N merge proposals, N ID-rename proposals)
  Conflicts: N found (N resolutions proposed, N flagged for user)
  Validated: N / M testable (N on A5, N reference-check)
  Possibly obsolete: N (flagged for user review)
  Candidates: N ready for release review, N kept

  External review: [codex: N comments / opencode: N comments / skipped]

  Changes proposed: N / 10 max
  Proposals deferred (reviewer flagged): N

  Items needing user input:
  - [ID]: [reason]

  Remaining work (re-invoke if needed):
  - [list]
```

---

## Mode 3: Validate Specific Entry (--validate EC-13)

Triggered by: `/aog-knowledge-maintain --validate PB-9`

Tests a single known issue against the current A5 environment:
1. Read the entry
2. If it has a reproducible test: run it
3. If it references code: check if the code/pattern still exists
4. Report: VALIDATED / POSSIBLY_FIXED / UNTESTABLE

---

## Mode 5: Promotion Audit (--auto-promote compatibility)

Runtime auto-promotion is disabled. This compatibility mode is read-only and
may only inspect pending candidate metadata, apply the C36–C40 review criteria
(generalization, dedup, conflict, cross-op transferability, and external
review), and write a recommendation report under the active workspace.

It MUST NOT rename or move bundled candidate blocks, allocate canonical IDs,
edit canonical entries or KB_INDEX.md, create promotion/block markers, invoke a
release promotion implementation, or create `.kb_merged`. Operator finalize
never calls this mode. A release maintainer may use the report in a separate,
reviewed release process outside the installed generation workflow.

## Mode 4: Learn from Official Docs (--learn)

Triggered by: `/aog-knowledge-maintain --learn` or `--learn <url>`

Default URL: `https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_00010.html`

This mode scrapes AscendC official best practices documentation and stages
reviewed findings as user-local c-tier intake entries. It never imports into
bundled b-tier. The documentation is JS-rendered, so **must use Playwright
plugin** (not WebFetch) via `mcp__plugin_playwright_playwright__browser_navigate`
+ `browser_snapshot` / `browser_evaluate`.

### Knowledge access path priority

Before scraping fresh from hiascend.com, consult these sources in order:

1. **Packaged KB** — search `${CLAUDE_PLUGIN_ROOT}/kb/target/ascendc/` and `${CLAUDE_PLUGIN_ROOT}/kb/okf/` first.
2. **Public hiascend.com CANN documentation (browser-rendered)** — URL slugs:
   - CANN 9.0.0 commercial: `https://www.hiascend.com/document/detail/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_XXXX.html`
   - CANN 9.1.0-beta.1 community (newer, has SIMT API): replace `canncommercial/900` → `CANNCommunityEdition/910beta1`
   - Known page IDs: Mmad=`0249`, LoadData=`00169`, CrossCoreSetFlag=`0273`, SIMT `__float2int_rz`=`10616`
   - **CRITICAL — owner-verified 2026-05-27 22:38Z + 22:43Z**: any product-support matrix row reading **"Atlas 350 加速卡" IS Ascend950PR / V300 / arch35**. Marker: Atlas 350 supports SIMT API; SIMT decorator `__simt_vf__` + header `simt_api/device_functions.h`; CANN 9.0.0 install on A5 has `*_simt.h` files under `arch35/` and `v35/` dirs. The Atlas 350 == arch35 == V300 == SIMT NPU equivalence is **codified, do not re-derive**.
3. **CANN install headers** — inspect the active target's `$CANN_PATH` include tree and grep `__CCE_AICORE__` / `__DAV_*__` for per-arch divergence; cite the concrete file/line.
4. **Optional local source checkout** — only if present and permitted by the current project policy. The community plugin does not package the old vendor tree or a scraped CANN docs mirror. Reading-to-distill is allowed only within policy; copying-to-deliverable remains forbidden.

When staging c-tier intake: cite the path used in the entry's provenance,
including page ID for hiascend pages and file:line for header/source.

### Step 1: Scrape Documentation TOC

Use dev-browser to navigate to the overview page and extract all sub-page links:

```
Sections to scrape:
├── SIMD 算子性能优化
│   ├── Tiling策略 (核间负载均衡)
│   ├── 头尾开销优化 (6 sub-pages)
│   ├── 流水编排 (DoubleBuffer, Iterate/IterateAll)
│   ├── 内存访问 (11 sub-pages: 大数据块, 512B对齐, 搬运API, 非对齐, Buffer复用, bank冲突, L2切分...)
│   ├── 矢量计算 (UB融合, Counter模式, 低延迟归约)
│   └── 矩阵计算 (BT/FP/L0C Buffer, L1驻留, AtomicAdd)
├── VF(SIMT)性能优化
│   ├── VF循环优化
│   ├── 指令双发优化
│   ├── 连续非对齐场景优化
│   └── VF融合优化
├── SIMD+SIMT 混合算子优化
│   ├── 内存访问 (UB提升效率)
│   └── 计算优化
└── 性能调优案例
    ├── FlashAttention
    ├── Other Matmul cases
    ├── GroupedMatmul
    └── Matmul系列 (6+ sub-pages)
```

### Step 2: Per-Page Content Extraction

For each sub-page, use dev-browser to extract:

```
{
  "title": "页面标题",
  "category": "Tiling策略 | 头尾开销 | 流水编排 | 内存访问 | 矢量计算 | 矩阵计算 | VF优化 | 混合优化",
  "positive_pattern": "推荐的做法（正模式）",
  "negative_pattern": "不推荐的做法（反模式）",
  "code_before": "优化前的代码示例（如有）",
  "code_after": "优化后的代码示例（如有）",
  "applicable_when": "什么场景下应用此优化",
  "performance_impact": "预期性能影响",
  "a5_regbase_opportunity": "是否可以在 A5 上用 reg-based 替代 mem-based（见下文）"
}
```

### Step 3: A5 Reg-based Enhancement Check

**CRITICAL**: 社区文档的 SIMD 优化是为 A2/A3 写的（mem-based，操作数在 UB）。A5 作为新一代芯片，可能支持 reg-based 实现以获得更好性能。

对每个提取的优化建议，额外检查：

1. **识别 mem-based 操作**：文档中使用 DataCopy(UB→UB)、UB 中间变量、TBuf 临时缓冲的地方
2. **查找 reg-based 替代 API**：在 AscendC API 文档中搜索同名指令是否有寄存器版本
   - 例如：`Add(dst_local, src1_local, src2_local, count)` 是否有 `Add(dst_reg, src1_reg, src2_reg)` 变体
   - 检查 SIMT 模式下的标量/寄存器操作是否能替代 SIMD 的 UB 操作
3. **标注替代可能性**：
   - `REG_POSSIBLE`: 有明确的 reg-based API 可用
   - `REG_INVESTIGATE`: API 可能存在但需要实验验证
   - `MEM_ONLY`: 该优化只能用 mem-based（如大批量 DMA 操作）

### Step 4: Dedup Against Existing KB

对每个提取的知识条目，与现有 KB 做文本匹配：

```
For each extracted entry:
  Search existing KB files for overlapping keywords (top 3 matches)
  If match > 70% overlap:
    SKIP (already in KB) — log as "dedup: matches [ID]"
  If match 30-70%:
    SUPPLEMENT — stage a c-tier refinement that cites the existing ID
  If match < 30%:
    NEW — stage a new c-tier candidate
```

### Step 5: Stage c-tier intake

按提取内容映射为用户 c 层条目，不分配内置 KB 编号，也不写内置文件：

| 提取内容 | c-tier `kind` | 元数据 |
|---------|----------------|--------|
| 编译/构建修复 | `anti_pattern` 或 `experience` | 在 provenance 中引用文档 |
| 性能优化建议 | `positive_pattern` | evidence 必须包含来源证据 |
| 可复用的优化技术 | `positive_pattern` | scope 标注适用算子类别 |
| A5 reg-based 替代建议 | `experience` | meta 标注待 A5 实验验证 |

每个条目必须标注来源：`- **Source**: hiascend.com best practices (采集日期)`
只写 orchestrator 指定的 workspace intake JSON；由 orchestrator 经
`Arbiter.write(..., "customer")` 持久化。

### Step 6: Intake validation

提交前对 intake 再运行一次 Mode 2 Step 1 的只读去重检查，确保候选
不与已有条目冲突；不得修改内置 KB。

### Step 7: Report

```
Knowledge Base Learn Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Pages scraped: N
  Entries extracted: N

  Staged for c-tier admission: N entries
    - positive_pattern: N
    - anti_pattern: N
    - experience: N

  Skipped (dedup): N
  Refinements staged: N

  A5 Reg-based opportunities:
    - REG_POSSIBLE: N
    - REG_INVESTIGATE: N
    - MEM_ONLY: N

  Pre-admission dedup: N conflicts found
```

---

## Integration Points

### After each op (automatic):
```
# In ascendc-op-gen Step 4c:
/aog-knowledge-maintain knowledge_update_path=workspace/{name}/knowledge_update.md
```
→ Mode 1: fast semantic review; orchestrator admits accepted entries to c-tier.

### Periodic maintenance (user-triggered):
```
/aog-knowledge-maintain --scan
```
→ Mode 2: full scan with validation. Run after every ~5 ops, or when KB feels stale.

### Spot-check (user-triggered):
```
/aog-knowledge-maintain --validate PB-9
```
→ Mode 3: test one specific entry. Use when you suspect a bug might be fixed.

### Knowledge enrichment (user-triggered):
```
/aog-knowledge-maintain --learn
/aog-knowledge-maintain --learn https://www.hiascend.com/document/detail/zh/...
/aog-knowledge-maintain --learn --hw-spec
```
→ Mode 4: scrape official AscendC best practices, extract patterns, check for A5 reg-based opportunities. Run when KB coverage feels insufficient, or after major CANN version updates.

`--hw-spec` sub-flag: targets the **hardware architecture spec pages** instead of best-practices pages. Scrapes the architecture version page for the current target chip (default: 351x) and stages sourced c-tier findings plus a workspace audit; it never updates bundled `hardware/target/ascend950pr.md`. Key URLs:
- 351x arch spec: `https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_00065.html`
- Best practices (default): `https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_00010.html`

**Note**: Mode 4 requires `dev-browser` skill (hiascend.com is JS-rendered, WebFetch cannot extract content). Ensure browser server is running before invoking.

**Lesson (2026-04-21)**: The 351x public page directly answered 5 of our "internal-only" hardware questions (UB bank conflict rules, AIV↔L1 hardware channel, SSBuffer, SIMT register specs, VEC data flow). Always scrape `--hw-spec` before escalating to internal experts.
