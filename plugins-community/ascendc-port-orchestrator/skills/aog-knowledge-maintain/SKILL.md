---
name: aog-knowledge-maintain
description: >
  Review AscendC runtime findings, deduplicate them against the installed
  cannbot-knowledge repository, and stage user-local c-tier intake entries.
  Use when an operator run produces new knowledge or the AscendC KB needs maintenance.
argument-hint: >
  Default: knowledge_update_path=workspace/{op}/knowledge_update.md;
  batch: --batch --scan-roots ROOTS; audit: --scan; validate: --validate ID;
  learn: --learn [url] [--hw-spec].
context: inline
---

# AscendC knowledge maintenance

Official b-tier knowledge lives only in the project-installed
`cannbot-knowledge` checkout. This skill never edits that checkout and never
falls back to removed plugin-local cards. Runtime findings are admitted only to
the user-local c-tier through an orchestrator-owned intake file.

Before a high-leverage decision, read
`${CLAUDE_PLUGIN_ROOT}/kb/shared/ANTI_PRESSURE_PROTOCOLS.md`.

## External knowledge access

1. Confirm the current target platform. Normalize A5/Ascend950PR/V300/arch35 to
   Registry platform `950`; keep A3/A2 as their registered platform values.
2. Invoke the installed `knowledge-query` skill for topic, symptom, runbook,
   concept, optimization, or operator knowledge. Use `domain=ops`,
   `technology=ascendc`, and the confirmed platform. Read selected returned
   cards fully.
3. Invoke `ascendc-api-knowledge-query` for an exact Ascend C API contract or
   overload. There is no monolithic API catalog or language-reference file.
4. Do not glob/grep the external knowledge tree, reconstruct paths from old
   IDs, rebuild its index, or scan the tree when a query fails. Report the
   configuration/index/query failure explicitly.
5. Directly read a canonical card only when the brief/query already supplied a
   unique `$CANNBOT_KNOWLEDGE_ROOT/knowledge/...` `local_path`.

The external repository and its index are read-only in every mode. New evidence
is c-tier material until a separate knowledge-repository maintainer performs a
governed ingest/release review.

## Runtime persistence contract

- Emit only the exact workspace intake path supplied by the orchestrator:
  `{"schema_version":1,"entries":[...]}`.
- Each entry may contain only `kind`, `claim`, `scope`, `key`, `evidence`,
  `provenance`, and `meta`.
- `kind` is `positive_pattern`, `anti_pattern`, or `experience`.
- Claims are English, code-facing prose. Verbatim Chinese error strings may be
  retained as evidence.
- Emit at most 10 entries per invocation and report any remainder.
- Emit `entries: []` when nothing is admissible.
- Never write directly to `ASCENDC_PORT_USER_KB`, its default root,
  `.kb_merged`, promotion markers, external cards, or external indexes.
- The orchestrator validates the intake and persists accepted entries through
  `Arbiter.write(entry, "customer")` / `CannbotCProvider`.
- Never invent or allocate EC/PB/OL/P-P/CAND IDs. Those are external-governance
  identifiers, not runtime c-tier identifiers.

## Mode 1: update one workspace

Input: `knowledge_update_path=workspace/{op}/knowledge_update.md`.

Also read `workspace/{op}/kb_draft_from_user_decision.md` when present. A
structured user-decision draft is provisional evidence; a heuristic-mode draft
is not auto-admitted and must be reported for owner rewrite. Preserve a source
anchor to the workspace/session in `provenance`.

For each proposed finding:

1. Require a concrete claim, applicability scope, evidence, and provenance.
2. Generalize away from one operator only when the evidence supports transfer.
   Otherwise keep a narrow scope.
3. Reject copied implementation bodies, CANN-internal identifiers, secrets,
   machine-specific credentials, unverifiable performance numbers, and claims
   derived only from an agent's assertion.
4. Query official b-tier by the claim's actual topic/symptom and platform.
   Query c-tier through the runtime provider. Do not infer absence from a single
   zero-result query; try one justified alternate phrasing, then record the gap.
5. Deduplicate:
   - same claim already in b-tier with no new evidence: omit;
   - same claim in c-tier: emit at most one evidence-refining candidate;
   - contradiction: omit and report both sides for owner review;
   - genuinely new, transferable evidence: emit an ID-free c-tier candidate.
6. Bind scope fields to concrete platform/CANN/dtype/operator constraints.
   Never silently broaden a target-specific observation to all targets.
7. Write the exact intake JSON. Do not create completion markers.

## Mode 1-batch: update multiple workspaces

Input: `--batch --scan-roots <roots>` plus the exact per-workspace intake paths
provided by the orchestrator.

Apply Mode 1 to every pending `knowledge_update.md`, then deduplicate across the
whole batch before writing one intake per workspace. Keep deterministic input
order. An entry present in multiple workspaces may be consolidated only when
their claims and scopes agree; preserve all independent evidence anchors.
Conflicts remain unadmitted and are reported. The orchestrator alone creates
completion markers after validating every intake.

## Mode 2: read-only audit

Input: `--scan`.

Audit the user c-tier plus query-selected official cards. Do not scan or mutate
the full external repository. The audit may write a report under the active
workspace and may stage c-tier corrections through an orchestrator-supplied
intake path.

Check:

- duplicate/near-duplicate claims;
- contradictory scopes or recommendations;
- stale evidence and unverifiable measurements;
- missing platform/CANN/dtype bounds;
- candidates supported by only one operator;
- external query/index gaps.

Recommend external release changes in the report only. If a user later
authorizes a release change, route it through the external repository's
maintainer ingest/lint workflow — that tooling lives outside this installed
plugin and is not one of its runtime skills.

## Mode 3: validate one known item

Input: `--validate <legacy-id-or-query>`.

Resolve the item through `knowledge-query`; do not derive a filename from a
legacy ID. Read the selected card fully, run only tests authorized by the
active operator workflow, and report `VALIDATED`, `POSSIBLY_FIXED`, or
`UNTESTABLE` with evidence. Any correction is staged as c-tier intake, never
written into the official card.

## Mode 4: learn from official documentation

Input: `--learn [url] [--hw-spec]`.

1. Query cannbot-knowledge first. If an applicable card already answers the
   question, use it and avoid redundant scraping.
2. For an unresolved gap, use the available browser-capable tool for the exact
   public hiascend.com page/version. Do not assume WebFetch can render the site.
3. For API declarations, use `ascendc-api-knowledge-query` and inspect the
   active target's CANN header when necessary.
4. Extract only sourced, transferable findings. Record page URL/version or
   header path/line in provenance.
5. Deduplicate with Mode 1 rules and stage accepted findings through the exact
   c-tier intake path.

`--hw-spec` targets the architecture page for the current platform. It never
updates the canonical target card directly.

## Promotion-audit compatibility mode

`--auto-promote` is read-only compatibility behavior. It may review candidate
metadata and write a workspace recommendation report. It must not allocate
canonical IDs, rename/move cards, create promotion markers, edit the external
repository, or create `.kb_merged`.

## Required report

Report:

- mode and workspaces reviewed;
- queries issued and cards actually read;
- entries admitted, omitted as duplicates, or rejected;
- contradictions and unresolved evidence gaps;
- exact intake paths written;
- external knowledge configuration/index failures, if any.

Never report an external card as read when only its search summary was seen.
