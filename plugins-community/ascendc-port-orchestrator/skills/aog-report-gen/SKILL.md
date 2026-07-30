---
name: aog-report-gen
description: >
  Generate or refresh a project-level REPORT.md for an output/{project}/ directory
  (cross-generation migration or backward-generation project). Wraps
  src/scripts/gen_report_tables.py
  (table injection via <!-- BEGIN-GEN:* --> markers) and the canonical 9-section
  structure defined in OUTPUT_PROJECT_LAYOUT.md §4. It is restricted to
  arch22→arch35 migration and backward-generation projects under `output/`.
  Use when an AscendC kernel project needs its report initialized, refreshed, or audited.
  Usage:
    /aog-report-gen {project_name}              # refresh tables in existing REPORT.md
    /aog-report-gen {project_name} --init       # create REPORT.md from template if absent
    /aog-report-gen {project_name} --audit      # dry-run: check sections + freshness, no edits
---

# /aog-report-gen


> **Anti-pressure protocols (MANDATORY, load first)**: read `${CLAUDE_PLUGIN_ROOT}/kb/shared/ANTI_PRESSURE_PROTOCOLS.md` before any decision. P1–P8 are LLM-pressure drift modes (user-watching, context-filling, batch-throughput, simple-op assumption, failure discomfort, infrastructure friction, closure desire, tool-path-of-least-resistance) that override technical rules under load. Cite the relevant Px at every high-leverage decision point (spawn / done / PARTIAL / skip-verify).

## When to use

- After a new op lands DONE (run to refresh §核心成果 table from its new verification.json)
- When a KB_USAGE_LOG / KB_COLD_START_RETROSPECTIVE gets updated (cross-link refresh)
- When starting a new output/<project>/ port (needs a skeleton REPORT.md per spec)
- Before session handover / archival (run `--audit` to confirm the project's report is fresh + complete)

## When NOT to use

- For per-op files (workspace/<op>/verification.json, knowledge_update.md) — those are produced by aog-kernel-worker, not by this skill
- For session-level retrospectives — use /aog-session-retrospective
- For KB changes — use /aog-knowledge-maintain

## Report ≠ Log — the cardinal rule (2026-06-19, owner direction)

**A REPORT presents conclusions as data; a LOG narrates how you got there. The front of every REPORT.md is a REPORT, never a log.** This rule fires on EVERY touch of a REPORT.md — full generation AND a one-line hand-edit. **If you are hand-editing a report section inline (not running this skill end-to-end), you MUST still apply this contract** — that is exactly where it gets violated.

**Front sections (§1–2 核心数据 / 核心成果) = comparison-data tables, data-only:**
- Every per-op / per-impl entry is a TABLE ROW, not a narrative paragraph. When ADDING a new impl/result to an existing report, **mirror the existing core-results table format** — do NOT append a prose subsection that tells the story.
- Every perf cell MUST carry the comparison vs the in-scope source-architecture baseline for migration, or the declared target baseline for backward generation. **A bare absolute device-time (e.g. "305.9µs") in a front table is FORBIDDEN** — an absolute with no comparison is meaningless to a reader.
- NO event narration in the front: "✅ NO-GO 已翻案", "复查发现盘上有三套…", dead-prototype absolutes, "vec_ratio 0.044→0.975 优化手法" — these are LOG.

**History / how-we-got-here → the END (or removed):** the investigation trace (failed prototypes, the optimization play-by-play, what flipped and why) goes in a late section explicitly labeled "历史 / 调查记录，非核心数据" (§7 pipeline/harness 发现 is a natural home), NEVER §1.

**Self-check before committing ANY REPORT.md edit:** would a customer reading ONLY §1–2 see (a) clean comparison tables with the vs-original-source ratio, (b) zero narrative of how we got here, (c) **NO internal-harness jargon** (cannbot / 商用「标准」 / PARTIAL_PERSIST / OL-/DEBT- IDs / competitor_mare / pass_a-pass_b / blackbox / bare commit-SHAs / task-IDs — owner's TOP concern: "客户觉得我们是 cannbot 套壳/骗人"), and (d) **an archive link under each core-results table** (归档: [kernel](src/kernels/<op>/) · [verification.json](…))? If no to any → rewrite it. **Now MECHANIZED — run the FOUR report-quality lints:**
- `python3 src/scripts/report_not_log_lint.py <REPORT.md>` — Report≠Log (a/b above)
- `python3 src/scripts/report_jargon_lint.py <REPORT.md>` — front jargon (c above)
- `python3 src/scripts/report_archive_link_lint.py <REPORT.md>` — table→archive links (d above)
- `python3 src/scripts/report_concision_advisory.py <REPORT.md>` — front-prose concision (ADVISORY only, never blocks — owner: "正式报告行文但不会太啰嗦")

All four are BLOCKING gates in Phase R2 + first-class DRIFT drivers in Phase R_audit (EXCEPT concision = advisory). Each is conservative + dual-fixture-tested for no-false-positive (`src/scripts/orchestrator/tests/test_report_quality_lints.py` + `test_report_not_log_lint.py`). No longer LLM-attention-reliant — but still run all four manually after any inline edit that skips the skill (the original incident's path).

**Why this exists** (owner 2026-06-19, repeated): a forward-report SIMD section was hand-edited as a log — NO-GO-翻案 narrative + a dead prototype's runtime + the optimization play-by-play — while the correct comparison table sat directly above as a model. The edit-trigger above closes that gap.

## Inputs

- **project_name** — an arch22→arch35 migration or backward-generation project under `output/`
- Optional flags: `--init` (create fresh from template), `--audit` (dry-run), `--output <path>` (override default `output/<project>/docs/REPORT.md`)

## REPRODUCE.md 写作规则（强制，不仅限 /aog-report-gen）

REPRODUCE.md 描述"任何人复现的步骤"，**禁止硬编码具体 IP / 账号 / 密码 / 容器名**：
- ❌ `A5 access: container npu_dev3 on 198.51.100.35`
- ✅ `A5 access: 任何可达的 Ascend950PR 容器；先运行本插件 init.sh，再按 engine/workspace/.ascendc_env.template 填写 workspace/.ascendc_env（A5_HOST / A5_CONTAINER / A5_USER / A5_PASSWORD / CANN_PATH / SOC_VERSION）`

原则：读者的硬件配置和我们的不一样。REPRODUCE.md 的职责是描述**配置指引**（运行 `init.sh` 生成模板 → 保存到 `.ascendc_env` → 下游 skill 自动读），而不是假定读者已经坐在我们的 A5 机器前面。

检查点（audit 模式也应检查）：
- grep `198.51.100.35` / `Ascend2026` / `npu_dev3` / `can_torch_cann_device_1` 等项目-specific 字符串 in REPRODUCE.md → 警告或自动替换成 `$A5_HOST` / `$A5_CONTAINER` 语义引用

## Default language (强制)

**REPORT.md 默认使用中文**（标题、prose 章节、表格表头、统计描述、commentary）。这是硬规则，不是项目偏好。

- 章节标题用中文：`## 核心成果` / `## 精度详情` / `## 架构决策` 等（不是 `## Overall Status` / `## Core Results`）
- 表格表头用中文：`算子 | 精度 | 确定性 | 性能比率 | 状态 | 说明`（不是 `Op | Precision | ...`）
- Prose/narrative 用中文
- **技术术语保留英文**：`verification.json`, `aclrtEvent`, `DataCopyPad`, `Ascend950PR`, `fp32`, `SIMD`, `msprof`, pattern 名 (`P-P62`), DEBT 编号, commit SHA, 文件路径, 命令, 代码片段
- **单位保留原样**：`ms`, `KB`, `GB/s`, `ratio`, `21.02x`
- **对比方向 / 单位歧义声明**可用中英夹杂（如 `> 比率约定（避免歧义）：...`）

所有迁移和反向生成项目都遵守此默认。--init 生成的 skeleton 必须中文。

跟随 memory `feedback_match_doc_primary_language.md`：中文主文档新增内容也必须中文，不引入大段英文 prose。

## Universal table-completeness rule (THE FULL IN-SCOPE COHORT) — 2026-05-20

**The §核心成果 table MUST list the FULL in-scope op set for the project — not just the ops that have shipped.** Readers need to see at a glance: what's done, what's deferred (with reason), what's blocked (with blocker), and what's still pending (with owner / trigger). A table that only shows DONE rows is misleading — it hides the remaining funnel and makes the project look further along than it is.

**Status enum** (extended from the precision-axis enum below):

| Status | Meaning |
|---|---|
| ✅ DONE | precision PASS / PASS_WITHIN_TOLERANCE; archive on origin/main; verification.json present |
| ⚠️ PARTIAL | precision PARTIAL_PASS (some cases waived / hw-floor / etc); archive present but with caveats |
| ❌ FAIL | precision FAIL; archive may or may not exist; needs rework |
| 🚧 IN_PROGRESS | actively being generated (workspace exists, no terminal verification.json) |
| ⏸ DEFERRED | intentionally postponed; cell MUST cite reason (e.g. "CANN arch35 upstream already exists — defer to upstream") |
| 🚫 BLOCKED | structural / dependency blocker; cell MUST cite blocker ID (DEBT-X / P0Y / task #N) |
| 📋 PENDING | scheduled but not yet started; cell MUST cite trigger condition (when this will begin) |

**Sourcing the in-scope op set** (per project_class):

- `kernel_port` (`port_a3_to_a5`) — full set = ROADMAP §F1 cohort OR explicit project scope-doc (e.g. `output/<project>/docs/SCOPE.md`); pending/deferred/blocked rows come from there even if they have no archive yet
- `single_project` (a backward-generation cohort) — full set declared in project scope-doc (typically header); usually small (5-15 ops)

**When generating the table, build the row set from the FULL cohort, then populate columns from each op's verification.json IF the archive exists; for non-archived ops, fill cells with the cohort-doc's reason + Status enum value.**

**Caveat for very large cohorts (>30 ops)**: keep the full table but bucket by level/category. Don't drop rows. If the table grows unreadable, add a "Status breakdown" rollup at the top (DONE: N / PARTIAL: M / DEFERRED: K / BLOCKED: J / PENDING: P) — the rollup summarizes, the table still enumerates.

**Why this rule exists** (user direction 2026-05-20): "can we keep the table list full set of in scope op so that we know which one are done/defer and which one are still pending. this is common rule for all report so please update the skills of report gen." Showing only the done set hides the funnel. Cross-skill rule — applies to /aog-roadmap-maintain and any future doc skill, not just /aog-report-gen.

**Cross-link**: memory `feedback_report_full_inscope_cohort_not_just_done.md` (2026-05-20).

## Canonical layout contract

Per `${CLAUDE_PLUGIN_ROOT}/kb/shared/OUTPUT_PROJECT_LAYOUT.md` §4, REPORT.md has 9 sections (§1–4 / §6 / §7 / §9 required; §5 / §8 recommended):

1. **顶部**（必须）— 项目名 + 1-句范围 + 跳转 REPRODUCE.md 锚点 + 比率方向约定
2. **核心成果表**（必须）— 列：`算子 | 模式 | 精度 | coverage_tier | 确定性 | 来源基线 | 目标耗时 | 性能比率 | 实现方案 | 状态`。**每个 impl/op 是表格行、不是叙述段；perf 列必须带相对基线的比率，禁止裸绝对值；不进任何「翻案/调查/优化手法」叙述——见上「Report ≠ Log」cardinal rule。**
3. **精度详情**（必须）— 分层说明 + 特殊情况（P-P58 tolerance / FMA grouping / OL-83 waiver 等）
4. **确定性验证**（必须）— 表格：每 op 的 3-run SHA256 或 by-construction 论证，对应 DET_POLICY
5. **架构决策**（推荐）— 表格：SIMT/SIMD 选择、tile 尺寸、核分区、KB 规则引用
6. **方法学**（必须）— 参考生成 / 输入集 / 验证 / 性能测量细则
7. **手动编译调试记录**（必须，2026-06-22 owner direction）— 每个 op 必须记录真实的手动 build + 调试过程，证明算子是被实际编译、上机、调通的（非不沾泥 review），并让下一个人能手动重现这条调试路径。至少含：① **手动编译**：精确编译命令 + 平台/CANN 版本/编译 flag + 命中的编译错误及其修法（挂 `EC-*` / 错误码 / `PB-*`）；② **精度调试**：哪一步 MERE/MARE/bit-diff 卡住、如何定位（minimal-repro / 逐 case / 中间张量 dump）、改了什么使其转 PASS；③ **性能调试**（若做了 perf）：msprof 关键读数（scalar/vector/cube/MTE 占比、bound 判定）+ 实际调优手法；④ **如何手动重编 + 重跑调试**（命令 + 入口）。**这是过程记录、属 Report≠Log 的「历史/调查」类**——放在核心数据（§1–4）之后，**绝不上移进 §1**；空着或写「无调试」需显式说明为何（如纯模板复用 0 编译错误）。
8. **Pipeline / harness 发现**（推荐）— 本次 port 反馈到 harness/KB 的经验
9. **总结 + 相关文档**（必须）— 状态总结 + 交叉链接（REPRODUCE, KB_USAGE_LOG, KB_COLD_START, RESULTS if applicable）

Script-generated sections use injection markers:
```
<!-- BEGIN-GEN:core_results --> ... <!-- END-GEN:core_results -->
<!-- BEGIN-GEN:determinism --> ... <!-- END-GEN:determinism -->
<!-- BEGIN-GEN:performance --> ... <!-- END-GEN:performance -->
```

Narrative sections (1, 3 prose, 5, 6, 7, 8, 9) stay human-written — the script does not touch them.

## Phase R0: Mandatory isolated-agent dispatch (2026-05-18, post-incident)

> **The caller agent (main orchestrator / user-facing agent) MUST NOT write
> REPORT.md directly.** All actual report work — init / refresh / audit, table
> injection, narrative reorder, commit — happens inside a freshly-spawned
> `aog-report-gen` subagent. The caller only parses argv, spawns the subagent,
> waits for its status line, and reports the status line up.

### Why

2026-05-07 commit `68d78caf` rewrote REPORT.md from 1191 → 148 lines and
silently dropped L1/L2 per-op detail tables AND the `<!-- BEGIN-GEN:* -->`
injection markers. Caught 2026-05-18 06:39Z by user: "你给我说我要依赖
这种skills生成报告?". Root cause: caller was a main agent under closure-desire
+ context-filling pressure during an in-flight op-gen run; it read the long
doc, decided "this is too verbose", and trimmed. A fresh-context subagent has
no such pressure. The hard rule below structurally prevents the recurrence —
the caller cannot edit REPORT.md because the subagent owns the writes.

### How

Caller MUST invoke:

```
Agent(
  subagent_type="aog-report-gen",
  description="{project}-rg-{N} {init|refresh|audit}",
  prompt=<self-contained brief — see template below>,
)
```

The subagent definition lives at `${CLAUDE_PLUGIN_ROOT}/agents/aog-report-gen.md`
(symlinked to `~/.claude/agents/aog-report-gen.md`). It has Read/Edit/Write/Bash/Grep/Glob
tools — sufficient for the full playbook. It does NOT have access to the
caller's conversation history; it reads disk for ground truth.

### Self-contained brief template (paste into prompt= verbatim, fill the {…} slots)

```
Project: {project_name}     # arch22→arch35 migration or backward generation
Mode:    {init|refresh|audit}
Flags:   {--init / --audit / --output <path> as applicable}

Read ${CLAUDE_PLUGIN_ROOT}/skills/aog-report-gen/SKILL.md and execute Phase R1 → R_{init|refresh|audit} → R2
mechanically. Disk is your only ground truth — ignore any inline framing about
"what the report should look like".

Hard invariants you enforce (also documented in your own agent definition):
1. <!-- BEGIN-GEN:* --> markers MUST be present before refresh; restore if missing
2. Reordering allowed; deletion is not — preserve all sections + content
3. PER_OP_DETAIL.md is regen-able; run the appropriate generator script
4. Honest "verification.json missing" rows — no fabrication

Return ONE status line per the contract in your agent definition (REPORT_OK /
REPORT_INIT_OK / REPORT_AUDIT_OK / REPORT_AUDIT_DRIFT / REPORT_FAILED).
```

### What the caller is FORBIDDEN from doing

- ❌ Directly editing `output/<project>/docs/REPORT.md`
- ❌ Directly running `gen_report_tables.py`
- ❌ Deciding "the doc is too long, let me trim" without spawning the subagent
- ❌ Cherry-picking which sections to update — the subagent updates whatever
  the playbook says is stale; caller doesn't pre-filter

### What the caller IS allowed

- ✅ Pass through user's project / flags / `--output` overrides
- ✅ Read the subagent's status line + actionable details
- ✅ Report status line up to the user verbatim
- ✅ Decide whether to push after the subagent's local commit (subagent
  commits but does not push — push is a caller-level decision)
- ✅ If subagent returns `REPORT_FAILED`, surface the failure to the user
  unaltered; do NOT attempt to fix it in caller context

### Exception (narrow)

The caller MAY:
- Run `--audit` mode directly via the subagent and read the result without
  committing anything (pure read-only is fine)
- Read REPORT.md to answer user questions (read-only fine; write NOT)

Below this point, all R-phases describe what the SUBAGENT executes inside
its isolated context. The caller does NOT execute R1/R_*/R2 itself.

## Phase R1: Dispatch on mode

```
if --audit:    go to Phase R_audit  (dry-run, read-only)
if --init:     go to Phase R_init   (create REPORT.md from template)
otherwise:     go to Phase R_refresh (re-inject tables)
```

## Phase R_init — create REPORT.md skeleton

1. Verify `output/<project>/docs/REPORT.md` does NOT already exist. If it does, abort — use refresh instead.
2. Inspect `output/<project>/src/kernels/`; both named and numbered operator
   directories use the same migration/backward project template.
3. Write skeleton per OUTPUT_PROJECT_LAYOUT §4, with:
   - §1 顶部 populated with project name + the canonical ratio-convention block
   - §2-4 as `<!-- BEGIN-GEN:* -->` / `<!-- END-GEN:* -->` marker pairs (script fills on refresh)
   - §3, §5, §6, §7 (手动编译调试记录), §8, §9 as prose stubs with TODO placeholders — human fills
4. Do NOT run the script yet — wait for user to fill narrative stubs, then `/aog-report-gen <project>` to inject tables.

## Phase R_refresh — inject tables

1. Run `python3 src/scripts/gen_report_tables.py output/<project> --output output/<project>/docs/REPORT.md`.
2. Script reads every `output/<project>/src/kernels/*/verification.json`, generates §核心成果 / §确定性 / §性能 tables, injects between existing marker pairs. Missing verification.json → row with `—` dashes.
3. Report what changed: stdout of the script, plus a diff summary (new rows / changed rows / dropped rows).
4. Cross-check: if a kernel verification.json has `status=PARTIAL` or `pass < total`, print a warning so the narrator doesn't silently accept a regression.

## Phase R_audit — dry-run freshness check

1. Parse existing REPORT.md. Enumerate present `<!-- BEGIN-GEN:* -->` sections.
2. Enumerate all `output/<project>/src/kernels/*/verification.json`.
3. For each kernel, check if its row is present + current (compare `verification.json` fields vs current table row). Flag mismatches.
4. Check all 9 canonical sections are present (by heading). **Missing a REQUIRED section (1, 2, 3, 4, 6, 7, 9) is a HARD DRIFT contributor** — not advisory. (§7 手动编译调试记录 is REQUIRED as of 2026-06-22 owner direction — a report with clean final numbers but no manual compile/debug record is DRIFT.) (Caught 2026-06-20: an audit returned `mismatches=0` while §4 确定性验证 was entirely missing despite verification.json carrying determinism data — the verdict must reflect that as DRIFT.)
5. Check cross-links are alive: REPRODUCE.md, README.md (or ../README.md), KB_USAGE_LOG.md, KB_COLD_START_RETROSPECTIVE.md.
6. **Report ≠ Log lint (FIRST-CLASS DRIFT driver, not advisory)** — run `python3 src/scripts/report_not_log_lint.py output/<project>/docs/REPORT.md`. Its `report_not_log_violations=N` mechanizes the "Report ≠ Log" cardinal rule for hand-authored narrative (previously documented-only / LLM-attention-reliant — the exact gap that let the §一-SIMD-as-a-log edit through). **N>0 → DRIFT even if rows + sections all match.** Report the offending line numbers + check codes (C1 log-event-phrase / C2 narrative-where-table-belongs in §1-2). The lint is conservative (skips `>` blockquote pointers + table-cell pass-markers like `✅ 达标` + `归档:` archive-link lines, excludes the intro 1-句范围) and dual-fixture-tested for no-false-positive — see `src/scripts/orchestrator/tests/test_report_not_log_lint.py`.
6a. **Jargon lint (FIRST-CLASS DRIFT driver)** — run `python3 src/scripts/report_jargon_lint.py output/<project>/docs/REPORT.md`. Its `report_jargon_violations=N` flags internal-harness jargon in the FRONT (核心数据) prose (owner's TOP concern, "cannbot 套壳/骗人"): cannbot / 商用 / PARTIAL_PERSIST / OL-/DEBT- IDs / competitor_mare / pass_a-pass_b / blackbox / lane-health / ratio-gate / bare commit-SHAs / task-IDs / P0-tags. **N>0 → DRIFT.** Conservative: PROSE only (skips tables / `>` blockquotes / headings / code fences), strips backtick code spans (legit `verification.json` etc. survive), excludes intro; jargon in §implementation/§history is allowed. Dual-fixture-tested — see `test_report_quality_lints.py`.
6b. **Archive-link lint (FIRST-CLASS DRIFT driver)** — run `python3 src/scripts/report_archive_link_lint.py output/<project>/docs/REPORT.md`. Its `report_archive_link_violations=N` flags any front core-results table NOT followed by a `归档:` markdown link to its kernel dir / verification.json (owner: "每个表格到 archive 的文件要有链接，早期报告有、我最近的漏了"). **N>0 → DRIFT.** Conservative: only checks core-results tables (header + separator + ≥2 data rows; tiny key=value tables skipped); a link anywhere in the contiguous post-table block satisfies it. Dual-fixture-tested — see `test_report_quality_lints.py`.
6c. **Concision advisory (ADVISORY only — NOT a DRIFT driver)** — run `python3 src/scripts/report_concision_advisory.py output/<project>/docs/REPORT.md`. Its `report_concision_advisories=N` nudges over-long front prose/blockquote blocks toward "data→table, prose→short 关键读法" (owner: "正式报告行文但不会太啰嗦"). Always exits 0 — surface as a non-blocking note, NEVER count toward DRIFT.
7. **0 `<!-- BEGIN-GEN:* -->` markers on a marker-style project → HARD WARNING** — a future `--refresh` would be a silent no-op (the script injects nothing). (Hand-authored single-project reports legitimately have no markers; flag only when project_class implies marker-based refresh.)
8. Print audit report. **DRIFT (exit 2) if ANY of: row mismatch / missing required section / `report_not_log_violations>0` / `report_jargon_violations>0` / `report_archive_link_violations>0` / silent-no-op markers.** (Concision advisories do NOT contribute to DRIFT.) Exit 0 only if all green.

## Phase R2: Verify + commit

After refresh or init:
1. `git diff output/<project>/docs/REPORT.md` — sanity-check the edit.
2. **Report ≠ Log gate (BLOCKING, write-time)** — run `python3 src/scripts/report_not_log_lint.py output/<project>/docs/REPORT.md`. If `report_not_log_violations>0`, do **NOT** commit — fix the flagged front-section log narrative first (move the how-we-got-here / event-phrases to a later 历史/调查 section; keep §1-2 comparison-data-only). This catches a Report≠Log regression at write-time, not only at the next audit. (The original incident was an inline hand-edit that never ran the skill — so ALSO apply this lint manually after any inline REPORT.md edit, per the "Report ≠ Log" cardinal rule's EDIT-TRIGGER.)
2a. **Jargon gate (BLOCKING, write-time)** — run `python3 src/scripts/report_jargon_lint.py output/<project>/docs/REPORT.md`. If `report_jargon_violations>0`, do **NOT** commit — rewrite the flagged front prose into plain customer-facing language (move cannbot / OL-/DEBT- / PARTIAL_PERSIST / commit-SHA / task-ID jargon out of §1-2 into §implementation/§history, or drop it). Owner's TOP report-quality concern.
2b. **Archive-link gate (BLOCKING, write-time)** — run `python3 src/scripts/report_archive_link_lint.py output/<project>/docs/REPORT.md`. If `report_archive_link_violations>0`, do **NOT** commit — add a `归档:` markdown link under each flagged core-results table pointing at its kernel dir (and `verification.json` when present), e.g. `归档: [<op>](src/kernels/<op>/) · [verification.json](src/kernels/<op>/verification.json)`. (Path is relative to the REPORT.md location.)
2c. **Concision advisory (NON-BLOCKING, write-time)** — run `python3 src/scripts/report_concision_advisory.py output/<project>/docs/REPORT.md`. Surface its `report_concision_advisories=N` nudges, but it NEVER blocks the commit (advisory only — a writing nudge, not a gate).
3. If refresh: `git add` + commit with message `aog-report-gen: refresh <project>/docs/REPORT.md from verification.json`.
4. If init: do NOT auto-commit (human needs to fill narrative stubs first).

## Migration/backward project status

The core-results table covers the full in-scope migration or backward cohort.
`precision.status` alone determines the status column: PASS and
PASS_WITHIN_TOLERANCE map to DONE, PARTIAL/PARTIAL_PASS map to PARTIAL, FAIL maps
to FAIL, and a missing `verification.json` remains explicit. Performance stays in
its own comparison column and never changes the precision status.

All projects use the standard `core_results`, `determinism`, and `performance`
injection markers. Large cohorts may be grouped by operator category while still
listing every in-scope row.

## Limitations (honest)

- Script-generated tables only cover 3 sections (§核心成果 / §确定性 / §性能). Other sections are human-written; this skill does not auto-generate them.
- Skill does not validate narrative content for correctness — it only checks section presence and cross-link aliveness.
- No anti-drift between REPORT.md narrative claims and verification.json numbers (hard problem; detect only if numbers quoted verbatim in narrative).
- Init template is static; if OUTPUT_PROJECT_LAYOUT §4 evolves, this skill must be updated (tied by convention, not by import).

## References

- `${CLAUDE_PLUGIN_ROOT}/kb/shared/OUTPUT_PROJECT_LAYOUT.md` — canonical directory + REPORT structure
- `src/scripts/gen_report_tables.py` — the table-injection engine
- Cross-generation and backward-generation reports produced by this plugin are the only examples in scope.
