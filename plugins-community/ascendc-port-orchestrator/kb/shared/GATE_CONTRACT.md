# GATE_CONTRACT.md — finalize gate requirements + verify-phase mandates

**Purpose**: this file is the canonical source of truth for what the
finalize → done gate (orchestrator side, `finalize_pipeline.check_finalize_eligibility()`)
checks for. Worker briefs reference this file by section anchor instead of
inlining the full text — V4 backend hit 300s timeout when kw_brief grew
past ~22K chars (DS finding 2026-05-08).

**Worker workflow**: Read this file at the start of Phase D — i.e. AFTER
kernel write + build pass but BEFORE running `verification_ascendc.py`.
The gate text below tells you what verification.json fields the gate
will inspect; load them now so you don't ship a kernel that the gate
will reject for a metadata gap.

**Format invariant**: every section header is also referenced by the lean
agent briefs (kw_brief.py, ko_brief.py, etc.). If you rename a section,
update the brief callers in the same commit.

---

## §Phase D Verify Gate

### model.py 接口铁规 (P0abg, 2026-05-08)

任何 workspace 里的 `model.py` 和 `model_cpu_truth.py`（Path-A CPU truth 分离文件）
必须定义 `get_input_groups()`，**绝不允许**只定义 `get_inputs()`。这是 vendor
benchmark 接口规范 —— vendor `benchmarks/NPUKernelBench/levelN/<op>.py`
的全部 31+ 算子都用 `get_input_groups()`，没有一个用 `get_inputs()`。
哪怕只有单个 case，也要返回单元素列表：

```python
def get_input_groups():
    # vendor 标准做法：从 sibling JSONL 加载多 case
    return [build_inputs(case) for case in load_jsonl(__file__)]

# NOT ALLOWED:
# def get_inputs(): return [tensor_a, tensor_b]
```

违规后果：finalize gate 直接 reject (P0abg)，rollback 到 `await_worker`
让你重写。verifier 内部对 `get_inputs()` 有 fallback 把它包成
`[inputs]`（pass_a.total=1）—— 这是历史遗留兼容性，不是允许形态。
我们 harness 的 lint gate 在 finalize 前拦下违规，避免触发 verifier
的 1/N 静默欺诈。

### model_cpu_truth.py — Path-A 精度 oracle 分离 (P0xxx, 2026-05-15)

当 prior-art 的 `Model.forward` 调用 `torch_npu.*` CANN op 无法在 CPU 上运行时，
worker 需要将性能/研究基线与终验 truth 分离，不能用 target NPU 输出补位：

| 文件 | 角色 | 使用者 |
|---|---|---|
| `model.py` | 可选 target prior-art / 性能研究基线 | `performance.py` 诊断性性能对比；不得作为终验 truth |
| `model_cpu_truth.py` | CPU fp64/autograd 精度 oracle | 反向生成终验；迁移终验另用 selected arch22 source-arch NPU capture |

规则：
- `model.py` 可保留 target prior-art 的原始实现（调用 `torch_npu.*`），但只作研究/性能基线
- 反向生成必须提供可追溯到前向契约的 `model_cpu_truth.py`（CPU fp64/autograd）
- 迁移必须提供当前 selected arch22 source 的 source-arch NPU capture
- 两个文件都必须定义 `get_input_groups()`
- `performance.py` 可以 target prior-art 为对比基线，但必须标为 diagnostic/advisory
- 缺少相应 source-arch capture 或 CPU fp64/autograd oracle 时，终验必须
  `FAIL_NO_INDEPENDENT_TRUTH`，不得 fallback 到 `model.py`

### Workspace 必须有 `{op}.json` (P0abf, 2026-05-07)

Phase O2 自动 sync 一份过来。如果没看到，从
`BENCHMARK_ROOT/<level>/{op}.json` 手动 cp。

### 合法跳过 case 的方式

Populate `precision.pass_a.skipped_cases` 为
`[{"case_idx": <int>, "reason": "<explicit reason>"}, ...]` —
每个跳过的 case 一项，必须给具体原因（CANN reject / 硬件限制等）。
`total + len(skipped_cases) >= benchmark_count` 才被 P0abd gate 接受。
单纯减小 total（"preserved kw-N baseline 1/1"）= 静默欺诈 = 拒绝。

### §4.5.3 small-value rule (P0abh, 2026-05-08)

Vendor 精度标准 2.1 §4.5.3 规定，当 `|golden|` 小于 dtype 对应的
small-value threshold（fp16: 2^-11, bf16: 2^-8, fp32: 2^-14）时，
相对误差度量 (MARE/MERE) 不稳定（除以接近 0 的分母放大），改用
绝对误差计数：`ErrorCount_npu ≤ 2 × max(ErrorCount_cann, 1)`。

**如果你的 inline `verification_ascendc.py` 报告 MARE 失败（高 MARE +
低 MERE 同时出现，提示是小值域放大伪 FAIL）**，立即跑：

```bash
python3 src/scripts/precision_eval_two_tier.py current_task --json /tmp/p2t.json
```

检查 `--json` 输出每个 case 的 `verdict`：
- `PASS_T1_SMALLVAL` = §4.5.3 救出来的 case，按 PASS 计数
- `PASS_T1` / `PASS_T2` = 标准路径，PASS
- `FAIL` = 真实精度差，不要用 §4.5.3 美化

在 verification.json `precision.pass_a.case_detail[i]` 加字段
`smallval_rule_fired: true` + `smallval_evidence`，并把这些 case
算进 `tier1_pass`。规则参考：
`src/skills/references/target/ascendc/PRECISION_STANDARD_v2.1.md` §4.5.3。
KB 索引：`src/skills/references/KB_INDEX.md` "Vendor-authoritative
Standards" 区块。

### Pass A / Pass B / Det / Perf checklist

- Pass A (inline benchmark): **MUST run ALL benchmark cases via canonical evaluator** (see below)
- Pass B (edge_dataset): bit-exact vs CPU truth
- Det: 3-run bit-identical
- Perf: **performance.py on ALL benchmark cases; ratio = mean of per-case speedups** (see §Performance Measurement Contract below)

### Performance Measurement Contract (P0xxx 2026-05-14)

**Requirement**: Worker MUST run `performance.py` on ALL benchmark cases
before emitting any non-FAIL handoff. One-case measurement is insufficient
— the perf gate compares the MEAN speedup across all cases against the
0.6x threshold.

**Canonical command**:
```bash
python3 utils/performance.py --output_dir current_task --warmup 5 --repeats 50 --output current_task/perf.json --markdown current_task/perf.md
```

**Fields to write into `verification.json.performance`**:
- `ratio` = perf.json.overall_speedup (mean of per-case speedups; CANONICAL,
  used by state machine gate and finalize gate)
- `ratio_mean` = same value (alias, backward compat)
- `ratio_median` = median of per-case speedups
- `ratio_min` / `ratio_max` = worst/best individual case speedup
- `n_cases` = number of cases successfully measured
- `threshold` = 0.6
- `harness` = the exact command used
- `status` = "PASS" if ratio >= 0.6 else "BELOW_THRESHOLD"

**The `ratio` field is the arithmetic mean** of per-case speedups across
ALL benchmark cases. This matches `performance.py`'s `overall_speedup`
computation (`statistics.mean(speedups)`). Geometric mean is NOT used for
the gate threshold.

**Failed cases**: If performance.py fails on some cases, report `n_cases`
of successfully measured cases only, skip failed cases in the mean, and
note failed case indices in a `failed_cases` field. `performance.py`
already handles this — if either reference or ascendc measurement fails
for a case, that case is absent from `per_case_speedup` and excluded from
the mean.

**OL-95 wall-clock check**: In addition to the multi-case performance.py
run, for the LARGEST case only, run an independent wall-clock measurement
to detect profiler noise-floor issues. Record under
`performance.independent_re_measure` per the existing contract.

### Canonical evaluator at Phase D (P0aau-c35, MANDATORY)

**Scope of fix**: catches API contract mismatches (multi-output return-shape divergence,
single-output schema mismatch where reference returns `Tuple[...]` but kernel returns
`Tensor`), dtype templating gaps (kernel fp16-only but fixture has bf16/fp32 cases),
benchmark coverage divergence between worker self-test surface vs Phase O5 SSH-based
canonical surface (P0kk).

**Origin (C35 audit, 2026-05-09)**: 3_FusionAttention cold-start shipped through 24
spawn.start events with a structurally-detectable API contract bug
(`ModelNew.forward` returned 1-D zero-length placeholder for FlashAttention auxiliary
outputs; reference returned `(B,N,S)` shape). Worker self-test passed via
`verification_ascendc.py` against pilot edge_dataset. The bug was caught only at
Phase O5 by `precision_eval_two_tier.py` running on the full fixture — 4
`phase_o5_block` events and a kw-4 fix-spawn ($4.18 + 11min wallclock) before
finalize. C35 (self-evolution gap) catalog filed; this section is the structural fix.

**Requirement at Phase D**: worker MUST run the canonical evaluator on the full
benchmark fixture before emitting any non-FAIL handoff. The canonical evaluator is:

```bash
python3 src/scripts/precision_eval_two_tier.py current_task --json /tmp/p2t.json
```

This evaluator:
- Iterates **all** cases from `model.py.get_input_groups()` (not just the pilot
  edge_dataset)
- Runs each case through the full triple `(cpu_model, npu_model, cand_model)`
- Compares ALL return tensors (main + auxiliary) against reference per OL-109
  T1/T2 framework + §4.5.3 small-value rule
- Returns per-case `verdict` (PASS_T1 / PASS_T2 / PASS_T1_SMALLVAL / FAIL / EVAL_ERR)
- Identical surface to Phase O5 SSH-based runner — same evaluator, same expected output
  schema → if worker's run passes, Phase O5's run passes (no surprise mismatch)

**Why this matters generally** (cross-op impact): catches the same bug class at the
**cheapest detection layer** for ALL ops where it applies. C35 cousin-op estimate:
~8 multi-output fused ops in current bench backlog (6_MoeFinalizeRouting, 9_TopKTopP,
10_SwigluQuant, 11_DequantSwigluQuant, 12_KvRmsnormRopeCache,
20_FusedRopeWithQkNormAndKvCacheUpdate, 27_MultiMaskAttentionAggregation,
29_TanhGatedResidualAddBackward) all share the structural risk and will benefit
from worker-side canonical evaluation.

**The legacy `verification_ascendc.py` inline path is allowed for fast iteration
during compile-fix loops** (when worker is iterating on a build-error or initial
precision residual). But:
- Final emission of any non-FAIL handoff (`done`, `PARTIAL_PERSIST`, `@aog-precision-probe`
  with verdict-class claim) MUST cite a recent canonical evaluator run
- The canonical run's `/tmp/p2t.json` output MUST be referenced in `verification.json`
  `precision.pass_a.method` field — the gate's pre-finalize check verifies this
  reference is non-empty

**verification.json format with canonical evidence**:
```json
{
  "precision": {
    "pass_a": {
      "status": "PASS",
      "tier1_pass": <int>,
      "total": <int>,
      "method": "precision_eval_two_tier.py canonical run on /tmp/p2t.json (timestamp)",
      "case_detail": [...]  // sampled or full, with per-case verdict
    }
  }
}
```

If the canonical run reports any `EVAL_ERR` cases (especially `shape mismatch` /
`schema mismatch` / `API contract` reasons): worker MUST diagnose + fix BEFORE
emitting any handoff. EVAL_ERR is structural, not numerical residual — does not
qualify for `@aog-precision-probe` (which is for numerical residuals); requires
worker to fix in same spawn or escalate to `@aog-fused-optimizer` if architectural.

### MANDATORY artifacts for finalize gate (P0aaf #108 + P0aba 2026-05-07)

The orchestrator's finalize → done gate REJECTS workspaces missing any of:

1. **`run_pass_b.py`** at workspace root — canonical filename for the
   independent Pass-B verifier. Phase O5 post-verify runner looks for
   this exact name. If Pass B is feasible (reference runnable, edge
   dataset usable), write a small wrapper script that loads
   `edge_dataset.pt`, runs the kernel + the reference, asserts each
   case bit-exact, prints "PASS_B_VERIFY: pass_a={N}/{TOTAL}". The
   orchestrator runs this independently of your inline verification.

2. **`verification.json` `precision.pass_b`** — concrete fields:
   - If Pass B ran: `{"status": "PASS", "tier1_pass": <N>, "total":
     <N>, "method": "<verifier name>"}`. The gate rejects PASS without
     concrete `tier1_pass` + `total` integers (no self-claim strings).
   - If Pass B is infeasible (Path A / OL-68 Case A — torch_npu
     reference unrunnable on most cases): `{"status": "N/A", "reason":
     "<explicit OL-68 Case A or harness-level reason>"}`. The gate
     rejects N/A without a reason string.

3. **`verification.json` `performance.independent_re_measure`** —
   either `{"ran": true, "ratio": <num>, "delta_vs_kw_self_report":
   <num>}` (you ran performance.py from a separate invocation, not
   the same process that did inline verify), OR `{"status": "N/A",
   "reason": "<Path A / explicit reason>"}`. CLAUDE.md hard rule:
   "NEVER trust skill-reported performance numbers — always re-run
   performance.py independently". The gate rejects perf=PASS for
   non-Path-A ops without this field's substantive contents.

4. **`verification.json` `performance.ratio_baseline`** — string that
   makes clear what the ratio is vs. A target CANN NPU op may be named as
   an advisory performance baseline, but this field never defines precision
   truth. Precision truth is selected-source capture for migration and CPU
   fp64/autograd for backward generation.

5. **Runner INVOCATION contract (P0aba — prevents exit-2 / JSON-parse rollback)**:
   `pass_a_runner.py` / `pass_b_runner.py` MUST be **bare-invocable** — Phase O5 re-runs
   `python3 pass_b_runner.py` with NO positional argument, and ALSO with an explicit
   `--json <path>`. So:
   - any `kernel_dir` positional MUST be OPTIONAL:
     `ap.add_argument("kernel_dir", nargs="?", default=".")`. A REQUIRED positional makes
     argparse `exit(2)` on the bare invocation → O5 `RUNNER_FAILED`.
   - the canonical JSON summary MUST be emitted on BOTH paths: the **LAST stdout line**
     (`print(json.dumps({"label": "pass_b", **summary}))`) AND written to the `--json`
     file when one is passed. If only one path emits JSON, the other O5 invocation fails
     "verifier stdout had no parseable JSON" → rollback.
   Anchor: rms_norm_backward_add (T5 op-gen 2026-06-29) spent 2 rollback passes on exactly
   this; the conforming runner is the reference pattern.

The orchestrator additionally produces (you do NOT need to write
these — they get auto-produced before finalize):
- `audit_self_critic_post_worker.md` (via `claude --print
  --skill aog-self-critic` post-worker)
- `.delegation_scan_passed` marker (via `scan_delegation_cheating.py`)

But you MUST write the verification.json fields above; the
orchestrator does NOT synthesize precision/performance numbers
from your output — it only audits what you wrote.

---

## §Phase E Knowledge Update

### knowledge_update.md schema (P0aax 2026-05-07, MANDATORY)

Write `workspace/{op}/knowledge_update.md` (≥ 100 bytes body, target
~500-2000) capturing **everything you learned during build/probe/verify
that future cold-starts should know**. The file is the ONLY mechanism
that propagates per-op learnings into KB.

Required structure (use these section headers verbatim — `aog-knowledge-maintain`
skill parses them):

```
# knowledge_update.md — {OP_NAME}

## Context
<one paragraph: op signature, dtypes, shape range, op-class>

## Findings
<numbered list — what surprised you / what you had to fix / what worked.
Each finding = one paragraph with concrete evidence (error msg, hex bit
pattern, perf delta, etc.). At least 1 finding required. Empty list →
reject.>

## KB-promotable patterns (proposed)
<list candidates for OL-/EC-/PB-/P-P entries with rationale. OK to leave
empty if nothing rises to KB level — but state "no new patterns" explicitly
so reviewer knows the worker considered it.

For each candidate, include scope tags so the kb_manager can apply the
regression-risk gate without re-deriving them:

- `applies_to: soc=...; cann=...; bisheng=...; op_class=...`
- `verified_on: soc=...; cann=...` (omit if same as applies_to)
- `unverified_on: soc=...` (when you suspect cross-arch may NOT transfer
  — e.g., "verified on A5/Ascend950PR but unknown for A3/Ascend910_V220")

When in doubt about cross-arch transfer, encode the uncertainty as
`unverified_on:` rather than silently broadening `applies_to=all`. The
kb_manager will route uncertain candidates to candidates.md for review
instead of merging them as authoritative.>

## Cited KB items
<which OL/EC/PB/P-P entries you actually loaded and used. Empty = either
you didn't load any, or you loaded but didn't use — both cases need the
"cited: none, reason: ..." line.>

## Anti-patterns avoided
<if you considered a known anti-pattern (P5 packaging, OL-83 waiver, etc.)
and chose differently, name it. Empty if none applicable.>
```

**Hard floor**: file present + body ≥ 100 bytes + at least 1 finding entry.
`finalize_pipeline.check_finalize_eligibility()` will REJECT precision=PASS
handoffs that lack this file, rolling back to `await_worker` with directive
"write knowledge_update.md only".

**Why mandatory**: the entire harness's ability to scale to harder ops
depends on per-op KB accretion. A "done" without `knowledge_update.md`
is silent learning loss — the next op that hits the same gotcha pays
the discovery cost again. P0aax regression caught this on op#6_QuantMatmul:
kw discovered AIC+AIV mixed-file build emits only AIC `.o` (split needed)
AND dequant write pipeline needs TQue not TBuf — both findings only in
PROGRESS.md, never reached KB.

### P-P88 compliance block (P0abi 2026-05-08, MANDATORY when applicable)

**Trigger**: kernel emits `AscendC::Tanh` (any namespace/template variant)
AND op-class is transcendental (gelu/silu/sigmoid/tanh/softmax/erf-bearing).

Scope narrowed 2026-05-08 post-DS portfolio scan: PB-24 Tanh bimodal
small-x failure is the regression-producing primitive; PB-25 Sigmoid
uniform 2-ULP doesn't have the failure mode and is NOT enforced.

If your kernel calls Tanh on a transcendental op, you MUST add a
structured YAML block to knowledge_update.md (the deterministic
compliance gate parses it):

```yaml
p_p88:
  status: applied | exempt | not_applicable
  primitives_detected: [Tanh]              # gate currently scoped to Tanh
  evidence:
    files: [kernel/<op>_kernel.h:LO-HI]   # line range of the call(s)
    rationale: "<concrete prose explaining the status>"
    isolated_primitive_measurements:      # REQUIRED if status=exempt
      - input_range: "|x| > 0.1"
        measured_ulp: 2
        vs_cpu_truth: true
  diff_refs: [kernel/<op>_kernel.h:42-67] # REQUIRED if status=applied
```

- `status: applied` — you rewrote to sigmoid-form (Exp + Reciprocal +
  Add per P-P88 vendor-source evidence). `diff_refs` lists where the
  rewrite lives.
- `status: exempt` — kernel keeps the primitive AND you have concrete
  isolated-primitive measurements showing PB-24/25's small-x failure
  regime doesn't reach this op (e.g., bounded input domain).
  `evidence.isolated_primitive_measurements` REQUIRED.
- `status: not_applicable` — primitive is in kernel but op-class isn't
  transcendental (gate is scoped to transcendental ops only — this
  status is rare).

**Why mandatory** (P0abi origin): 1_GELU regressed 50/50
(May-4 archive PASS_WITHIN_TOLERANCE) → 44/50 (May-8 cold-start PARTIAL)
because cold-start kw cited P-P88 as diagnosis but didn't apply the
prescribed sigmoid-form rewrite. Different tile-size choice (4096 → 6144)
routed Tanh primitive's internal SIMD differently on small-value inputs,
exposing PB-24's bimodal floor. Same op, same math, different
kernel-quality. The gate makes the cite-vs-apply distinction structural
— kw can't ship a kernel that uses Tanh on a transcendental op without
explicit YAML evidence either way.

---

## §P0cc Dual-count verification schema (2026-05-23, flat_quant incident)

**Trigger**: when `precision.pass_a.status` (or `pass_b.status`) is in the inclusive-promotion set:
- `PASS_WITHIN_TOLERANCE`
- `PARTIAL_PASS_WITHIN_TOLERANCE`
- `PARTIAL_PASS`

**Required schema in verification.json**:
```json
"precision": {
  "pass_a": {
    "status": "PASS_WITHIN_TOLERANCE",
    "tier1_pass": 4,              // STRICT bit-exact count only
    "tier1_pass_inclusive": 8,    // T1_BIT_EXACT + T2_PASS_WITHIN_TOLERANCE
    "total": 8,
    ...
  }
}
```

**Required schema in pass_a_runner.py stdout JSON tail** (consumed by phase_o5_runner._try_parse_json_tail):
```python
summary = {
    "n_pass": n_pass_strict,                  # legacy alias for tier1_pass
    "n_total": n_total,
    "tier1_pass": n_pass_strict,              # bit-exact only (T1_BIT_EXACT verdict count)
    "tier1_pass_inclusive": n_pass_inclusive, # T1_BIT_EXACT + T2_PASS_WITHIN_TOLERANCE
    "total": n_total,
    "status": "PASS" if n_pass_strict == n_total else (
        "PASS_WITHIN_TOLERANCE" if n_pass_inclusive == n_total else "FAIL"),
}
print(json.dumps(summary))
```

When per-case verdict logic distinguishes T1 vs T2 (e.g. bf16 quant op with qscale tolerance, fp8 dequant with E8M0 rounding), the runner counts them separately. For pure-PASS ops where every case is bit-exact, `tier1_pass_inclusive == tier1_pass` — both fields still emitted for schema compatibility.

**Why Phase O5 needs dual-count**:

Without dual-count, when worker emits `precision.pass_a.status = PASS_WITHIN_TOLERANCE` with `tier1_pass=4` strict, Phase O5 re-runs the runner and compares strict `tier1_pass=4` claimed vs strict `tier1_pass=?` measured. If the kernel has any cross-run non-reproducibility (workspace contention from concurrent op deploy / non-deterministic quantization / stale binary not re-built / different inputs on re-sync), O5 measurement can drift, producing MISMATCH → infinite finalize→await_worker loop. Caught flat_quant 2026-05-23 (84 phase_o5_mismatch rollbacks in 37 min before P0bb-loop-guard landed).

With dual-count, when worker declares PASS_WITHIN_TOLERANCE, phase_o5.py compares `tier1_pass_inclusive` (more stable across re-runs since it admits T2 cases) instead of strict `tier1_pass`. The inclusive count better matches the worker-declared semantic ("kernel passes within tolerance"), so O5 re-measurement is robust to per-case T1↔T2 drift.

**Backward-compat**: pure-PASS path (status NOT in inclusive-promotion set) falls through to the legacy strict-only comparison unchanged. EXACT_FIELDS = ("tier1_pass", "total") for those statuses.

**Cross-ref**:
- `phase_o5.py`: INCLUSIVE_FIELDS / INCLUSIVE_STATUSES + per-pass status-aware compare picker
- `kw_brief.py`: port_a3 phase block pre-done checklist assertion
- EC-59: incident + fix-path codification
- memory `feedback_no_patch_fix_harness_for_next_customer.md`: rule for "fix harness not patch single archive"

---

## §P0ff — Session-directive cross-check (knowledge_update.md schema, 2026-05-23)

**Owner directive 20:48Z 2026-05-23**: when main-agent writes a god-mode-researcher
directive in `workspace/<op>/user_decision.md`, orchestrator auto-extracts strategic
content into `kb_draft_from_user_decision.md`. Worker MUST validate/extend that
draft so kb_manager Mode 1 can promote into canonical KB → customer cold-clone
reproducibility without session-state band-aid.

**When to add**: ONLY when `workspace/<op>/kb_draft_from_user_decision.md` exists.
(If absent, skip this section — no user_decision.md was consumed in this op's history.)

**Required schema** in `knowledge_update.md`:

```
## Session-directive cross-check
- kb_draft_anchor: <one-line quote of kb_draft's `rule:` field>
- rule_confirmed: yes / no / extended (one-line how)
- evidence_from_my_work: <file:line in my emitted kernel/test that
                            supports OR refutes the rule>
- additional_applies_to_found: <if my work shows pattern applies to MORE
                                  op-classes than draft claimed; or
                                  "none — scope as stated">
- rule_caveats: <edge case / version dep / SoC dep observed during this
                  iter that should annotate the promoted KB entry>
```

**Semantics**:
- `rule_confirmed: yes` → kb_manager Mode 1 promotes kb_draft as-is
- `rule_confirmed: extended` → kb_manager merges `additional_applies_to_found`
  + `rule_caveats` into the kb_draft body before promoting
- `rule_confirmed: no` → kb_manager does NOT promote; writes
  KB-correction-pending marker to KB_USAGE_LOG.md flagging contradiction for
  owner review

Worker is a mini-researcher here — validate + extend the owner's distilled
directive, don't just echo. Closes the loop: owner white-box → kb_draft →
worker validation → canonical KB → customer cold-clone reproducibility.

**Cross-ref**:
- `src/scripts/orchestrator/orchestrator.py::_extract_kb_draft_from_user_decision` —
  auto-extraction primitive
- `src/skills/aog-knowledge-maintain/SKILL.md` Mode 1 — dual-source read +
  CONFIRM/EXTEND/REFUTE matrix
- `src/scripts/orchestrator/tests/test_p0ff_kb_draft_from_user_decision.py` — 8 regression tests


## §D.3a-port_a3 — FAILURE-ANALYSIS SOP (port_a3 mode only, mandatory)

Referenced from `kw_brief.py` Phase D.3a. Codified 2026-05-21 (post fused_quant_mat_mul
case 6 reward-hacking incident); externalized to here 2026-05-24 per DEBT-114 brief-size
trim. **Mandatory for port_a3 mode workers** when ANY case fails D.3 with
`max_abs_diff >> Tier-1 threshold`.

**Concrete trigger**: `max_abs_diff > 10× tier1_atol` for T1 ops, OR `max_abs_diff > 1.0`
absolute for T2 ops, OR `max_rel_diff > 100×`.

**FORBIDDEN claims under this trigger**:
- "OL-109 Tier-2 hardware floor / bf16 drift / ULP-floor" without source citation
- "A3 reference is non-canonical" without source citation
- "my kernel matches my model.py bit-exact, A3 is wrong"
- Running a hypothesis sweep on scale variations (×0.5, ×2, ×scale_a, ×scale_b, swap
  operands, etc.) — that's reward hacking on the reference, not debugging
- Declaring PARTIAL_PERSIST + shipping the archive

**REQUIRED steps before any precision-failure verdict**:

1. **Re-read A3 op_kernel** for THIS specific case's code path. Identify the algorithm
   sequence the case actually executes:
   - If `dtype_x1=int8, dtype_x2=int8, dtype_y=bf16, has_bias=true, has_gelu_erf=true`
     → read the matching template specialization (e.g.
     `EpilogueDequantGeluErf<bf16-out>::DequantCompute` + `ComputeGeluErf`), NOT a
     sibling specialization.
   - Identify EVERY transformation in order: scale→quant, dequant, bias, activation,
     cast. Cite file + line for each.

2. **Check your `model.py` against the citations**. For each transformation listed,
   does `model.py` perform the SAME transformation in the SAME order with the SAME
   numeric semantics (rounding mode, dtype promotion, broadcast axis)? If any
   mismatch, fix `model.py` first.

3. **Re-run Phase D after `model.py` fix**. If the case now passes, the original
   "failure" was a reference bug — close the loop, archive normally.

4. **Only if step 3 still fails** are you allowed to investigate A5-kernel-side
   causes. The next allowed claim is "A5 kernel output differs from CPU-truth
   model.py (which now matches A3 source semantics)" — that points the finger at the
   A5 kernel itself, not at A3.

5. **OL-109 Tier-2 / hardware-floor citation is allowed ONLY if**:
   - (a) you've completed steps 1-3 above with a `model.py` proven to match A3 source
     semantics line-by-line in `analysis.md`,
   - (b) the residual diff is at fp16/bf16 LSB scale (`max_abs_diff ≤ 2 LSB` of output
     dtype),
   - (c) you can cite the specific Ascend primitive (e.g., `Tanh`, `Sigmoid`,
     `AscendDequant`) whose hardware-precision floor matches the residual.

**Why this SOP exists**: 2026-05-21 fused_quant_mat_mul case 6 — worker found
`max_abs_diff=1.857` (way outside any LSB floor), ran a 14-hypothesis sweep on scale
variations, never read the A3 source's `EpilogueDequantGeluErf` template, and declared
"A3 non-canonical / PARTIAL_PERSIST". User caught: "what reward hacking — I want to
know where the difference actually is". The actual root cause was in the model.py
reference (missing `CAST_RINT` + missing `AscendDequant` bf16-scale path semantics) —
exactly what step 1-3 above would have caught in 15 minutes.


## §kw-exit-handoff — full handoff option catalog (kw_brief externalized)

Referenced from `kw_brief.py` EXIT HANDOFF OPTIONS section. Codified across
2026-05-04 (DEBT-077 P0g op#10 catch) + 2026-05-20 (historical implementation design
§4.3 structural_rewrite_needed) + 2026-05-23 (P0cc dual-count); externalized
here 2026-05-24 per DEBT-114 brief-size trim.

### Pre-done file-existence checklist (mandatory)

Before emitting `→ orchestrator: done`, verify on disk:
- `workspace/{op}/pass_a_runner.py` exists
- `workspace/{op}/pass_b_runner.py` exists
- `python3 src/scripts/orchestrator/check_verification_schema.py workspace/{op}/verification.json` exits clean

Schema requires `precision.pass_a.tier1_pass` (int) + `total` (int); `pass_b`
requires same OR `status=N/A` with `reason`. P0cc (2026-05-23): when
`pass_a.status` ∈ `PASS_WITHIN_TOLERANCE` / `PARTIAL_PASS*`, ALSO requires
`tier1_pass_inclusive` (T1+T2). See §P0cc.

**Empirical anchor** (Zheng 2026-05-21 catch): 7_Sum kw-1 emitted `done` with
kernel + canonical model wrapper + verification.json + verify.py but NO
`pass_a_runner.py` / `pass_b_runner.py`. Finalize phase_o5 rolled back →
kw-2 wrote missing runners (~5-10 min wasted). Same pattern hit 22_Nonzero
kw-1 + 20_Gather kw-1 — three documented occurrences in one day.

`verify.py` is NOT a substitute for `pass_a_runner.py` / `pass_b_runner.py`:
`verify.py` is the inline-verification helper the kernel imports; the
`*_runner.py` files are standalone scripts orchestrator runs independently
in Phase O5. Both required, different consumers.

### Handoff-line format (REQUIRED — single token, last line, no inline @aog-X)

The orchestrator's `extract_canonical_handoff` reverse-scans your stdout / PROGRESS
tail and takes the **LAST** line matching a canonical handoff prefix, then
`_parse_worker_signal` maps it to a state transition. Two rules keep that extraction
unambiguous:

1. **The verdict line is the LAST non-empty line** you emit (nothing — no prose, no
   summary — after it).
2. **It carries EXACTLY ONE handoff token and NO inline `@aog-X` substring.** A
   `→ orchestrator: <verdict>` line that ALSO mentions `@aog-fused-optimizer` (or any
   `@aog-X`) — on the same line OR on a later line — is AMBIGUOUS: the reverse-scan can
   grab the `@aog-X` agent-handoff over your `→ orchestrator:` verdict, and if that
   `@aog-X` form isn't a recognized transition from the current state, `_parse_worker_signal`
   returns `unknown` → spurious `await_worker → abort` ("worker exited without recognized
   handoff — contract violation"). Put any "recommend routing to @aog-X" reasoning EARLIER
   (in the analysis body). If you genuinely intend an agent handoff, emit JUST the `@aog-X`
   form as the last line (no `→ orchestrator:` prefix), not both.

**Worked example (the bug this rule prevents)** — flash_attention_score_grad kw-1
(2026-06-20): the kw correctly verdicted `→ orchestrator: structural_rewrite_needed — …
Recommend routing to @aog-fused-optimizer OR adopting the multi-launch approach.` on ONE
line. `extract_canonical_handoff` matched the trailing `@aog-fused-optimizer` (later in the
line) instead of the `structural_rewrite_needed` verdict → `unknown` → the run aborted at
`await_worker` instead of routing to the structural-rewrite path (await_worker HAS that
route). The verdict + the route both existed and were correct; ONLY the line format tripped
it. Fix forward: verdict alone on the last line, routing recommendations in the body above.

### Allowed handoffs

- `→ orchestrator: done — <summary>` — precision PASS + det PASS; orchestrator
  evaluates perf separately (V3.8.4 forcibly escalates to optimizer if
  perf < 0.6×, which is the correct path; do NOT short-circuit ko by claiming
  structural ceiling YOURSELF)
- `→ orchestrator: PARTIAL_PERSIST — <evidence>` — Tier-2 evidence per OL-109;
  schema_norm requires probe_report.md OR pass_b two-tier evidence to accept
- `→ orchestrator: structural_rewrite_needed — <reason>` — emit ONLY when ALL of:
  - (a) scope spans **≥2 of** {algorithm design, tile structure decision,
    primitive selection, cross-core sync discipline}
  - (b) AT LEAST ONE objective signal fires:
    - `pass_count <= current_baseline + 1` after ≥2 of your own iters, OR
    - rewrite would touch ≥2 distinct kernel files, OR
    - rewrite would touch ≥2 distinct kernel phases (preprocess / main / epilogue), OR
    - rewrite would introduce a new tiling layout (different block_M/N/K shape).
  - Examples: FA-class fused-attention cold-start PARTIAL ≤5/61 = yes-structural
    (4 axes); `foreach_sqrt` iter 5 pass_a 6/8 = NO (single-axis tuning, stay
    PARTIAL_PERSIST); `layernorm` iter 3 + Welford-vs-naive choice + per-row
    pipeline restructure = yes-structural (2 axes).
  - Routes through the standard worker. For an FA-class forward op, select the
    codified AscendC template-assembly recipe; otherwise keep the evidence-backed
    PARTIAL_PERSIST path.
- `@aog-precision-probe` — precision stuck ≥4 iters same signature
- `@aog-kernel-optimizer` — precision PASS, perf < 0.6× threshold; usually
  preferred over `→ orchestrator: done` if you have a specific tuning idea
- `@aog-fused-optimizer` — fused op + ko plateau, only if fo not yet run
- `@aog-determinism-analyzer` — det fails AND DET_POLICY=required
- `→ orchestrator: await_user_decision — <reason>` — soft-judgment call

### Forbidden handoffs

- `→ orchestrator: PARTIAL_PERF_STRUCTURAL_CEILING` — RESERVED for
  aog-kernel-optimizer / aog-fused-optimizer AFTER they have actually
  attempted optimization. As kw you have not. Emit `→ orchestrator: done`
  with HARD GATE preserved + accurate perf number; V3.8.4 will route to ko
  which will either improve or legitimately claim ceiling itself.
  (DEBT-077 P0g 2026-05-04: caught op#10 emitting this on first kw spawn.)
- `→ orchestrator: done` if perf < 0.6× — V3.8.4 schema_norm REJECTS this
  unless verification.json shows perf N/A (Path A / OL-68 case A)
- `to_state: done` / `partial_persist` / `await_orchestrator` — these are
  not YAML states; schema_norm normalizes (with evidence) or DROPS
- ANY entry in state_transitions.jsonl — orchestrator owns that file

---

## §FA-class-gates — 12 op-class-scoped gates (referenced by benchmark plugin gate_spec_for_brief, DEBT-123)

⚠ **OP-CLASS-SCOPED**: rows below fire ONLY when this workspace
classifies as FA-class (schema_norm._detect_op_class detects FUSED +
SOFTMAX signature). For non-FA ops (elementwise, reductions, etc.) the
gates short-circuit and this section is informational. Run
`python src/scripts/fa_class_self_check.py workspace/{op}/` before EXIT
HANDOFF and propagate non-zero exit into iteration, NOT into handoff.

### §FA-class-gates.A — 9 finalize hooks (extra_finalize_checks)

| # | Gate | What it enforces | Rollback verdict |
|---|---|---|---|
| 1 | `fa_class_structure_test5bis` | Antipattern A (inline Mmad) / Antipattern B forbidden; cube/vec split present; WorkspaceQueue ring buffer present | Test 5-bis FAIL → rebuild emit |
| 2 | `fa_class_acceptance_fresh_pass_a` | pass_a mtime > kernel mtime — proves verification was run AFTER kernel edits, not stale | mtime stale → re-run pass_a |
| 3 | `fa_class_regression_baseline` | `precision.pass_a.tier1_pass >= archive_baseline` — kernel iteration must not regress against archived ratio | regression detected → revert or improve |
| 4 | `fa_class_scope_out_of_scope_sentinel` | Unsupported (dtype × layout × sparse_mode) combos MUST raise explicit `_OutOfScope("<combo> not supported")`, NOT crash with TypeError | exception type/text wrong → emit explicit sentinel |
| 5 | `fa_class_source_contamination` | C34a identifier denylist + C34b compile-against-public-only + C34c copy-shape n-gram on kernel/* | CANN-internal symbols / copy-shape > 5% → rewrite |
| 6 | `fa_class_case_variant_map_present` | `workspace/case_variant_map.json` exists + non-empty mapping case_id → variant tag | missing/empty → author variant map |
| 7 | `fa_class_v2_atomicity` | `IterateAll<false>` all-or-revert behavior for sub-stage failures (no partial commits) | partial commit detected → atomicity fix |
| 8 | `fa_class_case6_structural_validity` | case_6 (atten_mask=None + sparse_mode=4) MUST be valid (not crash); case_variant_map MUST map case_6 to a non-error variant | case_6 errors / mismatched variant → structural fix |
| 9 | `tile_size_consistency` | emitted kernel tile constant (`FA_BLOCK_N`/`BLOCK_N` in kernel/*.h) MUST equal designer `tile_level` `block_N` — translator must honor the authoritative design tile, not a verbatim-ported sibling's fixed size (CAND-FA-TILESIZE-1) | mismatch → set tile = design block_N + re-derive all tile-dependent values, re-emit |

### §FA-class-gates.B — 4 brief-time gates (DEBT-121/122)

| # | Gate | What it enforces |
|---|---|---|
| 1 | EC-57/58 reminder | "Cross-core sync needs WorkspaceQueue, NOT raw CrossCoreSetFlag" — anti-Antipattern-A education before emit |
| 2 | Anti-pattern banlist | Inline Mmad in K-loop / single T.mma collapse / templated pipe (R1) / GM-struct copy-assign (R2) |
| 7 | Case-6 hard gate | Brief instructs worker: case_6 mapping MUST exist in case_variant_map.json before exit |
| 9 | Variant-dispatch context | Per-variant brief content (BSH / BNSD / BNSD_BSND / SBH layouts) — worker picks per case |

(Brief-time gates land via DEBT-122 / DEBT-121 — currently the cold-start
workflow authors `workspace/3_FusionAttention/optimization_directive.md`
manually; DEBT-122 will make the benchmark plugin auto-inject this
content into kw_brief for ALL future FA-class workspaces.)

### §FA-class-gates.C — Universal post-verify rows (all benchmark ops)

These are NOT FA-class-scoped — they apply to every benchmark archive:

- `precision.status ∈ {"PASS", "PASS_WITHIN_TOLERANCE", "PARTIAL", "FAIL"}`
- `precision.pass_a.status` matches `precision.status` consistency rule (DEBT-099)
- `performance.method` MUST declare symmetric measurement (P0ee METHODOLOGY_DECLARATION) — one of:
  - `torch_npu.profiler` + `operator_details` + `device_self_duration` (P141 schedule warmup=5/active=5)
  - `aclrtEvent` + `torch.npu.Event` device-event pair
  - `same_wrapper` / `symmetric=true` / `method_symmetric` explicit declaration
- `performance.independent_re_measure` REQUIRED — `{"ran": true, "ratio": ...}` OR `{"status": "N/A", "reason": "..."}`
- `audit_self_critic_post_worker.md` MUST exist (post-worker /aog-self-critic audit)
- `.delegation_scan_passed` marker mtime > kernel/* mtime (re-run scan_delegation_cheating.py after any kernel edit)
- `knowledge_update.md` MUST follow 5-section schema (Context / Findings / KB-promotable patterns / Cited KB items / Anti-patterns avoided)

### §FA-class-gates.evidence — Bug-class anchor

3_FusionAttention 7th cold-start (2026-05-15→23) spent 9 spawns + ~$25
rediscovering gates 1, 4, 5, 6, 8 via the rollback-and-retry path. Each
rediscovery cost 1-2 spawns. Brief-time checklist eliminates that cost.

### §FA-class-gates.how-to-use

1. If op is non-FA-class: §A + §B short-circuit (gates won't fire). §C still applies universally.
2. If op IS FA-class:
   - Read §A, §B, §C rows above
   - Run `python src/scripts/fa_class_self_check.py workspace/{op}/`
   - For each non-zero exit row: fix in workspace, re-verify, re-run script
   - Only exit when script exits 0 AND all §C rows satisfied
3. Propagate non-zero exit into ITERATION (kw spawn #2/#3/...), NOT into handoff.

## §Phase C pre-build self-audit (cold-start kernel write — MANDATORY before compile)

Relocated from `kw_brief._cold_start_phase_instructions_block()` Phase C
(2026-06-14, P0abm brief-size trim — the inlined self-audit detail pushed the
cold-start kw brief to 22949 chars, over the 21500 cap). The brief now carries
a thin pointer to this section; the full BLOCKING checklist lives here. These
are all run BEFORE compiling the kernel — the "compiles clean, crashes at
runtime" V220 error classes. Failing any BLOCKING item = do NOT build.

### ⚠ PRE-FLIGHT GATE — static analysis before compile (MANDATORY, OL-213/215, 2026-06-08, LightningIndexerGrad P128-P134 retrospective)

BEFORE compiling, run `python3 src/scripts/orchestrator/pre_build_check.py <kernel.h>`
on every kernel header file you wrote. This static analysis catches ~75% of
historical debug iterations:
- UB buffer overlaps / chain breaks (OL-215) — copy-paste errors in offset expressions
- MAX_UB_SIZE overflow (OL-215) — workspace exceeds available UB
- SyncAll in AiCore-only pipelines (OL-213) — #1 cause of multi-core hangs
- SyncAll inside batch loops (OL-213) — per-iteration cross-core barrier deadlock
- Event lifecycle violations (OL-213) — AllocEvent/FreeEvent pairing
- 32B alignment constraint violations

The tool runs in < 1 second with NO NPU/CANN needed. Exit code 0 = all clean.
If it reports ERRORS: fix them BEFORE compiling. Do NOT proceed to build with
failing pre-flight checks. If it reports WARNINGS: review each; fix if
applicable to your op's data paths.

### DataCopy alignment self-audit (MANDATORY before build, PB-22)

Every `DataCopy` call's element count MUST be 32B-aligned for the target dtype:
fp32 ≥ 8 elements, fp16/bf16 ≥ 16 elements. A non-aligned simple count is
rounded **down**, silently truncating the transfer. A count that is too large
(for example, bytes passed where elements are required) instead requests an
oversized transfer and can read or write beyond the buffer. Self-check BEFORE
build:

The third argument of the simple-count `DataCopy(dst, src, count)` overload is
the number of **elements/operands**, not bytes. Pass the aligned element count
directly; never multiply it by `sizeof(T)` or pass a byte-count variable. For
V220 `DataCopyParams`, `blockCount` is the DMA burst count and `blockLen` is the
length within each burst in 32-byte blocks, so
`DataCopyParams{element_count, 1, ...}` with zero gaps transfers
`element_count * 32` bytes, not `element_count` elements. For a
non-32-byte-aligned tail, consult EC-23/P-P98 and the target SDK before using
target-supported `DataCopyPad` with byte-based `DataCopyExtParams.blockLen`.

The mandatory `ascendc_static_check.py` rejects clear byte-count forms such as
`DataCopy(..., copyBytes)` and `DataCopy(..., count * sizeof(T))`; a failing
result is a memory-safety failure and blocks build/verification regardless of
numerical output. It also rejects the known ambiguous Params form
`blockLen=1; blockCount=curCount` when no genuine stride is configured.

```bash
grep -n 'DataCopy(' kernel.h kernel.cpp | while read line; do
  # verify count parameter is (multiple of 8 for fp32) or (multiple of 16 for fp16/bf16)
done
```

If any DataCopy has a variable count (not compile-time constant), verify the
runtime value is aligned in Init() or Process(). Use `AlignUp32(cnt)` pattern:
`((cnt + ELEMS_PER_32B - 1) / ELEMS_PER_32B) * ELEMS_PER_32B`. This is the #1
V220 crash class — see PB-22 for evidence.

### TBuf InitBuffer self-audit (MANDATORY before build, EC-62)

Every `TBuf<...>` member declared in the kernel class MUST have a corresponding
`pipe_.InitBuffer(tbuf_name, size_bytes)` call in `Init()`. This includes
workspace/scratch buffers — NOT just TQue buffers. Missing InitBuffer =
unallocated UB memory → 507035 vector core exception at runtime. Self-check
BEFORE build:

```bash
grep -E 'TBuf<.*> \w+_;' kernel.h | while read decl; do
  name=$(echo "$decl" | grep -oP '\w+(?=_;)')
  grep -q "InitBuffer($name," kernel.h || echo "MISSING InitBuffer for $name"
done
```

If the grep produces output → add the missing InitBuffer call. Do NOT proceed
to build with uninitialized TBufs.

### V220 correctness checklist (MANDATORY before build — OL-178..181, EC-47/60/61/62, PB-22)

These are the known "compiles clean, crashes at runtime" V220 error classes.
Verify ALL of the following before running the build:
1. [OL-178] Tiling struct has `#pragma pack(push,1)` / `#pragma pack(pop)`
2. [OL-179] TPipe is a CLASS MEMBER, not a function-local variable
3. [OL-180] pybind11 includes CANN env init or docs say `source set_env.sh`
4. [OL-181] Output buffer padded for DataCopy overflow (+16 elements)
5. [EC-60] pybind11 nblk ≥ 1 (never blockDim=0)
6. [EC-61] Accumulator uses VECCALC TBuf (not S-pipe float[]) when >8 elements
7. [PB-22] Every DataCopy count is 32B-aligned (self-audit above)
8. [EC-62] Every TBuf has InitBuffer (self-audit above)
9. [EC-63] pybind11 uses const char* not std::string (ABI crash on V220 ARM64)
10. [RW-01] No host-side compute — all math in kernel, pybind metadata+alloc only
11. [RW-02] nblk workload-appropriate — single-core with >256 elements needs justification

Missing 1-9 → fix BEFORE build (BLOCKING). 10-11 are WARNING — verify honestly.

**Automated check**: run `python3 src/scripts/checks/v220_prebuild_check.py`
before build. It verifies the above automatically and exits 1 (blocking) or 0
(pass with optional warnings).

### Classification gap fallback

If `workspace/<op>/op_classification.json` is missing or has empty tags, the KB
manifest block may load a wrong subset. In that case, you MUST manually read
ALL of these V220-critical files:
- `target/ascendc/PLATFORM_BUGS.md` (all PB- entries)
- `target/ascendc/ERROR_CORRECTIONS.md` (all EC- entries)
- `target/ascendc/OPERATIONAL_KNOWLEDGE.md` (search for "V220" + your op class)

Do NOT rely on the auto-loaded KB subset alone.
