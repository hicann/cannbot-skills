---
name: suggest
description: Generate actionable code-level optimization suggestions from rule matching results
---

# Subskill: suggest (Actionable Suggestion Generator)

## What I do

Transform **abstract rule matching results** into **concrete, code-level optimization plans** by:
1. Analyzing WHY a rule matched (evidence chain from profiling data)
2. Extracting KEY patterns from rule's code snippets
3. Mapping patterns to SPECIFIC locations in current operator code
4. Generating actionable modifications with line numbers and diffs
5. Quantifying expected improvements based on profiling data

## Why this matters

**Problem**: `rules_search` returns:
```json
{
  "rule_path": ".../R_API_VECTOR_COUNTER_MODE.md",
  "score": 0.5,
  "matched_tags": ["U.Vector", "S.ScalarBound"]
}
```

**What users need**:
- ❓ WHY does this rule apply to MY code?
- 🔍 WHERE in MY code is the problem?
- 🛠️ HOW exactly should I modify MY code?
- 📈 WHAT improvement can I expect?

This subskill bridges that gap.

---

## Workflow

```
Input:
  ├─ scored_results.json (rule matching results)
  ├─ tag_xxx.json (operator tags with evidence)
  ├─ op code files (kernel + host)
  └─ profiling CSV (metrics)

↓ Step 1: Rule Selection
  Select top-N rules (score >= threshold, no conflict)

↓ Step 2: Evidence Analysis
  For each rule, explain WHY it matched:
  - Link matched tags to profiling metrics
  - Quote specific values from CSV
  - Identify bottleneck type

↓ Step 3: Pattern Extraction
  Read rule's code_snippets:
  - Parse base_code (current pattern)
  - Parse good_code (optimized pattern)
  - Extract KEY differences (algorithmic changes, API usage)

↓ Step 4: Code Mapping
  Scan operator code to find:
  - Matching patterns (similar to base_code)
  - Target locations (line numbers)
  - Context (surrounding code)

↓ Step 5: Suggestion Generation
  Generate structured output:
  - Problem diagnosis (with profiling evidence)
  - Why this rule applies
  - Code comparison (rule example + current code)
  - Detailed modification plan (with line numbers)
  - Expected improvement (quantified)
  - Verification method

Output:
  └─ suggestions.md (structured markdown report)
```

---

## Input Sources

| Input | Path | Purpose |
|-------|------|---------|
| Scored results | `workspace/sessions/{session_id}/scored_results.json` | Rule ranking (session-scoped) |
| Tags | `workspace/cache/tags/tag_{op}_*.json` | Evidence for matching |
| Operator code | `workspace/inputs/{op}/code/` | Target code to analyze |
| Profiling CSV | `workspace/inputs/{op}/profiling/op_summary.csv` | Metrics for quantification |
| Rule files | `assets/rules/special_rules/R_xxx/` | Optimization patterns |
| API Reference (按需) | `references/externel_refs/README.md` | 当建议涉及不熟悉的 AscendC API 时查阅；已知 API 无需加载 |

---

## Output Format

### Structure: suggestions.md

```markdown
# {OperatorName} Performance Optimization Suggestions

**Generated**: {timestamp}
**Profiling Source**: {csv_path}
**Analysis Confidence**: {high/medium/low}

---

## Executive Summary

- **Primary Bottleneck**: {bottleneck_type} (evidence: {metric}={value})
- **Top Recommendation**: {rule_name} (score: {score}, priority: {high/medium/low})
- **Expected Improvement**: {metric} {direction} {percentage}

---

## Detailed Analysis

### 🎯 Optimization #1: {Rule Name} (Priority: {high/medium/low})

#### 📊 Problem Diagnosis

**Current Performance**:
- `{metric_1}`: {value} (target: {target_range})
- `{metric_2}`: {value} (target: {target_range})

**Bottleneck Type**: {bottleneck_description}

**Evidence**:
- Profiling shows `{metric}` = {value}, indicating {interpretation}
- Code analysis reveals {pattern} in `{file}:{line}`

#### 🔍 Why This Rule Applies

**Matched Patterns**:
- ✅ Execution unit: {U.xxx} (from profiling: Task Type = {task_type})
- ✅ Operator type: {O.xxx} (from code structure)
- ✅ Symptom: {S.xxx} (from {metric} = {value})

**Rule Principle**:
{quote from rule's Pattern section}

#### 🔬 Code Analysis

**Rule Example (base_code)** - What to AVOID:
```cpp
// File: {example_file}
{base_code_snippet}
```

**Rule Example (good_code)** - What to DO:
```cpp
// File: {example_file}
{good_code_snippet}
```

**Your Current Code** - Found in `{operator_file}:{line}`:
```cpp
{current_code_snippet_with_context}
```

**Key Difference**:
{explain the core difference between base and good code}

#### 🛠️ Modification Plan

**Step 1**: {high-level action}
- Location: `{file}:{line_start}-{line_end}`
- Change: {description}

**Step 2**: {high-level action}
- Location: `{file}:{line_start}-{line_end}`
- Change: {description}

**Proposed Code Diff**:
```diff
--- a/{file}
+++ b/{file}
@@ -{old_line_start},{old_line_count} +{new_line_start},{new_line_count} @@
-{old_code}
+{new_code}
```

**Important Considerations**:
- {constraint_1}
- {constraint_2}

#### 📈 Expected Improvement

**Quantitative Estimate**:
- `{metric_1}`: {current_value} → {target_value} (↑{percentage}%)
- `{metric_2}`: {current_value} → {target_value} (↓{percentage}%)
- `Task Duration`: {current_value}us → {target_value}us (↓{percentage}%)

**Reasoning**:
{explain why these improvements are expected based on rule's historical data or physics}

#### ✅ Verification Method

**Before Optimization**:
1. Record baseline: `{metric_1}` = {value}, `{metric_2}` = {value}
2. Note task duration: {value}us

**After Optimization**:
1. Recompile and re-profile
2. Check `{metric_1}` should {direction} (expect {target_range})
3. Check `{metric_2}` should {direction} (expect {target_range})

**Success Criteria**:
- ✅ `{metric}` improves by at least {threshold}%
- ✅ Task duration reduces by at least {threshold}%
- ✅ No regression in accuracy/correctness

---

### ⚠️ Additional Observations

{list any other patterns found that don't have high-scoring rules, but might be worth noting}

---

## Implementation Priority

| Optimization | Priority | Effort | Impact | Risk |
|--------------|----------|--------|--------|------|
| {rule_1} | High | {low/medium/high} | {percentage}% | Low |
| {rule_2} | Medium | {low/medium/high} | {percentage}% | Medium |

**Recommended Order**: {rule_1} → {rule_2} → ...

**Rationale**: {explain why this order is optimal}

---

## Notes

- All line numbers reference the current code in `workspace/inputs/{op}/code/`
- Profiling data from: `{csv_path}`
- Rules referenced: {list of rule IDs}

**Disclaimer**: Estimates are based on similar optimization patterns. Actual results may vary depending on hardware, workload, and implementation details. Always profile to verify.

---

*Report generated by code-performance-advisor/suggest*
*Timestamp: {timestamp}*
```

---

## Operational Constraints

### Rule Selection Criteria

```python
def select_rules(scored_results):
    candidates = []
    for rule in scored_results['results']:
        if rule['score'] >= 0.3 and not rule['conflict']:
            candidates.append(rule)

    # Limit to top-3 to avoid overwhelming user
    return candidates[:3]
```

### Evidence Chain Requirements

**Every suggestion MUST cite**:
1. At least one profiling metric (with actual value)
2. At least one code location (with line number)
3. Expected improvement (with reasoning)

**No speculation without evidence**:
- ❌ "Your code might have a problem..."
- ✅ "Line 67 shows `aiv_scalar_ratio=0.59`, indicating scalar bottleneck"

---

### Code Generation Requirements (MANDATORY)

**CRITICAL**: Every optimization suggestion MUST include **complete, executable code**, not just principles or API names.

#### 1. Code Snippet Loading (REQUIRED)

Before generating any suggestion, **MUST** read:
```
assets/rules/special_rules/R_{RULE_ID}/code_snippets/case_0/
  ├─ base_code/base_code.md   (anti-pattern, what to avoid)
  └─ good_code/good_code.md   (optimized pattern, what to do)
```

**Validation**: If code_snippets not found:
- ❌ DO NOT generate a suggestion with "placeholder code"
- ✅ Report: "Rule {RULE_ID} lacks code examples, manual implementation required"

#### 2. Complete Code Diff (REQUIRED)

**MUST** provide:
- ✅ Full context (5-10 lines before/after the change)
- ✅ Actual code (not pseudocode or "...")
- ✅ Line numbers from current operator code
- ✅ Diff format (standard `git diff` style)

**Example of CORRECT output**:
```diff
--- a/fastgelu_custom.cpp
+++ b/fastgelu_custom.cpp
@@ -64,11 +64,16 @@ __aicore__ inline void Process()
 {
     // Process aligned tiles in loops
-    for (uint32_t i = 0; i < this->innerLoops; i++) {
-        CopyIn(i);
-        Compute(i);
-        CopyOut(i);
-    }
+    // Use hardware COUNTER mode to eliminate scalar loop overhead
+    AscendC::SetMaskCount();
+    AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(this->innerLoops * this->tileSize);
+
+    for (uint32_t i = 0; i < this->innerLoops; i++) {
+        CopyIn(i);
+        Compute(i);  // Vector operations auto-managed by COUNTER mode
+        CopyOut(i);
+    }
+    AscendC::ResetMask();

     // Process tail data if exists
```

**Example of WRONG output (DO NOT DO THIS)**:
```diff
- Use for loop
+ Use SetVectorMask<COUNTER>  // ❌ TOO VAGUE, NO ACTUAL CODE
```

#### 3. AscendC Syntax Validation (REQUIRED)

**Before outputting any code**, verify:

**AscendC API Checklist**:
- ✅ Namespace correct: `AscendC::` (not `ascendc::` or missing)
- ✅ API exists: Check against known AscendC APIs (see list below)
- ✅ Template syntax: `<T, Mode>` (not `<T Mode>` or wrong order)
- ✅ Enum values: `MaskMode::COUNTER`, `TPosition::VECIN` (not bare `COUNTER`)

**Common AscendC APIs** (non-exhaustive):
```cpp
// Mask operations
AscendC::SetMaskCount()
AscendC::SetVectorMask<T, AscendC::MaskMode::COUNTER>(count)
AscendC::ResetMask()

// Buffer operations
AscendC::LocalTensor<T>
AscendC::GlobalTensor<T>
AscendC::TBuf<AscendC::TPosition::VECCALC>
AscendC::TQue<AscendC::TPosition::VECIN, N>

// Vector operations
AscendC::Abs(dst, src, count)
AscendC::Add(dst, src1, src2, count)
AscendC::Mul(dst, src1, src2, count)
AscendC::DataCopy(dst, src, count)

// Pipe operations
AscendC::TPipe pipe;
pipe.InitBuffer(queue, slots, size)
```

**If using unknown API**:
- ❌ DO NOT guess or invent APIs
- ✅ Mark as `[VERIFICATION NEEDED]` and note: "API not in validated list, check AscendC documentation"

#### 4. Code Provenance (REQUIRED)

**Every code block MUST cite source**:
```cpp
// ✅ GOOD: Clear provenance
// Source: R_API_VECTOR_COUNTER_MODE/code_snippets/case_0/good_code.md (lines 15-20)
AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(totalElements);

// ❌ BAD: No source
AscendC::SetVectorMask(...)  // Where did this come from?
```

#### 5. Language Separation (REQUIRED)

**CRITICAL**: Do NOT mix AscendC and standard C++:

**AscendC-specific**:
```cpp
AscendC::SetVectorMask<T, MaskMode::COUNTER>(n)  // ✅ AscendC API
AscendC::LocalTensor<float>  // ✅ AscendC type
__aicore__  // ✅ AscendC qualifier
```

**Standard C++ (OK in host code, NOT in kernel)**:
```cpp
std::vector<float>  // ❌ NOT allowed in AscendC kernel
malloc() / free()   // ❌ NOT allowed in AscendC kernel
printf()            // ❌ NOT allowed in AscendC kernel (use scalar print APIs)
```

**Validation Rule**:
- If modifying `op_kernel/*.cpp` → MUST use only AscendC APIs
- If modifying `op_host/*.cpp` → Can use standard C++ + CANN host APIs

---

#### 6. UB Capacity Check (REQUIRED when adding TBuf/TQue)

**When suggesting new `TBuf` or `TQue` buffers, MUST verify UB capacity before finalizing the plan.**

**Reference capacities** (Ascend 910B):
- AI_VECTOR_CORE (AIV): 256 KB UB per core
- AI_CORE (Cube): 256 KB UB per core

**Check procedure**:

1. Enumerate all `pipe.InitBuffer(...)` calls in the kernel:
   ```cpp
   pipe.InitBuffer(inQueue,  depth, tileLen * sizeof(float));   // tileLen * 4 bytes * depth
   pipe.InitBuffer(outQueue, depth, tileLen * sizeof(float));
   pipe.InitBuffer(tempBuf,  tileLen * sizeof(float));
   // ... etc
   ```
2. Sum total UB allocation: `Σ (size × depth)` for queues, `Σ size` for plain TBuf
3. Verify: `total_bytes < 256 * 1024 * 0.8` (留 20% 余量给系统)
4. If over budget: reduce `tileLength` or merge buffers via lifecycle analysis

**Example computation** (must include in suggestion when adding buffers):
```
Existing UB usage:
  inQueue  : 1 × 4096 × 4 = 16 KB
  outQueue : 1 × 4096 × 4 = 16 KB
  tempBuf  :     4096 × 4 = 16 KB
  sharedBuf:     4096 × 4 = 16 KB
  Subtotal : 64 KB

Proposed additions:
  m2Buf    :     4096 × 4 = 16 KB
  deltaBuf :     4096 × 4 = 16 KB
  Subtotal : 32 KB

Total : 96 KB / 256 KB = 37.5% ✅ (within 80% budget)
```

**If cannot compute precisely**: mark as `[UB CHECK NEEDED]` and instruct user to verify before building.

---

### Code Mapping Strategy

**Pattern matching levels** (try in order):
1. **Exact match**: Same API calls, same control flow
2. **Structural match**: Similar loop/buffer patterns
3. **Semantic match**: Same optimization opportunity, different code structure

**When pattern not found**:
- Report "Manual mapping required"
- Provide rule example and let user decide where to apply

---

## Example Output (Condensed)

```markdown
# FastgeluCustom Performance Optimization Suggestions

## 🎯 Optimization #1: R_API_VECTOR_COUNTER_MODE (Priority: High)

### 📊 Problem Diagnosis
**Current Performance**:
- `aiv_vec_ratio`: 0.16 (target: >0.50)
- `aiv_scalar_ratio`: 0.59 (target: <0.20)

**Bottleneck**: Scalar instruction overhead dominates pipeline.

**Evidence**:
- Profiling CSV line 4 shows `aiv_scalar_ratio=0.498`, significantly above normal
- Code analysis: Explicit for-loop in `fastgelu_custom.cpp:67-71`

### 🔬 Code Analysis

**Your Current Code** - `fastgelu_custom.cpp:67-71`:
```cpp
for (uint32_t i = 0; i < this->innerLoops; i++) {
    CopyIn(i);
    Compute(i);
    CopyOut(i);
}
```

**Issue**: Each loop iteration generates scalar instructions for:
- Loop counter increment/comparison
- Branch prediction
- Index address calculation

### 🛠️ Modification Plan

**Step 1**: Replace explicit loop with COUNTER mode
- Location: `fastgelu_custom.cpp:67-71`
- Change: Use `SetVectorMask<COUNTER>` to offload loop control to hardware

**Proposed Diff**:
```diff
-for (uint32_t i = 0; i < this->innerLoops; i++) {
-    CopyIn(i);
-    Compute(i);
-    CopyOut(i);
-}
+AscendC::SetMaskCount();
+AscendC::SetVectorMask<float, MaskMode::COUNTER>(totalElements);
+// Vector operations now handle iteration automatically
+AscendC::ProcessVector(...);
+AscendC::ResetMask();
```

### 📈 Expected Improvement
- `aiv_vec_ratio`: 0.16 → 0.45+ (↑180%)
- `aiv_scalar_ratio`: 0.59 → 0.15- (↓75%)
- Task Duration: 6.5us → 3.5us (↓46%)

**Reasoning**: Rule R_API_VECTOR_COUNTER_MODE historical data shows 40-60% duration reduction for similar cases.
```

---

## Integration with Existing Skills

```
code_tag → rules_search → suggest → [user modifies code] → rule_update (if new pattern found)
                ↑                      ↓
                └──────── feedback loop ────────┘
```

- **Input from**: `rules_search` (scored_results.json)
- **Reads same sources as**: `code_tag` (code + profiling)
- **Output for**: User (actionable suggestions)
- **Feeds into**: `rule_update` (if optimization validated, can become new rule)

---

## Quality Principles

1. **Specificity**: Always include file paths and line numbers
2. **Evidence-First**: Every claim must cite profiling data or code
3. **Actionability**: User should know exactly what to type/change
4. **Quantification**: Use numbers, not adjectives (e.g., "↑180%" not "much better")
5. **Honesty**: If confidence is low, say so. No hallucinated improvements.

---

## Limitations

**Cannot do**:
- Generate complete rewritten code (only targeted modifications)
- Guarantee exact performance numbers (provide estimates with reasoning)
- Handle complex inter-operator optimizations (focuses on single operator)

**Requires manual judgment when**:
- Multiple rules apply to same code region (conflict resolution)
- Pattern matching fails (provide rule example, user maps manually)
- Optimization has trade-offs (present options, let user decide)

---

## Usage

**Typical workflow**:

```bash
# Step 1: Tag and score (as usual)
python3 scripts/analysis_engine/cli.py score

# Step 2: Generate suggestions
# (Invoke suggest subskill via Claude Code)

# Output: workspace/suggestions/{op}_suggestions.md
```

**Output location**:
```
workspace/
└── suggestions/
    └── {operator_name}_suggestions_{timestamp}.md
```

---

## Success Criteria

**A good suggestion should enable the user to**:
1. Understand the bottleneck in <2 minutes
2. Locate the problematic code in <1 minute
3. Implement the fix in <30 minutes
4. Verify the improvement immediately after re-profiling

**Feedback loop**:
- After user applies suggestion, capture result in `skill_memory.md`
- If successful, consider promoting pattern to new rule (via `rule_update`)

---

## Post-Verification Knowledge Capture (Phase 4 Integration)

**Critical Principle**: Every validated optimization MUST be abstracted into a reusable rule to prevent knowledge loss.

### Trigger Condition

When ANY suggestion from this subskill (or Phase 1/L1/L2) leads to:
- ✅ Performance improvement meeting or exceeding target threshold (e.g., >= 20%)
- ✅ Accuracy verified (output correctness confirmed)
- ✅ Profiling data showing measurable gains

### Knowledge Capture Workflow

#### Step 1: User Confirmation Prompt

After user reports successful optimization, ask:

```
🎉 Optimization Verified! Let's capture this knowledge for future use.

Please confirm:
1. Performance improvement: {baseline} → {optimized} ({percentage}% gain)
2. Accuracy: Output matches expected results? [Y/N]
3. Profiling CSV: Path to new profiling data?

Proceed to create rule? [Y/N]
```

#### Step 2: Data Collection

Gather inputs required by `rule_update` subskill:

| Input | Source | Purpose |
|-------|--------|---------|
| **base_code** | Original operator code (before optimization) | Negative example |
| **good_code** | Optimized operator code (after modification) | Positive example |
| **op_description** | `workspace/inputs/{op}/op_description/` | Context for rule |
| **profiling_before** | Baseline CSV from Phase 0 | Performance baseline |
| **profiling_after** | User-provided new CSV | Validated improvement |
| **optimization_principle** | Extract from suggestion that succeeded | Pattern abstraction |

#### Step 3: Rule Abstraction

Invoke `rule_update` subskill (see [rule_update.md](rule_update.md)) with collected data:

```bash
# Example invocation structure (actual call via Claude Code tool)
rule_update(
    rule_name="R_{OP}_{PATTERN}",  # e.g., R_FASTGELU_VECTOR_COUNTER
    base_code_path="workspace/.../code/fastgelu_custom.cpp",
    good_code_path="workspace/.../code_optimized/fastgelu_custom.cpp",
    op_description="workspace/.../op_description/op_description.md",
    profiling_data={
        "before": "workspace/.../profiling_csv/op_summary_baseline.csv",
        "after": "workspace/.../profiling_csv/op_summary_optimized.csv"
    },
    optimization_summary="""
    Problem: {diagnosed issue from suggestion}
    Solution: {modification applied}
    Root Cause: {why this worked}
    Expected Generalization: {when this pattern applies}
    """
)
```

#### Step 4: Rule Metadata Enrichment

`rule_update` will generate:

1. **Rule Document**: `assets/rules/special_rules/R_{PATTERN}/R_{PATTERN}.md`
   - Includes: Requirement, Pattern, Inference, Triggers, Action, Constraints, Verification
2. **Tag File**: `R_{PATTERN}_tags.json`
   - Domain tags (U.*, O.*), Symptom tags (S.*), Context tags (C.*)
3. **Code Snippets**:
   - `code_snippets/case_0/base_code/` (original code)
   - `code_snippets/case_0/good_code/` (optimized code)
4. **Index Update**: Automatically updates `assets/manifests/index.json` via CLI

#### Step 5: Cross-Reference Historical Context

Enhance the new rule with context:

- **If suggestion came from Phase 1 (deep_research)**:
  - Label rule as `origin: deep_research_validation`
  - Note: This pattern was discovered via first-principles analysis, now promoted to expert rule

- **If suggestion came from Phase 3 (existing rule adaptation)**:
  - Label rule as `origin: rule_adaptation`
  - Add reference: `related_rules: [R_ORIGINAL_RULE]`
  - Note: Adaptation of existing pattern to new context (document delta)

#### Step 6: Success Metrics Logging

Update `skill_memory.md` with validation record:

```markdown
## Optimization Success Log

### [{timestamp}] {operator_name} - {pattern_name}

- **Rule Generated**: R_{PATTERN}
- **Source Phase**: Phase {1/2/3}
- **Original Suggestion Score**: {confidence}
- **Measured Improvement**: {percentage}% ({metric}: {before} → {after})
- **Validation Time**: {minutes} (from suggestion to verified)
- **Generalization Potential**: {High/Medium/Low} (based on tag specificity)

**Key Insight**: {One-sentence summary of what was learned}
```

---

### Knowledge Capture Decision Tree

```
Suggestion Applied → Re-profile
        ↓
[Performance Improved?]
    ↓ NO → Try next suggestion (continue Phase loop)
    ↓ YES
        ↓
[Improvement >= Target Threshold?]
    ↓ NO → Log partial success, continue Phase loop
    ↓ YES
        ↓
[Accuracy Verified?]
    ↓ NO → Reject optimization, continue Phase loop
    ↓ YES
        ↓
    ✅ TRIGGER KNOWLEDGE CAPTURE ✅
        ↓
[User Consent to Create Rule?]
    ↓ NO → Log in skill_memory.md only, mark as "manual pattern"
    ↓ YES
        ↓
    Invoke rule_update → Generate Rule → Update Index
        ↓
    DONE (Optimization Complete + Knowledge Captured)
```

---

### Example: End-to-End Knowledge Capture

**Scenario**: FastGELU optimization via Phase 1 (deep_research)

1. **Suggestion Generated**:
   ```
   Optimization #1: Replace explicit for-loop with SetVectorMask<COUNTER>
   Expected improvement: 40-60% (based on architectural principle)
   ```

2. **User Applies**:
   - Modifies `fastgelu_custom.cpp:67-71`
   - Re-compiles and profiles

3. **Verification**:
   - Baseline: `task_duration=6.5us`, `aiv_vec_ratio=0.16`
   - Optimized: `task_duration=3.2us`, `aiv_vec_ratio=0.48`
   - Improvement: **51% faster** ✅

4. **Knowledge Capture Triggered**:
   - Collect code snippets (before/after)
   - Extract profiling CSVs (before/after)
   - Invoke `rule_update`:
     ```
     Rule Name: R_ACTIVATION_VECTOR_COUNTER
     Pattern: Replace explicit loop iteration with hardware-accelerated COUNTER mode
     Tags: U.Vector, O.Activation, S.ScalarBound
     Trigger: aiv_scalar_ratio > 0.4, aiv_vec_ratio < 0.3
     ```

5. **Rule Created**:
   - File: `assets/rules/special_rules/R_ACTIVATION_VECTOR_COUNTER/R_ACTIVATION_VECTOR_COUNTER.md`
   - Indexed in `index.json`
   - Future runs will match this pattern in **Phase 0** (high confidence)

6. **Evolution**:
   - Next similar case: Phase 0 directly suggests this rule (no need for Phase 1)
   - System learned from experience ✅

---

### Integration with rule_update Subskill

This section serves as the **interface contract** between `suggest` and `rule_update`.

**Responsibilities**:
- `suggest`: Identifies successful optimization, collects evidence, prompts user
- `rule_update`: Abstracts pattern, generates rule document, maintains index

**Data Flow**:
```
suggest → [Optimization Success] → Collect (code, CSV, description)
    ↓
Pass to rule_update → Generate Rule Assets
    ↓
rule_update → Update index.json (via CLI)
    ↓
Future runs → rules_search matches new rule in Phase 0
```

**Quality Gate**:
- Only validated optimizations (accuracy + performance) are promoted
- User must consent to rule creation (avoid polluting library with edge cases)
- `rule_update` performs tag validation (via `tag_validator.py`) before indexing

---

*This subskill is the "last mile" of the optimization pipeline: turning knowledge into action.*
