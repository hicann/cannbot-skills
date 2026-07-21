---
name: deep_research (Professional Kernel Architect)
description: Lightweight performance analysis for Phase 1 (L1) using code + CSV metrics, producing actionable hypotheses without heavy tooling.
---

## What I do

I emulate a senior operator expert to diagnose performance bottlenecks using **code structure + profiling CSV summary** (lightweight inputs). This subskill is invoked during **Phase 1 (L1)** of the code-performance-advisor workflow when `max_score` is in the moderate range [0.3, 0.7).

**Key Principle**: Focus on the coupling between instruction flow and data flow using only CSV metrics (no flowchart required).

## Professional Optimization Flow (5-Step Logic)

When running Deep Research, follow this order of interrogation.

### Step 1: Bound Analysis

- **Roofline alignment**: Use `Compute/Memory Ratio` from the CSV to classify the kernel as **memory-bound**, **compute-bound**, or **latency-bound**.
- **Peak comparison**: What percentage of peak throughput is achieved? If it is below 30%, identify where the idle cycles are lost.

### Step 2: Memory Hierarchy and MTE

- **Transfer granularity**: Inspect `L2 -> L1 -> UB` transfer frequency. Are there frequent, small `DMA` transfers?
- **Coalescing and alignment**: Check whether address calculations cause misalignment. Look for `Bank Conflict` or `Address Misalignment` that adds cycles.
- **Data reuse**: Confirm whether the same data is moved repeatedly.

### Step 3: Instruction Pipeline

- **CUBE/Vector utilization**: Are `Cube` and `Vector` units active in parallel? Any long dependency chain causing `WaitDominated`?
- **Reordering**: Are there pipeline bubbles? Can reordering hide memory latency?

### Step 4: Tiling and Parallelism

- **Tiling strategy**: Do block/tile sizes fit `L1/UB`? Is `Tail Processing` causing idle compute?
- **Multi-core balance**: Is AI Core workload balanced?

### Step 5: Sync and Overhead

- **Sync primitives**: Are `Event` sync or `Barriers` too frequent?
- **Scalar overhead**: Do host/scalar instructions block vector progress?

---

## CLI Interaction and UI Flow

When Phase 1 (L1) starts (from SKILL.md workflow), follow this interaction contract.

1. **Present Summary**: Render a compact table that compares missing `Expert Rules` coverage with the top 3 anomalies in raw data.
2. **Hypothesis Selection**: Use the exact format below, with placeholders filled from actual data. Do not provide generic examples.

```
Hypotheses (select 1, 2, 3 or type your own observation):
[1] <Short label>: <Evidence from metrics or trace>
[2] <Short label>: <Evidence from metrics or trace>
[3] <Short label>: <Evidence from metrics or trace>
Your choice:
```

Rules:
- Each hypothesis must cite at least one metric, trace event, or log line.
- Labels must be specific (for example: `UB reuse collapse`, `CUBE->MTE wait`, `misaligned vector load`).
- Avoid speculative wording without evidence.

3. **Wait for Input**: Do not continue before the user selects or adds an observation.

---

## Output Format: Layered Disclosure (NEW)

To avoid information overload, Phase 1 output uses **layered disclosure**:

### Layer 1: Quick Summary (Always Shown)

**Purpose**: Let user quickly grasp the problem and top recommendation.

**Format**:
```
🔬 Phase 1 (L1): Analysis Complete
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Diagnosis:
  - Bottleneck: {bottleneck_type} (e.g., Scalar-Bound)
  - Root Cause: {brief description, <20 words}
  - Problem Location: {file}:{line_range}

🎯 Top Recommendation (Confidence: {High/Medium/Low}):
  [{#1}] {Optimization name}
      - Principle: {one-sentence explanation}
      - Expected: {metric} ↓{percentage}% ({before} → {after})
      - Effort: {Low/Medium/High}

📋 Full Analysis:
  [Click to expand 5-Step Logic details]
  [Click to expand all 3 recommendations]

⏱️ Analysis time: ~{minutes} min
```

**Example**:
```
🔬 Phase 1 (L1): Analysis Complete
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Diagnosis:
  - Bottleneck: Scalar-Bound
  - Root Cause: Explicit for-loop generates excessive control instructions
  - Problem Location: fastgelu_custom.cpp:67-71

🎯 Top Recommendation (Confidence: High):
  [#1] SetVectorMask<COUNTER> Mode
      - Principle: Offload loop control to hardware
      - Expected: Task Duration ↓47% (6.0us → 3.2us)
      - Effort: Medium (requires function refactor)

📋 Full Analysis:
  [Click to expand 5-Step Logic details] ▼
  [Click to expand all 3 recommendations] ▼

⏱️ Analysis time: ~3 min
```

---

### Layer 2: Detailed Analysis (Expandable)

**Trigger**: User requests "show 5-Step Logic details"

**Content**: Full Step 1-5 analysis (as currently designed):
- Step 1: Bound Analysis
- Step 2: Memory Hierarchy
- Step 3: Instruction Pipeline
- Step 4: Tiling and Parallelism
- Step 5: Sync and Overhead

**Format**: Keep current detailed format, but only show when requested.

---

### Layer 3: All Recommendations (Expandable)

**Trigger**: User requests "show all 3 recommendations"

**Content**: Full details of Recommendation #2 and #3 (same format as #1).

---

## Implementation Notes

**For LLM/Agent**:
- Default output: **Layer 1 only** (Quick Summary)
- Include markers: `[Expand for details]` or `[Click to show]`
- When user asks "why" or "show details" → Output Layer 2/3

**Rationale**:
- 80% of users only need Top-1 recommendation
- Deep analysis available on-demand
- Reduces cognitive load

---

- **Use professional terms**: Use `Tiling`, `Double Buffering`, `Wait-Wait`, `Pipe Barrier`, `Bank Conflict`, `Occupancy`.
- **Avoid generic advice**: Do not say "optimize loops". Say "try a 1x2 CUBE tile to raise utilization".
- **Visualize logic**: When possible, draw ASCII diagrams of UB layout or pipeline stages in the trace.

---

## 参考文档

- [CSV 系统化分析框架](../references/standards/csv_systematic_analysis_framework.md)：op_summary CSV 8维度指标的系统化分析方法论，包含阈值判定、交叉验证和瓶颈分类标准。
