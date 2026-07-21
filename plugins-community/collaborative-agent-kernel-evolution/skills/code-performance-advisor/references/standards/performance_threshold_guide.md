---
title: Performance Threshold Configuration Guide
description: How to define quantitative "success criteria" for optimization suggestions
version: 1.0
date: 2026-02-24
---

# Performance Threshold Configuration Guide

## Purpose

This guide defines how to configure **performance improvement thresholds** for the code-performance-advisor skill. These thresholds determine when an optimization is considered "successful" and trigger early stopping in the progressive disclosure workflow (Phase 1→2→3).

## Why This Matters

**Problem**: Without explicit thresholds, the system cannot decide:
- When to stop generating more suggestions (early exit)
- Whether an optimization "succeeded" or "failed"
- When to capture validated patterns into the rule library

**Solution**: Quantitative, operator-specific targets defined in `goal.md`.

---

## Configuration File Location

For each operator being optimized, create or update:

```
workspace/InputMessages/raw/{operator_name}/roofline/goal.md
```

**Example Path**:
```
workspace/InputMessages/raw/fastgelu/roofline/goal.md
```

---

## Configuration Format

### Template

```yaml
---
operator: {operator_name}
target_hardware: {Ascend910B / Ascend310P / etc.}
baseline_date: {YYYY-MM-DD}
optimization_goal: {brief description}
---

## Performance Targets

### Primary Metric: {metric_name}

**Baseline**: {current_value} {unit}
**Target**: {target_value} {unit}
**Improvement Required**: {percentage}% or {absolute_delta}

**Success Criteria**: {metric_name} {operator} {threshold}

### Secondary Metrics (Optional)

| Metric | Baseline | Target | Min Improvement | Priority |
|--------|----------|--------|-----------------|----------|
| {metric_2} | {value} | {value} | {percentage}% | Medium |
| {metric_3} | {value} | {value} | {percentage}% | Low |

---

## Threshold Logic

**Early Stop Condition** (ANY of the following):
- [ ] Primary metric meets target (absolute threshold)
- [ ] Primary metric improves by >= {min_percentage}% (relative threshold)
- [ ] All secondary metrics meet targets (composite success)

**Validation Requirements** (ALL must pass):
- [ ] Output accuracy verified (no correctness regression)
- [ ] No metric shows >= {regression_threshold}% degradation
- [ ] Improvement is reproducible (profiling run 2+ times)

---

## Example Configurations

### Example 1: Absolute Target (Task Duration)

```yaml
operator: fastgelu_custom
target_hardware: Ascend910B
baseline_date: 2026-02-10

## Performance Targets

### Primary Metric: Task Duration

**Baseline**: 6.5 us
**Target**: < 4.0 us
**Improvement Required**: >= 38%

**Success Criteria**: `Task Duration(us) < 4.0`

### Secondary Metrics

| Metric | Baseline | Target | Min Improvement | Priority |
|--------|----------|--------|-----------------|----------|
| aiv_vec_ratio | 0.16 | > 0.40 | absolute | Medium |
| aiv_scalar_ratio | 0.59 | < 0.25 | absolute | Low |
```

### Example 2: Relative Improvement (Memory-Bound Operator)

```yaml
operator: large_matmul
target_hardware: Ascend910B
baseline_date: 2026-02-15

## Performance Targets

### Primary Metric: Task Duration

**Baseline**: 145.3 us
**Target**: Not specified (open-ended optimization)
**Improvement Required**: >= 20%

**Success Criteria**: `Task Duration improves by >= 20%`

### Secondary Metrics

| Metric | Baseline | Target | Min Improvement | Priority |
|--------|----------|--------|-----------------|----------|
| aic_mte2_ratio | 0.82 | < 0.60 | 25% reduction | High |
| cube_utilization(%) | 23 | > 50 | absolute | Medium |
```

### Example 3: Composite Targets (Multi-Dimensional Optimization)

```yaml
operator: fused_attention
target_hardware: Ascend910B
baseline_date: 2026-02-20

## Performance Targets

### Primary Metric: Task Duration

**Baseline**: 89.2 us
**Target**: < 60 us
**Improvement Required**: >= 33%

**Success Criteria**:
- Task Duration < 60 us, OR
- (Task Duration improves >= 25%) AND (aic_mac_ratio > 0.70)

### Secondary Metrics

| Metric | Baseline | Target | Min Improvement | Priority |
|--------|----------|--------|-----------------|----------|
| aic_mac_ratio | 0.52 | > 0.70 | absolute | High |
| aic_scalar_ratio | 0.38 | < 0.15 | absolute | Medium |
| L1_hit_rate | 0.76 | > 0.90 | absolute | Low |
```

---

## Threshold Types

### 1. Absolute Threshold (Specific Value)

**Format**: `{metric} {operator} {value}`

**Examples**:
- `Task Duration(us) < 5.0`
- `cube_utilization(%) > 80`
- `aic_mte2_ratio < 0.50`

**When to Use**:
- You know the theoretical optimum (e.g., from hardware specs)
- You have a business requirement (e.g., "must fit in 5us budget")
- Expert rules specify concrete targets

**Advantages**: Clear, unambiguous success criteria
**Disadvantages**: May be unrealistic if baseline is far from target

---

### 2. Relative Improvement (Percentage Gain)

**Format**: `{metric} improves by >= {percentage}%`

**Examples**:
- `Task Duration improves by >= 20%`
- `aiv_vec_ratio increases by >= 50%` (relative to baseline)
- `aic_scalar_ratio decreases by >= 30%` (relative to baseline)

**When to Use**:
- Baseline is unknown or variable (different input shapes)
- Goal is incremental improvement (not absolute perfection)
- Early-stage optimization (accept "good enough" gains)

**Advantages**: Flexible, adapts to different baselines
**Disadvantages**: May stop too early if baseline is already poor

**Calculation**:
```python
# For "lower is better" metrics (e.g., Task Duration):
improvement_pct = (baseline - optimized) / baseline * 100

# For "higher is better" metrics (e.g., utilization):
improvement_pct = (optimized - baseline) / baseline * 100

# Success if improvement_pct >= threshold
```

---

### 3. Composite Logic (AND / OR Conditions)

**Format**: Boolean expressions combining multiple thresholds

**Examples**:
- `(Task Duration < 5.0) OR (Task Duration improves >= 30%)`
- `(aiv_vec_ratio > 0.5) AND (aiv_scalar_ratio < 0.2)`
- `(cube_utilization > 70) OR (aic_mac_ratio > 0.8 AND Task Duration < 10)`

**When to Use**:
- Multi-dimensional optimization (e.g., speed + efficiency)
- Trade-offs exist (e.g., faster but more memory)
- Fallback criteria (primary OR secondary)

**Advantages**: Captures complex optimization goals
**Disadvantages**: Can be hard to interpret

---

## Default Thresholds (If goal.md Missing)

When `goal.md` does not exist or is incomplete, use these defaults:

| Metric | Default Threshold | Rationale |
|--------|-------------------|-----------|
| Task Duration | >= 20% improvement | Standard "meaningful gain" heuristic |
| *_utilization(%) | >= 15% absolute increase | Indicates better hardware usage |
| *_ratio (lower is better) | >= 20% reduction | Meaningful efficiency gain |
| *_ratio (higher is better) | >= 25% increase | Noticeable improvement |

**Note**: Default thresholds are **conservative**. For production-critical operators, always define explicit targets.

---

## How code-performance-advisor Uses Thresholds

### Phase 0: Threshold Loading

```python
def load_thresholds(op_name):
    goal_path = f"workspace/InputMessages/raw/{op_name}/roofline/goal.md"

    if Path(goal_path).exists():
        # Parse goal.md (extract primary metric, target, min_improvement)
        config = parse_goal_file(goal_path)
    else:
        # Use defaults
        config = {
            "primary_metric": "Task Duration(us)",
            "threshold_type": "relative",
            "min_improvement_pct": 20
        }

    return config
```

### During Verification Loop (Phase 1/2/3)

```python
def check_success(baseline_csv, optimized_csv, thresholds):
    baseline_val = extract_metric(baseline_csv, thresholds["primary_metric"])
    optimized_val = extract_metric(optimized_csv, thresholds["primary_metric"])

    if thresholds["threshold_type"] == "absolute":
        # Check: optimized_val meets target
        success = eval(f"{optimized_val} {thresholds['operator']} {thresholds['target']}")

    elif thresholds["threshold_type"] == "relative":
        # Check: percentage improvement
        improvement_pct = (baseline_val - optimized_val) / baseline_val * 100
        success = improvement_pct >= thresholds["min_improvement_pct"]

    elif thresholds["threshold_type"] == "composite":
        # Evaluate boolean expression
        success = eval(thresholds["expression"])

    return success
```

### Early Stopping

```
Phase 1: Suggestion #1 → Verify → Success? YES → STOP (skip Phase 2/3)
                              ↓ NO
Phase 1: Suggestion #2 → Verify → Success? YES → STOP
                              ↓ NO
Phase 1: Suggestion #3 → Verify → Success? NO → Escalate to Phase 2
```

---

## Best Practices

### DO:
✅ Define realistic targets based on hardware limits (roofline model)
✅ Prioritize metrics (primary vs secondary)
✅ Use relative thresholds for variable workloads
✅ Document baseline conditions (date, shape, hardware config)
✅ Update thresholds after major code changes

### DON'T:
❌ Set unachievable targets (e.g., "100% utilization")
❌ Use only absolute thresholds for dynamic workloads
❌ Forget to validate accuracy alongside performance
❌ Change thresholds mid-optimization (creates confusion)
❌ Ignore secondary metrics (may reveal trade-offs)

---

## Validation Checklist

Before running optimization workflow, verify:

- [ ] `goal.md` exists in correct directory
- [ ] Primary metric is present in profiling CSV (`op_summary_*.csv`)
- [ ] Threshold type (absolute/relative/composite) is clearly specified
- [ ] Baseline values are from latest profiling run
- [ ] Success criteria are unambiguous (no "maybe" outcomes)
- [ ] Regression thresholds defined (what constitutes "failure")

---

## Example Workflow

1. **User prepares operator for optimization**:
   ```bash
   mkdir -p workspace/InputMessages/raw/myop/roofline
   vim workspace/InputMessages/raw/myop/roofline/goal.md
   # Define targets
   ```

2. **Run baseline profiling**:
   ```bash
   # Generate op_summary.csv
   # Record baseline: Task Duration = 12.3 us
   ```

3. **Update goal.md with baseline**:
   ```yaml
   operator: myop
   baseline_date: 2026-02-24

   ## Performance Targets
   ### Primary Metric: Task Duration
   **Baseline**: 12.3 us
   **Target**: < 8.0 us
   **Improvement Required**: >= 35%
   ```

4. **Invoke code-performance-advisor**:
   - System loads thresholds from `goal.md`
   - Generates suggestions
   - After each verification, checks if `Task Duration < 8.0` or improvement >= 35%
   - Stops early if threshold met

5. **Post-optimization**:
   - If successful, mark in `goal.md`:
     ```yaml
     status: OPTIMIZED
     achieved_value: 7.2 us
     improvement: 41%
     date_achieved: 2026-02-24
     ```

---

## Advanced: Dynamic Thresholds

For operators with variable shapes, use **shape-parameterized thresholds**:

```yaml
operator: dynamic_matmul

## Performance Targets (Shape-Dependent)

### Small Shapes (M*N*K < 10^6)
- Target: < 5 us
- Min Improvement: 30%

### Medium Shapes (10^6 <= M*N*K < 10^9)
- Target: < 50 us
- Min Improvement: 25%

### Large Shapes (M*N*K >= 10^9)
- Target: >= 20% improvement (no absolute target)
```

**Implementation**: Code-performance-advisor detects shape from `op_description.md` and selects appropriate threshold tier.

---

## Reference

- **Related Documents**:
  - [SKILL.md](../../SKILL.md) - Main workflow (uses thresholds in Phase 1/2/3)
  - [deep_research.md](../../subskills/deep_research.md) - Phase 1 analysis
  - [suggest.md](../../subskills/suggest.md) - Phase 3 knowledge capture (triggered on threshold success)

- **External Standards**:
  - [op_summary_header_guide.md](op_summary_header_guide.md) - Profiling metric definitions
  - [Roofline Model](../externel_refs/) - Theoretical performance limits

---

**Last Updated**: 2026-02-24
**Version**: 1.0
**Status**: Active
