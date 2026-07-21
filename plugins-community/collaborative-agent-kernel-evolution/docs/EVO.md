# Evolutionary Kernel Generation Documentation

This document explains the evolutionary kernel generation system for CAKE2, accessible through the `cake_evo` Claude Code agent.

## Table of Contents

1. [Quick Start](#quick-start)
2. [How to Use cake_evo](#how-to-use-cake_evo)
3. [How Inspirations and Diversity Work](#how-inspirations-and-diversity-work)
4. [Subagent Execution](#subagent-execution)
5. [Result Evaluation](#result-evaluation)
6. [Troubleshooting](#troubleshooting)

---

# Quick Start

## Prerequisites

- **CANN 8.3+** and NPU hardware required
- Local CANN environment for compilation

## Using cake_evo Agent

```bash
# 1. Create working directory
mkdir genop
cp -r .claude genop/
cd genop

# 2. Start Claude Code
opencode

# 3. Select cake_evo agent (press Tab)
# 4. Describe your operator
```

Example interaction:
```
User: Generate a FastGELU activation function operator with evolution.
Formula: y = x / (1 + exp(-1.702 * |x|)) * exp(0.851 * (x - |x|))

Agent: I'll set up evolutionary kernel generation for FastGELU.

Configuration (press Enter for defaults):
- Max rounds [2]:
- Parallel candidates [3]:
- Target speedup [1.5]:

Starting evolution...
```

## Output Structure

```
output/FastGELU_evo_20260213_143022/
├── shared/                   # Generated once (steps 1-4)
│   ├── FastGELU_op_desc.json
│   ├── FastGELU_reference.py
│   ├── FastGELU_functional.py
│   └── FastGELUCustom/       # CMake project template
├── round_1/
│   ├── parallel_0/           # Variant 0 (steps 5-9)
│   │   ├── FastGELU_op_desc.json     # (copied from shared)
│   │   ├── FastGELU_reference.py     # (copied from shared)
│   │   ├── FastGELU_dsl.py           # (unique per variant)
│   │   ├── FastGELUCustom/           # (kernel code modified)
│   │   └── evaluation_results.json
│   ├── parallel_1/           # Variant 1
│   └── parallel_2/           # Variant 2
├── round_2/                  # Next round with inspirations
```

---

# How to Use cake_evo

## Agent Workflow

The `cake_evo` agent orchestrates evolutionary optimization through these steps:

### 1. Configuration

Agent prompts for:
- **Operator name**: Short identifier (e.g., "FastGELU")
- **Operator description**: Natural language or formal spec
- **Max rounds**: Number of evolution cycles (default: 2)
- **Parallel candidates**: Variants per round (default: 3)
- **Target speedup**: Goal speedup vs PyTorch (default: 1.5x)

### 2. Environment Detection

Agent automatically detects local CANN environment:
- Requires `npu-smi` and CANN 8.3+ installation
- Uses local NPU compilation

### 3. Shared Pre-Generation (run once)

Agent runs steps 1-4 **once** and saves to a `shared/` directory:
1. Generates operator description JSON
2. Creates PyTorch reference
3. Converts to functional API
4. Generates Ascend call code and CMake project

These outputs are identical across all variants and rounds, so they are generated once and copied to each parallel directory.

### 4. Round 1: Parallel Kernel Generation (from step 5)

Agent copies shared files to each `round_1/parallel_{p}/` directory, then spawns N parallel subagents (default: 3), each starting from DSL generation:
1. Creates DSL baseline
2. Applies DSL lowering passes
3. Reviews and fixes AscendC code
4. Compiles and evaluates locally

**Meta-Prompt Cycling**: Each subagent receives a different meta-prompt for diversity:
- Subagent 0: `meta_prompts/general.txt`
- Subagent 1: `meta_prompts/elementwise.txt`
- Subagent 2: `meta_prompts/general.txt` (cycles)

### 5. Classification

After Round 1, agent classifies implementations into tiers:

| Tier | Percentage | Selection Strategy |
|------|------------|-------------------|
| **Good** | Top 30% | Exploitation (best performance) |
| **Medium** | Middle 40% | Balanced exploration |
| **Poor** | Bottom 30% | Diverse exploration |

### 6. Inspiration Selection

Agent samples from all tiers for next round:
- **1 from Good tier**: Best performer (exploitation)
- **1 from Medium tier**: Alternative approach (exploration)

### 7. Round 2+: Evolution

Agent copies shared files again, then spawns new subagents with inspirations:
- Same operator description
- **Plus**: Code from selected implementations
- **Plus**: Performance metrics and insights
- Different meta-prompts for diversity

### 8. Termination

Evolution stops when:
- Target speedup achieved
- Max rounds reached
- No successful implementations

---

# How Inspirations and Diversity Work

## The Core Mechanism

### Round 1: Pure Exploration

Shared artifacts (op_desc, reference, functional, CMake project) are already in each parallel directory. Each subagent receives:
```
Pre-generated files in output directory:
  {op_name}_op_desc.json, {op_name}_reference.py,
  {op_name}_functional.py, {op_name}Custom/

Meta-Prompt: general.txt (or elementwise.txt, etc.)
  - Memory access optimization
  - Parallelization strategies
  - Vectorization opportunities

Execute skills: dsl_baseline → dsl_lowering → review → eval
```

Result: 3 implementations with different approaches

### Round 2: Informed Exploration

All subagents receive inspirations **plus** different meta-prompts:

```
Operator Description:
  [Same as Round 1]

=== INSPIRATIONS ===
Inspiration 1: round_1_parallel_0 (1.8x speedup)
  DSL Code: [double buffering implementation]
  AscendC Code: [vectorized operations]
  Key optimizations:
    - Uses double buffering for memory
    - Vectorizes inner loops

Inspiration 2: round_1_parallel_1 (1.3x speedup)
  DSL Code: [row-wise processing]
  AscendC Code: [simple memory access]
  Key optimizations:
    - Row-wise tiling
    - Simplified logic
=== END INSPIRATIONS ===

Meta-Prompt: general.txt (different for each subagent)
  [Optimization hints]

Instruction: Learn from inspirations and explore new approaches

Execute skills: op_desc → pytorch → dsl → ascendc → eval
```

### How Diversity Emerges

**Same inspirations** + **Different meta-prompts** = **Different approaches**

- Subagent 0 (general.txt): Combines double buffering with cache optimization
- Subagent 1 (elementwise.txt): Focuses on wider SIMD operations
- Subagent 2 (reduction.txt): Explores different reduction patterns (if applicable)

## Tiered Sampling Benefits

1. **Exploitation**: Good tier ensures we keep improving
2. **Exploration**: Medium/Poor tiers prevent local optima
3. **Diversity**: Different approaches may work for different operators
4. **Robustness**: System doesn't get stuck on single strategy

## Example Evolution

### Round 1 Results
```
parallel_0: 1.3x (basic tiling)         → Poor tier
parallel_1: 1.8x (double buffering)     → Good tier ✓ Selected
parallel_2: 1.5x (row-wise processing)  → Medium tier ✓ Selected
```

### Round 2 Results (with inspirations)
```
parallel_0: 2.1x (combined best ideas)   → Good tier ✓ Best overall!
parallel_1: 1.9x (improved vectorization) → Good tier
parallel_2: 1.7x (different memory layout) → Medium tier
```

**Improvement**: 1.8x → 2.1x (17% gain from evolution)

---

# Subagent Execution

## Parallel Execution Model

The `cake_evo` agent first runs shared steps 1-4, then uses Claude Code's Task tool to spawn parallel subagents from step 5:

```
Agent runs steps 1-4 once → shared/ directory
    ↓
Agent copies shared files to each parallel directory
    ↓
Agent spawns 3 subagents simultaneously (from step 5):
  ├─ Subagent 0 (background) → output/round_1/parallel_0/
  ├─ Subagent 1 (background) → output/round_1/parallel_1/
  └─ Subagent 2 (background) → output/round_1/parallel_2/

Agent waits for all to complete, then evaluates results
```

## Subagent Prompt Structure

Each subagent receives:

1. **Pre-generated files**: op_desc, reference, functional, CMake project (already in output directory)
2. **Inspirations** (Round 2+): Previous successful implementations
3. **Meta-prompt**: Optimization hints (rotates)
4. **Output directory**: Isolated workspace
5. **Skill sequence**: Steps 5-9 to execute
6. **Local compilation mode**: Uses CANN environment

## Skill Execution

### Shared Steps (run once by cake_evo agent)
1. `op_desc_generation` - Generate operator JSON
2. `reference_generation` - Create PyTorch reference
3. `functional_conversion` - Convert to functional API
4. `ascend_call_generation` - Generate host code

### Per-Variant Steps (run by each parallel subagent)

#### Local Mode
1. `dsl_baseline_generation` - Generate DSL baseline
2. `dsl_lowering` - Apply lowering passes
3. `ascendc_code_review` - Check coding standards, review and fix code
4. `ascendc_evaluation` - Compile and evaluate locally

## Result Collection

After all subagents complete, agent:
1. Reads `evaluation_results.json` from each directory
2. Extracts: `speedup`, `compilation_success`, `precision_passed`
3. Reads generated code: DSL and AscendC
4. Classifies into tiers
5. Selects inspirations for next round

---

# Result Evaluation

## Evaluation Metrics

Each implementation is scored on:

### 1. Compilation Success
- ✅ **Success**: Code compiles without errors
- ❌ **Failure**: Compilation errors → Score = 0

### 2. Correctness
- ✅ **Pass**: Output matches PyTorch reference (99%+ match)
- ❌ **Fail**: Output mismatch → Score reduced

### 3. Performance
- **Speedup** = `pytorch_time / ascendc_time`
- Higher is better (>1.0x means faster than PyTorch)

## Scoring System

| Condition | Score | Interpretation |
|-----------|-------|----------------|
| Compilation failed | 0 | Not usable |
| Runtime error | 100 | Compiles but crashes |
| Precision failed | 200 | Runs but wrong output |
| All tests passed | 300 | Correct implementation |
| Speedup ≥ 2.0x | 350 | Optimal performance |

## Typical Speedup Ranges

| Operator Type | Expected Speedup |
|---------------|------------------|
| Element-wise | 2-5x |
| Reductions | 1.5-3x |
| Fused ops | 3-10x |
| Matrix ops | 1.2-2x |

## Classification Algorithm

```python
def classify_implementations(implementations):
    # Filter valid (speedup > 0, compiled successfully)
    valid = [impl for impl in implementations if impl.speedup > 0]

    # Sort by speedup (descending)
    valid.sort(key=lambda x: x.speedup, reverse=True)

    # Calculate tier sizes
    total = len(valid)
    good_count = max(1, int(total * 0.3))
    medium_count = max(1, int(total * 0.4))

    return {
        'good': valid[:good_count],
        'medium': valid[good_count:good_count + medium_count],
        'poor': valid[good_count + medium_count:]
    }
```

---

# Troubleshooting

## Common Issues

### All Subagents Failed

**Symptom**: No successful implementations in round

**Possible Causes**:
1. Operator description too vague
2. CANN environment issues
3. NPU memory issues

**Solutions**:
- Check `output/round_X/parallel_Y/execution.log` for errors
- Verify operator description is detailed
- Check CANN installation
- Verify NPU availability with `npu-smi`

### Low Speedup

**Symptom**: Best implementation slower than PyTorch

**Possible Causes**:
1. Operator is memory-bound (hard to optimize)
2. Small tensor sizes (overhead dominates)
3. PyTorch already well-optimized

**Solutions**:
- Increase `parallel_num` for more exploration
- Increase `max_rounds` for more iterations
- Try different meta-prompts manually
- Consider operator characteristics

### Compilation Failures

**Symptom**: AscendC compilation errors

**Solutions**:
```bash
# Check CANN installation
ascend-doctor

# Verify NPU devices
npu-smi info

# Check environment variables
echo $ASCEND_TOOLKIT_HOME
echo $LD_LIBRARY_PATH
```

### Subagent Timeout

**Symptom**: Subagent takes >30 minutes

**Causes**:
- Complex operator requiring many compilation iterations
- NPU under heavy load

**Solutions**:
- Be patient (complex ops take time)
- Check NPU utilization with `npu-smi`
- Reduce `parallel_num` to decrease load

## Best Practices

1. **Start small**: Use 2 rounds, 3 parallel candidates initially
2. **Monitor progress**: Check `PROGRESS.md` (updated by task-progress skill) for overall status
3. **Inspect failures**: Read execution logs for failed subagents
4. **Set realistic targets**: 1.5-2x speedup is typical, not 10x
5. **Detailed descriptions**: More detail = better results

## Debugging Commands

```bash
# View evolution progress
cat output/*_evo_*/PROGRESS.md

# Check specific subagent output
cat output/FastGELU_evo_*/round_1/parallel_0/execution.log

# View evaluation results
cat output/FastGELU_evo_*/round_1/parallel_0/evaluation_results.json

# Check all speedups in a round
grep "speedup" output/FastGELU_evo_*/round_1/*/evaluation_results.json

# Monitor NPU usage
npu-smi info -l
```

---

# Summary

## Key Concepts

1. **Evolutionary optimization**: Multiple rounds of parallel generation
2. **Tiered sampling**: Selection from good/medium/poor tiers for diversity
3. **Inspirations**: Learning from previous successful implementations
4. **Meta-prompts**: Different optimization hints for diverse approaches
5. **Agent-based**: Interactive workflow through Claude Code

## Workflow Recap

```
1. User describes operator to cake_evo agent
2. Agent detects local CANN environment
3. Agent runs shared steps 1-4 once (op_desc → reference → functional → ascend_call)
4. Round 1: Copy shared files, spawn N parallel subagents from step 5
5. Agent classifies results into tiers
6. Agent selects inspirations from all tiers
7. Round 2+: Copy shared files, generate variants with inspirations
8. Repeat until target achieved
9. Return top implementations
```

## When to Use Evolution

- **Complex operators**: Multiple optimization strategies possible
- **Performance-critical**: Need best-in-class performance
- **Uncertain approach**: Not sure which optimization works best
- **Learning**: Want to explore different implementations

## When NOT to Use Evolution

- **Simple operators**: Single obvious implementation
- **Time-constrained**: Need results quickly (use regular `cake` agent)
- **Good-enough performance**: 1.2x speedup sufficient

---

For more details, see:
- `evolution/README.md` - Module documentation
- `.claude/agents/cake_evo.md` - Agent definition
- `CLAUDE.md` - Project overview