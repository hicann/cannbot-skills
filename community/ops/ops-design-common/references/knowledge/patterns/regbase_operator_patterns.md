---
name: regbase_operator_patterns
description: Ascend950 regbase-first 直调算子的核心结构模式，覆盖 VF / Reg-MicroAPI 计算链、参考实现借鉴边界、规约后处理和 membase 兼容边界。
title: Regbase Operator Patterns
purpose: Choose the overall regbase kernel shape, fusion boundary, and greenfield-vs-reference route before line-level coding.
read_when:
  - The task is already in the regbase branch and needs a first design route.
  - You need to decide how much structure can be borrowed from an existing operator and what must stay greenfield.
not_for:
  - Detailed sync or build questions
  - Exact API signatures
keywords:
  - regbase operator
  - kernel shape
  - fusion boundary
  - reference reuse
next_reads:
  - ../regbase_development_guide.md
  - regbase_kernel_dataflow_patterns.md
  - ../dev-experience/regbase_kernel_case_notes.md
depth: foundation
topic_type: pattern
type: knowledge_card
platform: 950-regbase
verified: false
patterns: [operator, regbase, microapi_chain, fusion, direct_invoke]
---

# Ascend950 Regbase Operator Patterns

This file is the canonical `patterns/` home for regbase operator patterns. Use it when the target implementation uses `regbase` and you need to decide the kernel shape before coding.

## Applicability

- Platform: `ascend950`
- Branch: `regbase`
- Scope: direct-invoke, single-project kernel work
- Reuse rule: a candidate reference operator is optional; if none fits, keep the task in the regbase branch and design greenfield

## Core Route

1. Lock the math and dataflow first: pure elementwise, reduction plus post-process, broadcasted elementwise, or mixed fusion.
2. Decide whether there is a useful existing implementation to inspect. If there is, adapt only the reusable structure; if not, stay greenfield and record the reuse boundary explicitly.
3. Choose the tiling and branch form before final API selection.
4. Keep regbase-specific compute chains in `__VEC_SCOPE__` whenever the intermediate state is small and stable enough.

## Pattern 1: Single-Scope VF / Reg-MicroAPI Chain

Best for short elementwise or post-process chains such as:

- `cast -> muls -> add`
- `mul -> add -> relu`
- `sub(max) -> exp -> reciprocal`

Design guidance:

- Prefer one `__VEC_SCOPE__` for the hot path when the chain is linear and intermediate reuse is local.
- Delay write-back until the chain has produced a stable output, instead of bouncing temporary values through extra storage.
- Check API availability in `../api/index.md` before assuming a VF / Reg-MicroAPI chain is legal.

## Pattern 2: Load Once, Compute Many

Use this when one input tile can support several dependent operations after a single load:

- reduction followed by normalization or scale/bias
- elementwise preconditioning before a reduction
- multi-output accumulation derived from the same loaded tile

Design guidance:

- Treat the load boundary as expensive and the reg/UB compute chain as cheap.
- Keep the tile alive across consecutive operations when this removes duplicate GM traffic.
- Split the design only when live buffers, temporary state, or precision constraints force it.

## Pattern 3: Reduction Summary Plus Post-Process

Common regbase fusion shape:

1. Reduce a tile or row group into summary statistics.
2. Keep the summary in fp32 if the chain is precision-sensitive.
3. Apply the post-process immediately while the summary is still local.

Typical cases:

- RMSNorm / LayerNorm local phases
- Softmax local max-sum-normalize chain
- sum or max followed by broadcasted correction

Use [[reduction_patterns]] for the reduction route and [Precision Guide](../pitfalls/precision_guide.md) to decide when the post-process must stay in fp32.

## Pattern 4: Verified Reference Adaptation

When a useful existing reference implementation exists:

- reuse only the branch shape, tiling form, and proven control structure
- record what was reused, what changed, and what must be rewritten
- keep regbase API and dataflow decisions primary even if the reference came from an adjacent implementation family

When no useful reference implementation exists:

- do not invent reuse
- stay in the regbase branch
- use this patterns subtree plus `../../reference-ops/open_source_operator_table.md` only as reference-reading context

## Pattern 5: Membase Compatibility Boundary

`ascend910b / membase` remains a compatibility context, but this document is not a membase design guide.

When a design starts leaning on:

- `TPipe`
- `TQue`
- `DataCopyPad` as the central execution model
- membase-style queue choreography

then the task is no longer choosing a regbase pattern. At that point:

- stop importing membase habits into the regbase design
- re-check whether the branch or platform assumption is wrong
- fall back to the local regbase references first: `../api/index.md`, `tiling_patterns.md`, and `../pitfalls/regbase_vs_membase_confusions.md`

## Design Checklist

- Is the kernel primarily a VF / Reg-MicroAPI chain, a reduction chain, a broadcast chain, or a hybrid?
- Is the load boundary minimal, or are we reloading data that could stay local?
- Is any claimed reuse backed by an inspected reference implementation?
- Are fp32 summaries kept long enough for numerically sensitive post-processing?
- Are membase-only habits leaking into the regbase design?

## Related Documents

- [[tiling_patterns]]
- [[reduction_patterns]]
- [[broadcast_patterns]]
- [[kernel_design_patterns]]
- [Precision Guide](../pitfalls/precision_guide.md)
- [[../api/index]]
- [[../../reference-ops/open_source_operator_table]]
