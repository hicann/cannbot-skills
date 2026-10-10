# Patterns Index

This subtree captures the reusable implementation patterns that shape regbase-first kernel design, tiling, reduction routing, broadcast handling, and design-to-implementation decisions.

## How To Use This Index

- Read this index first when the task is already in the `regbase` branch and now needs a workable kernel shape.
- For one subproblem, shortlist by `purpose`, `read_when`, and `keywords`, then expand **at most 5 leaf documents** from this directory.
- If you still need more than 5 pattern notes, split the issue into smaller subproblems or move to `api/`, `pitfalls/`, or `dev-experience/` instead of scanning the whole subtree.

## Routing Table

| File | Purpose | Read When | Not For | Keywords |
|---|---|---|---|---|
| [Regbase Development Guide](../regbase_development_guide.md) | full practical mainline before choosing sub-patterns | you need the whole flow before drilling into one pattern | isolated API lookup | mainline, end-to-end |
| [Regbase Operator Patterns](./regbase_operator_patterns.md) | choose overall kernel shape and fusion boundary | first design route is still open | low-level sync or build questions | kernel shape, fusion boundary |
| [Regbase Kernel Dataflow Patterns](./regbase_kernel_dataflow_patterns.md) | separate shell, UB staging, and VF body | structuring a new kernel or reading a reference | host runtime-only questions | dataflow, VF, copyin/compute/copyout |
| [Regbase Buffer Partitioning](./regbase_buffer_partitioning.md) | plan GM, UB, and register residency | tile buffers and scratch placement are unclear | top-level route selection | GM, UB, RegTensor |
| [Regbase Sync Patterns](./regbase_sync_patterns.md) | choose sync by layer | handoff vs VF vs overlap boundaries are unclear | operator classification | sync, SetFlag, queue handoff |
| [Tiling Patterns](./tiling_patterns.md) | choose block and UB split strategy | tiling route must be fixed before coding | postmortem runtime debugging | tiling, block split |
| [Tiling Input Validation](./tiling_input_validation.md) | reject illegal dtype/format/dimension/attr/shape at tiling entry | tiling needs input validation or L2 exception cases need a rejection path | kernel-side runtime validation | validation, dtype, format, dimension, attr, shape, exception |
| [Reduction Patterns](./reduction_patterns.md) | route reductions by shape and algorithm family | reduction, norm, with-index, or ARA choices matter | pure broadcast work | reduction, AR, Welford |
| [Broadcast Patterns](./broadcast_patterns.md) | choose a broadcast route after dimension collapse | broadcast strategy affects the kernel shape | reduction-only design | broadcast, OneDim, NDDMA |
| [Kernel Design Patterns](./kernel_design_patterns.md) | keep design packets and repair loops coherent | moving from design into implementation or review | exact API signatures | walkthrough, repair loop |
| [Inplace Operator Patterns](./inplace_operator_patterns.md) | inplace 算子的 OpDef 同名规则、kernel 参数命名、冒烟测试与 TTK 用例 | 算子名含 Inplace 或需求文档声明输出复用输入地址 | 非 inplace 算子的常规开发流程 | inplace, OpDef 同名, var_out, output_inplace_indexes |

## Common Paths

- Need the full practical path before choosing a sub-pattern: [[../regbase_development_guide]]
- Need the overall regbase-first kernel shape and fusion boundary: [[regbase_operator_patterns]]
- Need the outer shell, UB flow, and VF decomposition: [[regbase_kernel_dataflow_patterns]]
- Need GM / UB / register domain planning: [[regbase_buffer_partitioning]]
- Need stage handoff, flags, and sync boundaries: [[regbase_sync_patterns]]
- Need block, UB, and split-axis planning: [[tiling_patterns]]
- Need tiling input validation (dtype/format/dimension/attr/shape): [[tiling_input_validation]]
- Need AR / ARA / multi-axis / with-index reduction routes: [[reduction_patterns]]
- Need OneDim / UB-broadcast / NDDMA-broadcast selection: [[broadcast_patterns]]
- Need design packet, implementation order, and repair loop habits: [[kernel_design_patterns]]
- Need inplace operator OpDef / kernel / test patterns: [[inplace_operator_patterns]]

## Related Documents

- [[../regbase_development_guide]]
- [[../index]]
- [[../api/index]]
- [Precision Guide](../pitfalls/precision_guide.md)
- [[inplace_operator_patterns]]
- [[../../reference-ops/open_source_operator_table]]
