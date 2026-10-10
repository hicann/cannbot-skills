# Development Experience Index

This subtree stores distilled development experience for regbase work. Unlike the rest of `knowledge/`, these notes are not API cards or pattern taxonomies. They answer questions such as “what do experienced contributors re-check after context loss?”, “what requirements are easy to miss before design?”, and “what build habits prevent false progress on ascend950?”

## How To Use This Index

- Read this index first when the task needs engineering judgment, execution habits, or reference-reading advice rather than raw API lookup.
- For one subproblem, shortlist the most relevant notes by `purpose`, `read_when`, and `keywords`, then expand **at most 5 leaf documents** from this directory.
- If 5 notes are still not enough, split the question into smaller subproblems or switch to `api/`, `patterns/`, or `pitfalls/` instead of loading the whole directory.

## Routing Table

| File | Purpose | Read When | Not For | Keywords |
|---|---|---|---|---|
| [Regbase Development Guide](../regbase_development_guide.md) | Main end-to-end regbase development path | you need the full practical flow before drilling down | isolated API lookup | mainline, end-to-end |
| [Process Memory Cards](./process_memory_cards.md) | restore routed context after memory loss or handoff | long sessions, compression, ownership handoff | new algorithm design | context reload, handoff |
| [Requirements Analysis Patterns](./requirements_analysis_patterns.md) | lock dtype, shape, and precision requirements early | the task definition is still ambiguous | final API signatures | requirements, dtype, shape |
| [Regbase Programming Notes](./regbase_programming_notes.md) | keep the regbase mental model stable | code is drifting toward membase thinking | build/package issues | mental model, SPMD |
| [Regbase Build Notes](./regbase_build_notes.md) | avoid false build and packaging progress | starting compile/package/install work | VF compute-body design | build, artifacts |
| [Complexity And Route Selection](./complexity_and_route_selection.md) | classify the operator before coding | route choice is still unclear | low-level sync or buffer details | route, complexity |
| [Working Vs Failed 950 Cases](./working_vs_failed_950_cases.md) | compare stable and broken 950 patterns | you want concrete success/failure evidence | exact API lookup | 950 cases, failure evidence |
| [Regbase Performance Practices](./regbase_performance_practices.md) | apply performance-minded habits early | functional shape is clear and perf concerns matter | first correctness pass | performance, full-load |
| [Regbase Kernel Case Notes](./regbase_kernel_case_notes.md) | choose real regbase references to inspect | you must study at least one existing operator before coding | abstract theory only | reference implementation, case study |
| [Tiling Review Notes](./tiling_review_notes.md) | review tiling implementation risks | tiling code exists or design is about to land | first operator classification | tiling review, storage shape |
| [Small Channel Transpose Notes](./small_channel_transpose_notes.md) | preserve a specialized small-channel transpose case | transpose/layout-change with very small channels | default regbase skeleton design | transpose, small channel |

## Common Paths

- Need one practical “how do I actually develop a regbase operator?” main guide: [Regbase Development Guide](../regbase_development_guide.md)
- Long session, context compression, or handoff recovery: [Process Memory Cards](./process_memory_cards.md)
- Requirement ambiguity before design or implementation: [Requirements Analysis Patterns](./requirements_analysis_patterns.md)
- Regbase vs membase mental-model drift: [Regbase Programming Notes](./regbase_programming_notes.md)
- `ascend950` build, package, install, and artifact validation: [Regbase Build Notes](./regbase_build_notes.md)
- Unsure how complex the operator really is or which route to pick: [Complexity And Route Selection](./complexity_and_route_selection.md)
- Need concrete “working vs failed” 950 lessons before coding: [Working Vs Failed 950 Cases](./working_vs_failed_950_cases.md)
- Need performance-oriented implementation habits instead of API lookup: [Regbase Performance Practices](./regbase_performance_practices.md)
- Need real implementation reading advice before writing a new kernel: [Regbase Kernel Case Notes](./regbase_kernel_case_notes.md)
- Need a high-yield review pass over tiling implementation risk: [Tiling Review Notes](./tiling_review_notes.md)
- Need a proven small-channel transpose compatibility case from 910b experience: [Small Channel Transpose Notes](./small_channel_transpose_notes.md)

## Related Documents

- [[../index]]
- [[../regbase_development_guide]]
- [[process_memory_cards]]
- [[requirements_analysis_patterns]]
- [[regbase_programming_notes]]
- [[regbase_build_notes]]
- [[complexity_and_route_selection]]
- [[working_vs_failed_950_cases]]
- [[regbase_performance_practices]]
- [[regbase_kernel_case_notes]]
- [[tiling_review_notes]]
- [[small_channel_transpose_notes]]
- [[../../reference-ops/open_source_operator_table]]
