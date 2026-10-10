# Knowledge Index

This subtree is the top-level knowledge map for regbase development. It routes agents into the four primary knowledge branches and the few stable cross-cutting entry points that stay useful across design, implementation, repair, and reference-implementation selection.

## How To Use This Index

- Read this first when you need the shortest path from a regbase task into the right knowledge branch.
- Choose the primary branch here, then read that directory's `index.md` before opening leaf notes.
- For one subproblem, expand **at most 5 leaf documents** inside the chosen directory. If that is still not enough, split the subproblem or move to another branch.

## Branch Selection Table

| Need | Read First | Then |
|---|---|---|
| Full practical end-to-end regbase development path | [Regbase Development Guide](./regbase_development_guide.md) | follow the linked notes as the implementation deepens |
| API constraints, VF boundaries, sync, or runtime guardrails | [API Index](./api/index.md) | pick leaf API notes by `purpose`, `read_when`, and `keywords` |
| Kernel structure, tiling, reduction, broadcast, or sync-by-layer patterns | [Patterns Index](./patterns/index.md) | pick the smallest pattern set that closes the current design question |
| Failure modes, precision risks, wrong-branch symptoms, or repair loops | [Pitfalls Index](./pitfalls/index.md) | branch from symptom or failure mechanism into the matching pitfall notes |
| Engineering judgment, route selection, build habits, or reference-reading advice | [Development Experience Index](./dev-experience/index.md) | choose experience notes that fit the current execution-stage question |
| Need at least one existing regbase implementation to inspect before coding | [Open Source Operator Table](../reference-ops/open_source_operator_table.md) | inspect the matched source implementation directly after the table |

## Related Documents

- [[../index]]
- [[regbase_development_guide]]
- [[api/index]]
- [[patterns/index]]
- [[pitfalls/index]]
- [[dev-experience/index]]
- [[../reference-ops/open_source_operator_table]]
