# API Index

This subtree is the regbase-authoritative API entry point for direct-invoke work. It groups the stable signature map, comparison and fallback notes, and the precision/runtime guardrails that usually decide whether an implementation is viable.

## How To Use This Index

- Read this index first when the question is primarily about API family, VF boundary, data movement, synchronization, or runtime guardrails.
- For one subproblem, shortlist by `purpose`, `read_when`, and `keywords`, then expand **at most 5 leaf documents** from this directory.
- If 5 API notes are still not enough, split the issue into narrower API subproblems or move to `patterns/` or `pitfalls/` instead of loading the whole subtree.

## Routing Table

| File | Purpose | Read When | Not For | Keywords |
|---|---|---|---|---|
| [Regbase API Whitelist](./regbase_api_whitelist.md) | verify allowed APIs and VF boundaries | you must confirm whether an API family is allowed before coding | perf tuning | whitelist, VF, Reg, MicroAPI |
| [Regbase API Reference](./regbase_api_reference.md) | compact signature map after availability is narrowed down | you need family-level call reminders | exhaustive header verification | signatures, LoadDist |
| [Regbase API Sync](./regbase_api_sync.md) | distinguish sync by layer | handoff, VF lane control, or overlap is the question | top-level route choice | sync, queue handoff |
| [DataCopy Best Practices](./datacopy_best_practices.md) | contrast compatibility DataCopy paths with regbase defaults | documenting a fallback or compatibility path | default regbase VF loads/stores | datacopy, fallback |
| [Pipeline and Buffer](./pipeline_and_buffer.md) | explain queued buffering only as compatibility material | comparing against TQue/TBuf fallback semantics | default regbase shell semantics | pipeline, TQue |
| [Arithmetic and Reduce](./arithmetic_and_reduce.md) | choose arithmetic and reduce API families | math is fixed but helper family is unclear | build questions | arithmetic, reduce |
| [Precision and Runtime](./precision_and_runtime.md) | combine cast, precision, and host launch guardrails | precision and runtime sequencing must be considered together | full kernel-structure design | precision, runtime |

## Common Paths

- Need API availability, VF-function boundary, and signature reminders: [[regbase_api_whitelist]] and [[regbase_api_reference]]
- Need copy and fallback compatibility notes: [[datacopy_best_practices]]
- Need buffering and compatibility notes: [[pipeline_and_buffer]] and [[regbase_api_sync]]
- Need scalar, broadcast, reduce, and compare constraints: [[arithmetic_and_reduce]]
- Need precision, host init, and runtime guardrails: [[precision_and_runtime]]

## Related Documents

- [[../index]]
- [[../patterns/index]]
- [[../pitfalls/index]]
- [[../dev-experience/index]]
- [[../../reference-ops/open_source_operator_table]]
