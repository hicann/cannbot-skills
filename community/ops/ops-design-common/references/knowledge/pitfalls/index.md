# Pitfalls Index

This subtree is the regbase troubleshooting entry point for failure modes, precision regressions, API misuse, and regbase-vs-membase confusion.

## How To Use This Index

- Read this index first when the task has already moved into repair, debugging, or review and the issue is about failure shape rather than greenfield design.
- For one subproblem, shortlist by `purpose`, `read_when`, and `keywords`, then expand **at most 5 leaf documents** from this directory.
- If 5 pitfall notes are still not enough, split the issue by symptom family or move to `api/` or `patterns/` to verify the structural assumptions.

## Routing Table

| File | Purpose | Read When | Not For | Keywords |
|---|---|---|---|---|
| [Precision Guide](./precision_guide.md) | primary precision-prevention and triage guide | reductions, nonlinear chains, casts, or mixed precision are involved | pure build issues | precision, mixed precision |
| [Common Traps](./common_traps.md) | shortest high-yield debugging trap list | you are entering a repair loop or review flagged a familiar failure | exact numerical mechanism analysis | traps, repair loop |
| [Symptom to Cause](./symptom_to_cause.md) | start from the visible failure shape | the symptom is clearer than the mechanism | greenfield design | symptom, triage |
| [API Misuse](./api_misuse.md) | catch wrong API-family decisions | outputs are structurally wrong or review points to wrong API surfaces | happy-path design | api misuse, wrong family |
| [Precision Failures](./precision_failures.md) | map numerical failures to repair directions | branch and API family already look correct but tolerance fails | structural branch mistakes | instability, reduction drift |
| [Regbase vs Membase Confusions](./regbase_vs_membase_confusions.md) | separate regbase from template-default thinking | code or review still behaves as if membase were authoritative | exact API signatures | regbase vs membase |

## Common Paths

- Need the main precision-prevention and triage guide: [Precision Guide](./precision_guide.md)
- Need a broad first-pass debug shortlist: [Common Traps](./common_traps.md)
- Need to start from the visible failure shape: [Symptom to Cause](./symptom_to_cause.md)
- Suspect wrong API-family usage: [API Misuse](./api_misuse.md)
- Know the issue is numerical but need failure mechanisms: [Precision Failures](./precision_failures.md)
- Suspect the task drifted from regbase into membase thinking: [Regbase vs Membase Confusions](./regbase_vs_membase_confusions.md)

## Related Documents

- [[precision_guide]]
- [[common_traps]]
- [[symptom_to_cause]]
- [[api_misuse]]
- [[precision_failures]]
- [[regbase_vs_membase_confusions]]
- [[../api/regbase_api_sync]]
- [[../api/regbase_api_whitelist]]
- [[../patterns/regbase_operator_patterns]]
