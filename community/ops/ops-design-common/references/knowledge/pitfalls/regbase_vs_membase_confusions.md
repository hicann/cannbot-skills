---
name: regbase_vs_membase_confusions
description: Decision aid for separating regbase-first guidance from membase compatibility habits that still leak into direct-invoke work.
title: Regbase vs Membase Confusions
purpose: Separate regbase-authoritative guidance from membase compatibility habits that keep leaking into direct-invoke work.
read_when:
  - Review feedback or code structure still behaves as if the template queue model were authoritative.
  - The task is in regbase but the mental model is drifting toward membase assumptions.
not_for:
  - Exact API signatures
  - Pure numerical failure analysis
keywords:
  - regbase vs membase
  - template drift
  - branch confusion
  - mental model
next_reads:
  - ../dev-experience/regbase_programming_notes.md
  - api_misuse.md
  - ../patterns/regbase_operator_patterns.md
depth: foundation
topic_type: pitfall
type: knowledge_card
platform: ascend950-regbase
tags: [regbase, membase, confusion, branch_selection, debugging]
---

# Regbase vs Membase Confusions

This document is for cases where the target implementation uses regbase, but the implementation or review conversation still behaves as if the authoritative guidance were membase-first.

## 1. The most common mistaken assumption

Mistake:

- "The default direct-invoke template shows the real implementation semantics, so regbase should follow the same queue and buffer model."

Correction:

- for regbase tasks, the template is an engineering scaffold
- regbase knowledge, whitelist, and pattern notes are the authoritative source for implementation semantics

## 2. Confusion table

| Confusion | Why It Is Wrong | What To Do Instead |
|---|---|---|
| "`ascend950` and `ascend910b` are interchangeable if the math is the same." | The math may match while API families, execution assumptions, and failure modes do not. | Keep the branch regbase-first for that target, and only treat membase as compatibility context. |
| "I can port a membase queue flow directly and optimize later." | That changes semantics before correctness is proven. | Re-derive the compute path from regbase docs, then use the template only for project skeleton concerns. |
| "If output is wrong, I should add more membase-style pipeline structure." | The issue may be wrong branch reasoning, wrong API choice, or unstable numerics. | First confirm API-family correctness and the first bad intermediate. |
| "Sync advice from membase examples applies unchanged." | Regbase and membase do not share identical synchronization assumptions or abstractions. | Use regbase sync guidance and inspect non-determinism before adding arbitrary barriers. |
| "A familiar API name means the usage contract is the same." | Similar names can hide different branch expectations and data movement assumptions. | Re-check the regbase whitelist and usage notes before implementation. |
| "Reference-ops or default template guidance outranks shared regbase docs." | In the regbase branch, shared regbase docs are the authoritative guidance source. | Use reference-ops only to choose an existing implementation to inspect; it does not outrank the shared regbase docs. |

## 3. How confusion shows up in practice

Typical failure shapes:

- review says the code "fell back to default template thinking"
- repair loops keep changing buffering structure while the same functional bug remains
- the implementation reads like a membase port with regbase vocabulary added afterwards
- precision debugging never converges because the branch semantics were wrong from the start

## 4. Recovery Pattern

When a regbase task shows membase leakage:

1. restate the branch decision: this task is regbase-first
2. identify the first place where membase assumptions entered the design
3. re-check API choice, synchronization, and compute structure against regbase guidance
4. only then continue with numeric tuning or performance repair

## 5. Safe Use of Membase Context

Membase context is still useful for:

- understanding compatibility constraints supplied by the caller
- recognizing which habits should not be copied into regbase
- reviewing whether a failure is actually caused by cross-branch confusion

It is not the source of truth for regbase implementation details.

## Related Documents

- [[api_misuse]]
- [[common_traps]]
- [[symptom_to_cause]]
- [[precision_guide]]
- [[../api/regbase_api_whitelist]]
- [[../patterns/regbase_operator_patterns]]
