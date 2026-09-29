---
name: regbase_api_misuse
description: API misuse patterns that frequently appear as regbase precision bugs, review failures, or broken repair loops.
title: API Misuse
purpose: Catch wrong API-family and wrong execution-model decisions before they are misdiagnosed as precision bugs.
read_when:
  - Outputs are structurally wrong, review points to wrong API families, or repair loops keep failing without a numeric explanation.
  - The task may have mixed regbase and membase assumptions.
not_for:
  - Happy-path greenfield design before any failure signal
  - Build or packaging troubleshooting
keywords:
  - api misuse
  - wrong family
  - regbase
  - review failure
next_reads:
  - ../api/regbase_api_whitelist.md
  - regbase_vs_membase_confusions.md
  - common_traps.md
depth: foundation
topic_type: pitfall
type: knowledge_card
platform: ascend950-regbase
tags: [api, misuse, regbase, review, debugging]
---

# API Misuse

Many "precision bugs" are actually API-family or execution-model mistakes. This document focuses on misuse classes that should be ruled out before tuning tolerances or widening precision.

## 1. Mixing regbase and membase mental models

Misuse:

- designing a regbase kernel as if the authoritative model were `LocalTensor + TPipe + TQue`

Why it fails:

- regbase work is defined by regbase APIs and patterns, not by the membase queue pipeline

Visible symptom:

- review says the branch fell back to default-template logic
- fixes keep rewriting buffer flow without improving the actual branch correctness

## 2. Skipping whitelist-first validation

Misuse:

- selecting an API because it looked familiar in another branch or example

Why it fails:

- the implementation can drift into unsupported or misleading usage before anyone verifies the regbase contract

Visible symptom:

- design and code disagree on which primitive is actually supposed to exist or be used

First check:

- read `../api/regbase_api_whitelist.md` before debugging implementation detail

## 3. Treating synchronization bugs as numeric noise

Misuse:

- blaming dtype or tolerance when the failure is unstable between runs

Why it fails:

- synchronization and event misuse often produce outputs that look like random numeric corruption

Visible symptom:

- repeated runs fail at different indices
- intermediate tensors look partially old or partially uninitialized

First check:

- read `../api/regbase_api_sync.md`
- determine whether the first bad value is stable across reruns

## 4. Assuming reduction validity across all shapes

Misuse:

- using a reduction path without validating width, tail, or alignment assumptions

Why it fails:

- the API may be mathematically right but operationally wrong for the tested shape

Visible symptom:

- only certain dimensions fail
- tails or minimum-size cases are unstable

## 5. Casting based on convenience instead of contract

Misuse:

- downcasting early to simplify storage or reuse
- comparing output against a reference that assumes a different cast mode

Why it fails:

- cast placement and rounding behavior can dominate the error budget

Visible symptom:

- stable systematic bias
- "small but stubborn" mismatches that survive algorithm fixes

## 6. Repairing the final output only

Misuse:

- clamping or recasting the last stage without checking where the first semantic misuse happened

Why it fails:

- the repair hides the symptom while preserving the wrong execution path

Visible symptom:

- one case improves, neighboring cases still fail
- review continues to flag wrong branch reasoning

## 7. Letting template scaffolding override regbase semantics

Misuse:

- treating the default direct-invoke template as the source of truth for how regbase compute must be structured

Why it fails:

- the template is a scaffold for project layout and build shape, not a proof that the compute path should inherit membase-style semantics

Visible symptom:

- queue- and buffer-oriented rewrites dominate the fix even when the real issue is API choice or numeric stability

## Repair Order

Before making a precision-only fix:

1. validate branch and API family
2. validate whitelist and sync assumptions
3. identify the first bad intermediate
4. only then tune dtype, cast point, or stabilization logic

## Related Documents

- [[regbase_vs_membase_confusions]]
- [[precision_guide]]
- [[symptom_to_cause]]
- [[precision_failures]]
- [[../api/regbase_api_sync]]
- [[../api/regbase_api_whitelist]]
