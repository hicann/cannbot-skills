---
name: regbase_common_traps
description: High-frequency regbase debugging traps that waste repair cycles when treated as isolated implementation mistakes.
title: Common Traps
purpose: Surface the highest-frequency regbase debugging traps before repair work turns into blind trial and error.
read_when:
  - Review or runtime failures feel familiar but the exact root cause is still broad.
  - You are about to enter a repair loop and want the shortest list of high-yield checks.
not_for:
  - Exact numerical mechanism analysis
  - API signature lookup
keywords:
  - traps
  - debugging
  - repair loop
  - first checks
next_reads:
  - symptom_to_cause.md
  - api_misuse.md
  - precision_guide.md
depth: foundation
topic_type: pitfall
type: knowledge_card
platform: ascend950-regbase
tags: [pitfalls, debugging, traps, regbase, direct_invoke]
---

# Common Traps

This document lists the failure patterns that most often send direct-invoke repair work in the wrong direction. Each trap is written as an operational warning: what it looks like, why it happens, and what to check first.

## 1. Treating a branch error as a precision error

Signal:

- outputs are wildly wrong, all zero, or obviously off-scale
- review mentions a default template or membase-style implementation showing through in a regbase task

Why it happens:

- the design borrowed the wrong API family or wrong execution mental model

Check first:

- [[regbase_vs_membase_confusions]]
- [[api_misuse]]

## 2. Promoting too late

Signal:

- `fp32` patches are added but the failure barely improves
- the first large diff appears before the promoted step

Why it happens:

- the code upgrades precision after an unstable low-precision reduction, subtract, or nonlinear operation already destroyed signal

Check first:

- identify the earliest bad intermediate
- move promotion in front of the risky step

## 3. Assuming small-shape success proves correctness

Signal:

- unit tests on tiny or aligned shapes pass
- production shapes, long axes, or tails fail

Why it happens:

- small aligned cases hide tail handling, partial-tile masking, and scale-dependent accumulation drift

Check first:

- add one aligned case, one non-aligned case, and one long-axis case
- inspect whether the first failure starts at a tail or merge boundary

## 4. Using a single failure metric

Signal:

- mean error looks fine but a few samples explode
- relative error looks huge only around zero

Why it happens:

- the chosen metric hides the real failure mode

Check first:

- inspect absolute and relative error together
- print worst-case samples and their neighborhoods

## 5. Chasing the final output instead of the first divergence

Signal:

- many speculative fixes, little understanding
- every patch changes the surface symptom but not the root cause

Why it happens:

- the debugging loop compares only the final tensor

Check first:

- stage the formula
- compare the first unstable intermediate
- if quick checks stall, switch to a binary-search style intermediate walk

## 6. Trusting the template as semantic truth

Signal:

- regbase code is forced into queue- or pipeline-centric design just because the default direct-invoke template uses it

Why it happens:

- the template is an engineering skeleton, not the authoritative regbase implementation model

Check first:

- re-read the regbase whitelist and operator pattern notes
- confirm the design is regbase-first, not "template-first"

## 7. Ignoring hardware and shape constraints

Signal:

- only certain axis lengths or tile sizes fail
- behavior flips around minimum-size thresholds

Why it happens:

- reduction width, alignment, tail masks, or buffer limits were assumed instead of checked

Check first:

- shape boundaries
- minimum reduction width
- tail and mask logic

## 8. Treating non-determinism as harmless noise

Signal:

- identical input produces different mismatches
- re-running the same case changes which elements fail

Why it happens:

- synchronization or branch-selection problems are masquerading as precision drift

Check first:

- [[../api/regbase_api_sync]]
- branch and API-family correctness

## 9. Blindly widening everything to `fp32`

Signal:

- the bug disappears but performance or design clarity collapses
- no one can explain which step actually needed wider precision

Why it happens:

- the fix was chosen for convenience, not diagnosis

Check first:

- narrow the repair to the unstable sub-chain
- prove which intermediate requires widening

## 10. Using weak reference outputs

Signal:

- NPU output is compared against a reference that shares the same numerical weakness
- `fp16` and `bf16` are judged against another reduced-precision path

Why it happens:

- the golden path is not strong enough to expose the real error budget

Check first:

- prefer a higher-precision reference
- verify rounding and cast behavior assumptions before claiming 1 ULP bugs

## Related Documents

- [[precision_guide]]
- [[symptom_to_cause]]
- [[api_misuse]]
- [[precision_failures]]
- [[regbase_vs_membase_confusions]]
