---
name: regbase_precision_failures
description: Failure-pattern library for numerical instability in regbase direct-invoke operators, organized by failure mechanism and repair direction.
title: Precision Failures
purpose: Group numerical failures by mechanism so repair work targets the real instability instead of symptoms.
read_when:
  - The branch and API family already look correct but tolerance still fails.
  - Errors grow with axis length, dtype downgrade, exp/log chains, or normalization stages.
not_for:
  - Structural wrong-branch issues
  - Environment or build failures
keywords:
  - precision failure
  - instability
  - reduction drift
  - norm
next_reads:
  - precision_guide.md
  - symptom_to_cause.md
  - ../api/precision_and_runtime.md
depth: intermediate
topic_type: pitfall
type: knowledge_card
platform: ascend950-regbase
tags: [precision, failures, softmax, reduce, norm, cancellation]
---

# Precision Failures

This document groups numerical failures by mechanism rather than by operator name. Use it when the code path is already known to be regbase-correct but still fails tolerance or produces unstable values.

## 1. Long Reduction Drift

Failure signature:

- error grows with axis length or shape size
- short reductions pass, long reductions fail

Common cause:

- accumulation stayed in low precision or merged too aggressively

Repair direction:

- promote the accumulator
- inspect partial sums and merge boundaries
- verify long-axis and tail cases explicitly

## 2. Overflow in Exponential Chains

Failure signature:

- `inf`, `nan`, or saturated outputs
- only large positive inputs fail
- softmax normalization collapses

Common cause:

- `exp` is applied before range shift or clamp strategy

Repair direction:

- subtract max before exponentiation
- keep the nonlinear core in `fp32`
- verify both pre-shift and post-shift ranges

## 3. Underflow or Collapse in Very Small Magnitudes

Failure signature:

- outputs collapse to zero
- small-magnitude signals disappear after multiple operations

Common cause:

- early cast-down or repeated low-precision transforms erase the signal

Repair direction:

- delay cast-down
- compare the first non-zero intermediate between reference and kernel

## 4. Cancellation Around Near-Equal Values

Failure signature:

- large relative error in a narrow numeric band
- subtracting two close values causes sudden degradation

Common cause:

- unstable algebraic form executed in low precision

Repair direction:

- reformulate the expression when possible
- promote before the subtract
- inspect the operands, not just the result

## 5. Reciprocal and Divide Instability

Failure signature:

- isolated huge values
- `nan` clusters around zeros or tiny denominators

Common cause:

- denominator guard is missing, misplaced, or too weak

Repair direction:

- add epsilon or clamp by explicit policy
- verify the denominator distribution before reciprocal

## 6. Norm and Variance Path Corruption

Failure signature:

- `rms_norm`, `layer_norm`, or variance-like paths fail more than plain elementwise kernels
- failure often appears after square-sum, mean, or `rsqrt`

Common cause:

- low-precision accumulation plus poor epsilon placement

Repair direction:

- promote the statistic path
- add epsilon before reciprocal or `rsqrt`
- verify pre-normalization statistics separately

## 7. Tail-Only Precision Failure

Failure signature:

- most of the tensor matches, but the final tile or final few elements fail

Common cause:

- tail mask, valid-length propagation, or boundary merge logic is wrong

Repair direction:

- compare the first failing index to tile boundaries
- isolate aligned and non-aligned runs

## 8. Reference-Quality Mismatch

Failure signature:

- low-precision output appears "wrong" only against a weak golden path
- 1-2 ULP style mismatches dominate

Common cause:

- the reference implementation does not match required cast behavior or uses insufficient precision

Repair direction:

- strengthen the golden path before changing the kernel
- separate "real bug" from "expected dtype budget"

## Fast Classification Questions

- Does error growth track axis length? If yes, start with reduction drift.
- Do failures cluster on large magnitudes? If yes, inspect overflow and cast placement.
- Do failures cluster near zero? If yes, inspect cancellation or denominator protection.
- Do only tail elements fail? If yes, inspect valid-length logic before precision math.

## Related Documents

- [[precision_guide]]
- [[common_traps]]
- [[symptom_to_cause]]
- [[api_misuse]]
