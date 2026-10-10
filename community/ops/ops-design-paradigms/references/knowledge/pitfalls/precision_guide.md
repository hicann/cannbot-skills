---
name: regbase_precision_guide
description: Regbase direct-invoke precision guide focused on prevention, failure triage, and stable mixed-precision design.
title: Precision Guide
purpose: Provide the primary precision-prevention and precision-triage guide for regbase direct-invoke work.
read_when:
  - Design or repair touches reductions, nonlinear chains, casts, norms, or mixed precision.
  - You need to decide whether the issue is truly numerical before changing code.
not_for:
  - Pure build/environment setup
  - General route selection with no precision signal
keywords:
  - precision
  - mixed precision
  - triage
  - prevention
next_reads:
  - precision_failures.md
  - api_misuse.md
  - ../api/precision_and_runtime.md
depth: foundation
topic_type: pitfall
type: knowledge_card
platform: ascend950-regbase
tags: [precision, regbase, mixed_precision, reduce, softmax, norm, debugging]
---

# Regbase Precision Guide

This guide keeps the highest-signal precision rules for regbase direct-invoke work. Use it at design time to avoid predictable numerical failures, and during repair when the implementation is functionally plausible but still fails tolerance.

## 1. Fast Triage

Ask these questions before changing code:

1. Is this a real precision problem, or is the output structurally wrong because the wrong branch, API family, or buffer path was used?
2. Does `fp32` pass while `fp16` or `bf16` fails?
3. Does the first large error appear at a reduction, cast-down, `exp`, `log`, `rsqrt`, or divide?
4. Does the error grow with axis length, input magnitude, or shape size?
5. Is the failure concentrated at tails, border tiles, or non-aligned shapes?

If the answer to 1 is uncertain, read [[api_misuse]] and [[regbase_vs_membase_confusions]] before treating the issue as pure precision drift.

## 2. Core Design Rules

### 2.1 Promote early, not after damage

Use higher precision before the risky operation, not after it:

- reductions and long accumulations should promote before accumulation
- `exp`, `log`, `rsqrt`, reciprocal, and division chains should keep critical intermediates in `fp32`
- norm-style pipelines should promote the square, sum, mean, epsilon-add, and reciprocal-sqrt path

Late promotion cannot recover information already lost in low precision.

### 2.2 Delay cast-down until the interface boundary

Downcast only when one of these is true:

- the output contract requires it
- a stable intermediate has already been finalized
- storage or bandwidth constraints force it and the error budget is still acceptable

Bad pattern:

- `fp16 -> exp -> reduce -> reciprocal -> fp16 normalize`

Safer pattern:

- `fp16 input -> fp32 critical chain -> final cast`

### 2.3 Treat long reductions as error amplifiers

Reduction errors usually scale with one or more of these:

- axis length
- mixed-sign accumulation
- repeated partial merges
- low-precision accumulator reuse

Use platform-supported reduce paths where possible, but still reason about accumulator dtype and partial merge order.

### 2.4 Stabilize before nonlinearity

Common stable transforms:

- softmax: subtract row max before `exp`
- log-sum-exp: factor out the max before summing exponentials
- norm and variance: accumulate in `fp32`, add epsilon before reciprocal or `rsqrt`
- division: protect tiny denominators before reciprocal or divide

### 2.5 Compare the right thing

Tolerance failures are often misread because the wrong metric is used:

- use absolute error for small-magnitude outputs near zero
- use relative error for scale-sensitive outputs
- inspect worst samples, not only aggregate means
- compare `fp16/bf16` output against a high-quality reference, not another weak baseline

## 3. High-Risk Operation Classes

### 3.1 Reduction and accumulation

Typical failures:

- `ReduceSum`, `mean`, variance, norm statistics drift as axis length grows
- outputs are close on small shapes but fail on large ones

Preferred response:

- move the accumulator to `fp32`
- inspect partial sums, not only final output
- verify tails and tile merges separately

### 3.2 Exponential and logarithmic chains

Typical failures:

- `inf`, `nan`, or saturated outputs
- softmax rows no longer sum near 1
- only large positive or very negative inputs fail

Preferred response:

- range-shift before `exp`
- guard log input domain
- keep the central chain in `fp32`

### 3.3 Cancellation-sensitive formulas

Typical failures:

- large relative error when subtracting two close values
- failure appears only in a narrow input band

Preferred response:

- algebraically reformulate when possible
- promote just before the subtract
- compare intermediate values around the first divergence

### 3.4 Divide and reciprocal chains

Typical failures:

- isolated `nan` or very large spikes
- error correlates with very small denominators

Preferred response:

- add or clamp epsilon with an intentional rule
- check whether zero and near-zero should map to zero, clamp, or fail fast

## 4. Regbase-Specific Precision Checks

Precision analysis in regbase must also confirm the execution path itself is regbase-correct:

- API choices should match the regbase whitelist before numerical tuning starts
- if a design borrowed membase `TPipe`, `TQue`, or queue-stage thinking as the semantic source, the failure may be a branch mistake instead of a precision issue
- synchronization mistakes can surface as unstable or garbage numerics; read [[../api/regbase_api_sync]] when diff patterns look non-deterministic

## 5. Verification Ladder

Use this order instead of jumping straight to large random tests:

1. aligned `fp32` sanity case to prove the formula path
2. aligned low-precision case to isolate dtype loss
3. non-aligned or tail case to isolate boundary handling
4. boundary-value set to trigger overflow, underflow, or divide instability
5. production-like shapes to confirm the repair survives scale

If a repair only fixes one layer of this ladder, it is not complete.

## 6. Repair Checklist

- [ ] Confirm the bug is not actually API-family misuse or branch confusion
- [ ] Identify the first numerically bad intermediate, not only the final mismatch
- [ ] Promote the smallest critical sub-chain that removes the instability
- [ ] Re-check reduction size, tails, and shape scaling
- [ ] Re-run both aligned and non-aligned cases
- [ ] Re-check tolerances against dtype and operator class

## Related Documents

- [[common_traps]]
- [[symptom_to_cause]]
- [[precision_failures]]
- [[api_misuse]]
- [[regbase_vs_membase_confusions]]
