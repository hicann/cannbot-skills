# Evidence Escalation and Source Discovery

Use this route only when a focused pattern/example does not answer the current
boundary, repository behavior conflicts with the guidance, or a claim of DSL
infeasibility is being considered. It is an escape hatch from the pattern fast
path, not a default source-reading checklist.

## Entry triggers

Enter source discovery when any of these is true:

- no indexed pattern matches the required dataflow;
- a pattern conflicts with a public facade, stub, simulator result, or warning;
- a new dtype, dynamic shape, tail, alignment, or instruction combination is
  outside the pattern's stated scope;
- the same failing boundary survives an evidence-driven fix;
- simulator and generated C++ or hardware disagree;
- the next report would claim that EasyASC or one device lacks a capability.

## Narrow the question first

Write one evidence question before opening source. Good questions name the
device, primitive, and ambiguity, for example:

> On A2, does `count_per_rep` change only the live lanes, or also the physical
> source footprint of `cadd`?

Do not ask a scout to "understand vec" or "find a solution for the kernel".

## Evidence order

Stop as soon as the current question is answered with reproducible evidence:

1. `easyasc/a2.py`, `easyasc/a3.py`, `easyasc/a5.py`, or `easyasc/a5pr.py` public exposure;
2. the primitive stub signature and defaults;
3. parser/codegen lowering;
4. simulator dispatch, mask, stride, and footprint behavior;
5. focused parser/codegen/simulator tests;
6. generated C++;
7. a minimal hardware probe when codegen or device behavior remains material.

Simulator source is authoritative for simulator behavior, not for C++ name
resolution, compiler templates, instruction availability, or board bit
exactness. Escalate those questions to generated code or hardware.

## Direct reading versus a scout

The parent should read source directly for one primitive in one known file. Use
one read-only evidence scout when the question crosses several layers such as a
stub, parser, simulator, and tests. The parent still owns the contract,
candidate, architecture, validation order, and final claim.

A scout returns only this evidence packet:

```text
Question
Observed API contract
Instruction access model
Evidence with path:line
Simulator/HW uncertainty
Minimal probe
Conclusion and confidence
```

Do not return raw transcripts or edit the candidate. Follow the agent-count and
context rules in `agent/references/workflow-state-and-context.md`.

## Minimal probe rule

When source inspection still leaves an execution question, write one probe
under `tmp/` that isolates the primitive boundary. Start with an aligned case,
then add only the tail, multi-repeat, or hardware case needed to distinguish the
remaining hypotheses. Record the command, result, and first failing boundary.

For a shared-device launch/allocation failure after a long batch of runs, rerun
the single case in a fresh process on an idle device before classifying it as a
kernel defect. The isolated rerun must still compare outputs; it does not waive
compilation, numerical, or synchronization validation.

## Capability-absence gate

Before claiming that the DSL or a device lacks a required operation, record:

1. the public facade search;
2. stub/parser evidence;
3. simulator semantics;
4. the nearest composable primitive or pattern considered;
5. a minimal probe result;
6. generated-C++ or hardware evidence when the gap may be below the simulator.

If this gate is incomplete, report only that the current candidate has not yet
found a valid composition. Do not promote candidate failure into a DSL
capability claim.

## Return to the main path

Compress the result into one invariant, update the workflow boundary/evidence
ledger, and resume the selected playbook. Promote a new reusable pattern only
after its scope and runnable reference are clear; keep one-off findings in a
probe or pitfall record.
