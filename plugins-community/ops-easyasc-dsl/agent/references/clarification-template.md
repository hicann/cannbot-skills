# Clarification Template

Use this only after repository evidence cannot resolve a semantic ambiguity.
Do not treat it as a workflow or a reason to ask broad design questions.

Ask one concise question that identifies:

- the exact unresolved behavior;
- the two or three materially different interpretations;
- the observable effect on ABI, numerical results, topology, or compatibility;
- the evidence already checked.

Example:

> The reference accepts both `[M, K]` and batched `[B, M, K]`, but the requested
> runtime ABI does not say whether batching is public or host-flattened. Should
> the kernel expose `B` as a runtime dimension, or is the contract strictly
> two-dimensional? This changes the `GMTensor` shape and core partitioning.

Do not ask the user to choose an implementation detail that source facts settle.
When a safe, reversible default is inside the stated contract, state the
assumption and continue.
