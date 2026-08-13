# Optional Decomposition Worker

Read this only when the user explicitly requests parallel decomposition. The
normal workflow is local and produces one plan.

## Ownership

One worker owns one complete `{plan_dir}/refs/` handoff. It may write only:

```text
refs/oracle.py
refs/compose.py
refs/dag.json
refs/plan.md
refs/<sub>_ref.py
refs/<sub>_check.py
```

It must not write kernels, candidates, manifests, reports, or shared root
files. Parallel workers need distinct plan directories.

## Prompt contract

Provide the worker with the original PyTorch function, target device,
`plan_tolerance`, `implementation_tolerance`, and plan directory. Require it to
follow `agent/playbooks/decompose-pytorch.md` and:

- freeze the original behavior unchanged in `oracle.py`;
- settle one complete node DAG and every edge ABI;
- model every dtype, layout, cast, lossy boundary, and matmul precision path in
  the planned refs;
- record both tolerance classes and any justified per-node implementation
  override in `plan.md`;
- keep refs pure PyTorch and independently runnable;
- stop with a blocker instead of publishing an open semantic question.

## Return gate

Before returning, the worker must run all sub checks, `compose.py`, Python
compilation, and DAG validation from the main playbook. Its response contains
only files changed, commands/results, and blockers. The main agent reviews the
handoff before declaring `refs/` verified and immutable.
