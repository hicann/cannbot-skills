- Start every conversation at `agent/ROUTER.md`. Select exactly one workflow,
  then read `agent/common-language.md` in full before opening the selected
  playbook. The router and common language are the fixed initial documentation
  baseline.
- Follow the selected workflow's evidence path. The normal order is
  `ROUTER.md` -> `common-language.md` -> one playbook -> the authoring preflight
  when applicable -> one device fact or focused pattern/example -> source only
  when smaller layers are insufficient.
- Do not preload a large catalog. Use selectors or indexes before catalogs and
  revisit only the relevant common-language section when resolving an alias.

## Repository evidence

- Before guessing about runtime behavior, inspect the implementation in this
  order: `easyasc/stub_functions/`, `easyasc/parser/`, then
  `easyasc/simulator/` when execution semantics matter.
- Treat `easyasc/a2.py`, `easyasc/a5.py`, and `easyasc/a5pr.py` as the public
  target facades. Do not mix target APIs or import private vector helpers from a
  kernel.
- Existing kernels are evidence, not copy-paste templates. Unless a task is
  tied to an existing kernel, create a new kernel file by default.
- A repository path mentioned by a document must resolve. Correct a stale path
  against the repository layout instead of trusting the document.

## Kernel work

- Authoring workflows must read `agent/references/authoring-preflight.md` before
  implementation. It owns the constant-condition, generated-name, public-I/O,
  synchronization, and device-limit checks.
- Build incrementally: settle tiling and dataflow first, then implement and
  validate each stage. Justify every operation, cast, buffer, synchronization
  edge, and datamove from the contract and hardware behavior.
- Before golden inputs enter the kernel, only shape-only transforms are allowed:
  `squeeze`, `unsqueeze`, and `reshape`. Do not change values or layout unless
  the user explicitly requests it.
- Treat warnings as evidence of an incomplete model. Investigate their cause;
  for `auto_sync` warnings, fix the kernel or propose a concrete parser change.
- Record every kernel change in `agent/diary.md` while making it. The diary is
  internal history only and holds only unresolved work. Closing a task is a
  paired action in one commit: move stable conclusions to the owning reference
  or catalog, then delete the diary entry. Never cite the diary from repository
  documentation.
- CANN Bench work (competition kernels, contract tests, evaluation reports,
  submission tooling, and its own diary) lives in the sibling
  `easyasc_cannbench_kernels` extension repository. The
  `cann-bench-aclnn-evaluation.md` playbook stays here and routes into it.

## Validation and environment

- Use `OpExec(..., simulator=True)` for execution and debugging unless the user
  explicitly asks for CANNSIM or real hardware.
- For a requested CANNSIM run use
  `OpExec(..., simulator=False, cannsim=True)`. It may run for several minutes
  and emit substantial logs.
- If `machine_specs.md` exists, it is the local, git-ignored owner for real-A5
  access. Do not copy machine details into tracked files.
- Use the `torch210npu` conda environment when available; otherwise use the
  default environment.
- Simulator multiprocessing reloads the main script. Put temporary runners
  under `tmp/<task>/`; do not launch them from stdin.
- Keep Python type hints compatible with Python 3.8. All code, error messages,
  and repository documentation must be in English.

## Owner documents

- repository layout and subsystem ownership: `agent/references/repo-map.md` and
  `doc/11_architecture_for_contributors.md`
- implementation lookup: `agent/references/code-paths.md`
- kernel and tool examples: `agent/references/examples/kernel-catalog.md` and
  `agent/references/examples/tool-catalog.md`
- tests and fixtures: `agent/example/testcases/README.md`
- public entry maps: `README.md`, `README_CN.md`, `doc/`, and `doc_cn/`
- reusable composite decompositions: `agent/references/composite-api-recipes.md`

## Python formatting

- Keep readable function signatures and calls compact. Do not mechanically put
  every argument on its own line or add trailing commas that force expansion.
