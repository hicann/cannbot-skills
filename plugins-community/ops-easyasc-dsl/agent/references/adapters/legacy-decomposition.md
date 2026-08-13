# Legacy Decomposition Adapter

This adapter exists only for an already-created plan that predates the current
`refs/` handoff. It is not part of default decomposition or authoring.

Legacy plans may contain an author manifest, candidate files, or differently
named oracle slices. Read those files only long enough to map them into the
current contract:

- original oracle -> `refs/oracle.py`;
- planned composition -> `refs/compose.py`;
- graph -> `refs/dag.json`;
- sub references/checks -> `<sub>_ref.py` and `<sub>_check.py`;
- ABI and error budgets -> `refs/plan.md` with explicit `plan_tolerance` and
  `implementation_tolerance`.

Run the full current handoff gate after adaptation. Do not infer a missing
tolerance from an old script, and do not preserve candidate or tuning metadata
in the normal authoring worklist. Existing tool support for reading a legacy
manifest may be used, but new plans must not emit one.
