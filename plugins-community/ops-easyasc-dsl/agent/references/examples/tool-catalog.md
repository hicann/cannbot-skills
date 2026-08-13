# Tool Catalog

Use this file to choose a repository tool before opening or modifying the script itself.
This is a selection index, not an implementation reference.

## Index schema

This file is also the machine-readable metadata source for `agent/index/tools.json`.
The index builder reads:
- each `###` entry heading as one tool record
- the surrounding `##` section as the tool category
- top-level entry bullets such as `purpose`, `use_for`, `not_for`, `important_outputs`, and `pair_with`
- nested bullet items under those fields as ordered list values

If you edit this catalog, keep that field structure stable.

## Current tools

### `agent/scripts/analyze_sim_trace.py`
- purpose:
  - summarize easyasc simulator Chrome trace JSON timing into total makespan and pipe utilization tables
  - compute the two standard utilization ratios, active/span and active/makespan, from exported timed events
- use_for:
  - interpreting `OpExec(..., simulator=True, trace=...)` outputs
  - comparing scheduling changes such as DBuff/lookahead traces
  - finding which pipe family is saturated across cores
  - drilling from global pipe averages into per-core pipe utilization
- not_for:
  - launching kernels
  - validating numerical correctness
  - replacing Chrome trace visual inspection for detailed dependency ordering
- important_outputs:
  - total trace makespan in simulator cycles when `args.time_domain == "cycle"`
  - average utilization by pipe across cores, with default `MTE2` split into `cube.MTE2` and `vec.MTE2`
  - per-core utilization by pipe
  - optional JSON output for downstream comparison scripts
- pair_with:
  - `agent/references/facts-simulator-opexec.md`
  - `agent/references/cycle-model.md`

Sample entry points:
- `python agent/scripts/analyze_sim_trace.py tmp/<task>/trace.json` — prints total time plus average and per-core utilization grouped by pipe label, with `cube.MTE2` and `vec.MTE2` separated
- `python agent/scripts/analyze_sim_trace.py tmp/<task>/trace.json --group-by lane-pipe` — keeps `vec0.MTE2` and `vec1.MTE2` as separate pipe labels
- `python agent/scripts/analyze_sim_trace.py tmp/<task>/trace.json --json` — emits the same metrics as JSON

### `agent/scripts/estimate_matmul_datamove.py`
- purpose:
  - estimate matmul data movement for candidate tile/core-split strategies
  - reject illegal tile-space combinations before kernel authoring goes too far
- use_for:
  - choosing `TILE_M`, `TILE_N`, `TILE_K`
  - comparing `split_m`, `split_n`, and `mix`
  - checking `dbuf_left`, `dbuf_right`, and fixed `dbuf_l0c` capacity assumptions
  - resolving large-matmul strategy questions before touching kernel code
- not_for:
  - launching kernels
  - validating numerical correctness
  - replacing simulator-based output checks
- important_outputs:
  - per-core datamove estimate
  - best strategy candidate set
  - PrettyTable strategy report
  - expansion ratio and DBUF-aware capacity decisions
- pair_with:
  - `agent/references/constraints/tiling.md`
  - large tiled examples such as `agent/example/kernels/a5/matmul/matmul_mknk_2dgrid_splitn.py`, `agent/example/kernels/a5/matmul/matmul_mknk_2dgrid_splitk.py`, and `agent/example/kernels/a5/matmul/matmul_mknk_2dgrid_splitk_add1.py`

Core APIs:
- `estimate_percore_datamove(m, n, k, TILEM, TILEN, TILEK, mode, dbuf_left=True, dbuf_right=True, dbuf_l0c=True)` — estimates single-core data movement for one fixed loop mode; valid `mode` values are `left_first`, `right_first`, and `balanced`
- `estimate_multi_core(m, n, k, m_split, n_split, TILEM, TILEN, TILEK, nonempty_only=False, dbuf_left=True, dbuf_right=True, dbuf_l0c=True)` — tries all three loop modes, returns minimum total data movement; `nonempty_only=True` counts only non-empty split blocks
- `estimate_strategy(m, n, k, num_core, split_mode, min_tile_m=None, min_tile_n=None, dbuf_l0c=True)` — brute-force searches split, tile, mode, and legal DBUF combinations; returns dict with `baseline_datamove`, `best_datamove`, `best_results`, and PrettyTable `table`

Capacity model:
- `MAX_TOTAL_TILE_ELEMENTS = 128 * 1024`
- `dbuf_l0c=True`: `TILEM * TILEN <= 32 * 1024`; `dbuf_l0c=False`: `TILEM * TILEN <= 64 * 1024`
- DBUF flags affect capacity only; they do not change data-movement formulas

Datamove rules:
- `TILEK` alignment is used only in data-movement calculation, not in capacity calculation
- When `TILEK != k`, the K-loop span is `CeilDiv(k, TILEK) * CeilDiv(TILEK, 256) * 256`, applied only to the operand traversed by the `TILEK` loop in the selected mode

Strategy search rules:
- Split candidates: `split_m` only `(num_core, 1)`; `split_n` only `(1, num_core)`; `mix` all factor pairs whose product is `num_core`
- Tile candidates: `TILEM`, `TILEN` from `[32, 64, 128, 256, 512]`; `TILEK` from `[32, 64, 128, 256, 512]` values `<= k`, plus `k` itself
- DBUF candidates are mode-constrained: `balanced` only `(True, True)`; `left_first` adds `(False, True)`; `right_first` adds `(True, False)`
- `dbuf_l0c` is a fixed input to `estimate_strategy(...)`, not searched
- `estimate_strategy(...)` always evaluates with `nonempty_only=False`

Sample entry points:
- Python API only: `python -c 'from tools.estimate_matmul_datamove import estimate_strategy; print(estimate_strategy(1024, 512, 128, 20, "split_n")["table"])'` — prints a strategy table for one concrete shape; direct `python agent/scripts/estimate_matmul_datamove.py` is currently silent

### `agent/scripts/select_kernel_example.py`
- purpose:
  - rank existing kernel examples for a new task using the generated kernel index
  - reduce manual catalog and source-file scanning before kernel authoring
- use_for:
  - selecting a first kernel to study by dataflow pattern, topology, query text, tags, or lightweight features
  - narrowing candidate examples before opening `agent/example/kernels/` source files
  - surfacing `study_for` and `do_not_copy_when` guidance together with the example path
- not_for:
  - generating kernel logic
  - proving that one example is the uniquely correct template
  - replacing constraint/reference reading for tiling, autosync, counters, or precision
- important_outputs:
  - ranked kernel paths
  - reusable dataflow-pattern ids carried by matching catalog entries
  - short match reasons
  - `study_for` and `do_not_copy_when` summaries
  - optional JSON output for tool chaining
- pair_with:
  - `agent/references/pattern-index.md`
  - `agent/references/examples/kernel-catalog.md`
  - `agent/index/kernels.json`
  - `agent/references/authoring-overview.md`
  - `agent/references/constraints/`

Sample entry points:
- `python agent/scripts/select_kernel_example.py --device a2 --pattern vec-row-reduce-broadcast --topology vec-only --catalog` — returns A2 examples already validated for that reusable dataflow
- `python agent/scripts/select_kernel_example.py --query "matmul add1" --topology 'cube->vec'` — ranks mixed-pipeline examples for an add-after-matmul task
- `python agent/scripts/select_kernel_example.py --topology cube-only --tag splitk` — narrows to pure cube split-k references

### `agent/scripts/gen_kernel_from_manifest.py`
- purpose:
  - validate legacy `author_manifest.json` handoff files from older decomposition flows
  - generate deterministic DSL kernel validation scaffolds for legacy manifest-based authoring
  - preserve the old scaffold path without making manifests part of the default plan-local workflow
- use_for:
  - checking that a legacy decomposition candidate is structurally consumable
  - generating `tmp/<task>/{hw_target}/{candidate_id}/{sub}.py` scaffolds from legacy manifest metadata
  - recovering or comparing old sandbox outputs that already contain an `author_manifest.json`
  - catching unsupported public tensor dtypes and missing symbolic-dimension bindings before simulator runtime
- not_for:
  - the normal plan-local `refs/` -> `agent/example/kernels/` authoring path in `agent/playbooks/author-kernel-from-decomposition.md`
  - implementing the DSL kernel body
  - choosing tile sizes or schedules
  - proving functional correctness after the body is filled in
- important_outputs:
  - `OK: ...` validation summary for `--validate-only`
  - generated scaffold paths, one per selected sub-kernel
  - full generated source on stdout with `--dry-run`
- pair_with:
  - `agent/references/adapters/legacy-decomposition.md`
  - `agent/playbooks/author-kernel-from-decomposition.md`
  - `agent/references/facts-simulator-opexec.md`
- sample_entry_points:
  - `python agent/scripts/gen_kernel_from_manifest.py --manifest foo_decomposition/A/author_manifest.json --validate-only` — checks a legacy manifest without writing files
  - `python agent/scripts/gen_kernel_from_manifest.py --manifest foo_decomposition/A/author_manifest.json --out tmp/<task>/a5/A --sub sub1 --overwrite` — writes one legacy generated sub-kernel scaffold

### `agent/scripts/check_kernel_catalog.py`
- purpose:
  - verify that the human-readable kernel catalog, generated kernel index, and current kernel-source guardrails stay consistent
  - catch source-of-truth drift before selectors or routing layers start returning stale metadata
- use_for:
  - checking that every catalog entry points to a real kernel file
  - checking that required metadata fields are present and non-empty
  - recursively checking that selectable files under `agent/example/kernels/**/*.py` with a top-level `@kernel` are not missing from the catalog
  - excluding runners/helpers, private `_*.py` diagnostics, `baseline_original/` snapshots, and `*_original.py` preserved copies from selectable-kernel coverage
  - checking that the lean human index and detailed catalog contain the same concrete `.py` paths
  - checking whether `agent/index/kernels.json` is stale relative to the current catalog
  - catching legacy UB row-scalar allocations whose second dimension is statically `1`
- not_for:
  - rewriting catalog entries automatically
  - ranking or selecting kernels for study
  - validating kernel numerical correctness
- important_outputs:
  - consistency warning codes such as missing file, missing field, recursively uncataloged kernel, lean-index drift, stale machine index, or legacy UB scalar layout
  - optional JSON output for CI or tool chaining
- pair_with:
  - `agent/references/examples/kernel-catalog.md`
  - `agent/index/kernels.json`
  - `agent/scripts/build_agent_index.py`
  - `agent/scripts/select_kernel_example.py`

Warning codes: `duplicate-entry-path`, `missing-file`, `missing-field`, `malformed-field`, `uncataloged-kernel`, `missing-lean-index`, `missing-lean-entry`, `orphan-lean-entry`, `ub-second-dim-one`, `invalid-kernel-python`, `missing-index`, `invalid-index-json`, `stale-index`.

Sample entry points:
- `python agent/scripts/check_kernel_catalog.py --fail-on-warning` — fails fast when catalog entries, generated index JSON, recursive selectable-kernel discovery, or recursive UB row-scalar guardrails drift

### `agent/scripts/build_agent_index.py`
- purpose:
  - regenerate machine-readable agent indexes from the human-readable catalogs
  - keep `agent/index/kernels.json` and `agent/index/tools.json` aligned with the current `kernel-catalog.md` and `tool-catalog.md` without introducing a second manifest format
- use_for:
  - refreshing `agent/index/kernels.json` and `agent/index/tools.json` after any catalog edit
  - running as a CI or pre-commit step after catalog changes
- not_for:
  - editing catalog content
  - validating kernel numerical correctness
  - replacing `check_kernel_catalog.py` consistency checks
- important_outputs:
  - `agent/index/kernels.json` — machine-readable kernel entry list with `schema_version`, `generated_by`, `source`, `entry_count`, and `entries`
  - `agent/index/tools.json` — same shape for tool entries
  - `agent/references/examples/kernel-index.md` — generated compact Markdown view with the computed entry count
- pair_with:
  - `agent/references/examples/kernel-catalog.md`
  - `agent/references/examples/tool-catalog.md`
  - `agent/scripts/check_kernel_catalog.py`

Sample entry points:
- `python agent/scripts/build_agent_index.py` — regenerates both JSON indexes and the compact Markdown kernel index from the current catalogs
- `python agent/scripts/build_agent_index.py --check` — compares all generated views without writing files and fails when any view is stale

### `agent/scripts/check_agent_docs.py`
- purpose:
  - validate the router-first agent documentation as one governed system
  - enforce the initial common-language load contract and its 250-line / 15-KiB size budget
  - reject broken links, stale generated views, catalog drift, removed-role residue, concrete temporary evidence, machine-specific access details in tracked text or archive payloads, and archive owner/cache metadata that can disclose a workstation identity
- use_for:
  - the final documentation gate after changing `AGENTS.md`, `agent/`, catalogs, tools, or public documentation entry maps
  - checking that every workflow playbook appears exactly once in `agent/ROUTER.md`
  - checking Markdown links and anchors plus backticked repository paths
  - running catalog and generated-index freshness checks through one command
- not_for:
  - validating kernel numerical correctness
  - rewriting documents or generated indexes
  - replacing focused parser, simulator, or tool pytest coverage
- important_outputs:
  - a zero-exit `Agent documentation checks passed.` result when all checks succeed
  - warning codes and source locations for each routing, link, path, hygiene, catalog, or freshness failure
  - optional JSON output for machine consumption
- pair_with:
  - `agent/ROUTER.md`
  - `agent/scripts/build_agent_index.py`
  - `agent/scripts/check_kernel_catalog.py`

Sample entry points:
- `python agent/scripts/check_agent_docs.py` — runs the complete agent-documentation gate
- `python agent/scripts/check_agent_docs.py --json` — emits the same findings as JSON

### `agent/scripts/run_preferred_simulator.py`
- purpose:
  - launch repository simulator-side Python checks while preferring WSL's `torch210npu` environment on Windows hosts
  - keep the target script unchanged while centralizing backend selection and `PYTHONPATH` setup
- use_for:
  - running project-level simulator checks such as `agent/example/projects/a5/kda_bwd/check_simulator_vs_ref.py`
  - running standalone kernel simulator smokes from `agent/example/projects/*/kernels/*.py`
  - keeping one command line that works on Windows hosts, WSL, and Linux
- not_for:
  - changing kernel logic or simulator configuration inside the target script
  - replacing project-specific onboard / CANNSIM wrappers
  - launching arbitrary non-Python tools
- important_outputs:
  - selected backend (`wsl` vs `local`)
  - fully rendered launch command
  - forwarded exit code and streamed target-script output
- pair_with:
  - project verification docs such as `agent/example/projects/a5/kda_bwd/README.md`
  - simulator entry scripts such as `agent/example/projects/a5/kda_bwd/check_simulator_vs_ref.py`

Sample entry points:
- `conda run -n torch210npu python agent/scripts/run_preferred_simulator.py agent/example/projects/a5/kda_bwd/check_simulator_vs_ref.py` — on Windows, offloads the simulator run into WSL `torch210npu` when available; elsewhere runs locally
- `conda run -n torch210npu python agent/scripts/run_preferred_simulator.py agent/example/projects/a5/kda_bwd/test_script/module_vs_ref.py --module leaf --upstream ref` — runs a per-module simulator check through the same backend selection
- `conda run -n torch210npu python agent/scripts/run_preferred_simulator.py --dry-run agent/example/projects/a5/kda_bwd/kernels/finalize_reduce.py` — prints the chosen backend and exact command without executing it

## Fast selection hint

If the question is "which tile/core split should I use before writing the kernel body?", start here:
- `agent/scripts/estimate_matmul_datamove.py`

If the question is "which existing kernel should I study before opening source files?", start here:
- `agent/scripts/select_kernel_example.py`

If the question is "did the kernel catalog or generated kernel index drift out of sync?", start here:
- `agent/scripts/check_kernel_catalog.py`

If the question is "how do I regenerate the machine-readable agent indexes after editing a catalog?", start here:
- `agent/scripts/build_agent_index.py`

If the question is "are routing, links, catalogs, generated views, and documentation hygiene all current?", start here:
- `agent/scripts/check_agent_docs.py`

If the question is "run this simulator check from Windows, but prefer the faster WSL torch210npu path when available", start here:
- `agent/scripts/run_preferred_simulator.py`
