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
- important outputs:
  - per-core datamove estimate
  - best strategy candidate set
  - PrettyTable strategy report
  - expansion ratio and DBUF-aware capacity decisions
- pair_with:
  - `agent/references/constraints/tiling.md`
  - `agent/scripts/tools_summary.md`
  - large tiled examples such as `agent/example/kernels/a5/matmul_mknk_2dgrid_splitn.py`, `agent/example/kernels/a5/matmul_mknk_2dgrid_splitk.py`, and `agent/example/kernels/a5/matmul_mknk_2dgrid_splitk_add1.py`

### `agent/scripts/select_kernel_example.py`
- purpose:
  - rank existing kernel examples for a new task using the generated kernel index
  - reduce manual catalog and source-file scanning before kernel authoring
- use_for:
  - selecting a first kernel to study by topology, query text, tags, or lightweight features
  - narrowing candidate examples before opening `agent/example/kernels/` source files
  - surfacing `study_for` and `do_not_copy_when` guidance together with the example path
- not_for:
  - generating kernel logic
  - proving that one example is the uniquely correct template
  - replacing constraint/reference reading for tiling, autosync, counters, or precision
- important outputs:
  - ranked kernel paths
  - short match reasons
  - `study_for` and `do_not_copy_when` summaries
  - optional JSON output for tool chaining
- pair_with:
  - `agent/references/examples/kernel-catalog.md`
  - `agent/index/kernels.json`
  - `agent/playbooks/kernel-authoring.md`
  - `agent/references/constraints/`

### `agent/scripts/check_counter_lifetimes.py`
- purpose:
  - flag suspicious counter sharing and stage-lifetime mixing with static AST checks
  - give a fast pre-review pass before deeper manual reasoning or simulator work
- use_for:
  - spotting counters incremented at multiple loop-owned stages
  - spotting suspicious delayed-stage sharing across sibling conditional branches
  - summarizing which buffers and positions each counter indexes
  - adding a lightweight CI gate with `--fail-on-warning`
- not_for:
  - proving full kernel correctness
  - replacing autosync reasoning or simulator validation
  - deciding automatically whether every multi-buffer counter is semantically wrong
- important_outputs:
  - warning codes with line numbers
  - per-counter buffer and position summary
  - optional JSON output for tool chaining or CI
- pair_with:
  - `agent/references/constraints/counters.md`
  - `agent/example/kernels/a5/matmul_mknk_2dgrid_splitn.py`
  - `agent/example/kernels/a5/matmul_mknk_2dgrid_splitk_add1.py`
  - `agent/example/kernels/a5/test_mla_entire.py`

### `agent/scripts/check_kernel_catalog.py`
- purpose:
  - verify that the human-readable kernel catalog, generated kernel index, and actual `agent/example/kernels/*.py` files stay consistent
  - catch source-of-truth drift before selectors or routing layers start returning stale metadata
- use_for:
  - checking that every catalog entry points to a real kernel file
  - checking that required metadata fields are present and non-empty
  - checking that newly added kernel files are not missing from the catalog
  - checking whether `agent/index/kernels.json` is stale relative to the current catalog
- not_for:
  - rewriting catalog entries automatically
  - ranking or selecting kernels for study
  - validating kernel numerical correctness
- important_outputs:
  - consistency warning codes such as missing file, missing field, uncataloged kernel, or stale index
  - optional JSON output for CI or tool chaining
- pair_with:
  - `agent/references/examples/kernel-catalog.md`
  - `agent/index/kernels.json`
  - `agent/scripts/build_agent_index.py`
  - `agent/scripts/select_kernel_example.py`

### `agent/scripts/gen_kernel_skeleton.py`
- purpose:
  - generate repository-style kernel scaffolds so new kernel work can start from a controlled skeleton instead of peeling an old file by hand
  - encode the current pure-cube, cube->vec, vec->cube, vec->cube->vec, cube->vec->cube->vec, and vec->cube->vec->cube template structure directly into a reusable CLI tool
- use_for:
  - creating a new `cube-only`, `cube->vec`, `vec->cube`, `vec->cube->vec`, `cube->vec->cube->vec`, or `vec->cube->vec->cube` kernel scaffold under `agent/example/kernels/`
  - previewing grid-aware cube skeletons with `tile-m`, `tile-n`, or `mix`
  - choosing whether the scaffold should include a tiled `K` loop or a one-shot cube call
  - selecting a topology-specific profile such as `cube-only --profile splitk`, `cube-only --profile kmkn`, `cube->vec --profile dual-output`, `cube->vec --profile normalize-two-pass`, `cube->vec --profile half-row-post`, `vec->cube --profile nz-publish`, `vec->cube --profile half-row-pre`, `cube->vec->cube->vec --profile lookahead-basic`, or `vec->cube->vec->cube --profile lookahead-basic`
  - getting explicit counter ownership and tiled loop structure before filling real matmul logic
- not_for:
  - generating a semantically complete kernel body
  - deciding exact tile/core split values automatically
  - replacing the need to choose the correct topology first
- important_outputs:
  - generated kernel file or printed scaffold text
  - tiled pure-cube, cube->vec, vec->cube, vec->cube->vec, cube->vec->cube->vec, or vec->cube->vec->cube loop skeletons
  - `CvMutex` handoff skeleton and vec postprocess placeholder for `cube->vec`, including `simple-post`, explicit split-`M` `half-row-post`, `normalize-simple`, `normalize-two-pass`, and `dual-output` variants
  - `VcMutex` publish skeleton and vec preprocess placeholder for `vec->cube`, including `nd-publish`, `nz-publish`, and explicit split-`M` `half-row-pre` variants
  - overlapped stage1/stage2 delayed-drain skeleton with `VcMutex` + `CvMutex` for `vec->cube->vec`, including explicit `overlap-basic`, `half-row-post`, and deeper-queue `delayed-post` variants
  - lookahead-style `cube->vec->cube->vec` streaming skeleton with `CvMutex` + `VcMutex` + `CvMutex`, separate delayed-stage counters, and a loop-external finalize stage
  - mirrored `vec->cube->vec->cube` streaming skeleton with `VcMutex` + `CvMutex` + `VcMutex`, streamed vec preprocess, delayed second-stage cube consume, and direct final cube writeback
  - `__main__` validation stub unless disabled
- pair_with:
  - `agent/playbooks/kernel-authoring.md`
  - `agent/references/patterns/cube-only.md`
  - `agent/scripts/select_kernel_example.py`
  - `agent/scripts/estimate_matmul_datamove.py`

## Fast selection hint

If the question is "which tile/core split should I use before writing the kernel body?", start here:
- `agent/scripts/estimate_matmul_datamove.py`

If the question is "which existing kernel should I study before opening source files?", start here:
- `agent/scripts/select_kernel_example.py`

If the question is "does this kernel have obvious counter-lifetime smells before I go deeper?", start here:
- `agent/scripts/check_counter_lifetimes.py`

If the question is "did the kernel catalog or generated kernel index drift out of sync?", start here:
- `agent/scripts/check_kernel_catalog.py`

If the question is "give me a new pure-cube or mixed-pipeline kernel scaffold that follows the repository's tiled template", start here:
- `agent/scripts/gen_kernel_skeleton.py`
