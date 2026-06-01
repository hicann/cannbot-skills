# Repository Map

Use this file when the question is about where something lives in the repository.
Workflow routing lives in `agent/ROUTER.md`; this file is not a second task router.

## Top-level areas

- `skill/`
  - skill entrypoint (`skill/SKILL.md`); the only file the skill surface is supposed to expose directly
- `agent/`
  - the reusable easyasc DSL to AscendC workflow used by the skill
  - router-first guidance, focused references, playbooks, scripts, archived payloads, and examples
- `README.md`, `README_CN.md`
  - top-level project and documentation entry points
- `AGENTS.md`, `CLAUDE.md`
  - repository-local working rules for agent-style contributors

Restored on demand by `agent/scripts/init.sh`:

- `easyasc/`
  - core DSL, parser/codegen, simulator, runtime models, and public API surface
- `doc/`
  - English project and API documentation
- `doc_cn/`
  - Chinese mirror of the documentation set
- `agent/example/kernels/`
  - curated kernel examples, reference kernels, and runnable validation stories
- `agent/example/demo/`
  - manual runners and compile/integration demos grouped by device family

Removed from the delivered skill bundle:

- `testcases/`
  - no longer part of the delivered tree; references to it elsewhere describe historical layout only

## `agent/` layout

- `agent/ROUTER.md`
  - first task router
- `agent/scripts/`
  - repository-maintenance scripts, including `init.sh` that restores archived trees
- `agent/assets/`
  - archived payloads (`ops-easyasc-dsl-runtime.tar.gz`, `ops-easyasc-dsl-example.tar.gz`)
- `agent/playbooks/`
  - short workflow guides for common tasks
- `agent/references/contract-intake.md`
  - intake rules when `clarify-first.md` needs to extract a contract from a reference file
- `agent/references/facts.md`
  - quick chooser for factual lookups
- `agent/references/facts-*.md`
  - focused fact sheets for device/runtime values, authoring rules, and simulator/`OpExec` gotchas
- `agent/references/constraints/`
  - topic-focused invariants such as tiling, autosync, counters, precision, tails, and device-specific limits
- `agent/references/patterns/`
  - topology-specific pipeline patterns, especially the a2 mixed-pipeline routes
- `agent/references/examples/`
  - human-readable kernel and tool catalogs, plus optional deep notes for a few complex kernels
- `agent/references/code-paths.md`
  - implementation-path lookup guide when the issue is in lowering or runtime behavior
- `agent/references/repo-map.md`
  - this file
- `agent/references/simulator-v2.md`
  - focused simulator runtime and bridge map
- `agent/index/`
  - generated machine-readable catalogs derived from the example markdown files
- `agent/example/`
  - restored kernel examples (`agent/example/kernels/`) and manual demos (`agent/example/demo/`)

## Owner files

Use the smallest owner file that answers the question:

- repository layout and area ownership -> `agent/references/repo-map.md`
- implementation-path lookup -> `agent/references/code-paths.md`
- contributor-facing architecture fallback -> `doc/11_architecture_for_contributors.md`
- kernel example metadata -> `agent/references/examples/kernel-index.md` and `agent/references/examples/kernel-catalog.md`
- tool summaries -> `agent/scripts/tools_summary.md`

## Reading rule

- use `agent/ROUTER.md` for workflow questions
- use this file for layout and ownership questions
- if archived content (`easyasc/`, `doc/`, `doc_cn/`, `agent/example/`) is missing, run `bash agent/scripts/init.sh`
- open source files only after the map is no longer enough
