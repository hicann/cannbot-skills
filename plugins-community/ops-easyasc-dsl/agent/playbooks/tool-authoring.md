# Tool Authoring Playbook

Use this playbook when adding or modifying a repository tool under `agent/scripts/`.
Keep tools small, deterministic, and easy to validate.

## Goal

Build a tool that:
- solves one recurring problem cleanly
- has a narrow interface
- is easy to test from the command line
- is summarized in `agent/scripts/tools_summary.md`

## 1. Confirm the tool belongs in `agent/scripts/`

A repository tool is a good fit when it:
- performs repeated estimation or analysis
- provides deterministic helper logic
- is useful across multiple kernel tasks
- should be runnable without loading a huge amount of prose context

If the work is only explanatory, prefer documentation instead of a new tool.

Before modifying a tool, check:
- `agent/references/examples/tool-catalog.md`

## 2. Define the exact job and interface

Before coding, write down:
- what the tool computes or checks
- required inputs
- optional inputs
- output format
- whether it is analysis-only or mutates files

Prefer a narrow API over a giant kitchen-sink script.

## 3. Keep the implementation deterministic

Repository preference:
- tools should be small and direct
- avoid hidden environment requirements unless truly needed
- avoid unnecessary side effects

If a script must make a repository decision repeatedly, encode that decision clearly instead of forcing the agent to rediscover it in prose.

## 4. Keep command-line usage obvious

A good tool should be runnable quickly from the repository root.
If it has multiple modes, keep the mode names explicit.
If output is tabular or numeric, make the result readable enough for direct inspection.

## 5. Validate with a representative sample

Actually run the tool.
Do not assume the script is correct because it is short.
If there are several modes, test a representative subset.

For estimators or analyzers, validate:
- a normal case
- an edge or failure case
- at least one case that exercises an option or mode switch

## 6. Update the tool summary

When adding or editing files under `agent/scripts/`, update:
- `agent/scripts/tools_summary.md`

Update the summary itself.
Do not turn it into a change log.

## 7. Refresh owner docs when done

After the tool work is complete and there are no scripts left to run, refresh the owner docs that actually describe the changed area:
- `agent/scripts/tools_summary.md`
- `agent/references/code-paths.md` if the implementation-path lookup changed
- `agent/references/repo-map.md` or `doc/11_architecture_for_contributors.md` if the repository structure or ownership changed

## 8. Keep Python style compact

Repository formatting preference:
- keep signatures and call sites compact
- do not force one argument per line unless readability really needs it
- keep the script easy to scan

## Fallback references

Read these if more detail is needed:
- `agent/scripts/tools_summary.md`
- `agent/references/code-paths.md`
- `doc/11_architecture_for_contributors.md`
- `README.md`
