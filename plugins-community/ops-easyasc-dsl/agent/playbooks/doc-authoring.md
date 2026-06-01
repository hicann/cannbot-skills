# Documentation Authoring Playbook

Use this playbook when adding or revising repository documentation.
Write for future agents and humans who need the shortest path to the right detail level.

## Goal

Produce documentation that:
- explains one layer of the system clearly
- avoids mixing overview, workflow, and low-level constraints into one blob
- matches the current repository behavior
- points readers to the next focused file instead of dumping everything at once

## 1. Choose the right documentation layer

Before editing, decide which layer the content belongs to:
- overview or navigation
- workflow playbook
- detailed constraint or reference
- example catalog
- API or framework documentation

Do not stuff every fact into a giant summary file.

## 2. Prefer progressive disclosure

Default read order should be:
1. router or overview
2. one playbook
3. one focused reference
4. one concrete example

If a file tries to do all four jobs at once, split it.

## 3. Keep purpose and audience obvious

At the top of a doc, make it obvious:
- what this file is for
- when to read it
- what it does not try to cover
- where to go next for deeper detail

This matters more than long introductions.

## 4. Separate workflow from facts

Keep these concerns separate:
- workflow: what to do step by step
- constraints: rules that must hold
- examples: which source files to study
- architecture: where logic lives in the repository

If readers need all of them at once, link them together instead of merging them into one monster file.

## 5. Keep examples concrete

Prefer:
- one direct example
- one clear counterexample
- one pointer to the exact source file

Avoid vague advice that cannot be traced to repository code.

## 6. Preserve repository truth

When behavior is subtle, derive documentation from the implementation path, not from memory.
Typical sources:
- `easyasc/`
- `agent/example/kernels/`
- `testcases/`
- `doc/`
- `agent/scripts/`

If you are not sure, inspect the real code before documenting the rule.

## 7. Keep English documentation in English

Repository rule:
- codes, error messages, and readme files should be written in English

If Chinese documentation also needs updating, keep the English source coherent first, then mirror as needed.

## 8. Update owner docs carefully

If a repository change affects the documentation map or high-level structure, refresh the owner docs that describe it:
- `README.md`, `README_CN.md` when the top-level documentation entry map changed
- `agent/references/repo-map.md` when top-level layout or ownership changed
- `doc/11_architecture_for_contributors.md` when contributor-facing architecture or subsystem ownership changed
- `testcases/README.md` when the test-suite layout or fixture/demo boundary changed

If kernel files are added, update:
- `agent/references/examples/kernel-catalog.md`

If tool files are added or changed, update:
- `agent/scripts/tools_summary.md`
- `agent/references/code-paths.md` if the implementation-path lookup changed

## 9. Prefer replacement-ready migration

When restructuring documentation, do not delete older guidance until the new layer is actually usable.
A migration is only done when the new route works on its own.

## Fallback references

Read these if more detail is needed:
- `README.md`
- `agent/references/repo-map.md`
- `doc/11_architecture_for_contributors.md`
- `doc/`
- `doc_cn/`
