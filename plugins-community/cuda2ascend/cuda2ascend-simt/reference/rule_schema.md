# Rule Schema

`grammar_rules.yaml` and `constraint_rules.yaml` are the source of truth for syntax-rewrite rules and migration constraint rules.

The Markdown files remain human-readable browse views:

- `grammar.md`
- `constraints.md`

Update the YAML files first. Keep the Markdown views aligned afterward.

## `grammar_rules.yaml`

Use `grammar_rules.yaml` to record syntax-level rewrite rules for CUDA source code that must be adjusted for Ascend C SIMT.

### Record Structure

```yaml
- rule_id: assert-header
  pattern: assert
  category: debug
  status: required
  action: include_header
  cuda_form: assert(...)
  ascend_form: '#include "utils/debug/asc_assert.h"'
  note: Add the Ascend SIMT assert header when assert is used.
  source:
    - reference/grammar.md
  reviewed_by: human
  reviewed_at: "2026-04-06"
```

### Field Definitions

- `rule_id`
  - Required.
  - Unique identifier for the rule.
  - Use lowercase letters, digits, and hyphens.

- `pattern`
  - Required.
  - The CUDA syntax pattern or construct this rule applies to.

- `category`
  - Required.
  - Suggested values:
    - `debug`
    - `memory-space`
    - `header`
    - `syntax`

- `status`
  - Required.
  - One of:
    - `required`: a fixed rule that must be applied
    - `rewrite-required`: syntax must be transformed
    - `conditional`: only applies in specific situations
    - `unknown`: rule not fully confirmed yet

- `action`
  - Required.
  - One of:
    - `include_header`
    - `syntax_replace`
    - `check_docs`
    - `manual_review`

- `cuda_form`
  - Required, may be empty.
  - Short representation of the CUDA-side form.

- `ascend_form`
  - Required, may be empty.
  - Short representation of the Ascend-side form.

- `note`
  - Required, may be empty.
  - Brief explanation of the rule or its scope.

- `source`
  - Required.
  - List of references that justify the rule.

- `reviewed_by`
  - Required, may be empty.
  - `human`, `agent`, or a more specific reviewer label.

- `reviewed_at`
  - Required, may be empty.
  - ISO date string.

### Maintenance Rules

- Add a new rule when the syntax transformation is reusable across operators.
- Do not encode one-off operator-specific hacks as global syntax rules.
- If the rewrite rule is uncertain, keep `status: unknown` or `status: conditional`.
- Prefer short `cuda_form` and `ascend_form` examples over long code blocks.

## `constraint_rules.yaml`

Use `constraint_rules.yaml` to record unsupported or restricted features in the current migration model.

### Record Structure

```yaml
- rule_id: cooperative-groups
  feature: Cooperative Groups
  category: execution-model
  status: unsupported
  action: stop_and_report
  workaround: ""
  note: No workaround in the current migration model.
  source:
    - reference/constraints.md
  reviewed_by: human
  reviewed_at: "2026-04-06"
```

### Field Definitions

- `rule_id`
  - Required.
  - Unique identifier for the constraint rule.
  - Use lowercase letters, digits, and hyphens.

- `feature`
  - Required.
  - Human-readable name of the constrained feature.

- `category`
  - Required.
  - Suggested values:
    - `graphics`
    - `execution-model`
    - `memory-model`
    - `performance-model`
    - `compilation`
    - `data-type`
    - `build`

- `status`
  - Required.
  - One of:
    - `unsupported`: the feature has no direct migration support in the current model; use `action` to distinguish blocking features from removable unsupported subpaths
    - `restricted`: only usable with a workaround or limitation
    - `conditional`: support depends on context or extra confirmation
    - `unknown`: not fully investigated yet

- `action`
  - Required.
  - One of:
    - `stop_and_report`: the selected migration path cannot continue
    - `remove_and_record`: remove or exclude the unsupported source subpath, record the exclusion, and continue only if supported behavior remains
    - `manual_implementation`
    - `check_docs`
    - `manual_review`

- `workaround`
  - Required, may be empty.
  - The accepted workaround when one exists.

- `note`
  - Required, may be empty.
  - Brief explanation of the constraint.

- `source`
  - Required.
  - List of references that justify the rule.

- `reviewed_by`
  - Required, may be empty.
  - `human`, `agent`, or a more specific reviewer label.

- `reviewed_at`
  - Required, may be empty.
  - ISO date string.

### Maintenance Rules

- Use `unsupported` only when the feature itself has no direct migration support in the current migration model.
- Use `action: stop_and_report` for blocking unsupported features that prevent the selected migration path from continuing.
- Use `action: remove_and_record` for unsupported subpaths that may be removed or excluded while preserving supported source-visible behavior.
- Use `restricted` when a workaround exists but carries limits or extra migration work.
- If the rule still needs confirmation, use `conditional` or `unknown`; do not overstate support.
- Keep workarounds short in YAML and move long explanations to Markdown if needed.

## General Update Workflow

When a human or agent updates rule references:

1. Update the relevant YAML file first.
2. Keep `rule_id` stable once introduced.
3. Add or revise `source`, `note`, and `reviewed_at`.
4. Sync the Markdown browse view if the rule meaning changed.
