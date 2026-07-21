# Hooks (Claude Code only)

> ⚠️ **Platform**: these hooks target **Claude Code**. They use Claude Code's
> hook schema (`hooks.json` with `Stop` / `PostToolUse` / `PostToolUseFailure`
> events and the `${CLAUDE_PLUGIN_ROOT}` variable). They will **not** run under
> OpenCode or other harnesses — those use a different hook mechanism.
>
> Running CAKE2 without these hooks is fully supported: the agents still
> generate and evaluate operators correctly. The hooks only add convenience
> automation (stop-gating + debug-skill suggestions); their absence degrades
> gracefully (nothing blocks, no auto-suggestions).

## What each hook does

| Event | Script | Purpose |
|-------|--------|---------|
| `Stop` | `check_eval_done.sh` | Blocks the agent from stopping until evaluation has passed. No-op unless `CAKE_OUTPUT_DIR` is set by an external batch runner (manual runs always approve). |
| `PostToolUse` (Bash) | `debug_skill_advisor.py --event PostToolUse` | On compile/runtime/precision failures detected in Bash output, suggests the relevant debug skills. |
| `PostToolUseFailure` (Bash) | `debug_skill_advisor.py --event PostToolUseFailure` | Same advisor, on tool-call failures. |

## Other scripts here (not wired into hooks.json)

These are repo-quality linters, invoked manually or via the optional
`.pre-commit-config.yaml` (a local pre-commit hook, also Claude-Code-independent):

- `skills_linter.py` — validate SKILL.md frontmatter / token budget.
- `agents_linter.py` — validate agent definition files.
- `_yaml_utils.py` — shared YAML helpers for the linters.

## Porting to OpenCode

Not provided. An OpenCode port would need the equivalent events expressed in
OpenCode's hook format (cf. the cross-platform `run-hook.cmd` wrapper used by
other cannbot community plugins) and is out of scope for this plugin today.
