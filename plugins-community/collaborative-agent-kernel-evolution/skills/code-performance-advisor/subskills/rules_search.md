---
name: rules-search
description: Score and rank rules from tags for manual review
---

# Subskill: rules_search (Rule Scoring and Ranking)

## What I do

Run the **programmatic** rule-scoring flow based on tags produced by `code_tag`, and write a ranked JSON for manual review.

## Workflow (Program-First)

This flow is implemented in code; the model should **not** re-implement it.

1) Read the latest tag file (or a user-provided tag file).
2) Load rule index from `assets/manifests/index.json`.
3) For each rule, load tags (prefer sidecar `Rxxx_tags.json`).
4) Compute weighted Jaccard score and coverage ratio.
5) Sort results and write `scored_results.json` for review.

## Inputs

- Tag file (required)
  - Default: `workspace/cache/tags/*.json` (latest)
- Rule index (required)
  - `assets/manifests/index.json`

### Missing check template

If required inputs are missing, the CLI will stop and ask for a valid path.



## Output Format

**Save to:**

```
assets/manifests/scored_results.json
```

**Fields per rule:**

- `rule_path`
- `is_general`
- `score` / `raw_score`
- `coverage_ratio`
- `conflict`
- `matched_tags` / `missing_tags`
- `check` (default false)
- `valid` (default false)

## Tool Entry (CLI)

```
python scripts/analysis_engine/cli.py score
```

Optional:

```
python scripts/analysis_engine/cli.py score --tag-file <path/to/tag.json>
```

## Notes

- `is_general` is inferred from folder name: `general_rules` vs `special_rules`.
- Tags are loaded from `Rxxx_tags.json` when available; otherwise from rule front-matter.
- If the CLI exists, **use it**; do not ask the model to rescore rules.
