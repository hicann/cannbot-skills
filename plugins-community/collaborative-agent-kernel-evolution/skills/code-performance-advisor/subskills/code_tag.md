---
name: code-tag
description: Tag code and profiling inputs for SQL indexing and recommendations
---

# Subskill: code_tag (Tagging for Code + Profiling)

## What I do

Turn **code + profiling data + roofline goals** into structured tags for SQL indexing and recommendation lookup.

## Workflow

1) Read inputs (code + profiling + flowchart + roofline + op description).
2) Check missing inputs and ask whether to proceed.
3) Parse profiling: read CSV first; only load header guide if needed.
4) Generate tags from the taxonomy (Domain/Symptom/Context).
5) Save JSON tags + a trace note.

## Input Sources

**Inputs are optional; missing data must be reported and confirmed before continuing.**

- Code (optional)
  - Host: `workspace/inputs/{op}/code/op_host/`
  - Kernel: `workspace/inputs/{op}/code/op_kernel/`
- Profiling (optional but recommended)
  - CSV: `workspace/inputs/{op}/profiling/`
  - Flowchart notes: `workspace/inputs/{op}/flowchart/`
- Roofline goal (optional)
  - `workspace/inputs/{op}/roofline/goal.md`
- Op description (optional)
  - `workspace/inputs/{op}/op_description/op_description.md`

### Missing check template

```
Missing:
- profiling_csv
- roofline_goal

Continue with partial inputs? (yes/no)
```

## Profiling CSV Rules

Read the CSV header directly. Only load the header guide when a field is unclear:

- `references/standards/op_summary_header_guide.md`

Avoid unnecessary document loading to save tokens.

## Tagging Rules

Tag taxonomy source:

- `references/standards/tag_taxonony.md`

### Domain Tags

- `U.*`: derived from Task Type (AI_CORE / AI_VECTOR_CORE / MIX / CPU / DMA).
- `O.*`: derived from op description + code structure (multi-tag allowed when evidence is clear).
- `T.*`: derived from input/output data types.

**O.* Inference Rules (avoid missing co-tags):**

| Primary tag | Condition | Add also |
|-------------|-----------|----------|
| `O.Norm` | kernel uses `ReduceSum` or `ReduceMean` API | `O.Reduce` |
| `O.Norm` | operator computes per-channel/per-token statistics | `O.Reduce` |
| `O.Fused` | fused op contains a Norm sub-computation | `O.Norm` |
| `O.Activation` | activation is computed inside a fused Norm (e.g., GeluNorm) | `O.Norm` |

**Rationale**: Norm operators inherently perform reduction (sum/mean over elements per channel/token). Omitting `O.Reduce` causes rules like `R_NORM_WELFORD_ALGORITHM` to score lower than intended because they expect both tags.

### Symptom Tags

- derived from profiling values + flowchart interpretation.
- only tag when evidence exists (no guessing).

### Context Tags

- derived from shapes/layout/arch and code constraints (alignment/UB).
- context is for ranking and explanation, not hard triggering.

## Output Format

### JSON tag output (machine-readable)

**Save to:**

```
workspace/cache/tags/tag_<op>_<timestamp>.json
```

**JSON template:**

```json
{
  "version": "1.0",
  "op": {"name": "", "type": "", "task_type": ""},
  "inputs_used": {"code_host": "", "code_kernel": "", "profiling_csv": ""},
  "missing_inputs": [],
  "domain_tags": [],
  "symptom_tags": [],
  "context_tags": [],
  "evidence": {"<tag>": "<reason>"},
  "profiling_summary": {},
  "roofline_goal": {},
  "notes": []
}
```

### Trace output (human-readable)

**Save to:**

```
workspace/cache/tags/tag_trace_<op>_<timestamp>.md
```

**Trace must include:**
- input file list
- tag list
- evidence summary
- WaitDominated status

## Example (mhc_post)

- JSON: `workspace/cache/tags/tag_mhc_post_20260120012224.json`
- Trace: `workspace/cache/tags/tag_trace_mhc_post_20260120012224.md`

## Quality Principles

- Evidence-first: every tag must be traceable to CSV/flowchart/code.
- Keep it small: avoid over-tagging.
- Maintainable: add new tags by updating taxonomy only.

## Tag Validation (Automatic)

**IMPORTANT**: After generating tags, they are **automatically validated** against `tag_taxonony.md` via a hook.

**Validation Process**:
1. Tag JSON is generated in `workspace/cache/tags/`
2. Hook `hooks/post_tag_generation.py` is automatically triggered
3. Tags are checked against canonical taxonomy
4. If invalid tags found:
   - ❌ Validation fails with error message
   - 💡 Suggestions provided for typos (fuzzy matching)
   - 📝 User must fix tags before proceeding

**Example validation output** (if tags are invalid):
```
[HOOK] ❌ Validation failed - 2 invalid tag(s) found:
  - S.HighScalarRatio
    💡 Did you mean: S.ScalarBound?
  - U.InvalidUnit
    💡 Did you mean: U.Vector, U.Cube?

[HOOK] Please fix tags in: workspace/cache/tags/tag_xxx.json
[HOOK] Reference: references/standards/tag_taxonony.md
```

**Manual validation**:
```bash
# Validate all rules
python3 scripts/analysis_engine/tag_validator.py

# Validate specific tag file
python3 scripts/analysis_engine/tag_validator.py tag_pattern
```

**Adding new valid tags**:
If you need a tag that doesn't exist in the taxonomy:
1. Edit `references/standards/tag_taxonony.md`
2. Add your tag following the format: `` `X.YourTag`: description ``
3. Re-run validation

**Why this matters**:
- Prevents typos that would break rule matching
- Ensures tags are consistent across all rules
- Provides immediate feedback with correction suggestions
- Maintains taxonomy integrity as single source of truth
