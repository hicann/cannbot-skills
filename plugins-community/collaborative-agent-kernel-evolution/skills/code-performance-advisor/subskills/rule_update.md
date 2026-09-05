---
name: rule-update
description: Generate or update performance optimization rules from user inputs
---

# Subskill: rule_update (Rule Update and Completion)

## What I do

Based on user-provided operator descriptions, code, or performance analysis results, **automatically supplement and optimize** rules in the `assets/rules/` directory. This subskill follows the **"Rigorous Priority, No Completion without Basis"** principle—only supplementing content with clear evidence and prompting the user for confirmation on missing or uncertain information.

Additionally, this skill is responsible for calling the CLI to **maintain the rule index** (`assets/manifests/index.json`) after a rule is added or updated, ensuring that subsequent `rules_search` can directly execute scoring based on the latest index.

## Scope Boundary (Relationship with rules_search)

- `rule_update` is responsible for: Rule content generation/update, tag sidecar generation, and index build/update.
- `rules_search` is responsible for: Rule scoring and ranking based on tags (`score` flow).
- `rule_update` **does not re-implement or re-calculate** scoring logic; when scoring is needed, it hands off to `rules_search` or calls `cli.py score`.

## Knowledge Routing: Local Rules

**Local rules** (`assets/rules/`) are curated expert knowledge — stable, validated, and used by the scoring engine. **Do NOT add new rules locally** unless they are generic, hardware-architecture-level patterns that apply across many operators.

For new discoveries during optimization:

```
New optimization found after APPLY→EVALUATE
  │
  ├─ Matches an existing local rule? → Update that rule's supplementary info (code_snippets, constraints)
  │
  └─ Novel pattern (not covered by any local rule)? → Log to session notes only, skip
```

## Workflow (Rigorous-First Incremental Completion)

### Phase 1: Input Analysis and Gap Analysis

1. **Read Input**
   - Check the input types provided by the user:
     - Operator description files (`op_description.md` or similar)
     - Code files (AscendC kernel/host code)
     - Profiling data (CSV or analysis results)
     - Free-text descriptions

2. **Gap Analysis**
   - Compare against `rule_template.md` to identify which parts have information and which are missing.
   - Generate a **supplementary checklist**, marking the completeness of each category:
     - ✅ Explicit information available
     - ⚠️ Partial information, needs confirmation
     - ❌ Missing information, requires user supplement

### Phase 2: Incremental Generation (On-Demand)

**Core Principle**: Generate content for each section only when there is sufficient evidence.

#### 2.1 Title (Rule Name)
- Infer from input or ask the user.
- Format: `R_{OperatorName}_{OptimizationTheme}`

#### 2.2 Requirement
- **Auto-Extractable**:
  - Business logic: Extract from operator description or code comments.
  - Shape/Data type context: Extract from `Input Shapes` / `Output Data Types`.
- **Requires User Confirmation**:
  - Qualitative description of performance bottlenecks (e.g., transfer-dominated vs. compute-dominated).

#### 2.3 Pattern
- **Auto-Extractable**:
  - Summarize optimization ideas from code comparison (`base_code` vs. `good_code`).
- **Requires User Confirmation**:
  - Physical goal of the optimization (e.g., "reducing transfer volume" needs profiling data support).

#### 2.4 Inference / Physics
- **Requires Expert Review**:
  - Any content involving formulas or causal inferences.
  - Generate a draft by default but mark it with `[Review Required]`.

#### 2.5 Triggers
- **Auto-Extractable** (Based on `op_summary` headers):
  - Extract relevant fields from profiling data.
  - Default includes: `Task Duration(us)`, `Block Dim`, `cube_utilization(%)`.
  - Add based on bottleneck type:
    - Transfer bottleneck: `aic_mte2_ratio`, `aiv_mte2_ratio`
    - Compute bottleneck: `aic_mac_ratio`, `aiv_vec_ratio`
    - Pipeline bottleneck: `aic_scalar_ratio`, `aic_icache_miss_rate`
- **Interpretation**: Provide qualitative interpretation directions, avoiding specific thresholds.

#### 2.6 Action
- **Auto-Extractable**:
  - Code snippet paths (if `code_snippets/` examples exist)
  - Summarize key implementation steps from `good_code`
- **Requires User Provision**:
  - If no code examples exist, require user supplement or mark as `[To be supplemented]`

#### 2.7 Constraints
- **Inferable**:
  - Memory/UB usage: Infer from workspace allocation in code
  - Preliminary tags for applicable scenarios
- **Requires User Confirmation**:
  - Non-applicable scenarios (usually rely on experience, avoid auto-inference)

#### 2.8 Verification
- **Auto-Template Generation**:
  - Based on trigger signal fields, generate verification items for before/after comparison
  - Use qualitative descriptions (Improvement/Increase/Decrease) instead of numeric thresholds
- **Requires User Provision**:
  - Specific verification methods or test cases

#### 2.9 Tags
- **Auto-Extractable**:
  - Domain tags (`U.*`, `O.*`): Infer from operator type/execution unit
  - Symptom tags (`S.*`): Infer from bottleneck characteristics in profiling data
  - Context tags (`C.*`): Infer from shape/architecture/data type
- **Reference Standard**:
  - `references/standards/tag_taxonony.md`

### Phase 3: Tag File Generation (JSON)

Generate `{rule_id}_tags.json` in the following format:

```json
{
  "rule_id": "RXXX_{OP}_{PATTERN}",
  "domain_tags": ["U.Cube", "O.MatMul"],
  "symptom_tags": ["S.TransferDominated", "S.MemoryBound"],
  "context_tags": ["C.K.Large", "C.MN.Small", "C.Arch.910B"],
  "required_tags": ["U.Cube", "O.MatMul", "S.TransferDominated"],
  "tags": ["U.Cube", "O.MatMul", "S.TransferDominated", "S.MemoryBound", ...]
}
```

### Phase 3.1: Tag Validation (Compliance Check)

To maintain the rigor of the rule library, all tags used in `{rule_id}_tags.json` **MUST** exist in [tag_taxonony.md](../references/standards/tag_taxonony.md).

1. Execute the validation script:
```bash
python scripts/analysis_engine/tag_validator.py <rule_id>
```
2. If validation fails:
   - Check for typos in the tags.
   - If the tag is necessary but genuinely new, update [tag_taxonony.md](../references/standards/tag_taxonony.md) first, then re-run validation.

### Phase 3.5: Synchronized Rule Index Update (CLI, Mandatory)

After the rule document and `{rule_id}_tags.json` are written, call the CLI to maintain the index:

1. If `assets/manifests/index.json` does not exist:

```bash
python scripts/analysis_engine/cli.py build-index --rule <path/to/R_xxx.md>
```

2. If index already exists (adding or modifying a rule):

```bash
python scripts/analysis_engine/cli.py update-index --rule <path/to/R_xxx.md>
```

Note:
- `is_general` is automatically inferred from the rule path (`general_rules` vs. `special_rules`), no manual specification needed.
- This skill only maintains the index and does not execute `score` internally.

### Phase 4: User Confirmation and Gap Prompting

**For each missing or uncertain piece of content, generate a confirmation prompt**:

```
Please confirm the following (Optional, leave blank to skip):

1. [Verification] Are there specific verification test cases?
2. [Constraints] In what situations will this optimization fail?
3. [Inference] Does the formula/inference need correction?
```

## Inputs

| Input Type | Path/Format | Required | Description |
|------------|-------------|----------|-------------|
| Operator Description | `workspace/inputs/op_description/*.md` or user input | Recommended | Prefer `op_description_template.md` format |
| Code Snippets | `workspace/inputs/code/*` or user input | Recommended | Kernel/host code |
| Profiling Data | `workspace/inputs/profiling_data/profiling_csv/*.csv` | Recommended | Used to extract trigger signals |
| Existing Rules | `assets/rules/**/R_*.md` | Optional | Used for updates or reference |

## Output Format

**Primary Rule File**: `assets/rules/{special_rules,common_rules}/R_{ID}/R_{ID}.md`

**Tag File**: `assets/rules/{special_rules,common_rules}/R_{ID}/R_{ID}_tags.json`

**Code Snippet Directory**: `assets/rules/{special_rules,common_rules}/R_{ID}/code_snippets/case_{x}/`

**Index File (Updated via CLI)**: `assets/manifests/index.json`

### Rule Directory Structure Template

```
assets/rules/special_rules/R_XXX/
├── R_XXX.md                    # Main rule document
├── R_XXX_tags.json             # Tag definitions
└── code_snippets/              # Code examples (optional)
    └── case_0/
        ├── base_code/          # Negative/Original implementation
        │   └── base_code.md
        └── good_code/          # Positive/Optimized implementation
            └── good_code.md
```

## Gap Checklist (Must verify before generation)

Before generating the rule, check for the following gaps and prompt the user:

| Check Item | Description | Default Behavior |
|------------|-------------|------------------|
| `rule_id` | Whether there is a clear ID | Auto-generated, can be overridden |
| Domain Tags | Whether the execution unit/operator type can be determined | Auto-inferred, prompt if uncertain |
| Symptom Tags | Whether the bottleneck type can be determined | Inferred from profiling, prompt if no data |
| Triggers | Whether there are corresponding `op_summary` fields | List relevant fields, leave blank if no data |
| Code Snippets | Whether base_code/good_code exist | Mark as `[To be supplemented]` if missing |
| Tag Validation | Whether tags match the taxonomy | MUST PASS via `tag_validator.py` |
| Verification | Whether there are verification methods | Generate qualitative template, mark if no test cases |

## Interactive Supplementary Flow

When information is insufficient, prompt the user based on priority:

1. **High Priority (Blocks generation)**:
   - Rule ID conflicts or naming issues
   - Unable to determine Domain tags (affects rule matching)

2. **Medium Priority (Generate but mark)**:
   - Symptom tags uncertain → Generate draft and mark `[To be confirmed]`
   - Code snippets missing → Generate framework and mark `[To be supplemented]`

3. **Low Priority (Optional supplement)**:
   - Verification test cases → Can be added later
   - Constraints/Side effects → Can be added later

## Example: Generating Rules from Operator Description

**User Input**:
```
I have a MatMul operator with shape (32, 4096) x (4096, 32), aic_mte2_ratio is very high, 
performance improves after Cut-K. Please help me generate a rule.
```

**Workflow**:
1. Identify Domain: `U.Cube`, `O.MatMul`
2. Identify Symptom: `S.TransferDominated`, `S.MemoryBound`
3. Identify Context: `C.MN.Small`, `C.K.Large`
4. Reference `R_MATMUL_CUTK.md` template
5. Generate rule framework, confirm tags
6. Ask: Are there code snippets? Is there specific verification data?

## Reference

- `assets/templates/rule_template.md` — Rule template
- `assets/templates/op_description_template.md` — Operator description template
- `references/standards/tag_taxonony.md` — Tag taxonomy
- `references/standards/op_summary_header_guide.md` — Profiling field guide
- `assets/rules/special_rules/MATMUL_CUTK/` — Complete rule example
- `subskills/rules_search.md` — Scoring and ranking responsibilities
- `scripts/analysis_engine/cli.py` — Unified entry for build-index / update-index / score
