---
name: init-workspace
description: Initialize workspace/inputs from CAKE2 output artifacts, with overwrite support and latest-CSV selection
---

# Subskill: init_workspace

## What I do

Initialize workspace input files by:
1. **Scanning** CAKE2 `output/{op}/` for code and profiling artifacts
2. **Copying** code (`op_host/`, `op_kernel/`) and latest profiling CSV
3. **Selecting** newest `op_summary_*.csv` by datetime token in path (mtime fallback)

Target directory: `workspace/inputs/{operator_name}/`

This subskill enables **repeatable, one-command bootstrap** before `code_tag` and optimization phases.

---

## Key Behaviors

### Latest CSV Selection

Profiling output directories contain datetime tokens (e.g. `ModelNew_device0_20260227_075433_310761`).

`init_workspace.py` selects the **newest** `op_summary_*.csv` by:
1. Datetime token in path/filename (preferred)
2. File mtime fallback

Example: given both `Model_device0_20260227_075431_*` and `ModelNew_device0_20260227_075433_*`, the `075433` one is selected.

### Overwrite Behavior

By default, existing files are **skipped** (not overwritten). Use `--overwrite` to replace existing files.

### Cleanup

`init_workspace.py` does **not** auto-clean. To re-initialize after code or profiling changes, delete the existing directory first, or use `--overwrite`:
```bash
rm -rf workspace/inputs/{op}/
python3 scripts/analysis_engine/init_workspace.py --op {op}
# or simply:
python3 scripts/analysis_engine/init_workspace.py --op {op} --overwrite
```

Stale tag cache files must be cleaned manually if needed:
```
workspace/cache/tags/tag_{op}*.json
```

---

## Source and Target

- **Source root**: `output/{operator_name}/`
- **Target root**: `workspace/inputs/{operator_name}/`

Target layout per operator:

```text
workspace/inputs/{op}/
├── code/
│   ├── op_host/
│   └── op_kernel/
└── profiling/
    └── op_summary.csv
```

Copied content:
- `code/op_host/` from generated output (recursive)
- `code/op_kernel/` from generated output (recursive)
- latest profiling summary CSV: `op_summary_*.csv` -> `profiling/op_summary.csv` (selected by timestamp, newest wins)

Exclusion rule:
- Skip all files matching `CMakeLists*`

## Factual completion policy

When auto-generating `op_description.md` and `goal.md`:
- Must only include facts that can be inferred from current workspace data (e.g., operator directory name, discovered profiling file path).
- Unknown fields stay blank or TODO-style placeholders.
- Never fabricate formulas, thresholds, shape constraints, or kernel behavior.
- `op_description.md` is generated from:
	`assets/templates/op_description_template.md`

## Date rule for profiling

Profiling output contains date/time tokens in path names (e.g. `ModelNew_device0_20260212_090034`).

This skill selects the **latest** `op_summary_*.csv` by:
1. datetime token in path/name (preferred)
2. file mtime fallback

## Command

### Basic Usage

```bash
# Initialize workspace for specific operator
python3 scripts/analysis_engine/init_workspace.py --op fastgelu

# Initialize all operators in output/
python3 scripts/analysis_engine/init_workspace.py

# Overwrite existing files (use after re-profiling)
python3 scripts/analysis_engine/init_workspace.py --op fastgelu --overwrite

# Optional: Specify custom CAKE2 root
python3 scripts/analysis_engine/init_workspace.py --root /path/to/CAKE2
```

### Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--op` | No | All operators | Operator name (e.g., `fastgelu`) |
| `--root` | No | Auto-detect | CAKE2 project root path |
| `--dry-run` | No | False | Preview changes without copying files |
| `--overwrite` | No | False | Overwrite existing files (default: skip) |
| `--code-only` | No | False | Copy only code (skip profiling) |
| `--profiling-only` | No | False | Copy only profiling (skip code) |
| `--no-verify` | No | False | Skip copy verification |
| `--compute-hash` | No | False | Compute code hash for change detection |
| `--save-report` | No | - | Save report to JSON file |

### Output

```
ℹ️  Processing operator: fastgelu
ℹ️    Copying op_host: .../op_host -> workspace/inputs/fastgelu/code/op_host
ℹ️    Copying op_kernel: .../op_kernel -> workspace/inputs/fastgelu/code/op_kernel
ℹ️    Copying profiling: op_summary_20260212090038.csv -> workspace/inputs/fastgelu/profiling/op_summary.csv

============================================================
📊 INITIALIZATION SUMMARY
============================================================
✅ Mode: full

Operators: 1 total, 1 success, 0 failed
Files: 4 copied, 0 skipped, 0 failed

------------------------------------------------------------
📋 OPERATOR DETAILS
------------------------------------------------------------

✅ fastgelu
   Target: workspace/inputs/fastgelu/
   Files: 4 copied, 0 skipped, 0 failed

============================================================
```

---

## Integration with CAKE2 Main Workflow

### Automatic Invocation (Recommended)

**In CAKE2 main pipeline**, before calling `code-performance-advisor` skill:

```python
# CAKE2/scripts/optimize_operator.py (example)

def optimize_operator(op_name):
    """
    Optimize a single operator using code-performance-advisor.
    """
    # Step 1: Auto-initialize workspace (cleanup + fresh data)
    subprocess.run([
        "python3",
        "skills/code-performance-advisor/scripts/analysis_engine/init_workspace.py",
        "--op", op_name
    ], check=True)

    # Step 2: Invoke optimization skill
    subprocess.run([
        "claude", "code",  # Or your CLI invocation method
        "-s", "code-performance-advisor",
        "--", op_name
    ], check=True)
```

### Manual Workflow

```bash
# 1. Generate operator code + profiling (CAKE2 workflow)
cd CAKE2
python scripts/generate_operator.py --op fastgelu
python scripts/profile_operator.py --op fastgelu

# 2. Initialize workspace (auto-cleanup)
cd skills/code-performance-advisor
python3 scripts/analysis_engine/init_workspace.py --op fastgelu

# 3. Start optimization
# (Invoke code-performance-advisor skill via Claude Code or automation)
```

---

## Success criteria

For each operator, target directory should contain the full layout above.

## Typical workflow

1. Generate operator code/profiling under `output/{op}/`
2. Run `init_workspace.py --op {op}`
3. Verify `workspace/inputs/{op}/` contents
4. Run `code_tag` subskill, then start `workflow.py run --op {op}`

## Troubleshooting

- If no files are copied, confirm source paths exist under `output/{op}/`
- If no summary is found, verify profiling generated `op_summary_*.csv`
- If command exits with code `2`, check `[ERROR]` lines in log output
