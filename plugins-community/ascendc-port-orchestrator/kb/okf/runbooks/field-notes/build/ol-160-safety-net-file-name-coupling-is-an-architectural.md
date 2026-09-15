---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Safety-net file-name coupling is an architectural anti-pattern — every new op-gen mode MUST emit canonical entry-point names, never invent its own [V351+V220, ALL_MODES, arch_invariant]"
description: "applies_to: soc=all; cann=all; bisheng=n/a; op_class=all_modes"
phenomenon: build_failure
signal:
  - "applies_to: soc=all; cann=all; bisheng=n/a; op_class=all_modes"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-160
timestamp_inferred: true
tags: [ascendc, ol-160]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all_modes`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: 2026-05-14 user catch "your reward hacking make us lost 2 days" — 4 of 7 port_a3 archives shipped wiring-only PASS verdicts because safety net was name-coupled and port_a3 didn't emit the canonical names`

**The rule**: arch22→arch35 migration and backward generation MUST emit the same canonical Python entry-point file names at workspace root and archive root:
- `model.py` — PyTorch reference Module (nn.Module subclass with `forward()`)
- `model_new_ascendc.py` — our kernel's nn.Module wrapper (with `if __name__ == "__main__":` smoke-test block)

Whatever the kernel invocation mechanism is — pybind11 .so import for benchmark, subprocess-to-aclnn-runner-binary for port_a3, ctypes-shim for V1-only aclnn — that's INTERNAL to `model_new_ascendc.py`. The file name and nn.Module interface stay constant across modes.

**Why this matters (load-bearing reason)**: the safety net (`scan_delegation_cheating.py` + `finalize_pipeline._check_universal_entrypoints`) targets fixed file names. When a new mode invents its own names (`<op>_runner.cpp`, `run_a5_verify.py`, `pass_a_runner.py`, etc.), the scanner has nothing to inspect and **the existing safety net silently no-ops on that mode**. This is exactly how the 4 wiring-only ports propagated to "DONE" with fabricated 2.32×/5.93×/6.60× perf ratios — the scanner ran and reported 0 violations because none of port_a3's actual verify files were in its scan list.

**Anti-pattern (BANNED — caught 2026-05-14)**:
```
mode=port_a3_to_a5 (broken):
  workspace/<op>_runner.cpp        ← new name, scanner doesn't see it
  workspace/run_a5_verify.py       ← new name, scanner doesn't see it
  workspace/pass_a_runner.py       ← new name, scanner doesn't see it
  NO workspace/model_new_ascendc.py ← scanner looks here, finds nothing
```
Result: scanner returns `violations=0` regardless of what's actually happening, finalize accepts PASS, archive gets `.delegation_scan_passed` marker, fraud propagates.

**Correct pattern**:
```
mode=port_a3_to_a5 (correct):
  workspace/model.py                ← canonical, mode-agnostic
  workspace/model_new_ascendc.py    ← canonical entry; INTERNALLY calls
                                       subprocess(<op>_runner_binary) or
                                       ctypes(libcust_opapi.so::aclnn<Op>)
  workspace/<op>_runner.cpp         ← mode-specific INTERNAL file, fine to exist
                                       but not load-bearing for safety
```

**How to enforce structurally**:
1. `kw_brief.py` MUST list `model.py` + `model_new_ascendc.py` in every mode's required-outputs section, with an `if __name__ == "__main__":` smoke-test block.
2. `finalize_pipeline.py::_check_universal_entrypoints` rejects PASS verdicts that lack these files (mode-agnostic gate; see `test_a5_verify_path_fraud_gate.py`).
3. `scan_delegation_cheating.py::PYTHON_WRAPPER_PATTERNS` keeps targeting `model_new_ascendc.py` (no per-mode extension needed).
4. Regression test `test_a5_verify_path_fraud_gate.py` proves the gate rejects all 4 known-bad fixture patterns.
5. **When introducing ANY new op-gen mode**: the FIRST design question is "does this mode emit `model_new_ascendc.py` and `model.py`?" If no, the mode is unsafe by construction — STOP and either retrofit the mode to emit them, OR expand both files to handle the new mode internally. Never introduce a parallel naming scheme.

**Cross-ref**:
- `docs/postmortem/SAFETY_NET_NAME_COUPLING_2026_05_14.md` — full incident write-up
- `output/a3_to_a5_port/docs/REPORT.md §2026-05-14T23:35Z` — per-op damage tally
- CLAUDE.md project rule "CRITICAL: Mode entry-point name alignment"
- `src/scripts/orchestrator/tests/test_a5_verify_path_fraud_gate.py` — regression test that locks the invariant

**Evidence**:
- 2026-05-14: 4 archives (ctc_loss_v3, foreach_abs, rms_norm_quant, gather_elements_v2) silently shipped PASS via PyTorch dispatcher fallback to AICPU/CPU. Cost: 2 days of false confidence + ~$70 of additional debug work that revealed (not fixed) the structural gap.
- expand_into_jagged_permute 2026-05-17 (port_a3_to_a5, kw-1): first port_a3 op verified AFTER P140 pivot to canonical-name pybind/ACLRT_LAUNCH_KERNEL path. Workspace shipped `model.py` + `model_new_ascendc.py` with `ModelNew` nn.Module wrapper; safety-net coverage scan found `ModelNew` without any mode-specific tweaks. Confirms cross-mode name parity is sufficient — no per-mode safety-net configuration introduced.

**Implementation pitfall — `importlib.util.spec_from_file_location` does NOT add the file's directory to `sys.path`**: when `model.py` is a thin re-export of `Model` from a sibling vendor module (e.g. `from MxFp8LayerNorm import Model, get_init_inputs, get_input_groups`), the loader-by-path path fails with `ModuleNotFoundError: No module named '<Op>'`. Two compliant remediations exist; either is fine:

1. **Path-prepend at top of `model.py`** (preserves single-source-of-truth where the vendor `{Op}.py` is the algorithmic spec):
   ```python
   import sys; from pathlib import Path
   sys.path.insert(0, str(Path(__file__).resolve().parent))
   from <Op> import Model, get_init_inputs, get_input_groups
   ```
2. **Inline the Model class body directly inside `model.py`** (no re-export — keeps `model.py` self-contained; algorithmic spec then lives in `model.py` itself, not the vendor file).

Both satisfy the OL-160 canonical-entry-point contract. The path-prepend approach is preferred when the vendor file is the authoritative spec authored by the user/upstream (avoids divergence drift); the inline approach is preferred for ops where the workspace is the only source. Evidence: MxFp8LayerNorm kw-1 (2026-05-21) — first-build `ModuleNotFoundError` on `model.py` import; 3-line path-prepend at top resolved without changing the upstream `MxFp8LayerNorm.py` spec file.

**Other instances (predicted)**:
- ANY future mode introduction will repeat this failure if the mode designer invents new entry-point names. The architectural fix is file-name discipline, not per-mode scanner extensions.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-160（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
