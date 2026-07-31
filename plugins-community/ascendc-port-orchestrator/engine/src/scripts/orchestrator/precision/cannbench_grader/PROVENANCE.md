# cann-bench ecosystem precision grader — VERBATIM (byte-identical) VENDOR COPY

This directory is the **actual 生态 (ecosystem) op-gen precision grader**, vendored verbatim so
a5_ops grades against the SAME source of truth the ecosystem op-gen uses — **NOT a re-implementation**.

## Source
- Repo: `gitcode.com/cann/cann-bench`
- Commit: `007855b` (master, "fix: profiler step 前同步 NPU stream")
- Tree: `src/kernel_eval/utils/`
- Vendored on: 2026-06-30, under owner-directed precision-standard correction
  (our prior `precision_cannbot_adapter` had drifted to the cannbot DESCRIPTION-library formulas
  — 商用 dual-baseline ratio + an invented ①③ absolute floor — instead of this real `compare.py`).

## Files (grading logic preserved from the clone — md5 verified)
| file           | md5                                | provenance |
|----------------|------------------------------------|------------|
| `compare.py`   | `3f683648c42a77c9862ad5fa1b0701f8`  | clone `src/kernel_eval/utils/compare.py` + the two sanctioned, verdict-neutral changes below — **grading logic 0 edits** (verbatim base was `905b701eaf5861d8ea944266277d42c6`) |
| `thresholds.py`| `be42d99ac574aef6a32f84bdbb4a4440`  | VERBATIM from clone `src/kernel_eval/utils/thresholds.py` — **0 edits** |

`__init__.py` and this `PROVENANCE.md` are **our wrapper files** (not from the clone); they add the
package surface + provenance and do NOT alter grader logic.

## Import-path edits
**NONE.** `compare.py`'s only intra-repo import is `from .thresholds import (...)` (relative). By
vendoring both files into THIS package (made a package by `__init__.py`), that relative import
resolves to `cannbench_grader.thresholds` — so neither vendored file required any edit, and md5
stays identical to the clone. (Had any import edit been necessary it would be itemized here with the
exact line so the diff is provably import-only, not logic.)

## Dependency closure (self-contained, 0 external cann-bench imports)
- `compare.py` imports: `logging`, `math`, `traceback`, `typing`, `dataclasses` (stdlib), `torch`,
  and `from .thresholds import (...)`.
- `thresholds.py` imports: `typing` (stdlib) only.
- ⇒ the FULL closure of the grader is exactly `{compare.py, thresholds.py}`. No `kernel_eval.*` or
  any other cann-bench module is reachable. Verified by importing the package from a clean process
  with cann-bench NOT on `sys.path` (see forward_fix_log.md import-closure proof).

## a5_ops sanctioned patches (grading VERDICT-NEUTRAL)

### Additive diagnostic field (sanctioned 2026-06-30, main-decided)

One additive change on top of verbatim `007855b` `compare.py`, authorized because the **(c) kernel-fail vs
(d) native-provisioning-gap** distinction — the whole purpose of the option-b forward gate — needs
compare.py's TRUE normal-region signal, which the vendor COMPUTES but does not EXPOSE:

- **What**: exposed `normal_mismatch_count` (the Stage-2 normal-region over-threshold count already
  computed at `compare.py:~605`) as a new `CompareResult` field, threaded through
  `_compare_single_tensor`'s final return + `compare_tensors`' aggregation + `to_dict`.
- **Grading UNCHANGED**: NO `passed` / threshold / mask / carve-out logic was touched. The pass/fail
  VERDICT is byte-identical with or without the field — `git diff` vs `905b701…` shows ONLY the new
  field declaration + threading + comments (zero logic lines). Proven by
  `test_compare_py_additive_field_verdict_consistency` (verdicts unchanged; field populated correctly).
- **Why not recompute in a wrapper**: a parallel re-implementation of compare.py's region masks would
  be a SECOND source that silently DRIFTS from the vendor on re-vendor — exactly the drift class this
  patch avoids by reusing compare.py's own already-computed value.
- **RE-VENDOR DISCIPLINE**: when re-pulling a newer cann-bench `compare.py`, this additive field MUST be
  RE-APPLIED at 3 sites (`CompareResult` dataclass + `_compare_single_tensor` final return +
  `compare_tensors` aggregate return; plus `to_dict`) and the verdict-consistency test must stay green.
  The grading logic is still tracked verbatim against the upstream commit; only this one exposure rides
  on top. Any FURTHER change needs the same main-decision + verdict-consistency proof.

### Device-neutral CPU normalization (sanctioned portability cleanup)

- **What**: the pre-comparison transfer guard now moves any tensor whose device type is not `cpu`
  back to CPU. This replaces an accelerator-specific condition without changing the CPU/NPU path.
- **Grading unchanged**: shapes, thresholds, masks, error metrics, carve-outs, and pass/fail decisions
  are untouched. The supported CPU and NPU inputs take the same transfer path as before.
- **RE-VENDOR DISCIPLINE**: reapply this two-line device-neutral guard when refreshing the upstream
  file, then update the full-file hash and rerun the verdict-consistency regression.

## The grading method (for reviewers)
`_compare_single_tensor(output, golden, threshold, dtype, native_output)` (compare.py L184):
- golden = fp64 CPU; `output` cast back to its native dtype; per-dtype `threshold` from thresholds.py
  (fp32 2^-13, fp16 2^-10, bf16 2^-7).
- **Stage 1** (L484): overall MERE < threshold AND overall MARE < 10·threshold → PASS.
- **Stage 2** (L505-647, if Stage-1 fails): over-threshold points in the NORMAL value region → FAIL;
  else small-value (|golden|<2^-14, err>2^-30) + cancellation (|out|≈0 & golden∈[2^-14,2^-8)) carve-outs
  judged by NPU error-count / `native_output` (CPU-same-precision) error-count ≤ 2. With
  `native_output=None` the CPU baseline error-count is 0 ⇒ NPU must be exact in the carve-out region
  (stricter — never looser). This is the REAL grader behavior; provisioning a real CPU-same-precision
  `native_output` (Phase 2) is what correctly relaxes the near-zero region.

DO NOT edit the **GRADING LOGIC** of `compare.py` / `thresholds.py` (Stage-1/Stage-2 pass/fail,
thresholds, masks, carve-outs) — that is the upstream standard. The only sanctioned changes are the
verdict-neutral diagnostic exposure and device-neutral CPU normalization documented above; any
further change needs the same main-decision + verdict-consistency proof.
