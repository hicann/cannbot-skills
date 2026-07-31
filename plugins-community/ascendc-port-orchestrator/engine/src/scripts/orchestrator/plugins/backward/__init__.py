# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""backward mode plugin — auto-generate a reverse (gradient) AscendC operator
from a forward op so generated ops can participate in TRAINING, not just inference.

Design: docs/design/BACKWARD_PLUGIN_DESIGN.md (merged #270). The plugin owns the
complete supported backward path: state-driven dispatch, CPU-fp64 autograd truth,
generation brief, target-NPU remeasurement, finalize gates, and archive layout.

Concern split (design §4):
- C1 backward *generation* = THIS standalone mode (torch.autograd.grad as the exact
  reference oracle). Selected explicitly via `--backward` / `opgen_mode == "backward"`,
  NEVER inferred from op shape.
- C2 backward *perf* (OL-200 MIX_AIC cube/vec cross-taskId software pipelining) =
  cross-cutting op_class knowledge, NOT owned by this plugin (so e.g. LIG consumes it
  from its own mode without being force-migrated here).

Mode is an AscendC paradigm but uses the **pybind11 `.so` paradigm**
(`kernel/pybind11.cpp` + `<op>_kernels.cpp`) — not the migration `op_host/`
deliverable — so it overrides PB-33 `check_op_host_completeness`; the real
deliverable model_new_ascendc.py + verifiable .so is still enforced by _check_universal_entrypoints).
Detection is state-driven and therefore mutually exclusive with the migration mode.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..base import BasePlugin


class BackwardPlugin(BasePlugin):
    name = "backward"
    cli_flag = "--backward"

    def detect(self, workspace: Path) -> bool:
        """Match ONLY when .opgen_state.json declares opgen_mode == 'backward'.

        No layout heuristic: the explicit state declaration is the sole trigger.
        """
        state_file = workspace / ".opgen_state.json"
        if not state_file.is_file():
            return False
        try:
            state = json.loads(state_file.read_text())
        except Exception:
            return False
        return state.get("opgen_mode") == self.name

    def kernel_logic_files(self) -> tuple[str, ...]:
        """Backward mode emits the canonical deliverable — this module's
        header: "model_new_ascendc.py + verifiable .so", enforced by
        `_check_universal_entrypoints`. The grad kernel is reached through
        the canonical entry-point; there is no paradigm-native second
        source, so the canonical name holds this mode's Python compute.

        `model.py` is excluded on purpose: it is the forward oracle that
        `torch.autograd.grad` differentiates for truth. It is SUPPOSED to
        be pure torch, and scanning it for delegation would flag the
        reference itself.
        """
        return ("model_new_ascendc.py",)

    def kernel_cpp_dirs(self) -> tuple[str, ...]:
        """Backward mode uses the standalone-pybind `.so` paradigm:
        `kernel/pybind11.cpp` + `<op>_kernels.cpp` +
        `<op>_kernel.h` + `<op>_tiling.h` under the flat `kernel/` dir. That
        is where its AscendC C++ compute lives, so declare it for scanner
        coverage (DEBT-211 directory-level).
        """
        return ("kernel",)

    def truth_source(self) -> Optional[str]:
        """Backward truth = fp64 torch.autograd.grad over the forward
        (model.py oracle), compared by the worker's verify_<op>.py (B3.3b:
        NOT edge_dataset; no run_pass_a/edge_dataset.pt). Plugin-method
        dispatch (DEBT-161 paradigm) — burns down the phase_o5
        `opgen_mode == "backward"` literal-branch.
        """
        return "backward_autograd"

    def should_run_phase_o5_perf_capture(self) -> bool:
        """Backward mode does NOT use phase_o5_perf_capture — it has its own NPU
        perf path: backward_verify_runner runs verify_<op>.py on the NPU (via
        {TARGET}_NPU_PYTHON_BIN) and parses its perf summary (author != measurer).

        Without this override the inherited BasePlugin default (True) makes the
        orchestrator fall into phase_o5_perf_capture, whose subprocess runs
        `python3` (the orchestrator's LOCAL python, no torch_npu) -> crashes on
        `import torch_npu` -> perf=N/A. Caught 2026-06-19 on the GQA-bwd orch
        archive run. Backward perf is the backward_verify_runner's job.
        """
        return False

    def check_op_host_completeness(self, workspace: Path) -> Optional[str]:
        """PB-33 gate does NOT apply to backward mode — returns None (B3.3b).

        Empirically confirmed via finalize probe 2026-05-30: the backward
        hardware-proven archives (mul_grad / rms_norm_grad / layer_norm_grad /
        mm_grad / swiglu_grad) build via the **standalone pybind11 `.so`**
        paradigm — `kernel/pybind11.cpp` + `<op>_kernels.cpp` + `<op>_kernel.h`
        + `<op>_tiling.h` → `_<op>_ext`. They do
        NOT emit/consume op_host's `<op>_def.cpp` / `<op>_tiling.cpp` /
        `CMakeLists.txt`; op_host is **vestigial-for-build** here. Without this
        override the inherited BasePlugin PB-33 body blocks every backward
        archive at finalize on "workspace/op_host/ missing".

        Backward-mode completeness is NOT left unguarded: the real deliverable
        (`model_new_ascendc.py` + verifiable `.so` + self-contained
        `verify_<op>.py` vs autograd truth) is enforced by
        `_check_universal_entrypoints` (GateID.A5_VERIFY_PATH_FRAUD / OL-160)
        and the phase_o5 backward re-measure (author≠measurer). Scoping
        op_host→port_a3 is gate-scoping-to-where-load-bearing, NOT fraud-net
        loosening: PB-33 stays STRICT for port_a3_to_a5 (op_host IS the
        deliverable). The standalone pybind deliverable is verified through
        its own `.so`, not `op_host/`.
        """
        return None

    def _verification_script(self, workspace: Path, vj: dict) -> Optional[Path]:
        op = vj.get("op")
        if not op:
            try:
                op = json.loads((workspace / ".opgen_state.json").read_text()).get("op")
            except Exception:
                op = None
        candidates = [workspace / f"verify_{op}.py"] if op else []
        candidates.append(workspace / "verify.py")
        return next((path for path in candidates if path.is_file()), None)

    def check_binary_provenance(
        self, workspace: Path, vj: dict
    ) -> Optional[str]:
        """Require the locally archived binary that O5 loads and executes."""
        status = (vj.get("precision", {}) or {}).get("status")
        if status not in ("PASS", "PASS_WITHIN_TOLERANCE"):
            return None
        root = workspace.resolve()
        for candidate in workspace.rglob("*.so"):
            try:
                if candidate.is_symlink():
                    continue
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root)
                if resolved.is_file() and resolved.stat().st_size > 0:
                    return None
            except (OSError, ValueError):
                continue
        return (
            "backward precision PASS has no non-empty workspace-local .so; "
            "the generated kernel must be compiled before O5 verification"
        )

    def check_verifier_uses_modelnew(self, workspace: Path, vj: dict) -> Optional[str]:
        """Require the backward verifier to invoke the generated wrapper in code."""
        import ast

        verifier = self._verification_script(workspace, vj)
        if verifier is None:
            return "backward verifier missing (expected verify_<op>.py or verify.py)"
        try:
            tree = ast.parse(verifier.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as exc:
            return f"backward verifier cannot be inspected: {exc}"

        imports_wrapper = any(
            (
                isinstance(node, ast.ImportFrom)
                and node.module == "model_new_ascendc"
            )
            or (
                isinstance(node, ast.Import)
                and any(alias.name == "model_new_ascendc" for alias in node.names)
            )
            for node in ast.walk(tree)
        )
        if not imports_wrapper:
            return "backward verifier does not import model_new_ascendc"

        instances: set[str] = set()
        instance_lines: dict[str, int] = {}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if not isinstance(value, ast.Call):
                continue
            func = value.func
            is_model_new = (
                isinstance(func, ast.Name) and func.id == "ModelNew"
            ) or (
                isinstance(func, ast.Attribute)
                and func.attr == "ModelNew"
                and isinstance(func.value, ast.Name)
                and func.value.id == "model_new_ascendc"
            )
            if not is_model_new:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    instances.add(target.id)
                    instance_lines[target.id] = node.lineno
        if not instances:
            return "backward verifier does not instantiate ModelNew()"

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                name = func.value.id
            if name in instances and node.lineno > instance_lines[name]:
                break
        else:
            return "backward verifier instantiates ModelNew but never calls it"

        return None

    def check_verify_path_provenance(
        self, workspace: Path, vj: dict
    ) -> Optional[str]:
        usage_error = self.check_verifier_uses_modelnew(workspace, vj)
        if usage_error:
            return usage_error
        stamp = vj.get("harness_pristine", {}) or {}
        if (
            stamp.get("sampled_at") != "o5_post_verify"
            or stamp.get("o5_verdict") not in ("VERIFIED", "PROVISIONAL")
        ):
            return (
                "backward verifier has no O5 independent NPU execution stamp; "
                "worker-authored counts alone are not accepted"
            )
        return None

    @staticmethod
    def _check_pass_a_concrete(workspace: Path, vj: dict) -> Optional[str]:
        precision = vj.get("precision", {}) or {}
        if precision.get("status") not in ("PASS", "PASS_WITHIN_TOLERANCE"):
            return None
        pass_a = precision.get("pass_a", {}) or {}
        passed = pass_a.get("tier1_pass")
        total = pass_a.get("total")
        if (
            pass_a.get("status") != "PASS"
            or not isinstance(passed, int)
            or isinstance(passed, bool)
            or not isinstance(total, int)
            or isinstance(total, bool)
            or passed < 0
            or total <= 0
        ):
            return (
                "backward precision.pass_a requires status=PASS with concrete "
                "non-negative tier1_pass and positive total from the independent "
                "NPU verifier."
            )
        return None

    def extra_finalize_checks(self):
        return [("pass_count", self._check_pass_a_concrete)]

    def archive_project_subdir(self) -> Optional[str]:
        """Keep backward deliverables in a mode-owned archive tree."""
        return "backward_ops"

    def kw_brief_phase_block(
        self, *, op: str = "", workspace=None,
        iter_cap_remaining: int = 0,
        directive_text=None,
        handoff_from_prior_agent=None,
        env=None,
    ):
        """B3.3a: backward GENERATION Phase A-D worker brief.

        Fires via kw_brief's plugin-override hook for opgen_mode==backward (the
        dispatch reaches this only when directive_text is None — a respawn
        directive wins upstream). Encodes the self-contained verify generation
        contract (BACKWARD_PLUGIN_DESIGN §5.5) validated on mul_grad /
        rms_norm_grad / layer_norm_grad: reference = the B3.2-generated autograd
        model.py; verify is self-contained (seed per backward_ref.json recipe,
        compare vs autograd truth), NOT edge_dataset.

        Lessons baked in (from the 3 hardware proofs this session):
        - FAIR perf baseline: A/B vs the RAW vendor C-API (e.g.
          torch_npu.npu_<op>_backward / npu_<op>grad with stats precomputed),
          NOT autograd-through-the-forward (that adds torch autograd-engine
          Python overhead and inflates the ratio — layer_norm_grad worker hit this).
        - VEC transcendental precision: 910B/A2 (V220) VEC Exp/Sigmoid/Tanh/
          Rsqrt/Reciprocal/Sqrt/Div are ~fp16-mantissa piecewise-poly approx, NOT
          fp32-grade (OPERATIONAL_KNOWLEDGE.md, the "transcendental fp16-grade" OL).
          For a precision-sensitive grad, refine with Newton-Raphson on the scalar.

        The O2.5 backward branch produces the self-contained CPU-fp64 autograd
        truth, then the regular worker and O5/finalize path builds and verifies
        the generated operator on the selected target NPU.
        """
        from pathlib import Path as _P
        # FA-class BACKWARD architecture route (2026-06-20, owner via the C19 finding):
        # when this backward op is an attention-family op (flash_attention_score_grad,
        # GQA/SWA/cross grad, …), DEFAULT to the proven MULTI-LAUNCH approach
        # (CAND-FA-GQA-BWD-1 — precision-core-complete: 45/45 incl S>128+fp32). The fused
        # single-launch stitch (P-P103 BACKWARD) is BN2-only (S≤128, no fp32) + ~0.2× on
        # small shapes (its large-S perf value is an UNMEASURED hypothesis), so it is
        # GATED behind an explicit opt-in (`fa_backward_arch=="fused"` / `fa_backward_large_s`),
        # NEVER the default. PREPEND the selected architecture brief; the analytic-derive
        # Phase A-D prose below carries the B3.3b verify/finalize SCHEMA CONTRACT + cannbot
        # judge (verify is identical — fp64 autograd oracle either way). The kw_brief forward
        # FA block returns None for FA-grad ops precisely so this override runs.
        fa_stitch_prefix = ""
        try:
            import json as _json0
            op_class = ""
            if workspace is not None:
                _cls_p = _P(workspace) / "op_classification.json"
                if _cls_p.is_file():
                    _tags = _json0.loads(_cls_p.read_text()).get("op_class_tags") or []
                    op_class = " ".join(_tags) if isinstance(_tags, list) else str(_tags)
            from briefs.kw_brief import (
                _is_fa_class_backward as _is_fa_bw,
                _fused_fa_backward_requested as _fused_req,
                _fa_class_backward_stitch_block as _fa_stitch,
                _fa_class_backward_multilaunch_block as _fa_multilaunch,
            )
            if _is_fa_bw(op, op_class, workspace):
                if _fused_req(op, workspace):
                    # EXPLICIT opt-in → the fused single-launch stitch recipe.
                    fa_stitch_prefix = _fa_stitch(op, op_class) + (
                        "\n\n---\n\n# BACKWARD-MODE VERIFY / FINALIZE CONTRACT "
                        "(below — the FA-grad stitch above is the GENERATION recipe;\n"
                        "the Phase A-D prose below is the SELF-CONTAINED autograd verify + "
                        "the B3.3b finalize schema. For an FA-grad op: use the trusted\n"
                        "source only to recover interfaces, algorithm semantics, and\n"
                        "constraints; independently author the implementation. Do not\n"
                        "copy source blocks or lines. Verify per the contract below.)\n\n"
                    )
                else:
                    # DEFAULT → the proven MULTI-LAUNCH approach (precision-core-complete).
                    fa_stitch_prefix = _fa_multilaunch(op, op_class) + (
                        "\n\n---\n\n# BACKWARD-MODE VERIFY / FINALIZE CONTRACT "
                        "(below — the MULTI-LAUNCH architecture above is the GENERATION\n"
                        "recipe; the Phase A-D prose below is the SELF-CONTAINED autograd "
                        "verify + the B3.3b finalize schema. The multi-launch path IS the\n"
                        "'derive the per-grad math, REAL AscendC' default below — follow it;\n"
                        "verify per the contract below.)\n\n"
                    )
        except Exception:
            fa_stitch_prefix = ""
        bref = None
        try:
            if workspace is not None:
                bref_p = _P(workspace) / "backward_ref.json"
                if bref_p.is_file():
                    import json as _json
                    bref = _json.loads(bref_p.read_text())
        except Exception:
            bref = None
        wrt = (bref or {}).get("wrt", ["<forward inputs to differentiate>"])
        grad_out = (bref or {}).get("grad_output", "explicit")
        recipe = (bref or {}).get("seeding_recipe", "see backward_ref.json")
        return fa_stitch_prefix + f"""# PHASES (cold-start aog-kernel-worker — BACKWARD generation, §5.5 self-contained)

This is a **backward (gradient) op** (opgen_mode=backward). Produce grads for {wrt}
(grad_output={grad_out!r}). The reference truth is SELF-CONTAINED autograd — NOT edge_dataset.
A. **KB Manifest LOAD** + read `analysis.md` plan. Load the V220/membase transcendental-
   precision OL (Exp/Sigmoid/Tanh/Rsqrt/Reciprocal/Sqrt/Div are ~fp16-mantissa approx on
   V220 — refine precision-sensitive scalars with Newton-Raphson).

B. **Reference + signature** — the autograd reference is `workspace/{op}/model.py`
   (B3.2-generated: `Model().forward(*forward_inputs, gy) -> tuple(grads)` via fp64
   torch.autograd.grad). Read `forward_spec.py` for the differentiable forward; DERIVE the
   analytic backward by hand from it. Inputs = forward tensors the grad math needs + gy;
   outputs = one grad per wrt ({wrt}) — multi-output. Write `analysis.md` (forward formula,
   analytic backward per wrt, DET_POLICY).

C. **Author the AscendC backward kernel** — REAL AscendC only (DataCopy / VEC / reductions /
   TQue·TBuf). **FORBIDDEN**: in-kernel torch.autograd, CPU gradient compute, aclnn grad-op
   delegation. The autograd `model.py` is the REFERENCE ORACLE ONLY — NEVER call it / autograd
   as the kernel's compute. **archive-blind**: do NOT copy another op's `output/` kernel; reuse
   only KB techniques (fold-reduction, NR refinement). Emit canonical `model_new_ascendc.py`
   (OL-160) with the multi-output forward signature. **Name the kernel-wrapper class `ModelNew`
   (canonical, preferred) — NOT `Model`: `Model` collides with the `model.py` autograd ORACLE's
   class and is only tolerated by the scanner (`finalize_pipeline.py:3807` accepts both), so
   `ModelNew` keeps the entry-point unambiguous.** Build via `deploy_to_npu_lane.sh --build`.

D. **SELF-CONTAINED verify + perf** — write `verify_<op>.py`: **LOAD each case directly from
   `backward_cpu_truth.pt`** — `torch.load(...)["records"]`, iterate `status=="ok"` records;
   each record carries `inputs` (dict name→tensor), `grad_outputs` (tensor or list), `grads`
   (dict name→fp64 truth), `dtype`, and a value-`profile` (one of randn/zeros/large/small/
   boundary — the input_gen edge coverage, 2026-06-11). **Use `record["inputs"]` +
   `record["grad_outputs"]` AS-IS — do NOT re-seed from the recipe** (inputs are edge-profiled;
   re-seeding `torch.randn` would produce the WRONG inputs and mismatch the truth). The recipe
   ({recipe[:70]}...) in backward_ref.json is for REGENERATION only. (`status!="ok"` records are
   §5.4 non-finite truth — skipped automatically by the case builder below.)

   The verifier MUST `import model_new_ascendc`, instantiate `ModelNew()`, and call that
   instance for every NPU result. A direct extension call or a printed count that bypasses
   `ModelNew` is rejected. Keep the compiled, non-empty `.so` inside the workspace so the
   independent O5 run and final archive refer to the same generated binary.

   **PRECISION = THE CANNBOT SINGLE JUDGE (owner 2026-06-18, 完全照抄 cannbot — REQUIRED;
   replaces the old hand-rolled two-tier-tol T1 + the separate additive T2):** do NOT hand-compare
   grads with ad-hoc tolerances and do NOT compute any backward-private statistic. Build the
   canonical cannbot `cases` and route them through the SAME `grade_batch` that
   pass_a/benchmark/port_a3 use, via the two backward helpers (deployed in the precision/ dir):

       from backward_cdv_collect import cases_from_records   # VERIFY-side; NO re-seed (守 no-reseed)
       from backward_cdv_grade import grade_cases            # cannbot single judge (grade_batch)

       records = torch.load("backward_cpu_truth.pt", weights_only=False)["records"]
       cases = cases_from_records(records)   [[HASH]] per record: npu=kernel(record inputs+grad_outputs);
                                             [[HASH]] golden=record["grads"] (fp64); native=CPU-SAME-PRECISION
                                             [[HASH]] autograd on the SAME record inputs (the 生态 compare.py
                                             [[HASH]] carve-out baseline, native_kind="cpu_same_precision");
                                             [[HASH]] third_party=same tensor for the OPTIONAL 商用 ratio;
                                             [[HASH]] is_edge=(profile≠representative). ONE case/output×record.
       res = grade_cases(cases, "<op>", op_class="float", precision_level="<L0|L1|L2>",  # tier per op
                         persist_outputs_path="npu_outputs.pt")  # persist raw arrays → offline re-grade

   `grade_cases` → `grade_batch` grades by the **DEFAULT 生态 standard** = the VERBATIM-vendored
   cann-bench `compare.py` (`cannbench_grader/`). golden=fp64 CPU; `native` (CPU-same-precision)
   feeds compare.py's small-value/cancellation carve-out (compare.py is the COMPLETE grader —
   MERE/MARE Stage-1 + small-value/cancel Stage-2 + inf/nan). The 商用 double-baseline **② ratio**
   is an OPTIONAL non-default route (`route="commercial"`, needs a 标杆) — NOT used by `grade_cases`'s
   default. `is_edge` cases (zeros/large/small/boundary) are EXCLUDED from the statistical verdict
   (bug-find stream only). This is ONE source of truth — do NOT add a second gate, a manual tol, or
   relax the judge. `persist_outputs_path` saves a bounded subset of the raw npu/golden/native arrays
   alongside verification.json so a failed op can be re-graded offline under a corrected metric with
   NO NPU re-run.

   **⚠ NATIVE-FEED PITFALL (2026-07-01, archive-wide gap — 3 backward archives shipped it):**
   PREFER `cases_from_records` (it wires native for you). If you hand-roll an inline verify
   loop instead (e.g. to also emit per-output informational diagnostics), you MUST still pass
   `native_grads=<the cpu_same_precision ref>` to `_build_record_cases` — the SAME
   `_competitor_same_dtype(...)` tensor you build for the competitor IS the 生态 native_output
   baseline (one tensor, two consumers: 生态 native + optional 商用 ratio). Omit it and
   `case['native']=None` → compare.py's Stage-2 small-value/cancellation carve-out silently
   falls to its STRICT baseline → 生态 UNDER-reports with no error (real bug: ss-bwd graded
   6/48 instead of the true 9/48). A finalize gate enforces this (DEBT-182).

   **FAIL-CLOSED (never fabricate PASS):** `grade_cases` returns
   `verdict ∈ {{PASS, FAIL, REJECT_FAIL_CLOSED}}` plus the canonical `res["cannbot"]` block.
   Empty cases / no representative / inconclusive → `REJECT_FAIL_CLOSED` → emit `status:"FAIL"`
   (a REJECT is NOT a pass — honest "blocked" over a fabricated green). The DEFAULT judge is the
   生态 vendored compare.py (golden=fp64 truth + native=CPU-same-precision carve-out baseline); the
   old additive T2 `pass_a_t2` block is GONE and the 商用 ② ratio is an OPTIONAL non-default route
   (route="commercial"), NOT the default gate. Carry the `grade_batch` result as evidence under
   `verification.json.precision.pass_a.cannbot`, NOT a separate `pass_a_t2`.

   **Perf A/B — FAIR
   baseline**: back-to-back same-NPU vs the RAW vendor C-API (`torch_npu.npu_<op>_backward` /
   `npu_<op>grad` / `torch.ops.aten.<op>_backward` with mean/rstd/stats PRECOMPUTED, untimed) —
   NOT autograd-through-the-forward (Python autograd-engine overhead inflates the ratio).
   **ATTENTION-FAMILY (FA-class) ops — the vendor C-API IS `torch_npu.npu_fusion_attention` +
   `npu_fusion_attention_grad`** (GQA: separate Q vs K/V `head_num`; sliding-window: `sparse_mode` /
   `atten_mask`; cross: Sq≠Skv inferred from shapes). Run the FORWARD once to precompute softmax
   stats (OUTSIDE the timed region), then time ONLY `npu_fusion_attention_grad`. **DO NOT classify an
   attention-family backward as parity / N/A** — it HAS a fused vendor backward; you MUST measure the
   A/B (see `CAND-FA-GQA-BWD-1` / `CAND-FA-SWA-BWD-1/2` / `CAND-FA-CROSS-BWD-1` in KB for the exact
   params + reference ratios 1.391× / 0.852× / 0.960×). **Mask FAIRNESS (CAND-FA-SWA-BWD-2)**: any
   `[S,S]` mask MUST be pre-built / cached OUTSIDE the timed region on BOTH sides — never penalize
   ours with a per-call host mask-build that the vendor (given a pre-built mask) does not incur.
   **TIMING METHOD (REQUIRED — author-time, or finalize P0ee METHODOLOGY_DECLARATION rolls you
   back, esp. for any ratio >1.0×)**: time BOTH sides with `torch_npu.profiler`
   device_self_duration (operator_details / kernel_details), warmup>=5 + active>=5, take the
   median on-device time. **DO NOT use `aclrtEvent` for backward perf** — P97
   PERF_METHODOLOGY_ASYMMETRY: Event record/elapsed straddles the launch and absorbs host dispatch
   overhead ASYMMETRICALLY between a single fused vendor C-API call and our multi-launch backward
   kernel, INFLATING the ratio (measured the SAME Sum kernel at 0.86×-Event vs 0.56×-profiler — the
   Event number is fiction). Profiler device_self_duration is pure on-device kernel time, symmetric
   across both sides, and is the ONLY method the finalize gate trusts for backward. **NOT**
   wall-clock `time.time()`/median-of-iters either (host-time, launch/sync overhead, rejected by
   P0ee). Same profiler method symmetrically for our kernel + the vendor baseline (OL-163).
   **`performance.method` MUST literally contain ALL THREE tokens the P97 gate parses for:
   `torch_npu.profiler` AND `warmup=5` AND `active=5`** — e.g. `"torch_npu.profiler
   device_self_duration (kernel_details.csv); schedule warmup=5, active=5; symmetric both sides"`.
   Omitting the `warmup=5, active=5` schedule tokens (even if you ran that schedule) → P97
   PERF_METHODOLOGY_ASYMMETRY rollback (costs an extra worker iteration — caught 2026-06-11
   rotary_mul_grad kw-1). Honest; flag shared-NPU contention.

   **B3.3b FINALIZE SCHEMA CONTRACT (REQUIRED — orch --backward end-to-end gates depend on it):**
   - `verify_<op>.py` MUST print, as the LAST JSON **object** on stdout, a summary
     `{{"tier1_pass": <n_pass>, "total": <n_rep>, "status": "PASS"|"FAIL",
     "performance": {{...perf block...}}}}` where the counts come from the cannbot judge over the
     REPRESENTATIVE stream: `total = res["cannbot"]["n_representative"]`,
     `tier1_pass = round(res["cannbot"]["pass_rate"] * total)`,
     `status = "PASS" if res["verdict"] == "PASS" else "FAIL"` (a `REJECT_FAIL_CLOSED` verdict →
     `status:"FAIL"`). The per-case `JSON:[...]` array is informational — phase_o5's backward
     re-measure parses the final object to independently re-verify your pass_a claim
     (author≠measurer). You MAY ride the canonical `res["cannbot"]` block along under a `"cannbot"`
     key as evidence (scenario / verdict_basis / gate / ci_upper), but it MUST NOT change
     `status`/`tier1_pass` — those are the single judge's verdict.
   - **B3.3c (REQUIRED): the stdout summary object MUST include `"performance"`** — the SAME perf
     block you write to `verification.json.performance` (`{{"status", "ratio", "median_ratio",
     "method", "rows": [...]}}`). phase_o5's `backward_verify_runner` runs THIS script on the NPU
     and parses `summary["performance"]` as the **orchestrator's independent perf re-measure**
     (NOT your verification.json self-report) → it populates `performance.independent_re_measure`
     so the finalize perf gate (`NEVER trust skill-reported performance`) closes legitimately.
     OMITTING `performance` from the stdout summary → the gate finds `independent_re_measure`
     missing → **finalize ROLLBACK** (the exact gap that blocked abs_nocase_grad 2026-06-17, before
     this contract). Emit perf in the summary even when status is `N/A` (carry the reason).
   - `verification.json.precision.pass_a` = `{{"status": "PASS"|"FAIL", "tier1_pass": <n_pass>,
     "total": <n_rep>, "method": "cannbot single judge (grade_cases → precision_cannbot_adapter.
     grade_batch over per-output×record cases vs fp64 autograd truth; DEFAULT 生态 vendored
     compare.py with native=CPU-same-precision)",
     "cannbot": res["cannbot"]}}` — the SAME counts as the stdout summary; they MUST equal the
     re-measured verify counts. The `cannbot` sub-block carries the canonical adapter result
     (scenario 生态 (default) / verdict_basis / bootstrap_valid / pass_rate / ci_upper / gate /
     n_representative / n_edge / per_case / edge) — this is the SINGLE source of truth; do NOT
     relax it and do NOT add a second backward-private gate.
   - **No `pass_a_t2`** — the old additive T2 double-baseline block is GONE; the DEFAULT judge is
     the 生态 vendored compare.py (the 商用 ② ratio is an OPTIONAL non-default route, not a
     side-channel). Do NOT emit a separate `verification.json.precision.pass_a_t2`.
   - `verification.json.precision.pass_b` = `{{"status": "N/A", "reason": "backward mode:
     pass_b degenerate by design — the fp64 autograd grad IS the truth source and verify_<op>.py
     runs every case as pass_a (analogous to port_a3 edge_dataset IS truth); no separate
     edge_dataset/pass_b", "method": "n/a — backward mode pass_b not applicable"}}`.
   - `verification.json.performance.status` MUST be a CANONICAL value — `"PASS"` /
     `"PASS_WITHIN_TOLERANCE"` (with a numeric `ratio`) / `"N/A"` (with `reason`). NEVER a prose
     sentence / "DEFERRED" / "PENDING" (finalize _check_universal_entrypoints rejects non-canonical;
     a parity-class op with no clear win → `"N/A"` + reason, NOT prose).
   - Also emit `precision.{{pass,total,status,cases[],dtype_summary}}` (native per-output detail)
     + `build` + `determinism` blocks as before.
   - **`knowledge_update.md` AUTHOR-TIME (REQUIRED — or finalize kb_writeup gate rolls you back;
     write it NOW, not after a rollback)**: ≥100 bytes, Phase E structure with a literal
     `## Findings` section (what the backward needed: analytic grad form, V220-transcendental-NR if
     used, race/sync fixes) + `## KB-promotable patterns` (or "consumes OL-xxx, none new") +
     `## KB entries touched`. The gate hard-requires the `## Findings` header.
   Exit `→ orchestrator: done` or handoff @aog-precision-probe / @aog-kernel-optimizer.""".replace(
        "[[HASH]]", "#"
    )

    # C2 backward-PERF guidance (OL-200 MIX_AIC pipelining) is injected separately
    # by kw_brief._backward_perf_c2_block (op_class-driven, cross-mode). The block
    # above is C1 GENERATION. Other hooks inherit BasePlugin neutral defaults until
    # B3.3b wires the end-to-end run (O2.5 boundary flip + finalize/O5 backward
    # self-contained truth_source).


_INSTANCE = BackwardPlugin()


def get_plugin_instance():
    return _INSTANCE


# Self-register at import time
from .. import register_plugin  # noqa: E402

register_plugin(_INSTANCE)
