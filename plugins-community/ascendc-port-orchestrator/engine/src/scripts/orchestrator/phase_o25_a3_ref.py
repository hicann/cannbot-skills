# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Phase O2.5 reference provider — A3-CANN variant (W4, ROADMAP §1.5).

For the port_a3_to_a5 OPGEN_MODE the ground-truth reference is NOT
CPU-PyTorch output; it's the output of the existing A2/A3-merged kernel
running on A3 hardware via the aclnn API. This module captures that
reference and emits the standard Phase O2.5 artifacts so downstream
phases (kw_brief, post_verify, finalize) can consume them uniformly.

Compared to the legacy generic reference-artifact path,
counterpart) this variant:

- Reads the algorithm spec from `<op_dir>/op_kernel/` + `<op_dir>/op_host/`
  (existing A3 ascendc source) rather than a Model.forward benchmark.
- Generates inputs either via the op's own CANN UT script
  (`tests/ut/op_kernel/<op>_data/gen_data.py`) or via SCHEMA case_gen fallback
  (when the op doesn't ship a UT generator — e.g. ctc_loss_v3).
- Captures A3 outputs by ssh-ing into the A3 container, running the
  aclnn entry-point binary (`examples/test_aclnn_<op>.cpp`), and pulling
  outputs back.
- Cross-op dependency discovery: parses `op_host/CMakeLists.txt` for
  `DEPENDENCIES <peer_op>` lines so kw_brief (W5) knows which peer ops'
  routers may need to be patched (e.g. ctc_loss_v3 depends on ctc_loss_v2
  whose aclnn router needs 3 surgical edits to route v3 on A5).

Emits to workspace/:
- `a3_reference_runnable.json`: verdict + paths + cross-op deps
- `edge_dataset.pt`: {inputs, a3_outputs} when successful (a3_outputs replaces
  a synthetic CPU-only output)
- `a3_baseline_perf.json`: A3-CANN per-case timings for the perf ratio in
  Phase O5

This W4 commit lands the pure-function helpers + the orchestrator
function with a defensive run_remote injection point. The actual ssh+
docker exec calls (build_a3_reference_kernel / run_a3_reference) are
exercised live in W14 PoC; for W4 we ship the deterministic logic
+ tests + a marker (`requires_a3_runner=True`) the caller can branch on.
"""
from __future__ import annotations
import logging

import json
import os
import re
import shlex
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from logging_config import get_logger  # cv-agent style logger
log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Decomposition facade (2026-07-06): this module was a 2889-line god-file. It is
# now split into cohesive leaves; the top-level orchestration entry
# `provision_a3_reference` stays HERE and calls the extracted helpers by name,
# which are re-imported below so both `provision_a3_reference`'s bare-name
# resolution AND external `phase_o25_a3_ref.<symbol>` access (+ monkeypatch)
# keep working. Leaf edges are unidirectional (leaves never import this facade):
#   a3_ref_common     — path/env + report dataclasses + shared case helpers
#   a3_ref_derive     — aclnn-entry / input-gen / signature derivation
#   a3_ref_npu        — NPU busy-gating + host-workspace config
#   a3_ref_validate   — CPU-truth capture + model-contract validation
#   a3_ref_provision  — remote-exec transport + build/run + gates + block msg
# ---------------------------------------------------------------------------
from a3_ref_common import (  # noqa: F401  (re-exported for callers + patch points)
    RunRemote,
    _A3_CONTAINER_HOME_DEFAULT,
    _a3_container_home,
    _ascendc_env_path,
    _case_model_kwargs,
    _coerce_case_list,
    _translate_to_a3_container_path,
    ModelContractResult,
    O25A3Report,
)
from a3_ref_derive import (  # noqa: F401
    _GE_DTYPE_TO_TORCH,
    derive_aclnn_entry,
    derive_input_gen_source,
    derive_op_dependencies,
    ensure_edge_inputs,
    generate_edge_inputs_from_signature,
    parse_a3_perf_log,
    parse_op_def_signature,
)
from a3_ref_npu import (  # noqa: F401
    _a3_host_workspace_root_from_env,
    _read_a3_npu_gate_config,
    _read_a3_npu_range_config,
    _run_npu_smi,
    check_a3_npu_busy,
    parse_aicore_pct,
    parse_npu_range,
    pick_idle_npu_in_range,
)
from a3_ref_validate import (  # noqa: F401
    _CONTRACT_FIX_HINT,
    _count_a3_outputs,
    _detect_npu_delegation,
    _diagnose_kwargs_signature,
    _diagnose_output_shape,
    _first_sample_case,
    _strip_py_comments,
    _strip_strings_and_comments,
    _validate_a3_capture,
    _write_a3_reference_runnable_json,
    provision_cpu_truth,
    provision_native_capture,
    validate_model_contract,
    validate_a3_capture_provenance,
    write_a3_capture_provenance,
)
from a3_ref_provision import (  # noqa: F401
    _default_run_remote,
    _emit_cpu_truth_scaffold,
    _run_prior_art_scan,
    _scp_pull_files,
    _scp_push_dir,
    _source_arch_complete_gate,
    _try_a3_author,
    build_a3_reference_kernel,
    format_block_message,
    probe_a3_reachable,
    run_a3_reference,
)
from source_op import resolve_logical_op_name


def provision_a3_reference(
    op_dir: Path,
    workspace: Path,
    *,
    source_build_op_dir: Optional[Path] = None,
    a3_host: str,
    a3_user: str = "root",
    a3_container: str = "npu-a3",
    # Neutral fallback; real flow passes A3_CANN_PATH from .ascendc_env
    # (orchestrator.py:665) — matches template + validate_ds_env.sh
    # (genericize, not employee-path).
    a3_cann_path: str = "/usr/local/Ascend/cann",
    run_remote: Optional[RunRemote] = None,
    probe_only: bool = False,
    npu_busy_check: Optional[Callable[..., tuple[bool, float, str]]] = None,
    npu_picker: Optional[Callable[..., tuple[Optional[int], dict, str]]] = None,
) -> O25A3Report:
    """Phase O2.5 A3-CANN reference provisioning.

    Args:
        op_dir: ops-nn op directory (e.g. ~/workspace/cann/ops-nn/loss/ctc_loss_v3)
        workspace: workspace/<op>/ to write artifacts into
        source_build_op_dir: original ops-nn operator path used only by the
            trusted source-package fallback. ``op_dir`` may be the immutable
            ``.source_arch22`` analysis snapshot, whose parents do not contain
            the repository-level ``build.sh``.
        a3_host / a3_user / a3_container / a3_cann_path: A3 connection info
        run_remote: dependency-injection point for tests (mocks ssh+docker exec)
        probe_only: deprecated compatibility switch. When True, provisioning
                    fails closed with ``LIVE_CAPTURE_REQUIRED``; a reachability
                    probe is never accepted as migration truth.

    Returns:
        O25A3Report with verdict + paths + errors.

    Side effects on success:
        Writes `workspace/a3_reference_runnable.json` (always — verdict + status).
        Writes `workspace/edge_dataset.pt` after live execution succeeds.
        Writes `workspace/a3_baseline_perf.json` after live execution succeeds.
    """
    rep = O25A3Report(verdict="UNKNOWN")
    logical_op_name = resolve_logical_op_name(op_dir)

    # A migration verdict is bound to a fresh live source-device execution.
    # Quarantine every prior canonical capture before any runner-discovery or
    # reachability exit so an earlier run can never remain reusable as truth.
    stale_fallback_markers = (
        workspace / ".truth_source_override",
        workspace / ".a3_ref_unavailable_cpu_truth_deferred",
    )
    has_stale_fallback = any(marker.exists() for marker in stale_fallback_markers)
    if has_stale_fallback:
        rep.recommendations.append(
            "stale fallback marker detected; cached reference is ineligible and a "
            "fresh live A3 capture is required"
        )
    stale_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    def _quarantine_capture(label: str, *, include_runnable: bool) -> bool:
        names = [
            "edge_dataset.pt",
            "a3_baseline_perf.json",
            "a3_capture_manifest.json",
        ]
        if include_runnable:
            names.append("a3_reference_runnable.json")
        for name in names:
            cached = workspace / name
            if not cached.exists():
                continue
            archived = cached.with_name(
                f"{cached.name}.stale-{label}-{stale_stamp}-{uuid.uuid4().hex[:8]}"
            )
            try:
                cached.rename(archived)
            except OSError as exc:
                rep.verdict = "STALE_REFERENCE"
                rep.errors.append(
                    f"could not quarantine prior reference {cached.name}: {exc}"
                )
                return False
        return True

    if not _quarantine_capture("pre-capture", include_runnable=True):
        _write_a3_reference_runnable_json(workspace, rep)
        return rep

    # P124 (2026-05-16): translate local-host op_dir → A3 container path
    # ONCE here at the entry point (per DS convention codifier 19:57Z).
    # Runner analysis uses the translated immutable stage.  The source-package
    # fallback separately translates the original ops-nn path because only its
    # repository parents contain build.sh.  The local `op_dir` stays the source
    # of truth for snapshot validation and prior-art inventory on this host.
    op_dir_a3 = _translate_to_a3_container_path(op_dir)
    source_build_op_dir_a3 = _translate_to_a3_container_path(
        source_build_op_dir if source_build_op_dir is not None else op_dir
    )

    # Inventory provenance-tracked prior art before generation. This is
    # advisory context only; it never replaces the live arch22 source capture.
    _run_prior_art_scan(op_dir, workspace)

    # DEBT-165 port-ENTRY gate (architecture-based, the load-bearing fix). Reject a
    # candidate whose SOURCE-arch kernel is a pure `#include "<target-arch>/"` dispatch
    # shell — no source-arch algorithm to extract (deformable_offsets exemplar: a
    # 39-line <op>_apt.cpp that includes only arch35/). Rejecting HERE, before build/exec,
    # roots out the whole "FA difficulty" class BEFORE generation; the finalize
    # arch35-wrap / GE_OPHOST / copy-shape gates are the belt for ACTUAL copies.
    _src_complete, _src_reason = _source_arch_complete_gate(op_dir)
    if not _src_complete:
        log.info(f"[phase_o25_a3_ref] port-ENTRY gate REJECT (DEBT-165): {_src_reason}")
        rep.verdict = "SOURCE_ARCH_INCOMPLETE"
        rep.errors.append(_src_reason)
        rep.summary = (
            f"port-entry gate (DEBT-165): '{logical_op_name}' has NO source-arch algorithm "
            f"to extract — pure target-arch dispatch shell. Porting it would copy the "
            f"target-arch vendor answer (which the customer lacks) = not a real port."
        )
        _write_a3_reference_runnable_json(workspace, rep)
        return rep

    if probe_only:
        rep.verdict = "LIVE_CAPTURE_REQUIRED"
        rep.errors.append(
            "probe-only reference provisioning is disabled; a fresh live arch22 "
            "capture is mandatory"
        )
        rep.summary = "reachability alone is not migration truth"
        _write_a3_reference_runnable_json(workspace, rep)
        return rep

    # Step 0.5 (apply_adam_w_quant aftermath, 2026-05-14T17:08Z): CPU-truth
    # ground-truth scaffolding. User direction: "如果你100%确认，你应该把这个
    # 机制固化到我们的pipeline的前期分析中". Without a CPU fp64 ground truth,
    # "precision alignment" against A3-NPU-output is meaningless when the
    # upstream-reference itself may be buggy. apply_adam_w_quant pp-3
    # H12/H13 hypothesis ("A3 drops m_old") was investigated for $26 of
    # probe spawns; would have been resolved in 1 CPU-truth measurement.
    # This step emits `cpu_truth_template.md` with the algorithm spec
    # discovery + a sample skeleton. kw is then expected to author
    # `cpu_truth_reference.py` (per-op fp64 oracle) in Phase A.5.
    _emit_cpu_truth_scaffold(op_dir, workspace)

    # Generate workspace/edge_inputs.pt before live arch22 capture so the runner
    # receives a deterministic fixture. Input generation never substitutes for
    # the mandatory live reference result. See `ensure_edge_inputs` for the
    # preference order (existing fixture / input_gen.py / op_def signature).
    _edge_ok, _edge_msg = ensure_edge_inputs(op_dir, workspace)
    rep.summary = (rep.summary + "; " if rep.summary else "") + f"edge_inputs: {_edge_msg}"

    # Step 0.75 (item①, DEBT-199): model.py↔input_gen CONTRACT validation. Run model.forward on ONE
    # sample case BEFORE cpu_truth/native provisioning and FAIL LOUD with the SPECIFIC mismatch
    # (NPU-delegation / signature-mismatch / output-shape) so the 生态 golden path can never silently
    # produce no/wrong golden. Best-effort (never raises); records .model_contract_validation.json +,
    # on failure, a prominent error + actionable recommendation on the report.
    try:
        _contract = validate_model_contract(workspace)
        try:
            (workspace / ".model_contract_validation.json").write_text(json.dumps({
                "ok": _contract.ok,
                "reason_code": _contract.reason_code,
                "message": _contract.message,
                "detail": _contract.detail,
                "validated_ts": datetime.now(timezone.utc).isoformat(),
            }, indent=2, ensure_ascii=False))
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
        rep.summary = (rep.summary + "; " if rep.summary else "") + \
            f"model_contract: {_contract.reason_code}"
        if not _contract.ok:
            log.error(f"[phase_o25_a3_ref] model.py contract validation FAILED "
                      f"({_contract.reason_code}): {_contract.message}")
            rep.errors.append(f"model.py contract ({_contract.reason_code}): {_contract.message}")
            rep.recommendations.append(_CONTRACT_FIX_HINT)
        elif _contract.detail.get("fp64_downcast_warning"):
            log.info(f"[phase_o25_a3_ref] model.py contract WARN: "
                     f"{_contract.detail['fp64_downcast_warning']}")
    except Exception as e:  # never let validation crash O2.5
        log.info(f"[phase_o25_a3_ref] model contract validation WARN (non-fatal): {e!r}")

    # Step 0.8 (② PROVIDER-AUTO native_capture, owner 2026-06-30): auto-emit native_capture.pt =
    # model.py.Model.forward @ native dtype on CPU over edge_inputs.pt — the CPU-same-precision
    # baseline the 生态 compare.py forward grader needs for its small-value carve-out. HARNESS-run
    # (not worker-authored) so EVERY forward port_a3 op gets it; without it native=None → the fp32
    # near-zero false-FAIL silently returns. Runs here (pure CPU, no A3) so it's independent of A3
    # reachability / probe_only. Best-effort: failure leaves native=None (stricter, safe).
    _nat_ok, _nat_msg = provision_native_capture(workspace)
    rep.summary = (rep.summary + "; " if rep.summary else "") + f"native_capture: {_nat_msg}"

    # DEBT-199 (2026-07-03): ALSO auto-provision the 生态 T1 golden cpu_truth_outputs.pt (fp64 CPU
    # model.forward). Without it, when A3 capture SUCCEEDS the canonical grader has no cpu_truth → falls
    # to the worker's vs-A3 pass_a_runner.py (T3-axis) = the precision-standard drift. HARNESS-run so
    # every CPU-runnable port_a3 op grades T1=ours-vs-cpu_truth (A3 stays T2 competitor). Best-effort:
    # a genuinely NPU-only op leaves cpu_truth absent (grader handles it), fail-closed.
    _cput_ok, _cput_msg = provision_cpu_truth(workspace)
    rep.summary = (rep.summary + "; " if rep.summary else "") + f"cpu_truth: {_cput_msg}"

    # Step 1: pure helpers
    rep.aclnn_entry = derive_aclnn_entry(op_dir)
    if rep.aclnn_entry is None:
        rep.verdict = "MISSING_ENTRY"
        rep.errors.append(
            f"no aclnn entry point in {op_dir} (looked for "
            f"examples/test_aclnn_<op>.cpp, pytorch/<op>.asc, "
            f"tests/ut/op_host/test_aclnn_<op>.cpp)"
        )
        rep.recommendations.append(
            "for ops without an aclnn entry, the A3 reference must come from "
            "another harness (the op may not be aclnn-callable; check op_def.cpp "
            "for the registration pattern)"
        )
        _write_a3_reference_runnable_json(workspace, rep)
        return rep

    rep.gen_data_source = derive_input_gen_source(op_dir)
    rep.peer_op_dependencies = derive_op_dependencies(op_dir)

    # Step 2: probe A3 reachability
    reachable, msg = probe_a3_reachable(
        a3_user, a3_host, a3_container, run_remote=run_remote,
    )
    if not reachable:
        rep.verdict = "A3_UNREACHABLE"
        rep.errors.append(msg)
        rep.recommendations.append(
            f"check ssh {a3_user}@{a3_host} works + container {a3_container} "
            f"is running (docker ps); see workspace/.ascendc_env A3_* fields"
        )
        _write_a3_reference_runnable_json(workspace, rep)
        return rep

    # Step 3.5: NPU lane pick + busy gate (2026-05-13).
    # When A3_NPU_RANGE is set, auto-pick the idlest chip from the range
    # (supports parallel orchestrator instances each binding a distinct chip).
    # Otherwise fall back to A3_DEFAULT_NPU_ID single-chip behavior.
    # In either case, refuse capture if ALL candidates are above threshold —
    # contaminated baseline would inflate A5/A3 ratio.
    a3_npu_candidates, a3_busy_threshold = _read_a3_npu_range_config(workspace)
    # If caller provided an explicit picker, use it (test injection); else
    # if the legacy npu_busy_check is provided (older test fixtures), wrap
    # it to return the first candidate as idle (matches pre-range semantic);
    # else fall back to live npu-smi picker.
    if npu_picker is not None:
        _picker = npu_picker
    elif npu_busy_check is not None:
        def _picker(_au, _ah, cands, _th, **_kw):
            busy, pct, busy_log = npu_busy_check(_au, _ah, cands[0], _th, **_kw)
            return (None if busy else cands[0]), {cands[0]: pct}, busy_log
    else:
        _picker = pick_idle_npu_in_range
    chosen_npu_id, observed_pcts, npu_smi_log = _picker(
        a3_user, a3_host, a3_npu_candidates, a3_busy_threshold,
        run_remote=run_remote,
    )
    if chosen_npu_id is None:
        rep.verdict = "A3_BUSY"
        rep.errors.append(
            f"A3 NPU pick failed: all candidates {a3_npu_candidates} above "
            f"threshold={a3_busy_threshold}%. observed={observed_pcts}"
        )
        rep.recommendations.append(
            f"wait for workloads to finish (re-run after `npu-smi info` shows "
            f"AICore<{a3_busy_threshold}% on at least one of {a3_npu_candidates}), "
            f"or widen A3_NPU_RANGE, or raise A3_AICORE_BUSY_THRESHOLD."
        )
        rep.summary = f"A3 NPU pick failed (all of {a3_npu_candidates} busy)"
        _write_a3_reference_runnable_json(workspace, rep)
        return rep
    # Store chosen chip for downstream runner cmd
    rep.summary = f"A3 NPU picked: chip {chosen_npu_id} (range={a3_npu_candidates}, observed={observed_pcts})"

    # Step 4: full live mode (W14b — build + exec + capture outputs).
    if run_remote is None:
        run_remote = _default_run_remote(a3_user, a3_host, a3_container)

    # P128 (2026-05-17): for ops with shipped aclnn registration (e.g.
    # `aclnnForeachReciprocal` in installed CANN 9.0.0), the runner uses
    # `torch._foreach_<op>(...)` via torch_npu dispatch — no source build
    # needed. The earlier `build_a3_reference_kernel` step is only useful
    # for ops NOT yet in shipped CANN (newly upstreamed ops). The runner
    # never loads the build output `.so` (it uses torch_npu's dispatch),
    # so for shipped ops the build was wasted work.
    #
    # New order: locate/author runner FIRST → try executing → only fall
    # back to build_a3_reference_kernel if the runner fails with a
    # "symbol not found" / "op not registered" pattern. User caught
    # 2026-05-17 ~00:30Z: "arch22 to arch35 需要在 a3 编译 cann 代码么？不是直接用
    # pytorch 调用的么？" — correct, for shipped ops we just dispatch.

    # Step 4b: locate the per-op aclnn runner script. For W14b we expect
    # the caller to have authored workspace/run_a3_reference.py — this is
    # a small per-op script that loads edge_inputs.pt, calls aclnnX for
    # each case, writes outputs + timings. Generic harness can't author
    # this because aclnn signatures differ per op.
    runner_script = workspace / "run_a3_reference.py"
    if not runner_script.is_file():
        # Gap-A fix (2026-05-13): try the aog-a3-author skill to auto-generate
        # the runner + input_gen + manifest. Triggers ONLY when
        # opgen_mode==port_a3_to_a5 AND runner missing (the should_trigger
        # gate enforces both). If skill succeeds, we proceed; if not, fall
        # through to RUNNER_MISSING below so caller sees the exact failure.
        author_attempted, author_result = _try_a3_author(workspace, op_dir)
        if author_attempted and author_result.get("success"):
            log.info(
                f"[phase_o25_a3_ref] aog-a3-author skill authored "
                f"run_a3_reference.py + input_gen.py "
                f"(verdict={author_result.get('verdict')})"
            )
            # Refresh — skill should have written it
        elif author_attempted:
            rep.verdict = "RUNNER_MISSING"
            rep.errors.append(
                f"per-op A3 runner missing AND aog-a3-author skill failed: "
                f"verdict={author_result.get('verdict')} "
                f"errors={author_result.get('errors')}"
            )
            rep.recommendations.append(
                "inspect workspace/.a3_author_log.jsonl for skill failure detail; "
                "fix the upstream test_aclnn_<op>.cpp parsing issue OR hand-author "
                "run_a3_reference.py as a one-off."
            )
            _write_a3_reference_runnable_json(workspace, rep)
            return rep
        # else: author_attempted=False because should_trigger said no
        # (wrong mode etc.) — fall through to RUNNER_MISSING.

        if not runner_script.is_file():
            rep.verdict = "RUNNER_MISSING"
            rep.errors.append(
                f"per-op A3 runner missing: {runner_script}. "
                f"Author a script that loads edge_inputs.pt + invokes the aclnn API "
                f"for {logical_op_name} + writes a3_outputs + a3_baseline_perf.json."
            )
            rep.recommendations.append(
                "for ctc_loss_v3 PoC, the runner template lives in "
                "workspace/<op>/run_a3_reference.py and is hand-authored per W14c."
            )
            _write_a3_reference_runnable_json(workspace, rep)
            return rep

    # Step 4c: stage inputs + run + capture (with auto-picked NPU id from §3.5).
    # P124: pass translated container path, NOT local path.
    # P128: try runner FIRST. For shipped aclnn ops the runner succeeds via
    # torch_npu dispatch without any source build. Only fall back to build
    # if runner fails with a symbol-not-found / unregistered-op signature.
    rep.capture_id = uuid.uuid4().hex
    capture_started_ts = datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    rep.a3_exec_attempted = True
    exec_ok, exec_msg = run_a3_reference(
        op_dir_a3, workspace, a3_user, a3_host, a3_container, a3_cann_path,
        runner_script=runner_script, run_remote=run_remote,
        npu_id=chosen_npu_id,
    )
    if not exec_ok:
        # P128: detect "op not registered in shipped CANN" signatures —
        # only then do we actually need to build from ops-nn source.
        _unshipped_signatures = (
            "NotImplementedError",
            "could not find an implementation",
            "Could not run",  # PyTorch dispatch missing
            "aclnn",  # generic symbol-resolution failures
            "undefined symbol",
            "561103",  # source kernel binary is not registered in shipped OPP
            "no binary",
            "kernel not registered",
            "kernel-not-registered",
        )
        exec_msg_folded = exec_msg.casefold()
        looks_unshipped = any(
            signature.casefold() in exec_msg_folded
            for signature in _unshipped_signatures
        )
        if looks_unshipped:
            log.info(
                "[phase_o25_a3_ref] runner failed with unshipped-op signature; "
                "falling back to build_a3_reference_kernel"
            )
            # The package must never be installed into the system OPP tree.
            # Keep it outside the scp-replaced per-op workspace (a retry pushes
            # that directory again), but bind it to this workspace + capture id
            # so concurrent runs cannot overwrite or silently reuse each other.
            custom_opp_install_path = (
                f"{_a3_container_home().rstrip('/')}"
                f"/workspace/a5_ops_a3_to_a5/custom_opp/"
                f"{workspace.name}/{rep.capture_id}"
            )
            custom_opp_env_script = (
                f"{custom_opp_install_path}/vendors/custom_nn/"
                f"bin/set_env.bash"
            )
            build_ok, build_log = build_a3_reference_kernel(
                source_build_op_dir_a3, a3_cann_path, run_remote=run_remote,
                logical_op_name=logical_op_name,
                custom_opp_install_path=custom_opp_install_path,
            )
            if not build_ok:
                rep.verdict = "BUILD_FAILED"
                rep.errors.append(
                    f"runner failed AND build failed; runner err:\n{exec_msg[-300:]}\n\n"
                    f"build err:\n{build_log[-500:]}"
                )
                _write_a3_reference_runnable_json(workspace, rep)
                return rep
            # Retry with only this run's isolated custom OPP activated.  Merely
            # building the package does not register its kernel (561103).
            if not _quarantine_capture("pre-retry", include_runnable=False):
                _write_a3_reference_runnable_json(workspace, rep)
                return rep
            rep.capture_id = uuid.uuid4().hex
            capture_started_ts = datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            )
            exec_ok, exec_msg = run_a3_reference(
                op_dir_a3, workspace, a3_user, a3_host, a3_container, a3_cann_path,
                runner_script=runner_script, run_remote=run_remote,
                npu_id=chosen_npu_id,
                custom_opp_env_script=custom_opp_env_script,
            )
    if not exec_ok:
        rep.verdict = "EXEC_FAILED"
        rep.errors.append(exec_msg)
        _write_a3_reference_runnable_json(workspace, rep)
        return rep

    # Success — paths should exist now. task#25: exec_ok=True means the runner
    # returned 0, but that does NOT guarantee it populated outputs for every case
    # (§6: fused_quant_mat_mul exec returned ok yet captured 0/73 + empty perf).
    # Validate captured CONTENT before declaring READY; previously this set
    # verdict=READY unconditionally while its own summary said 'outputs MISSING'.
    a3_outputs = workspace / "edge_dataset.pt"
    a3_perf = workspace / "a3_baseline_perf.json"
    if a3_outputs.is_file():
        rep.a3_outputs_path = a3_outputs
    if a3_perf.is_file():
        rep.a3_perf_path = a3_perf
    _cap_ok, _cap_reason = _validate_a3_capture(workspace)
    if _cap_ok:
        provenance_ok, provenance_reason, provenance_path = (
            write_a3_capture_provenance(
                workspace,
                capture_id=rep.capture_id or "",
                capture_started_ts=capture_started_ts,
                npu_id=chosen_npu_id,
            )
        )
        if not provenance_ok:
            rep.verdict = "CAPTURE_INCOMPLETE"
            rep.errors.append(
                f"live A3 capture provenance invalid: {provenance_reason}"
            )
            rep.summary = (
                f"aclnn={rep.aclnn_entry.name}; live A3 exec ran but "
                "provenance binding failed"
            )
            _write_a3_reference_runnable_json(workspace, rep)
            return rep
        rep.capture_manifest_path = provenance_path
        rep.verdict = "READY"
        rep.summary = (
            f"aclnn={rep.aclnn_entry.name}; {provenance_reason}; {_cap_reason}"
        )
        for marker in stale_fallback_markers:
            try:
                marker.unlink(missing_ok=True)
            except OSError:
                pass
    else:
        rep.verdict = "CAPTURE_INCOMPLETE"
        rep.errors.append(
            f"A3 exec returned success but capture invalid: {_cap_reason}"
        )
        rep.recommendations.append(
            "the per-op run_a3_reference.py exited 0 but did not populate "
            "a3_outputs/perf for all cases — inspect its per-case a3_error fields. "
            "Do NOT fall back to CPU-truth for quant ops (misleading fp32 oracle)."
        )
        rep.summary = f"aclnn={rep.aclnn_entry.name}; A3 exec ran but {_cap_reason}"
    _write_a3_reference_runnable_json(workspace, rep)
    return rep
