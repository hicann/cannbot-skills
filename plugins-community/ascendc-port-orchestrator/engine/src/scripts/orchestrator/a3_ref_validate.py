# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Phase O2.5 A3-CANN reference — CPU-truth capture + model-contract validation (decomposed 2026-07-06).

Cohesive LEAF: synthesize/native/CPU-truth dataset capture, plus the model.py↔
input_gen contract validator (NPU-delegation static scan, kwargs/output-shape
diagnosis) and the a3-capture completeness validator + ref_runnable.json writer.
Imports stdlib (+ call-time torch/importlib/subprocess/datetime inside functions)
and the shared dataclasses/helpers from a3_ref_common. NEVER imports from the
phase_o25_a3_ref facade (unidirectional edge, no cycle).
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from a3_ref_common import (
    ModelContractResult,
    O25A3Report,
    _case_model_kwargs,
    _coerce_case_list,
    log,
)


def _remove_stale_capture(
    out_path: Path, capture_name: str, failure_template: str
) -> str | None:
    """Delete an old harness artifact or return the existing fail-closed message."""
    try:
        out_path.unlink()
    except FileNotFoundError:
        return None
    except OSError as error:
        log.warning(
            "%s: could not remove stale %s: %r — treating as provisioning failure",
            capture_name,
            out_path,
            error,
        )
        return failure_template.format(error=error)
    return None


def _try_generate_edge_inputs(
    workspace: Path, edge_inputs: Path, python_executable: str
) -> None:
    """Run an input generator if and only if its expected file is absent."""
    if edge_inputs.is_file() or not (workspace / "input_gen.py").is_file():
        return
    try:
        import subprocess as _sp

        _sp.call(
            [python_executable, "input_gen.py"],
            cwd=str(workspace),
            stdout=_sp.DEVNULL,
            stderr=_sp.DEVNULL,
        )
    except Exception as error:
        logging.getLogger(__name__).debug(
            "Recoverable operation failed.", exc_info=error
        )


def _read_capture_cases(torch, edge_inputs: Path, capture_name: str):
    """Load the accepted edge-input schemas, retaining their established diagnostics."""
    try:
        blob = torch.load(edge_inputs, weights_only=False)  # type: ignore[arg-type]
    except Exception as error:
        return None, f"{capture_name}: torch.load(edge_inputs.pt) failed: {error!r}"
    if isinstance(blob, list):
        cases = blob
    elif isinstance(blob, dict) and "cases" in blob:
        cases = blob["cases"]
    elif isinstance(blob, dict) and "inputs" in blob:
        cases = blob["inputs"]
    else:
        return None, (
            f"{capture_name}: edge_inputs.pt unrecognized shape "
            f"({type(blob).__name__})"
        )
    if not cases:
        return None, f"{capture_name}: edge_inputs.pt cases list is empty"
    return cases, None


def _load_capture_context(
    workspace: Path,
    capture_name: str,
    module_prefix: str,
    missing_inputs_template: str,
    python_executable: str,
):
    """Load the model instance and its cases for either harness-owned capture."""
    edge_inputs = workspace / "edge_inputs.pt"
    model_py = workspace / "model.py"
    _try_generate_edge_inputs(workspace, edge_inputs, python_executable)
    if not edge_inputs.is_file():
        return None, missing_inputs_template.format(edge_inputs=edge_inputs)
    if not model_py.is_file():
        return None, f"{capture_name}: model.py missing at {model_py}"
    try:
        import torch  # type: ignore
    except Exception as error:
        return None, f"{capture_name}: torch import failed: {error!r}"
    import importlib.util

    try:
        spec = importlib.util.spec_from_file_location(
            f"{module_prefix}_{workspace.name}", str(model_py)
        )
        if spec is None or spec.loader is None:
            return None, f"{capture_name}: importlib couldn't build spec for {model_py}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception as error:
        return None, f"{capture_name}: model.py exec_module failed: {error!r}"
    if not hasattr(module, "Model"):
        return None, f"{capture_name}: model.py does not define `Model`"
    cases, error = _read_capture_cases(torch, edge_inputs, capture_name)
    if error is not None:
        return None, error
    try:
        model_inst = module.Model()  # type: ignore[attr-defined]
    except Exception as error:
        return None, f"{capture_name}: Model() instantiation failed: {error!r}"
    return (torch, model_inst, cases), None


def _invoke_model(model_inst, case_kwargs, transform):
    """Dispatch one case through a model using its preserved input contract."""
    if isinstance(case_kwargs, dict):
        return model_inst(**{key: transform(value) for key, value in case_kwargs.items()})
    if isinstance(case_kwargs, (list, tuple)):
        return model_inst(*[transform(value) for value in case_kwargs])
    return model_inst(transform(case_kwargs))


def _native_dtype(torch, case):
    """Infer a case's native dtype from its first floating tensor."""
    values = case.values() if isinstance(case, dict) else (
        case if isinstance(case, (list, tuple)) else [case]
    )
    for value in values:
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.dtype
    return torch.float32


def _collect_native_outputs(torch, model_inst, cases):
    """Capture all native outputs, including the existing fp32 fallback behavior."""
    outputs, dtypes, failures = [], set(), []
    n_fp32_fallback = 0
    for index, case in enumerate(cases):
        try:
            case_kwargs = _case_model_kwargs(case)
            native_dtype = _native_dtype(torch, case_kwargs)

            def cast(value, dtype=native_dtype):
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    return value.to(dtype)
                return value

            used_fallback = False
            try:
                output = _invoke_model(model_inst, case_kwargs, cast)
            except Exception:
                output = _invoke_model(
                    model_inst, case_kwargs, lambda value: cast(value, torch.float32)
                )
                used_fallback = True
            if (
                isinstance(output, torch.Tensor)
                and output.is_floating_point()
                and output.dtype != native_dtype
            ):
                output = output.to(native_dtype)
            outputs.append(output)
            n_fp32_fallback += int(used_fallback)
            if isinstance(output, torch.Tensor):
                dtypes.add(str(output.dtype).replace("torch.", ""))
        except Exception as error:
            failures.append(f"case {index}: {error!r}")
            outputs.append(None)
    return outputs, dtypes, failures, n_fp32_fallback


def _collect_fp64_outputs(torch, model_inst, cases):
    """Capture every CPU golden at fp64, retaining the all-or-nothing result."""
    def cast(value):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.to(torch.float64)
        return value

    outputs, failures = [], []
    for index, case in enumerate(cases):
        try:
            outputs.append(_invoke_model(model_inst, _case_model_kwargs(case), cast))
        except Exception as error:
            failures.append(f"case {index}: {error!r}")
            outputs.append(None)
    return outputs, failures


def provision_native_capture(workspace: Path) -> tuple[bool, str]:
    """② (owner-directed 2026-06-30) — PROVIDER-AUTO native_capture.pt (the load-bearing fix).

    Auto-generates `workspace/native_capture.pt` = `model.py.Model.forward()` re-run at the op's
    NATIVE dtype on CPU over `edge_inputs.pt`. This is the CPU-SAME-PRECISION baseline the 生态
    compare.py forward grader (precision_eval_port_a3_two_tier.load_and_classify) needs for its
    small-value/cancellation carve-out. It is HARNESS-run (NOT worker-authored), so every forward
    port_a3 op gets it automatically — WITHOUT it native=None → compare.py falls to its stricter
    baseline and the fp32 near-zero false-FAIL silently returns (the whole fix no-ops).

    KEY: it is a REAL native-dtype REFERENCE RUN (model.forward over the native-dtype inputs), NOT a
    cast of the fp64 cpu_truth. Uses the SAME generic model.forward-over-edge_inputs mechanism as
    the shared model-forward case format. Shape-aligned: one output entry per case.

    Stored as {"native_kind":"cpu_same_precision","dtype":<str>,"outputs":[per-case native tensor]}
    (load_and_classify's _to_tensor_list reads the 'outputs' key).

    Returns (success, message). Best-effort: on failure leaves NO native_capture.pt (fail-closed).
    """
    out_path = workspace / "native_capture.pt"
    # ① STALE-DELETE (codex01 P1/P3, blocking): remove any pre-existing (worker/stale) native_capture.pt
    # BEFORE we start, so on ANY failure path below native_capture.pt is ABSENT — never a stale file
    # that O5 (now local-wins via FORCE_UPDATE_SCRIPTS) would FORCE-PUSH to the remote as authoritative.
    # A fresh one is written ONLY on full success (all-or-nothing, P3).
    stale_error = _remove_stale_capture(
        out_path,
        "native_capture",
        "native_capture stale-unlink failed (stale native may survive): {error!r}",
    )
    if stale_error is not None:
        return False, stale_error
    # P1 (codex01): input_gen.py ops have NO edge_inputs.pt at O2.5 Step 0.8 time (ensure_edge_inputs
    # leaves generation to the input_gen path). Run input_gen.py here so native IS produced for the
    # input_gen class too — else it falls back to native=None and the fp32 near-zero false-FAIL
    # silently returns for those ops. (_run_a3_reference_remote ALSO retries native after it runs
    # input_gen.py, as a belt for the probe_only→live transition.)
    import sys as _sys
    context, error = _load_capture_context(
        workspace,
        "native_capture",
        "model_native_for",
        (
            "native_capture: edge_inputs.pt missing at {edge_inputs} "
            "(no input_gen.py present, or it did not produce edge_inputs.pt)"
        ),
        _sys.executable,
    )
    if error is not None:
        return False, error
    torch, model_inst, cases = context
    outputs, dtypes, failures, n_fp32_fallback = _collect_native_outputs(
        torch, model_inst, cases
    )

    # P3 FAIL-CLOSED (codex01): a partial native (any case = None) must NOT be written + treated as
    # provisioned — that is the exact silent native=None the whole fix guards against. Require ALL cases
    # to produce native; otherwise write NOTHING → native is ABSENT → the ① grader falls to its strict
    # baseline → the near-zero cases FAIL VISIBLY (loud), never a silent pass-through.
    n_missing = sum(1 for o in outputs if o is None)
    if n_missing:
        return False, (f"native_capture FAIL-CLOSED: {n_missing}/{len(cases)} case(s) produced no native "
                       f"(model.forward raised); NOT writing a partial native_capture.pt (would "
                       f"masquerade as provisioned → silent near-zero pass). first 3: {failures[:3]}")

    # P2 provenance (codex01): the fp32 fallback is NOT a true native-dtype run — tag it honestly so the
    # grader/audit is never misled. cpu_same_precision ONLY when EVERY case ran at the true native dtype.
    native_kind = "cpu_same_precision" if n_fp32_fallback == 0 else "cpu_fp32_fallback"
    provenance = ("provider-auto: model.py.Model.forward @ native dtype on CPU over edge_inputs.pt "
                  "(NOT a cast of fp64 cpu_truth)" if n_fp32_fallback == 0 else
                  f"provider-auto: model.py.Model.forward — {n_fp32_fallback}/{len(cases)} case(s) fell "
                  f"back to fp32-compute then downcast to native (CPU native-dtype unsupported); tagged "
                  f"cpu_fp32_fallback (NOT a strict same-precision run, NOT a cast of fp64 truth)")
    blob = {
        "native_kind": native_kind,
        "dtype": sorted(dtypes)[0] if dtypes else "float32",
        "n_fp32_fallback": n_fp32_fallback,
        "provenance": provenance,
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "outputs": outputs,
    }
    try:
        torch.save(blob, out_path)
    except Exception as e:
        return False, f"native_capture: torch.save failed: {e!r}"
    msg = (f"native_capture: AUTO-emitted {len(cases)}/{len(cases)} cases @ native dtype {sorted(dtypes)} "
           f"(native_kind={native_kind}"
           + (f", {n_fp32_fallback} fp32-fallback" if n_fp32_fallback else "") + f") → {out_path.name}")
    return True, msg


def provision_cpu_truth(workspace: Path) -> tuple[bool, str]:
    """DEBT-199 (2026-07-03) — PROVIDER-AUTO cpu_truth_outputs.pt (the 生态 T1 golden).

    Auto-generates `workspace/cpu_truth_outputs.pt` = `model.py.Model.forward()` re-run at **FP64**
    on CPU over `edge_inputs.pt`. This is the 生态 golden (owner 2026-06-30, cann-bench compare.py)
    that the canonical grader (precision_eval_port_a3_two_tier.load_and_classify:426) needs to compute
    T1 = ours-vs-cpu_truth. HARNESS-run (NOT worker-authored), so every CPU-runnable port_a3 op gets
    it automatically — mirroring provision_native_capture.

    This supplemental oracle is emitted separately from the mandatory live arch22 capture. It never
    fills the ``a3_outputs`` slot and never authorizes the migration to continue when live capture
    fails. Absent a separate cpu_truth, the canonical grader fails visibly on the missing T1 golden.

    DISTINCTIONS from provision_native_capture (a DIFFERENT, complementary artifact — both needed):
      - computes at FP64 (the golden), NOT the op's native dtype (native_capture.pt = the carve-out
        baseline). fp64 is always CPU-supported → no fp32-fallback branch.
      - writes cpu_truth_outputs.pt (SEPARATE file); does NOT touch edge_dataset.pt's real a3_outputs
        (A3 stays the T2 competitor: ours_MARE ≤ a3_MARE, both vs this cpu_truth).

    FAIL-CLOSED (mirrors provision_native_capture P3): if ANY case's model.forward raises, write
    NOTHING → cpu_truth ABSENT → the canonical grader FAILs visibly (never a silent partial pass).
    A genuinely NPU-only op (model.py can't run on CPU) thus legitimately leaves cpu_truth absent.

    Stored {"golden_kind":"cpu_fp64","dtype":"float64","provenance":<str>,"generated_ts":<iso>,
    "outputs":[per-case fp64 tensor]} (load_and_classify's _to_tensor_list reads the 'outputs' key).
    Returns (success, message). Best-effort: on failure leaves NO cpu_truth_outputs.pt.
    """
    out_path = workspace / "cpu_truth_outputs.pt"
    # STALE-DELETE first (all-or-nothing, mirrors provision_native_capture): a stale cpu_truth must
    # never survive a failure path (O5 local-wins would force-push it as authoritative).
    stale_error = _remove_stale_capture(
        out_path,
        "cpu_truth",
        "cpu_truth stale-unlink failed (stale may survive): {error!r}",
    )
    if stale_error is not None:
        return False, stale_error
    context, error = _load_capture_context(
        workspace,
        "cpu_truth",
        "model_cputruth_for",
        "cpu_truth: edge_inputs.pt missing at {edge_inputs}",
        "python3",
    )
    if error is not None:
        return False, error
    torch, model_inst, cases = context
    outputs, failures = _collect_fp64_outputs(torch, model_inst, cases)

    # FAIL-CLOSED (mirrors provision_native_capture P3): all cases must produce a golden; else write
    # NOTHING → cpu_truth ABSENT → canonical grader FAILs visibly, never a silent partial pass.
    n_missing = sum(1 for o in outputs if o is None)
    if n_missing:
        return False, (f"cpu_truth FAIL-CLOSED: {n_missing}/{len(cases)} case(s) produced no fp64 golden "
                       f"(model.forward raised — likely NPU-only op); NOT writing a partial "
                       f"cpu_truth_outputs.pt. first 3: {failures[:3]}")

    blob = {
        "golden_kind": "cpu_fp64",
        "dtype": "float64",
        "provenance": ("provider-auto (DEBT-199): model.py.Model.forward @ fp64 on CPU over "
                       "edge_inputs.pt — the 生态 T1 golden (owner 2026-06-30 cann-bench compare.py)"),
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "outputs": outputs,
    }
    try:
        torch.save(blob, out_path)
    except Exception as e:
        return False, f"cpu_truth: torch.save failed: {e!r}"
    return True, (f"cpu_truth: AUTO-emitted {len(cases)}/{len(cases)} cases @ fp64 → {out_path.name} "
                  "(生态 T1 golden)")


# ---------------------------------------------------------------------------
# model.py <-> input_gen contract validation
# ---------------------------------------------------------------------------
_LABEL_TORCH_NPU = "torch_npu"
_LABEL_NPU_CALL = "npu_*() call (e.g. npu_fusion_attention)"
_LABEL_DOT_NPU = ".npu() device move"
_LABEL_IMPORT_TORCH_NPU = "import torch_npu"

_NPU_DELEGATION_PATTERNS = [
    (re.compile(r"\btorch_npu\b"), _LABEL_TORCH_NPU),
    (re.compile(r"\bnpu_[A-Za-z0-9_]+\s*\("), _LABEL_NPU_CALL),
    (re.compile(r"\.npu\s*\("), _LABEL_DOT_NPU),
    (re.compile(r"\bimport\s+torch_npu\b"), _LABEL_IMPORT_TORCH_NPU),
]

_CONTRACT_FIX_HINT = (
    "FIX model.py: it is the CPU fp64 REFERENCE (生态 T1 golden) and MUST be "
    "(a) CPU-PURE — no torch_npu / npu_* / .npu(); use plain torch CPU math — and "
    "(b) accept the input_gen key contract — either a module-level get_input_groups() "
    "returning positional arg-lists, OR a Model.forward that accepts the emitted keys "
    "(add **kwargs, or rename params to match, e.g. input_layout not layout)."
)


def _strip_py_comments(src: str) -> str:
    """Remove `#` line comments (keeps string content roughly intact enough for a
    conservative token scan — we only need to avoid matching commented-out code)."""
    out_lines = []
    for line in src.splitlines():
        h = line.find("#")
        out_lines.append(line if h < 0 else line[:h])
    return "\n".join(out_lines)


def _strip_strings_and_comments(src: str) -> str:
    """Best-effort removal of triple-quoted docstrings, single-line string literals, and
    `#` comments. ONLY used as the fallback when model.py fails to AST-parse — the
    primary detector is AST-based. Regex-level, so imperfect on pathological nesting, but
    enough to keep a vendor-op NAME inside a docstring/string from tripping the scan."""
    s = re.sub(r'"""(?:.|\n)*?"""', '""', src)
    s = re.sub(r"'''(?:.|\n)*?'''", "''", s)
    s = re.sub(r'"(?:\\.|[^"\\\n])*"', '""', s)
    s = re.sub(r"'(?:\\.|[^'\\\n])*'", "''", s)
    return _strip_py_comments(s)


def _regex_npu_delegation_hits(src: str) -> list[str]:
    """Return conservative matches for source that cannot be parsed as Python."""
    stripped = _strip_strings_and_comments(src)
    return sorted(
        {label for pattern, label in _NPU_DELEGATION_PATTERNS if pattern.search(stripped)}
    )


def _npu_delegation_labels(node: ast.AST) -> set[str]:
    """Return the delegation labels represented by one parsed AST node."""
    if isinstance(node, ast.Import):
        if any(alias.name.split(".")[0] == "torch_npu" for alias in node.names):
            return {_LABEL_IMPORT_TORCH_NPU, _LABEL_TORCH_NPU}
        return set()
    if isinstance(node, ast.ImportFrom):
        if node.module and node.module.split(".")[0] == "torch_npu":
            return {_LABEL_IMPORT_TORCH_NPU, _LABEL_TORCH_NPU}
        return set()
    if isinstance(node, ast.Name):
        return {_LABEL_TORCH_NPU} if node.id == "torch_npu" else set()
    if not isinstance(node, ast.Call):
        return set()
    if isinstance(node.func, ast.Name) and node.func.id.startswith("npu_"):
        return {_LABEL_NPU_CALL}
    if not isinstance(node.func, ast.Attribute):
        return set()
    if node.func.attr == "npu":
        return {_LABEL_DOT_NPU}
    return {_LABEL_NPU_CALL} if node.func.attr.startswith("npu_") else set()


def _detect_npu_delegation(src: str) -> list[str]:
    """Return the sorted NPU-delegation labels that appear as REAL code in `src` — an
    actual `import torch_npu`, a `npu_*(...)` / `torch_npu.*(...)` call, or an `x.npu()`
    method move. Names occurring only inside docstrings / string literals / comments do
    NOT trip it (that was the archived-FA false-positive: a pure-CPU torch-SDPA model.py
    whose MODULE DOCSTRING mentioned `npu_fusion_attention`).

    AST-based; falls back to a string-literal + comment-stripped regex scan only if the
    source doesn't parse (SyntaxError)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return _regex_npu_delegation_hits(src)

    hits: set[str] = set()
    for node in ast.walk(tree):
        hits.update(_npu_delegation_labels(node))
    return sorted(hits)


def _sample_from_input_groups(workspace: Path, model_py: Path):
    """Return the first authoritative positional group, if the model exposes one."""
    import importlib.util as _ilu

    try:
        import torch  # noqa: F401

        spec = _ilu.spec_from_file_location(f"model_probe_{workspace.name}", str(model_py))
        if spec is None or spec.loader is None:
            return None
        module = _ilu.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        groups = getattr(module, "get_input_groups", lambda: None)()
        return ("positional", list(groups[0])) if groups else None
    except Exception as error:
        logging.getLogger(__name__).debug(
            "Could not build a sample from get_input_groups; falling back to edge inputs.",
            exc_info=error,
        )
        return None


def _load_first_edge_case(edge_inputs: Path):
    """Load the first edge-input case, retaining the existing schema diagnostics."""
    try:
        import torch

        blob = torch.load(edge_inputs, weights_only=False)
    except Exception as error:
        return None, f"edge_inputs.pt unreadable: {error!r}"
    if isinstance(blob, list):
        cases = blob
    elif isinstance(blob, dict) and "cases" in blob:
        cases = blob["cases"]
    elif isinstance(blob, dict) and "inputs" in blob:
        cases = blob["inputs"]
    else:
        return None, f"edge_inputs.pt unrecognized shape ({type(blob).__name__})"
    if not cases:
        return None, "edge_inputs.pt cases list empty"
    return _case_model_kwargs(cases[0]), None


def _sample_call_form(case):
    """Normalize one edge-input case to the model invocation contract."""
    if isinstance(case, dict):
        return "kwargs", case
    if isinstance(case, (list, tuple)):
        return "positional", list(case)
    return "single", case


def _first_sample_case(workspace: Path):
    """Return ONE sample forward-arg set for contract validation, or (None, reason).

    Preference (coordinator steer 2026-07-03 + get_input_groups() being the op's
    authoritative case→forward-arg constructor):
      1. module-level get_input_groups() → first positional group (list) → ("positional", grp)
      2. edge_inputs.pt first case → _case_model_kwargs unwrap → ("kwargs"|"positional"|"single", args)
    """
    model_py = workspace / "model.py"
    # get_input_groups() path (authoritative arg constructor when present)
    group_sample = _sample_from_input_groups(workspace, model_py)
    if group_sample is not None:
        return group_sample, None
    edge_inputs = workspace / "edge_inputs.pt"
    if not edge_inputs.is_file():
        return None, "no edge_inputs.pt and no usable get_input_groups()"
    case, error = _load_first_edge_case(edge_inputs)
    if error is not None:
        return None, error
    return _sample_call_form(case), None


def _static_contract_result(model_py: Path) -> ModelContractResult | None:
    """Return a terminal static contract result, or None when execution may continue."""
    try:
        source = model_py.read_text(errors="ignore")
    except OSError as error:
        return ModelContractResult(
            True, "SKIPPED_NO_MODEL", f"model.py unreadable: {error!r}"
        )
    hits = _detect_npu_delegation(source)
    if not hits:
        return None
    return ModelContractResult(
        False,
        "NPU_DELEGATION",
        f"model.py delegates to the NPU ({', '.join(hits)}) — it CANNOT produce a "
        f"CPU fp64 golden (this is exactly the archived-FA model.py bug: it called "
        f"npu_fusion_attention instead of pure CPU math). {_CONTRACT_FIX_HINT}",
        {"npu_tokens": hits},
    )


def _load_contract_model(workspace: Path, model_py: Path):
    """Load Model for a contract probe, returning the established terminal error."""
    try:
        import torch  # noqa: F401
        import importlib.util as _ilu

        spec = _ilu.spec_from_file_location(f"model_contract_{workspace.name}", str(model_py))
        if spec is None or spec.loader is None:
            return None, ModelContractResult(
                False, "FORWARD_RAISED", f"importlib couldn't build spec for {model_py}"
            )
        module = _ilu.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception as error:
        return None, ModelContractResult(
            False,
            "FORWARD_RAISED",
            f"model.py exec_module failed: {error!r}. {_CONTRACT_FIX_HINT}",
        )
    if not hasattr(module, "Model"):
        return None, ModelContractResult(
            False,
            "NO_MODEL_CLASS",
            f"model.py does not define a `Model` class. {_CONTRACT_FIX_HINT}",
        )
    try:
        return module.Model(), None  # type: ignore[attr-defined]
    except Exception as error:
        return None, ModelContractResult(
            False,
            "FORWARD_RAISED",
            f"Model() instantiation failed: {error!r}. {_CONTRACT_FIX_HINT}",
        )


def _run_contract_sample(model_inst, call_kind: str, args):
    """Invoke the sampled model call and preserve classified failure results."""
    try:
        if call_kind == "kwargs":
            return model_inst(**args), None
        if call_kind == "positional":
            return model_inst(*args), None
        return model_inst(args), None
    except TypeError as error:
        return None, ModelContractResult(
            False,
            "SIGNATURE_MISMATCH",
            f"model.forward rejected the input_gen contract: {error!r}. {_CONTRACT_FIX_HINT}",
            {"call_kind": call_kind},
        )
    except Exception as error:
        return None, ModelContractResult(
            False,
            "FORWARD_RAISED",
            f"model.forward raised on a sample case: {error!r}. {_CONTRACT_FIX_HINT}",
            {"call_kind": call_kind},
        )


def _fp64_contract_detail(call_kind: str, output) -> dict:
    """Return the non-blocking fp64 soundness detail for a successful probe."""
    detail: dict = {"call_kind": call_kind}
    try:
        import torch

        if (
            isinstance(output, torch.Tensor)
            and output.is_floating_point()
            and output.dtype != torch.float64
        ):
            detail["fp64_downcast_warning"] = (
                f"forward returned {output.dtype} on a sample case; provision_cpu_truth casts "
                f"INPUTS to fp64 but if model.forward downcasts internally (e.g. .to(float32)) "
                f"the golden is NOT truly fp64. Make model.forward honor the input dtype."
            )
    except Exception as error:
        logging.getLogger(__name__).debug("Recoverable operation failed.", exc_info=error)
    return detail


def validate_model_contract(workspace: Path) -> ModelContractResult:
    """Phase-O2.5 model.py↔input_gen contract validation (item①).

    Runs model.forward on ONE sample case and FAILS LOUD with the SPECIFIC mismatch.
    Pure-CPU + best-effort: never raises (returns a result). Called by
    provision_a3_reference before cpu_truth provisioning; also unit-testable directly.
    """
    model_py = workspace / "model.py"
    if not model_py.is_file():
        return ModelContractResult(True, "SKIPPED_NO_MODEL",
                                   "no model.py to validate (non-blocking)")

    # (1) STATIC NPU-delegation scan — model.py can't produce a CPU golden if it
    # delegates to the NPU. Catch it WITHOUT running (fastest, unambiguous signal).
    # AST-based so a vendor-op NAME in a docstring / string / comment is NOT a false hit.
    static_result = _static_contract_result(model_py)
    if static_result is not None:
        return static_result

    # (2) get a sample case to actually exercise forward.
    sample, why = _first_sample_case(workspace)
    if sample is None:
        return ModelContractResult(True, "SKIPPED_NO_INPUTS",
                                   f"no sample case to validate contract: {why}")

    model_inst, load_result = _load_contract_model(workspace, model_py)
    if load_result is not None:
        return load_result

    kind, args = sample
    # (3) For the kwargs form, pre-diff the signature so a mismatch names the EXACT
    # extra/missing/renamed kwargs (better than a bare TypeError).
    if kind == "kwargs":
        sig_issue = _diagnose_kwargs_signature(model_inst, args)
        if sig_issue is not None:
            return sig_issue

    # (4) Actually RUN forward on the sample case.
    out, run_result = _run_contract_sample(model_inst, kind, args)
    if run_result is not None:
        return run_result

    # (5) output-shape cross-check vs a3_outputs (when the A3 capture exists).
    shape_issue = _diagnose_output_shape(workspace, out)
    if shape_issue is not None:
        return shape_issue

    # (6) fp64-golden soundness probe (coordinator note): a genuine fp64 golden needs
    # forward to run at fp64, not compute-then-downcast. If floating fp64 inputs yield a
    # lower-precision floating output, the cpu_truth golden won't be true fp64 — record a
    # WARNING (non-blocking: some ops legitimately return int/bool; don't over-strict-FAIL).
    return ModelContractResult(
        True,
        "OK",
        "model.forward ran on a sample case",
        _fp64_contract_detail(kind, out),
    )


def _diagnose_kwargs_signature(model_inst, case_kwargs: dict):
    """Return a SIGNATURE_MISMATCH ModelContractResult if forward can't accept
    `case_kwargs`, else None. Names the EXACT extra/missing/renamed kwargs."""
    import inspect
    import difflib
    fwd = getattr(model_inst, "forward", None) or getattr(type(model_inst), "__call__", None)
    if fwd is None:
        return None
    try:
        sig = inspect.signature(fwd)
    except (TypeError, ValueError):
        return None
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None  # forward accepts **kwargs → any keys OK.
    named: set[str] = set()
    required: set[str] = set()
    named_parameter_kinds = (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
    for name, parameter in params.items():
        if parameter.kind in named_parameter_kinds and name != "self":
            named.add(name)
        if (
            parameter.default is inspect.Parameter.empty
            and parameter.kind in named_parameter_kinds
            and name != "self"
        ):
            required.add(name)
    case_keys = set(case_kwargs.keys())
    extra = sorted(case_keys - named)
    missing = sorted(required - case_keys)
    if not extra and not missing:
        return None
    # rename hints: pair each extra key with its closest missing param.
    renames = []
    for e in extra:
        close = difflib.get_close_matches(e, missing, n=1, cutoff=0.5)
        if close:
            renames.append(f"{e!r} → likely meant param {close[0]!r}")
    msg = ("model.forward signature does not accept the input_gen keys — "
           f"extra (unexpected) kwargs={extra}; missing (required) params={missing}")
    if renames:
        msg += "; probable renames: " + "; ".join(renames)
    msg += f". {_CONTRACT_FIX_HINT}"
    return ModelContractResult(False, "SIGNATURE_MISMATCH", msg,
                               {"extra_kwargs": extra, "missing_params": missing})


def _diagnose_output_shape(workspace: Path, out):
    """Return an OUTPUT_SHAPE_MISMATCH ModelContractResult if the sample forward output
    shape disagrees with the captured a3_outputs[0] shape, else None. Best-effort — a
    missing/unparseable edge_dataset.pt just skips this cross-check (returns None)."""
    edge_dataset = workspace / "edge_dataset.pt"
    if not edge_dataset.is_file():
        return None
    try:
        import torch
        ds = torch.load(edge_dataset, weights_only=False)
    except Exception:
        return None
    ds = _coerce_case_list(ds)
    a3_first = None
    if isinstance(ds, dict):
        outs = ds.get("a3_outputs")
        if isinstance(outs, (list, tuple)) and outs:
            a3_first = outs[0]
    elif isinstance(ds, (list, tuple)) and ds:
        c0 = ds[0]
        a3_first = c0.get("a3_outputs") if isinstance(c0, dict) else None
    try:
        import torch

        def _shape(x):
            if isinstance(x, torch.Tensor):
                return tuple(x.shape)
            if isinstance(x, (list, tuple)) and x and isinstance(x[0], torch.Tensor):
                return tuple(x[0].shape)
            return None
        os_, as_ = _shape(out), _shape(a3_first)
        if os_ is not None and as_ is not None and os_ != as_:
            return ModelContractResult(
                False, "OUTPUT_SHAPE_MISMATCH",
                f"model.forward output shape {os_} != captured a3_outputs shape {as_} — "
                f"the CPU reference and the A3 capture disagree on output shape; the golden "
                f"would be graded against a mis-shaped truth. {_CONTRACT_FIX_HINT}",
                {"model_shape": list(os_), "a3_shape": list(as_)})
    except Exception:
        return None
    return None


def _count_a3_outputs(dataset) -> tuple[int, int]:
    """Return (n_captured, n_total) of A3 outputs across BOTH edge_dataset.pt schemas.

    Schema 1 (CPU-truth synth, this module ~line 2025):
        {"inputs": [...], "a3_outputs": [...]}  — top-level aligned lists.
    Schema 2 (per-op run_a3_reference.py, e.g. fused_quant_mat_mul §6):
        [{case_id, inputs, ..., a3_outputs?, a3_error?}, ...]  — per-case dicts;
        a case counts as captured only if it has a non-None a3_outputs AND no
        a3_error.
    Int-keyed dict {0:case,...,N} (some authors, e.g. FA) is coerced to Schema 2
    first via _coerce_case_list (else it falsely reads as 0 cases).
    Unknown shapes return (0, 0) → caller treats as not-captured.
    """
    dataset = _coerce_case_list(dataset)
    if isinstance(dataset, dict):
        outs = dataset.get("a3_outputs")
        ins = dataset.get("inputs")
        if ins is not None:
            n_total = len(ins)
        elif outs is not None:
            try:
                n_total = len(outs)
            except TypeError:
                n_total = 1
        else:
            n_total = 0
        if outs is None:
            return 0, n_total
        try:
            n_cap = len(outs)
        except TypeError:
            n_cap = 1  # single tensor for a single-case dataset
        return n_cap, n_total
    if isinstance(dataset, (list, tuple)):
        n_total = len(dataset)
        n_cap = 0
        for case in dataset:
            if (
                isinstance(case, dict)
                and case.get("a3_outputs") is not None
                and not case.get("a3_error")
            ):
                n_cap += 1
        return n_cap, n_total
    return 0, 0


def _validate_a3_capture(workspace: Path) -> tuple[bool, str]:
    """Content-level gate for an A3 reference capture BEFORE declaring READY.

    Returns (ok, reason). ok=True iff:
      - edge_dataset.pt exists AND every case has a captured a3_outputs
        (n_captured == n_total, n_total > 0), AND
      - a3_baseline_perf.json exists AND median_ms_per_case is a non-empty mapping.

    task#25 (main decision 2026-06-01): require FULL capture (n_captured ==
    n_total). A partial reference verifies only the covered cases; the uncovered
    ones would 'pass' unmeasured = coverage fraud (feedback_input_requirements_
    immutable). Empty/partial capture must NOT silently degrade to CPU-truth for
    quant ops (misleading fp32 oracle = a fake-pass per the no-CPU-fallback rule);
    the caller emits CAPTURE_INCOMPLETE (fail-fast, NOT fallback-eligible).

    Surfaced by §6 cube-MIX confirm-run: a3-author declared verdict=READY over
    edge_dataset.pt with 0/73 captured + median_ms_per_case={}, wasting a 406s /
    $2.28 worker spawn that had to self-detect the empty baseline.

    Counts are derived from the captured dataset bytes.  A per-op runner's
    self-authored manifest is never authoritative for coverage.
    """
    edge = workspace / "edge_dataset.pt"
    perf = workspace / "a3_baseline_perf.json"
    if not edge.is_file():
        return False, f"edge_dataset.pt absent ({edge})"
    if not perf.is_file():
        return False, f"a3_baseline_perf.json absent ({perf})"

    # --- perf check (cheap, torch-free) ---
    try:
        perf_obj = json.loads(perf.read_text())
    except Exception as e:  # noqa: BLE001
        return False, f"a3_baseline_perf.json unreadable: {e!r}"
    median = perf_obj.get("median_ms_per_case")
    if not (isinstance(median, dict) and len(median) > 0):
        return False, (
            "a3_baseline_perf.json median_ms_per_case empty — no per-case A3 "
            "timing captured (perf baseline missing)"
        )

    try:
        import torch  # lazy: producer host has torch, but keep import local
        ds = torch.load(edge, weights_only=False)
    except Exception as e:  # noqa: BLE001
        return False, f"edge_dataset.pt unreadable for capture validation: {e!r}"
    n_cap, n_total = _count_a3_outputs(ds)
    if n_total == 0:
        return False, "edge_dataset.pt has 0 cases"
    if n_cap == n_total:
        return True, f"A3 capture complete ({n_cap}/{n_total} cases have a3_outputs + perf)"
    return False, (
        f"edge_dataset.pt: {n_cap}/{n_total} cases have a3_outputs — A3 capture "
        f"incomplete (require full capture; partial = coverage fraud)"
    )


def _capture_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_CAPTURE_EVIDENCE_FILENAMES = {
    "edge_inputs": "edge_inputs.pt",
    "edge_dataset": "edge_dataset.pt",
    "a3_baseline_perf": "a3_baseline_perf.json",
    "runner": "run_a3_reference.py",
}


def _capture_evidence_paths(workspace: Path) -> dict[str, Path]:
    """Return the fixed evidence inventory expected for every live A3 capture."""
    return {
        label: workspace / filename
        for label, filename in _CAPTURE_EVIDENCE_FILENAMES.items()
    }


def _capture_evidence_inventory(workspace: Path):
    """Build the regular-file hash inventory or return its fail-closed reason."""
    files: dict[str, dict[str, object]] = {}
    for label, path in _capture_evidence_paths(workspace).items():
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("missing, symlinked, or not a regular file")
            files[label] = {
                "path": path.name,
                "size": path.stat().st_size,
                "sha256": _capture_file_sha256(path),
            }
        except Exception as error:
            return None, f"capture evidence {label} invalid: {error}"
    return files, None


def _capture_dataset_counts(dataset_path: Path, error_template: str):
    """Load a captured dataset and return its output counts or the supplied error."""
    try:
        import torch

        dataset = torch.load(dataset_path, weights_only=False)
        return _count_a3_outputs(dataset), None
    except Exception as error:
        return None, error_template.format(error=error)


def _capture_manifest_payload(
    state: dict,
    capture_id: str,
    capture_started_ts: str,
    npu_id: int,
    counts: tuple[int, int],
    files: dict[str, dict[str, object]],
) -> dict:
    """Assemble the canonical live-capture provenance payload."""
    n_captured, n_total = counts
    return {
        "schema": "a3_capture/v2",
        "capture_id": capture_id,
        "live_exec": True,
        "capture_started_ts": capture_started_ts,
        "capture_completed_ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_stage_digest": state["source_stage_digest"],
        "source_arch": "arch22",
        "device": {"kind": "NPU", "npu_id": npu_id},
        "n_total": n_total,
        "n_captured": n_captured,
        "files": files,
    }


def write_a3_capture_provenance(
    workspace: Path,
    *,
    capture_id: str,
    capture_started_ts: str,
    npu_id: int,
) -> tuple[bool, str, Path | None]:
    """Write harness-owned provenance for one successful live A3 execution."""
    from source_arch import verify_source_stage

    state_path = workspace / ".opgen_state.json"
    try:
        state = json.loads(state_path.read_text())
    except Exception as exc:
        return False, f"migration state unreadable: {exc}", None
    valid_stage, stage_reason, _stage_manifest = verify_source_stage(workspace, state)
    if not valid_stage:
        return False, f"source stage invalid at capture time: {stage_reason}", None
    capture_ok, capture_reason = _validate_a3_capture(workspace)
    if not capture_ok:
        return False, capture_reason, None

    files, error = _capture_evidence_inventory(workspace)
    if error is not None:
        return False, error, None
    counts, error = _capture_dataset_counts(
        workspace / _CAPTURE_EVIDENCE_FILENAMES["edge_dataset"],
        "could not count captured A3 cases: {error}",
    )
    if error is not None:
        return False, error, None
    payload = _capture_manifest_payload(
        state, capture_id, capture_started_ts, npu_id, counts, files
    )
    path = workspace / "a3_capture_manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    valid, reason, _loaded = validate_a3_capture_provenance(workspace)
    if not valid:
        path.unlink(missing_ok=True)
        return False, reason, None
    return True, reason, path


def _capture_provenance_inputs(workspace: Path):
    """Read manifest and migration state together or return the original reason."""
    try:
        manifest = workspace / "a3_capture_manifest.json"
        payload = json.loads(manifest.read_text())
        state = json.loads((workspace / ".opgen_state.json").read_text())
        return payload, state, None
    except Exception as error:
        return None, None, f"capture provenance unreadable: {error}"


def _capture_manifest_header_error(payload: dict) -> str | None:
    """Validate the schema and id fields that precede source-stage checking."""
    if payload.get("schema") != "a3_capture/v2" or payload.get("live_exec") is not True:
        return "capture is not harness-owned live-exec schema v2"
    if not isinstance(payload.get("capture_id"), str) or not payload["capture_id"]:
        return "capture_id missing"
    return None


def _capture_manifest_origin_error(payload: dict, state: dict) -> str | None:
    """Validate source identity and NPU-device provenance after stage verification."""
    if payload.get("source_stage_digest") != state.get("source_stage_digest"):
        return "capture source-stage digest does not match current run"
    if payload.get("source_arch") != "arch22":
        return "capture source architecture is not arch22"
    device = payload.get("device")
    if not isinstance(device, dict) or device.get("kind") != "NPU":
        return "capture has no source-NPU device provenance"
    if not isinstance(device.get("npu_id"), int) or device["npu_id"] < 0:
        return "capture NPU id is invalid"
    return None


def _capture_timestamp_error(payload: dict) -> str | None:
    """Validate the timezone-aware and ordered capture timestamps."""
    try:
        started = datetime.fromisoformat(str(payload["capture_started_ts"]).replace("Z", "+00:00"))
        completed = datetime.fromisoformat(
            str(payload["capture_completed_ts"]).replace("Z", "+00:00")
        )
        if started.tzinfo is None or completed.tzinfo is None or completed < started:
            raise ValueError("invalid timestamp ordering/timezone")
    except Exception as error:
        return f"capture timestamps invalid: {error}"
    return None


def _capture_file_inventory_error(workspace: Path, payload: dict) -> str | None:
    """Check every manifest evidence entry against its regular local file."""
    files = payload.get("files")
    if not isinstance(files, dict):
        return "capture file inventory missing"
    for label, filename in _CAPTURE_EVIDENCE_FILENAMES.items():
        entry = files.get(label)
        path = workspace / filename
        if not isinstance(entry, dict) or entry.get("path") != filename:
            return f"capture file entry invalid: {label}"
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("missing/symlink/non-file")
            if entry.get("size") != path.stat().st_size:
                raise ValueError("size mismatch")
            if entry.get("sha256") != _capture_file_sha256(path):
                raise ValueError("SHA256 mismatch")
        except Exception as error:
            return f"capture file {label} invalid: {error}"
    return None


def validate_a3_capture_provenance(
    workspace: Path,
) -> tuple[bool, str, dict]:
    """Validate fresh live-capture provenance against current files and state."""
    from source_arch import verify_source_stage

    payload, state, error = _capture_provenance_inputs(workspace)
    if error is not None:
        return False, error, {}
    header_error = _capture_manifest_header_error(payload)
    if header_error is not None:
        return False, header_error, payload
    valid_stage, stage_reason, _stage_manifest = verify_source_stage(workspace, state)
    if not valid_stage:
        return False, f"source stage invalid: {stage_reason}", payload
    origin_error = _capture_manifest_origin_error(payload, state)
    if origin_error is not None:
        return False, origin_error, payload
    timestamp_error = _capture_timestamp_error(payload)
    if timestamp_error is not None:
        return False, timestamp_error, payload
    inventory_error = _capture_file_inventory_error(workspace, payload)
    if inventory_error is not None:
        return False, inventory_error, payload

    capture_ok, capture_reason = _validate_a3_capture(workspace)
    if not capture_ok:
        return False, capture_reason, payload
    counts, error = _capture_dataset_counts(
        workspace / _CAPTURE_EVIDENCE_FILENAMES["edge_dataset"],
        "captured dataset unreadable: {error}",
    )
    if error is not None:
        return False, error, payload
    n_captured, n_total = counts
    if payload.get("n_total") != n_total or payload.get("n_captured") != n_captured:
        return False, "capture count provenance does not match dataset", payload
    if n_total <= 0 or n_captured != n_total:
        return False, "capture is not complete", payload
    return True, f"fresh live A3 capture verified ({n_captured}/{n_total})", payload


def _write_a3_reference_runnable_json(workspace: Path, rep: O25A3Report) -> Path:
    """Emit a3_reference_runnable.json to workspace/. Parallel to phase_o25's
    ref_runnable.json, but with A3-specific fields."""
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / "a3_reference_runnable.json"
    payload = {
        "verdict": rep.verdict,
        "aclnn_entry": str(rep.aclnn_entry) if rep.aclnn_entry else None,
        "gen_data_source": str(rep.gen_data_source) if rep.gen_data_source else None,
        "peer_op_dependencies": rep.peer_op_dependencies,
        "a3_outputs_path": str(rep.a3_outputs_path) if rep.a3_outputs_path else None,
        "a3_perf_path": str(rep.a3_perf_path) if rep.a3_perf_path else None,
        "a3_exec_attempted": rep.a3_exec_attempted,
        "capture_id": rep.capture_id,
        "capture_manifest_path": (
            str(rep.capture_manifest_path) if rep.capture_manifest_path else None
        ),
        "errors": rep.errors,
        "recommendations": rep.recommendations,
        "summary": rep.summary,
    }
    try:
        state = json.loads((workspace / ".opgen_state.json").read_text())
        if state.get("source_arch") == "arch22" and state.get("target_arch") == "arch35":
            payload["migration"] = {
                "source_arch": "arch22",
                "target_arch": "arch35",
                "source_arch_detection": state.get("source_arch_detection", {}),
            }
    except Exception as error:
        logging.getLogger(__name__).debug(
            "Recoverable operation failed.", exc_info=error
        )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path
