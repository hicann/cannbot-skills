# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bridge: sync workspace eval entry → async ``op_autoresearch`` verifier + worker.

``eval_kernel(EvalRequest(...))`` reads the current
``kernel.py`` + reference, registers a LocalWorker (or RemoteWorker), runs
``KernelVerifier.run()`` then ``run_profile()``, and returns a dict in the
schema ``phase_machine`` / ``workflow`` already consume (``outcome`` /
``correctness`` / ``metrics{...}`` / ``error`` / ``error_source``).
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from op_autoresearch.op.verifier import aggregate
from op_autoresearch.op.verifier.adapters.factory import get_dsl_adapter

from .profile_plan import ProfilePlan, ProfilePlanInput, plan_profile
from .settings import (
    default_reference_data_timeout,
    eval_repeats,
    eval_warmup,
    target_backend,
    target_dsl,
    target_framework,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalRequest:
    task_dir: str
    config: Any
    device_id: Any = 0
    worker_url: Optional[str] = None
    current_step: int = 0
    verify_only: bool = False


@dataclass(frozen=True)
class TargetSpec:
    backend: str
    arch: str
    framework: str
    dsl: str
    device_ids: list[int]


@dataclass(frozen=True)
class EvalSession:
    request: EvalRequest
    target: TargetSpec
    worker: Any
    worker_manager: Any
    verifier: Any
    task_info: Dict[str, Any]
    timeout_s: int
    keep_res: bool


def eval_kernel(request: EvalRequest) -> Dict[str, Any]:
    """Run op_autoresearch verify + profile against current kernel.py.

    ``request.worker_url`` selects a remote worker; otherwise the requested
    local device is used. ``current_step`` distinguishes round log directories.
    """
    return asyncio.run(_eval_async(request))


def _normalize_device_ids(device_id: Any) -> list[int]:
    if device_id is None:
        return []
    if isinstance(device_id, str):
        text = device_id.strip()
        if not text:
            return []
        return [int(x.strip()) for x in text.split(",") if x.strip()]
    if isinstance(device_id, (list, tuple, set)):
        return [int(x) for x in device_id]
    return [int(device_id)]


def _load_seed_files(task_dir: str, ref_file: str, kernel_file: str):
    """Return (kernel_code, ref_code) or (None, infra_fail_dict).

    ``kernel_file`` is the primary editable filename (TaskConfig.
    editable_files[0]); DSL-driven so e.g. ascendc / catlass / triton all
    work the same way without ``kernel.py`` literal hardcoding here.
    """
    kernel_path = os.path.join(task_dir, kernel_file)
    ref_path = os.path.join(task_dir, ref_file)
    if not os.path.exists(kernel_path):
        return None, _infra_fail(
            f"primary editable {kernel_file} not found in {task_dir}")
    if not os.path.exists(ref_path):
        return None, _infra_fail(
            f"reference file {ref_file} not found in {task_dir}")
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()
    with open(ref_path, "r", encoding="utf-8") as f:
        ref_code = f.read()
    return (kernel_code, ref_code), None


def _remote_host_for_url(worker_url: str) -> Optional[Tuple[str, str]]:
    """Find the ``remote_worker.hosts.<alias>`` entry whose tunneled port
    matches ``worker_url``. Used to wire an auto-reconnect callback into
    ``register_remote_worker`` — only valid when worker_url points at a
    tunneled local port that op-autoresearch set up.

    Returns ``(alias, cfg_path)`` so callers pass the same yaml back to
    ``load_remote_host_config`` — eval might be running outside the
    workspace cwd, can't rely on the cwd-search fallback in op-autoresearch.
    """
    import urllib.parse
    try:
        port = urllib.parse.urlparse(
            worker_url if "://" in worker_url else f"http://{worker_url}"
        ).port
    except Exception:
        return None
    if port is None:
        return None
    # workspace_autoresearch/config.yaml is the canonical place; op-autoresearch
    # uses the same yaml via --remote-config (default cwd/config.yaml).
    import yaml
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "..", "config.yaml")
    cfg_path = os.path.abspath(cfg_path)
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return None
    # Match by yaml `worker.port` (single configured tunnel port).
    if int((data.get("worker") or {}).get("port", -1)) != port:
        return None
    hosts = (data.get("remote_worker") or {}).get("hosts") or {}
    # Pick the first host; multi-host setups would need a per-host port
    # mapping, but the current config model is single-tunnel-per-yaml.
    alias = next(iter(hosts.keys()), None)
    if alias is None:
        return None
    return (alias, cfg_path)


def _make_reconnect_callback(worker_url: str):
    """Build a hook that reconnects the ssh -L tunnel matching this
    worker_url. Returns None when the url isn't a tunneled local port
    (e.g. direct remote host, or url not declared in config.yaml).
    """
    resolved = _remote_host_for_url(worker_url)
    if not resolved:
        return None
    alias, cfg_path = resolved
    import urllib.parse

    def _reconnect():
        from op_autoresearch.cli.service import remote_dispatch
        # Pass the absolute cfg_path resolved above —  if eval runs outside
        # workspace cwd, the load_remote_host_config(alias, None) fallback
        # to cwd/config.yaml silently misses, and reconnect goes no-op.
        host_cfg = remote_dispatch.load_remote_host_config(alias, cfg_path)
        if host_cfg is None:
            return
        port = int(urllib.parse.urlparse(
            worker_url if "://" in worker_url else f"http://{worker_url}"
        ).port)
        remote_dispatch.dispatch_reconnect_tunnel(alias, host_cfg, port)
    return _reconnect


async def _acquire_worker(backend: str, arch: str, device_ids: list[int],
                          worker_url: Optional[str]):
    """Register + select a worker. Returns (worker, manager) on success,
    or (None, infra_fail_dict) on failure.
    """
    from op_autoresearch.core.worker.manager import (
        RemoteWorkerRegistration,
        get_worker_manager,
        register_local_worker,
        register_remote_worker,
    )
    try:
        if worker_url:
            # OP_AUTORESEARCH's register_remote_worker uses urllib's URL parser, which
            # needs an explicit scheme. CA / WA CLIs accept bare host:port
            # (e.g. `--worker-url 127.0.0.1:9111`); normalise here so the
            # call site stays terse.
            url = worker_url if worker_url.startswith(("http://", "https://")) \
                else f"http://{worker_url}"
            worker = await register_remote_worker(RemoteWorkerRegistration(
                backend=backend, arch=arch, worker_url=url,
                expected_device_ids=device_ids or None,
                on_transient_failure=_make_reconnect_callback(url),
            ))
        else:
            worker = await register_local_worker(
                device_ids=device_ids or [0], backend=backend, arch=arch)
    except Exception as e:
        return None, _infra_fail(f"worker registration failed: {e}")
    wm = get_worker_manager()
    if not await wm.reserve(worker):
        return None, _infra_fail(
            f"registered worker disappeared for backend={backend} arch={arch}")
    return (worker, wm), None


def _resolve_eval_timeout_s(task_dir: str, config) -> int:
    """Compute the wall-clock subprocess timeout for one verify / profile
    invocation as ``eval_timeout * num_cases``.

    ``TaskConfig.eval_timeout`` is a PER-SHAPE budget (see TaskConfig
    docstring). For a multi-shape op the verify subprocess iterates
    every input group, so the wall-clock cap is N × per-shape budget —
    otherwise a 30-case op with a 600s per-shape budget hits the worker
    protocol default (300s) long before it can finish.

    ``num_cases`` resolution order: explicit ``config.num_cases``
    (scaffold-probed at task creation, written into task.yaml) > runtime
    probe via ``utils.input_groups.num_cases`` (covers the case where
    scaffold's probe failed because the dev host lacks torch) > 1
    (single-shape fallback).
    """
    num_cases = int(getattr(config, "num_cases", 0) or 0)
    if num_cases <= 0:
        from .input_groups import num_cases_from_ref
        num_cases = num_cases_from_ref(os.path.join(task_dir, config.ref_file))
    return int(config.eval_timeout) * num_cases


def _sticky_baseline_override(task_dir: str,
                              sidecar: Optional[dict]):
    """Committed AR baseline as ``StickyOverride`` (metric + per_shape_us),
    or ``None`` when no fingerprint-matched anchor exists. Caller wraps
    into a Section so multi-shape per_case data survives the worker hop.

    Anchor key is ``(num_cases, shape_signature)`` — run_times-independent
    (see :mod:`utils.baseline_anchor`), so baseline reuse survives
    :func:`plan_profile` varying ``run_times`` per round.
    """
    sidecar = sidecar if isinstance(sidecar, dict) else {}
    per_case = sidecar.get("per_case") or []
    descs = [c.get("case_desc") for c in per_case
             if isinstance(c, dict) and c.get("case_desc")]
    num_cases = int(sidecar.get("num_cases") or len(descs) or 1)
    try:
        from phase_machine.state_store import load_progress

        from utils.baseline_anchor import (
            current_fingerprint,
            sticky_override_from_progress,
        )
        decision = sticky_override_from_progress(
            load_progress(task_dir), current_fingerprint(num_cases, descs))
    except Exception:
        return None
    return decision.override


def _profile_plan(sidecar: Optional[dict], config, *,
                  base_only: bool = False, override=None) -> ProfilePlan:
    """Turn the verify sidecar + committed baseline ``override`` (resolved
    once by the caller) into a :class:`ProfilePlan`. Per-shape walls come
    from verify; sizing / precedence stays in :func:`plan_profile`.
    """
    per_case = (sidecar or {}).get("per_case") or []

    def _walls(key: str) -> list:
        walls = []
        for case in per_case:
            if not isinstance(case, dict):
                continue
            value = case.get(key)
            if isinstance(value, (int, float)) and value > 0:
                walls.append(float(value))
        return walls

    sticky_section = None
    if not base_only and override is not None:
        from op_autoresearch.op.verifier.profiler_utils import make_profile_section
        sticky_section = make_profile_section(
            override.metric, per_case_us=override.per_shape_us,
            method="override")
    return plan_profile(ProfilePlanInput(
        ref_walls=_walls("ref_wall_us"),
        impl_walls=_walls("impl_wall_us"),
        eval_timeout=float(getattr(config, "eval_timeout", 0) or 0),
        warmup=eval_warmup(),
        repeats=eval_repeats(),
        sticky_section=sticky_section,
        base_only=base_only,
    ))


def _build_verifier(request: EvalRequest, ref_code: str,
                    target: TargetSpec, worker):
    from op_autoresearch.op.verifier.kernel_verifier import KernelVerifier
    task_dir = request.task_dir
    config = request.config
    log_dir = os.path.join(task_dir, ".ar_state", "op_autoresearch_verify")
    os.makedirs(log_dir, exist_ok=True)
    # warmup/repeats are global eval knobs in config.yaml — not on
    # TaskConfig — mirroring CA's `eval_runner` -> `settings.
    # eval_warmup/eval_repeats` pattern. Both runs (baseline ref +
    # per-round kernel) read the same values so timing stays comparable.
    config_dict: Dict[str, Any] = {
        "log_dir": log_dir,
        "verify_timeout": _resolve_eval_timeout_s(task_dir, config),
        "reference_data_timeout": default_reference_data_timeout(),
        "warmup_times": eval_warmup(),
        "run_times": eval_repeats(),
        "task_dir": task_dir,
        "framework_filename": os.path.basename(str(config.ref_file)),
        "framework_module_name": os.path.splitext(
            os.path.basename(str(config.ref_file))
        )[0],
    }
    # Forward all per-DSL knobs from TaskConfig.dsl_config. adapter's
    # prepare_config() reads them at run/run_profile time. New DSL
    # adding a new key is zero code change here.
    config_dict.update(getattr(config, "dsl_config", None) or {})
    # Sidecar files are materialized exactly as declared in task.yaml.
    # AR tasks normally use reference.py/reference.json; the verifier gets
    # the same framework filename/module above, so no sidecar rename is
    # needed.
    data_files = getattr(config, "data_files", None) or []
    framework_aux_files: Dict[str, Any] = {}
    for rel in data_files:
        if not isinstance(rel, str) or not rel:
            continue
        src = os.path.join(task_dir, rel)
        if not os.path.isfile(src):
            continue
        with open(src, "rb") as f:
            framework_aux_files[rel] = f.read()
    if framework_aux_files:
        config_dict["framework_aux_files"] = framework_aux_files
    return KernelVerifier(
        op_name=config.name,
        task_id=config.name,
        framework_code=ref_code,
        framework=target.framework,
        dsl=target.dsl,
        backend=target.backend,
        arch=target.arch,
        config=config_dict,
        worker=worker,
    )


def _base_metrics_from_profile(profile_result: dict,
                               op_name: str) -> Dict[str, Any]:
    base_us = _float(profile_result.get("base_time"))
    if base_us is None or base_us <= 0:
        return {}

    per_shape_base = list(profile_result.get("per_shape_base_us") or [])
    if not per_shape_base:
        per_shape_base = [base_us]
    num_cases = len(per_shape_base)

    descs = list(profile_result.get("case_descs") or [])
    if len(descs) != num_cases:
        if descs:
            logger.warning(
                "[%s] base-only case_descs len %s != per_shape_base_us "
                "len %s; regenerating labels",
                op_name,
                len(descs),
                num_cases,
            )
        descs = [f"case{i}" for i in range(num_cases)]

    return {
        "ref_latency_us": base_us,
        "num_cases": num_cases,
        "per_shape_base_us": per_shape_base,
        "per_shape_descs": descs,
    }


def _add_failure_signals(payload: Dict[str, Any], log: str) -> None:
    """Best-effort: tag the payload with extracted failure signals. Import is
    local + swallowed so a missing/broken failure_extractor never breaks the
    eval result. Mutates ``payload`` in place.
    """
    try:
        from .failure_extractor import extract_failure_signals
        diag = extract_failure_signals(log)
        if not diag.is_empty:
            payload["failure_signals"] = diag.to_dict()
    except Exception:
        logger.debug("Could not extract structured failure signals", exc_info=True)


def _verify_fail_payload(verify_log: Optional[str],
                         sidecar: Optional[dict] = None,
                         metrics: Optional[Dict[str, Any]] = None,
                         report_path: Optional[str] = None
                         ) -> Dict[str, Any]:
    """Build the dict downstream eval_client consumes. When the verify script
    wrote a per-case sidecar (kernel_verify_template_refactored.j2 ->
    verify_result.json -> KernelVerifier.last_verify_sidecar), use its
    `error_source` so a ref-side crash flips to "ref". The per-case results are
    folded into the SAME ``per_shape_*`` metric arrays a passing round uses, so
    the FAIL prints through the one per_shape_table path. When ``report_path``
    is given the FULL per-case + complete log is written there (verify.py-style
    artifact) and surfaced as ``fail_report`` for the agent to open by file.
    """
    payload: Dict[str, Any] = {
        "outcome": "kernel_fail",
        "correctness": False,
        "metrics": dict(metrics or {}),
        "error": (verify_log or "verify failed")[-2000:],
        "error_source": "kernel",
        "raw_output_tail": (verify_log or "")[-4000:],
    }
    per_case: list = []
    if isinstance(sidecar, dict):
        es = sidecar.get("error_source")
        if es in ("ref", "kernel"):
            payload["error_source"] = es
        per_case = sidecar.get("per_case") or []
        if sidecar.get("failed_indices") is not None:
            payload["failed_indices"] = sidecar["failed_indices"]
    if per_case:
        # gen/base = the free per-shape verify walls (None where unmeasured);
        # status = PASS / failure_kind. Same arrays the success table reads.
        payload["metrics"].update({
            "num_cases": len(per_case),
            "per_shape_descs": [c.get("case_desc", "") for c in per_case],
            "per_shape_status": ["PASS" if c.get("correctness")
                                 else (c.get("failure_kind") or "FAIL")
                                 for c in per_case],
            "per_shape_gen_us": [c.get("impl_wall_us") for c in per_case],
            "per_shape_base_us": [c.get("ref_wall_us") for c in per_case],
        })
    _add_failure_signals(payload, verify_log or "")
    if report_path:
        from .eval_summary import write_artifact
        payload["fail_report"] = write_artifact(report_path, {
            "outcome": payload["outcome"],
            "error_source": payload["error_source"],
            "per_case": per_case,
            "failure_signals": payload.get("failure_signals", {}),
            "verify_log": verify_log or "",
        })
    return payload


def _profile_fail_payload(exc: Exception) -> Dict[str, Any]:
    tb_str = traceback.format_exc()
    payload = {
        "outcome": "kernel_fail",
        "correctness": True,
        "metrics": {},
        "error": f"profile raised: {exc}",
        "error_source": "kernel",
        "raw_output_tail": tb_str[-4000:],
    }
    _add_failure_signals(payload, tb_str)
    return payload


def _too_slow_payload(reason: str, sidecar: Optional[dict]) -> Dict[str, Any]:
    """Kernel is correct but a single call exceeds the per-shape budget.
    ``kernel_fail`` (agent can fix by optimising), deliberately NOT ``inf``,
    which ``_make_ok_payload`` collapses to ``infra_fail`` (operator-only).
    Mirrors :func:`_profile_fail_payload` (correctness stays True).
    """
    payload: Dict[str, Any] = {
        "outcome": "kernel_fail",
        "correctness": True,
        "metrics": {},
        "error": reason,
        "error_source": "kernel",
    }
    if isinstance(sidecar, dict) and sidecar.get("per_case"):
        payload["per_case"] = sidecar["per_case"]
    return payload


EVAL_FAIL_REPORT = "eval_fail_report.json"


def _iteration_verify_dir(task_dir: str, op_name: str, current_step: int) -> str:
    """Mirror KernelVerifier._create_verify_dir naming; needed because the
    verifier doesn't expose ``last_verify_dir`` and we want the adapter's
    post_iteration_cleanup hook to run after each round.
    """
    return os.path.join(
        task_dir, ".ar_state", "op_autoresearch_verify", op_name,
        f"Iteration{op_name}_Step{current_step}_verify",
    )


def _make_verify_ok_payload(sidecar: Optional[dict]) -> Dict[str, Any]:
    """Return a success payload for verify-only callers."""
    sidecar = sidecar if isinstance(sidecar, dict) else {}
    metrics = {
        "num_cases": int(sidecar.get("num_cases") or 1),
        "max_abs_diff": sidecar.get("worst_max_abs_diff"),
    }
    payload: Dict[str, Any] = {
        "outcome": "ok",
        "correctness": True,
        "metrics": metrics,
        "error": None,
        "error_source": None,
    }
    if sidecar.get("per_case"):
        payload["per_case"] = sidecar["per_case"]
    return payload


def _aligned_profile_metadata(
    profile_result: dict,
    op_name: str,
    per_shape_gen: list,
) -> tuple[Optional[float], list, Optional[float], list[str]]:
    num_cases = len(per_shape_gen)
    base_us = _float(profile_result.get("base_time"))
    per_shape_base = list(profile_result.get("per_shape_base_us") or [])
    if base_us is not None and len(per_shape_base) != num_cases:
        logger.warning(
            "[%s] dropping per_shape_base_us "
            "(len %s != per_shape_gen_us len %s)",
            op_name,
            len(per_shape_base),
            num_cases,
        )
        per_shape_base = []
        base_us = None
    speedup = (
        aggregate.geomean_ratio(per_shape_base, per_shape_gen)
        if per_shape_base
        else None
    )
    descriptions = list(profile_result.get("case_descs") or [])
    if len(descriptions) != num_cases:
        if descriptions:
            logger.warning(
                "[%s] case_descs len %s != per_shape_gen_us len %s — "
                "regenerating with generic caseN labels",
                op_name,
                len(descriptions),
                num_cases,
            )
        descriptions = [f"case{i}" for i in range(num_cases)]
    return base_us, per_shape_base, speedup, descriptions


def _make_ok_payload(profile_result: dict, op_name: str) -> Dict[str, Any]:
    """Pack profile_result into the workspace's eval-result schema. Returns
    an ``infra_fail`` payload when gen_time is missing / non-positive, or
    when the canonical per-shape arrays are absent (chain regression).

    ``profile_result`` is the canonical per-shape dict surfaced by
    ``KernelVerifier.run_profile``:

        {
          "gen_time": float,                    # aggregate (mean of per_shape)
          "base_time": float | None,            # cross-backend → None
          "speedup": float | None,
          "per_shape_gen_us": list[float],      # always populated; len == num_cases
          "per_shape_base_us": list[float],     # [] when base skipped
          "case_descs": list[str],              # from verify sidecar
          ...roofline fields...
        }

    No fallback for missing ``per_shape_gen_us`` — the profile template +
    profiler_utils + local_worker + KernelVerifier chain guarantees it.
    Empty here means a chain regression and we surface that explicitly.
    Missing ``case_descs`` is softer (verify sidecar may genuinely be
    skipped on some paths) — we synthesize generic ``caseN`` labels.
    """
    gen_us = _float(profile_result.get("gen_time"))
    if gen_us is None or gen_us <= 0:
        return _infra_fail(
            "profile returned invalid "
            f"gen_time={profile_result.get('gen_time')!r}"
        )

    per_shape_gen = list(profile_result.get("per_shape_gen_us") or [])
    if not per_shape_gen:
        return _infra_fail(
            "profile_result missing per_shape_gen_us "
            f"(gen_time={gen_us:.2f}us is set, but per-case breakdown is "
            "empty — chain regression in profiler_utils / LocalWorker / "
            "KernelVerifier)"
        )
    base_us, per_shape_base, speedup, descs = _aligned_profile_metadata(
        profile_result,
        op_name,
        per_shape_gen,
    )
    num_cases = len(per_shape_gen)

    return {
        "outcome": "ok",
        "correctness": True,
        "metrics": {
            "latency_us": gen_us,
            "ref_latency_us": base_us,
            "speedup_vs_ref": speedup,
            "num_cases": num_cases,
            "per_shape_gen_us": per_shape_gen,
            "per_shape_base_us": per_shape_base,
            "per_shape_descs": descs,
        },
        "error": None,
        "error_source": None,
    }


def _resolve_target_spec(request: EvalRequest,
                         ) -> tuple[Optional[TargetSpec], Optional[dict]]:
    arch = request.config.arch
    if not arch and request.worker_url:
        arch = _arch_from_worker(request.worker_url)
    if not arch:
        return None, _infra_fail(
            "task.yaml missing arch and could not derive it from worker "
            f"/status (worker_url={request.worker_url!r})")
    return TargetSpec(
        backend=target_backend(),
        arch=arch,
        framework=target_framework(),
        dsl=target_dsl(),
        device_ids=_normalize_device_ids(request.device_id),
    ), None


def _trace_setting(request: EvalRequest, target: TargetSpec) -> bool:
    keep_res = os.environ.get("OP_AUTORESEARCH_PROF_KEEP_RES") == "1"
    if keep_res and target.backend != "ascend":
        logger.warning(
            "[%s] --trace ignored: msprof trace is Ascend-only (backend=%s)",
            request.config.name,
            target.backend,
        )
        return False
    return keep_res


async def _open_eval_session(request: EvalRequest, target: TargetSpec,
                             kernel_code: str, ref_code: str,
                             ) -> tuple[Optional[EvalSession], Optional[dict]]:
    acquired, error = await _acquire_worker(
        target.backend, target.arch, target.device_ids, request.worker_url)
    if error is not None:
        return None, error
    worker, worker_manager = acquired
    verifier = _build_verifier(request, ref_code, target, worker)
    task_info: Dict[str, Any] = {
        "coder_code": kernel_code,
        "task_dir": request.task_dir,
        **(getattr(request.config, "dsl_config", None) or {}),
    }
    return EvalSession(
        request=request,
        target=target,
        worker=worker,
        worker_manager=worker_manager,
        verifier=verifier,
        task_info=task_info,
        timeout_s=_resolve_eval_timeout_s(request.task_dir, request.config),
        keep_res=_trace_setting(request, target),
    ), None


def _profile_settings(session: EvalSession, plan: ProfilePlan) -> dict:
    return {
        "timeout": session.timeout_s,
        "keep_res": session.keep_res,
        **plan.settings,
    }


async def _reference_metrics_after_failure(
        session: EvalSession, sidecar: Optional[dict], override,
        ) -> Dict[str, Any]:
    if session.request.verify_only or override is not None:
        return {}
    try:
        plan = _profile_plan(sidecar, session.request.config, base_only=True)
        profile = await session.verifier.run_profile(
            session.task_info,
            current_step=session.request.current_step,
            profile_settings=_profile_settings(session, plan),
        )
        return _base_metrics_from_profile(
            profile, session.request.config.name)
    except Exception as exc:  # boundary around backend/worker implementations
        logger.warning(
            "[%s] failed to profile reference after verify failure: %s",
            session.request.config.name,
            exc,
            exc_info=True,
        )
        return {}


async def _failed_verification_payload(
        session: EvalSession, verify_log: Optional[str],
        sidecar: Optional[dict], override,
        ) -> Dict[str, Any]:
    metrics = await _reference_metrics_after_failure(
        session, sidecar, override)
    report_path = None
    if not session.request.verify_only:
        report_path = os.path.join(
            _iteration_verify_dir(
                session.request.task_dir,
                session.request.config.name,
                session.request.current_step,
            ),
            EVAL_FAIL_REPORT,
        )
    return _verify_fail_payload(
        verify_log, sidecar, metrics, report_path=report_path)


async def _execute_eval_session(session: EvalSession) -> Dict[str, Any]:
    request = session.request
    verified, verify_log = await session.verifier.run(
        session.task_info, current_step=request.current_step)
    sidecar = getattr(session.verifier, "last_verify_sidecar", None)
    override = _sticky_baseline_override(request.task_dir, sidecar)
    if not verified:
        return await _failed_verification_payload(
            session, verify_log, sidecar, override)
    if request.verify_only:
        return _make_verify_ok_payload(sidecar)
    plan = _profile_plan(sidecar, request.config, override=override)
    if plan.too_slow:
        return _too_slow_payload(plan.too_slow, sidecar)
    try:
        profile = await session.verifier.run_profile(
            session.task_info,
            current_step=request.current_step,
            profile_settings=_profile_settings(session, plan),
        )
    except Exception as exc:  # boundary around generated profile programs
        return _profile_fail_payload(exc)
    return _make_ok_payload(profile, request.config.name)


async def _cleanup_eval_session(session: EvalSession) -> None:
    request = session.request
    try:
        session.verifier.dsl_adapter.post_iteration_cleanup(
            _iteration_verify_dir(
                request.task_dir, request.config.name, request.current_step))
    except Exception:  # cleanup must not mask the evaluation result
        logger.debug("Post-iteration cleanup failed", exc_info=True)
    try:
        await session.worker_manager.release(session.worker)
    except Exception:  # cleanup must not mask the evaluation result
        logger.debug("Worker release failed", exc_info=True)


async def _eval_async(request: EvalRequest) -> Dict[str, Any]:
    # Resolve the entry file through the same DSL adapter used by scaffold.
    kernel_file = get_dsl_adapter(target_dsl()).entry_filename_template.format(
        op_name=request.config.name)
    seed, error = _load_seed_files(
        request.task_dir, request.config.ref_file, kernel_file)
    if error is not None:
        return error
    kernel_code, ref_code = seed

    target, error = _resolve_target_spec(request)
    if error is not None:
        return error
    session, error = await _open_eval_session(
        request, target, kernel_code, ref_code)
    if error is not None:
        return error
    try:
        return await _execute_eval_session(session)
    finally:
        await _cleanup_eval_session(session)


def _arch_from_worker(worker_url: str) -> Optional[str]:
    """Curl http://<worker_url>/api/v1/status and return the arch field.
    Returns None on any failure; the caller surfaces the infra_fail.
    """
    import json as _json
    import urllib.request
    url = worker_url if worker_url.startswith(("http://", "https://")) \
        else f"http://{worker_url}"
    try:
        with urllib.request.urlopen(f"{url}/api/v1/status", timeout=5) as r:
            return _json.loads(r.read().decode("utf-8")).get("arch")
    except Exception:
        return None


def _float(x) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _infra_fail(msg: str) -> Dict[str, Any]:
    return {
        "outcome": "infra_fail",
        "correctness": False,
        "metrics": {},
        "error": msg,
        "error_source": None,
    }
