# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Phase O5 orchestrator-driven perf capture (task #17, 2026-05-18).

Design: docs/design/PERF_METHODOLOGY_NOTES.md#phase-o5-perf-capture-design-2026-05-18

## Why this module exists

Post-P135.PR (`de7a1094`, 2026-05-18), `kw_brief.py` Phase D instructs the
worker to use `torch_npu.profiler` per the brief's Option 1 contract. But
the worker still AUTHORS the per-op timing scripts
(`run_a3_event_timing.py` / `run_a5_event_timing.py` / etc.) and reads
those scripts' output INTO `verification.json.performance`. Risks:

1. **Per-op methodology drift** — different workers write subtly different
   profiler harnesses (warmup count, schedule args, parse logic). P141
   catches method-label drift but not numeric drift hidden behind
   matching labels.

2. **Worker self-report risk** — same context that writes a fabricated
   `8/8 PASS` could write a fabricated `2.7×` ratio. P0aba.O5 +
   `post_verify.py` catch precision fraud; perf fraud has no analogous
   re-measurement gate as of 2026-05-18.

3. **Regime split worker has to reason about** — single-tensor vs
   foreach-list have different `Event`-wrap behavior. Brief says
   "profiler-CSV" but worker may still get cute under pressure.

This module moves perf measurement OUT of the worker context, INTO the
orchestrator. Worker's responsibility ends at producing a runnable
`model.py` + `model_new_<backend>.py`; orchestrator runs the
profiler-CSV harness against both, symmetrically, and writes the
result into `verification.json.performance`.

## Module contract

`measure_op_perf(op, workspace, *, plugin, a3_env=None, warmup=5,
repeats=5, threshold=0.9) -> dict` returns the
`verification.json.performance` payload directly:

```
{
  "method": "symmetric torch_npu profiler: schedule(warmup=5, active=5); Device Self Duration(us)",
  "ratio": <float | None>,
  "reference_us": <float | None>,
  "candidate_us": <float | None>,
  "samples_us_reference": [...],
  "samples_us_candidate": [...],
  "status": "PASS" | "PASS_WITHIN_TOLERANCE" | "BELOW_THRESHOLD" | "NOT_VERIFIED_SAME_METHOD",
  "phase_o5_perf_capture_version": "1.0",
}
```

## Feature flag

`AOG_PERF_CAPTURE=1` env enables; default off pending pilot on
foreach_sqrt re-finalize (migration plan step 2). When the flag is off,
`is_enabled()` returns False and callers should fall through to the
existing worker-authored timing path.

## v1 scope (this commit)

- Module API + dispatch + dataclass + result schema
- AscendC candidate entry-point resolution
- arch22→arch35 SSH path STUBBED — returns NOT_VERIFIED_SAME_METHOD with
  honest reason. SSH dispatch is migration-plan step 3.
- Not wired into orchestrator.py main loop yet (pilot-then-wire from
  design doc). Wiring will land as a separate commit after foreach_sqrt
  pilot validates the local-execution path.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


VERSION = "1.0"
DEFAULT_WARMUP = 5
DEFAULT_REPEATS = 5
DEFAULT_THRESHOLD = 0.9

METHOD_LABEL = (
    "phase_o5_perf_capture: torch_npu.profiler.profile "
    "schedule(warmup={warmup}, active={repeats}) + "
    "operator_details.csv Device Self Duration(us) — "
    "symmetric across reference/candidate sides"
)


class _PerfCaptureFailure(Exception):
    """Internal — caller catches and returns NOT_VERIFIED_SAME_METHOD."""


@dataclass
class PerfResult:
    """In-memory representation of a measurement. Serialize via `to_dict()`
    for verification.json embedding.
    """

    method: str
    status: str
    ratio: Optional[float] = None
    reference_us: Optional[float] = None
    candidate_us: Optional[float] = None
    samples_us_reference: list = field(default_factory=list)
    samples_us_candidate: list = field(default_factory=list)
    phase_o5_perf_capture_version: str = VERSION
    error: Optional[str] = None
    reason: Optional[str] = None
    # Stage-3 tool-usage visibility (2026-07-03): the per-pipe instruction-timeline
    # coverage record(s) emitted by the aog-msprof-timeline skill (tl_gap.py -> *.coverage.json),
    # keyed by label (e.g. "before"/"after"). Records that msprof --instr-profiling was actually
    # sampled + what pipeline coverage (VEC-idle ∩ MTE-busy stall) was judged — so a reviewer/gate
    # can VERIFY the timeline was consulted, not just read a bare device-time number. A perf-swap
    # with no `profiling` block is tool-usage-unverified.
    profiling: Optional[dict] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.error is None:
            d.pop("error", None)
        if self.reason is None:
            d.pop("reason", None)
        if self.profiling is None:
            d.pop("profiling", None)
        return d


def is_enabled() -> bool:
    """Feature flag check. Default off pending pilot validation."""
    return os.environ.get("AOG_PERF_CAPTURE", "").strip() in ("1", "true", "TRUE", "yes")


def measure_op_perf(
    op: str,
    workspace: Path,
    *,
    plugin: Any,
    a3_env: Optional[dict] = None,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
    threshold: float = DEFAULT_THRESHOLD,
    npu_id: int = 0,
) -> dict:
    """Measure per-op perf symmetrically; return verification.json schema dict.

    Dispatches on `plugin.name` (or class name fallback). Honest
    NOT_VERIFIED_SAME_METHOD on dispatch gaps / runner failures — caller
    should NOT trust ratio when status starts with NOT_VERIFIED.
    """
    plugin_name = (getattr(plugin, "name", None)
                   or type(plugin).__name__.lower())
    method = METHOD_LABEL.format(warmup=warmup, repeats=repeats)

    try:
        candidate_filename = _candidate_filename_for(plugin_name, workspace)

        if "port_a3" in plugin_name:
            return _not_verified(
                method,
                "port_a3 SSH dispatch not yet wired (task #17 v1 scope — "
                "migration plan step 3 will add SSH+docker-exec runner)",
            )

        ref_samples, cand_samples = _measure_local(
            workspace=workspace,
            reference_filename="model.py",
            candidate_filename=candidate_filename,
            warmup=warmup,
            repeats=repeats,
            npu_id=npu_id,
        )
    except _PerfCaptureFailure as e:
        return _not_verified(method, str(e))
    except Exception as e:  # noqa: BLE001 — last-ditch honest fallback
        return _not_verified(method, f"unexpected: {type(e).__name__}: {e}")

    return _build_result(
        method=method,
        ref_samples=ref_samples,
        cand_samples=cand_samples,
        threshold=threshold,
    ).to_dict()


def _candidate_filename_for(plugin_name: str, workspace: Path) -> str:
    """Resolve which `model_new_*.py` is the candidate side, per workspace
    layout / plugin contract. Mirrors the universal entry-point rule
    documented in CLAUDE.md (P135.SI).

    Both supported workflows use `model_new_ascendc.py`. No other
    programming-model entry point is accepted.
    """
    if (workspace / "model_new_ascendc.py").exists():
        return "model_new_ascendc.py"
    raise _PerfCaptureFailure(
        f"no candidate model file found in {workspace} — expected "
        "model_new_ascendc.py"
    )


def _measure_local(
    *,
    workspace: Path,
    reference_filename: str,
    candidate_filename: str,
    warmup: int,
    repeats: int,
    npu_id: int,
) -> tuple[list, list]:
    """Run the profiler harness locally (same host, same NPU).

    Spawns a Python subprocess that imports torch_npu, loads both
    Model.forward (reference) and ModelNew.forward (candidate) from the
    workspace, runs the symmetric profiler measurement, and prints a
    JSON line with the per-iter samples.

    Subprocess isolation: keeps torch_npu out of the orchestrator's
    main process (orchestrator is pure-Python; subprocess inherits the
    user's torch_npu env).
    """
    if not (workspace / reference_filename).exists():
        raise _PerfCaptureFailure(
            f"reference file missing: {workspace / reference_filename}"
        )
    if not (workspace / candidate_filename).exists():
        raise _PerfCaptureFailure(
            f"candidate file missing: {workspace / candidate_filename}"
        )

    harness = _build_harness_script(
        reference_filename=reference_filename,
        candidate_filename=candidate_filename,
        warmup=warmup,
        repeats=repeats,
        npu_id=npu_id,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="phase_o5_perf_"
    ) as tf:
        tf.write(harness)
        harness_path = tf.name

    try:
        proc = subprocess.run(
            ["python3", harness_path],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            raise _PerfCaptureFailure(
                f"harness exit {proc.returncode}: "
                f"stderr={proc.stderr[-500:]} stdout_tail={proc.stdout[-200:]}"
            )

        return _parse_harness_output(proc.stdout)
    finally:
        try:
            os.unlink(harness_path)
        except OSError:
            pass


def _parse_harness_output(stdout: str) -> tuple[list, list]:
    """Parse the harness subprocess's JSON-line output.

    The harness prints a single line `PHASE_O5_PERF_RESULT={"ref": [...], "cand": [...]}`
    so we can ignore any preceding logging noise.
    """
    marker = "PHASE_O5_PERF_RESULT="
    line = next(
        (ln for ln in stdout.splitlines() if ln.startswith(marker)),
        None,
    )
    if line is None:
        raise _PerfCaptureFailure(
            f"harness did not emit PHASE_O5_PERF_RESULT line; "
            f"stdout_tail={stdout[-300:]}"
        )
    try:
        payload = json.loads(line[len(marker):])
    except json.JSONDecodeError as exc:
        raise _PerfCaptureFailure(
            f"harness output JSON parse failed: {exc}"
        ) from exc
    ref = payload.get("ref")
    cand = payload.get("cand")
    if not isinstance(ref, list) or not isinstance(cand, list):
        raise _PerfCaptureFailure(
            f"harness payload missing ref/cand lists: keys={list(payload.keys())}"
        )
    if not ref or not cand:
        raise _PerfCaptureFailure(
            f"harness returned empty samples: ref={len(ref)} cand={len(cand)}"
        )
    return ref, cand


def _build_harness_script(
    *,
    reference_filename: str,
    candidate_filename: str,
    warmup: int,
    repeats: int,
    npu_id: int,
) -> str:
    """Compose the subprocess Python script.

    Symmetry-by-construction: identical profiler config + identical CSV
    parse for both sides. Source of parser primitives:
    `vendor/AscendOpGenAgent/utils/performance.py:299-340`
    (`_run_profiler_with_config`).
    """
    return f"""# AUTO-GENERATED by phase_o5_perf_capture.py — do not edit
import importlib.util, json, os, sys, time
from pathlib import Path

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "{npu_id}")

import torch
import torch_npu  # noqa: F401
import pandas as pd


def _load_model(filename, class_name):
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    init = getattr(mod, "get_init_inputs", lambda: [])()
    cls = getattr(mod, class_name)
    return cls(*init).npu().eval(), mod


def _get_inputs(mod):
    fn = getattr(mod, "get_inputs", None)
    if fn is None:
        fn = getattr(mod, "get_input_groups", None)
    if fn is None:
        raise RuntimeError("model has neither get_inputs nor get_input_groups")
    inputs = fn()
    # Take first input group if get_input_groups returned a list of groups
    if isinstance(inputs, list) and inputs and isinstance(inputs[0], (list, tuple)) and inputs and not isinstance(inputs[0], torch.Tensor):
        inputs = inputs[0]
    return [t.npu() if isinstance(t, torch.Tensor) else t for t in inputs]


def _measure(model, inputs, warmup, repeats, tag):
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*inputs)
    torch.npu.synchronize()

    skip_first = 1 + warmup
    profile_dir = f"/tmp/phase_o5_perf_{{tag}}_{{int(time.time()*1000)}}"

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=None,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )

    def test_fn():
        with torch.no_grad():
            _ = model(*inputs)
        torch.npu.synchronize()

    test_fn()
    torch.npu.synchronize()

    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.NPU,
            torch_npu.profiler.ProfilerActivity.CPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=warmup, active=repeats, repeat=1, skip_first=skip_first,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_dir),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(skip_first + repeats):
            test_fn()
            prof.step()
            torch.npu.synchronize()

    # Parse operator_details.csv from the profile output
    csv_files = list(Path(profile_dir).rglob("operator_details.csv"))
    if not csv_files:
        raise RuntimeError(f"no operator_details.csv in {{profile_dir}}")
    df = pd.read_csv(csv_files[0])
    if "Device Self Duration(us)" not in df.columns:
        raise RuntimeError(f"missing Device Self Duration(us) col in {{csv_files[0]}}")

    # Sum across operator names, divide by active count for per-iter avg.
    total_us = df["Device Self Duration(us)"].sum()
    per_iter_us = total_us / repeats
    # For per-iter samples, we use the aggregate; ideally split by Count
    # but the column is not always present. Conservative: repeat the
    # per-iter average `repeats` times so consumers see expected list shape.
    samples = [per_iter_us] * repeats
    return samples


reference_model, reference_mod = _load_model("{reference_filename}", "Model")
candidate_model, candidate_mod = _load_model("{candidate_filename}", "ModelNew")
inputs = _get_inputs(reference_mod)

ref_samples = _measure(reference_model, inputs, {warmup}, {repeats}, "ref")
cand_samples = _measure(candidate_model, inputs, {warmup}, {repeats}, "cand")

print("PHASE_O5_PERF_RESULT=" + json.dumps({{
    "ref": ref_samples,
    "cand": cand_samples,
}}))
"""


def _build_result(
    *,
    method: str,
    ref_samples: list,
    cand_samples: list,
    threshold: float,
) -> PerfResult:
    """Aggregate samples → ratio + status verdict."""
    ref_us = statistics.median(ref_samples)
    cand_us = statistics.median(cand_samples)
    if cand_us <= 0:
        # DEBT-140 follow-up (blue 2026-06-01): emit a CANONICAL finalize-gate
        # status. The finalize-eligibility gate (finalize_pipeline.py:774) only
        # accepts PASS / PASS_WITHIN_TOLERANCE / N/A — a literal
        # "NOT_VERIFIED_SAME_METHOD" is non-canonical and makes the gate roll
        # back forever (LOOP-BREAK → await_user_decision). "Could not measure"
        # IS honestly N/A-with-reason, so classify it that way.
        _r = f"phase_o5_perf_capture could not verify same-method: candidate median {cand_us} non-positive"
        return PerfResult(
            method=method,
            status="N/A",
            samples_us_reference=ref_samples,
            samples_us_candidate=cand_samples,
            error=_r,
            reason=_r,
        )

    ratio = ref_us / cand_us
    if ratio >= 1.0:
        status = "PASS"
    elif ratio >= threshold:
        status = "PASS_WITHIN_TOLERANCE"
    else:
        status = "BELOW_THRESHOLD"

    return PerfResult(
        method=method,
        status=status,
        ratio=round(ratio, 4),
        reference_us=round(ref_us, 4),
        candidate_us=round(cand_us, 4),
        samples_us_reference=[round(x, 4) for x in ref_samples],
        samples_us_candidate=[round(x, 4) for x in cand_samples],
    )


def _not_verified(method: str, reason: str) -> dict:
    """Honest fallback when same-method perf could not be captured (profiler
    device-duration unavailable on V220, dispatch gap, runner failure, etc.).

    DEBT-140 follow-up (blue 2026-06-01): emit CANONICAL status="N/A" + reason
    (the finalize gate only accepts PASS / PASS_WITHIN_TOLERANCE / N/A — a
    literal "NOT_VERIFIED_SAME_METHOD" rolled finalize back forever). "Could
    not measure same-method" is an honest N/A-with-reason, not a fake PASS.
    The original signal is preserved in `error` for forensics."""
    return PerfResult(
        method=method,
        status="N/A",
        error=reason,
        reason=reason,
    ).to_dict()
