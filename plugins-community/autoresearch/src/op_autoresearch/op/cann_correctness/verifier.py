# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""Generate verification and profiling projects for CANN-Bench tasks."""

import json
import logging
import os
import shutil
from dataclasses import dataclass

import yaml

from op_autoresearch import get_project_root
from op_autoresearch.core.worker.eval_config import (
    resolve_run_times,
    resolve_warmup_times,
)
from op_autoresearch.op.verifier.profile_project import ProfileProjectSpec
from op_autoresearch.op.verifier.project_common import (
    adapter_template_vars,
    candidate_profile_enabled,
    copy_required_files,
    materialize_implementation,
    render_base_profile,
    render_template_to_file,
    stage_runtime,
)

logger = logging.getLogger(__name__)

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_TEMPLATES_DIR = os.path.join(_PKG_DIR, "templates")
_CORE_PY_PATH = os.path.join(_PKG_DIR, "core.py")

PROF_CANN_BASE_TEMPLATE_PATH = os.path.join(_TEMPLATES_DIR, "prof_cann_base.j2")
PROF_CANN_GENERATION_TEMPLATE_PATH = os.path.join(
    _TEMPLATES_DIR, "prof_cann_generation.j2"
)

CANN_BENCH_SRC_DIR = os.environ.get(
    "OP_AUTORESEARCH_CANN_BENCH_SRC",
    os.path.abspath(
        os.path.join(get_project_root(), "..", "..", "thirdparty", "cann-bench", "src")
    ),
)
CANN_DATA_FILES = ["proto.yaml", "golden.py", "cases.yaml"]


@dataclass(frozen=True)
class _ProfileRenderContext:
    device_id: int
    warmup_times: int
    run_times: int
    schema: str


def generate_cann_verify_project(
    verifier, impl_code: str, verify_dir: str, device_id: int = 0
) -> None:
    """Generate CANN-Bench verification project files into verify_dir."""
    logger.info(
        "[%s] generating CANN-Bench verification project in %s on device %s",
        verifier.op_name,
        verify_dir,
        device_id,
    )
    problem_dir = _require_problem_dir(verifier)
    _copy_cann_inputs(problem_dir, verify_dir, overwrite=True)
    _stage_correctness_module(verify_dir, overwrite=True)
    materialize_implementation(verifier, impl_code, verify_dir)

    operator_spec = _load_operator_spec(verify_dir)
    template_vars = _verify_template_vars(verifier, device_id, operator_spec)
    verify_file = os.path.join(verify_dir, f"verify_{verifier.op_name}.py")
    render_template_to_file(
        os.path.join(_TEMPLATES_DIR, "verify_cann.j2"),
        verify_file,
        template_vars,
    )


def generate_cann_profile_project(
    verifier,
    verify_dir: str,
    spec: ProfileProjectSpec | None = None,
) -> None:
    """Generate CANN-Bench base and implementation profile scripts."""
    spec = spec or ProfileProjectSpec()
    warmup_times = resolve_warmup_times(spec.warmup_times)
    run_times = resolve_run_times(spec.run_times)
    logger.info(
        "[%s] generating CANN-Bench profile project in %s on device %s",
        verifier.op_name,
        verify_dir,
        spec.device_id,
    )
    problem_dir = _require_problem_dir(verifier)
    _copy_cann_inputs(problem_dir, verify_dir, overwrite=False)
    _stage_correctness_module(verify_dir, overwrite=False)
    context = _ProfileRenderContext(
        device_id=spec.device_id,
        warmup_times=warmup_times,
        run_times=run_times,
        schema=_load_operator_spec(verify_dir).get("schema", ""),
    )

    if spec.skip_base:
        logger.info("[%s] skipping CANN base profile generation", verifier.op_name)
    else:
        _generate_base_profile(verifier, verify_dir, context)

    if candidate_profile_enabled(verifier, "CANN", logger):
        _generate_implementation_profile(verifier, verify_dir, context)


def _require_problem_dir(verifier) -> str:
    problem_dir = verifier.config.get("cann_problem_dir")
    if problem_dir and os.path.exists(problem_dir):
        return problem_dir
    raise ValueError(f"cann_problem_dir is missing or does not exist: {problem_dir}")


def _copy_cann_inputs(problem_dir: str, verify_dir: str, *, overwrite: bool) -> None:
    copy_required_files(
        problem_dir,
        verify_dir,
        CANN_DATA_FILES,
        overwrite=overwrite,
        suite="CANN",
    )

    desc_source = os.path.join(problem_dir, "desc.md")
    desc_destination = os.path.join(verify_dir, "desc.md")
    if os.path.exists(desc_source) and (
        overwrite or not os.path.exists(desc_destination)
    ):
        shutil.copy2(desc_source, desc_destination)


def _stage_correctness_module(verify_dir: str, *, overwrite: bool) -> None:
    stage_runtime(
        _CORE_PY_PATH,
        verify_dir,
        "cann_correctness.py",
        overwrite=overwrite,
    )


def _load_operator_spec(verify_dir: str) -> dict:
    with open(
        os.path.join(verify_dir, "proto.yaml"), "r", encoding="utf-8"
    ) as file:
        proto = yaml.safe_load(file)
    return proto.get("operator", {})


def _verify_template_vars(verifier, device_id: int, operator_spec: dict) -> dict:
    outputs = operator_spec.get("outputs", [])
    ignored_outputs = [
        index
        for index, output in enumerate(outputs)
        if output.get("compare", True) is False
    ]
    precision_thresholds = operator_spec.get("precision_thresholds")
    variables = {
        "op_name": verifier.op_name,
        "framework": verifier.framework,
        "backend": verifier.backend,
        "arch": verifier.arch,
        "dsl": verifier.dsl,
        "device_id": device_id,
        "precision_thresholds_yaml": (
            json.dumps(precision_thresholds) if precision_thresholds else "None"
        ),
        "ignore_output_indices": ignored_outputs,
        "schema": operator_spec.get("schema", ""),
        "cann_bench_src_dir": CANN_BENCH_SRC_DIR,
    }
    variables.update(adapter_template_vars(verifier, device_id))
    return variables


def _generate_base_profile(
    verifier,
    verify_dir: str,
    context: _ProfileRenderContext,
) -> None:
    destination = render_base_profile(
        verifier,
        verify_dir,
        context.device_id,
        _profile_identity_vars(verifier, context),
        PROF_CANN_BASE_TEMPLATE_PATH,
    )
    logger.info(
        "[%s] wrote CANN base profile script: %s",
        verifier.op_name,
        destination,
    )


def _generate_implementation_profile(
    verifier,
    verify_dir: str,
    context: _ProfileRenderContext,
) -> None:
    template_vars = _profile_identity_vars(verifier, context)
    template_vars.update(
        _get_cann_common_template_vars(verifier, context.device_id)
    )
    destination = os.path.join(
        verify_dir, f"profile_{verifier.op_name}_generation.py"
    )
    render_template_to_file(
        PROF_CANN_GENERATION_TEMPLATE_PATH, destination, template_vars
    )
    logger.info(
        "[%s] wrote CANN implementation profile script: %s",
        verifier.op_name,
        destination,
    )


def _profile_identity_vars(verifier, context: _ProfileRenderContext) -> dict:
    return {
        "op_name": verifier.op_name,
        "backend": verifier.backend,
        "arch": verifier.arch,
        "dsl": verifier.dsl,
        "device_id": context.device_id,
        "warmup_times": context.warmup_times,
        "run_times": context.run_times,
        "schema": context.schema,
        "cann_bench_src_dir": CANN_BENCH_SRC_DIR,
    }


def _get_cann_common_template_vars(verifier, device_id: int) -> dict:
    """Get common template variables for CANN profile scripts."""
    variables = adapter_template_vars(verifier, device_id)
    variables["cann_bench_src_dir"] = CANN_BENCH_SRC_DIR
    return variables
