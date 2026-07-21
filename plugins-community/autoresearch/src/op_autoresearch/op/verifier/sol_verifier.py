# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

"""Generate verification and profiling projects for SOL-ExecBench tasks."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from op_autoresearch import get_project_root
from op_autoresearch.core.worker.eval_config import (
    resolve_run_times,
    resolve_warmup_times,
)
from op_autoresearch.op.verifier import project_common
from op_autoresearch.op.verifier.profile_project import ProfileProjectSpec

logger = logging.getLogger(__name__)

_RESOURCE_ROOT = Path(get_project_root(), "op", "resources")
_TEMPLATE_ROOT = _RESOURCE_ROOT / "templates"
_SOL_CORRECTNESS_PATH = _RESOURCE_ROOT / "utils" / "sol_correctness.py"
_SOL_EXECBENCH_SRC_DIR = str(
    (Path(get_project_root()) / ".." / ".." / "thirdparty" / "sol-execbench" / "src").resolve()
)
_SOL_INPUT_FILES = ("definition.json", "workload.jsonl", "reference.py")

PROF_SOL_BASE_TEMPLATE_PATH = _TEMPLATE_ROOT / "prof_sol_base_template.j2"
PROF_SOL_GENERATION_TEMPLATE_PATH = _TEMPLATE_ROOT / "prof_sol_generation_template.j2"
_VERIFY_TEMPLATE_PATH = _TEMPLATE_ROOT / "verify_sol_template.j2"


@dataclass(frozen=True)
class _ProfileRenderContext:
    device_id: int
    warmup_times: int
    run_times: int


def generate_sol_verify_project(
    verifier, impl_code: str, verify_dir: str, device_id: int = 0
) -> None:
    """Generate a SOL-ExecBench verification project."""
    logger.info(
        "[%s] generating SOL-ExecBench verification project in %s on device %s",
        verifier.op_name,
        verify_dir,
        device_id,
    )
    problem_dir = _require_problem_dir(verifier)
    _copy_sol_inputs(problem_dir, verify_dir, overwrite=True)
    _stage_correctness_module(verify_dir, overwrite=True)
    project_common.materialize_implementation(verifier, impl_code, verify_dir)

    template_vars = _verify_template_vars(verifier, device_id)
    destination = os.path.join(verify_dir, f"verify_{verifier.op_name}.py")
    project_common.render_template_to_file(
        _VERIFY_TEMPLATE_PATH, destination, template_vars
    )


def generate_sol_profile_project(
    verifier,
    verify_dir: str,
    spec: ProfileProjectSpec | None = None,
) -> None:
    """Generate SOL-ExecBench base and implementation profile scripts."""
    spec = spec or ProfileProjectSpec()
    context = _ProfileRenderContext(
        device_id=spec.device_id,
        warmup_times=resolve_warmup_times(spec.warmup_times),
        run_times=resolve_run_times(spec.run_times),
    )
    logger.info(
        "[%s] generating SOL-ExecBench profile project in %s on device %s",
        verifier.op_name,
        verify_dir,
        context.device_id,
    )
    problem_dir = _require_problem_dir(verifier)
    _copy_sol_inputs(problem_dir, verify_dir, overwrite=False)
    _stage_correctness_module(verify_dir, overwrite=False)

    if spec.skip_base:
        logger.info("[%s] skipping SOL base profile generation", verifier.op_name)
    else:
        _generate_base_profile(verifier, verify_dir, context)

    if project_common.candidate_profile_enabled(verifier, "SOL", logger):
        _generate_implementation_profile(verifier, verify_dir, context)


def _require_problem_dir(verifier) -> str:
    problem_dir = verifier.config.get("sol_problem_dir")
    if problem_dir and os.path.exists(problem_dir):
        return problem_dir
    raise ValueError(f"sol_problem_dir is missing or does not exist: {problem_dir}")


def _copy_sol_inputs(problem_dir: str, verify_dir: str, *, overwrite: bool) -> None:
    project_common.copy_required_files(
        problem_dir,
        verify_dir,
        _SOL_INPUT_FILES,
        overwrite=overwrite,
        suite="SOL",
    )


def _stage_correctness_module(verify_dir: str, *, overwrite: bool) -> None:
    project_common.stage_runtime(
        _SOL_CORRECTNESS_PATH,
        verify_dir,
        "sol_correctness.py",
        overwrite=overwrite,
    )


def _verify_template_vars(verifier, device_id: int) -> dict:
    attributes = ("op_name", "framework", "backend", "arch", "dsl")
    identity = {name: getattr(verifier, name) for name in attributes}
    identity["device_id"] = device_id
    return identity | _get_sol_common_template_vars(verifier, device_id)


def _get_sol_common_template_vars(verifier, device_id: int) -> dict:
    """Build adapter-owned template values shared by SOL scripts."""
    variables = project_common.adapter_template_vars(verifier, device_id)
    variables["sol_execbench_src_dir"] = _SOL_EXECBENCH_SRC_DIR
    return variables


def _generate_base_profile(
    verifier, verify_dir: str, context: _ProfileRenderContext
) -> None:
    destination = project_common.render_base_profile(
        verifier,
        verify_dir,
        context.device_id,
        _profile_identity_vars(verifier, context),
        PROF_SOL_BASE_TEMPLATE_PATH,
    )
    logger.info(
        "[%s] wrote SOL base profile script: %s",
        verifier.op_name,
        destination,
    )


def _generate_implementation_profile(
    verifier, verify_dir: str, context: _ProfileRenderContext
) -> None:
    template_vars = _profile_identity_vars(verifier, context)
    template_vars["dsl"] = verifier.dsl
    template_vars.update(
        _get_sol_common_template_vars(verifier, context.device_id)
    )
    destination = os.path.join(
        verify_dir, f"profile_{verifier.op_name}_generation.py"
    )
    project_common.render_template_to_file(
        PROF_SOL_GENERATION_TEMPLATE_PATH, destination, template_vars
    )
    logger.info(
        "[%s] wrote SOL implementation profile script: %s",
        verifier.op_name,
        destination,
    )


def _profile_identity_vars(
    verifier, context: _ProfileRenderContext
) -> dict:
    return {
        "op_name": verifier.op_name,
        "backend": verifier.backend,
        "arch": verifier.arch,
        "device_id": context.device_id,
        "warmup_times": context.warmup_times,
        "run_times": context.run_times,
        "sol_execbench_src_dir": _SOL_EXECBENCH_SRC_DIR,
    }
