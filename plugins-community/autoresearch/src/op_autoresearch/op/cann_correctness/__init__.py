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

"""CANN-Bench evaluation standard — single, isolated owner.

Everything cannbench-specific (MERE/MARE precision, per-op thresholds/registries,
FP64 dual reference, the cann verify/profile project generators and their
templates, task loading) lives in this package. op_autoresearch core reaches it only through
the exports below; the one switch is the boolean ``DSLAdapter.uses_cannbench_precision``
(True -> cannbench precision path; False/default -> framework precision).

The package root is also used while generating verification projects.  Keep
that path importable on an orchestrator without the worker-only torch runtime;
comparison APIs from ``core`` are loaded on first access.
"""

from importlib import import_module

from ._paths import CORE_PY_PATH, TEMPLATES_DIR, stage_core_into
from .task_loader import (
    get_cann_task_desc_for_prompt,
    inject_cann_into_config,
    is_cann_task_dir,
    load_cann_desc,
    load_cann_golden,
    load_cann_proto,
    load_cann_task_for_runner,
    load_cann_task_source,
)
from .verifier import (
    CANN_BENCH_SRC_DIR,
    generate_cann_profile_project,
    generate_cann_verify_project,
)
from .verify_snippets import (
    compare_snippet,
    reference_call_snippet,
)

_CORE_EXPORTS = frozenset({
    "OutputAssertionContext",
    "TensorComparisonOptions",
    "assert_outputs",
    "compare_tensors",
    "dual_reference",
    "set_seed",
    "validate_index_output",
})


def __getattr__(name: str):
    """Load torch-dependent comparison APIs only in the worker runtime."""
    if name not in _CORE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.core"), name)
    globals()[name] = value
    return value

__all__ = [
    "CORE_PY_PATH",
    "TEMPLATES_DIR",
    "stage_core_into",
    "TensorComparisonOptions",
    "OutputAssertionContext",
    "compare_tensors",
    "assert_outputs",
    "dual_reference",
    "validate_index_output",
    "set_seed",
    "reference_call_snippet",
    "compare_snippet",
    "is_cann_task_dir",
    "load_cann_task_source",
    "inject_cann_into_config",
    "get_cann_task_desc_for_prompt",
    "load_cann_task_for_runner",
    "load_cann_proto",
    "load_cann_golden",
    "load_cann_desc",
    "generate_cann_verify_project",
    "generate_cann_profile_project",
    "CANN_BENCH_SRC_DIR",
]
