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

"""Shared project-building primitives for verification suites."""

import logging
import os
import shutil
from collections.abc import Iterable

from jinja2 import Template

from op_autoresearch.op.verifier.adapters.dsl.base import MaterializeSpec
from op_autoresearch.op.verifier.adapters.factory import (
    get_backend_adapter,
    get_dsl_adapter,
    get_framework_adapter,
)


def copy_required_files(
    problem_dir: str,
    verify_dir: str,
    file_names: Iterable[str],
    *,
    overwrite: bool,
    suite: str,
) -> None:
    """Copy one suite's required problem description files."""
    for file_name in file_names:
        source = os.path.join(problem_dir, file_name)
        destination = os.path.join(verify_dir, file_name)
        if not overwrite and os.path.exists(destination):
            continue
        if not os.path.exists(source):
            raise FileNotFoundError(f"Missing required {suite} file: {source}")
        shutil.copy2(source, destination)


def stage_runtime(
    source_path: str,
    verify_dir: str,
    file_name: str,
    *,
    overwrite: bool,
) -> None:
    """Stage a suite runtime module into a generated project."""
    destination = os.path.join(verify_dir, file_name)
    if not overwrite and os.path.exists(destination):
        return
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Missing verification runtime: {source_path}")
    shutil.copy2(source_path, destination)


def materialize_implementation(verifier, impl_code: str, verify_dir: str) -> None:
    """Write a single-file implementation or delegate a project-backed DSL."""
    dsl_adapter = get_dsl_adapter(verifier.dsl)
    if dsl_adapter.kernel_arg_is_directory:
        dsl_adapter.prepare_config(verifier.config, task_info=None)
        dsl_adapter.materialize_impl(
            MaterializeSpec(
                impl_code=impl_code,
                verify_dir=verify_dir,
                op_name=verifier.op_name,
                framework=verifier.framework,
                dsl_name=verifier.dsl,
                config=verifier.config,
            )
        )
        return
    destination = os.path.join(
        verify_dir, f"{verifier.op_name}_{verifier.dsl}_impl.py"
    )
    imports = dsl_adapter.get_import_statements(verifier.framework)
    with open(destination, "w", encoding="utf-8") as file:
        file.write(imports + impl_code)


def adapter_template_vars(verifier, device_id: int) -> dict:
    """Build the adapter-owned variables shared by generated projects."""
    framework, dsl, backend = _adapters_for(verifier)
    backend.setup_environment(device_id, verifier.arch)
    names = ("device_setup_code", "dsl_imports", "create_impl_code")
    values = (
        _prepared_device_setup(verifier, framework, device_id),
        _dsl_import_block(verifier, dsl),
        verifier.prepare_code_lines(
            dsl.create_impl_module(verifier.framework, framework)
        ),
    )
    return dict(zip(names, values))


def base_profile_template_vars(
    verifier, device_id: int, identity_vars: dict
) -> dict:
    """Add framework device setup to suite-specific base profile values."""
    framework, _dsl, backend = _adapters_for(verifier)
    backend.setup_environment(device_id, verifier.arch)
    return {
        **identity_vars,
        "device_setup_code": _prepared_device_setup(
            verifier, framework, device_id
        ),
    }


def _adapters_for(verifier):
    factories = (get_framework_adapter, get_dsl_adapter, get_backend_adapter)
    names = (verifier.framework, verifier.dsl, verifier.backend)
    return tuple(factory(name) for factory, name in zip(factories, names))


def _prepared_device_setup(verifier, framework_adapter, device_id: int):
    setup = framework_adapter.get_device_setup_code(
        verifier.backend, verifier.arch, device_id
    )
    return verifier.prepare_code_lines(setup)


def render_base_profile(
    verifier,
    verify_dir: str,
    device_id: int,
    identity_vars: dict,
    template_path: str,
) -> str:
    """Render one suite's framework baseline profile script."""
    variables = base_profile_template_vars(verifier, device_id, identity_vars)
    destination = os.path.join(
        verify_dir, f"profile_{verifier.op_name}_base.py"
    )
    render_template_to_file(template_path, destination, variables)
    return destination


def candidate_profile_enabled(
    verifier, suite: str, suite_logger: logging.Logger
) -> bool:
    """Report and return whether candidate profiling is enabled."""
    if verifier.profile_generation_enabled:
        return True
    suite_logger.info(
        "[%s] skipping %s candidate profile because verification did not pass",
        verifier.op_name,
        suite,
    )
    return False


def render_template_to_file(
    template_path: str, destination: str, template_vars: dict
) -> None:
    """Render a Jinja project template to one destination file."""
    with open(template_path, "r", encoding="utf-8") as file:
        template = Template(file.read())
    rendered = template.render(**template_vars)
    with open(destination, "w", encoding="utf-8") as file:
        file.write(rendered)


def _dsl_import_block(verifier, dsl_adapter) -> str:
    imports = dsl_adapter.get_import_statements(verifier.framework)
    impl_import = _normalize_impl_import(
        dsl_adapter.get_impl_import(
            verifier.op_name, verifier.impl_func_name
        ).strip()
    )
    dsl_adapter.prepare_config(verifier.config, task_info=None)
    special_setup = dsl_adapter.get_special_setup_code(
        framework=verifier.framework
    )
    parts = [imports, impl_import]
    if special_setup:
        parts.append(special_setup)
    return "\n".join(parts)


def _normalize_impl_import(statement: str) -> str:
    parts = statement.split()
    if not (
        len(parts) >= 4
        and parts[0] == "from"
        and parts[1]
        and parts[1][0].isdigit()
    ):
        return statement
    module_name = parts[1]
    import_name = parts[3]
    return "\n".join(
        (
            "import importlib.util",
            "import sys",
            (
                "spec = importlib.util.spec_from_file_location"
                f"('{module_name}', '{module_name}.py')"
            ),
            "module = importlib.util.module_from_spec(spec)",
            f"sys.modules['{module_name}'] = module",
            "spec.loader.exec_module(module)",
            f"{import_name} = getattr(module, '{import_name}')",
        )
    )
