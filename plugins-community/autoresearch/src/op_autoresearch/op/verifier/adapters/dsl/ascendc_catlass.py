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

"""AscendC + CATLASS pybind DSL adapter for KernelBench / AR verify."""

from __future__ import annotations

import logging
import os
import shutil
import textwrap
from typing import Any, Dict, Optional

from op_autoresearch.core.worker.eval_config import resolve_eval_timeout
from op_autoresearch.op.utils.catlass_runtime import arch_to_catlass_arch

from .base import DSLAdapter, MaterializeSpec

logger = logging.getLogger(__name__)

# KernelVerifier arch → catlass cmake arch id (2201 / 3510).
# Pass BOTH -DNPU_ARCH and -DCATLASS_ARCH: pipeline CMakeLists vary
# (option(NPU_ARCH) vs if(NOT DEFINED CATLASS_ARCH)).
arch_to_npu_arch = arch_to_catlass_arch


class DSLAdapterAscendCCatlass(DSLAdapter):
    """CATLASS pybind + ModelNew wrapper on Ascend NPU."""

    impl_func_name_template = "ModelNew"
    model_wrapper_eval_mode = True
    uses_cannbench_precision = True

    def __init__(self) -> None:
        self._setup_arch: Optional[str] = None
        self._setup_catlass_root: Optional[str] = None

    def get_import_statements(self, framework: str) -> str:
        return "import torch\nimport torch_npu\n"

    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        return "from kernel import ModelNew\n"

    # catlass kernel handoff is a directory: the catlass_op/ project
    # subtree sitting next to a Python wrapper (kernel.py).
    kernel_arg_is_directory = True
    kernel_project_dir_name = "catlass_op"
    kernel_project_files = [
        "catlass_op/kernel/catlass_kernel.asc",
        "catlass_op/include/catlass_kernel.h",
        "catlass_op/src/catlass_torch.cpp",
        "catlass_op/CMakeLists.txt",
    ]

    def get_special_setup_code(self, framework: str = "torch") -> str:
        # arch + catlass_root resolved at prepare_config() time into
        # self._setup_arch / self._setup_catlass_root. Fall back to
        # the ascend910b3 / env-driven defaults if prepare_config
        # was bypassed (e.g. unit tests).
        arch = self._setup_arch
        if not arch:
            raise RuntimeError(
                "ascendc_catlass requires config['arch'] before special setup runs"
            )
        catlass_root = self._setup_catlass_root
        catlass_arch = arch_to_catlass_arch(arch)
        catlass_root_repr = repr(catlass_root) if catlass_root else "None"
        timeout = resolve_eval_timeout()
        return textwrap.dedent(
            f"""
        # --- ascendc_catlass rebuild ---
        import os
        from op_autoresearch.op.utils.catlass_runtime import (
            ensure_catlass_library as _ensure_catlass_library,
        )

        _task_root = os.path.dirname(os.path.abspath(__file__))
        _catlass_arch = "{catlass_arch}"
        print(
            f"[INFO]: catlass cmake flags: -DNPU_ARCH={{_catlass_arch}} "
            f"-DCATLASS_ARCH={{_catlass_arch}} -DCATLASS_ROOT={catlass_root_repr}"
        )
        _lib_so = _ensure_catlass_library(
            _task_root,
            arch={arch!r},
            catlass_root={catlass_root_repr},
            timeout={timeout},
        )
        print(f"[INFO]: catlass library ready: {{_lib_so}}")
        """
        )

    # ------------------------------------------------------------------
    # Extension hooks (override DSLAdapter defaults)
    # ------------------------------------------------------------------

    def materialize_impl(self, spec: MaterializeSpec) -> None:
        """Write the primary wrapper + copy catlass_op tree into verify_dir."""
        verify_dir = spec.verify_dir
        op_name = spec.op_name
        kernel_file = os.path.join(
            verify_dir, self.entry_filename_template.format(op_name=op_name))
        with open(kernel_file, "w", encoding="utf-8") as f:
            f.write(spec.impl_code)

        from op_autoresearch.op.utils.catlass_paths import merge_catlass_config
        cfg = dict(spec.config) if spec.config else {}
        merge_catlass_config(cfg, task_info=spec.task_info)
        catlass_op_src = cfg.get("catlass_op_src")
        if not catlass_op_src or not os.path.isdir(catlass_op_src):
            raise ValueError(
                f"[{op_name}] catlass_op_src not found or not a directory. "
                f"Set config catlass_op_src / task_dir + catlass_op/, or task.yaml "
                f"catlass.op_dir. Got: {catlass_op_src!r}"
            )
        catlass_op_dst = os.path.join(verify_dir, "catlass_op")
        if os.path.isdir(catlass_op_dst):
            shutil.rmtree(catlass_op_dst)
        shutil.copytree(
            catlass_op_src,
            catlass_op_dst,
            ignore=shutil.ignore_patterns("build", "__pycache__", "*.pyc", "*.so"),
        )
        logger.debug("[%s] catlass_op copied: %s -> %s",
                     op_name, catlass_op_src, catlass_op_dst)

    def expected_artifacts(self, verify_dir: str, op_name: str,
                           framework: str, dsl_filename_hint: str) -> list:
        wrapper_name = self.entry_filename_template.format(op_name=op_name)
        return [
            os.path.join(verify_dir, wrapper_name),
            os.path.join(verify_dir, "catlass_op", "CMakeLists.txt"),
        ]

    def prepare_config(self, config: Dict[str, Any],
                       task_info: Optional[Dict[str, Any]] = None) -> None:
        """Resolve CATLASS_ROOT + catlass_op_src into config, and remember
        arch / catlass_root for get_special_setup_code (which has the
        ABC-fixed signature ``(framework)``).
        """
        from op_autoresearch.op.utils.catlass_paths import merge_catlass_config
        merge_catlass_config(config, task_info=task_info)
        # Stash for get_special_setup_code; not a config key so that
        # cross-DSL config inspection stays clean.
        self._setup_arch = config.get("arch")
        self._setup_catlass_root = config.get("catlass_root")

    benchmark_requires_l2_clear = False
    profile_via_python_script = True

    def post_iteration_cleanup(self, verify_dir: str) -> None:
        """Drop catlass_op/build for this round; keep profile JSON + sources."""
        build_dir = os.path.join(verify_dir, "catlass_op", "build")
        if os.path.isdir(build_dir):
            shutil.rmtree(build_dir, ignore_errors=True)

    def materialize_project_tree(self, dst_dir: str,
                                 project_src: Optional[str],
                                 project_dir_name: Optional[str] = None) -> None:
        """Copy catlass_op tree into ``dst_dir`` and patch its
        CMakeLists.txt for the AR task layout.
        """
        if not project_src:
            return
        from op_autoresearch.op.utils.catlass_paths import patch_catlass_op_cmake
        dst = os.path.join(dst_dir, project_dir_name or self.kernel_project_dir_name)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(
            project_src,
            dst,
            ignore=shutil.ignore_patterns("build", "__pycache__", "*.so"),
        )
        patch_catlass_op_cmake(dst)
