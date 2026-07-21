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

"""AscendC direct-invoke DSL adapter.

The canonical ``ascendc`` format is now a CANNBot-style project tree next
to a small Python ``ModelNew`` wrapper:

    kernel.py
    ascendc_op/
        CMakeLists.txt
        op_kernel/
        op_host/
        op_extension/
        scripts/

The wrapper is responsible for loading the built shared object and calling
``torch.ops.npu.<op>(...)``.  The adapter owns project-tree copy, CMake
rebuild, arch patching, and KernelVerifier integration.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from op_autoresearch.core.worker.eval_config import resolve_eval_timeout
from op_autoresearch.op.utils.arch_normalize import ascend_direct_invoke_npu_arch

from .base import (
    DSLAdapter,
    MaterializeSpec,
    render_code_template,
)

logger = logging.getLogger(__name__)

_TEXT_SUFFIXES = {
    ".asc",
    ".c",
    ".cc",
    ".cmake",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
}
_TEXT_FILENAMES = {"CMakeLists.txt", "CMakePresets.json", "run.sh"}
_IGNORE_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
    "CMakeFiles",
    "output",
}
_EDITABLE_PROJECT_ROOTS = {
    "CMakeLists.txt",
    "CMakePresets.json",
    "cmake",
    "op_kernel",
    "op_extension",
    "op_host",
    "src",
    "include",
    "common",
}
_CMAKE_NPU_ARCH_FLAG_RE = re.compile(r"--npu-arch=[A-Za-z0-9_-]+")
_CMAKE_NPU_ARCH_VAR_RE = re.compile(r"\b(?:NPU_ARCH|ASCENDC_NPU_ARCH)\b")
_CMAKE_TEXT_SUFFIXES = {".cmake", ".txt"}


def _copy_project_tree(src: str, dst: str) -> None:
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns(
            ".git",
            ".pytest_cache",
            "__pycache__",
            "build",
            "CMakeFiles",
            "output",
            "*.o",
            "*.pyc",
            "*.so",
        ),
    )


def _patch_cmake_npu_arch(project_dir: str, npu_arch: str) -> bool:
    """Rewrite literal ``--npu-arch=...`` flags in a copied project."""
    cmake_path = os.path.join(os.path.abspath(project_dir), "CMakeLists.txt")
    if not os.path.isfile(cmake_path):
        return False
    with open(cmake_path, "r", encoding="utf-8") as f:
        text = f.read()
    patched, count = _CMAKE_NPU_ARCH_FLAG_RE.subn(f"--npu-arch={npu_arch}", text)
    if count:
        with open(cmake_path, "w", encoding="utf-8") as f:
            f.write(patched)
    return bool(count)


def _iter_cmake_text_files(project_dir: str):
    root = Path(project_dir).resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in _IGNORE_DIRS for part in rel_parts):
            continue
        if path.name == "CMakeLists.txt" or path.suffix in _CMAKE_TEXT_SUFFIXES:
            yield path


def _project_consumes_cmake_npu_arch_vars(project_dir: str) -> bool:
    """Return whether project CMake files reference OP_AUTORESEARCH's arch variables."""
    for path in _iter_cmake_text_files(project_dir):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if _CMAKE_NPU_ARCH_VAR_RE.search(text):
            return True
    return False


def _assert_cmake_has_npu_arch_channel(project_dir: str, op_name: str) -> None:
    """Fail before build when OP_AUTORESEARCH cannot steer direct-invoke arch.

    A project is acceptable when either:
    - it has a literal ``--npu-arch=...`` flag for us to patch, or
    - its CMake files consume ``NPU_ARCH`` / ``ASCENDC_NPU_ARCH`` passed by
      the verifier runtime.
    """
    cmake_path = os.path.join(os.path.abspath(project_dir), "CMakeLists.txt")
    if not os.path.isfile(cmake_path):
        return
    with open(cmake_path, "r", encoding="utf-8") as f:
        text = f.read()
    if _CMAKE_NPU_ARCH_FLAG_RE.search(text):
        return
    if _project_consumes_cmake_npu_arch_vars(project_dir):
        return
    raise ValueError(
        f"[{op_name}] ascendc CMake has no controllable NPU arch channel. "
        "Add a literal --npu-arch=<dav-token> flag to CMakeLists.txt, or "
        "consume ${NPU_ARCH} / ${ASCENDC_NPU_ARCH} in the project CMake files."
    )


class DSLAdapterAscendC(DSLAdapter):
    """CANNBot-style AscendC direct-invoke project adapter."""

    impl_func_name_template = "ModelNew"
    model_wrapper_eval_mode = True
    profile_via_python_script = True
    benchmark_requires_l2_clear = False
    uses_cannbench_precision = True

    kernel_arg_is_directory = True
    kernel_project_dir_name = "ascendc_op"
    kernel_project_files = ["ascendc_op/CMakeLists.txt"]

    def __init__(self) -> None:
        self._setup_arch: Optional[str] = None
        self._setup_npu_arch: Optional[str] = None
        self._setup_timeout: Optional[int] = None
        self._setup_project_dir_name: Optional[str] = None

    def get_import_statements(self, framework: str) -> str:
        if framework != "torch":
            raise ValueError(
                f"ascendc direct-invoke currently supports torch only, got {framework!r}"
            )
        return "import torch\nimport torch_npu\n"

    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        return "from kernel import ModelNew\n"

    def prepare_config(
        self,
        config: Dict[str, Any],
        task_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        task_info = task_info or {}
        project_dir_name = (
            config.get("ascendc_op_dir")
            or config.get("ascendc_project_dir")
            or task_info.get("ascendc_op_dir")
            or task_info.get("ascendc_project_dir")
            or self.kernel_project_dir_name
        )
        config["ascendc_op_dir"] = project_dir_name

        project_src = (
            config.get("ascendc_op_src")
            or config.get("ascendc_project_src")
            or task_info.get("ascendc_op_src")
            or task_info.get("ascendc_project_src")
            or task_info.get("kernel_project_src")
        )
        task_dir = task_info.get("task_dir")
        if not project_src and task_dir:
            project_src = os.path.join(task_dir, project_dir_name)
        if project_src:
            config["ascendc_op_src"] = os.path.abspath(project_src)

        arch = config.get("arch")
        self._setup_arch = arch
        self._setup_npu_arch = ascend_direct_invoke_npu_arch(arch or "")
        self._setup_timeout = int(
            config.get("verify_timeout")
            or config.get("timeout")
            or resolve_eval_timeout()
        )
        self._setup_project_dir_name = project_dir_name

    def get_special_setup_code(self, framework: str = "torch") -> str:
        arch = self._setup_arch
        npu_arch = self._setup_npu_arch
        if not arch:
            raise RuntimeError(
                "ascendc requires config['arch'] before special setup runs"
            )
        if not npu_arch:
            raise RuntimeError(
                f"ascendc cannot derive direct-invoke --npu-arch from {arch!r}"
            )
        project_dir_name = (
            self._setup_project_dir_name or self.kernel_project_dir_name
        )
        timeout = self._setup_timeout or resolve_eval_timeout()
        return render_code_template(
            "ascendc_setup.py.j2",
            project_dir_name_repr=repr(project_dir_name),
            npu_arch=npu_arch,
            timeout=timeout,
        )

    def materialize_impl(self, spec: MaterializeSpec) -> None:
        verify_dir = spec.verify_dir
        op_name = spec.op_name
        kernel_file = os.path.join(
            verify_dir, self.entry_filename_template.format(op_name=op_name)
        )
        with open(kernel_file, "w", encoding="utf-8") as f:
            f.write(spec.impl_code)

        cfg = dict(spec.config or {})
        project_src = cfg.get("ascendc_op_src")
        if not project_src and spec.task_info:
            project_src = (
                spec.task_info.get("ascendc_op_src")
                or spec.task_info.get("ascendc_project_src")
                or spec.task_info.get("kernel_project_src")
            )
        project_dir_name = cfg.get("ascendc_op_dir") or self.kernel_project_dir_name
        if (not project_src or not os.path.isdir(project_src)) and cfg.get("task_dir"):
            candidate = os.path.join(os.path.abspath(cfg["task_dir"]), project_dir_name)
            if os.path.isdir(candidate):
                project_src = candidate
        if not project_src or not os.path.isdir(project_src):
            raise ValueError(
                f"[{op_name}] ascendc_op_src not found or not a directory. "
                "Pass --kernel pointing to the ascendc_op project tree, or "
                "set config/task_info ascendc_op_src. "
                f"Got: {project_src!r}"
            )

        project_dst = os.path.join(verify_dir, project_dir_name)
        _copy_project_tree(project_src, project_dst)
        _assert_cmake_has_npu_arch_channel(project_dst, op_name)
        npu_arch = self._setup_npu_arch
        if npu_arch:
            _patch_cmake_npu_arch(project_dst, npu_arch)
        logger.debug(
            "[%s] ascendc project copied: %s -> %s",
            op_name,
            project_src,
            project_dst,
        )

    def expected_artifacts(
        self,
        verify_dir: str,
        op_name: str,
        framework: str,
        dsl_filename_hint: str,
    ) -> list:
        wrapper_name = self.entry_filename_template.format(op_name=op_name)
        return [
            os.path.join(verify_dir, wrapper_name),
            os.path.join(
                verify_dir,
                self._setup_project_dir_name or self.kernel_project_dir_name,
                "CMakeLists.txt",
            ),
        ]

    def post_iteration_cleanup(self, verify_dir: str) -> None:
        if os.environ.get("AR_KEEP_BATCH_VERIFY_TEMP") == "1":
            return
        project_dir = os.path.join(
            verify_dir,
            self._setup_project_dir_name or self.kernel_project_dir_name,
        )
        build_dir = os.path.join(project_dir, "build")
        if os.path.isdir(build_dir):
            shutil.rmtree(build_dir, ignore_errors=True)

    def materialize_project_tree(
        self,
        dst_dir: str,
        project_src: Optional[str],
        project_dir_name: Optional[str] = None,
    ) -> None:
        if not project_src:
            return
        dst = os.path.join(
            dst_dir,
            project_dir_name or self.kernel_project_dir_name,
        )
        _copy_project_tree(project_src, dst)

    def list_kernel_project_files(
        self,
        project_src: Optional[str] = None,
        op_name: Optional[str] = None,
        project_dir_name: Optional[str] = None,
    ) -> list:
        if not project_src or not os.path.isdir(project_src):
            return list(self.kernel_project_files)
        project_src_path = Path(project_src).resolve()
        dst_dir = project_dir_name or self.kernel_project_dir_name
        files: list[str] = []
        for path in sorted(project_src_path.rglob("*")):
            if not path.is_file():
                continue
            project_rel_parts = path.relative_to(project_src_path).parts
            if any(part in _IGNORE_DIRS for part in project_rel_parts):
                continue
            if path.name not in _TEXT_FILENAMES and path.suffix not in _TEXT_SUFFIXES:
                continue
            # Keep the copied project broad, but the WA edit surface narrow:
            # only core direct-invoke implementation/build files are mutable.
            if not project_rel_parts or project_rel_parts[0] not in _EDITABLE_PROJECT_ROOTS:
                continue
            rel = path.relative_to(project_src_path).as_posix()
            files.append(f"{dst_dir}/{rel}")
        return files or list(self.kernel_project_files)
