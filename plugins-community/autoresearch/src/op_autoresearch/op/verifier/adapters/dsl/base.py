# Copyright 2025 Huawei Technologies Co., Ltd
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

"""Base class for DSL adapters."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Template


@lru_cache(maxsize=8)
def load_code_template(template_name: str) -> str:
    """Load a generated-code template from package resources."""
    template_path = (
        Path(__file__).resolve().parents[3]
        / "resources"
        / "templates"
        / template_name
    )
    return template_path.read_text(encoding="utf-8")


def render_code_template(template_name: str, **context: Any) -> str:
    """Render one generated-code template without changing its indentation."""
    return Template(
        load_code_template(template_name), keep_trailing_newline=True
    ).render(**context)


def render_model_wrapper_setup(
    framework: str,
    init_params_var: str,
    device_var: str,
    *,
    eval_mode: bool,
) -> str:
    """Render the common ``ModelNew`` construction sequence."""
    lines = [f"impl_model = ModelNew(*{init_params_var})"]
    if framework == "torch":
        lines.append(f"impl_model = impl_model.to({device_var})")
    if eval_mode:
        lines.append("impl_model.eval()")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class BenchmarkSpec:
    """Inputs that define one generated implementation benchmark."""

    inputs: str
    warmup: int
    runs: int
    backend: str
    op_name: str
    case_idx: int = 0
    clear_l2_cache: bool = True
    framework: str = "torch"


@dataclass(frozen=True)
class MaterializeSpec:
    """Files and task metadata needed to materialize one implementation."""

    impl_code: str
    verify_dir: str
    op_name: str
    framework: str
    dsl_name: str
    task_info: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None


class DSLAdapter(ABC):
    """Abstract base class for DSL adapters.

    DSL adapters provide a unified interface for different implementation languages
    (Triton Ascend, AscendC, and CATLASS) to handle calls, benchmarking, and
    other DSL-specific operations. DSL adapters are unaware of autotune logic.
    """

    @abstractmethod
    def get_import_statements(self, framework: str) -> str:
        """Return import statements for the DSL.

        Args:
            framework: Framework name (torch, mindspore, numpy)

        Returns:
            str: Import statements as a string
        """
        pass

    @abstractmethod
    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        """Return import statement for implementation function.

        Args:
            op_name: Operator name
            impl_func_name: Implementation function name

        Returns:
            str: Import statement for the generated implementation module.
        """
        pass

    def create_impl_module(
        self,
        framework: str,
        framework_adapter: Any,
        init_params_var: str = "init_params",
        device_var: str = "device",
    ) -> str:
        """Render setup for adapters that expose a ModelNew wrapper."""
        if self.model_wrapper_eval_mode is None:
            return ""
        return render_model_wrapper_setup(
            framework,
            init_params_var,
            device_var,
            eval_mode=self.model_wrapper_eval_mode,
        )

    def call_impl(self, inputs: str) -> str:
        """Return code string to call implementation function.

        Args:
            inputs: Input variable name (e.g., "inputs_for_impl")

        Returns:
            str: Code string to call the implementation
        """
        return (
            "from op_autoresearch.op.utils.code_checker.runtime_guard import "
            "guarded_call as _op_autoresearch_guarded_call\n"
            f"impl_output = _op_autoresearch_guarded_call(lambda: impl_model(*{inputs}))\n"
        )

    # needs_binary_io: True iff an implementation uses file-based I/O.
    # needs_compilation: True iff impl must be compiled before import/use.
    # static_check_via_python_ast: True iff LLM-submitted source is
    # parseable Python; CodeChecker skips when False.
    needs_binary_io: bool = False
    needs_compilation: bool = False
    static_check_via_python_ast: bool = True

    # ------------------------------------------------------------------
    # Kernel project structure — DSL-knowable data describing how this
    # DSL packages its kernel sources. Used by both op_autoresearch (verifier needs
    # the file list for materialize_impl / expected_artifacts) and any
    # outer driver that has to know where to copy / what to expose as
    # editable. The driver derives its own policy (e.g. WA scaffold
    # builds task.yaml ``editable_files`` from this list); the adapter
    # only states what the project IS.
    # ------------------------------------------------------------------
    # Directory-backed DSLs hand off a wrapper plus a project subtree.
    kernel_arg_is_directory: bool = False
    # Project subtree relative to the per-op root.
    kernel_project_dir_name: Optional[str] = None
    # Files (relative to the kernel's Python wrapper) that belong to the
    # DSL's kernel project besides the wrapper itself — sources, headers,
    # build files. Single-file DSLs leave this empty.
    kernel_project_files: list = []
    # Op entry filename — the file the LLM mainly edits. Format-string
    # with optional ``{op_name}`` slot. The supported adapters use a Python
    # wrapper and may also materialize a directory-backed project.
    # Consumers should NOT assume Python — check
    # ``static_check_via_python_ast`` for that.
    entry_filename_template: str = "kernel.py"

    def benchmark_impl(self, spec: BenchmarkSpec) -> str:
        """Render the generic implementation benchmark."""
        framework_arg = (
            f', framework="{spec.framework}"'
            if spec.framework == "mindspore"
            else ""
        )
        return render_code_template(
            "default_dsl_benchmark.py.j2",
            is_ascend=spec.backend == "ascend",
            inputs=spec.inputs,
            warmup=spec.warmup,
            runs=spec.runs,
            total_runs=spec.warmup + spec.runs,
            clear_l2_cache=spec.clear_l2_cache,
            framework_arg=framework_arg,
        )

    def get_special_setup_code(self, framework: str = "torch") -> str:
        """Return special setup code required by the DSL.

        Args:
            framework: Framework type ("torch" or "mindspore")

        Returns:
            str: Setup code as string (empty if not needed)
        """
        return ""

    # The ONE interface between op_autoresearch core and the cannbench eval-standard. True
    # makes kernel_verifier route reference + compare + comparator-staging through
    # the cann_correctness package (MERE/MARE, fp64 dual reference); False/default
    # uses the framework's generic allclose. That flag is the whole surface — no
    # cannbench hooks or logic live on the adapter; op_autoresearch reads it and pulls
    # everything from the package.
    uses_cannbench_precision: bool = False

    # ------------------------------------------------------------------
    # Extension hooks — KernelVerifier / eval_bridge / LocalWorker delegate
    # per-DSL behavior here so new DSLs need only override these instead
    # of editing the call sites with if/elif chains.
    # ------------------------------------------------------------------

    def materialize_impl(self, spec: MaterializeSpec) -> None:
        """Write the generated kernel into verify_dir for this DSL.

        Default: write ``<op>_<dsl_name>_impl.py`` with the adapter's
        own import statements prepended (the Triton convention). DSLs that
        need a project tree
        (AscendC-CATLASS) override to drop their own files.
        """
        impl_path = os.path.join(
            spec.verify_dir,
            f"{spec.op_name}_{spec.dsl_name}_impl.py",
        )
        imports = self.get_import_statements(spec.framework)
        with open(impl_path, "w", encoding="utf-8") as f:
            f.write(imports + spec.impl_code)

    def expected_artifacts(self, verify_dir: str, op_name: str,
                           framework: str, dsl_filename_hint: str) -> list:
        """Files that must exist in verify_dir before profile can run.

        Default: framework file + ``<op>_<dsl>_impl.py``. SOL / CANN /
        catlass override.
        """
        return [
            os.path.join(verify_dir, f"{op_name}_{framework}.py"),
            os.path.join(verify_dir, f"{op_name}_{dsl_filename_hint}_impl.py"),
        ]

    def prepare_config(self, config: Dict[str, Any],
                       task_info: Optional[Dict[str, Any]] = None) -> None:
        """Mutate ``config`` in place before verify/profile runs (e.g.
        resolve CATLASS_ROOT). Default: no-op.
        """
        return None

    # Pure per-adapter flags / templates (no runtime state). Override as
    # a class attribute on the subclass — don't wrap in a method just
    # to return a constant.
    #
    #   benchmark_requires_l2_clear: should the base benchmark template
    #     clear L2 cache between runs? AscendC-CATLASS = False (cmake-
    #     built kernel keeps state across iterations).
    #
    #   profile_via_python_script: LocalWorker dispatch — True → run
    #     profile scripts via Python + read JSON; False → compile-then-
    #     launch flow (AscendC).
    #
    # Default implementation name; adapters override the format as needed.
    benchmark_requires_l2_clear: bool = True
    profile_via_python_script: bool = False
    impl_func_name_template: str = "{op_name}_{dsl}_{framework}"
    model_wrapper_eval_mode: Optional[bool] = None
    profiler_dsl: str = "other"
    supports_autotune_configs: bool = False
    emits_autotune_artifacts: bool = False

    def post_iteration_cleanup(self, verify_dir: str) -> None:
        """Drop per-round artifacts that should not survive into the
        next iteration (e.g. catlass build dir). Default: no-op.
        """
        return None

    def get_runtime_env_override_code(self, **kwargs) -> str:
        """Emit an optional `__main__`-side environment override snippet.
        Default: no-op so KernelVerifier
        can always call it without a hasattr() guard. Override on the
        DSL that actually needs it.
        """
        return ""

    def read_kernel_source(self, kernel_arg: str,
                           op_name: Optional[str] = None) -> tuple:
        """Resolve a kernel handoff path into ``(source_code, project_dir_or_None)``.

        ``source_code`` is the text of the Python wrapper exposing
        ModelNew (or whatever entry the DSL uses); ``project_dir`` is
        the supplementary source tree the wrapper sits next to when
        ``kernel_arg_is_directory=True``, or ``None`` for single-file
        DSLs. Callers that need the full project tree forward
        ``project_dir`` to :meth:`materialize_project_tree`.

        Directory-backed adapters use the sibling ``kernel.py`` or
        ``<op>_kernel.py`` wrapper; single-file adapters read ``kernel_arg``.
        """
        if self.kernel_arg_is_directory:
            return self._read_project_wrapper(kernel_arg, op_name)
        if not os.path.isfile(kernel_arg):
            raise FileNotFoundError(
                f"kernel handoff must be a file for this DSL; got {kernel_arg!r}"
            )
        with open(kernel_arg, "r", encoding="utf-8") as f:
            return f.read(), None

    def materialize_project_tree(self, dst_dir: str,
                                 project_src: Optional[str],
                                 project_dir_name: Optional[str] = None) -> None:
        """Copy the DSL's project tree from ``project_src`` into ``dst_dir``
        with any DSL-specific patching (e.g. catlass cmake rewrite).
        Default: no-op (single-file DSLs have nothing to copy beyond the
        wrapper, which the caller already wrote out).
        """
        return None

    def list_kernel_project_files(self, project_src: Optional[str] = None,
                                  op_name: Optional[str] = None,
                                  project_dir_name: Optional[str] = None) -> list:
        """Return editable files belonging to this DSL project tree.

        Most multi-file DSLs have a fixed project shape and can use the
        class-level ``kernel_project_files`` list. DSLs whose project files
        are operator-named (AscendC direct-invoke) override this to discover
        text sources from ``project_src`` after the handoff path is known.
        Entries are relative to the wrapper's directory.
        """
        files = list(self.kernel_project_files)
        src_dir = self.kernel_project_dir_name
        dst_dir = project_dir_name or src_dir
        if src_dir and dst_dir and dst_dir != src_dir:
            prefix = f"{src_dir}/"
            files = [
                f"{dst_dir}/{path[len(prefix):]}"
                if path.startswith(prefix) else path
                for path in files
            ]
        return files

    def get_autotune_info(self, case_idx: int) -> Optional[Dict]:
        """Get autotune information (only for triton_ascend in profiling).

        Args:
            case_idx: Case index

        Returns:
            dict or None: Autotune information
        """
        return None

    def get_binary_io_functions(self) -> str:
        """Get optional binary I/O helper functions.

        Returns:
            str: Function definitions as string (empty if not needed)
        """
        return ""

    def _read_project_wrapper(
        self, kernel_arg: str, op_name: Optional[str]
    ) -> tuple:
        if not os.path.isdir(kernel_arg):
            raise FileNotFoundError(
                f"kernel handoff must be a project directory; got {kernel_arg!r}"
            )
        project_dir = os.path.abspath(kernel_arg)
        parent = os.path.dirname(project_dir)
        candidates = [self.entry_filename_template.format(op_name=op_name or "")]
        if op_name:
            candidates.append(f"{op_name}_kernel.py")
        for name in candidates:
            sibling = os.path.join(parent, name)
            if os.path.isfile(sibling):
                with open(sibling, "r", encoding="utf-8") as file:
                    return file.read(), project_dir
        raise FileNotFoundError(
            "directory-backed kernel handoff requires sibling "
            f"{' or '.join(candidates)} at {parent}"
        )
