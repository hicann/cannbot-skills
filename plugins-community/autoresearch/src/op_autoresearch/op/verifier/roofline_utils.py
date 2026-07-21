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

"""SOLAR roofline 集成工具。

设计约束：
1. 不修改 / patch SOLAR 仓库。
2. OP_AUTORESEARCH 运行时只依赖“已安装的 solar Python 包”，不依赖本地 SOLAR 工作树。
3. 之前只存在于本地 SOLAR 改动里的辅助逻辑（如 solbench wrapper、Ascend arch config）
   迁移到 OP_AUTORESEARCH 自己维护。
4. roofline 失败只能降级，不能影响原有 profile 主流程。
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from op_autoresearch.core.worker.eval_config import resolve_eval_timeout
from op_autoresearch.op.utils.json_safe import sanitize_floats
from op_autoresearch.op.verifier.aggregate import geomean_ratio, mean_us

logger = logging.getLogger(__name__)

ROOFLINE_MODEL = "fused"
ROOFLINE_RESULT_JSON = "roofline_profile_result.json"
SOLAR_INSTALL_HINT = "bash ./download.sh --with_solar"

ARCH_ALIAS_TO_CONFIG_KEY = {
    "ascend910b1": "ascend910b1",
    "ascend910b2": "ascend910b2",
    "ascend910b2c": "ascend910b2",
    "ascend910b3": "ascend910b3",
    "ascend910b4": "ascend910b4",
    "ascend910_9362": "ascend910b4",
    "ascend910_9372": "ascend910b4",
    "ascend910_9381": "ascend910b4",
    "ascend910_9382": "ascend910b4",
    "ascend910_9391": "ascend910b4",
    "ascend910_9392": "ascend910b4",
    "ascend950dt_95a": "ascend950_pr",
    "ascend950pr_950z": "ascend950_pr",
    "ascend950pr_9572": "ascend950_pr",
    "ascend950pr_9574": "ascend950_pr",
    "ascend950pr_9575": "ascend950_pr",
    "ascend950pr_9576": "ascend950_pr",
    "ascend950pr_9577": "ascend950_pr",
    "ascend950pr_9578": "ascend950_pr",
    "ascend950pr_9579": "ascend950_pr",
    "ascend950pr_957b": "ascend950_pr",
    "ascend950pr_957d": "ascend950_pr",
    "ascend950pr_9581": "ascend950_pr",
    "ascend950pr_9582": "ascend950_pr",
    "ascend950pr_9584": "ascend950_pr",
    "ascend950pr_9587": "ascend950_pr",
    "ascend950pr_9588": "ascend950_pr",
    "ascend950pr_9589": "ascend950_pr",
    "ascend950pr_958a": "ascend950_pr",
    "ascend950pr_958b": "ascend950_pr",
    "ascend950pr_9591": "ascend950_pr",
    "ascend950pr_9592": "ascend950_pr",
    "ascend950pr_9599": "ascend950_pr",
}

# 这些配置原先依赖本地 SOLAR 改动；现在由 OP_AUTORESEARCH 自己维护，避免依赖 drop commit。
OP_AUTORESEARCH_ROOFLINE_ARCH_CONFIGS = {
    "ascend910b1": {
        "name": "Ascend910B1",
        "freq_GHz": 1.5,
        "DRAM_byte_per_cycle": 1.8e12 / 1.5e9,
        "MAC_per_cycle_fp16_tc": 245e12 / (2 * 1.5e9),
        "MAC_per_cycle_fp32_tc": 61e12 / (2 * 1.5e9),
    },
    "ascend910b2": {
        "name": "Ascend910B2",
        "freq_GHz": 1.5,
        "DRAM_byte_per_cycle": 1.8e12 / 1.5e9,
        "MAC_per_cycle_fp16_tc": 245e12 / (2 * 1.5e9),
        "MAC_per_cycle_fp32_tc": 61e12 / (2 * 1.5e9),
    },
    "ascend910b3": {
        "name": "Ascend910B3",
        "freq_GHz": 1.5,
        "DRAM_byte_per_cycle": 1.6e12 / 1.5e9,
        "MAC_per_cycle_fp16_tc": 245e12 / (2 * 1.5e9),
        "MAC_per_cycle_fp32_tc": 61e12 / (2 * 1.5e9),
    },
    "ascend910b4": {
        "name": "Ascend910B4",
        "freq_GHz": 1.5,
        "DRAM_byte_per_cycle": 0.8e12 / 1.5e9,
        "MAC_per_cycle_fp16_tc": 245e12 / (2 * 1.5e9),
        "MAC_per_cycle_fp32_tc": 61e12 / (2 * 1.5e9),
    },
    "ascend950_pr": {
        "name": "Ascend950_PR",
        "freq_GHz": 1.65,
        "DRAM_byte_per_cycle": 1.6e12 / 1.65e9,
        "MAC_per_cycle_fp16_tc": 380e12 / (2 * 1.65e9),
        "MAC_per_cycle_fp32_tc": 27e12 / (2 * 1.65e9),
    },
}

_PRECISION_ALIASES = {
    "fp32": "fp32",
    "float32": "fp32",
    "float": "fp32",
    "torch.float32": "fp32",
    "torch.float": "fp32",
    "tf32": "tf32",
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "torch.float16": "fp16",
    "torch.half": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "torch.bfloat16": "bf16",
    "fp64": "fp64",
    "float64": "fp64",
    "double": "fp64",
    "torch.float64": "fp64",
    "torch.double": "fp64",
    "int8": "int8",
    "torch.int8": "int8",
    "fp8": "fp8",
    "float8": "fp8",
    "nvfp4": "nvfp4",
    "fp4": "nvfp4",
    "float4": "nvfp4",
}

_PREFERRED_PRECISION_ORDER = ["nvfp4", "fp8", "bf16", "fp16", "fp32", "int8", "fp64"]


@dataclass(frozen=True)
class RooflineContext:
    """SOLAR API and target settings shared by every roofline case."""

    verify_dir: Path
    op_name: str
    task_id: str
    solar_api: Dict[str, Any]
    arch_spec: str
    precision_override: Optional[str]
    timeout: int


@dataclass(frozen=True)
class RooflineCase:
    """One Python model source analyzed into an isolated output directory."""

    source_file: Path
    output_dir: Path
    label: str


@dataclass(frozen=True)
class RooflineSetup:
    """Validated dispatch decision for one profile directory."""

    bench_type: str
    arch: str
    framework: str
    context: Optional[RooflineContext] = None
    skipped: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class RooflineArtifacts:
    """Intermediate directories produced by the SOLAR pipeline."""

    graph_dir: Path
    einsum_dir: Path
    analysis_dir: Path
    perf_dir: Path


def compute_roofline_profile(
    verify_dir: str,
    op_name: str,
    task_id: str,
    profile_settings: Dict[str, Any],
) -> Dict[str, Any]:
    """为当前 verify_dir 计算 SOLAR roofline。"""
    setup = _resolve_roofline_setup(
        Path(verify_dir),
        op_name,
        task_id,
        profile_settings,
    )
    if setup.skipped is not None:
        return setup.skipped
    try:
        result = _dispatch_roofline(setup)
        result.setdefault("bench_type", setup.bench_type)
        result.setdefault("arch", setup.arch)
        result.setdefault("task_id", task_id)
        result.setdefault("op_name", op_name)
        return result
    except Exception as exc:
        logger.warning("[%s:%s] roofline 计算失败: %s", task_id, op_name, exc, exc_info=True)
        return {
            "success": False,
            "skipped": False,
            "source": "solar",
            "model": ROOFLINE_MODEL,
            "bench_type": setup.bench_type,
            "arch": setup.arch,
            "op_name": op_name,
            "task_id": task_id,
            "error": str(exc),
        }


def _resolve_roofline_setup(
    verify_dir: Path,
    op_name: str,
    task_id: str,
    profile_settings: Dict[str, Any],
) -> RooflineSetup:
    bench_type = _infer_bench_type(verify_dir, profile_settings.get("bench_type"))
    backend = str(profile_settings.get("backend", "")).lower()
    arch = str(profile_settings.get("arch", "")).lower()
    framework = str(profile_settings.get("framework", "torch")).lower()
    skip_reason = _roofline_skip_reason(profile_settings, backend, bench_type)
    if skip_reason:
        return _skipped_setup(bench_type, arch, framework, skip_reason)

    solar_api, import_error = _import_solar_api()
    if solar_api is None:
        reason = (
            "未安装 solar Python 包。"
            f"可执行: {SOLAR_INSTALL_HINT}. 原始错误: {import_error}"
        )
        return _skipped_setup(bench_type, arch, framework, reason)
    arch_spec = resolve_arch_spec(
        arch,
        verify_dir,
        profile_settings.get("roofline_arch_config"),
    )
    if arch_spec is None:
        return _skipped_setup(
            bench_type,
            arch,
            framework,
            f"arch={arch} 当前没有可用的 roofline 架构配置",
        )
    context = RooflineContext(
        verify_dir,
        op_name,
        task_id,
        solar_api,
        arch_spec,
        profile_settings.get("roofline_precision"),
        resolve_eval_timeout(profile_settings.get("timeout")),
    )
    return RooflineSetup(bench_type, arch, framework, context=context)


def _skipped_setup(
    bench_type: str,
    arch: str,
    framework: str,
    reason: str,
) -> RooflineSetup:
    return RooflineSetup(
        bench_type,
        arch,
        framework,
        skipped=_skipped_result(reason, bench_type, arch),
    )


def _roofline_skip_reason(
    profile_settings: Dict[str, Any],
    backend: str,
    bench_type: str,
) -> Optional[str]:
    if not profile_settings.get("enable_roofline", True):
        return "roofline 已显式关闭"
    if backend != "ascend":
        return f"backend={backend} 当前不支持 roofline"
    if bench_type == "cann":
        return "CANN-Bench 暂不支持 roofline 评分"
    return None


def _dispatch_roofline(setup: RooflineSetup) -> Dict[str, Any]:
    if setup.context is None:
        raise RuntimeError("roofline dispatch requires a validated context")
    if setup.bench_type == "sol":
        return _compute_sol_roofline(setup.context)
    return _compute_kernelbench_roofline(setup.context, setup.framework)


def augment_roofline_metrics(
    roofline_result: Dict[str, Any],
    gen_time_us: Optional[float] = None,
    base_time_us: Optional[float] = None,
    gen_per_shape_us: Optional[list] = None,
    base_per_shape_us: Optional[list] = None,
) -> Dict[str, Any]:
    """补充与 profile 实测时间相关的 roofline 指标。

    ``speedup_vs_generated`` / ``speedup_vs_baseline`` are the geomean of the
    per-shape ratios ``roofline_case[i] / gen[i]`` when the per-shape arrays
    are present AND length-aligned with the roofline cases — same rule as
    ``speedup_vs_ref``. Otherwise fall back to the scalar
    ``roofline_time / gen_time`` (both arithmetic-mean aggregates), so a
    backend that doesn't surface per-shape arrays still gets a number.
    """
    augmented = dict(roofline_result or {})
    roofline_time = augmented.get("time_us")
    case_times = list(augmented.get("case_times_us") or [])
    gen_ps = list(gen_per_shape_us or [])
    base_ps = list(base_per_shape_us or [])

    def _ratio(numer_ps, numer_scalar, denom_ps, denom_scalar):
        """geomean(numer_ps[i]/denom_ps[i]) when arrays align with the
        roofline cases; else scalar numer/denom; else None.
        """
        if (case_times and len(numer_ps) == len(case_times)
                and len(denom_ps) == len(case_times)):
            geo = geomean_ratio(numer_ps, denom_ps)
            if geo is not None:
                return geo
        if (_is_valid_positive_number(numer_scalar)
                and _is_valid_positive_number(denom_scalar)):
            return float(numer_scalar) / float(denom_scalar)
        return None

    svg = _ratio(case_times, roofline_time, gen_ps, gen_time_us)
    augmented["speedup_vs_generated"] = svg if svg is not None else 0.0
    gap = _ratio(gen_ps, gen_time_us, case_times, roofline_time)
    augmented["gap_vs_generated"] = gap  # None when unmeasurable

    svb = _ratio(case_times, roofline_time, base_ps, base_time_us)
    augmented["speedup_vs_baseline"] = svb if svb is not None else 0.0

    return augmented


def write_roofline_profile_result(verify_dir: str, roofline_result: Dict[str, Any]) -> str:
    """将 roofline 结果写入 verify_dir/roofline_profile_result.json。"""
    output_path = Path(verify_dir) / ROOFLINE_RESULT_JSON
    output_path.write_text(
        json.dumps(sanitize_floats(roofline_result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(output_path)


def resolve_arch_spec(
    arch: str,
    verify_dir: Path,
    explicit_arch_config: Optional[str] = None,
) -> Optional[str]:
    """将 OP_AUTORESEARCH arch 解析为 roofline arch-config 参数。"""
    if explicit_arch_config:
        explicit_path = Path(os.path.expanduser(explicit_arch_config))
        return str(explicit_path.resolve()) if explicit_path.exists() else explicit_arch_config

    config_key = ARCH_ALIAS_TO_CONFIG_KEY.get(arch)
    arch_config = OP_AUTORESEARCH_ROOFLINE_ARCH_CONFIGS.get(config_key or "")
    if arch_config is None:
        return None

    custom_dir = verify_dir / "_roofline_arch"
    custom_dir.mkdir(parents=True, exist_ok=True)
    custom_path = custom_dir / f"{config_key}.yaml"
    custom_path.write_text(yaml.safe_dump(arch_config, sort_keys=False), encoding="utf-8")
    return str(custom_path)


def _compute_kernelbench_roofline(
    context: RooflineContext,
    framework: str,
) -> Dict[str, Any]:
    source_file = _find_framework_source_file(
        context.verify_dir,
        context.op_name,
        framework,
    )
    case_result = _compute_single_case_roofline(
        context,
        RooflineCase(
            source_file,
            context.verify_dir / "_roofline_kernelbench",
            source_file.stem,
        ),
    )
    return _aggregate_case_results([case_result], bench_type="kernelbench")


def _compute_sol_roofline(
    context: RooflineContext,
) -> Dict[str, Any]:
    workload_path = context.verify_dir / "workload.jsonl"
    if not workload_path.is_file():
        raise FileNotFoundError(f"SOL workload 文件不存在: {workload_path}")

    workloads = [line for line in workload_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not workloads:
        raise ValueError(f"SOL workload 为空: {workload_path}")

    wrappers_dir = context.verify_dir / "_roofline_sol_wrappers"
    wrappers_dir.mkdir(parents=True, exist_ok=True)

    case_results = []
    for workload_idx in range(len(workloads)):
        wrapper_file = wrappers_dir / f"{context.op_name}_w{workload_idx:03d}.py"
        _create_solbench_wrapper(context.verify_dir, wrapper_file, workload_idx)

        case_output_dir = (
            context.verify_dir / "_roofline_sol" / f"w{workload_idx:03d}"
        )
        case_results.append(
            _compute_single_case_roofline(
                context,
                RooflineCase(
                    wrapper_file,
                    case_output_dir,
                    f"w{workload_idx:03d}",
                ),
            )
        )

    result = _aggregate_case_results(case_results, bench_type="sol")
    result["workload_count"] = len(case_results)
    return result


def _compute_single_case_roofline(
    context: RooflineContext,
    case: RooflineCase,
) -> Dict[str, Any]:
    artifacts = _create_roofline_artifacts(case.output_dir)
    graph_path, precision = _process_roofline_graph(context, case, artifacts)
    analysis_path = _analyze_roofline_graph(
        context,
        artifacts,
        graph_path,
        precision,
    )
    perf_data = _predict_roofline(
        context,
        artifacts,
        analysis_path,
        precision,
    )
    return _format_case_result(case, context, perf_data, precision)


def _create_roofline_artifacts(output_dir: Path) -> RooflineArtifacts:
    artifacts = RooflineArtifacts(
        output_dir / "graph",
        output_dir / "einsum",
        output_dir / "analysis",
        output_dir / "perf",
    )
    for directory in (
        artifacts.graph_dir,
        artifacts.einsum_dir,
        artifacts.analysis_dir,
        artifacts.perf_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return artifacts


def _process_roofline_graph(
    context: RooflineContext,
    case: RooflineCase,
    artifacts: RooflineArtifacts,
) -> tuple[Path, str]:
    processing_config = context.solar_api["ProcessingConfig"](
        save_graph=False,
        force_rerun=True,
        timeout=context.timeout,
        output_dir=str(artifacts.graph_dir),
        debug=False,
        safe_mode=True,
    )
    processor = context.solar_api["PyTorchProcessor"](processing_config)
    if not processor.process_model_file(
        str(case.source_file),
        str(artifacts.graph_dir),
    ):
        raise RuntimeError(f"[process_model] 处理失败: {case.source_file}")

    graph_path = artifacts.graph_dir / "pytorch_graph.yaml"
    if not graph_path.is_file():
        raise FileNotFoundError(f"未生成 pytorch_graph.yaml: {graph_path}")
    precision = (
        _normalize_precision_name(context.precision_override)
        if context.precision_override
        else _infer_graph_precision(graph_path)
    )
    return graph_path, precision


def _analyze_roofline_graph(
    context: RooflineContext,
    artifacts: RooflineArtifacts,
    graph_path: Path,
    precision: str,
) -> Path:
    converter = context.solar_api["PyTorchToEinsum"](
        debug=False,
        enable_agent=False,
        cache_dir=str(artifacts.graph_dir.parent / "solar_handlers_cache"),
    )
    convert_result = converter.convert(
        graph_path,
        artifacts.einsum_dir,
        copy_graph=True,
        expand_complex_ops=True,
        enable_rename=True,
    )
    if convert_result is None:
        raise RuntimeError(f"[toeinsum_model] 转换失败: {graph_path}")

    einsum_graph_path = artifacts.einsum_dir / "einsum_graph_renamed.yaml"
    if not einsum_graph_path.is_file():
        raise FileNotFoundError(f"未生成 einsum_graph_renamed.yaml: {einsum_graph_path}")

    analyzer = context.solar_api["EinsumGraphAnalyzer"](debug=False)
    analysis_result = analyzer.analyze_graph(
        einsum_graph_path,
        artifacts.analysis_dir,
        precision=precision,
        copy_graph=True,
    )
    if analysis_result is None:
        raise RuntimeError(f"[analyze_model] 分析失败: {einsum_graph_path}")

    analysis_path = artifacts.analysis_dir / "analysis.yaml"
    if not analysis_path.is_file():
        raise FileNotFoundError(f"未生成 analysis.yaml: {analysis_path}")
    return analysis_path


def _predict_roofline(
    context: RooflineContext,
    artifacts: RooflineArtifacts,
    analysis_path: Path,
    precision: str,
) -> Dict[str, Any]:
    perf_model = context.solar_api["EinsumGraphPerfModel"](debug=False)
    perf_result = perf_model.predict(
        analysis_path,
        artifacts.perf_dir,
        arch_config=context.arch_spec,
        precision=precision,
        copy_analysis=True,
    )
    if perf_result is None:
        raise RuntimeError(f"[predict_perf_model] 预测失败: {analysis_path}")
    return perf_result


def _format_case_result(
    case: RooflineCase,
    context: RooflineContext,
    perf_data: Dict[str, Any],
    precision: str,
) -> Dict[str, Any]:
    arch_info = perf_data.get("arch") or {}
    fused_info = perf_data.get(ROOFLINE_MODEL) or {}
    freq_ghz = float(arch_info.get("freq_GHz") or 0.0)
    runtime_ms = float(fused_info.get("runtime_ms") or 0.0)
    compute_cycles = float(fused_info.get("compute_cycles") or 0.0)
    memory_cycles = float(fused_info.get("memory_cycles") or 0.0)

    return {
        "success": True,
        "case_label": case.label,
        "precision": precision,
        "arch_name": str(arch_info.get("name") or context.arch_spec),
        "time_us": runtime_ms * 1000.0,
        "compute_time_us": _cycles_to_us(compute_cycles, freq_ghz),
        "memory_time_us": _cycles_to_us(memory_cycles, freq_ghz),
        "bottleneck": str(fused_info.get("bottleneck") or ""),
    }


def _aggregate_case_results(case_results: list[Dict[str, Any]], bench_type: str) -> Dict[str, Any]:
    if not case_results:
        raise ValueError("没有可聚合的 roofline case 结果")

    failures = [item for item in case_results if not item.get("success")]
    if failures:
        first = failures[0]
        return {
            "success": False,
            "skipped": False,
            "source": "solar",
            "model": ROOFLINE_MODEL,
            "bench_type": bench_type,
            "error": first.get("error") or "roofline case failed",
            "case_count": len(case_results),
        }

    case_labels = [str(item["case_label"]) for item in case_results]
    case_times_us = [float(item["time_us"]) for item in case_results]
    compute_times_us = [float(item["compute_time_us"]) for item in case_results]
    memory_times_us = [float(item["memory_time_us"]) for item in case_results]
    bottlenecks = [str(item.get("bottleneck") or "") for item in case_results if item.get("bottleneck")]
    precisions = [str(item.get("precision") or "") for item in case_results if item.get("precision")]
    arch_names = [str(item.get("arch_name") or "") for item in case_results if item.get("arch_name")]

    return {
        "success": True,
        "skipped": False,
        "source": "solar",
        "model": ROOFLINE_MODEL,
        "bench_type": bench_type,
        "case_count": len(case_results),
        "case_labels": case_labels,
        "case_times_us": case_times_us,
        # Latencies aggregate as arithmetic mean (per-call cost); the
        # per-shape ``case_times_us`` stays for the geomean speedup ratios
        # computed in ``augment_roofline_metrics``.
        "time_us": mean_us(case_times_us),
        "compute_time_us": mean_us(compute_times_us),
        "memory_time_us": mean_us(memory_times_us),
        "bottleneck": _merge_strings_keep_mixed(bottlenecks),
        "precision": _merge_strings_keep_mixed(precisions),
        "arch_name": _merge_strings_keep_mixed(arch_names),
    }


def _create_solbench_wrapper(verify_dir: Path, wrapper_path: Path, workload_idx: int) -> None:
    """Render the SOLBench reference wrapper for one workload."""
    from jinja2 import Template

    definition = json.loads((verify_dir / "definition.json").read_text(encoding="utf-8"))
    template_path = (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "templates"
        / "roofline_solbench_wrapper.py.j2"
    )
    template_values = {
        "reference_path_repr": repr(str(verify_dir / "reference.py")),
        "workload_path_repr": repr(str(verify_dir / "workload.jsonl")),
        "workload_idx": workload_idx,
        "input_order_repr": repr(list((definition.get("inputs") or {}).keys())),
        "entrypoint_repr": repr(definition.get("custom_inputs_entrypoint") or "get_inputs"),
        "axes_spec_repr": repr(definition.get("axes") or {}),
        "inputs_spec_repr": repr(definition.get("inputs") or {}),
    }
    rendered = Template(template_path.read_text(encoding="utf-8")).render(**template_values)
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text(rendered.rstrip() + "\n", encoding="utf-8")


def _import_solar_api() -> tuple[Optional[Dict[str, Any]], Optional[Exception]]:
    try:
        common_types = importlib.import_module("solar.common.types")
        graph_mod = importlib.import_module("solar.graph")
        einsum_mod = importlib.import_module("solar.einsum")
        analysis_mod = importlib.import_module("solar.analysis")
        perf_mod = importlib.import_module("solar.perf")
        return (
            {
                "ProcessingConfig": common_types.ProcessingConfig,
                "PyTorchProcessor": graph_mod.PyTorchProcessor,
                "PyTorchToEinsum": einsum_mod.PyTorchToEinsum,
                "EinsumGraphAnalyzer": analysis_mod.EinsumGraphAnalyzer,
                "EinsumGraphPerfModel": perf_mod.EinsumGraphPerfModel,
            },
            None,
        )
    except Exception as exc:
        return None, exc


def _find_framework_source_file(verify_dir: Path, op_name: str, framework: str) -> Path:
    exact = verify_dir / f"{op_name}_{framework}.py"
    if exact.is_file():
        return exact

    matches = sorted(verify_dir.glob(f"*_{framework}.py"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"未找到 framework 源文件: expected={exact} or any '*_{framework}.py' in {verify_dir}"
    )


def _infer_bench_type(verify_dir: Path, explicit_bench_type: Optional[str]) -> str:
    if explicit_bench_type in {"sol", "kernelbench", "cann"}:
        return explicit_bench_type
    if (verify_dir / "definition.json").is_file() and (verify_dir / "workload.jsonl").is_file():
        return "sol"
    if (verify_dir / "proto.yaml").is_file() and (verify_dir / "golden.py").is_file():
        return "cann"
    return "kernelbench"


def _normalize_precision_name(raw: Optional[str]) -> str:
    if raw is None:
        return "fp32"
    text = str(raw).strip().lower()
    if text not in _PRECISION_ALIASES:
        raise ValueError(f"不支持的 precision: {raw}")
    return _PRECISION_ALIASES[text]


def _layer_precisions(layer: Dict[str, Any]) -> list[str]:
    precisions = []
    for key in ("input_dtypes", "output_dtypes"):
        for dtype in layer.get(key) or []:
            mapped = _map_graph_dtype(dtype)
            if mapped is not None:
                precisions.append(mapped)
    return precisions


def _infer_graph_precision(graph_path: Path) -> str:
    graph = yaml.safe_load(graph_path.read_text(encoding="utf-8")) or {}
    counts: Dict[str, int] = {}

    for layer in (graph.get("layers") or {}).values():
        for precision in _layer_precisions(layer):
            counts[precision] = counts.get(precision, 0) + 1

    if not counts:
        return "fp32"

    return sorted(
        counts.items(),
        key=lambda item: (
            -item[1],
            _PREFERRED_PRECISION_ORDER.index(item[0]) if item[0] in _PREFERRED_PRECISION_ORDER else 999,
            item[0],
        ),
    )[0][0]


def _map_graph_dtype(raw: Any) -> Optional[str]:
    text = str(raw).strip().lower()
    if not text or text in {"torch.bool", "bool"}:
        return None
    if text.startswith("torch.int") or text.startswith("torch.uint") or text in {"int", "long", "torch.long"}:
        return None
    try:
        return _normalize_precision_name(text)
    except ValueError:
        return None


def _merge_strings_keep_mixed(values: list[str]) -> Optional[str]:
    filtered = [value for value in values if value]
    if not filtered:
        return None
    if all(value == filtered[0] for value in filtered):
        return filtered[0]
    return "mixed"


def _cycles_to_us(cycles: float, freq_ghz: float) -> float:
    return cycles / (freq_ghz * 1e3) if freq_ghz > 0 else 0.0


def _is_valid_positive_number(value: Optional[float]) -> bool:
    return value is not None and isinstance(value, (int, float)) and float(value) > 0 and math.isfinite(float(value))


def _skipped_result(reason: str, bench_type: str, arch: str) -> Dict[str, Any]:
    return {
        "success": False,
        "skipped": True,
        "source": "solar",
        "model": ROOFLINE_MODEL,
        "bench_type": bench_type,
        "arch": arch,
        "error": reason,
    }
