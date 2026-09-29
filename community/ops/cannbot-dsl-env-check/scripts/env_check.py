#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""执行 CANNBotDSL 分层环境检查并生成 environment.yaml。"""

from __future__ import annotations

import logging
import sys
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any
import unicodedata

import yaml

HERE = Path(__file__).resolve().parent
PROBE = HERE / "probe_compile_stage.py"


LOGGER = logging.getLogger(__name__)


def one_line(value: Any, limit: int = 120) -> str:
    """将动态诊断文本规整为单行，便于生成逐行带注释的 YAML。"""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def run(command: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command, text=True, capture_output=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return subprocess.CompletedProcess(
            command, 124, stdout, stderr or f"命令超时（{timeout}s）"
        )
    except OSError as exc:
        return subprocess.CompletedProcess(
            command, 127, "", f"{type(exc).__name__}: {exc}"
        )


def last_detail(proc: subprocess.CompletedProcess[str]) -> str:
    combined = "\n".join(
        part.strip() for part in (proc.stdout, proc.stderr) if part.strip()
    )
    if not combined:
        return f"命令退出码 {proc.returncode}"
    for marker in ("RESULT=FAIL:", "RESULT=PASS:"):
        if marker in combined:
            return one_line(combined.rsplit(marker, 1)[1])
    return one_line(combined.splitlines()[-1])


def result(
    status: str,
    code: str,
    summary: str,
    *,
    detail: str = "",
    causes: list[str] | None = None,
    actions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "code": code,
        "summary": summary,
        "detail": one_line(detail),
        "possible_causes": causes or [],
        "actions": actions or [],
    }


def check_l0() -> tuple[dict[str, Any], dict[str, Any]]:
    statement = (
        "import json, cannbotdsl; "
        "print(json.dumps({'module_file': cannbotdsl.__file__, "
        "'version': getattr(cannbotdsl, '__version__', None)}, ensure_ascii=False))"
    )
    proc = run([sys.executable, "-c", statement], timeout=30)
    context = {"module_file": None, "version": None}
    if proc.returncode == 0:
        try:
            context.update(json.loads(proc.stdout.strip().splitlines()[-1]))
        except (json.JSONDecodeError, TypeError) as exc:
            return (
                result(
                    "FAIL",
                    "IMPORT_METADATA_INVALID",
                    "CANNBotDSL 已导入，但模块信息无法解析",
                    detail=str(exc),
                    actions=["检查 cannbotdsl 导入时是否向标准输出写入额外内容"],
                ),
                context,
            )
        return result("PASS", "OK", "当前解释器可以导入 CANNBotDSL"), context
    return (
        result(
            "FAIL",
            "CANNBOTDSL_IMPORT_FAILED",
            "当前解释器无法导入 CANNBotDSL",
            detail=last_detail(proc),
            causes=[
                "当前解释器未安装 CANNBotDSL wheel",
                "wheel 安装到了其他 Python 环境",
                "wheel 与当前 Python ABI、系统架构或依赖不匹配",
            ],
            actions=[
                "使用当前解释器执行 python -m pip show cannbotdsl",
                "使用当前解释器执行 python -m pip check",
                "为当前解释器安装匹配的 CANNBotDSL wheel",
            ],
        ),
        context,
    )


def check_compile_stage(stage: str) -> dict[str, Any]:
    proc = run([sys.executable, str(PROBE), stage], timeout=180)
    if proc.returncode == 0:
        summary = (
            "CANNBotDSL 前端与 AscendC 翻译通过"
            if stage == "translate"
            else "CANNBotDSL 编译后端通过并生成 .so"
        )
        return result("PASS", "OK", summary, detail=last_detail(proc))
    probe_errors = {
        "PROBE_INTERFACE_MISMATCH": (
            "环境探针与当前 CANNBotDSL 接口不匹配",
            "按已安装版本的公开接口更新固定探针",
        ),
        "PROBE_ARTIFACT_INVALID": (
            "编译产物不满足探针检查条件",
            "检查编译阶段设置及产物路径、大小",
        ),
        "PROBE_CLEANUP_FAILED": (
            "编译返回对象释放失败",
            "检查当前版本的程序对象生命周期接口及原始异常",
        ),
    }
    code = next(
        (
            line.removeprefix("PROBE_ERROR=")
            for line in proc.stdout.splitlines()
            if line.startswith("PROBE_ERROR=")
        ),
        "",
    )
    if code in probe_errors:
        summary, action = probe_errors[code]
        return result(
            "FAIL",
            code,
            summary,
            detail=last_detail(proc),
            actions=[
                "核对实际导入的 CANNBotDSL 路径和版本",
                action,
                "直接运行 probe_compile_stage.py 的对应阶段查看完整异常",
            ],
        )
    if stage == "translate":
        return result(
            "FAIL",
            "DSL_TRANSLATE_FAILED",
            "CANNBotDSL 无法生成 AscendC 编译输入",
            detail=last_detail(proc),
            causes=[
                "固定探针调用与当前接口签名或语义不匹配",
                "CANNBotDSL wheel 内部组件不完整或版本不一致",
                "CANNBotDSL 与当前 CANN Toolkit 不匹配",
                "AscendC 翻译依赖未正确加载",
            ],
            actions=[
                "核对实际导入的 CANNBotDSL 路径和版本",
                "核对 wheel 与 CANN Toolkit 的版本关系",
                "检查固定探针是否匹配当前接口，并保留原始异常",
            ],
        )
    return result(
        "FAIL",
        "DSL_COMPILE_FAILED",
        "CANNBotDSL 无法生成可加载的 .so",
        detail=last_detail(proc),
        causes=[
            "固定探针调用与当前接口签名或语义不匹配",
            "CANN Toolkit 未安装或环境未加载",
            "bisheng 不可执行或相关编译依赖不可见",
            "wheel 与 CANN Toolkit 版本不匹配",
            "编译输出目录或缓存目录不可写",
        ],
        actions=[
            "检查 ASCEND_HOME_PATH、PATH 和 LD_LIBRARY_PATH",
            "检查当前环境能否找到并执行 bisheng",
            "核对 CANNBotDSL wheel 与 CANN Toolkit 的版本关系",
        ],
    )


def skipped(code: str, summary: str) -> dict[str, Any]:
    return result("SKIP", code, summary)


NPU_RUNTIME_PROBE = """
import json

try:
    import torch
except Exception as exc:
    detail = f"{type(exc).__name__}: {exc}"
    code = (
        "TORCH_BACKEND_AUTOLOAD_FAILED"
        if "torch_npu" in detail or "backend extension" in detail
        else "TORCH_IMPORT_FAILED"
    )
    print(json.dumps({"code": code, "detail": detail}, ensure_ascii=False))
    raise SystemExit(10)
try:
    import torch_npu
except Exception as exc:
    print(
        json.dumps(
            {"code": "TORCH_NPU_IMPORT_FAILED", "detail": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False
        )
    )
    raise SystemExit(11)
try:
    available = bool(torch.npu.is_available())
    count = int(torch.npu.device_count()) if available else 0
    devices = []
    for device_id in range(count):
        try:
            name = " ".join(str(torch.npu.get_device_name(device_id)).split())
        except Exception:
            name = None
        devices.append({"logical_device_id": device_id, "name": name})
except Exception as exc:
    print(
        json.dumps({"code": "NPU_QUERY_FAILED", "detail": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)
    )
    raise SystemExit(12)
print(
    json.dumps(
        {"code": "OK" if available else "NPU_UNAVAILABLE", "visible_device_count": count, "devices": devices},
        ensure_ascii=False,
    )
)
raise SystemExit(0 if available else 13)
"""


def check_l3() -> tuple[dict[str, Any], dict[str, Any]]:
    proc = run([sys.executable, "-c", NPU_RUNTIME_PROBE], timeout=30)
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        payload = {"code": "NPU_CHECK_FAILED", "detail": last_detail(proc)}
    runtime = {
        "visible_device_count": payload.get("visible_device_count", 0),
        "devices": payload.get("devices", []),
    }
    if proc.returncode == 0:
        return result("PASS", "OK", "PyTorch NPU 后端识别到可用设备"), runtime

    code_name = payload.get("code", "NPU_CHECK_FAILED")
    messages = {
        "TORCH_IMPORT_FAILED": "当前解释器无法导入 torch",
        "TORCH_BACKEND_AUTOLOAD_FAILED": "torch 导入时自动加载 torch_npu 后端失败",
        "TORCH_NPU_IMPORT_FAILED": "当前解释器无法导入 torch_npu",
        "NPU_QUERY_FAILED": "查询 PyTorch NPU 状态时发生错误",
        "NPU_UNAVAILABLE": "PyTorch NPU 后端未识别到可用设备",
        "NPU_CHECK_FAILED": "NPU 可用性检查未正常完成",
    }
    causes = {
        "TORCH_IMPORT_FAILED": ["当前解释器未安装 torch，或 torch 安装损坏"],
        "TORCH_BACKEND_AUTOLOAD_FAILED": [
            "torch 与 torch_npu 版本不匹配",
            "torch_npu 后端扩展或其 CANN 依赖加载失败",
        ],
        "TORCH_NPU_IMPORT_FAILED": [
            "torch_npu 未安装",
            "torch 与 torch_npu 版本不匹配",
            "CANN 运行库不可见",
        ],
        "NPU_QUERY_FAILED": [
            "驱动或 CANN 运行时初始化失败",
            "torch_npu 与运行环境不匹配",
        ],
        "NPU_UNAVAILABLE": [
            "机器没有映射 NPU",
            "驱动未加载",
            "容器未映射设备",
            "CANN 运行环境未加载",
        ],
        "NPU_CHECK_FAILED": ["NPU 检查子进程异常退出"],
    }
    actions = {
        "TORCH_IMPORT_FAILED": ["在当前解释器中检查或安装与环境匹配的 torch"],
        "TORCH_BACKEND_AUTOLOAD_FAILED": [
            "核对 torch、torch_npu 与 CANN 的版本配套，并检查后端扩展加载错误"
        ],
        "TORCH_NPU_IMPORT_FAILED": [
            "核对并修复 torch、torch_npu 与 CANN 的版本配套关系"
        ],
        "NPU_QUERY_FAILED": ["检查驱动、CANN 运行时、设备权限和 torch_npu 初始化日志"],
        "NPU_UNAVAILABLE": [
            "使用设备管理工具确认 NPU 是否可见、健康并已映射到当前环境"
        ],
        "NPU_CHECK_FAILED": ["检查 NPU 探测子进程的原始错误"],
    }
    return (
        result(
            "FAIL",
            code_name,
            messages.get(code_name, messages["NPU_CHECK_FAILED"]),
            detail=payload.get("detail", last_detail(proc)),
            causes=causes.get(code_name, causes["NPU_CHECK_FAILED"]),
            actions=actions.get(code_name, actions["NPU_CHECK_FAILED"]),
        ),
        runtime,
    )


def inspect_cann_context() -> dict[str, Any]:
    home = os.environ.get("ASCEND_HOME_PATH")
    version = None
    if home:
        version_file = Path(home) / "compiler" / "version.info"
        try:
            for line in version_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("Version="):
                    version = line.split("=", 1)[1].strip().strip('"')
        except OSError:
            pass
    opp = os.environ.get("ASCEND_OPP_PATH")
    vendors = inspect_vendors(opp)
    simulator = inspect_simulators(home)
    toolkit_ready = bool(
        home and Path(home).is_dir() and (Path(home) / "compiler").is_dir()
    )
    warnings: list[str] = []
    if not home:
        warnings.append("ASCEND_HOME_PATH 未设置")
    elif not toolkit_ready:
        warnings.append("ASCEND_HOME_PATH 存在，但未找到 compiler 目录")
    if not opp:
        warnings.append(
            "ASCEND_OPP_PATH 未设置；DSL 编译可能可用，但算子运行环境可能不完整"
        )

    return {
        "status": "PASS" if toolkit_ready and not warnings else "WARNING",
        "ascend_home_path": home,
        "cann_version": version,
        "set_env_script_exists": bool(home and (Path(home) / "set_env.sh").is_file()),
        "ascend_opp_path": opp,
        "opp_path_exists": bool(opp and Path(opp).is_dir()),
        "vendors": vendors,
        "simulator": simulator,
        "tools": {
            "msprof": shutil.which("msprof"),
            "cannsim": shutil.which("cannsim"),
        },
        "warnings": warnings,
    }


def parse_asys_field(text: str, label: str) -> str | None:
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] == label and len(cells) > 1:
            return cells[1] or None
    return None


def inspect_host_npu() -> dict[str, Any]:
    inventory: dict[str, Any] = {
        "source": "unavailable",
        "detected_count": None,
        "devices": [],
        "architecture": {
            "full_soc": None,
            "npu_arch": None,
            "source": None,
        },
        "warnings": [],
    }

    inspect_npu_smi(inventory)
    inspect_asys(inventory)

    if inventory["source"] == "unavailable":
        inventory["warnings"].append("无法通过 npu-smi 或 asys 获取服务器物理设备清单")
    inventory["note"] = (
        "物理设备清单用于环境诊断；设备名称不从 npu-smi Chip Name 推导，核数未知时不猜测"
    )
    return inventory


def build_env_selected_soc(report: dict[str, Any]) -> dict[str, Any]:
    """选择第一张运行时可用卡，并记录探针直接获得的芯片身份。"""
    visible = report["server_information"]["npu"]["runtime_visible"]
    devices = visible.get("devices", []) if visible.get("available") else []
    selected = next(
        (
            device
            for device in devices
            if isinstance(device, dict) and device.get("name")
        ),
        None,
    )
    architecture = report["server_information"]["npu"]["physical_inventory"].get(
        "architecture", {}
    )
    selection = {
        "selection_status": "NO_AVAILABLE_DEVICE",
        "policy": "first_runtime_visible_device",
        "logical_device_id": None,
        "device_name": None,
        "message": "environment.yaml 未识别到可用且带名称的运行时设备",
    }
    if selected is None:
        return {
            "selection": selection,
            "chip_identification": None,
        }

    device_name = str(selected["name"])
    full_soc = architecture.get("full_soc")
    npu_arch = architecture.get("npu_arch")
    selection.update(
        {
            "selection_status": "SELECTED",
            "logical_device_id": selected.get("logical_device_id"),
            "device_name": device_name,
            "message": "已选择 environment.yaml 中第一张运行时可用卡",
        }
    )
    return {
        "selection": selection,
        "chip_identification": {
            "device_name": device_name,
            "full_soc": full_soc,
            "npu_arch": npu_arch,
            "runtime_source": "torch_npu",
            "architecture_source": architecture.get("source"),
        },
    }


def build_report() -> dict[str, Any]:
    l0, module = check_l0()
    l1 = (
        check_compile_stage("translate")
        if l0["status"] == "PASS"
        else skipped("CANNBOTDSL_IMPORT_REQUIRED", "未执行：CANNBotDSL 导入检查未通过")
    )
    l2 = (
        check_compile_stage("compile")
        if l1["status"] == "PASS"
        else skipped("DSL_TRANSLATION_REQUIRED", "未执行：DSL 翻译检查未通过")
    )
    l3, npu = check_l3()
    checks = {
        "cannbotdsl_import": l0,
        "dsl_translation": l1,
        "dsl_compilation": l2,
        "npu_runtime": l3,
    }
    gate, execution = execution_capabilities(checks, l3)
    cann = inspect_cann_context()
    host_npu = inspect_host_npu()
    overall_status, overall_summary = overall_result(gate, execution)

    return {
        "final_status": {
            "status": overall_status,
            "summary": overall_summary,
            "compile_gate": gate["status"],
            "execution_mode": execution["mode"],
            "reason": execution["reason"],
            "allowed_actions": execution["allowed_actions"],
            "blocked_actions": execution["blocked_actions"],
        },
        "stage_status": checks,
        "software_environment": {
            "runtime": {
                "python_executable": sys.executable,
                "python_version": sys.version.split()[0],
                "cannbotdsl_module_file": module.get("module_file"),
                "cannbotdsl_version": module.get("version"),
            },
            "cann": cann,
        },
        "server_information": server_information(l3, npu, host_npu),
        "checked_at": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
    }


class IndentedSafeDumper(yaml.SafeDumper):
    """让序列相对父键缩进，便于逐行注释和人工阅读。"""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> Any:
        return super().increase_indent(flow, False)


FIELD_COMMENTS = {
    "final_status": "本次环境检查的最终结论；优先阅读本节",
    "compile_gate": "DSL 生成与编译门禁状态",
    "execution_mode": "当前可执行范围：full、compile_only 或 blocked",
    "stage_status": "各检查阶段的独立结果；用于定位最终结论的原因",
    "cannbotdsl_import": "当前解释器导入 CANNBotDSL 的结果",
    "dsl_translation": "固定探针完成 DSL 前端与 AscendC 翻译的结果",
    "dsl_compilation": "固定探针经编译后端生成真实 .so 的结果",
    "npu_runtime": "当前 Python 环境识别和访问 NPU 的结果",
    "status": "当前检查状态；核心能力为 PASS/FAIL/SKIP，环境概览可为 WARNING",
    "code": "稳定的问题代码；OK 表示该检查通过",
    "summary": "面向用户的检查结论",
    "detail": "探针或命令返回的原始关键证据",
    "possible_causes": "检查失败时的可能原因列表",
    "actions": "建议执行的处理动作列表",
    "requirement": "门禁要求满足的核心能力",
    "reason": "当前门禁或执行模式的判定原因",
    "mode": "执行模式：full、compile_only 或 blocked",
    "allowed_actions": "当前环境允许继续执行的动作",
    "blocked_actions": "当前环境禁止执行或禁止宣称已验证的动作",
    "software_environment": "本次检查实际使用的软件环境",
    "runtime": "Python 与 CANNBotDSL 运行上下文",
    "python_executable": "实际执行检查的 Python 解释器",
    "python_version": "实际执行检查的 Python 版本",
    "cannbotdsl_module_file": "实际导入的 CANNBotDSL 模块路径",
    "cannbotdsl_version": "CANNBotDSL 自身声明的版本；未声明时为 null",
    "cann": "当前进程中的 CANN Toolkit、OPP、Simulator 与工具信息",
    "ascend_home_path": "当前 ASCEND_HOME_PATH；未设置时为 null",
    "cann_version": "从当前 Toolkit 的 compiler/version.info 读取的 CANN 版本",
    "set_env_script_exists": "当前 Toolkit 根目录是否存在 set_env.sh",
    "ascend_opp_path": "当前 ASCEND_OPP_PATH；未设置时为 null",
    "opp_path_exists": "ASCEND_OPP_PATH 指向的目录是否真实存在",
    "vendors": "OPP 中发现的 vendor 及其 op_api 动态库数量",
    "name": "当前条目的名称",
    "op_api_library_count": "该 vendor 中发现的 op_api .so 数量",
    "simulator": "当前 CANN 环境中发现的 Simulator 平台",
    "platform": "Simulator 平台目录名称",
    "available": "该能力、设备或 Simulator 是否可用",
    "tools": "当前 PATH 中发现的可选 CANN 工具",
    "msprof": "msprof 可执行文件路径；未找到时为 null",
    "cannsim": "cannsim 可执行文件路径；未找到时为 null",
    "warnings": "不直接改变核心编译门禁的环境警告",
    "server_information": "服务器硬件及设备信息",
    "npu": "服务器 NPU 信息",
    "runtime_visible": "当前进程通过 torch_npu 实际可见的逻辑设备",
    "physical_inventory": "通过 npu-smi 或 asys 获取的服务器物理设备信息",
    "scope": "可见设备统计的作用域",
    "visible_device_count": "当前进程通过 torch_npu 可见的逻辑设备数量",
    "device_count": "当前进程可见的逻辑设备数量",
    "devices": "在当前作用域内发现的设备列表",
    "host_inventory": "通过 npu-smi 或 asys 获取的服务器物理设备信息",
    "source": "当前信息的探测来源",
    "detected_count": "设备工具检测到的服务器物理设备数量；未知时为 null",
    "physical_device_id": "设备工具报告的物理设备编号",
    "health": "npu-smi 结构化健康查询结果；未知时为 null",
    "architecture": "asys 硬件报告提供的芯片与 NPU 架构信息",
    "full_soc": "asys Chip Info 提供的完整芯片信息",
    "npu_arch": "asys Arch Info 提供的 NPU 架构编号",
    "note": "该部分数据的使用限制和解释说明",
    "checked_at": "本次检查完成时的 UTC 时间",
    "selection": "默认使用卡的选择结果",
    "selection_status": "默认使用卡的选择状态",
    "policy": "默认卡选择策略",
    "logical_device_id": "environment.yaml 中被选中卡的逻辑设备编号",
    "device_name": "torch_npu 识别的设备名称",
    "message": "默认卡选择结果的中文说明",
    "chip_identification": "环境探针获取的芯片身份信息；不包含推导参数",
    "runtime_source": "设备名称的探测来源",
    "architecture_source": "完整芯片名和架构号的探测来源",
}

LIST_ITEM_COMMENTS = {
    "possible_causes": "一个可能原因",
    "actions": "一个建议处理动作",
    "allowed_actions": "一个允许继续执行的动作",
    "blocked_actions": "一个禁止执行或禁止宣称已验证的动作",
    "warnings": "一条环境警告",
    "vendors": "一个 OPP vendor",
    "simulator": "一个 Simulator 平台",
    "devices": "一个设备条目",
}


def annotate_yaml(text: str, preferred_column: int = 64) -> str:
    """逐行追加中文注释；按相邻层级对齐，且不受全局长值影响。"""
    rows: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stripped = line.strip()
        key = ""
        is_list = stripped.startswith("- ")
        body = stripped[2:] if is_list else stripped
        if ":" in body:
            key = body.split(":", 1)[0].strip().strip("'\"")
        parent = stack[-1][1] if stack else ""
        comment = FIELD_COMMENTS.get(key)
        if is_list and not comment:
            comment = LIST_ITEM_COMMENTS.get(parent, f"{parent or '列表'}中的一个条目")
        if not comment:
            comment = f"{key or parent} 字段"
        rows.append((line, comment))
        if not is_list and body.endswith(":"):
            stack.append((indent, key))

    def display_width(value: str) -> int:
        return sum(
            2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in value
        )

    output: list[str] = []
    for line, comment in rows:
        width = display_width(line)
        # 普通行固定在适中的列开始注释；长路径或长诊断只留两个空格，
        # 不再把整份文件的注释列推到数百字符之外。
        padding = preferred_column - width if width < preferred_column else 2
        output.append(f"{line}{' ' * max(padding, 2)}# {comment}")
    return "\n".join(output) + "\n"


def write_yaml_atomic(report: dict[str, Any], destination: Path) -> None:
    raw = yaml.dump(
        report,
        Dumper=IndentedSafeDumper,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=10000,
    )
    text = annotate_yaml(raw)
    if yaml.safe_load(text) != report:
        raise RuntimeError("environment.yaml 回读结果与源数据不一致")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=".environment-", suffix=".yaml.tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, destination)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def print_summary(
    report: dict[str, Any], selected_soc: dict[str, Any], destination: Path
) -> None:
    labels = {
        "cannbotdsl_import": "CANNBotDSL 导入",
        "dsl_translation": "DSL 翻译",
        "dsl_compilation": "DSL 编译",
        "npu_runtime": "NPU 运行环境",
    }
    LOGGER.info("CANNBotDSL 环境检查")
    for key, label in labels.items():
        item = report["stage_status"][key]
        LOGGER.info(f"  {label}: {item['status']} — {item['summary']}")
        if item["status"] == "FAIL":
            LOGGER.info(f"    问题代码: {item['code']}")
            if item["detail"]:
                LOGGER.info(f"    原始信息: {item['detail']}")
            for action in item["actions"]:
                LOGGER.info(f"    建议: {action}")
    final = report["final_status"]
    LOGGER.info(f"  最终状态: {final['status']} — {final['summary']}")
    LOGGER.info(f"  编译门禁: {final['compile_gate']}")
    LOGGER.info(f"  执行模式: {final['execution_mode']} — {final['reason']}")
    selection = selected_soc["selection"]
    LOGGER.info(
        f"  默认使用卡: {selection['selection_status']} — {selection['message']}"
    )
    LOGGER.info(f"  产物: {destination}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="产物根目录；其下生成 environment",
    )
    args = parser.parse_args()
    output_root = args.output_dir.expanduser().resolve()
    if output_root == Path(output_root.anchor):
        parser.error("--output-dir 不能是文件系统根目录")
    environment_dir = output_root / "environment"
    destination = environment_dir / "environment.yaml"
    selected_soc_destination = environment_dir / "env_selected_soc.yaml"
    report = build_report()
    selected_soc = build_env_selected_soc(report)
    write_yaml_atomic(report, destination)
    write_yaml_atomic(selected_soc, selected_soc_destination)
    try:
        display = destination.relative_to(Path.cwd().resolve())
    except ValueError:
        display = destination
    print_summary(report, selected_soc, display)
    return 0 if report["final_status"]["compile_gate"] == "PASS" else 1


def inspect_vendors(opp):
    vendors: list[dict[str, Any]] = []
    if opp:
        vendor_root = Path(opp) / "vendors"
        if vendor_root.is_dir():
            for vendor in sorted(
                path for path in vendor_root.iterdir() if path.is_dir()
            ):
                libraries = list(vendor.glob("op_api/lib/*.so"))
                libraries += list(vendor.glob("op_impl/ai_core/tbe/op_api/lib/*.so"))
                vendors.append(
                    {"name": vendor.name, "op_api_library_count": len(libraries)}
                )

    return vendors


def inspect_simulators(home):
    simulator: list[dict[str, Any]] = []
    if home:
        for host_arch in ("x86_64-linux", "aarch64-linux"):
            root = Path(home) / host_arch / "simulator"
            if root.is_dir():
                for platform in sorted(
                    path for path in root.iterdir() if path.is_dir()
                ):
                    simulator.append(
                        {
                            "platform": platform.name,
                            "available": (
                                platform / "lib" / "libruntime_camodel.so"
                            ).is_file(),
                        }
                    )

    return simulator


def inspect_npu_smi(inventory):
    npu_smi = shutil.which("npu-smi")
    if npu_smi:
        proc = run([npu_smi, "info", "-m"], timeout=15)
        if proc.returncode == 0:
            device_ids = physical_device_ids(proc.stdout)
            if device_ids:
                inventory["source"] = "npu-smi"
                inventory["detected_count"] = len(device_ids)
                for device_id in device_ids:
                    health_proc = run(
                        [npu_smi, "info", "-t", "health", "-i", str(device_id)],
                        timeout=10,
                    )
                    health_match = re.search(
                        r"^\s*Health\s*:\s*(.+?)\s*$", health_proc.stdout, re.MULTILINE
                    )
                    inventory["devices"].append(
                        {
                            "physical_device_id": device_id,
                            "health": health_match.group(1) if health_match else None,
                        }
                    )
            else:
                inventory["warnings"].append("npu-smi info -m 未返回可识别的设备 ID")
        else:
            inventory["warnings"].append(
                f"npu-smi info -m 执行失败：{last_detail(proc)}"
            )


def inspect_asys(inventory):
    home = os.environ.get("ASCEND_HOME_PATH")
    asys_candidates = []
    if home:
        asys_candidates.append(
            Path(home) / "tools" / "ascend_system_advisor" / "asys" / "asys"
        )
    asys_in_path = shutil.which("asys")
    if asys_in_path:
        asys_candidates.append(Path(asys_in_path))
    asys = next(
        (
            path
            for path in asys_candidates
            if path.is_file() and os.access(path, os.X_OK)
        ),
        None,
    )
    if asys:
        hardware = run([str(asys), "info", "-r=hardware"], timeout=20)
        if hardware.returncode == 0:
            full_soc = parse_asys_field(hardware.stdout, "Chip Info")
            npu_arch = parse_asys_field(hardware.stdout, "Arch Info")
            inventory["architecture"] = {
                "full_soc": full_soc,
                "npu_arch": npu_arch,
                "source": "asys" if full_soc or npu_arch else None,
            }
            if inventory["detected_count"] is None:
                count = parse_asys_field(hardware.stdout, "NPU Count")
                if count and count.isdigit():
                    inventory["detected_count"] = int(count)
                    inventory["source"] = "asys"
        else:
            inventory["warnings"].append(f"asys 硬件查询失败：{last_detail(hardware)}")


def execution_capabilities(checks, l3):
    chain_passed = all(
        checks[key]["status"] == "PASS"
        for key in ("cannbotdsl_import", "dsl_translation", "dsl_compilation")
    )
    if not chain_passed:
        gate = {
            "status": "FAIL",
            "requirement": "CANNBotDSL 可导入、DSL 可翻译且可编译",
            "reason": "DSL 编译能力链未全部通过",
        }
        execution = {
            "mode": "blocked",
            "reason": "编译能力链未通过，先修复首个失败级别",
            "allowed_actions": ["environment_diagnosis"],
            "blocked_actions": [
                "dsl_generate",
                "dsl_compile",
                "npu_correctness",
                "npu_performance",
            ],
        }
    elif l3["status"] == "PASS":
        gate = {
            "status": "PASS",
            "requirement": "CANNBotDSL 可导入、DSL 可翻译且可编译",
            "reason": "DSL 编译能力链通过",
        }
        execution = {
            "mode": "full",
            "reason": "编译能力与 NPU 可用性均已确认",
            "allowed_actions": [
                "dsl_generate",
                "dsl_compile",
                "npu_correctness",
                "npu_performance",
            ],
            "blocked_actions": [],
        }
    else:
        gate = {
            "status": "PASS",
            "requirement": "CANNBotDSL 可导入、DSL 可翻译且可编译",
            "reason": "DSL 编译能力链通过",
        }
        execution = {
            "mode": "compile_only",
            "reason": "编译能力已确认，但 NPU 不可用；仅跳过真机验证",
            "allowed_actions": ["dsl_generate", "dsl_compile"],
            "blocked_actions": ["npu_correctness", "npu_performance"],
        }

    return gate, execution


def server_information(l3, npu, host_npu):
    return {
        "npu": {
            "runtime_visible": {
                "scope": "current_process_visible_devices",
                "available": l3["status"] == "PASS",
                "device_count": npu["visible_device_count"],
                "devices": npu["devices"],
            },
            "physical_inventory": host_npu,
            "note": "运行时可见设备用于任务执行判断；物理设备信息来自 npu-smi/asys，无法可靠获取的核数不猜测",
        },
    }


def physical_device_ids(output):
    device_ids: list[int] = []
    for line in output.splitlines():
        match = re.match(r"^\s*(\d+)\s+", line)
        if match:
            device_id = int(match.group(1))
            if device_id not in device_ids:
                device_ids.append(device_id)
    return device_ids


def overall_result(gate, execution):
    if gate["status"] == "FAIL":
        overall_status = "FAIL"
        overall_summary = "环境未通过 DSL 编译门禁，当前只能进行环境诊断"
    elif execution["mode"] == "compile_only":
        overall_status = "WARNING"
        overall_summary = "DSL 生成与编译可用，但当前不能进行 NPU 真机验证"
    else:
        overall_status = "PASS"
        overall_summary = (
            "DSL 翻译、编译通过且 NPU 可访问；尚未执行 DSL kernel 或验证精度、性能"
        )

    return overall_status, overall_summary


if __name__ == "__main__":
    raise SystemExit(main())
