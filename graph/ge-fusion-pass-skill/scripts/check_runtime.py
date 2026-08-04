#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Probe the Python/ES/custom-op runtime without modifying the environment.

This is deliberately a capability probe.  Missing TensorFlow, TorchAir, the
GE Python bridge, or a custom NPU operator produces NOT_RUN; it is never
reported as a successful pass validation.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path


OUTPUT_LOGGER = logging.getLogger(f"{__name__}.stdout")
OUTPUT_LOGGER.setLevel(logging.INFO)
OUTPUT_LOGGER.propagate = False


PROBE = r'''
import importlib
import json
import os
import platform
import sys

result = {
    "python": sys.executable,
    "python_version": sys.version.split()[0],
    "platform": platform.machine(),
    "modules": {},
    "custom_op": {},
    "es_custom": {},
}

for name in ("torch_npu", "torchair", "ge", "ge.passes", "tensorflow", "npu_bridge"):
    try:
        module = importlib.import_module(name)
        result["modules"][name] = {"status": "PASSED", "file": getattr(module, "__file__", None),
                                    "version": getattr(module, "__version__", None)}
    except Exception as exc:
        result["modules"][name] = {"status": "NOT_RUN", "reason": f"{type(exc).__name__}: {exc}"}

try:
    module = importlib.import_module("ge.es.custom")
    getattr(module, "AddCustom")
    result["es_custom"] = {"status": "PASSED", "file": getattr(module, "__file__", None)}
except Exception as exc:
    result["es_custom"] = {"status": "NOT_RUN", "reason": f"{type(exc).__name__}: {exc}"}

op_name = os.environ.get("GE_FUSION_CUSTOM_OP", "")
if op_name:
    try:
        import torch
        torch._C._dispatch_find_schema_or_throw(op_name, "")
        has_kernel = bool(torch._C._dispatch_has_kernel_for_dispatch_key(op_name, "PrivateUse1"))
        result["custom_op"] = {"status": "PASSED" if has_kernel else "NOT_RUN", "name": op_name,
                                "registered": True, "privateuse1_kernel": has_kernel}
        if not has_kernel:
            result["custom_op"]["reason"] = "operator schema exists but no PrivateUse1/NPU kernel is registered"
    except Exception as exc:
        result["custom_op"] = {"status": "NOT_RUN", "name": op_name,
                                "registered": False, "privateuse1_kernel": False,
                                "reason": f"{type(exc).__name__}: {exc}"}

bridge = os.environ.get("GE_PYTHON_PASS_BRIDGE")
if not bridge:
    passes = result["modules"].get("ge.passes", {})
    module_file = passes.get("file")
    if module_file:
        root = os.path.dirname(os.path.dirname(module_file))
        for current, _, names in os.walk(root):
            if "libge_python_pass_bridge.so" in names:
                bridge = os.path.join(current, "libge_python_pass_bridge.so")
                break
result["python_pass_bridge"] = {
    "status": "PASSED" if bridge and os.path.isfile(bridge) else "NOT_RUN",
    "path": bridge if bridge and os.path.isfile(bridge) else None,
}
print(json.dumps(result, ensure_ascii=False))
'''


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", dest="python_bin", default=sys.executable,
                        help="要探测的 Python 解释器，默认当前解释器")
    parser.add_argument("--mode", choices=("python-pass", "tf1", "custom-op"), default="python-pass",
                        help="运行时类型；tf1 不要求 TorchAir/GE Python bridge")
    parser.add_argument("--require-custom-op", action="store_true",
                        help="要求 GE_FUSION_CUSTOM_OP 指定的算子已注册")
    parser.add_argument("--custom-op", help="自定义算子全名，例如 npu::npu_add_custom")
    parser.add_argument("--require-es-custom", action="store_true",
                        help="要求 ge.es.custom.AddCustom 可导入")
    parser.add_argument("--require-tf1", action="store_true",
                        help="要求 Python 3.7 + TensorFlow 1.15 + npu_bridge")
    parser.add_argument("--out-json")
    parser.add_argument("--strict", action="store_true", help="NOT_RUN 时返回非零退出码")
    return parser


def _parse_probe_output(completed, command):
    for line in reversed(completed.stdout.splitlines()):
        if not line.strip().startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {
        "status": "NOT_RUN",
        "reason": completed.stderr.strip() or "runtime probe produced no JSON",
        "command": command,
        "returncode": completed.returncode,
    }


def _run_probe(command, probe_env):
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            env=probe_env,
            check=False,
        )
    except OSError as exc:
        return {"status": "NOT_RUN", "reason": str(exc), "command": command}
    return _parse_probe_output(completed, command)


def _python_pass_missing(result):
    missing = []
    modules = result.get("modules", {})
    for name in ("torch_npu", "torchair", "ge", "ge.passes"):
        if modules.get(name, {}).get("status") != "PASSED":
            missing.append(name)
    if result.get("python_pass_bridge", {}).get("status") != "PASSED":
        missing.append("python_pass_bridge")
    return missing


def _tf1_missing(result):
    missing = []
    python_version = result.get("python_version", "")
    modules = result.get("modules", {})
    tensorflow = modules.get("tensorflow", {})
    if not python_version.startswith("3.7"):
        missing.append("python3.7")
    tensorflow_version = str(tensorflow.get("version", ""))
    if tensorflow.get("status") != "PASSED" or not tensorflow_version.startswith("1.15"):
        missing.append("tensorflow==1.15")
    if modules.get("npu_bridge", {}).get("status") != "PASSED":
        missing.append("npu_bridge")
    return missing


def _required_missing(args, result):
    require_tf1 = args.require_tf1 or args.mode == "tf1"
    require_custom_op = args.require_custom_op or args.mode == "custom-op"
    missing = [] if args.mode == "tf1" else _python_pass_missing(result)
    if args.require_es_custom and result.get("es_custom", {}).get("status") != "PASSED":
        missing.append("ge.es.custom.AddCustom")
    if require_custom_op and result.get("custom_op", {}).get("status") != "PASSED":
        missing.append(args.custom_op or os.environ.get("GE_FUSION_CUSTOM_OP", "custom_op_name"))
    if require_tf1:
        missing.extend(_tf1_missing(result))
    return missing, require_custom_op, require_tf1


def _environment_snapshot(probe_env):
    result = {}
    names = (
        "ASCEND_HOME_PATH",
        "ASCEND_OPP_PATH",
        "GE_ES_API_PYTHONPATH",
        "GE_PYTHON_PASS_BRIDGE",
        "GE_FUSION_CUSTOM_OP",
    )
    for name in names:
        if probe_env.get(name):
            result[name] = probe_env.get(name)
    return result


def _finalize_result(args, result, command, probe_env):
    if not isinstance(result.get("modules"), dict):
        return result
    missing, require_custom_op, require_tf1 = _required_missing(args, result)
    result["required_missing"] = missing
    result["status"] = "PASSED" if not missing else "NOT_RUN"
    result["command"] = command
    result["environment"] = _environment_snapshot(probe_env)
    result["probe_requirements"] = {
        "mode": args.mode,
        "require_es_custom": args.require_es_custom,
        "require_custom_op": require_custom_op,
        "require_tf1": require_tf1,
    }
    return result


def _write_result(path, result):
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def _emit_json(value):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    OUTPUT_LOGGER.handlers = [handler]
    OUTPUT_LOGGER.info("%s", json.dumps(value, ensure_ascii=False, indent=2))


def main():
    args = _build_parser().parse_args()

    command = [args.python_bin, "-c", PROBE]
    probe_env = os.environ.copy()
    if args.custom_op:
        probe_env["GE_FUSION_CUSTOM_OP"] = args.custom_op
    result = _run_probe(command, probe_env)
    result = _finalize_result(args, result, command, probe_env)
    _write_result(args.out_json, result)
    _emit_json(result)
    return 1 if args.strict and result.get("status") != "PASSED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
