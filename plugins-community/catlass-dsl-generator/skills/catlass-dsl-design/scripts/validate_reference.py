#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Execute the Torch reference embedded in definition.json for every workload."""

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
import uuid
from pathlib import Path


class ReferenceError(ValueError):
    pass


OPAQUE_CASE_AXES = {
    "case", "case_id", "case_index", "case_number", "config", "config_id",
    "config_index",
}


def _load_bench():
    path = (
        Path(__file__).resolve().parents[2]
        / "catlass-dsl-bench"
        / "scripts"
        / "bench.py"
    )
    spec = importlib.util.spec_from_file_location("catlass_dsl_bench_for_design", path)
    if spec is None or spec.loader is None:
        raise ReferenceError("无法加载 catlass-dsl-bench/scripts/bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_reference(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ReferenceError("reference.py 不存在或不是普通文件")
    source = path.read_text(encoding="utf-8")
    return source


def _output_summary(value, torch, path="$"):
    if torch.is_tensor(value):
        return [
            {
                "path": path,
                "shape": list(value.shape),
                "dtype": str(value.dtype).split(".")[-1],
                "device": str(value.device),
            }
        ]
    if type(value) not in (tuple, list) or not value:
        raise ReferenceError("reference 输出必须是非空 Tensor 或 Tensor tuple/list")
    leaves = []
    for index, item in enumerate(value):
        leaves.extend(_output_summary(item, torch, "{}[{}]".format(path, index)))
    return leaves


def _validate_outputs(leaves, definition, axes):
    expected = list(definition["outputs"].items())
    if len(leaves) != len(expected):
        raise ReferenceError(
            "reference 输出数量不一致：{} != {}".format(len(leaves), len(expected))
        )
    for index, (name, spec) in enumerate(expected):
        leaf = leaves[index]
        shape = (
            None
            if spec.get("shape") is None
            else [
                int(axis) if axis.isdigit() else axes[axis]
                for axis in spec.get("shape")
            ]
        )
        if shape is not None and leaf["shape"] != shape:
            raise ReferenceError(
                "输出 {} shape 不一致：{} != {}".format(name, leaf["shape"], shape)
            )
        if spec.get("dtype") is not None and leaf["dtype"] != spec.get("dtype"):
            raise ReferenceError(
                "输出 {} dtype 不一致：{} != {}".format(
                    name, leaf["dtype"], spec["dtype"]
                )
            )


def validate_reference(
    reference,
    definition,
    workload,
    *,
    device="cpu",
    seed=1,
):
    bench = _load_bench()
    source = _load_reference(reference)
    definition_data = bench.load_definition(definition)
    workloads, workload_path, workload_sha256 = bench.load_workloads(workload)
    opaque_axes = sorted(
        name for name in definition_data["axes"]
        if name.lower() in OPAQUE_CASE_AXES
    )
    if opaque_axes:
        raise ReferenceError(
            "case 完整配置必须保存在 workload.jsonl，禁止不透明 case/config axis: {}"
            .format(opaque_axes)
        )
    if definition_data["reference"] != source:
        raise ReferenceError(
            "definition.reference 必须与 reference.py 的完整内容逐字一致"
        )

    import torch

    synchronize = bench.configure_device(device, torch)
    module = bench._load_module(
        Path(reference).resolve(),
        "catlass_dsl_design_reference_{}".format(uuid.uuid4().hex),
    )
    run = getattr(module, "run", None)
    if not callable(run):
        raise ReferenceError("reference.py 必须导出 callable run")
    if hasattr(module, "reference_cases"):
        raise ReferenceError("用例必须使用 workload.jsonl，不得导出 reference_cases")
    records = []
    for workload_data in workloads:
        inputs, axes = bench.generate_inputs(
            definition_data,
            workload_data,
            module,
            device,
            seed,
            Path(workload_path).parent,
        )
        output = run(
            *[bench.clone_value(item, torch) for item in inputs]
        )
        synchronize()
        try:
            outputs = bench.normalize_outputs(
                output, definition_data["outputs"], torch
            )
        except Exception as exc:
            raise ReferenceError("reference 输出不符合 definition: {}".format(exc)) from exc
        output_values = list(outputs.values())
        leaves = _output_summary(
            output_values[0] if len(output_values) == 1 else output_values, torch
        )
        _validate_outputs(leaves, definition_data, axes)
        records.append(
            {
                "case_id": workload_data["uuid"],
                "status": "passed",
                "line": workload_data["_line"],
                "device": device,
                "seed": seed,
                "axes": axes,
                "outputs": leaves,
            }
        )

    reference_path = Path(reference).resolve()
    return {
        "status": "passed",
        "reference": str(reference_path),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "definition": definition_data["_path"],
        "definition_sha256": definition_data["_sha256"],
        "workload": workload_path,
        "workload_sha256": workload_sha256,
        "cases": records,
    }


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".reference.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def update_state(path, result):
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink():
        raise ReferenceError("state.json 不存在或不是普通文件")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema") != "catlass.dsl.workflow.v3":
        raise ReferenceError("state schema 必须是 catlass.dsl.workflow.v3")
    config = state.get("config")
    if not isinstance(config, dict):
        raise ReferenceError("state.config 必须是对象")
    config["reference_validation"] = result
    approval = config.get("approval")
    if isinstance(approval, dict):
        approval["config_digest"] = "<computed>"
    state.pop("config_digest", None)
    _write_json(path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--definition", required=True, type=Path)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    try:
        result = validate_reference(
            args.reference,
            args.definition,
            args.workload,
            device=args.device,
            seed=args.seed,
        )
        update_state(args.state, result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        result = {
            "status": "failed",
            "reference": str(args.reference.resolve()),
            "definition": str(args.definition.resolve()),
            "workload": str(args.workload.resolve()),
            "error": "{}: {}".format(type(exc).__name__, exc),
        }
        try:
            update_state(args.state, result)
        except Exception as state_exc:
            result["state_error"] = "{}: {}".format(type(state_exc).__name__, state_exc)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
