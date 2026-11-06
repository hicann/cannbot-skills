# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Installed-wheel contract checks, called by smoke.py in its new user project."""

import json
from pathlib import Path
import re
import shutil

from accel_model.app.case.native_case import run_native_case
from accel_model.model.case import resolve_cli_hardware, resolve_eval_hardware
from accel_model.model.validation import score_case
import yaml

ROOT = Path(__file__).resolve().parents[1]


def save(root, name, value):
    (root / "definition" / name).write_text(yaml.safe_dump(value, sort_keys=False))


def base_model(name):
    root = Path("work") / name
    (root / "definition").mkdir(parents=True)
    op = dict(
        name=name,
        hardware="ascend_950DT",
        params=dict(rows=16, cols=32, num_tasks=1, num_groups=1),
        buffers=[],
        pipeline=dict(phases=[["Copy"]], tokens={}, dataflow=[]),
    )
    return root, op


def python_stage():
    root, op = base_model("python_copy")
    shutil.copyfile(
        ROOT / "assets/examples/hooks/copy_stage.py", root / "copy_stage.py"
    )
    (root / "__init__.py").write_text(
        'from .copy_stage import CopyStage\nSTAGES = {"Copy": CopyStage}\n'
    )
    op["buffers"] = [
        dict(name="scratch", mem="UB", bytes="rows * cols * fp32", slots=1)
    ]
    save(root, "operator.yaml", op)
    cases = []
    for rows in (16, 32):
        _, report, _ = score_case(
            root, dict(rows=rows), gates=["runtime", "backend_agreement"]
        )
        assert report["passed"] and not report["qualification"]["verified"]
        assert report["numeric_evidence"]["checks"]["numeric_execution"] == "not_run"
        for result in report["results"].values():
            assert result["scope"] == "native_aggregate"
            assert not result["physical_assignment_verified"]
            core = result["cores"][0]
            assert core["instructions"] == 2
            assert core["bytes"] == {"GM->UB": rows * 32 * 4, "UB->GM": rows * 32 * 4}
        cases.append(dict(rows=rows, cycles=report["cycles"]))
    return cases


def physical_stage():
    root = Path("work/physical_python")
    shutil.copytree(ROOT / "assets/examples/physical", root)
    dag_path = root / "definition/dag.yaml"
    dag = yaml.safe_load(dag_path.read_text())
    stage = dag["kernels"].pop("Vector")
    dag_path.write_text(yaml.safe_dump(dag))
    (root / "__init__.py").write_text(
        "from accel_model.kernel.stages_base import Stage\n"
        "class Vector(Stage):\n"
        "    def build(self, context):\n"
        "        assert context.owner is not None\n"
        f"        return {stage!r}\n"
        'STAGES = {"Vector": Vector}\n'
    )
    case = yaml.safe_load((root / "case.yaml").read_text())
    report = run_native_case(
        root, case, output_dir="work/physical_python_result", backend="both"
    )
    baseline = json.loads(Path("work/physical/result.json").read_text())
    assert report["passed"]
    assert report["total_cycles"] == baseline["total_cycles"]
    assert report["validation"] == baseline["validation"]
    return dict(cycles=report["total_cycles"], yaml_python_agreement=True)


def numeric_contract():
    root, op = base_model("numeric_decode")
    save(
        root,
        "workload.yaml",
        dict(
            parameters=dict(required=[], defaults={}),
            derive=[],
            assertions=[],
            semantic_bindings={},
            core_projection=dict(owner="operator"),
        ),
    )
    source = ROOT / "references/framework/numeric-execution-validation.md"
    block = re.findall(r"^```yaml\n(.*?)^```", source.read_text(), re.M | re.S)[0]
    numeric = yaml.safe_load(block)["numeric"]
    encoded, decoded = numeric["reads"][0], numeric["writes"][0]
    x = {**encoded, "allocation": "x", "memory": "GM", "owner": "device"}
    y = {**decoded, "allocation": "y", "memory": "GM", "owner": "device"}

    def move(name, src, dst):
        return dict(
            move=dict(
                id=name,
                src=dict(tensor=src["allocation"], mem=src["memory"]),
                dst=dict(tensor=dst["allocation"], mem=dst["memory"]),
                size=src["shape"],
                aiv_presplit=True,
                numeric=dict(operation="copy", reads=[src], writes=[dst]),
            )
        )

    stage = dict(
        body=[
            move("load", x, encoded),
            dict(
                vec=dict(
                    id="decode",
                    name="Decode",
                    cycles=2,
                    aiv_presplit=True,
                    numeric=numeric,
                )
            ),
            move("store", decoded, y),
        ]
    )
    save(
        root,
        "dag.yaml",
        dict(
            task_graph=dict(
                nodes=["Copy"], edges=[], implementation={"Copy": ["decode"]}
            ),
            kernels=dict(
                dtypes=dict(x="uint8", encoded="uint8", decoded="fp32", y="fp32"),
                Copy=stage,
            ),
        ),
    )
    save(
        root,
        "semantic.yaml",
        dict(
            dimensions={},
            tensors=[
                dict(name="x", shape=[1, 32], dtype="uint8", role="input"),
                dict(name="y", shape=[1, 32], dtype="fp32", role="output"),
            ],
            operations=[
                dict(
                    name="decode",
                    operation="decode_quantized",
                    inputs=[dict(tensor="x", version=0)],
                    outputs=[dict(tensor="y", version=0)],
                    attrs=dict(format="e4m3"),
                    state_effects=[],
                )
            ],
            outputs=[dict(tensor="y", version=0)],
            layouts=[],
            final_state_versions=[],
            numeric=dict(mode="exact", atol=0, rtol=0, equal_nan=False),
            tensor_numeric=[],
            transform_sources=[],
        ),
    )
    # E4M3 encodes 1 as 0x38; expected values are independent of model evaluation.
    case = dict(
        name="ones",
        inputs={"x": [56] * 32},
        expected={"y": [1] * 32},
        input_bindings={"x": x},
        output_bindings={"y": y},
    )
    op["validation"] = dict(semantic=True, execution_cases=[case])
    save(root, "operator.yaml", op)
    _, report, _ = score_case(root)
    assert report["qualification"]["verified"]
    assert report["validation"]["numeric_execution"]["status"] == "passed"
    case["expected"]["y"][0] = 2
    save(root, "operator.yaml", op)
    try:
        score_case(root)
    except ValueError:
        rejected = True
    else:
        raise AssertionError(
            "Wrong expected value must fail execution numeric validation"
        )
    return dict(execution_numeric="passed", wrong_expected_rejected=rejected)


def hardware_selection():
    model = "work/my_matmul"
    selected = Path(resolve_eval_hardware(model))
    assert selected.name == "ascend_950DT.yaml" and selected.is_file()
    explicit = Path(model, "hardware.yaml").resolve()
    assert Path(resolve_eval_hardware(model, str(explicit))) == explicit
    try:
        resolve_cli_hardware(model, "theoretical")
    except ValueError:
        pass
    else:
        raise AssertionError("A non-hardware alias must not select a profile")
    return dict(selected=selected.name, explicit_preserved=True)


if __name__ == "__main__":
    print(
        json.dumps(
            dict(
                python_stage=python_stage(),
                physical_stage=physical_stage(),
                numeric_contract=numeric_contract(),
                hardware_selection=hardware_selection(),
            ),
            ensure_ascii=False,
        )
    )
