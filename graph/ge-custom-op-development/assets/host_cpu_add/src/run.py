#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Build a constant graph and verify HostCpu constant folding in Python."""

import math
import os
import traceback

from ge.es.graph_builder import GraphBuilder
from ge.ge_global import GeApi
from ge.session import Session
from ge.graph.types import DataType

try:
    from ge.es.custom import AddCustom
except ImportError as import_error:
    raise RuntimeError(
        "Custom-op ES APIs are unavailable. Run run.sh first."
    ) from import_error


GRAPH_ID = 0
DEVICE_ID = int(os.environ.get("DEVICE_ID", "0"))
LEFT_VALUE = 1.0
RIGHT_VALUE = 2.0
EXPECTED_VALUE = 3.0
TOLERANCE = 1.0e-6


def build_graph(name):
    builder = GraphBuilder(name)
    left = builder.create_const_float([LEFT_VALUE], [1])
    right = builder.create_const_float([RIGHT_VALUE], [1])
    builder.set_graph_output(AddCustom(left, right), 0)
    return builder.build_and_reset()


def verify_output(outputs):
    if len(outputs) != 1:
        raise RuntimeError("RunGraph returned {} outputs".format(len(outputs)))
    output = outputs[0]
    if list(output.shape) != [1] or output.data_type != DataType.DT_FLOAT:
        raise RuntimeError("unexpected output shape or dtype")
    values = list(output.data)
    print("output shape: [{}]".format(", ".join(str(dim) for dim in output.shape)))
    print("output values: {}".format(" ".join(str(value) for value in values)))
    if (
        len(values) != 1
        or not math.isfinite(values[0])
        or abs(values[0] - EXPECTED_VALUE) > TOLERANCE
    ):
        raise RuntimeError("output verification failed: {}".format(values))


def run_graph():
    options = {"ge.exec.deviceId": str(DEVICE_ID)}
    ge_api = GeApi()
    session = None
    ge_initialized = False
    graph_added = False

    try:
        ge_api.ge_initialize(options)
        ge_initialized = True
        session = Session(options)
        session.add_graph(GRAPH_ID, build_graph("host_cpu_constant_folding_python"))
        graph_added = True
        verify_output(session.run_graph(GRAPH_ID, []))
        return 0
    except Exception as exc:
        print("[ConstantFoldingPython] run_graph failed: {}".format(exc))
        traceback.print_exc()
        return 1
    finally:
        if session is not None:
            if graph_added:
                session.remove_graph(GRAPH_ID)
            session = None
        if ge_initialized:
            ge_api.ge_finalize()


if __name__ == "__main__":
    raise SystemExit(run_graph())
