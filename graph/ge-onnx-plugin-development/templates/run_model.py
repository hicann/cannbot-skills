#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""OM 模型执行与结果比对模板（自包含 ACL runner）。

填充指引（使用前删除本段）：
- TODO(填充) 处替换为真实输入与 PyTorch 参考输出；
- dtype 与模型输出一致（默认 float32）。
"""

import argparse
import logging

import numpy as np

import acl

logging.basicConfig(level=logging.INFO, format="%(message)s")

ACL_SUCCESS = 0
ACL_MEM_MALLOC_NORMAL_ONLY = 2
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
DEVICE_ID = 0


def check_ret(name, ret):
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"{name} failed ret={ret}")


def _prepare_dataset(model_desc, io_type):
    if io_type == "input":
        io_num = acl.mdl.get_num_inputs(model_desc)
        get_size_by_index = acl.mdl.get_input_size_by_index
    else:
        io_num = acl.mdl.get_num_outputs(model_desc)
        get_size_by_index = acl.mdl.get_output_size_by_index
    dataset = acl.mdl.create_dataset()
    items = []
    for index in range(io_num):
        buffer_size = get_size_by_index(model_desc, index)
        buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY)
        check_ret(f"acl.rt.malloc({io_type})", ret)
        data_buffer = acl.create_data_buffer(buffer, buffer_size)
        _, ret = acl.mdl.add_dataset_buffer(dataset, data_buffer)
        check_ret(f"acl.mdl.add_dataset_buffer({io_type})", ret)
        items.append({"buffer": buffer, "size": buffer_size})
    return dataset, items


def _copy_inputs(input_data, inputs):
    for index, input_array in enumerate(inputs):
        bytes_data = input_array.tobytes()
        bytes_ptr = acl.util.bytes_to_ptr(bytes_data)
        check_ret(
            f"acl.rt.memcpy(input_{index})",
            acl.rt.memcpy(
                input_data[index]["buffer"],
                input_data[index]["size"],
                bytes_ptr,
                len(bytes_data),
                ACL_MEMCPY_HOST_TO_DEVICE,
            ),
        )


def _collect_outputs(model_desc, output_data, dtype):
    outputs = []
    for index, item in enumerate(output_data):
        host_buffer, ret = acl.rt.malloc_host(item["size"])
        check_ret(f"acl.rt.malloc_host(output_{index})", ret)
        try:
            check_ret(
                f"acl.rt.memcpy(output_{index})",
                acl.rt.memcpy(
                    host_buffer,
                    item["size"],
                    item["buffer"],
                    item["size"],
                    ACL_MEMCPY_DEVICE_TO_HOST,
                ),
            )
            output_bytes = acl.util.ptr_to_bytes(host_buffer, item["size"])
        finally:
            check_ret(
                f"acl.rt.free_host(output_{index})", acl.rt.free_host(host_buffer)
            )
        dims, ret = acl.mdl.get_output_dims(model_desc, index)
        check_ret(f"acl.mdl.get_output_dims(output_{index})", ret)
        shape = tuple(dims["dims"][: dims["dimCount"]])
        outputs.append(np.frombuffer(output_bytes, dtype=dtype).reshape(shape))
    return outputs


def _release_dataset(dataset):
    if not dataset:
        return
    num = acl.mdl.get_dataset_num_buffers(dataset)
    for index in range(num):
        data_buf = acl.mdl.get_dataset_buffer(dataset, index)
        if data_buf:
            data = acl.get_data_buffer_addr(data_buf)
            check_ret("acl.rt.free", acl.rt.free(data))
            check_ret("acl.destroy_data_buffer", acl.destroy_data_buffer(data_buf))
    check_ret("acl.mdl.destroy_dataset", acl.mdl.destroy_dataset(dataset))


def run_model(model_path, inputs, dtype=np.float32):
    """加载 OM 并执行，返回输出列表（numpy 数组，按模型输出顺序）。"""
    model_id = None
    model_desc = None
    input_dataset = None
    output_dataset = None
    check_ret("acl.init", acl.init())
    try:
        check_ret("acl.rt.set_device", acl.rt.set_device(DEVICE_ID))
        model_id, ret = acl.mdl.load_from_file(model_path)
        check_ret("acl.mdl.load_from_file", ret)
        model_desc = acl.mdl.create_desc()
        check_ret("acl.mdl.get_desc", acl.mdl.get_desc(model_desc, model_id))
        input_dataset, input_data = _prepare_dataset(model_desc, "input")
        output_dataset, output_data = _prepare_dataset(model_desc, "output")
        _copy_inputs(input_data, inputs)
        check_ret(
            "acl.mdl.execute", acl.mdl.execute(model_id, input_dataset, output_dataset)
        )
        return _collect_outputs(model_desc, output_data, dtype)
    finally:
        _release_dataset(input_dataset)
        _release_dataset(output_dataset)
        if model_desc is not None:
            check_ret("acl.mdl.destroy_desc", acl.mdl.destroy_desc(model_desc))
        if model_id is not None:
            check_ret("acl.mdl.unload", acl.mdl.unload(model_id))
        acl.rt.reset_device(DEVICE_ID)
        acl.finalize()


def main():
    parser = argparse.ArgumentParser(description="Run OM model and compare outputs.")
    parser.add_argument("--model", required=True, help="Generated OM model path.")
    args = parser.parse_args()

    input_tensor = np.array([[-1.0, 0.5, 1.5], [2.0, -2.0, 3.0]], dtype=np.float32)
    # TODO(填充): PyTorch 参考输出（与导出侧 forward 计算一致），按模型输出顺序列出
    expected = [input_tensor * 2.0]

    outputs = run_model(args.model, [input_tensor])
    # 容差说明：NPU 算子多为 fp16 内部精度（输出仍为 fp32），默认 1e-3（与官方样例一致）；
    # 纯整数/位运算类算子可收紧到 1e-5，按算子类型调整
    for index, (actual, expect) in enumerate(zip(outputs, expected)):
        logging.info("Output[%d]:\n%s", index, actual)
        np.testing.assert_allclose(actual, expect, rtol=1e-3, atol=1e-3)
    logging.info("[Success] OM executed; output matches PyTorch reference.")


if __name__ == "__main__":
    main()
