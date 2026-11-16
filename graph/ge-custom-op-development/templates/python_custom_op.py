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
"""Python 声明式骨架，需提供匹配的 my_kernel.asc 后才能运行。

参考 GE python_compilable_add_custom。此骨架固定 float32、每块 1024 元素，
拒绝空张量和尾块；alpha 仅展示属性签名，未接入 kernel 时拒绝非 1 值。
若需其他契约，同时修改 kernel、参数、校验及测试。
可增加 execute 覆盖动态图；Portable、流和部署细节见 references/python-track.md。
"""

import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from ge.custom_op import (
    AnnotatedKernelLaunchInfo,
    get_compile_platform_info,
    get_declare_launch_args_ctx,
    register_op,
    register_op_impl,
)
from ge.runtime import Tensor, TensorDesc

OP_TYPE = "MyPythonCustomOp"
KERNEL_NAME = "my_kernel"  # 必须核对实际 device 入口
BLOCK_SIZE = 1024
MAX_BLOCK_DIM = 65535


@register_op(op_type=OP_TYPE)
def my_custom_op_infer_meta(
    x: TensorDesc, y: TensorDesc, *, alpha: float = 1.0
) -> TensorDesc:
    if x.shape.origin_shape.dims != y.shape.origin_shape.dims:
        raise ValueError("inputs must have the same shape")
    if x.data_type != y.data_type:
        raise ValueError("inputs must have the same data type")
    if alpha != 1.0:
        raise ValueError("template kernel does not implement alpha != 1")
    return TensorDesc(x.shape, x.data_type)


@dataclass(frozen=True)
class KernelArtifact:
    kernel_name: str
    kernel_bin: bytes
    block_dim: int


def _check_device_elf(data: bytes) -> None:
    # 仅对应本骨架的 Ascend C device ELF 路线，不是任意 binary 格式校验器。
    if (
        len(data) < 64
        or data[:4] != b"\x7fELF"
        or data[4] != 2
        or data[5] not in (1, 2)
    ):
        raise ValueError("expected a complete ELF64 device binary")
    order = "little" if data[5] == 1 else "big"
    if (
        int.from_bytes(data[16:18], order) != 2
        or int.from_bytes(data[18:20], order) != 0x1029
    ):
        raise ValueError("expected executable DaVinci ELF, not a host object")


def _extract_device_binary(host_obj: Path, dev_bin: Path) -> None:
    subprocess.run(
        [
            "llvm-objcopy",
            "-O",
            "binary",
            "--only-section=.aicore_binary",
            str(host_obj),
            str(dev_bin),
        ],
        check=True,
    )
    _check_device_elf(dev_bin.read_bytes())


def _compile_kernel(source: Path, soc: str, arch: str) -> bytes:
    """每次 Compile 独立构建；不复用可能已过期或跨工程冲突的磁盘缓存。"""
    source = source.resolve(strict=True)
    arch = str(arch).strip()
    if arch.startswith("dav-"):
        arch = arch[4:]
    if not arch.isdigit():
        raise ValueError("expected numeric NpuArch, got {!r}".format(arch))
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if not ascend_home:
        raise RuntimeError("ASCEND_HOME_PATH is required")
    root = Path(ascend_home)
    candidates = [
        root / "asc/include",
        root / "aarch64-linux/asc/include",
        root / "x86_64-linux/asc/include",
    ]
    include = next((p for p in candidates if p.is_dir()), None)
    if include is None:
        raise FileNotFoundError("Ascend C headers not found under {}".format(root))
    build_root = os.environ.get("MY_OP_BUILD_DIR")
    if build_root:
        Path(build_root).mkdir(parents=True, exist_ok=True)
    # 编译器参数对齐基线示例，其他版本按工具链核对；soc 用于诊断而非复用缓存。
    with tempfile.TemporaryDirectory(prefix="ge-custom-op-", dir=build_root) as work:
        host_obj, dev_bin = Path(work) / "host.o", Path(work) / "device.o"
        subprocess.run(
            [
                "bisheng",
                "-c",
                str(source),
                "-o",
                str(host_obj),
                "--npu-arch=dav-{}".format(arch),
                "-I{}".format(include),
            ],
            check=True,
        )
        _extract_device_binary(host_obj, dev_bin)
        data = dev_bin.read_bytes()
        if not data:
            raise RuntimeError("empty device binary for {}".format(soc))
        return data


def _tensor_key(x: Tensor, y: Tensor, z: Tensor, alpha: float) -> Tuple:
    def metadata(tensor):
        return (
            tuple(int(d) for d in tensor.origin_shape.dims),
            tuple(int(d) for d in tensor.storage_shape.dims),
            int(tensor.data_type),
            int(tensor.origin_format),
            int(tensor.storage_format),
            int(tensor.expand_dims_type.full_size),
        )

    return metadata(x), metadata(y), metadata(z), float(alpha)


def _validate_tensors(x: Tensor, y: Tensor, z: Tensor, alpha: float) -> int:
    if alpha != 1.0:
        raise ValueError("template kernel does not implement alpha != 1")
    shapes = [tuple(int(d) for d in t.storage_shape.dims) for t in (x, y, z)]
    if shapes[0] != shapes[1] or shapes[0] != shapes[2]:
        raise ValueError("template requires identical storage shapes")
    for t in (x, y, z):
        if int(t.data_type) != 0:
            raise ValueError("template kernel only supports float32")
        if tuple(int(d) for d in t.origin_shape.dims) != shapes[0]:
            raise ValueError("template requires identical origin and storage shapes")
        if (
            int(t.origin_format) != 2
            or int(t.storage_format) != 2
            or t.expand_dims_type.full_size != 0
        ):
            raise ValueError("template kernel only supports ND format")
    if not shapes[0] or any(d <= 0 for d in shapes[0]):
        raise ValueError("template requires concrete positive dimensions")
    n = 1
    for dim in shapes[0]:
        n *= dim
    if n % BLOCK_SIZE or n // BLOCK_SIZE > MAX_BLOCK_DIM:
        raise ValueError(
            "template requires full 1024-element blocks and valid block_dim"
        )
    return n


@register_op_impl(op_type=OP_TYPE)
class MyPythonCustomOp:
    def __init__(self) -> None:
        self._artifacts: Dict[Tuple, KernelArtifact] = {}
        self._target = None
        self._lock = threading.RLock()

    def compile(self, x: Tensor, y: Tensor, z: Tensor, *, alpha: float) -> None:
        n = _validate_tensors(x, y, z, alpha)
        info = get_compile_platform_info()
        arch = str(info.get_platform_resource("version", "NpuArch")).strip()
        soc = str(info.get_soc_version()).strip()
        key = _tensor_key(x, y, z, alpha)
        with self._lock:
            # 声明回调没有独立的平台参数；此骨架限制同实例一个目标，避免跨平台同key覆盖。
            if self._target is not None and self._target != (soc, arch):
                raise ValueError(
                    "use a separate process for a different target platform"
                )
            self._target = (soc, arch)
            # 不因已有相同元信息跳过编译，源码/依赖变更会重新构建。
            binary = _compile_kernel(Path(__file__).parent / "my_kernel.asc", soc, arch)
            self._artifacts[key] = KernelArtifact(KERNEL_NAME, binary, n // BLOCK_SIZE)

    def declare_launch_args(
        self, x: Tensor, y: Tensor, z: Tensor, *, alpha: float
    ) -> None:
        _validate_tensors(x, y, z, alpha)
        key = _tensor_key(x, y, z, alpha)
        with self._lock:
            artifact = self._artifacts.get(key)
        if artifact is None:
            raise RuntimeError(
                "missing compiled artifact for tensor metadata and attrs"
            )
        ctx = get_declare_launch_args_ctx()
        args = ctx.create_kernel_args()
        args.append_input(0, x)
        args.append_input(1, y)
        args.append_output(0, z)
        ctx.add_launch(
            AnnotatedKernelLaunchInfo(
                kernel_name=artifact.kernel_name,
                kernel_bin=artifact.kernel_bin,
                block_dim=artifact.block_dim,
                stream_id=ctx.get_stream_id(),
            ),
            args,
        )


class AddCustomPortable:
    """可选的自有额外状态示例；合并进实际实现类并定义自己的格式。"""

    def __init__(self) -> None:
        self._payload = None

    def serialize(self) -> bytes:
        if not isinstance(self._payload, bytes) or not self._payload:
            raise RuntimeError("portable state has not been prepared")
        return self._payload

    def deserialize(self, data: bytes) -> None:
        if type(data) is not bytes or not data:
            raise ValueError("portable state must be nonempty bytes")
        self._payload = data
