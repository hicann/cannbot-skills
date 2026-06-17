#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""探测当前 NPU 的 SocVersion 与 NpuArch（dav-*）。

供工作流阶段 0（准备）使用：判定目标芯片，决定是否走
Ascend950 / dav-3510 的 Reg API 路径，并将结果记入 docs/{OP}/STATE.md。

输出（stdout）:
    SocVersion: <完整 SoC 名，含子型号，如 Ascend910B3>
    NpuArch:    dav-<arch>
"""
import configparser
import ctypes
import glob
import logging
import os
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_CANN_ENV_VARS = (
    "ASCEND_TOOLKIT_HOME",
    "ASCEND_HOME_PATH",
    "ASCEND_HOME",
    "ASCEND_CANN_HOME",
)


class ChipInfo(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_char * 32),
        ("name", ctypes.c_char * 32),
        ("version", ctypes.c_char * 32),
    ]


def detect_soc():
    # 1. SocVersion
    hal = ctypes.cdll.LoadLibrary("libascend_hal.so")
    info = ChipInfo()
    if hal.halGetChipInfo(0, ctypes.byref(info)) != 0:
        raise RuntimeError("halGetChipInfo failed")
    soc = info.type.decode().strip() + info.name.decode().strip()

    # 2. CANN home
    cann = None
    for var in _CANN_ENV_VARS:
        path = os.environ.get(var)
        if path and os.path.isdir(path):
            cann = path
            break
    if not cann:
        raise RuntimeError("CANN not found")

    # 3. NpuArch (glob to find ini, avoids hardcoding arch dir)
    ini = glob.glob(f"{cann}/*/data/platform_config/{soc}.ini")
    if not ini:
        raise RuntimeError(f"NpuArch ini not found for {soc}")
    cfg = configparser.ConfigParser()
    cfg.read(ini[0])
    npu = cfg.get("version", "NpuArch", fallback="")
    if not npu:
        raise RuntimeError(f"NpuArch not found in {ini[0]}")

    logger.info("SocVersion: %s", soc)
    logger.info("NpuArch:    dav-%s", npu)


if __name__ == "__main__":
    try:
        detect_soc()
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)
