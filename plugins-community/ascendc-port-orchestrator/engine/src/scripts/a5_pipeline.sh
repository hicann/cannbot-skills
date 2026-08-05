#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# a5_pipeline.sh — Backwards-compatibility wrapper around npu_pipeline.sh (V3.4)
# Pre-V3.4 callers expect this script and an A5-only execution path.
# We now delegate to npu_pipeline.sh with TARGET forced to a5.
#
# New code should invoke `bash src/scripts/npu_pipeline.sh <cmd>` directly so
# `--target=a3` / TARGET=a3 overrides work without going through this shim.
exec env TARGET=a5 bash "$(dirname "$0")/npu_pipeline.sh" "$@"
