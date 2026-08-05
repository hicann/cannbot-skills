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
# deploy_to_a5.sh — Backwards-compatibility wrapper. V3.4 routes deploys
# through deploy_to_npu.sh (multi-target). Pre-V3.4 callers (aog-kernel-worker
# briefs, ad-hoc scripts) still invoke this name; we honor it by forcing
# TARGET=a5 and delegating.
#
# The original A5-only deploy logic is preserved at deploy_to_a5.sh.legacy
# for the (rare) case where the multi-target plumbing has a regression.
exec env TARGET=a5 bash "$(dirname "$0")/deploy_to_npu.sh" "$@"
