#!/usr/bin/env bash
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# 探测本机真实 ATC soc_version。宿主中立：只用环境变量定位 CANN，
# 不访问任何默认安装路径（对齐 knowledge-base「禁硬编码默认路径」）。
# 输出含 `ATC argument: --soc_version=<完整值>` 一行，供离线编译命令提取。
set -euo pipefail

find_ascend_home() {
  local env_candidates=() path

  # 优先级：ASCEND_HOME_PATH > ASCEND_TOOLKIT_HOME（来自环境）
  [[ -n "${ASCEND_HOME_PATH:-}" ]]     && env_candidates+=("${ASCEND_HOME_PATH}")
  [[ -n "${ASCEND_TOOLKIT_HOME:-}" ]]  && env_candidates+=("${ASCEND_TOOLKIT_HOME}")

  for path in "${env_candidates[@]}"; do
    if [[ -f "${path}/tools/msaicerr/ms_interface/dsmi_interface.py" ]]; then
      printf '%s\n' "${path}"
      return 0
    fi
  done

  return 1
}

ASCEND_HOME="$(find_ascend_home)" || {
  echo "环境缺失：未找到 CANN DSMI helper。请设置 ASCEND_HOME_PATH 指向 CANN 安装路径后重试；" >&2
  echo "在无 CANN 环境下，调用方应如实标注 soc_version「未确认」并跳过离线编译验证（门禁 G3）。" >&2
  exit 1
}

if [[ -f "${ASCEND_HOME}/set_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${ASCEND_HOME}/set_env.sh" >/dev/null 2>&1 || true
fi

export PYTHONPATH="${ASCEND_HOME}/tools/msaicerr/ms_interface:${PYTHONPATH:-}"
command -v python3 >/dev/null 2>&1 || {
  echo "环境缺失：python3 不在 PATH，无法调用 CANN DSMI helper。" >&2
  exit 1
}

printf 'CANN_ROOT=%s\n' "$ASCEND_HOME"

python3 - <<'PY'
from dsmi_interface import DSMIInterface

dsmi = DSMIInterface()
device_count = dsmi.get_device_count()
print(f"device_count={device_count}")

first_soc = None
for device_id in range(device_count):
    chip = dsmi.get_chip_info(device_id)
    if chip is None:
        print(f"device {device_id}: <failed to query>")
        continue
    soc = chip.get_complete_platform()
    ver = chip.get_ver()
    if first_soc is None:
        first_soc = soc
    print(f"device {device_id}: soc_version={soc} chip_ver={ver}")

if first_soc:
    # 完整值（如 Ascend910_9362）；ATC 需要完整值，不要用 short family（如 Ascend910_93）
    print(f"ATC argument: --soc_version={first_soc}")
else:
    raise SystemExit("No Ascend device soc_version detected.")
PY
