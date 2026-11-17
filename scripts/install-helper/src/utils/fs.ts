// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { writeFileSync, renameSync, existsSync, unlinkSync } from "fs";
import { dirname } from "path";

export function atomicWriteFileSync(filePath: string, data: string, encoding: BufferEncoding = "utf-8"): void {
  const dir = dirname(filePath);
  if (!existsSync(dir)) {
    throw new Error(`Directory does not exist: ${dir}`);
  }

  const tmpPath = `${filePath}.tmp`;
  writeFileSync(tmpPath, data, encoding);
  try {
    renameSync(tmpPath, filePath);
  } catch (e) {
    try { unlinkSync(tmpPath); } catch {}
    throw e;
  }
}
