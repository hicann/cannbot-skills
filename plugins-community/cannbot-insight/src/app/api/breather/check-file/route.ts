// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextResponse } from "next/server"
import fs from "node:fs"
import path from "node:path"

export async function GET(request: Request) {
  const url = new URL(request.url)
  const filePath = url.searchParams.get("path")

  if (!filePath) {
    return NextResponse.json({ error: "缺少 path 参数" }, { status: 400 })
  }

  const resolved = path.resolve(filePath)
  const exists = fs.existsSync(resolved)

  return NextResponse.json({ exists, path: resolved })
}
