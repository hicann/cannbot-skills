// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// npm install -g 后置钩子：预生成 Prisma client + 预构建 export-view bundle。
// 失败只 warn 不阻断安装 —— bin 启动器有运行时兜底（缺则现场补）。
// 不依赖 PATH（用 require.resolve 解析 bin 入口 + spawnSync node 调），全局装场景 cwd=PKG_DIR 但 PATH 不含 node_modules/.bin。

import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const here = path.dirname(fileURLToPath(import.meta.url))
const PKG_DIR = path.resolve(here, '..')
const require = createRequire(PKG_DIR + '/')

function warn(msg) { console.warn(`[cannbot-insight postinstall] ${msg}`) }

// 1) Prisma client 生成（运行时 @prisma/client 需要，否则 query 报错）
try {
  const prismaBin = require.resolve('prisma/build/index.js', { paths: [PKG_DIR] })
  const r = spawnSync(process.execPath, [prismaBin, 'generate'], { cwd: PKG_DIR, stdio: 'inherit' })
  if (r.status !== 0) warn(`prisma generate 失败（启动时兜底重试）: exit ${r.status}`)
} catch (e) {
  warn(`prisma generate 失败（启动时兜底重试）: ${e instanceof Error ? e.message : e}`)
}

// 2) export-view bundle 预构建（HTML 导出依赖；esbuild 转译 src/export-view/entry.tsx → public/export-view.js）
//    包发布时已预构建随包（1.9MB public/export-view.js），此处仅兜底缺失场景。
const exportView = path.join(PKG_DIR, 'public', 'export-view.js')
if (!existsSync(exportView)) {
  try {
    const r = spawnSync(process.execPath, [path.join('scripts', 'build-export-view.mjs')], { cwd: PKG_DIR, stdio: 'inherit' })
    if (r.status !== 0) warn(`export-view bundle 构建失败（HTML 导出功能不可用，其余正常）: exit ${r.status}`)
  } catch (e) {
    warn(`export-view bundle 构建失败（HTML 导出功能不可用，其余正常）: ${e instanceof Error ? e.message : e}`)
  }
}
