// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//
// Server-only（fs/path）：被 skill-eval-audit.ts 等客户端可达文件不能 import fs，故本扫描逻辑独立成文件，
// 仅由 audit-agenteval 服务端 route + 测试 import。

import fs from "node:fs"
import path from "node:path"

/**
 * agent 对账的 .md 声明恢复：agent .md 不在 session（dispatch args 只带任务 prompt，
 * opencode 运行时加载 agent .md 当 system prompt 不持久化），故从本地文件系统扫。
 *
 * 多插件歧义：skills-dev 里 plugins-official/<plugin>/agents/ 多个插件都有 developer.md 等。
 * 一个 session 用一个插件（运行时配的那个），故"覆盖本 session 大多数被 dispatch agent 名
 * 的那个 agents/ 目录"就是该 session 用的插件 → 按覆盖率消歧认对插件，不靠 first-match 拿错。
 */

export interface AgentDir {
  dir: string
  names: Set<string>
}

/**
 * 解析扫描根：AGENTS_SCAN_ROOT env 优先；未配则自动探测 skills-dev 根
 * （cannbot-insight 位于 <skills-dev>/tests/cannbot-insight，process.cwd()../.. 即根，
 *  若该根下有 plugins-official/ 则用）。探测不到 → null（route 据 503 关 agent 对账）。
 */
export function resolveScanRoot(): string | null {
  const env = process.env.AGENTS_SCAN_ROOT
  if (env) return env
  const root = path.resolve(process.cwd(), "..", "..")
  if (fs.existsSync(path.join(root, "plugins-official"))) return root
  return null
}

/**
 * 扫描 scanRoot 下所有 agents/ 目录及其 .md 文件名集合。
 * 覆盖：plugins-official/<plugin>/agents/、顶层 <domain>/agents/（排除 tests/node_modules/plugins-official/隐藏）、
 * .opencode/agents/。同一目录去重；空目录不入结果。
 */
export function scanAgentDirs(scanRoot: string): AgentDir[] {
  const out: AgentDir[] = []
  const seen = new Set<string>()

  function add(dir: string): void {
    const real = path.resolve(dir)
    if (seen.has(real)) return
    let files: string[]
    try {
      files = fs.readdirSync(real)
    } catch {
      return
    }
    const names = new Set<string>()
    for (const f of files) if (f.endsWith(".md")) names.add(f.slice(0, -3))
    if (names.size > 0) {
      seen.add(real)
      out.push({ dir: real, names })
    }
  }

  const po = path.join(scanRoot, "plugins-official")
  try {
    for (const e of fs.readdirSync(po, { withFileTypes: true })) {
      if (e.isDirectory()) add(path.join(po, e.name, "agents"))
    }
  } catch {
    /* plugins-official 不存在则跳过 */
  }

  try {
    for (const e of fs.readdirSync(scanRoot, { withFileTypes: true })) {
      if (!e.isDirectory()) continue
      if (e.name === "tests" || e.name === "node_modules" || e.name === "plugins-official" || e.name.startsWith(".")) continue
      add(path.join(scanRoot, e.name, "agents"))
    }
  } catch {
    /* 根不可读则跳过 */
  }

  add(path.join(scanRoot, ".opencode", "agents"))
  return out
}

/**
 * 为请求的 agentName 选定 .md 路径：
 * - 0 个候选 → null（route 404）
 * - 1 个 → 直接用
 * - >1 个 → 按 sessionAgentNames 覆盖率消歧（候选目录覆盖本 session 被派发 agent 名最多的那个
 *   = 本 session 用的插件）；并列取目录名字母序
 *
 * sessionAgentNames 为空（查不到 session 派发记录）时退化为字母序第一个，仍能给出一个 .md。
 */
export function resolveAgentMd(
  agentName: string,
  dirs: AgentDir[],
  sessionAgentNames: Set<string>,
): { mdPath: string; dir: string } | null {
  const candidates = dirs.filter((d) => d.names.has(agentName))
  if (candidates.length === 0) return null
  if (candidates.length === 1) {
    return { mdPath: path.join(candidates[0].dir, `${agentName}.md`), dir: candidates[0].dir }
  }
  let best: AgentDir | null = null
  let bestCov = -1
  for (const c of candidates) {
    let cov = 0
    for (const n of sessionAgentNames) if (c.names.has(n)) cov++
    if (best === null || cov > bestCov || (cov === bestCov && c.dir < best.dir)) {
      best = c
      bestCov = cov
    }
  }
  return best ? { mdPath: path.join(best.dir, `${agentName}.md`), dir: best.dir } : null
}
