// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

/**
 * Server-only（fs/path）：主 agent workflow skill 的真名解析。
 *
 * 主 agent 的 workflow 声明在 session 首条 user turn（注入系统提示，剥了 frontmatter），
 * 只有 body 没 name。但盘上对应 SKILL.md 有 frontmatter `name:`（真 identifier）。
 * turn0 的 skill 部分与盘上 SKILL.md body 逐字一致（实测验证），故扫盘按 body 前缀匹配
 * 可稳准对到该 skill → 读 frontmatter name，作 Skills/Audit 行的真名显示（替代合成名）。
 */
import fs from "node:fs"
import path from "node:path"
import { resolveScanRoot } from "@/lib/agent-md-scan"

const SKIP_DIRS = new Set(["node_modules", ".next", ".git", "dist", "tests", ".turbo", "coverage"])
const MAX_DEPTH = 8

interface SkillMd {
  name: string
  body: string
}

/** 递归收集 scanRoot 下所有 SKILL.md（跳过 node_modules/.git/.next/tests 等）。 */
function collectSkillMds(scanRoot: string): SkillMd[] {
  const out: SkillMd[] = []
  const stack: Array<{ dir: string; depth: number }> = [{ dir: scanRoot, depth: 0 }]
  while (stack.length) {
    const { dir, depth } = stack.pop()!
    let entries: fs.Dirent[]
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true })
    } catch {
      continue
    }
    for (const e of entries) {
      if (e.isDirectory()) {
        if (depth >= MAX_DEPTH || SKIP_DIRS.has(e.name) || e.name.startsWith(".")) continue
        stack.push({ dir: path.join(dir, e.name), depth: depth + 1 })
      } else if (e.isFile() && e.name.toLowerCase() === "skill.md") {
        const parsed = parseSkillMd(path.join(dir, e.name))
        if (parsed) out.push(parsed)
      }
    }
  }
  return out
}

/** 解析 SKILL.md：剥 frontmatter 取 name（无 frontmatter 用父目录名）+ body。 */
function parseSkillMd(filePath: string): SkillMd | null {
  let raw: string
  try {
    raw = fs.readFileSync(filePath, "utf8")
  } catch {
    return null
  }
  const parent = path.basename(path.dirname(filePath))
  // frontmatter：文件以 --- 开头，到下一个 --- 结束
  if (raw.startsWith("---")) {
    const end = raw.indexOf("\n---", 3)
    if (end >= 0) {
      const fm = raw.slice(3, end)
      // 剥 frontmatter 后的 body：--- 后是 "\n\n# body"，strip 全部前导换行（与 turn0 注入版
      // body 对齐——turn0 以 "# body" 开头无前导换行），否则 startsWith 匹配失败。
      const body = raw.slice(end + 4).replace(/^[\r\n]+/, "")
      const m = fm.match(/^name:\s*(.+?)\s*$/m)
      return { name: (m?.[1] ?? parent).trim(), body }
    }
  }
  return { name: parent, body: raw.replace(/^[\r\n]+/, "") }
}

/**
 * 按 turn0 body 前缀匹配盘上 SKILL.md，返回该 workflow skill 的真名（frontmatter name）。
 * turn0 = skill body + 末尾用户查询，故 turn0 以盘上 body 为前缀 → startsWith(body[:500]) 命中。
 * 无匹配 / 无 scan root → null。
 */
export function resolveWorkflowSkillName(turn0Body: string, scanRoot: string | null): string | null {
  if (!scanRoot) return null
  const prefix = turn0Body.slice(0, 500)
  if (prefix.length < 50) return null
  for (const s of collectSkillMds(scanRoot)) {
    if (s.body.length < 200) continue
    if (turn0Body.startsWith(s.body.slice(0, 500))) return s.name
  }
  return null
}

/** 便捷封装：自动取 scanRoot + 解析。供 route 调用。 */
export function resolveWorkflowSkillNameAuto(turn0Body: string): string | null {
  return resolveWorkflowSkillName(turn0Body, resolveScanRoot())
}
