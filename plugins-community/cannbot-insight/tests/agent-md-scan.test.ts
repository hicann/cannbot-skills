// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, beforeEach, afterEach } from "vitest"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"
import { scanAgentDirs, resolveAgentMd } from "@/lib/agent-md-scan"

/**
 * agent-md-scan：扫描 agents/ 目录 + 多插件覆盖率消歧。纯 FS，用 tmp 目录搭 fixture。
 */

let root = ""

function writeAgents(dir: string, names: string[]): void {
  fs.mkdirSync(dir, { recursive: true })
  for (const n of names) fs.writeFileSync(path.join(dir, `${n}.md`), `---\nname: ${n}\n---\nbody\n`)
}

beforeEach(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), "agent-scan-"))
})
afterEach(() => {
  fs.rmSync(root, { recursive: true, force: true })
})

describe("scanAgentDirs", () => {
  it("扫 plugins-official/<plugin>/agents/ + 顶层 <domain>/agents/ + .opencode/agents/", () => {
    writeAgents(path.join(root, "plugins-official", "plugin-a", "agents"), ["developer", "verifier"])
    writeAgents(path.join(root, "plugins-official", "plugin-b", "agents"), ["developer", "designer"])
    writeAgents(path.join(root, "ops", "agents"), ["developer", "kernel-developer"])
    writeAgents(path.join(root, ".opencode", "agents"), ["st-verifier"])
    // tests/agents 应被排除（fixture stub）
    writeAgents(path.join(root, "tests", "agents"), ["fake-dev"])

    const dirs = scanAgentDirs(root)
    const allNames = new Set<string>()
    for (const d of dirs) for (const n of d.names) allNames.add(n)
    expect(allNames.has("developer")).toBe(true)
    expect(allNames.has("verifier")).toBe(true)
    expect(allNames.has("designer")).toBe(true)
    expect(allNames.has("kernel-developer")).toBe(true)
    expect(allNames.has("st-verifier")).toBe(true)
    expect(allNames.has("fake-dev")).toBe(false) // tests 被排除
    expect(dirs.length).toBe(4) // plugin-a, plugin-b, ops, .opencode
  })

  it("根下无 agents 目录 → 空数组", () => {
    fs.mkdirSync(path.join(root, "plugins-official", "x"), { recursive: true }) // 无 agents 子目录
    expect(scanAgentDirs(root)).toEqual([])
  })

  it("空 agents 目录不入结果", () => {
    fs.mkdirSync(path.join(root, "plugins-official", "p", "agents"), { recursive: true })
    expect(scanAgentDirs(root)).toEqual([])
  })
})

describe("resolveAgentMd", () => {
  beforeEach(() => {
    writeAgents(path.join(root, "plugins-official", "plugin-a", "agents"), ["developer", "verifier", "x"])
    writeAgents(path.join(root, "plugins-official", "plugin-b", "agents"), ["developer", "verifier", "y"])
    writeAgents(path.join(root, "ops", "agents"), ["developer", "z"])
  })

  it("0 个候选 → null", () => {
    const dirs = scanAgentDirs(root)
    expect(resolveAgentMd("ghost", dirs, new Set())).toBeNull()
  })

  it("1 个候选 → 直接用（x 只在 plugin-a）", () => {
    const dirs = scanAgentDirs(root)
    const r = resolveAgentMd("x", dirs, new Set())
    expect(r).not.toBeNull()
    expect(r!.mdPath.endsWith("plugin-a/agents/x.md")).toBe(true)
  })

  it(">1 个候选 → 按覆盖率消歧：sessionAgentNames 含 x → 选 plugin-a（x 只在 plugin-a）", () => {
    const dirs = scanAgentDirs(root)
    const r = resolveAgentMd("developer", dirs, new Set(["x"]))
    expect(r!.mdPath.endsWith("plugin-a/agents/developer.md")).toBe(true)
  })

  it(">1 个候选 → 覆盖率消歧：sessionAgentNames 含 y → 选 plugin-b", () => {
    const dirs = scanAgentDirs(root)
    const r = resolveAgentMd("developer", dirs, new Set(["y"]))
    expect(r!.mdPath.endsWith("plugin-b/agents/developer.md")).toBe(true)
  })

  it("sessionAgentNames 为空 → 退化字母序（不崩）", () => {
    const dirs = scanAgentDirs(root)
    const r = resolveAgentMd("developer", dirs, new Set())
    expect(r).not.toBeNull()
    expect(r!.mdPath.endsWith("developer.md")).toBe(true)
  })

  it("返回的 mdPath 真实存在于盘", () => {
    const dirs = scanAgentDirs(root)
    const r = resolveAgentMd("verifier", dirs, new Set(["x"]))
    expect(fs.existsSync(r!.mdPath)).toBe(true)
  })
})
