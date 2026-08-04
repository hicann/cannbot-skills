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
import { resolveWorkflowSkillName } from "@/lib/skill-md-scan"

/**
 * skill-md-scan：扫盘 SKILL.md → 按 turn0 body 前缀匹配 → frontmatter name（真 identifier）。
 * 纯 FS，tmp 目录搭 fixture。
 */

let root = ""

function writeSkill(fileRel: string, name: string, body: string, withFrontmatter = true): void {
  const p = path.join(root, fileRel)
  fs.mkdirSync(path.dirname(p), { recursive: true })
  const content = withFrontmatter ? `---\nname: ${name}\ndescription: x\n---\n${body}` : body
  fs.writeFileSync(p, content)
}

beforeEach(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), "skill-md-scan-"))
})
afterEach(() => {
  fs.rmSync(root, { recursive: true, force: true })
})

describe("resolveWorkflowSkillName", () => {
  it("turn0 body 前缀匹配 disk SKILL.md → frontmatter name", () => {
    const body = "# 基于 ACLNN 的算子开发工作流\n\n你是纯编排者（Orchestrator），" + "x".repeat(500)
    writeSkill("plugins-official/ops-registry-invoke-glacier/skills/ops-registry-invoke-glacier/SKILL.md", "ops-registry-invoke-glacier", body)
    // turn0 = skill body + 末尾用户查询
    const turn0 = body + "\n\neuclidean_norm,参考/.../需求文档.md"
    expect(resolveWorkflowSkillName(turn0, root)).toBe("ops-registry-invoke-glacier")
  })

  it("无 frontmatter 的 SKILL.md → 用父目录名当 name", () => {
    const body = "# 工作流\n\n" + "x".repeat(500)
    writeSkill("ops/my-workflow/SKILL.md", "ignored", body, false)
    expect(resolveWorkflowSkillName(body + "\n\n用户查询", root)).toBe("my-workflow")
  })

  it("多个 SKILL.md 只匹配 body 前缀对得上的那个", () => {
    const a = "# 工作流A\n\n" + "a".repeat(500)
    const b = "# 工作流B\n\n" + "b".repeat(500)
    writeSkill("ops/a/SKILL.md", "skill-a", a)
    writeSkill("ops/b/SKILL.md", "skill-b", b)
    expect(resolveWorkflowSkillName(b + "\n\nq", root)).toBe("skill-b")
  })

  it("无匹配 → null", () => {
    writeSkill("ops/x/SKILL.md", "skill-x", "# 完全不同的内容\n\n" + "x".repeat(500))
    expect(resolveWorkflowSkillName("# 别的 workflow\n\n" + "y".repeat(500), root)).toBeNull()
  })

  it("scanRoot=null → null", () => {
    expect(resolveWorkflowSkillName("# whatever\n\n" + "x".repeat(500), null)).toBeNull()
  })

  it("turn0 过短（<50）→ null", () => {
    writeSkill("ops/x/SKILL.md", "skill-x", "# x\n\n" + "x".repeat(500))
    expect(resolveWorkflowSkillName("短", root)).toBeNull()
  })

  it("跳过 node_modules/.git/tests 等目录", () => {
    const body = "# 跳过\n\n" + "x".repeat(500)
    writeSkill("node_modules/pkg/SKILL.md", "should-not-match", body)
    writeSkill(".git/x/SKILL.md", "should-not-match2", body)
    writeSkill("tests/x/SKILL.md", "should-not-match3", body)
    expect(resolveWorkflowSkillName(body + "\n\nq", root)).toBeNull()
  })
})
