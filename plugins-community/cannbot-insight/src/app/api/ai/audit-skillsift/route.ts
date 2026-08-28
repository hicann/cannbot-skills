// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextResponse } from "next/server"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"
import { prisma } from "@/lib/db"
import {
  recoverSkillBody,
  recoverWorkflowDeclaration,
  buildAuditArgs,
  buildStructuredRecords,
  buildTranscriptArgs,
} from "@/lib/sift-audit"
import { llmExtractWorkflowDeclaration } from "@/lib/llm-workflow-extract"
import { makeStreamingAuditResponse } from "@/lib/sift-runner"

/**
 * 对当前 session 的某个 skill 跑 sift 对账（审"真实执行 vs SKILL.md 声明"）。
 *
 * 数据流：taskId → 解出 session.id + sourcePath(opencode .db) → 恢复该 skill 正文
 * （invoke ToolCall.resultJson 的 <skill_content>）→ 物化临时 SKILL.md →
 * `sift audit <tmp> --db <sourcePath> --session <taskId> -o <out>` → 读 audit-report.json。
 *
 * sift 自带 LLM（走它自己的 claude CLI，注入 529 绕过 env），所以这里不透传 provider。
 * 输出读文件不读 stdout（--format json 的 stdout 被 rich 折行 + 尾行污染）。
 */
export async function POST(req: Request) {
  const { taskId, skillName, kind, framework, extract } = await req.json()
  if (!taskId) {
    return NextResponse.json({ error: "Missing taskId" }, { status: 400 })
  }
  // kind: "skill"(默认)| "root"(审顶层主 agent 编排 vs workflow skill 编排规程)。
  //        "llm-root" = root + 用 LLM（claude CLI）从 session 行为提取编排规程。
  // root 的声明是 workflow skill 的编排规程（SKILL.md body / task-prompts.md），
  // 从主 agent Skill invoke 定位 skillName → recoverWorkflowDeclaration 恢复。
  // 按 sift 设计意图（slice.py："root 声明常是 workflow 级 SKILL.md"）。
  // extract: "llm" → 同 llm-root（兼容直接传 extract 参数）。
  const auditKind: "skill" | "root" = kind === "root" || kind === "llm-root" ? "root" : "skill"
  const useLlmExtract = kind === "llm-root" || extract === "llm"
  // skill 路需 skillName（恢复该 skill 正文）；root 路不用 skillName（声明从 workflow skill 取）。
  if (auditKind !== "root" && !skillName) {
    return NextResponse.json({ error: "Missing skillName" }, { status: 400 })
  }

  const session = await prisma.session.findFirst({
    where: { taskId, ...(framework ? { framework } : {}) },
  })
  if (!session?.sourcePath) {
    return NextResponse.json(
      { error: "该 session 无 sourcePath（非从 .db 导入？sift --db 需要 .db）" },
      { status: 404 },
    )
  }

  const decl =
    auditKind === "root"
      ? useLlmExtract
        ? await llmExtractWorkflowDeclaration(session.id, prisma)
        : await recoverWorkflowDeclaration(session.id, prisma)
      : null
  const body =
    decl?.content ?? (auditKind === "root" ? null : await recoverSkillBody(session.id, skillName, prisma))
  if (!body) {
    return NextResponse.json(
      auditKind === "root"
        ? { error: useLlmExtract
          ? "LLM 提取编排规程失败（claude CLI 超时/无输出/无 dispatch 前 turns）"
          : "此 session 无可对账的 workflow 编排规程（无主 agent Skill invoke / 无 skill resources / 无 SKILL.md body）" }
        : { error: `在此 session 恢复不到 skill "${skillName}" 的正文（无 invoke 事件 / resultJson）` },
      { status: 404 },
    )
  }

  const skillTmp = fs.mkdtempSync(path.join(os.tmpdir(), "sift-audit-"))
  try {
    const safeName = auditKind === "root"
      ? (useLlmExtract ? "llm-workflow" : "main-agent-workflow")
      : String(skillName).replace(/[^a-zA-Z0-9._-]/g, "_")
    const skillPath = path.join(skillTmp, safeName)
    fs.mkdirSync(skillPath, { recursive: true })
    // 恢复的正文无 YAML frontmatter，而 sift 的 parse_skill_md 要 `---\nname:` 才能
    // 得名（--kind skill 切段用它匹配 transcript 里 input.skill；--kind root 不按 name 切、
    // name 无意义，但 parse_skill_md 仍需 frontmatter 才解析指令）。注入：root 用固定名，
    // skill 用真实 skillName（与 transcript input.skill 对得上）。
    const declName = auditKind === "root" ? "main-agent-workflow" : skillName
    fs.writeFileSync(
      path.join(skillPath, "SKILL.md"),
      `---\nname: ${declName}\ndescription: recovered from session\n---\n${body}\n`,
    )

    const outputDir = path.join(skillTmp, "out")
    fs.mkdirSync(outputDir, { recursive: true })

    // framework 决定喂法:opencode 原生 → --db(直读 sourcePath 的 sessions.db);
    // 其余(cannbot-insight / claude-code)→ 结构化 records JSON(--transcript),把 insight 已
    // 归一化的 Turn/ToolCall 直喂 sift,绕开 opencode-native db 的格式限制。
    const useTranscript = session.framework !== "opencode"
    let args: string[]
    if (useTranscript) {
      const transcriptPath = path.join(skillTmp, "session.json")
      fs.writeFileSync(
        transcriptPath,
        JSON.stringify(await buildStructuredRecords(session.id, prisma)),
      )
      args = buildTranscriptArgs({ skillPath, transcriptPath, outputDir, kind: auditKind })
    } else {
      args = buildAuditArgs({
        skillPath,
        dbPath: session.sourcePath,
        sessionId: taskId,
        outputDir,
        kind: auditKind,
      })
    }

    // 流式 NDJSON：spawn sift，stdout 解析进度回传 progress 事件，结束回传 result/error。
    // tmp 清理放进流的 finally（等 sift 跑完），不能在路由 return 时删。
    return makeStreamingAuditResponse(args, outputDir, () =>
      fs.rmSync(skillTmp, { recursive: true, force: true }),
    )
  } catch (e) {
    fs.rmSync(skillTmp, { recursive: true, force: true })
    return NextResponse.json(
      { error: `准备 skill 对账失败：${e instanceof Error ? e.message : String(e)}` },
      { status: 500 },
    )
  }
}
