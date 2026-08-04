// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from "vitest"
import { alignTurns, computeAlignStats, type TurnData } from "@/lib/compare/turn-align"

function makeTurn(overrides: Partial<TurnData> & { turnIndex: number; role: string }): TurnData {
  return {
    turnId: `t-${overrides.turnIndex}`,
    turnIndex: overrides.turnIndex,
    role: overrides.role,
    content: overrides.content ?? null,
    contentSummary: overrides.contentSummary ?? null,
    totalTokens: overrides.totalTokens ?? 0,
    inputTokens: overrides.inputTokens ?? 0,
    outputTokens: overrides.outputTokens ?? 0,
    reasoningTokens: overrides.reasoningTokens ?? 0,
    latencyMs: overrides.latencyMs ?? 0,
    model: overrides.model ?? null,
    toolCalls: overrides.toolCalls ?? [],
    skillEvents: overrides.skillEvents ?? [],
  }
}

function diagnoseAlignment(pairs: ReturnType<typeof alignTurns>, label: string) {
  console.log(`\n=== ${label} ===`)
  console.log(`Total pairs: ${pairs.length}`)
  const stats = computeAlignStats(pairs)
  console.log(`matched=${stats.matched} aOnly=${stats.aOnly} bOnly=${stats.bOnly} avgSim=${stats.avgSimilarity.toFixed(3)} high=${stats.highSimilarity} mid=${stats.mediumSimilarity} low=${stats.lowSimilarity}`)
  for (const p of pairs) {
    if (p.type === "match") {
      const aRole = p.a?.role ?? "?"
      const bRole = p.b?.role ?? "?"
      const aPreview = (p.a?.content ?? p.a?.contentSummary ?? "").substring(0, 50).replace(/\n/g, "\\n")
      const bPreview = (p.b?.content ?? p.b?.contentSummary ?? "").substring(0, 50).replace(/\n/g, "\\n")
      const aTools = p.a?.toolCalls.map(tc => tc.toolName).join(",") ?? ""
      const bTools = p.b?.toolCalls.map(tc => tc.toolName).join(",") ?? ""
      console.log(`  MATCH sim=${p.similarity.toFixed(3)} A#${p.a?.turnIndex}(${aRole})[${aTools}] ↔ B#${p.b?.turnIndex}(${bRole})[${bTools}]`)
      console.log(`    A: ${aPreview}`)
      console.log(`    B: ${bPreview}`)
    } else if (p.type === "aOnly") {
      const aPreview = (p.a?.content ?? "").substring(0, 50).replace(/\n/g, "\\n")
      console.log(`  A-ONLY A#${p.a?.turnIndex}(${p.a?.role}) | ${aPreview}`)
    } else {
      const bPreview = (p.b?.content ?? "").substring(0, 50).replace(/\n/g, "\\n")
      console.log(`  B-ONLY B#${p.b?.turnIndex}(${p.b?.role}) | ${bPreview}`)
    }
  }
}

const SCENARIOS: Array<{ label: string; a: TurnData[]; b: TurnData[]; expectDesc: string }> = [
  {
    label: "S1: 同query同模型（理想情况）",
    expectDesc: "全部高相似对齐",
    a: [
      makeTurn({ turnIndex: 0, role: "user", content: "帮我修复登录页面的认证bug" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "<thinking>用户要求修复登录页面的认证bug，我需要检查认证逻辑代码</thinking>\n我来帮你修复登录认证bug。首先让我查看认证模块代码。", toolCalls: [
        { toolCallId: "tc1", toolName: "Read", state: "ok", durationMs: 200 },
        { toolCallId: "tc2", toolName: "Read", state: "ok", durationMs: 150 },
      ] }),
      makeTurn({ turnIndex: 2, role: "user", content: "找到bug了吗" }),
      makeTurn({ turnIndex: 3, role: "assistant", content: "<thinking>已经找到了认证逻辑中的问题</thinking>\n找到了！认证模块中密码验证的逻辑有问题，我来修复。", toolCalls: [
        { toolCallId: "tc3", toolName: "Write", state: "ok", durationMs: 300 },
      ] }),
      makeTurn({ turnIndex: 4, role: "user", content: "修好了吗，测试一下" }),
      makeTurn({ turnIndex: 5, role: "assistant", content: "已修复并测试通过。认证bug已解决。", toolCalls: [
        { toolCallId: "tc4", toolName: "Bash", state: "ok", durationMs: 500 },
      ] }),
    ],
    b: [
      makeTurn({ turnIndex: 0, role: "user", content: "帮我修复登录页面的认证bug" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "<thinking>用户要求修复登录认证bug</thinking>\n我来帮你修复认证bug，先看看代码。", toolCalls: [
        { toolCallId: "tc5", toolName: "Read", state: "ok", durationMs: 180 },
        { toolCallId: "tc6", toolName: "Read", state: "ok", durationMs: 160 },
      ] }),
      makeTurn({ turnIndex: 2, role: "user", content: "找到bug了吗" }),
      makeTurn({ turnIndex: 3, role: "assistant", content: "<thinking>认证逻辑中发现问题</thinking>\n找到了，认证密码验证有bug，我来修改。", toolCalls: [
        { toolCallId: "tc7", toolName: "Write", state: "ok", durationMs: 250 },
      ] }),
      makeTurn({ turnIndex: 4, role: "user", content: "修好了吗，测试一下" }),
      makeTurn({ turnIndex: 5, role: "assistant", content: "已修复，测试通过了。", toolCalls: [
        { toolCallId: "tc8", toolName: "Bash", state: "ok", durationMs: 400 },
      ] }),
    ],
  },
  {
    label: "S2: 同query不同模型（不同输出）",
    expectDesc: "user高相似，assistant中等相似（主题同但表达不同）",
    a: [
      makeTurn({ turnIndex: 0, role: "user", content: "帮我修复登录页面的认证bug" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "<thinking>The user wants to fix an authentication bug in the login page. I should read the auth module first.</thinking>\nI'll investigate the authentication module. Let me read the relevant files.", toolCalls: [
        { toolCallId: "tc1", toolName: "Read", state: "ok", durationMs: 200 },
      ] }),
      makeTurn({ turnIndex: 2, role: "user", content: "怎么样了" }),
      makeTurn({ turnIndex: 3, role: "assistant", content: "I found the issue in the password validation logic. Here's the fix:", toolCalls: [
        { toolCallId: "tc2", toolName: "Write", state: "ok", durationMs: 300 },
      ] }),
    ],
    b: [
      makeTurn({ turnIndex: 0, role: "user", content: "帮我修复登录页面的认证bug" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "<thinking>用户需要修复认证bug</thinking>\n我来检查认证代码，找到bug并修复。", toolCalls: [
        { toolCallId: "tc3", toolName: "Read", state: "ok", durationMs: 180 },
      ] }),
      makeTurn({ turnIndex: 2, role: "user", content: "怎么样了" }),
      makeTurn({ turnIndex: 3, role: "assistant", content: "找到了认证逻辑的问题，已修复：", toolCalls: [
        { toolCallId: "tc4", toolName: "Write", state: "ok", durationMs: 250 },
      ] }),
    ],
  },
  {
    label: "S3: 同query不同深度（A有子代理，B单代理）",
    expectDesc: "user对齐，初始assistant对齐，A的子代理turn应为A-only",
    a: [
      makeTurn({ turnIndex: 0, role: "user", content: "请你仔细阅读cannbot-insight仓的代码，深入理解" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "<thinking>用户要求深入理解cannbot-insight仓库，需要系统性探索</thinking>\n我来深入阅读这个仓库。", toolCalls: [
        { toolCallId: "tc1", toolName: "Read", state: "ok", durationMs: 200 },
        { toolCallId: "tc2", toolName: "Read", state: "ok", durationMs: 150 },
        { toolCallId: "tc3", toolName: "Read", state: "ok", durationMs: 100 },
      ] }),
      makeTurn({ turnIndex: 2, role: "assistant", content: "", toolCalls: [
        { toolCallId: "tc4", toolName: "task", state: "ok", durationMs: 0 },
      ] }),
      makeTurn({ turnIndex: 3, role: "assistant", content: "<thinking>探索代理已返回全面的总结</thinking>\n已深入阅读并理解 cannbot-insight 仓库。以下是架构概览：\n\n项目定位：LLM coding agent (opencode) 的 session 级可观测性工具\n\n技术栈：Next.js 16 App Router + Prisma + SQLite" }),
      makeTurn({ turnIndex: 4, role: "assistant", content: "", toolCalls: [
        { toolCallId: "tc5", toolName: "read", state: "ok", durationMs: 100 },
        { toolCallId: "tc6", toolName: "bash", state: "ok", durationMs: 200 },
      ] }),
      makeTurn({ turnIndex: 5, role: "assistant", content: "", toolCalls: [
        { toolCallId: "tc7", toolName: "read", state: "ok", durationMs: 150 },
        { toolCallId: "tc8", toolName: "read", state: "ok", durationMs: 120 },
      ] }),
      makeTurn({ turnIndex: 6, role: "assistant", content: "", toolCalls: [
        { toolCallId: "tc9", toolName: "read", state: "ok", durationMs: 200 },
      ] }),
    ],
    b: [
      makeTurn({ turnIndex: 0, role: "user", content: "好吧，那请你阅读一下cannbot-insight仓" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "<thinking>用户要求阅读cannbot-insight仓库</thinking>\n我来阅读这个仓库。", toolCalls: [
        { toolCallId: "tc10", toolName: "Read", state: "ok", durationMs: 200 },
        { toolCallId: "tc11", toolName: "Read", state: "ok", durationMs: 150 },
        { toolCallId: "tc12", toolName: "Read", state: "ok", durationMs: 100 },
      ] }),
      makeTurn({ turnIndex: 2, role: "assistant", content: "", toolCalls: [
        { toolCallId: "tc13", toolName: "read", state: "ok", durationMs: 100 },
        { toolCallId: "tc14", toolName: "read", state: "ok", durationMs: 150 },
        { toolCallId: "tc15", toolName: "read", state: "ok", durationMs: 120 },
      ] }),
      makeTurn({ turnIndex: 3, role: "assistant", content: "", toolCalls: [
        { toolCallId: "tc16", toolName: "read", state: "ok", durationMs: 200 },
        { toolCallId: "tc17", toolName: "read", state: "ok", durationMs: 180 },
      ] }),
      makeTurn({ turnIndex: 4, role: "assistant", content: "已阅读完毕。以下是 cannbot-insight 仓库的整体概览：\n\n**项目定位**：LLM coding agent (opencode) 的 session 级可观测性工具" }),
    ],
  },
  {
    label: "S4: 不同query相同工具（不应错误对齐）",
    expectDesc: "不应大量匹配，多数应为aOnly/bOnly",
    a: [
      makeTurn({ turnIndex: 0, role: "user", content: "实现数据库迁移脚本" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "我来编写数据库迁移脚本", toolCalls: [
        { toolCallId: "tc1", toolName: "Read", state: "ok", durationMs: 200 },
        { toolCallId: "tc2", toolName: "Write", state: "ok", durationMs: 300 },
      ] }),
      makeTurn({ turnIndex: 2, role: "user", content: "添加测试用例" }),
      makeTurn({ turnIndex: 3, role: "assistant", content: "添加数据库迁移测试", toolCalls: [
        { toolCallId: "tc3", toolName: "Write", state: "ok", durationMs: 200 },
      ] }),
    ],
    b: [
      makeTurn({ turnIndex: 0, role: "user", content: "配置网络防火墙设置" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "我来配置防火墙", toolCalls: [
        { toolCallId: "tc4", toolName: "Read", state: "ok", durationMs: 200 },
        { toolCallId: "tc5", toolName: "Write", state: "ok", durationMs: 300 },
      ] }),
      makeTurn({ turnIndex: 2, role: "user", content: "验证防火墙规则" }),
      makeTurn({ turnIndex: 3, role: "assistant", content: "验证防火墙配置", toolCalls: [
        { toolCallId: "tc6", toolName: "Bash", state: "ok", durationMs: 500 },
      ] }),
    ],
  },
  {
    label: "S5: 纯中文对话",
    expectDesc: "CJK单字提取应使相似query对齐",
    a: [
      makeTurn({ turnIndex: 0, role: "user", content: "晚上好 请问南京天气怎么样" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "晚上好！我无法查询实时天气数据，建议你查看天气应用。" }),
      makeTurn({ turnIndex: 2, role: "user", content: "好吧，那请你阅读一下cannbot-insight仓" }),
      makeTurn({ turnIndex: 3, role: "assistant", content: "我来阅读cannbot-insight仓库，了解其架构。", toolCalls: [
        { toolCallId: "tc1", toolName: "read", state: "ok", durationMs: 200 },
      ] }),
    ],
    b: [
      makeTurn({ turnIndex: 0, role: "user", content: "晚上好 请问南京天气怎么样" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "你好！我没有天气查询功能，请使用天气服务查看。" }),
      makeTurn({ turnIndex: 2, role: "user", content: "请你仔细阅读cannbot-insight仓的代码，深入理解" }),
      makeTurn({ turnIndex: 3, role: "assistant", content: "我来深入阅读cannbot-insight的代码和架构。", toolCalls: [
        { toolCallId: "tc2", toolName: "read", state: "ok", durationMs: 180 },
      ] }),
    ],
  },
  {
    label: "S6: 中英混合对话（实际最常见的模式）",
    expectDesc: "中文关键词+英文术语应同时参与cosine",
    a: [
      makeTurn({ turnIndex: 0, role: "user", content: "帮我优化这个 Prisma schema 的查询性能，当前有很多 N+1 问题" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "<thinking>用户遇到Prisma的N+1查询问题，需要优化schema和数据获取策略</thinking>\n我来分析 Prisma schema 中的 N+1 问题并优化。", toolCalls: [
        { toolCallId: "tc1", toolName: "Read", state: "ok", durationMs: 200 },
      ] }),
      makeTurn({ turnIndex: 2, role: "user", content: "看看 session detail page 的数据加载，有很多不必要的查询" }),
      makeTurn({ turnIndex: 3, role: "assistant", content: "session detail page 的数据加载确实有 N+1 问题。我来重构查询逻辑，使用 include 和 select 优化。", toolCalls: [
        { toolCallId: "tc2", toolName: "Read", state: "ok", durationMs: 150 },
        { toolCallId: "tc3", toolName: "Write", state: "ok", durationMs: 300 },
      ] }),
    ],
    b: [
      makeTurn({ turnIndex: 0, role: "user", content: "帮我优化 Prisma schema 查询性能，N+1 问题严重" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "<thinking>Prisma的N+1问题需要优化include策略</thinking>\n我来检查并优化 Prisma N+1 查询。", toolCalls: [
        { toolCallId: "tc4", toolName: "Read", state: "ok", durationMs: 200 },
      ] }),
      makeTurn({ turnIndex: 2, role: "user", content: "session detail 页面的查询太多了，需要优化" }),
      makeTurn({ turnIndex: 3, role: "assistant", content: "session detail 页面的 N+1 问题我来修复，用 Prisma include 优化。", toolCalls: [
        { toolCallId: "tc5", toolName: "Read", state: "ok", durationMs: 150 },
        { toolCallId: "tc6", toolName: "Write", state: "ok", durationMs: 280 },
      ] }),
    ],
  },
  {
    label: "S7: B有system prompt，A没有",
    expectDesc: "system turn应为B-only，后续user/assistant应正常对齐",
    a: [
      makeTurn({ turnIndex: 0, role: "user", content: "请阅读项目代码" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "我来阅读项目代码。", toolCalls: [
        { toolCallId: "tc1", toolName: "Read", state: "ok", durationMs: 200 },
      ] }),
      makeTurn({ turnIndex: 2, role: "user", content: "写个测试" }),
      makeTurn({ turnIndex: 3, role: "assistant", content: "我来写测试用例。", toolCalls: [
        { toolCallId: "tc2", toolName: "Write", state: "ok", durationMs: 300 },
      ] }),
    ],
    b: [
      makeTurn({ turnIndex: 0, role: "system", content: "You are a helpful coding assistant. Use tools to read and write files. Always run tests after making changes." }),
      makeTurn({ turnIndex: 1, role: "user", content: "请阅读项目代码" }),
      makeTurn({ turnIndex: 2, role: "assistant", content: "我来阅读项目代码，了解其结构。", toolCalls: [
        { toolCallId: "tc3", toolName: "Read", state: "ok", durationMs: 200 },
      ] }),
      makeTurn({ turnIndex: 3, role: "user", content: "写个测试" }),
      makeTurn({ turnIndex: 4, role: "assistant", content: "我来写测试，并在完成后运行验证。", toolCalls: [
        { toolCallId: "tc4", toolName: "Write", state: "ok", durationMs: 300 },
        { toolCallId: "tc5", toolName: "Bash", state: "ok", durationMs: 500 },
      ] }),
    ],
  },
  {
    label: "S8: 纯tool-call对话（无文本内容）",
    expectDesc: "应主要靠工具名对齐，相似工具序列的turn匹配",
    a: [
      makeTurn({ turnIndex: 0, role: "user", content: "读取src目录下的所有文件" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "", toolCalls: [
        { toolCallId: "tc1", toolName: "Glob", state: "ok", durationMs: 50 },
        { toolCallId: "tc2", toolName: "Read", state: "ok", durationMs: 100 },
        { toolCallId: "tc3", toolName: "Read", state: "ok", durationMs: 120 },
        { toolCallId: "tc4", toolName: "Read", state: "ok", durationMs: 150 },
      ] }),
      makeTurn({ turnIndex: 2, role: "assistant", content: "", toolCalls: [
        { toolCallId: "tc5", toolName: "Read", state: "ok", durationMs: 200 },
      ] }),
      makeTurn({ turnIndex: 3, role: "assistant", content: "已读取完所有文件，共15个文件。以下是项目结构概览..." }),
    ],
    b: [
      makeTurn({ turnIndex: 0, role: "user", content: "读取src目录下的所有文件" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "", toolCalls: [
        { toolCallId: "tc6", toolName: "Glob", state: "ok", durationMs: 60 },
        { toolCallId: "tc7", toolName: "Read", state: "ok", durationMs: 110 },
        { toolCallId: "tc8", toolName: "Read", state: "ok", durationMs: 130 },
      ] }),
      makeTurn({ turnIndex: 2, role: "assistant", content: "已读取所有文件。以下是项目概览..." }),
    ],
  },
  {
    label: "S9: 一方只有1个turn（极端不对称）",
    expectDesc: "只有一个pair，应为match或aOnly/bOnly",
    a: [
      makeTurn({ turnIndex: 0, role: "user", content: "你好" }),
    ],
    b: [
      makeTurn({ turnIndex: 0, role: "user", content: "帮我写个脚本" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "我来帮你写脚本。" }),
    ],
  },
  {
    label: "S10: 多轮对话中user/assistant交替错位",
    expectDesc: "应按内容对齐而非按index，避免assistant↔user错配",
    a: [
      makeTurn({ turnIndex: 0, role: "user", content: "问题A1" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "回答A1" }),
      makeTurn({ turnIndex: 2, role: "user", content: "问题A2" }),
      makeTurn({ turnIndex: 3, role: "assistant", content: "回答A2" }),
    ],
    b: [
      makeTurn({ turnIndex: 0, role: "system", content: "系统提示" }),
      makeTurn({ turnIndex: 1, role: "user", content: "问题B1" }),
      makeTurn({ turnIndex: 2, role: "assistant", content: "回答B1" }),
      makeTurn({ turnIndex: 3, role: "user", content: "问题B2" }),
      makeTurn({ turnIndex: 4, role: "assistant", content: "回答B2" }),
    ],
  },
]

describe("Complex alignment stress tests", () => {
  for (const { label, a, b, expectDesc } of SCENARIOS) {
    it(`${label} — ${expectDesc}`, () => {
      const pairs = alignTurns(a, b)
      diagnoseAlignment(pairs, label)
      expect(pairs.length).toBeGreaterThan(0)

      const sNum = label.match(/^S(\d+):/)?.[1]
      if (sNum === "1") {
        const matches = pairs.filter(p => p.type === "match")
        expect(matches.length).toBeGreaterThanOrEqual(a.length - 1)
        expect(matches.every(p => p.similarity >= 0.4)).toBe(true)
      }

      if (sNum === "2") {
        const userMatches = pairs.filter(p => p.type === "match" && p.a?.role === "user" && p.b?.role === "user")
        expect(userMatches.length).toBeGreaterThanOrEqual(2)
        expect(userMatches.every(p => p.similarity >= 0.35)).toBe(true)
        const assMatches = pairs.filter(p => p.type === "match" && p.a?.role === "assistant" && p.b?.role === "assistant")
        expect(assMatches.length).toBeGreaterThanOrEqual(2)
        expect(assMatches.every(p => p.similarity >= 0.3)).toBe(true)
      }

      if (sNum === "3") {
        const userMatches = pairs.filter(p => p.type === "match" && p.a?.role === "user" && p.b?.role === "user")
        expect(userMatches.length).toBeGreaterThanOrEqual(1)
        const aOnlyCount = pairs.filter(p => p.type === "aOnly").length
        expect(aOnlyCount).toBeGreaterThanOrEqual(2)
      }

      if (sNum === "4") {
        const matches = pairs.filter(p => p.type === "match")
        expect(matches.length).toBeLessThanOrEqual(2)
        if (matches.length > 0) {
          expect(matches.every(p => p.similarity < 0.5)).toBe(true)
        }
      }

      if (sNum === "5") {
        const matches = pairs.filter(p => p.type === "match")
        expect(matches.length).toBeGreaterThanOrEqual(3)
        const userMatches = matches.filter(p => p.a?.role === "user" && p.b?.role === "user")
        expect(userMatches.length).toBeGreaterThanOrEqual(2)
      }

      if (sNum === "6") {
        const matches = pairs.filter(p => p.type === "match")
        expect(matches.length).toBeGreaterThanOrEqual(3)
        expect(matches.some(p => p.similarity >= 0.5)).toBe(true)
      }

      if (sNum === "7") {
        const systemBOnly = pairs.filter(p => p.type === "bOnly" && p.b?.role === "system")
        expect(systemBOnly.length).toBe(1)
        const userMatches = pairs.filter(p => p.type === "match" && p.a?.role === "user" && p.b?.role === "user")
        expect(userMatches.length).toBeGreaterThanOrEqual(1)
      }

      if (sNum === "8") {
        const matches = pairs.filter(p => p.type === "match")
        expect(matches.length).toBeGreaterThanOrEqual(2)
        const userMatch = matches.find(p => p.a?.role === "user" && p.b?.role === "user")
        expect(userMatch).toBeDefined()
      }

      if (sNum === "9") {
        const matches = pairs.filter(p => p.type === "match")
        expect(matches.length).toBe(0)
        const aOnly = pairs.filter(p => p.type === "aOnly")
        const bOnly = pairs.filter(p => p.type === "bOnly")
        expect(aOnly.length + bOnly.length).toBe(pairs.length)
      }

      if (sNum === "10") {
        const roleMismatchPairs = pairs.filter(p => p.type === "match" && p.a?.role !== p.b?.role)
        expect(roleMismatchPairs.length).toBe(0)
      }
    })
  }
})
