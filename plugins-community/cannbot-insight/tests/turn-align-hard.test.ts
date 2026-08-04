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

function diagnose(pairs: ReturnType<typeof alignTurns>, label: string) {
  const stats = computeAlignStats(pairs)
  const lines: string[] = [
    `\n=== ${label} ===`,
    `pairs=${pairs.length} matched=${stats.matched} aOnly=${stats.aOnly} bOnly=${stats.bOnly} avgSim=${stats.avgSimilarity.toFixed(3)} high=${stats.highSimilarity} mid=${stats.mediumSimilarity} low=${stats.lowSimilarity}`,
  ]
  for (const p of pairs) {
    if (p.type === "match") {
      const aIdx = p.a?.turnIndex ?? "?"
      const bIdx = p.b?.turnIndex ?? "?"
      const aR = p.a?.role ?? "?"
      const bR = p.b?.role ?? "?"
      const aT = (p.a?.toolCalls ?? []).map(tc => tc.toolName).join(",")
      const bT = (p.b?.toolCalls ?? []).map(tc => tc.toolName).join(",")
      lines.push(`  M sim=${p.similarity.toFixed(3)} A#${aIdx}(${aR})[${aT}] ↔ B#${bIdx}(${bR})[${bT}]`)
    } else if (p.type === "aOnly") {
      lines.push(`  A-O A#${p.a?.turnIndex}(${p.a?.role})[${(p.a?.toolCalls ?? []).map(tc => tc.toolName).join(",")}]`)
    } else {
      lines.push(`  B-O B#${p.b?.turnIndex}(${p.b?.role})[${(p.b?.toolCalls ?? []).map(tc => tc.toolName).join(",")}]`)
    }
  }
  console.log(lines.join("\n"))
}

function t(idx: number, role: string, content: string | null, tools?: Array<{ toolCallId: string; toolName: string; state: string; durationMs: number }>, skills?: Array<{ skillName: string; eventType: string; success: boolean }>): TurnData {
  return makeTurn({ turnIndex: idx, role, content, toolCalls: tools ?? [], skillEvents: skills ?? [] })
}

const tc = (id: string, name: string, state: string = "ok", dur: number = 100) => ({ toolCallId: id, toolName: name, state, durationMs: dur })
const se = (name: string, event: string = "load", success: boolean = true) => ({ skillName: name, eventType: event, success })

const HARD: Array<{ label: string; a: TurnData[]; b: TurnData[]; assert: (pairs: ReturnType<typeof alignTurns>) => void }> = [
  {
    label: "H1: 15轮编码对话(同一bug,不同模型,中英混合)",
    a: [
      t(0, "user", "帮我修复 TurnTimeline 组件中的 hydration error，button 嵌套问题"),
      t(1, "assistant", "<thinking>用户要求修复TurnTimeline的hydration错误，需要检查button嵌套</thinking>\n我来检查 TurnTimeline 组件的 button 嵌套问题。", [tc("1","Read"), tc("2","Read")]),
      t(2, "assistant", "", [tc("3","Read"), tc("4","Read"), tc("5","Read")]),
      t(3, "assistant", "已找到 hydration 错误的原因：TurnTimeline 中 inner expand buttons 使用了 `<button>` 嵌套在 `<button>` 内，违反 HTML 规范。需要改为 `<span role=\"button\">`。"),
      t(4, "user", "好的，请修复这个问题，同时保持样式不变"),
      t(5, "assistant", "<thinking>需要将嵌套button改为span[role=button]，同时保持样式和交互</thinking>\n我来修改 TurnTimeline，将嵌套的 `<button>` 改为 `<span role=\"button\">`。", [tc("6","Write")]),
      t(6, "assistant", "", [tc("7","Bash")]),
      t(7, "assistant", "修改完成，TypeScript 编译通过。hydration 错误应该已修复。"),
      t(8, "user", "跑一下测试看看有没有问题"),
      t(9, "assistant", "我来运行测试验证修复。", [tc("8","Bash")]),
      t(10, "assistant", "所有测试通过。hydration 修复成功。", [tc("9","Bash")]),
      t(11, "user", "顺便看看 TurnTimeline 还有其他 hydration 风险吗"),
      t(12, "assistant", "<thinking>检查TurnTimeline的其他hydration风险点</thinking>\n我来全面检查 TurnTimeline 的 hydration 风险。", [tc("10","Read")]),
      t(13, "assistant", "检查完毕，还有一处 `<a>` 嵌套 `<a>` 的潜在问题在 collapse/expand 区域。建议也改为 `<span>`。"),
      t(14, "user", "一并修复吧"),
      t(15, "assistant", "一并修复了嵌套 `<a>` 的问题。所有修改通过编译和测试。", [tc("11","Write"), tc("12","Bash")]),
    ],
    b: [
      t(0, "user", "修复 TurnTimeline 的 hydration error — button 嵌套导致的问题"),
      t(1, "assistant", "<thinking>TurnTimeline的hydration错误是button嵌套导致的</thinking>\n我来定位并修复 TurnTimeline 的 hydration 问题。", [tc("13","Read"), tc("14","Read")]),
      t(2, "assistant", "", [tc("15","Read"), tc("16","Glob")]),
      t(3, "assistant", "找到问题了：inner expand buttons 在 outer `<button>` 内嵌套，需要用 `<span role=button>` 替代。"),
      t(4, "user", "修一下，样式不能变"),
      t(5, "assistant", "<thinking>将嵌套button替换为span，保持样式和交互</thinking>\n修改 TurnTimeline 内部按钮为 `<span role=\"button\">`。", [tc("17","Write")]),
      t(6, "assistant", "编译通过。hydration 修复完毕。"),
      t(7, "user", "跑个测试"),
      t(8, "assistant", "运行测试验证。", [tc("18","Bash")]),
      t(9, "assistant", "测试全部通过。"),
      t(10, "user", "检查下 TurnTimeline 还有没有其他 hydration 风险"),
      t(11, "assistant", "<thinking>检查其他hydration风险</thinking>\n我来检查 TurnTimeline 其他可能的 hydration 问题。", [tc("19","Read")]),
      t(12, "assistant", "发现一处 `<a>` 嵌套 `<a>` 的潜在 hydration 问题。建议也改 `<span>`。"),
      t(13, "user", "一并修"),
      t(14, "assistant", "一并修复了 `<a>` 嵌套问题。编译和测试通过。", [tc("20","Write"), tc("21","Bash")]),
    ],
    assert: (pairs) => {
      const m = pairs.filter(p => p.type === "match")
      const userM = m.filter(p => p.a?.role === "user" && p.b?.role === "user")
      expect(userM.length).toBeGreaterThanOrEqual(3)
      expect(userM.every(p => p.similarity >= 0.35)).toBe(true)
      const assM = m.filter(p => p.a?.role === "assistant" && p.b?.role === "assistant")
      expect(assM.length).toBeGreaterThanOrEqual(4)
      expect(assM.some(p => p.similarity >= 0.5)).toBe(true)
    },
  },
  {
    label: "H2: 20轮对话(A有子代理深度探索,B无子代理)",
    a: [
      t(0, "user", "请深入阅读 cannbot-insight 仓库的完整代码架构，理解所有模块"),
      t(1, "assistant", "<thinking>需要系统性探索整个仓库架构</thinking>\n我来深入阅读整个仓库。", [tc("1","Read"), tc("2","Read"), tc("3","Read")]),
      t(2, "assistant", "", [tc("4","task")], [se("explore")]),
      t(3, "user", "Thoroughly explore the cannbot-insight codebase...", null, null),
      t(4, "assistant", "<thinking>Systematic exploration needed</thinking>\nLet me explore all directories.", [tc("5","Read"), tc("6","Read")]),
      t(5, "assistant", "", [tc("7","Read"), tc("8","Read"), tc("9","bash")]),
      t(6, "assistant", "", [tc("10","Read"), tc("11","Read"), tc("12","Read")]),
      t(7, "assistant", "", [tc("13","Read"), tc("14","Read")]),
      t(8, "assistant", "Comprehensive exploration complete. Here's the architecture overview:\n\n- Data flow: opencode sessions.db → adapter → normalize → turn-split → bridge-builder → execution-split → Prisma\n- 9 tab views\n- CLI with Ink v7\n- Breather monitor module"),
      t(9, "assistant", "<thinking>子代理探索完毕，整理结果给用户</thinking>\n已深入阅读 cannbot-insight 仓库。以下是架构概览：\n\n**数据流**：opencode sessions.db → adapter → normalize → turn-split → bridge-builder → execution-split → data-service → Prisma\n\n**Web UI**：9 个 tab（Overview, Turns, Workflow, Trace, Subagents, Skills, Interactions, AI Workflow, Context）\n\n**CLI**：Ink v7 + Commander.js，10 个命令\n\n**Breather**：上下文监控模块"),
      t(10, "user", "看看 bridge-builder 模块的实现细节"),
      t(11, "assistant", "<thinking>用户要求查看bridge-builder细节</thinking>\n我来查看 bridge-builder 的实现细节。", [tc("15","Read")]),
      t(12, "assistant", "bridge-builder 的核心逻辑是追踪主代理→子代理的 dispatch/response 。它通过 ToolCall 的 dispatchBridgeId 和 Execution 的 parentExecutionId 建立桥接关系。"),
      t(13, "user", "写个集成测试覆盖 bridge-builder 的完整数据流"),
      t(14, "assistant", "<thinking>需要为bridge-builder写集成测试</thinking>\n我来写 bridge-builder 的集成测试。", [tc("16","Read"), tc("17","Write")]),
      t(15, "assistant", "测试已编写。bridge-builder IT 覆盖了 dispatch→response→subagentSessionId 的完整链路。", [tc("18","Bash")]),
      t(16, "user", "测试跑通了吗"),
      t(17, "assistant", "所有 bridge-builder 测试通过。", [tc("19","Bash")]),
      t(18, "user", "好的，现在看看 execution-split 的逻辑"),
      t(19, "assistant", "<thinking>用户要求查看execution-split</thinking>\n我来查看 execution-split 的实现。", [tc("20","Read")]),
    ],
    b: [
      t(0, "user", "请阅读 cannbot-insight 仓库的代码架构"),
      t(1, "assistant", "<thinking>用户要求阅读仓库架构</thinking>\n我来阅读这个仓库。", [tc("21","Read"), tc("22","Read"), tc("23","Read")]),
      t(2, "assistant", "", [tc("24","Read"), tc("25","Read"), tc("26","Read")]),
      t(3, "assistant", "", [tc("27","Read"), tc("28","Read")]),
      t(4, "assistant", "已阅读完毕。以下是架构概览：\n\n**数据流**：sessions.db → adapter → normalize → turn-split → bridge-builder → execution-split → Prisma\n\n**9 tab views**：Overview, Turns, Workflow, Trace, Subagents, Skills, Interactions, AI Workflow, Context"),
      t(5, "user", "看看 bridge-builder 的实现细节"),
      t(6, "assistant", "<thinking>查看bridge-builder细节</thinking>\n我来查看 bridge-builder 的核心实现。", [tc("29","Read")]),
      t(7, "assistant", "bridge-builder 通过 dispatchBridgeId 和 parentExecutionId 建立主代理→子代理的桥接。"),
      t(8, "user", "写个集成测试覆盖 bridge-builder"),
      t(9, "assistant", "我来写 bridge-builder 的集成测试。", [tc("30","Read"), tc("31","Write")]),
      t(10, "assistant", "测试完成，覆盖了完整链路。", [tc("32","Bash")]),
      t(11, "user", "测试结果"),
      t(12, "assistant", "所有测试通过。"),
      t(13, "user", "看看 execution-split"),
      t(14, "assistant", "<thinking>查看execution-split</thinking>\n我来查看 execution-split 的实现。", [tc("33","Read")]),
    ],
    assert: (pairs) => {
      const m = pairs.filter(p => p.type === "match")
      const userM = m.filter(p => p.a?.role === "user" && p.b?.role === "user")
      expect(userM.length).toBeGreaterThanOrEqual(4)
      const aOnly = pairs.filter(p => p.type === "aOnly")
      expect(aOnly.length).toBeGreaterThanOrEqual(4)
      const aOnlyAss = aOnly.filter(p => p.a?.role === "assistant")
      expect(aOnlyAss.length).toBeGreaterThanOrEqual(3)
    },
  },
  {
    label: "H3: 两个完全不同的编码任务(不应交叉对齐)",
    a: [
      t(0, "user", "实现一个新的 SkillEvent 分组功能，按 skill family 分类显示"),
      t(1, "assistant", "<thinking>需要实现SkillEvent按family分组</thinking>\n我来实现 SkillEvent 分组功能。", [tc("1","Read"), tc("2","Read")]),
      t(2, "assistant", "", [tc("3","Read"), tc("4","Glob")]),
      t(3, "assistant", "已了解 SkillEvent 的数据结构。我来实现 skill-family-config.ts 和 SkillCharts 的分组逻辑。", [tc("5","Write")]),
      t(4, "user", "更新对应的 tab 组件来显示分组结果"),
      t(5, "assistant", "我来更新 SkillsTab 的显示逻辑，按 family 分组。", [tc("6","Read"), tc("7","Write")]),
      t(6, "user", "跑测试验证"),
      t(7, "assistant", "运行测试。", [tc("8","Bash")]),
      t(8, "assistant", "所有测试通过。分组功能正常。"),
    ],
    b: [
      t(0, "user", "给 CompareTurns 添加 Beyond Compare 风格的智能对齐功能"),
      t(1, "assistant", "<thinking>需要实现Beyond Compare式的智能turn对齐</thinking>\n我来设计并实现 Turn-by-Turn 的智能对齐算法。", [tc("9","Read"), tc("10","Read")]),
      t(2, "assistant", "", [tc("11","Read")]),
      t(3, "assistant", "设计方案：使用 Needleman-Wunsch 算法做全局序列对齐，余弦相似度计算 turn 匹配度。", [tc("12","Write")]),
      t(4, "user", "UI 也要做 Beyond Compare 风格的重设计"),
      t(5, "assistant", "我来重新设计 CompareTurns 的 UI，加入 Overview Gutter 和内容对比。", [tc("13","Write")]),
      t(6, "user", "跑测试"),
      t(7, "assistant", "运行测试。", [tc("14","Bash")]),
      t(8, "assistant", "全部通过。智能对齐功能完成。"),
    ],
    assert: (pairs) => {
      const m = pairs.filter(p => p.type === "match")
      expect(m.length).toBeLessThanOrEqual(7)
      const genericShortMatches = m.filter(p => p.similarity >= 0.6)
      const genericPairs = genericShortMatches.filter(p => {
        const aContent = p.a?.content ?? p.a?.contentSummary ?? ""
        const bContent = p.b?.content ?? p.b?.contentSummary ?? ""
        return aContent.length < 30 && bContent.length < 30
      })
      expect(genericPairs.length).toBeLessThanOrEqual(2)
      const gaps = pairs.filter(p => p.type === "aOnly" || p.type === "bOnly")
      expect(gaps.length).toBeGreaterThanOrEqual(Math.floor(pairs.length * 0.35))
    },
  },
  {
    label: "H4: 相同任务但一方中途换了方向",
    a: [
      t(0, "user", "优化 ContextTracker 的性能，当前加载太慢"),
      t(1, "assistant", "<thinking>ContextTracker性能问题需要分析</thinking>\n我来分析 ContextTracker 的性能瓶颈。", [tc("1","Read"), tc("2","Read")]),
      t(2, "assistant", "ContextTracker 的主要瓶颈在 API 调用：每次加载都重新 fetch 所有 turns。需要加缓存。"),
      t(3, "user", "先加 client-side 缓存看看效果"),
      t(4, "assistant", "我来给 ContextTracker 加 client-side 缓存，用 useRef 存储 turns 数据。", [tc("3","Write")]),
      t(5, "assistant", "缓存已添加。加载时间从 3s 降到 0.5s。"),
      t(6, "user", "再优化下 context window 配置的加载"),
      t(7, "assistant", "<thinking>优化context-window配置加载</thinking>\n我来优化 context-window-config 的加载。", [tc("4","Read"), tc("5","Write")]),
      t(8, "assistant", "context-window 配置已优化为预计算。"),
      t(9, "user", "整体性能怎么样了"),
      t(10, "assistant", "整体性能提升了约 80%。ContextTracker 加载时间从 3s 降到 0.3s。", [tc("6","Bash")]),
    ],
    b: [
      t(0, "user", "优化 ContextTracker 的性能"),
      t(1, "assistant", "<thinking>分析ContextTracker性能问题</thinking>\n我来分析 ContextTracker 性能瓶颈。", [tc("7","Read")]),
      t(2, "assistant", "瓶颈在 API 调用重复加载所有 turns。"),
      t(3, "user", "不要加缓存了，直接改 API 让它只返回必要字段"),
      t(4, "assistant", "<thinking>用户要求改API返回精简数据而非加缓存</thinking>\n我来修改 API，让 /observe/session/turns 只返回 context tracker 需要的字段。", [tc("8","Read"), tc("9","Write")]),
      t(5, "assistant", "API 已优化，只返回 turnIndex/totalTokens/inputTokens/role/subagentSessionId。"),
      t(6, "user", "前端也要相应调整"),
      t(7, "assistant", "我来调整 ContextTracker 的数据获取逻辑，使用精简 API。", [tc("10","Write")]),
      t(8, "assistant", "前端调整完成。整体加载从 3s 降到 0.2s。"),
      t(9, "user", "性能怎么样了"),
      t(10, "assistant", "性能提升 90%。ContextTracker 加载从 3s 降到 0.2s。"),
    ],
    assert: (pairs) => {
      const userM = pairs.filter(p => p.type === "match" && p.a?.role === "user" && p.b?.role === "user")
      expect(userM.length).toBeGreaterThanOrEqual(2)
      const assM = pairs.filter(p => p.type === "match" && p.a?.role === "assistant" && p.b?.role === "assistant")
      expect(assM.length).toBeGreaterThanOrEqual(3)
      const firstTwoAss = assM.filter(p => (p.a?.turnIndex ?? 0) <= 2 && (p.b?.turnIndex ?? 0) <= 2)
      expect(firstTwoAss.every(p => p.similarity >= 0.4)).toBe(true)
      const divergeAss = assM.find(p => (p.a?.turnIndex ?? 0) >= 4 && (p.b?.turnIndex ?? 0) >= 4)
      if (divergeAss) {
        expect(divergeAss.similarity).toBeLessThan(0.7)
      }
    },
  },
  {
    label: "H5: skill加载密集型对话",
    a: [
      t(0, "user", "使用 cannbot-skill-review 技能检视 ops/skills 目录下的所有 skill"),
      t(1, "assistant", "<thinking>用户要求使用skill-review检视所有skill</thinking>\n我来用 cannbot-skill-review 检视所有 ops/skills 下的 skill 模块。", [tc("1","Glob"), tc("2","Glob")], [se("cannbot-skill-review")]),
      t(2, "assistant", "", [tc("3","task")], [se("cannbot-skill-review")]),
      t(3, "assistant", "", [], [se("cannbot-skill-review")]),
      t(4, "assistant", "cannbot-skill-review 检视完毕。发现 3 个 skill 有 SKILL.md 格式问题和 2 个有越界执行风险。详细结果如下...", [tc("4","Read"), tc("5","Write")], [se("cannbot-skill-review")]),
      t(5, "user", "修复这 5 个有问题的 skill"),
      t(6, "assistant", "<thinking>需要修复5个有问题的skill</thinking>\n我来逐一修复这 5 个 skill。", [tc("6","Read"), tc("7","Write")]),
      t(7, "assistant", "", [tc("8","Read"), tc("9","Write")]),
      t(8, "assistant", "5 个 skill 全部修复完成。格式和越界问题已解决。", [tc("10","Bash")]),
    ],
    b: [
      t(0, "user", "用 cannbot-skill-review 检视 ops 目录下所有 skill 模块"),
      t(1, "assistant", "<thinking>检视ops目录的skill模块</thinking>\n我来检视 ops 目录下的所有 skill 模块。", [tc("11","Glob")], [se("cannbot-skill-review")]),
      t(2, "assistant", "", [tc("12","task")], [se("cannbot-skill-review")]),
      t(3, "assistant", "cannbot-skill-review 检视结果：发现 4 个格式问题和 1 个越界风险。详细报告...", [], [se("cannbot-skill-review")]),
      t(4, "user", "修复这些有问题的 skill"),
      t(5, "assistant", "<thinking>修复有问题的skill</thinking>\n我来修复检视发现的问题。", [tc("13","Read"), tc("14","Write")]),
      t(6, "assistant", "修复完成。所有 skill 已符合规范。", [tc("15","Bash")]),
    ],
    assert: (pairs) => {
      const skillMatches = pairs.filter(p => p.type === "match" && (p.a?.skillEvents.length > 0 || p.b?.skillEvents.length > 0))
      expect(skillMatches.length).toBeGreaterThanOrEqual(2)
      expect(skillMatches.every(p => p.similarity >= 0.3)).toBe(true)
      const userM = pairs.filter(p => p.type === "match" && p.a?.role === "user" && p.b?.role === "user")
      expect(userM.length).toBeGreaterThanOrEqual(2)
    },
  },
  {
    label: "H6: 一方全是空content的tool调用序列(12轮)",
    a: [
      t(0, "user", "读取 src/lib/ingest 下的所有文件并分析 pipeline"),
      t(1, "assistant", "", [tc("1","Glob")]),
      t(2, "assistant", "", [tc("2","Read"), tc("3","Read"), tc("4","Read"), tc("5","Read")]),
      t(3, "assistant", "", [tc("6","Read"), tc("7","Read"), tc("8","Read")]),
      t(4, "assistant", "", [tc("9","Read"), tc("10","Read")]),
      t(5, "assistant", "已分析 ingest pipeline。数据流为：adapter → normalize → turn-split → bridge-builder → execution-split → phase-split → merge → data-service → Prisma write"),
      t(6, "user", "继续读取 src/app/api 下的所有路由"),
      t(7, "assistant", "", [tc("11","Glob")]),
      t(8, "assistant", "", [tc("12","Read"), tc("13","Read"), tc("14","Read"), tc("15","Read")]),
      t(9, "assistant", "", [tc("16","Read"), tc("17","Read"), tc("18","Read")]),
      t(10, "assistant", "API 路由分析完毕。共 3 组：/api/ingest(7路由)、/api/observe(9路由)、/api/ai(2路由)、/api/breather(7路由)"),
      t(11, "user", "写个 pipeline 覆盖的集成测试"),
      t(12, "assistant", "", [tc("19","Write")]),
      t(13, "assistant", "测试已写好并通过。覆盖了完整的 adapter → Prisma pipeline。", [tc("20","Bash")]),
    ],
    b: [
      t(0, "user", "读取 src/lib/ingest 的文件，分析 pipeline 流程"),
      t(1, "assistant", "", [tc("21","Glob")]),
      t(2, "assistant", "", [tc("22","Read"), tc("23","Read"), tc("24","Read")]),
      t(3, "assistant", "", [tc("25","Read"), tc("26","Read")]),
      t(4, "assistant", "ingest pipeline 分析：adapter → normalize → turn-split → bridge-builder → execution-split → data-service → Prisma"),
      t(5, "user", "继续看 src/app/api 的路由"),
      t(6, "assistant", "", [tc("27","Glob")]),
      t(7, "assistant", "", [tc("28","Read"), tc("29","Read"), tc("30","Read")]),
      t(8, "assistant", "API 路由共 3 组：ingest(7)、observe(9)、ai(2)、breather(7)"),
      t(9, "user", "写个 pipeline 集成测试"),
      t(10, "assistant", "", [tc("31","Write")]),
      t(11, "assistant", "测试完成，覆盖了完整 pipeline。", [tc("32","Bash")]),
    ],
    assert: (pairs) => {
      const m = pairs.filter(p => p.type === "match")
      expect(m.length).toBeGreaterThanOrEqual(4)
      const userM = m.filter(p => p.a?.role === "user" && p.b?.role === "user")
      expect(userM.length).toBeGreaterThanOrEqual(3)
      const contentM = m.filter(p =>
        (p.a?.content && p.a?.content.trim()) || (p.b?.content && p.b?.content.trim())
      )
      expect(contentM.length).toBeGreaterThanOrEqual(3)
      const skeletonM = m.filter(p =>
        p.a?.role === "assistant" && p.b?.role === "assistant" && !(p.a?.content) && !(p.b?.content)
      )
      expect(skeletonM.length).toBe(0)
    },
  },
  {
    label: "H7: 一方有compaction/system/prelude turn,另一方没有",
    a: [
      t(0, "system", "You are a coding assistant with access to tools for reading, writing, and executing code. Always test your changes."),
      t(1, "user", "修复 TokenBarChart 的 SVG 渲染问题"),
      t(2, "assistant", "<thinking>TokenBarChart的SVG渲染问题需要排查</thinking>\n我来排查 TokenBarChart 的 SVG 渲染问题。", [tc("1","Read")]),
      t(3, "assistant", "", [tc("2","Read"), tc("3","Read")]),
      t(4, "assistant", "问题在于 SVG viewBox 计算逻辑。barHeight 函数在某些 token 值为 0 时返回 0，导致 SVG 元素不可见。需要 Math.max(h, 1)。"),
      t(5, "user", "修一下"),
      t(6, "assistant", "我来修复 TokenBarChart 的 SVG 渲染。", [tc("4","Write")]),
      t(7, "assistant", "修复完成。SVG 渲染正常。", [tc("5","Bash")]),
      t(8, "user", "跑下 lint"),
      t(9, "assistant", "运行 eslint 检查。", [tc("6","Bash")]),
      t(10, "assistant", "lint 通过。TokenBarChart 修复完毕。"),
    ],
    b: [
      t(0, "user", "修复 TokenBarChart 的 SVG 渲染 bug"),
      t(1, "assistant", "<thinking>排查TokenBarChart SVG渲染bug</thinking>\n我来排查 SVG 渲染问题。", [tc("7","Read")]),
      t(2, "assistant", "", [tc("8","Read")]),
      t(3, "assistant", "barHeight 在 token=0 时返回 0 导致 SVG 不可见。需要 Math.max(h, 1)。"),
      t(4, "user", "修"),
      t(5, "assistant", "修复 TokenBarChart SVG 渲染。", [tc("9","Write")]),
      t(6, "assistant", "修复完成。", [tc("10","Bash")]),
      t(7, "user", "跑 lint"),
      t(8, "assistant", "eslint 通过。", [tc("11","Bash")]),
      t(9, "assistant", "lint 通过。修复完毕。"),
    ],
    assert: (pairs) => {
      const systemOnly = pairs.filter(p => p.type === "aOnly" && p.a?.role === "system")
      expect(systemOnly.length).toBe(1)
      const userM = pairs.filter(p => p.type === "match" && p.a?.role === "user" && p.b?.role === "user")
      expect(userM.length).toBeGreaterThanOrEqual(2)
      expect(userM.every(p => p.similarity >= 0.4)).toBe(true)
    },
  },
  {
    label: "H8: 同一任务一方成功了另一方失败了(错误处理turn)",
    a: [
      t(0, "user", "部署应用到生产服务器"),
      t(1, "assistant", "<thinking>需要部署应用到生产</thinking>\n我来部署应用到生产服务器。", [tc("1","Bash")]),
      t(2, "assistant", "部署成功。应用已上线。", [tc("2","Bash")]),
      t(3, "user", "验证一下生产环境"),
      t(4, "assistant", "我来验证生产环境状态。", [tc("3","Bash")]),
      t(5, "assistant", "生产环境验证通过。所有健康检查正常。"),
    ],
    b: [
      t(0, "user", "部署应用到生产服务器"),
      t(1, "assistant", "<thinking>部署应用到生产</thinking>\n我来尝试部署。", [tc("4","Bash")]),
      t(2, "assistant", "部署失败！连接超时。错误信息：Connection refused to production server。", [tc("5","Bash", "error")]),
      t(3, "assistant", "", [tc("6","Bash", "error")]),
      t(4, "user", "部署失败了，检查下网络配置"),
      t(5, "assistant", "<thinking>检查网络配置</thinking>\n我来检查网络配置。", [tc("7","Read")]),
      t(6, "assistant", "防火墙规则阻止了 SSH 连接。我来调整配置。", [tc("8","Write")]),
      t(7, "user", "重新部署"),
      t(8, "assistant", "重新部署。", [tc("9","Bash")]),
      t(9, "assistant", "部署成功。应用已上线。"),
    ],
    assert: (pairs) => {
      const userM = pairs.filter(p => p.type === "match" && p.a?.role === "user" && p.b?.role === "user")
      expect(userM.length).toBeGreaterThanOrEqual(1)
      const firstUser = pairs.find(p => p.type === "match" && p.a?.turnIndex === 0 && p.b?.turnIndex === 0)
      expect(firstUser).toBeDefined()
      expect(firstUser!.similarity).toBeGreaterThanOrEqual(0.6)
    },
  },
  {
    label: "H9: 大量轮次(30vs15)的极端不对称",
    a: [
      t(0, "user", "从零开始构建一个完整的 Next.js 可观测性工具"),
      t(1, "assistant", "<thinking>这是一个大项目需要系统性规划</thinking>\n我来规划并构建这个项目。", [tc("1","Bash")]),
      t(2, "assistant", "", [tc("2","Bash"), tc("3","Write")]),
      t(3, "assistant", "项目初始化完成。Next.js 16 + Prisma + SQLite。"),
      t(4, "assistant", "", [tc("4","Write"), tc("5","Write")]),
      t(5, "assistant", "", [tc("6","Write")]),
      t(6, "assistant", "", [tc("7","Write"), tc("8","Write")]),
      t(7, "assistant", "数据模型和 API 路由已完成。"),
      t(8, "user", "继续做 UI"),
      t(9, "assistant", "", [tc("9","Write"), tc("10","Write")]),
      t(10, "assistant", "", [tc("11","Write"), tc("12","Write")]),
      t(11, "assistant", "", [tc("13","Write")]),
      t(12, "assistant", "首页和 SessionList 已完成。"),
      t(13, "user", "做 session 详情页"),
      t(14, "assistant", "", [tc("14","Write"), tc("15","Write")]),
      t(15, "assistant", "", [tc("16","Write"), tc("17","Write")]),
      t(16, "assistant", "session 详情页和 9 个 tab 完成。"),
      t(17, "user", "添加 TurnTimeline 组件"),
      t(18, "assistant", "", [tc("18","Write")]),
      t(19, "assistant", "TurnTimeline 完成。"),
      t(20, "user", "加 CLI 前端"),
      t(21, "assistant", "", [tc("19","Write"), tc("20","Write")]),
      t(22, "assistant", "", [tc("21","Write"), tc("22","Write")]),
      t(23, "assistant", "CLI TUI 完成。SessionList + OverviewTab + TurnDetail。"),
      t(24, "user", "跑一下整体测试"),
      t(25, "assistant", "运行全部测试。", [tc("23","Bash")]),
      t(26, "assistant", "187 个测试全部通过。项目构建完成。"),
      t(27, "user", "好的，提交代码"),
      t(28, "assistant", "我来提交代码。", [tc("24","Bash")]),
      t(29, "assistant", "代码已提交。项目完成。"),
    ],
    b: [
      t(0, "user", "从零构建 Next.js 可观测性工具"),
      t(1, "assistant", "<thinking>规划项目架构</thinking>\n我来快速构建这个项目。", [tc("25","Bash"), tc("26","Write")]),
      t(2, "assistant", "项目初始化和数据模型完成。"),
      t(3, "assistant", "", [tc("27","Write"), tc("28","Write")]),
      t(4, "assistant", "API 和首页完成。"),
      t(5, "user", "做 session 详情页和 tab"),
      t(6, "assistant", "", [tc("29","Write"), tc("30","Write")]),
      t(7, "assistant", "session 详情和所有 tab 完成。"),
      t(8, "user", "加 CLI"),
      t(9, "assistant", "", [tc("31","Write"), tc("32","Write")]),
      t(10, "assistant", "CLI 完成。"),
      t(11, "user", "跑测试提交"),
      t(12, "assistant", "运行测试并提交。", [tc("33","Bash")]),
      t(13, "assistant", "测试通过，代码已提交。"),
    ],
    assert: (pairs) => {
      const userM = pairs.filter(p => p.type === "match" && p.a?.role === "user" && p.b?.role === "user")
      expect(userM.length).toBeGreaterThanOrEqual(4)
      const aOnly = pairs.filter(p => p.type === "aOnly")
      expect(aOnly.length).toBeGreaterThanOrEqual(8)
    },
  },
  {
    label: "H10: 编码对话+subagent+skill+error综合场景",
    a: [
      t(0, "system", "You are an expert coding assistant. Use skills when available. Always validate changes."),
      t(1, "user", "为 cannbot-insight 实现完整的 export 功能，支持 markdown 导出"),
      t(2, "assistant", "<thinking>需要实现session的markdown导出功能</thinking>\n我来实现 markdown 导出功能。先看看现有的数据结构和导出需求。", [tc("1","Read"), tc("2","Read")]),
      t(3, "assistant", "", [tc("3","Read"), tc("4","Read")]),
      t(4, "assistant", "已了解数据结构。我来实现 MarkdownExporter 类。", [tc("5","Write")]),
      t(5, "assistant", "", [tc("6","Write")]),
      t(6, "assistant", "", [tc("7","task")], [se("cannbot-skill-review")]),
      t(7, "assistant", "", [], [se("cannbot-skill-review")]),
      t(8, "assistant", "skill-review 检视了导出模块。发现一处 error handling 缺失：stream 模式下 chunk 拼接可能导致 JSON 解析失败。", [tc("8","Read")], [se("cannbot-skill-review")]),
      t(9, "user", "修复 error handling，然后加集成测试"),
      t(10, "assistant", "<thinking>修复error handling并加测试</thinking>\n我来修复 stream error handling 并加测试。", [tc("9","Write"), tc("10","Write")]),
      t(11, "assistant", "", [tc("11","Bash", "error")]),
      t(12, "assistant", "stream error handling 修复完成。集成测试已通过。", [tc("12","Bash")]),
    ],
    b: [
      t(0, "user", "实现 cannbot-insight 的 export 功能，支持 markdown 导出"),
      t(1, "assistant", "<thinking>实现markdown导出功能</thinking>\n我来实现 markdown 导出。", [tc("13","Read"), tc("14","Read")]),
      t(2, "assistant", "", [tc("15","Read")]),
      t(3, "assistant", "设计完毕。我来写 MarkdownExporter。", [tc("16","Write")]),
      t(4, "assistant", "", [tc("17","Write")]),
      t(5, "user", "加集成测试"),
      t(6, "assistant", "我来写 export 的集成测试。", [tc("18","Write")]),
      t(7, "assistant", "测试通过。导出功能完成。", [tc("19","Bash")]),
    ],
    assert: (pairs) => {
      const userM = pairs.filter(p => p.type === "match" && p.a?.role === "user" && p.b?.role === "user")
      expect(userM.length).toBeGreaterThanOrEqual(2)
      const skillInAny = pairs.filter(p =>
        (p.a?.skillEvents.length > 0 || p.b?.skillEvents.length > 0) &&
        p.type !== "match"
      )
      expect(skillInAny.length).toBeGreaterThanOrEqual(1)
      const aOnly = pairs.filter(p => p.type === "aOnly")
      expect(aOnly.length).toBeGreaterThanOrEqual(3)
    },
  },
]

describe("Hard alignment stress tests", () => {
  for (const { label, a, b, assert } of HARD) {
    it(`${label}`, () => {
      const pairs = alignTurns(a, b)
      diagnose(pairs, label)
      assert(pairs)
    })
  }
})
