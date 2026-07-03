// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import fs from "node:fs"
import path from "node:path"

interface StageInfo {
  lastCompletedStep: string
  lastCompletedSubagent: string
  nextStep: string
  stageName: string
  operatorName: string
  statusRows: Array<{ step: string; status: string }>
}

const SUBAGENT_MAP: Record<string, string> = {
  "开发准备": "general",
  "需求分析": "ascendc-ops-architect",
  "spec生成": "ascendc-ops-architect",
  "spec自审": "ascendc-ops-architect",
  "方案设计": "ascendc-ops-architect",
  "方案评审": "ascendc-ops-architect",
  "测试设计": "ascendc-ops-tester",
  "测试设计评审": "ascendc-ops-tester",
  "骨架搭建": "ascendc-ops-developer",
  "穿刺验证": "ascendc-ops-developer",
  "核心路径UT": "ascendc-ops-developer",
  "策略整合": "ascendc-ops-developer",
  "Tiling分支UT": "ascendc-ops-developer",
  "全功能实现": "ascendc-ops-developer",
  "全覆盖UT": "ascendc-ops-developer",
  "汇合验证": "ascendc-ops-developer",
  "白盒测试": "ascendc-ops-developer",
  "C++标准用例": "ascendc-ops-tester",
  "C++多shape用例": "ascendc-ops-tester",
  "C++全量用例": "ascendc-ops-tester",
  "ST测试开发": "ascendc-ops-tester",
  "PyTorch ST测试": "ascendc-ops-tester",
  "测试工程师验收": "ascendc-ops-tester",
  "精度验收": "ascendc-ops-tester",
  "性能达标验收": "ascendc-ops-developer",
  "代码检视": "ascendc-ops-reviewer",
  "开发总结": "general",
  "文档与示例": "general",
}

export function findLogMd(projectPath: string): string | null {
  const operatorsDir = path.join(projectPath, "operators")
  if (!fs.existsSync(operatorsDir)) return null

  const entries = fs.readdirSync(operatorsDir)
  for (const entry of entries) {
    const logPath = path.join(operatorsDir, entry, "docs", "LOG.md")
    if (fs.existsSync(logPath)) return logPath
  }
  return null
}

export function parseLogMd(logPath: string): StageInfo | null {
  if (!fs.existsSync(logPath)) return null

  const content = fs.readFileSync(logPath, "utf-8")
  const operatorName = extractOperatorName(logPath)
  const statusRows = extractStatusTable(content)

  if (statusRows.length === 0) return null

  let lastCompletedIdx = -1
  for (let i = statusRows.length - 1; i >= 0; i--) {
    if (statusRows[i].status.includes("✅")) {
      lastCompletedIdx = i
      break
    }
  }

  if (lastCompletedIdx < 0) {
    return {
      lastCompletedStep: "",
      lastCompletedSubagent: "",
      nextStep: statusRows[0]?.step ?? "",
      stageName: "未开始",
      operatorName,
      statusRows,
    }
  }

  const lastCompleted = statusRows[lastCompletedIdx]
  const nextStep =
    lastCompletedIdx + 1 < statusRows.length
      ? statusRows[lastCompletedIdx + 1].step
      : "已完成"

  const stageName = determineStageName(lastCompleted.step)
  const subagent = SUBAGENT_MAP[lastCompleted.step] ?? "unknown"

  return {
    lastCompletedStep: lastCompleted.step,
    lastCompletedSubagent: subagent,
    nextStep,
    stageName,
    operatorName,
    statusRows,
  }
}

function extractOperatorName(logPath: string): string {
  const parts = logPath.split("/")
  const opsIdx = parts.indexOf("operators")
  if (opsIdx >= 0 && opsIdx + 1 < parts.length) return parts[opsIdx + 1]
  return "unknown"
}

function extractStatusTable(content: string): Array<{ step: string; status: string }> {
  const rows: Array<{ step: string; status: string }> = []
  const lines = content.split("\n")

  for (const line of lines) {
    if (!line.startsWith("|")) continue
    if (line.includes("---")) continue

    const cells = line
      .split("|")
      .map((c: any) => c.trim())
      .filter((c: any) => c.length > 0)

    if (cells.length >= 2) {
      const step = cells[0]
      const status = cells[1]
      rows.push({ step, status })
    }
  }

  return rows
}

function determineStageName(step: string): string {
  if (step.startsWith("1.") || step.includes("需求") || step.includes("设计") || step.includes("spec"))
    return "阶段一：需求与设计"
  if (step.includes("迭代一") || step.includes("骨架搭建") || step.includes("穿刺验证"))
    return "阶段二/迭代一"
  if (step.includes("迭代二") || step.includes("策略整合") || step.includes("Tiling"))
    return "阶段二/迭代二"
  if (step.includes("迭代三") || step.includes("全功能") || step.includes("全覆盖"))
    return "阶段二/迭代三"
  if (step.includes("验收") && !step.includes("迭代"))
    return "阶段三：验收"
  if (step.startsWith("4.") || step.includes("上库") || step.includes("检视"))
    return "阶段四：上库"
  return "未知阶段"
}
