"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { cn } from "@/lib/utils"
import { WorkflowAnalyseTab } from "@/components/observe/WorkflowAnalyseTab"
import { SkillAuditTab } from "@/components/observe/SkillAuditTab"

/**
 * Audit 板块容器：把原 Audit（workflow 审计 v1-v4）与 skill 审计（sift 对账）收进同一主 tab，
 * 内部拆两个子 tab（受控：sub/onSubChange 由 page 持有，便于 Skills tab 的"对账 ↗"跨 tab 跳转时
 * 由 page 直接 setSub("skill") + setSelected(skill) 一气呵成，无需 effect/外部 store）。
 * WorkflowAnalyseTab 自身状态（audit-job 模块级 + localStorage）已能跨 unmount 恢复。
 */
type SubTab = "workflow" | "skill"

interface Props {
  taskId: string
  framework?: string
  skillEvents: { skillName: string; eventType: string }[]
  /** 主 agent 编排 对账目标是否可用（session 首条 user turn 达阈值）。 */
  hasMainAgentWorkflow?: boolean
  /** 主 agent 编排 的真名（扫盘反查的 identifier）；未取/回退时由 SkillAuditTab 用合成名。 */
  mainAgentWorkflowName?: string | null
  onJumpToTurn?: (turn: number) => void
  sub: SubTab
  onSubChange: (s: SubTab) => void
  /** Skill 子 tab 当前选中（受控 {name, kind}，由 page 持有），转发给 SkillAuditTab。 */
  skillSelected: { name: string; kind: "skill" | "agent" | "root" | "llm-root" } | null
  onSkillSelectedChange: (s: { name: string; kind: "skill" | "agent" | "root" | "llm-root" } | null) => void
}

export function AuditBoardTab({ taskId, framework, skillEvents, hasMainAgentWorkflow, mainAgentWorkflowName, onJumpToTurn, sub, onSubChange, skillSelected, onSkillSelectedChange }: Props) {
  return (
    <div className="h-full flex flex-col min-h-0">
      <div className="shrink-0 flex justify-center border-b py-2">
        <div className="inline-flex items-center rounded-full bg-muted p-0.5 text-xs font-medium">
          {(["workflow", "skill"] as const).map(k => (
            <button
              key={k}
              onClick={() => onSubChange(k)}
              className={cn(
                "px-5 py-1 rounded-full transition-all",
                sub === k
                  ? "bg-background shadow-sm text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {k === "workflow" ? "Workflow" : "Skill"}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 min-h-0 flex flex-col">
        {sub === "workflow" ? (
          <WorkflowAnalyseTab taskId={taskId} framework={framework} onJumpToTurn={onJumpToTurn} />
        ) : (
          <SkillAuditTab
            taskId={taskId}
            framework={framework}
            skillEvents={skillEvents}
            hasMainAgentWorkflow={hasMainAgentWorkflow}
            mainAgentWorkflowName={mainAgentWorkflowName}
            selected={skillSelected}
            onSelectedChange={onSkillSelectedChange}
            onJumpToTurn={onJumpToTurn}
          />
        )}
      </div>
    </div>
  )
}
