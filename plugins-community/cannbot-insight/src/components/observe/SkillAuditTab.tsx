"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useEffect, useMemo, useState, useRef } from "react"
import { useSyncExternalStore } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { deriveAuditableTargets } from "@/lib/sift-audit"
import { MAIN_AGENT_WORKFLOW_NAME } from "@/lib/sift-audit"
import { AuditReportView } from "@/components/observe/skill-audit/AuditReportView"
import {
  skillAuditKey,
  subscribeSkillAudit,
  getSkillAuditSnapshot,
  getSkillAuditServerSnapshot,
  startSkillAudit,
  hydrateSkillAuditFromStorage,
  type AuditKind,
} from "@/lib/skill-audit-job"

/**
 * Skill audit 子 tab：列出本 session 可对账的声明单元——skill（invoke/use）与 agent（dispatch），
 * 跑 sift audit（不 re-run）。running/result/error 由模块级 skill-audit-job store 持有
 * （cross-tab resume：切到别的 tab 组件 unmount 后 fetch 继续在后台、状态保留；切回来
 * useSyncExternalStore 读回 running/result）。skill 正文从 session 恢复，agent .md 从本地扫描。
 */
interface SelEntry {
  name: string
  kind: AuditKind
}

interface Props {
  taskId: string
  framework?: string
  skillEvents: { skillName: string; eventType: string }[]
  /** 主 agent 编排 对账目标是否可用（STATE.md 可恢复 → 合成 root 目标）。 */
  hasMainAgentWorkflow?: boolean
  /** 主 agent 编排 的真名（扫盘反查的 identifier）；未取/回退时用合成名 MAIN_AGENT_WORKFLOW_NAME。 */
  mainAgentWorkflowName?: string | null
  selected: SelEntry | null
  onSelectedChange: (s: SelEntry | null) => void
  /** 点击 finding 的 turn_refs #N → 跳转到 Turns tab 对应对话 turn。 */
  onJumpToTurn?: (turn: number) => void
}

/**
 * 合成的"主 agent 编排"目标：root kind，声明从 STATE.md（运行时任务清单）恢复。
 * 显示名用真名（扫盘反查，如 ops-registry-invoke-glacier）；未取到回退 MAIN_AGENT_WORKFLOW_NAME。
 * 内部 key 用真名/回退名（与 SkillDetail 对账按钮传的 name 一致），路由按 kind=root 忽略 skillName。
 */
const MAIN_AGENT_WORKFLOW_TARGET: SelEntry = { name: MAIN_AGENT_WORKFLOW_NAME, kind: "root" }
const LLM_WORKFLOW_TARGET: SelEntry = { name: `${MAIN_AGENT_WORKFLOW_NAME}（LLM）`, kind: "llm-root" }

export function SkillAuditTab({ taskId, framework, skillEvents, hasMainAgentWorkflow, mainAgentWorkflowName, selected, onSelectedChange, onJumpToTurn }: Props) {
  const targets = useMemo(() => deriveAuditableTargets(skillEvents), [skillEvents])
  // 主 agent 编排 显示名：真名（扫盘）→ 回退合成名。对账按钮传此 name，路由 kind=root 忽略。
  const workflowName = mainAgentWorkflowName || MAIN_AGENT_WORKFLOW_NAME
  const entries = useMemo(
    () => {
      const skillEntries = targets.flatMap(t => t.kinds.map(k => ({ name: t.name, kind: k } as SelEntry)))
      // 主 agent 编排 目标置顶（独立于 skillEvents——主 agent 不 invoke skill）
      // 确定性提取（root）+ LLM 提取（llm-root）并排，方便对比
      const rootEntries = hasMainAgentWorkflow
        ? [{ ...MAIN_AGENT_WORKFLOW_TARGET, name: workflowName }, LLM_WORKFLOW_TARGET]
        : [LLM_WORKFLOW_TARGET]
      return [...rootEntries, ...skillEntries]
    },
    [targets, hasMainAgentWorkflow, workflowName],
  )

  // 首次挂载/reload：把 sessionStorage 已有结果灌进 store（store 已有 live state 则跳过）
  useEffect(() => {
    hydrateSkillAuditFromStorage(taskId, entries)
  }, [taskId, entries])

  return (
    <div className="h-full flex flex-col min-h-0">
      <div className="flex-1 min-h-0 flex">
        <div className="w-64 shrink-0 border-r overflow-y-auto">
          {entries.length === 0 ? (
            <div className="p-4 text-xs text-muted-foreground">
              本 session 无可对账的声明单元（无 invoke/use/dispatch 事件）。
            </div>
          ) : (
            <ul className="py-1">
              {entries.map(e => (
                <EntryRow
                  key={`${e.kind}:${e.name}`}
                  taskId={taskId}
                  entry={e}
                  isActive={!!selected && selected.name === e.name && selected.kind === e.kind}
                  onSelect={() => onSelectedChange(e)}
                />
              ))}
            </ul>
          )}
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto p-3">
          {selected ? (
            <EntryPanel taskId={taskId} framework={framework} entry={selected} onJumpToTurn={onJumpToTurn} />
          ) : (
            <div className="h-full flex items-center justify-center text-sm text-muted-foreground">
              从左侧选一个声明单元跑对账
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function EntryRow({
  taskId,
  entry,
  isActive,
  onSelect,
}: {
  taskId: string
  entry: SelEntry
  isActive: boolean
  onSelect: () => void
}) {
  const key = skillAuditKey(taskId, entry.kind, entry.name)
  const st = useSyncExternalStore(subscribeSkillAudit, () => getSkillAuditSnapshot(key), getSkillAuditServerSnapshot)
  return (
    <li>
      <button
        onClick={onSelect}
        className={cn(
          "w-full text-left px-3 py-1.5 text-xs flex items-center gap-1.5 transition-colors",
          isActive ? "bg-accent text-accent-foreground" : "hover:bg-accent/40",
        )}
      >
        <Badge variant={entry.kind === "skill" ? "blue" : entry.kind === "root" || entry.kind === "llm-root" ? "yellow" : "purple"} className="text-[10px] px-1 py-0 h-4 shrink-0">
          {entry.kind}
        </Badge>
        <span className="truncate flex-1" title={entry.name}>{entry.name}</span>
        {st.running && <span className="text-muted-foreground shrink-0">…</span>}
        {st.result && !st.running && (
          <Badge variant={st.result.summary?.fail ? "red" : "green"} className="text-[10px] px-1 py-0 h-4 shrink-0 font-mono">
            {st.result.summary?.fail ?? 0}/{st.result.summary?.total ?? 0}
          </Badge>
        )}
      </button>
    </li>
  )
}

function EntryPanel({
  taskId,
  framework,
  entry,
  onJumpToTurn,
}: {
  taskId: string
  framework?: string
  entry: SelEntry
  onJumpToTurn?: (turn: number) => void
}) {
  const key = skillAuditKey(taskId, entry.kind, entry.name)
  const st = useSyncExternalStore(subscribeSkillAudit, () => getSkillAuditSnapshot(key), getSkillAuditServerSnapshot)
  // 运行中每秒刷新耗时计时器(Date.now 只在 effect 的 interval 回调里跑,不进 render——react-hooks/purity;
  // 不在 effect body 直接 setState,经 tick() 间接调用)
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    if (!st.running) return
    const tick = () => setElapsed(st.startedAt ? Math.floor((Date.now() - st.startedAt) / 1000) : 0)
    tick()
    const t = setInterval(tick, 1000)
    return () => clearInterval(t)
  }, [st.running, st.startedAt])

  // 进度条平滑插值：收到新目标后，ease-out 动画从当前值滑到目标值。
  // ease-out（快开始慢结束）比线性更自然；目标值只增不减（防退回）。
  // 用 st.percent 初始化 ref/state（修 remount 后退回 0% 的 bug）。
  const animPctRef = useRef(st.percent)
  const [animPct, setAnimPct] = useState(st.percent)
  const animTargetRef = useRef(st.percent)
  const animRef = useRef({ from: 0, to: 0, start: 0 })
  useEffect(() => {
    if (!st.running) return
    // 目标只增不减（防退回）
    const target = Math.max(st.percent, animTargetRef.current)
    animTargetRef.current = target
    animRef.current = { from: animPctRef.current, to: target, start: Date.now() }
    const timer = setInterval(() => {
      const elapsedMs = Date.now() - animRef.current.start
      const duration = 10000 // 10s 到达目标
      const t = Math.min(elapsedMs / duration, 1)
      // ease-out cubic：快开始慢结束
      const eased = 1 - Math.pow(1 - t, 3)
      const val = animRef.current.from + (animRef.current.to - animRef.current.from) * eased
      animPctRef.current = val
      setAnimPct(val)
    }, 100)
    return () => clearInterval(timer)
  }, [st.percent, st.running])

  const start = () => startSkillAudit({ taskId, kind: entry.kind, name: entry.name, framework })

  if (st.running) {
    const smoothPct = Math.max(0, Math.min(100, animPct))
    const elapsedStr = elapsed >= 60 ? `${Math.floor(elapsed / 60)}m${elapsed % 60}s` : `${elapsed}s`
    return (
      <div className="h-full flex flex-col items-center justify-center gap-3 text-sm text-muted-foreground px-6">
        <div className="w-full max-w-md">
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="animate-pulse">正在跑 sift audit…</span>
            <span className="tabular-nums font-medium text-foreground">{Math.round(smoothPct)}% · {elapsedStr}</span>
          </div>
          <div className="w-full h-2 rounded-full bg-muted overflow-hidden">
            <div
              className="h-full rounded-full bg-primary"
              style={{ width: `${smoothPct}%` }}
            />
          </div>
        </div>
        <div className="text-xs">审 <span className="font-medium text-foreground">{entry.name}</span>（{entry.kind}）的真实执行 vs 声明</div>
        {st.progress && (
          <div className="w-full max-w-md text-[11px] text-muted-foreground/80 line-clamp-3 font-mono break-all">
            {st.progress}
          </div>
        )}
        <div className="text-xs">可切到别的 tab，对账在后台继续，切回可看结果。</div>
      </div>
    )
  }
  if (st.error) {
    return (
      <div className="space-y-2">
        <div className="text-sm font-medium text-red-600 dark:text-red-400">失败</div>
        <pre className="text-xs text-red-600 dark:text-red-400 whitespace-pre-wrap break-words">{st.error}</pre>
        <Button size="sm" variant="outline" onClick={start}>重跑</Button>
      </div>
    )
  }
  if (st.result) {
    return <AuditReportView result={st.result} onRerun={start} onJumpToTurn={onJumpToTurn} />
  }
  return (
    <div className="h-full flex flex-col items-center justify-center gap-3 text-sm text-muted-foreground">
      <div>
        对 <span className="font-medium text-foreground">{entry.name}</span>（{entry.kind}）跑 sift 对账
        <div className="text-xs mt-1">
          {entry.kind === "skill"
            ? "不 re-run，对账真实执行 vs SKILL.md 声明（正文从 session 恢复）"
            : entry.kind === "root"
              ? "不 re-run，--kind root 切顶层主 agent 作用域，审主 agent 编排 vs workflow skill 编排规程（确定性提取）"
              : entry.kind === "llm-root"
                ? "不 re-run，--kind root + LLM 提取编排规程（claude CLI 读 session 前 few turns 总结），审主 agent 编排 vs LLM 提取的规程"
                : "不 re-run，对账真实执行 vs agent .md 声明（.md 从本地 AGENTS_SCAN_ROOT 扫描）"}
        </div>
      </div>
      <Button size="sm" onClick={start}>对账</Button>
    </div>
  )
}
