"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Card, CardContent } from "@/components/ui/card"
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import {
  LayoutGridIcon,
  CheckCircle2Icon,
  ArrowDownCircleIcon,
  SparklesIcon,
  CircleDashedIcon,
  HelpCircleIcon,
  Loader2Icon,
  AlertCircleIcon,
} from "lucide-react"
import { useSkillCoverage, type SkillCoverageResponse } from "@/components/observe/use-skill-coverage"
import type { CoverageItem } from "@/lib/skill-coverage"

interface SkillCoverageCardProps {
  taskId: string
  framework?: string
}

const STATUS_META = {
  invoked: {
    icon: CheckCircle2Icon,
    label: "已调用",
    ring: "#10b981",
    chip: "border-emerald-400/50 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
  },
  loaded: {
    icon: ArrowDownCircleIcon,
    label: "仅加载",
    ring: "#3b82f6",
    chip: "border-blue-400/50 bg-blue-500/10 text-blue-700 dark:text-blue-300",
  },
  dispatched: {
    icon: SparklesIcon,
    label: "子代理",
    ring: "#8b5cf6",
    chip: "border-purple-400/50 bg-purple-500/10 text-purple-700 dark:text-purple-300",
  },
  unused: {
    icon: CircleDashedIcon,
    label: "未使用",
    ring: "#94a3b8",
    chip: "border-slate-300/60 bg-slate-500/5 text-slate-500 dark:text-slate-400",
  },
  extra: {
    icon: HelpCircleIcon,
    label: "全集外",
    ring: "#f97316",
    chip: "border-orange-400/50 bg-orange-500/10 text-orange-700 dark:text-orange-300",
  },
} as const

function coverageRing(data: SkillCoverageResponse) {
  const size = 104
  const stroke = 10
  const r = (size - stroke) / 2
  const c = size / 2
  const circ = 2 * Math.PI * r
  const { stats } = data
  const total = stats.availableTotal
  const segments = total > 0
    ? [
        { n: stats.unused, color: "#e2e8f0" },
        { n: stats.invoked, color: STATUS_META.invoked.ring },
        { n: stats.loaded, color: STATUS_META.loaded.ring },
        { n: stats.dispatched, color: STATUS_META.dispatched.ring },
      ]
    : []
  let offset = 0
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0">
      <circle cx={c} cy={c} r={r} fill="none" stroke="#e2e8f0" strokeWidth={stroke} className="dark:opacity-25" />
      {segments.map((seg, i) => {
        if (seg.n <= 0) return null
        const len = (seg.n / total) * circ
        const el = (
          <circle
            key={i}
            cx={c}
            cy={c}
            r={r}
            fill="none"
            stroke={seg.color}
            strokeWidth={stroke}
            strokeDasharray={`${Math.max(len - 2, 0.1)} ${circ}`}
            strokeDashoffset={-offset}
            strokeLinecap="round"
            transform={`rotate(-90 ${c} ${c})`}
          />
        )
        offset += len
        return el
      })}
      <text x={c} y={c - 3} textAnchor="middle" className="fill-foreground text-[17px] font-semibold tabular-nums">
        {stats.unused}
      </text>
      <text x={c} y={c + 14} textAnchor="middle" className="fill-muted-foreground text-[10px]">
        / {total} 未使用
      </text>
    </svg>
  )
}

function chipTooltipBody(item: CoverageItem): string {
  const parts: string[] = [STATUS_META[item.status].label]
  if (item.invokeCount > 0) parts.push(`调用 ${item.invokeCount} 次`)
  if (item.loadCount > 0) parts.push(`加载 ${item.loadCount} 次`)
  if (item.dispatchCount > 0) parts.push(`派发 ${item.dispatchCount} 次`)
  if (item.description) parts.push(`\n${item.description}`)
  if (item.origin) parts.push(`\n来源: ${item.origin}`)
  return parts.join(" · ")
}

export function SkillCoverageCard({ taskId, framework }: SkillCoverageCardProps) {
  const { data, loading, error } = useSkillCoverage(taskId, framework)

  if (loading) {
    return (
      <Card size="sm">
        <CardContent className="py-3 flex items-center gap-2 text-xs text-muted-foreground">
          <Loader2Icon className="size-3.5 animate-spin" />
          正在统计 Skill 覆盖度…
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card size="sm">
        <CardContent className="py-3 flex items-center gap-2 text-xs text-destructive">
          <AlertCircleIcon className="size-3.5" />
          覆盖度获取失败：{error}
        </CardContent>
      </Card>
    )
  }

  if (!data || !data.hasAvailableList || data.items.length === 0) return null

  return (
    <Card size="sm">
      <CardContent className="py-3 space-y-3">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span className="inline-flex items-center gap-1.5 text-xs font-semibold">
            <LayoutGridIcon className="size-3.5" />
            Skill 覆盖度
          </span>
          <span className="text-xs text-muted-foreground">
            proxy 捕获的系统提示注入了 {data.stats.availableTotal} 个可用 skills，其中哪些始终未使用（灰）—— 已使用的按 调用 / 仅加载 / 子代理派发 区分
          </span>
        </div>

        <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
          <div className="flex items-center gap-3">
            {coverageRing(data)}
            <div className="flex flex-col gap-1 text-[11px]">
              {(["unused", "invoked", "loaded", "dispatched", "extra"] as const)
                .filter(k => k === "extra" ? data.stats.extra > 0 : true)
                .map(k => {
                  const meta = STATUS_META[k]
                  const n = k === "extra" ? data.stats.extra
                    : k === "invoked" ? data.stats.invoked
                    : k === "loaded" ? data.stats.loaded
                    : k === "dispatched" ? data.stats.dispatched
                    : data.stats.unused
                  return (
                    <span key={k} className={cn("inline-flex items-center gap-1.5 tabular-nums", k === "unused" && "font-semibold")}>
                      <span className="size-2 rounded-full shrink-0" style={{ background: meta.ring }} />
                      <span className="text-muted-foreground">{meta.label}</span>
                      <span className={cn(k === "unused" && "text-foreground")}>{n}</span>
                    </span>
                  )
                })}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap gap-1.5">
          {data.items.map(item => {
            const meta = STATUS_META[item.status]
            const Icon = meta.icon
            return (
              <Tooltip key={`${item.status}-${item.name}`}>
                <TooltipTrigger
                  render={
                    <span
                      className={cn(
                        "inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-[11px] font-medium max-w-[18rem] cursor-default",
                        meta.chip,
                        item.status === "unused" && "opacity-65"
                      )}
                    >
                      <Icon className="size-3 shrink-0" />
                      <span className="truncate">{item.name}</span>
                      {item.invokeCount > 1 && (
                        <span className="tabular-nums opacity-70">×{item.invokeCount}</span>
                      )}
                    </span>
                  }
                  delay={0}
                  closeDelay={0}
                />
                <TooltipContent side="top" className="max-w-[24rem] whitespace-pre-wrap">
                  <span className="font-semibold">{item.name}</span>
                  {"\n"}
                  {chipTooltipBody(item)}
                </TooltipContent>
              </Tooltip>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
