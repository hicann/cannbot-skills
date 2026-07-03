"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useMemo, useState, useCallback, useRef } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ZoomInIcon, ZoomOutIcon, RotateCcwIcon } from "lucide-react"

interface AgentItem {
  executionId: string
  agentName: string | null
  agentSessionId: string | null
  isSubagent: boolean
  parentExecutionId: string | null
  tokens: number
  maxSingleCallTokens: number
  cost: number
  toolCallCount: number
  skillLoadCount: number
  model: string | null
  createdAt: string
  latencyMs: number
  firstPrompt: string | null
}

interface BridgeItem {
  bridgeId: string
  dispatchExecutionId: string
  dispatchTurnId: string | null
  dispatchToolCallId: string | null
  dispatchContent: string | null
  dispatchTimestamp: string | null
  responseExecutionId: string | null
  responseTurnId: string | null
  responseContent: string | null
  responseTimestamp: string | null
  subagentSessionId: string | null
  subagentType: string | null
  subagentName: string | null
  agentName: string | null
  status: string
  subagentTokens: number
  subagentLatencyMs: number
}

interface AgentRelationGraphProps {
  agents: AgentItem[]
  bridges: BridgeItem[]
  onViewTurns?: (agentSessionId: string | null) => void
}

const NODE_W = 168
const NODE_H = 64
const V_CHILD_GAP = 18
const H_WAVE_GAP = 22
const H_FORK_DROP = 14
const H_PARENT_FORK = 14
const H_SEQ_GAP = 14
const PAD = 24
const R = 8
const FORK_STROKE = 3.5
const MINI_BAR_H = 4
const MINI_BAR_TRACK_H = 6
const MINI_BAR_PAD_X = 8

const AGENT_COLORS = [
  "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
  "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
  "#14b8a6", "#e11d48", "#a855f7", "#22c55e", "#d946ef",
]

const agentColorCache = new Map<string, string>()
let agentColorIdx = 0

function getAgentColor(name: string | null): string {
  const key = name ?? "root"
  if (agentColorCache.has(key)) return agentColorCache.get(key)!
  agentColorCache.set(key, AGENT_COLORS[agentColorIdx % AGENT_COLORS.length])
  agentColorIdx++
  return agentColorCache.get(key)!
}

function fmtMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  return `${(ms / 60000).toFixed(1)}min`
}

function fmtTok(n: number): string {
  if (n === 0) return ""
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return `${n}`
}

const STATUS_BORDER: Record<string, string> = {
  completed: "#10b981",
  failed: "#ef4444",
  running: "#f59e0b",
  dispatched: "#6b7280",
  timeout: "#f97316",
}

const STATUS_ICON: Record<string, string> = {
  completed: "",
  failed: "x",
  running: "~",
  dispatched: ">",
  timeout: "!",
}

interface Wave {
  waveIndex: number
  childIds: string[]
  startMs: number
  endMs: number
  isConcurrent: boolean
}

interface NodeInfo {
  id: string
  label: string
  color: string
  agent: AgentItem
  bridge: BridgeItem | null
  depth: number
  waves: Wave[]
  childIds: string[]
}

interface Pos { x: number; y: number }

interface ForkBarInfo {
  parentId: string
  waveIndex: number
  x: number
  centerY: number
  topY: number
  bottomY: number
  color: string
  isConcurrent: boolean
  waveLabel: string
}

interface WaveArrow {
  parentId: string
  fromWave: number
  toWave: number
  fromX: number
  toX: number
  centerY: number
  color: string
}

interface TooltipData {
  x: number
  y: number
  nodeLabel: string
  maxSingleCallTokens: number
  latencyMs: number
  toolCallCount: number
  model: string | null
  depth: number
  status: string
  dispatchContent: string | null
  waveInfo: string | null
  startMs: number
  endMs: number
}

function computeWaves(
  childIds: string[],
  nodeMap: Map<string, NodeInfo>
): Wave[] {
  if (childIds.length === 0) return []

  const windows: Array<{ id: string; startMs: number; endMs: number }> = []
  for (const cid of childIds) {
    const node = nodeMap.get(cid)
    if (!node) continue
    const b = node.bridge
    const startMs = b?.dispatchTimestamp
      ? new Date(b.dispatchTimestamp).getTime()
      : new Date(node.agent.createdAt).getTime()
    const endMs = startMs + node.agent.latencyMs
    windows.push({ id: cid, startMs, endMs })
  }

  windows.sort((a, b) => a.startMs - b.startMs)

  const waves: Wave[] = []
  let group: Array<{ id: string; startMs: number; endMs: number }> = []
  let groupEnd = 0

  for (const w of windows) {
    if (group.length === 0) {
      group = [w]
      groupEnd = w.endMs
    } else if (w.startMs <= groupEnd) {
      group.push(w)
      groupEnd = Math.max(groupEnd, w.endMs)
    } else {
      waves.push({
        waveIndex: waves.length + 1,
        childIds: group.map(g => g.id),
        startMs: Math.min(...group.map(g => g.startMs)),
        endMs: groupEnd,
        isConcurrent: group.length > 1,
      })
      group = [w]
      groupEnd = w.endMs
    }
  }

  if (group.length > 0) {
    waves.push({
      waveIndex: waves.length + 1,
      childIds: group.map(g => g.id),
      startMs: Math.min(...group.map(g => g.startMs)),
      endMs: groupEnd,
      isConcurrent: group.length > 1,
    })
  }

  return waves
}

export function AgentRelationGraph({ agents, bridges, onViewTurns }: AgentRelationGraphProps) {
  const [tooltip, setTooltip] = useState<TooltipData | null>(null)
  const [zoom, setZoom] = useState(1)
  const containerRef = useRef<HTMLDivElement>(null)
  const ZOOM_STEP = 0.15
  const ZOOM_MIN = 0.3
  const ZOOM_MAX = 3

  const layout = useMemo(() => {
    const bridgeBySession = new Map<string, BridgeItem>()
    for (const b of bridges) {
      if (b.subagentSessionId) bridgeBySession.set(b.subagentSessionId, b)
    }

    const sortedAgents = [...agents].sort(
      (a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
    )

    const globalStartMs = sortedAgents.length > 0
      ? Math.min(...sortedAgents.map(a => new Date(a.createdAt).getTime()))
      : 0
    const globalEndMs = sortedAgents.length > 0
      ? Math.max(...sortedAgents.map(a => new Date(a.createdAt).getTime() + a.latencyMs))
      : globalStartMs + 1
    const globalDuration = globalEndMs - globalStartMs || 1

    const nodeMap = new Map<string, NodeInfo>()
    const childMap = new Map<string, string[]>()
    const agentById = new Map(sortedAgents.map(a => [a.executionId, a]))
    const depthMap = new Map<string, number>()

    function computeDepth(execId: string): number {
      if (depthMap.has(execId)) return depthMap.get(execId)!
      const a = agentById.get(execId)
      if (!a || !a.isSubagent || !a.parentExecutionId) {
        depthMap.set(execId, 0)
        return 0
      }
      const d = computeDepth(a.parentExecutionId) + 1
      depthMap.set(execId, d)
      return d
    }

    for (const agent of sortedAgents) {
      const depth = computeDepth(agent.executionId)
      const label = agent.agentName ?? (agent.isSubagent ? "subagent" : "root")
      const color = getAgentColor(agent.agentName)
      const bridge = agent.isSubagent && agent.agentSessionId
        ? bridgeBySession.get(agent.agentSessionId) ?? null
        : null

      nodeMap.set(agent.executionId, {
        id: agent.executionId,
        label,
        color,
        agent,
        bridge,
        depth,
        waves: [],
        childIds: [],
      })

      if (agent.isSubagent && agent.parentExecutionId) {
        const ch = childMap.get(agent.parentExecutionId) ?? []
        ch.push(agent.executionId)
        childMap.set(agent.parentExecutionId, ch)
      }
    }

    for (const [parentId, cids] of childMap) {
      const parent = nodeMap.get(parentId)
      if (parent) {
        parent.childIds = cids
        parent.waves = computeWaves(cids, nodeMap)
      }
    }

    const rootIds = sortedAgents.filter(a => !a.isSubagent || !a.parentExecutionId).map(a => a.executionId)

    const subtreeW = new Map<string, number>()
    const subtreeH = new Map<string, number>()

    function sw(id: string): number {
      if (subtreeW.has(id)) return subtreeW.get(id)!
      const node = nodeMap.get(id)
      if (!node || node.waves.length === 0) {
        subtreeW.set(id, NODE_W)
        return NODE_W
      }
      let totalW = NODE_W
      for (let wi = 0; wi < node.waves.length; wi++) {
        const wave = node.waves[wi]
        const childWs = wave.childIds.map(cid => sw(cid))
        if (wave.isConcurrent) {
          totalW += H_PARENT_FORK + H_FORK_DROP + Math.max(...childWs)
        } else {
          totalW += H_SEQ_GAP + childWs[0]
        }
        if (wi < node.waves.length - 1) totalW += H_WAVE_GAP
      }
      subtreeW.set(id, totalW)
      return totalW
    }

    function sh(id: string): number {
      if (subtreeH.has(id)) return subtreeH.get(id)!
      const node = nodeMap.get(id)
      if (!node || node.waves.length === 0) {
        subtreeH.set(id, NODE_H)
        return NODE_H
      }
      let maxH = NODE_H
      for (const wave of node.waves) {
        const childHs = wave.childIds.map(cid => sh(cid))
        const waveH = wave.isConcurrent
          ? childHs.reduce((s, ch) => s + ch, 0) + (childHs.length - 1) * V_CHILD_GAP
          : childHs[0]
        maxH = Math.max(maxH, waveH)
      }
      subtreeH.set(id, maxH)
      return maxH
    }

    for (const id of nodeMap.keys()) { sw(id); sh(id) }

    const positions = new Map<string, Pos>()
    const forkBars: ForkBarInfo[] = []
    const waveArrows: WaveArrow[] = []

    function layoutNode(id: string, startX: number, centerY: number) {
      const node = nodeMap.get(id)
      if (!node) return

      positions.set(id, {
        x: startX,
        y: centerY - NODE_H / 2,
      })

      if (node.waves.length === 0) return

      let currentX = startX + NODE_W

      for (let wi = 0; wi < node.waves.length; wi++) {
        const wave = node.waves[wi]
        const childHs = wave.childIds.map(cid => sh(cid))

        if (wave.isConcurrent) {
          const forkX = currentX + H_PARENT_FORK
          const childStartX = forkX + H_FORK_DROP

          const totalChildH = childHs.reduce((s, ch) => s + ch, 0) + (childHs.length - 1) * V_CHILD_GAP
          const startCY = centerY - totalChildH / 2

          const childCenterYs: number[] = []
          let cy = startCY
          for (let ci = 0; ci < wave.childIds.length; ci++) {
            const childCY = cy + childHs[ci] / 2
            childCenterYs.push(childCY)
            layoutNode(wave.childIds[ci], childStartX, childCY)
            cy += childHs[ci] + V_CHILD_GAP
          }

          forkBars.push({
            parentId: id,
            waveIndex: wave.waveIndex,
            x: forkX,
            centerY,
            topY: childCenterYs[0],
            bottomY: childCenterYs[childCenterYs.length - 1],
            color: node.color,
            isConcurrent: true,
            waveLabel: `Wave ${wave.waveIndex}`,
          })

          currentX = childStartX + Math.max(...wave.childIds.map(cid => sw(cid)))
        } else {
          const childId = wave.childIds[0]
          const childStartX = currentX + H_SEQ_GAP

          forkBars.push({
            parentId: id,
            waveIndex: wave.waveIndex,
            x: childStartX - 2,
            centerY,
            topY: centerY,
            bottomY: centerY,
            color: node.color,
            isConcurrent: false,
            waveLabel: `#${wave.waveIndex}`,
          })

          layoutNode(childId, childStartX, centerY)
          currentX = childStartX + sw(childId)
        }

        if (wi < node.waves.length - 1) {
          waveArrows.push({
            parentId: id,
            fromWave: wave.waveIndex,
            toWave: node.waves[wi + 1].waveIndex,
            fromX: currentX,
            toX: currentX + H_WAVE_GAP,
            centerY,
            color: node.color,
          })
          currentX += H_WAVE_GAP
        }
      }
    }

    let totalRootH = 0
    let maxRootW = 0
    for (const rid of rootIds) {
      totalRootH += sh(rid)
      maxRootW = Math.max(maxRootW, sw(rid))
    }
    totalRootH += (rootIds.length - 1) * V_CHILD_GAP

    const svgW = maxRootW + PAD * 2
    const svgH = Math.max(totalRootH + PAD * 2, NODE_H + PAD * 2)

    if (rootIds.length > 1) {
      let cy = PAD + totalRootH / 2 - (rootIds.reduce((s, rid) => s + sh(rid), 0) + (rootIds.length - 1) * V_CHILD_GAP) / 2
      for (const rid of rootIds) {
        layoutNode(rid, PAD, cy + sh(rid) / 2)
        cy += sh(rid) + V_CHILD_GAP
      }
    } else if (rootIds.length === 1) {
      layoutNode(rootIds[0], PAD, PAD + sh(rootIds[0]) / 2)
    }

    const miniBars = new Map<string, { pctStart: number; pctWidth: number }>()
    for (const [id, node] of nodeMap) {
      const a = node.agent
      const startMs = new Date(a.createdAt).getTime()
      miniBars.set(id, {
        pctStart: Math.max(0, ((startMs - globalStartMs) / globalDuration) * 100),
        pctWidth: Math.max(3, (a.latencyMs / globalDuration) * 100),
      })
    }

    return {
      positions,
      forkBars,
      waveArrows,
      nodeMap,
      miniBars,
      svgW,
      svgH,
      globalStartMs,
      globalEndMs,
      globalDuration,
    }
  }, [agents, bridges])

  const handleHover = useCallback((e: React.MouseEvent, node: NodeInfo) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
    const parent = node.agent.parentExecutionId ? layout.nodeMap.get(node.agent.parentExecutionId) : null
    const waveInfo = parent
      ? parent.waves.find(wave => wave.childIds.includes(node.id))
      : null

    setTooltip({
      x: rect.left + rect.width / 2,
      y: rect.top,
      nodeLabel: node.label,
      maxSingleCallTokens: node.agent.maxSingleCallTokens,
      latencyMs: node.agent.latencyMs,
      toolCallCount: node.agent.toolCallCount,
      model: node.agent.model,
      depth: node.depth,
      status: node.bridge?.status ?? (node.agent.isSubagent ? "running" : "completed"),
      dispatchContent: node.bridge?.dispatchContent ?? node.agent.firstPrompt ?? null,
      waveInfo: waveInfo
        ? `${waveInfo.isConcurrent ? "Concurrent" : "Sequential"} Wave ${waveInfo.waveIndex} (${waveInfo.childIds.length} agents)`
        : null,
      startMs: new Date(node.agent.createdAt).getTime(),
      endMs: new Date(node.agent.createdAt).getTime() + node.agent.latencyMs,
    })
  }, [layout.nodeMap])

  const handleLeave = useCallback(() => setTooltip(null), [])

  if (agents.length === 0) {
    return (
      <Card size="sm">
        <CardHeader><CardTitle>Agent Call Topology (0)</CardTitle></CardHeader>
        <CardContent><div className="text-sm text-muted-foreground">No agents found</div></CardContent>
      </Card>
    )
  }

  const { positions, forkBars, waveArrows, nodeMap, miniBars, svgW, svgH } = layout

  return (
    <Card size="sm">
      <CardHeader className="flex-row items-center justify-between space-y-0">
        <CardTitle>Agent Call Topology ({agents.length})</CardTitle>
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={() => setZoom(z => Math.max(ZOOM_MIN, z - ZOOM_STEP))} title="缩小">
            <ZoomOutIcon className="size-3.5" />
          </Button>
          <span className="text-xs text-muted-foreground tabular-nums w-[36px] text-center">{Math.round(zoom * 100)}%</span>
          <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={() => setZoom(z => Math.min(ZOOM_MAX, z + ZOOM_STEP))} title="放大">
            <ZoomInIcon className="size-3.5" />
          </Button>
          <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={() => setZoom(1)} title="重置">
            <RotateCcwIcon className="size-3.5" />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div
          ref={containerRef}
          className="overflow-auto border rounded-md"
          style={{ maxWidth: "100%", maxHeight: 400 }}
          onWheel={(e) => {
            if (e.ctrlKey || e.metaKey) {
              e.preventDefault()
              const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP
              setZoom(z => Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, z + delta)))
            }
          }}
        >
          <svg
            width={svgW * zoom}
            height={svgH * zoom}
            viewBox={`0 0 ${svgW} ${svgH}`}
            className="block mx-auto"
          >
            <defs>
              <marker id="arr-seq" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
                <polygon points="0 0, 6 2, 0 4" fill="#9ca3af" />
              </marker>
            </defs>

            {waveArrows.map((wa, i) => (
              <g key={`wa-${i}`}>
                <line
                  x1={wa.fromX}
                  y1={wa.centerY}
                  x2={wa.toX}
                  y2={wa.centerY}
                  stroke="#9ca3af"
                  strokeWidth={1.5}
                  strokeDasharray="4 2"
                  markerEnd="url(#arr-seq)"
                />
                <text
                  x={(wa.fromX + wa.toX) / 2}
                  y={wa.centerY - 8}
                  fontSize={8}
                  fill="#9ca3af"
                  textAnchor="middle"
                >
                  → W{wa.toWave}
                </text>
              </g>
            ))}

            {forkBars.map((fb, i) => {
              const parentPos = positions.get(fb.parentId)
              if (!parentPos) return null

              const parentRX = parentPos.x + NODE_W
              const parentCY = parentPos.y + NODE_H / 2

              if (fb.isConcurrent) {
                return (
                  <g key={`fb-${i}`}>
                    <line
                      x1={parentRX}
                      y1={parentCY}
                      x2={fb.x}
                      y2={fb.centerY}
                      stroke={fb.color}
                      strokeWidth={2}
                      opacity={0.7}
                    />
                    <line
                      x1={fb.x}
                      y1={fb.topY}
                      x2={fb.x}
                      y2={fb.bottomY}
                      stroke={fb.color}
                      strokeWidth={FORK_STROKE}
                      strokeLinecap="round"
                      opacity={0.85}
                    />
                    <circle cx={fb.x} cy={fb.centerY} r={3} fill={fb.color} opacity={0.85} />

                    {(() => {
                      const parentNode = nodeMap.get(fb.parentId)
                      const wave = parentNode?.waves.find(w => w.waveIndex === fb.waveIndex)
                      if (!wave) return null
                      return wave.childIds.map((cid) => {
                        const childPos = positions.get(cid)
                        if (!childPos) return null
                        const childCY = childPos.y + NODE_H / 2
                        return (
                          <line
                            key={`stem-${cid}`}
                            x1={fb.x + 1}
                            y1={childCY}
                            x2={childPos.x}
                            y2={childCY}
                            stroke={fb.color}
                            strokeWidth={1.5}
                            opacity={0.6}
                          />
                        )
                      })
                    })()}

                  </g>
                )
              } else {
                const childId = (() => {
                  const parentNode = nodeMap.get(fb.parentId)
                  const wave = parentNode?.waves.find(w => w.waveIndex === fb.waveIndex)
                  return wave?.childIds[0]
                })()
                if (!childId) return null
                const childPos = positions.get(childId)
                if (!childPos) return null
                const childCY = childPos.y + NODE_H / 2

                return (
                  <g key={`fb-${i}`}>
                    <line
                      x1={parentRX}
                      y1={parentCY}
                      x2={childPos.x}
                      y2={childCY}
                      stroke={fb.color}
                      strokeWidth={2}
                      opacity={0.6}
                    />
                  </g>
                )
              }
            })}

            {[...nodeMap.values()].map(node => {
              const pos = positions.get(node.id)
              if (!pos) return null

              const mb = miniBars.get(node.id)
              const statusIcon = STATUS_ICON[node.bridge?.status ?? "completed"] ?? ""
              const isError = node.bridge?.status === "failed"
              const innerW = NODE_W - MINI_BAR_PAD_X * 2

              return (
                <g
                  key={node.id}
                  className="cursor-pointer"
                  onClick={() => onViewTurns?.(node.agent.isSubagent ? node.agent.agentSessionId : null)}
                  onMouseEnter={(e) => handleHover(e, node)}
                  onMouseLeave={handleLeave}
                >
                  <rect
                    x={pos.x}
                    y={pos.y}
                    width={NODE_W}
                    height={NODE_H}
                    rx={R}
                    fill={node.color + "12"}
                    stroke={isError ? "#ef4444" : node.color}
                    strokeWidth={isError ? 2.5 : 1.5}
                  />
                  {node.depth > 0 && (
                    <rect
                      x={pos.x}
                      y={pos.y}
                      width={4}
                      height={NODE_H}
                      rx={2}
                      fill={node.color}
                      opacity={0.8}
                    />
                  )}
                  <text
                    x={pos.x + (node.depth > 0 ? 10 : 8)}
                    y={pos.y + 15}
                    fontSize={11}
                    fontWeight="bold"
                    fill={isError ? "#ef4444" : node.color}
                  >
                    {statusIcon ? `${statusIcon} ` : ""}{node.label}
                  </text>
                  <text
                    x={pos.x + (node.depth > 0 ? 10 : 8)}
                    y={pos.y + 28}
                    fontSize={9}
                    fill="#6b7280"
                  >
                    {fmtTok(node.agent.maxSingleCallTokens)} token | {fmtMs(node.agent.latencyMs)}
                  </text>
                  <text
                    x={pos.x + (node.depth > 0 ? 10 : 8)}
                    y={pos.y + 40}
                    fontSize={8}
                    fill="#9ca3af"
                  >
                    {node.agent.toolCallCount} tools{node.agent.model ? ` | ${node.agent.model}` : ""}
                  </text>

                  <rect
                    x={pos.x + MINI_BAR_PAD_X}
                    y={pos.y + NODE_H - 12}
                    width={innerW}
                    height={MINI_BAR_TRACK_H}
                    rx={1.5}
                    fill="#e5e7eb20"
                    stroke="#d1d5db30"
                    strokeWidth={0.5}
                  />
                  {mb && (
                    <rect
                      x={pos.x + MINI_BAR_PAD_X + (mb.pctStart / 100) * innerW}
                      y={pos.y + NODE_H - 12 + (MINI_BAR_TRACK_H - MINI_BAR_H) / 2}
                      width={(mb.pctWidth / 100) * innerW}
                      height={MINI_BAR_H}
                      rx={1}
                      fill={node.color}
                      opacity={0.75}
                    />
                  )}

                  {(() => {
                    const parent = node.agent.parentExecutionId ? nodeMap.get(node.agent.parentExecutionId) : null
                    const wave = parent?.waves.find(w => w.childIds.includes(node.id))
                    if (!wave || !wave.isConcurrent) return null
                    const orderInWave = wave.childIds.indexOf(node.id) + 1
                    return (
                      <text
                        x={pos.x + NODE_W - 8}
                        y={pos.y + 15}
                        textAnchor="end"
                        fontSize={8}
                        fill={node.color}
                        fontWeight="bold"
                      >
                        ∥{orderInWave}
                      </text>
                    )
                  })()}
                </g>
              )
            })}
          </svg>
        </div>

        <div className="flex flex-wrap gap-3 pt-1">
          <div className="flex items-center gap-1 text-xs">
            <svg width="16" height="16" viewBox="0 0 16 16">
              <line x1="3" y1="3" x2="3" y2="13" stroke="#3b82f6" strokeWidth={3} strokeLinecap="round" />
            </svg>
            <span className="text-muted-foreground">∥ Concurrent</span>
          </div>
          <div className="flex items-center gap-1 text-xs">
            <svg width="16" height="16" viewBox="0 0 16 16">
              <line x1="3" y1="8" x2="13" y2="8" stroke="#9ca3af" strokeWidth={1.5} strokeDasharray="4 2" />
            </svg>
            <span className="text-muted-foreground">→ Sequential</span>
          </div>
          <div className="flex items-center gap-1 text-xs">
            <div style={{ height: MINI_BAR_H, width: 24, borderRadius: 1, backgroundColor: "#3b82f6" }} />
            <span className="text-muted-foreground">Time window</span>
          </div>
          {Object.entries(STATUS_BORDER).map(([status, color]) => {
            const count = bridges.filter(b => b.status === status).length
            if (count === 0) return null
            return (
              <div key={status} className="flex items-center gap-1 text-xs">
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: color }} />
                <span className="text-muted-foreground">{status} ({count})</span>
              </div>
            )
          })}
        </div>
      </CardContent>

      {tooltip && (
        <div
          className="fixed z-[9999] bg-popover border rounded-md shadow-lg px-3 py-2 text-xs whitespace-nowrap pointer-events-none"
          style={{ left: tooltip.x, top: tooltip.y - 8, transform: "translate(-50%, -100%)" }}
        >
          <div className="font-medium mb-1">{tooltip.nodeLabel}</div>
          <div className="text-muted-foreground space-y-0.5">
            <div>{fmtTok(tooltip.maxSingleCallTokens)} token | {fmtMs(tooltip.latencyMs)}</div>
            <div>{tooltip.toolCallCount} tools{tooltip.model ? ` | ${tooltip.model}` : ""}</div>
            <div>Depth: {tooltip.depth} | Status: {tooltip.status}</div>
            {tooltip.waveInfo && <div className="font-medium text-foreground/70">{tooltip.waveInfo}</div>}
            {tooltip.dispatchContent && (
              <div className="max-w-[200px] truncate">{tooltip.dispatchContent}</div>
            )}
          </div>
        </div>
      )}
    </Card>
  )
}
