"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useMemo, useState } from "react"
import { Badge } from "@/components/ui/badge"

type Severity = "high" | "medium" | "low"
export interface Problem {
  type: string
  severity: Severity
  evidence?: string
  diagnosis?: string
  suggestion?: string
}
type DimKey = "G1" | "G2" | "G3" | "G4" | "G5" | "S1" | "S2" | "S3"
type Rating = "pass" | "weak" | "fail" | "n-a"
type StaticCategory = "ambiguity" | "io-unclear" | "asymmetry" | "structure" | "reference"
interface StaticCheck {
  category: StaticCategory
  severity: Severity
  issue: string
  snippet?: string
  suggestion?: string
}
interface DimRating {
  rating: Rating
  note: string
  evidence?: string
  diagnosis?: string
  suggestion?: string
  staticChecks?: StaticCheck[]
}
interface SkillQuality {
  skill: string
  occurrences: number
  ratings: Record<DimKey, DimRating>
  summary: string
}
interface FlowNode {
  id: string
  skill: string
  step: string
  type: "invoke" | "dispatch" | "gate" | "terminal"
  turn: number
  parallel: string | null
  retryOf: string | null
  status: string
  problems: Problem[]
}
export interface Analysis {
  sessionSummary: string
  sessionMeta: Record<string, unknown>
  flow: FlowNode[]
  workflowLevelIssues: Problem[]
  optimizationPriorities: Array<{ priority: number; target: string; action: string; expectedGain: string }>
  skillQuality?: SkillQuality[]
}

const DIMS: DimKey[] = ["G1", "G2", "G3", "G4", "G5", "S1", "S2", "S3"]
const DIM_LABEL: Record<DimKey, string> = {
  G1: "G1 正确性", G2: "G2 指令遵循", G3: "G3 安全性", G4: "G4 完整性", G5: "G5 鲁棒性",
  S1: "S1 可执行性", S2: "S2 成本意识", S3: "S3 可维护性",
}
const RATING_COLOR: Record<Rating, string> = { pass: "#16a34a", weak: "#d97706", fail: "#dc2626", "n-a": "#9ca3af" }
const RATING_LABEL: Record<Rating, string> = { pass: "✓", weak: "△", fail: "✗", "n-a": "—" }
const STATIC_CAT_LABEL: Record<StaticCategory, string> = {
  ambiguity: "歧义", "io-unclear": "输入输出不明确", asymmetry: "不对称", structure: "结构", reference: "引用",
}
const DIM_DESC: Record<DimKey, { group: string; desc: string; principles?: Array<{ name: string; source: string }> }> = {
  G1: { group: "G · 任务产出质量", desc: "正确性：skill 输出是否正确达成目标。证据=评审 outcome(✅/❌)+重试周期" },
  G2: { group: "G · 任务产出质量", desc: "指令遵循：是否遵循格式/约束。证据=校验脚本失败(spec 9-stage/组装校验/checklist)" },
  G3: { group: "G · 任务产出质量", desc: "安全性：数值稳定性/溢出/资源/内存。证据=spec/DESIGN 设计层；代码层需阶段二/四" },
  G4: { group: "G · 任务产出质量", desc: "完整性：是否覆盖所有必要方面。证据=评审条款覆盖缺口" },
  G5: { group: "G · 任务产出质量", desc: "鲁棒性：边界/异常处理。证据=TEST.md 测试设计；执行验证需阶段三" },
  S1: { group: "S · skill 本身质量", desc: "可执行性：指令是否清晰具体可操作。证据=嵌入 skill 文本+子代理行为+静态分析", principles: [
    { name: "祈使语气", source: "Prefer using the imperative form" },
    { name: "示例驱动", source: "用 Input/Output 对展示" },
    { name: "理论思维", source: "Use theory of mind" },
  ] },
  S2: { group: "S · skill 本身质量", desc: "成本意识：输出是否简洁无冗余。证据=token/调用次数/冗余调用" },
  S3: { group: "S · skill 本身质量", desc: "可维护性：结构是否清晰、分段合理、易改。证据=嵌入 skill 文本", principles: [
    { name: "解释 Why", source: "explain why in lieu of heavy-handed MUSTs" },
    { name: "避免过度约束", source: "大写 ALWAYS/NEVER 是黄牌，改用推理解释" },
  ] },
}

const NODE_W = 188
const NODE_H = 54
const COL_GAP = 14
const ROW_GAP = 30
const LEFT_LABEL_W = 168
const PAD = 28
const RIGHT_RETRY_LANE = 60

const SEV_COLOR: Record<Severity, string> = {
  high: "#dc2626",
  medium: "#d97706",
  low: "#6b7280",
}
const SEV_LABEL: Record<Severity, string> = { high: "高", medium: "中", low: "低" }

function shortName(s: string): string {
  return s.replace(/^ascendc-ops-/, "").replace(/^ascendc-/, "").replace(/^ops-registry-invoke-/, "wf-")
}

interface WorkflowFlowChartProps {
  analysis: Analysis
}

export function WorkflowFlowChart({ analysis }: WorkflowFlowChartProps) {
  const [selected, setSelected] = useState<string | null>(
    analysis.flow.find(n => n.problems.length > 0)?.id ?? analysis.flow[0]?.id ?? null,
  )
  const [selectedSq, setSelectedSq] = useState<{ skill: string; dim: DimKey } | null>(null)

  const layout = useMemo(() => {
    const groups: { step: string; nodes: FlowNode[] }[] = []
    for (const n of analysis.flow) {
      const g = groups.find(x => x.step === n.step)
      if (g) g.nodes.push(n)
      else groups.push({ step: n.step, nodes: [n] })
    }
    const maxRowNodes = Math.max(...groups.map(g => g.nodes.length))
    const width = LEFT_LABEL_W + maxRowNodes * (NODE_W + COL_GAP) + RIGHT_RETRY_LANE + PAD * 2
    const pos = new Map<string, { x: number; y: number; cx: number; cy: number; rowY: number; rowH: number }>()
    const rows: { step: string; y: number; nodes: FlowNode[]; nodeXs: number[] }[] = []
    let y = PAD + 40
    for (const g of groups) {
      const startX = LEFT_LABEL_W + PAD
      const nodeXs: number[] = []
      g.nodes.forEach((n, i) => {
        const x = startX + i * (NODE_W + COL_GAP)
        nodeXs.push(x)
        pos.set(n.id, { x, y, cx: x + NODE_W / 2, cy: y + NODE_H / 2, rowY: y, rowH: NODE_H })
      })
      rows.push({ step: g.step, y, nodes: g.nodes, nodeXs })
      y += NODE_H + ROW_GAP
    }
    const height = y + PAD
    return { rows, pos, width, height, totalW: LEFT_LABEL_W + PAD + maxRowNodes * (NODE_W + COL_GAP) }
  }, [analysis])

  const selectedNode = analysis.flow.find(n => n.id === selected) ?? null

  const retryEdges = useMemo(() => {
    const edges: { from: string; to: string; key: string }[] = []
    for (const n of analysis.flow) {
      if (n.retryOf) edges.push({ from: n.id, to: n.retryOf, key: `${n.id}->${n.retryOf}` })
    }
    return edges
  }, [analysis])

  const sqBySkill = useMemo(() => {
    const m = new Map<string, SkillQuality>()
    for (const sq of analysis.skillQuality ?? []) m.set(sq.skill, sq)
    return m
  }, [analysis])

  function weakFailDims(sq: SkillQuality): DimKey[] {
    return DIMS.filter(d => sq.ratings[d].rating === "weak" || sq.ratings[d].rating === "fail")
  }

  function jumpToBoard(skill: string) {
    const sq = sqBySkill.get(skill)
    if (!sq) return
    const dim = weakFailDims(sq)[0] ?? "S1"
    setSelectedSq({ skill, dim })
    requestAnimationFrame(() => {
      document.getElementById("skill-quality-board")?.scrollIntoView({ behavior: "smooth", block: "start" })
    })
  }

  function nodeMaxSev(n: FlowNode): Severity | null {
    if (n.problems.length === 0) return null
    const sev = n.problems.map(p => p.severity)
    if (sev.includes("high")) return "high"
    if (sev.includes("medium")) return "medium"
    return "low"
  }

  return (
    <div className="flex flex-col gap-4 p-4">
      <div className="rounded-lg border bg-card p-4">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <h2 className="text-lg font-semibold">Workflow 实际流程 · 问题分析</h2>
            <p className="text-sm text-muted-foreground mt-1">{analysis.sessionSummary}</p>
          </div>
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant="outline">operator: {String(analysis.sessionMeta.operator)}</Badge>
            <Badge variant="outline">{String(analysis.sessionMeta.model)}</Badge>
            <Badge variant="outline">{String(analysis.sessionMeta.duration)}</Badge>
            <Badge variant="outline">{String(analysis.sessionMeta.tokens)} tok</Badge>
            <Badge variant="blue">{String(analysis.sessionMeta.autonomy)}</Badge>
            <Badge variant="green">{String(analysis.sessionMeta.reachedPhase)}</Badge>
          </div>
        </div>
        <div className="flex gap-2 mt-3 text-xs flex-wrap">
          <span className="text-muted-foreground">CP 执行:</span>
          {(analysis.sessionMeta.cpsExecuted as string[]).map(cp => (
            <Badge key={cp} variant="green">{cp} ✅</Badge>
          ))}
          {(analysis.sessionMeta.cpsMissing as string[]).map(cp => (
            <Badge key={cp} variant="gray">{cp} 未到</Badge>
          ))}
          <span className="text-muted-foreground ml-2">未达阶段:</span>
          {(analysis.sessionMeta.phasesNotReached as string[]).map(p => (
            <Badge key={p} variant="gray">{p}</Badge>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[1fr_360px] gap-4">
        <div className="rounded-lg border bg-card overflow-auto">
          <svg width={layout.width} height={layout.height} className="block" style={{ minWidth: layout.width }}>
            <defs>
              <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L8,3 L0,6 Z" fill="#94a3b8" />
              </marker>
              <marker id="retry-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L8,3 L0,6 Z" fill="#dc2626" />
              </marker>
            </defs>

            {layout.rows.map((row, ri) => {
              const next = layout.rows[ri + 1]
              const x = LEFT_LABEL_W / 2 + PAD
              return (
                <g key={row.step}>
                  <text x={PAD} y={row.y + NODE_H / 2 + 4} className="fill-muted-foreground" fontSize="11" fontWeight={600}>
                    {row.step}
                  </text>
                  {next && (
                    <line
                      x1={x} y1={row.y + NODE_H} x2={x} y2={next.y}
                      stroke="#cbd5e1" strokeWidth={1.5} markerEnd="url(#arrow)"
                    />
                  )}
                  {row.nodes.map((n) => {
                    const p = layout.pos.get(n.id)!
                    const sev = nodeMaxSev(n)
                    const isGate = n.type === "gate" || n.type === "terminal"
                    const stroke = sev === "high" ? "#dc2626" : sev === "medium" ? "#d97706" : isGate ? "#6366f1" : "#10b981"
                    const fill = n.type === "terminal" ? "#ede9fe" : n.type === "gate" ? "#e0e7ff" : "#f8fafc"
                    return (
                      <g key={n.id} transform={`translate(${p.x},${p.y})`} className="cursor-pointer" onClick={() => setSelected(n.id)}>
                        <rect
                          width={NODE_W} height={NODE_H} rx={isGate ? 26 : 8}
                          fill={fill} stroke={stroke} strokeWidth={selected === n.id ? 2.5 : 1.5}
                        />
                        <text x={NODE_W / 2} y={20} textAnchor="middle" fontSize="11" fontWeight={700} className="fill-foreground">
                          {shortName(n.skill)}
                        </text>
                        <text x={NODE_W / 2} y={36} textAnchor="middle" fontSize="9" className="fill-muted-foreground">
                          {n.type === "gate" ? n.status : `turn ${n.turn} · ${n.type}`}
                        </text>
                        {n.retryOf && (
                          <g>
                            <circle cx={NODE_W - 8} cy={8} r={8} fill="#dc2626" />
                            <text x={NODE_W - 8} y={11.5} textAnchor="middle" fontSize="9" fill="white" fontWeight={700}>↻</text>
                          </g>
                        )}
                        {sev && !n.retryOf && (
                          <g>
                            <circle cx={NODE_W - 8} cy={8} r={7} fill={SEV_COLOR[sev]} />
                            <text x={NODE_W - 8} y={11.5} textAnchor="middle" fontSize="8" fill="white" fontWeight={700}>!</text>
                          </g>
                        )}
                        {n.problems.length > 1 && (
                          <text x={NODE_W - 18} y={11.5} textAnchor="middle" fontSize="8" fill="white" fontWeight={700}>×{n.problems.length}</text>
                        )}
                        {(() => {
                          const sq = sqBySkill.get(n.skill)
                          if (!sq) return null
                          const wf = weakFailDims(sq)
                          if (wf.length === 0) return null
                          const hasFail = wf.some(d => sq.ratings[d].rating === "fail")
                          const col = hasFail ? "#dc2626" : "#d97706"
                          return (
                            <g>
                              <circle cx={11} cy={NODE_H - 11} r={8} fill={col} />
                              <text x={11} y={NODE_H - 7.5} textAnchor="middle" fontSize="8" fill="white" fontWeight={700}>{wf.length}</text>
                              <title>{`${n.skill} 质量看板: ${wf.length} 项 weak/fail (${wf.join(", ")})`}</title>
                            </g>
                          )
                        })()}
                      </g>
                    )
                  })}
                </g>
              )
            })}

            {retryEdges.map(e => {
              const from = layout.pos.get(e.from)!
              const to = layout.pos.get(e.to)!
              const laneX = layout.totalW + 20
              const fromY = from.cy
              const toY = to.cy
              const d = `M ${from.x + NODE_W} ${fromY} L ${laneX} ${fromY} L ${laneX} ${toY} L ${to.x + NODE_W} ${toY}`
              return (
                <path key={e.key} d={d} fill="none" stroke="#dc2626" strokeWidth={1.5} strokeDasharray="5 3" markerEnd="url(#retry-arrow)" opacity={0.7} />
              )
            })}
          </svg>
          <div className="flex gap-4 px-4 py-2 text-xs border-t flex-wrap">
            <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-sm border" style={{ borderColor: "#10b981" }} />正常</span>
            <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-sm border" style={{ borderColor: "#d97706" }} />有中等问题</span>
            <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-sm border" style={{ borderColor: "#6366f1" }} />门控/终点</span>
            <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5" style={{ background: "#dc2626" }} />↻ 重试回退</span>
            <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full" style={{ background: "#d97706" }} />节点左下=质量 weak/fail 数</span>
          </div>
        </div>

        <div className="flex flex-col gap-4 xl:sticky xl:top-4 xl:self-start xl:max-h-[calc(100vh-2rem)] xl:overflow-auto xl:pr-1">
          {selectedNode && (
            <div className="rounded-lg border bg-card p-4">
              <div className="flex items-center justify-between gap-2">
                <h3 className="font-semibold text-sm">{selectedNode.skill}</h3>
                <Badge variant="outline">{selectedNode.step}</Badge>
              </div>
              <p className="text-xs text-muted-foreground mt-1">turn {selectedNode.turn} · {selectedNode.type} · {selectedNode.status}</p>
              {selectedNode.retryOf && (
                <p className="text-xs text-red-600 mt-1">↻ 重试于 {selectedNode.retryOf}</p>
              )}
              {selectedNode.problems.length === 0 ? (
                <p className="text-xs text-green-600 mt-3">无流程问题</p>
              ) : (
                <div className="mt-3 space-y-3">
                  {selectedNode.problems.map((p, i) => (
                    <div key={i} className="rounded border p-2 text-xs space-y-1" style={{ borderColor: SEV_COLOR[p.severity] + "55" }}>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" style={{ color: SEV_COLOR[p.severity], borderColor: SEV_COLOR[p.severity] }}>{SEV_LABEL[p.severity]}</Badge>
                        <span className="font-mono">{p.type}</span>
                      </div>
                      {p.evidence && <p className="text-muted-foreground"><span className="font-semibold">证据:</span> {p.evidence}</p>}
                      {p.diagnosis && <p><span className="font-semibold">诊断:</span> {p.diagnosis}</p>}
                      {p.suggestion && <p className="text-blue-600"><span className="font-semibold">建议:</span> {p.suggestion}</p>}
                    </div>
                  ))}
                </div>
              )}

              {(() => {
                const sq = sqBySkill.get(selectedNode.skill)
                if (!sq) return null
                const wf = weakFailDims(sq)
                const allPass = wf.length === 0
                return (
                  <div className="mt-3 pt-3 border-t">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-semibold">Skill 质量评分（G/S 八维）</span>
                      <button type="button" onClick={() => jumpToBoard(sq.skill)} className="text-[10px] text-blue-600 hover:underline">
                        在看板中查看 →
                      </button>
                    </div>
                    {allPass ? (
                      <p className="text-xs text-green-600 mt-1">全部通过</p>
                    ) : (
                      <div className="mt-2 space-y-1.5">
                        {wf.map(d => (
                          <div key={d} className="text-xs rounded border p-1.5" style={{ borderColor: RATING_COLOR[sq.ratings[d].rating] + "55" }}>
                            <div className="flex items-center gap-2">
                              <Badge variant="outline" style={{ color: RATING_COLOR[sq.ratings[d].rating], borderColor: RATING_COLOR[sq.ratings[d].rating] }}>
                                {sq.ratings[d].rating}
                              </Badge>
                              <span className="font-semibold">{DIM_LABEL[d]}</span>
                              {sq.ratings[d].staticChecks && sq.ratings[d].staticChecks!.length > 0 && (
                                <span className="text-[10px] text-muted-foreground">含 {sq.ratings[d].staticChecks!.length} 条静态检查</span>
                              )}
                            </div>
                            <p className="text-muted-foreground mt-1">{sq.ratings[d].note}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )
              })()}
            </div>
          )}

          <div className="rounded-lg border bg-card p-4">
            <h3 className="font-semibold text-sm mb-2">Workflow 级问题</h3>
            <div className="space-y-2">
              {analysis.workflowLevelIssues.map(iss => (
                <div key={iss.id} className="text-xs rounded border p-2" style={{ borderColor: SEV_COLOR[iss.severity] + "55" }}>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" style={{ color: SEV_COLOR[iss.severity], borderColor: SEV_COLOR[iss.severity] }}>{SEV_LABEL[iss.severity]}</Badge>
                    <span className="font-semibold">{iss.title}</span>
                  </div>
                  <p className="text-muted-foreground mt-1">{iss.detail}</p>
                  <p className="text-blue-600 mt-1">{iss.suggestion}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-lg border bg-card p-4">
            <h3 className="font-semibold text-sm mb-2">优化优先级</h3>
            <ol className="space-y-2">
              {analysis.optimizationPriorities.map(op => (
                <li key={op.priority} className="text-xs">
                  <div className="flex items-center gap-2">
                    <Badge variant="purple">P{op.priority}</Badge>
                    <span className="font-mono text-[10px] text-muted-foreground">{op.target}</span>
                  </div>
                  <p className="mt-1">{op.action}</p>
                  <p className="text-green-600 mt-0.5">↑ {op.expectedGain}</p>
                </li>
              ))}
            </ol>
          </div>
        </div>
      </div>

      {analysis.skillQuality && analysis.skillQuality.length > 0 && (
        <div id="skill-quality-board" className="rounded-lg border bg-card p-4 scroll-mt-4">
          <div className="flex items-center justify-between flex-wrap gap-2 mb-3">
            <div>
              <h2 className="text-lg font-semibold">Skill 质量看板 · G/S 八维</h2>
              <p className="text-sm text-muted-foreground mt-0.5">
                G=任务产出质量（skill 做得好不好）· S=skill 本身写得好不好。点击单元格查看证据。
              </p>
            </div>
            <div className="flex gap-3 text-xs">
              {(["pass", "weak", "fail", "n-a"] as Rating[]).map(r => (
                <span key={r} className="flex items-center gap-1">
                  <span className="inline-block w-3 h-3 rounded-sm" style={{ background: RATING_COLOR[r] }} />
                  {r === "pass" ? "通过" : r === "weak" ? "偏弱" : r === "fail" ? "不达标" : "不可测"}
                </span>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-[1fr_340px] gap-4 items-start">
          <div className="overflow-auto">
            <table className="text-xs border-collapse">
              <thead>
                <tr>
                  <th className="text-left font-semibold p-2 border-b sticky left-0 bg-card whitespace-nowrap">Skill</th>
                  <th className="text-center font-semibold p-2 border-b whitespace-nowrap">#</th>
                  {DIMS.map(d => (
                    <th key={d} className="text-center font-semibold p-2 border-b whitespace-nowrap" title={DIM_LABEL[d]}>
                      {d}
                    </th>
                  ))}
                  <th className="text-left font-semibold p-2 border-b">小结</th>
                </tr>
              </thead>
              <tbody>
                {analysis.skillQuality.map(sq => (
                  <tr key={sq.skill} className="hover:bg-muted/40">
                    <td className="p-2 border-b font-mono whitespace-nowrap sticky left-0 bg-card">{shortName(sq.skill)}</td>
                    <td className="p-2 border-b text-center text-muted-foreground">{sq.occurrences}</td>
                    {DIMS.map(d => {
                      const r = sq.ratings[d]
                      const active = selectedSq?.skill === sq.skill && selectedSq?.dim === d
                      return (
                        <td key={d} className="p-1 border-b text-center">
                          <button
                            type="button"
                            onClick={() => setSelectedSq(active ? null : { skill: sq.skill, dim: d })}
                            className="w-9 h-7 rounded inline-flex items-center justify-center text-white font-bold text-sm transition"
                            style={{
                              background: RATING_COLOR[r.rating],
                              outline: active ? "2px solid #3b82f6" : "none",
                              outlineOffset: 1,
                            }}
                            title={`${DIM_LABEL[d]}: ${r.rating}\n${r.note}`}
                          >
                            {RATING_LABEL[r.rating]}
                          </button>
                        </td>
                      )
                    })}
                    <td className="p-2 border-b text-muted-foreground">{sq.summary}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="space-y-3 xl:sticky xl:top-4 xl:self-start xl:max-h-[calc(100vh-2rem)] xl:overflow-auto xl:pr-1">
            {selectedSq ? (() => {
            const sq = analysis.skillQuality!.find(s => s.skill === selectedSq.skill)!
            const r = sq.ratings[selectedSq.dim]
            return (
              <div className="rounded border p-3 text-xs bg-muted/30 space-y-2">
                <div className="flex items-center gap-2">
                  <span className="font-mono font-semibold">{shortName(sq.skill)}</span>
                  <Badge variant="outline">{DIM_LABEL[selectedSq.dim]}</Badge>
                  <Badge variant="outline" style={{ color: RATING_COLOR[r.rating], borderColor: RATING_COLOR[r.rating] }}>
                    {r.rating}
                  </Badge>
                  <span className="text-muted-foreground">出现 {sq.occurrences} 次</span>
                </div>
                {r.note && <p className="text-foreground">{r.note}</p>}
                {r.evidence && <p className="text-muted-foreground"><span className="font-semibold">证据:</span> {r.evidence}</p>}
                {r.diagnosis && <p><span className="font-semibold">诊断:</span> {r.diagnosis}</p>}
                {r.suggestion && <p className="text-blue-600"><span className="font-semibold">建议:</span> {r.suggestion}</p>}
                {r.staticChecks && r.staticChecks.length > 0 && (
                  <div className="mt-2 pt-2 border-t">
                    <p className="font-semibold mb-1">静态分析（扫 skill 文本）</p>
                    <div className="space-y-2">
                      {r.staticChecks.map((c, i) => (
                        <div key={i} className="rounded border p-2" style={{ borderColor: SEV_COLOR[c.severity] + "55" }}>
                          <div className="flex items-center gap-2 flex-wrap">
                            <Badge variant="outline" style={{ color: SEV_COLOR[c.severity], borderColor: SEV_COLOR[c.severity] }}>{SEV_LABEL[c.severity]}</Badge>
                            <Badge variant="gray">{STATIC_CAT_LABEL[c.category]}</Badge>
                          </div>
                          <p className="mt-1">{c.issue}</p>
                          {c.snippet && <p className="mt-1 font-mono text-[10px] text-muted-foreground bg-muted/40 rounded p-1">{c.snippet}</p>}
                          {c.suggestion && <p className="mt-1 text-blue-600"><span className="font-semibold">建议:</span> {c.suggestion}</p>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {!r.note && !r.evidence && !r.diagnosis && !r.suggestion && (!r.staticChecks || r.staticChecks.length === 0) && (
                  <p className="text-muted-foreground">（无说明）</p>
                )}
              </div>
            )
          })() : (
              <div className="rounded border p-3 text-xs text-muted-foreground bg-muted/30">
                点击左侧某 skill 的维度单元格，在此查看该维度的证据 / 诊断 / 建议 / 静态检查。
              </div>
            )}

            <div className="rounded-lg border bg-muted/20 p-3">
              <h4 className="font-semibold text-xs mb-2">八维标准</h4>
              <div className="text-[11px] space-y-1.5">
                {DIMS.map(d => (
                  <div key={d} className={`rounded p-1 -m-1 ${selectedSq?.dim === d ? "bg-blue-50 dark:bg-blue-950/40" : ""}`}>
                    <div>
                      <span className="font-mono font-semibold">{d}</span>
                      <span className="text-muted-foreground"> · {DIM_DESC[d].desc}</span>
                    </div>
                    {DIM_DESC[d].principles && DIM_DESC[d].principles.length > 0 && (
                      <ul className="pl-4 mt-0.5 space-y-0.5">
                        {DIM_DESC[d].principles.map(p => (
                          <li key={p.name} className="text-muted-foreground/90">
                            <span className="font-semibold text-foreground/80">{p.name}</span>
                            <span className="font-mono text-[10px]"> — {p.source}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
        </div>
      )}
    </div>
  )
}
