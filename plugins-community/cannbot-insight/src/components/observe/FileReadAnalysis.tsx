"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useState, useEffect } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollTextIcon, InfoIcon, FolderArchiveIcon } from "lucide-react"

interface ReadEntry {
  turnId: string
  turnIndex: number
  agent: string
  prompt: string | null
  subagentSessionId: string | null
  llmOutput: string | null
  range: {
    type: "full" | "partial"
    start: number
    end: number | null
  }
}

interface FileAnalysis {
  path: string
  displayPath: string
  reads: ReadEntry[]
  totalReads: number
  overlappingReads: number
  totalLinesRead: number
  uniqueLinesRead: number
  redundancyRate: number
}

interface FileReadsResponse {
  files: FileAnalysis[]
  summary: {
    totalFiles: number
    totalReads: number
    filesWithOverlap: number
    redundancyRate: number
  }
}

interface RestoreLine {
  n: number
  content: string | null
  source: "read" | "write" | "gap"
}

interface RestoreResponse {
  path: string
  lines: RestoreLine[]
  maxLine: number
  opsUsed: number
}

interface FileReadAnalysisProps {
  taskId: string
  onNavigateToTurn?: (turnId: string) => void
}

export function FileReadAnalysis({ taskId, onNavigateToTurn }: FileReadAnalysisProps) {
  const [data, setData] = useState<FileReadsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState<"all" | "overlap">("all")
  const [expandedFiles, setExpandedFiles] = useState<Set<string>>(new Set())
  const [restoring, setRestoring] = useState<Set<string>>(new Set())
  const [restored, setRestored] = useState<Map<string, RestoreResponse>>(new Map())
  const [restoreError, setRestoreError] = useState<Map<string, string>>(new Map())

  async function restoreContent(path: string) {
    setRestoring(prev => new Set(prev).add(path))
    setRestoreError(prev => { const n = new Map(prev); n.delete(path); return n })
    try {
      const r = await fetch(
        `/api/observe/session/file-restore?taskId=${encodeURIComponent(taskId)}&filePath=${encodeURIComponent(path)}`
      )
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const d: RestoreResponse = await r.json()
      setRestored(prev => new Map(prev).set(path, d))
    } catch (e) {
      setRestoreError(prev => new Map(prev).set(path, e instanceof Error ? e.message : String(e)))
    } finally {
      setRestoring(prev => { const n = new Set(prev); n.delete(path); return n })
    }
  }

  function downloadRestored(path: string) {
    const data = restored.get(path)
    if (!data) return
    const text = data.lines
      .map(l => (l.content === null ? `--line ${l.n} not found --` : l.content))
      .join("\n")
    const basename = path.split("/").pop() || "restored"
    const dot = basename.lastIndexOf(".")
    const stem = dot > 0 ? basename.slice(0, dot) : basename
    const ext = dot > 0 ? basename.slice(dot) : ""
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" })
    const url = URL.createObjectURL(blob)
    const a = window.document.createElement("a")
    a.href = url
    a.download = `${stem}.restored${ext}`
    window.document.body.appendChild(a)
    a.click()
    window.document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  function clearRestored(path: string) {
    setRestored(prev => { const n = new Map(prev); n.delete(path); return n })
    setRestoreError(prev => { const n = new Map(prev); n.delete(path); return n })
  }

  const [restoringAll, setRestoringAll] = useState(false)
  const [allError, setAllError] = useState<string | null>(null)
  const [allDone, setAllDone] = useState<{ count: number; totalLines: number; gapLines: number } | null>(null)

  async function restoreAll() {
    setRestoringAll(true)
    setAllError(null)
    setAllDone(null)
    try {
      const r = await fetch(`/api/observe/session/dir-restore?taskId=${encodeURIComponent(taskId)}`)
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const blob = await r.blob()
      const url = URL.createObjectURL(blob)
      const a = window.document.createElement("a")
      a.href = url
      const cd = r.headers.get("Content-Disposition") || ""
      const m = cd.match(/filename="?([^"]+)"?/)
      a.download = m ? decodeURIComponent(m[1]) : `session_${taskId}_restored.zip`
      window.document.body.appendChild(a)
      a.click()
      window.document.body.removeChild(a)
      URL.revokeObjectURL(url)
      setAllDone({
        count: Number(r.headers.get("X-Restored-Count") ?? 0),
        totalLines: Number(r.headers.get("X-Restored-Total-Lines") ?? 0),
        gapLines: Number(r.headers.get("X-Restored-Gap-Lines") ?? 0),
      })
    } catch (e) {
      setAllError(e instanceof Error ? e.message : String(e))
    } finally {
      setRestoringAll(false)
    }
  }

  useEffect(() => {
    let cancelled = false
    fetch(`/api/observe/session/file-reads?taskId=${encodeURIComponent(taskId)}`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then(d => {
        if (!cancelled) {
          setData(d)
          setLoading(false)
          setError(null)
        }
      })
      .catch(e => {
        if (!cancelled) {
          setError(e.message)
          setLoading(false)
        }
      })
    return () => { cancelled = true }
  }, [taskId])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <span className="text-muted-foreground">Loading file reads data...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <span className="text-destructive">Error: {error}</span>
      </div>
    )
  }

  if (!data || data.files.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <span className="text-muted-foreground">No file read data available for this session</span>
      </div>
    )
  }

  const filteredFiles = filter === "overlap"
    ? data.files.filter(f => f.overlappingReads > 0)
    : data.files

  const maxReads = Math.max(...filteredFiles.map(f => f.totalReads), 1)

  return (
    <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-4">
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold">{data.summary.totalFiles}</div>
            <div className="text-sm text-muted-foreground">Files Accessed</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold">{data.summary.totalReads}</div>
            <div className="text-sm text-muted-foreground">Total Reads</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold">{data.summary.filesWithOverlap}</div>
            <div className="text-sm text-muted-foreground">Files w/ Overlap</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold">{(data.summary.redundancyRate * 100).toFixed(1)}%</div>
            <div className="text-sm text-muted-foreground">Redundancy Rate</div>
          </CardContent>
        </Card>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Badge
          variant={filter === "all" ? "default" : "outline"}
          className="cursor-pointer"
          onClick={() => setFilter("all")}
        >
          All ({data.files.length})
        </Badge>
        <Badge
          variant={filter === "overlap" ? "default" : "outline"}
          className="cursor-pointer"
          onClick={() => setFilter("overlap")}
        >
          With Overlap ({data.summary.filesWithOverlap})
        </Badge>
        <span className="ml-auto" />
        <span
          role="button"
          title="重编汇册：按时间戳逐行重建本会话工作目录下全部被读/写过的文件，保留路径树打包下载 zip"
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold border transition-colors ${
            allDone
              ? "border-teal-500 bg-teal-500/25 text-teal-700 dark:text-teal-200"
              : "border-teal-500/50 bg-teal-500/15 text-teal-600 dark:text-teal-300 hover:bg-teal-500/25 hover:border-teal-500"
          }`}
          onClick={() => {
            if (restoringAll) return
            if (allDone || allError) { setAllDone(null); setAllError(null); return }
            restoreAll()
          }}
        >
          <FolderArchiveIcon className="size-3.5" />
          {restoringAll ? "重编中…" : "重编汇册 Gather & Rebuild Directory"}
        </span>
      </div>

      {(allDone || allError) && (
        <div className="p-3 border rounded-md border-teal-400/30 bg-teal-500/5 space-y-1">
          <div className="flex items-center justify-between">
            <span className="inline-flex items-center gap-1 text-xs font-medium text-teal-600 dark:text-teal-400">
              <FolderArchiveIcon className="size-3.5" />
              {allError ? "重编失败" : `已打包下载 ${allDone?.count ?? 0} 个文件`}
              {allDone && (
                <span className="text-muted-foreground">
                  （共 {allDone.totalLines} 行，{allDone.gapLines} 行未采集）
                </span>
              )}
            </span>
            <span
              role="button"
              title="关闭"
              className="text-muted-foreground cursor-pointer hover:text-foreground text-sm leading-none px-1"
              onClick={() => { setAllDone(null); setAllError(null) }}
            >
              ×
            </span>
          </div>
          <div className="text-[11px] text-muted-foreground leading-snug">
            按时间顺序逐行拼合本会话内每个被读、被写文件的内容，后写入者覆盖先前。局部改写（str_replace）未纳入，未被读到的行标注「未采集」并写入 zip。
          </div>
          {allError && (
            <div className="text-xs text-destructive">Error: {allError}</div>
          )}
        </div>
      )}

      <div className="space-y-2">
        {filteredFiles.map(file => (
          <div key={file.path}>
            <div
              className="cursor-pointer hover:bg-accent/50 rounded p-2 transition-colors"
              onClick={() => {
                const next = new Set(expandedFiles)
                if (next.has(file.path)) {
                  next.delete(file.path)
                } else {
                  next.add(file.path)
                }
                setExpandedFiles(next)
              }}
            >
              <div className="grid grid-cols-[1fr_auto] items-center gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-medium truncate block pr-2">{file.displayPath}</span>
                  </div>
                  <div
                    className="h-2.5 bg-muted/30 rounded overflow-hidden w-4/5"
                    title={`${file.totalReads} reads`}
                  >
                    <div
                      className="bg-blue-500 h-full rounded transition-all"
                      style={{
                        width: `${Math.max((file.totalReads / maxReads) * 100, 2)}%`,
                      }}
                    />
                  </div>
                </div>
                <div className="flex items-center gap-2 justify-end shrink-0">
                  <Badge variant="secondary">{file.totalReads} reads</Badge>
                  {file.overlappingReads > 0 && (
                    <Badge variant="destructive">{file.overlappingReads} overlap</Badge>
                  )}
                  <span
                    role="button"
                    title="循迹复卷：按时间戳逐行重建该文件的全部内容"
                    className={`inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-semibold border transition-colors ${
                      restored.has(file.path)
                        ? "border-teal-500 bg-teal-500/25 text-teal-700 dark:text-teal-200"
                        : "border-teal-500/50 bg-teal-500/15 text-teal-600 dark:text-teal-300 hover:bg-teal-500/25 hover:border-teal-500"
                    }`}
                    onClick={(e) => {
                      e.stopPropagation()
                      if (restored.has(file.path)) {
                        clearRestored(file.path)
                      } else if (!restoring.has(file.path)) {
                        restoreContent(file.path)
                      }
                    }}
                  >
                    <ScrollTextIcon className="size-3.5" />
                    {restoring.has(file.path) ? "复卷中…" : restored.has(file.path) ? "已复卷·收起" : "循迹复卷"}
                  </span>
                </div>
              </div>
            </div>

            {(restoring.has(file.path) || restored.has(file.path) || restoreError.has(file.path)) && (
              <div className="ml-4 mt-2 p-3 border rounded-md border-teal-400/30 bg-teal-500/5 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="inline-flex items-center gap-1 text-xs font-medium text-teal-600 dark:text-teal-400">
                    <ScrollTextIcon className="size-3.5" />
                    循迹复卷 · {restored.get(file.path)?.lines.length ?? 0} 行
                    {restored.get(file.path) && restored.get(file.path)!.opsUsed > 0 && (
                      <span className="text-muted-foreground">
                        （{restored.get(file.path)!.opsUsed} 次读写，{restored.get(file.path)!.lines.filter(l => l.source === 'gap').length} 行未采集）
                      </span>
                    )}
                    <InfoIcon
                      className="size-3.5 text-muted-foreground cursor-help"
                      title="按时间顺序拼合本会话中该文件的读写片段，后写入者覆盖先前内容。局部改写（str_replace）未纳入，未被读取的行标注为「未采集」，故重建结果未必等于任一时刻的真实文件。"
                    />
                  </span>
                  <span className="flex items-center gap-3">
                    {restored.has(file.path) && (
                      <span
                        role="button"
                        className="text-xs font-medium text-blue-500 cursor-pointer hover:underline"
                        onClick={(e) => {
                          e.stopPropagation()
                          downloadRestored(file.path)
                        }}
                      >
                        Download
                      </span>
                    )}
                    <span
                      role="button"
                      title="收起"
                      className="text-muted-foreground cursor-pointer hover:text-foreground text-sm leading-none px-1"
                      onClick={(e) => {
                        e.stopPropagation()
                        clearRestored(file.path)
                      }}
                    >
                      ×
                    </span>
                  </span>
                </div>
                <div className="text-[11px] text-muted-foreground leading-snug">
                  按时间顺序拼合本会话中该文件的读写片段，后写入者覆盖先前内容。局部改写（str_replace）未纳入，未被读取的行标注为「未采集」，故重建结果未必等于任一时刻的真实文件。
                </div>
                {restoreError.has(file.path) && (
                  <div className="text-xs text-destructive">
                    Error: {restoreError.get(file.path)}
                  </div>
                )}
                {restored.has(file.path) && (() => {
                  const r = restored.get(file.path)!
                  return (
                    <pre className="max-h-96 overflow-auto rounded border bg-background p-2 text-xs font-mono whitespace-pre">
                      {r.lines.map(l => (
                        <div
                          key={l.n}
                          className={l.content === null ? "text-muted-foreground italic" : ""}
                        >
                          {l.content === null ? `--line ${l.n} not found --` : `${l.n}\t${l.content}`}
                        </div>
                      ))}
                    </pre>
                  )
                })()}
              </div>
            )}

            {expandedFiles.has(file.path) && (
              <div className="ml-4 mt-2 p-3 border rounded-md bg-muted/30 space-y-2">
                <div className="text-xs text-muted-foreground">
                  {file.totalLinesRead > 0 ? (
                    <span>
                      {file.totalLinesRead} lines read, {file.uniqueLinesRead} unique
                      ({(file.redundancyRate * 100).toFixed(1)}% redundant)
                    </span>
                  ) : (
                    <span>All full reads — line metrics N/A</span>
                  )}
                </div>
                <div className="space-y-3">
                  {(() => {
                    const groups: { key: string; agent: string; prompt: string | null; reads: ReadEntry[] }[] = []
                    const groupMap = new Map<string, { agent: string; prompt: string | null; reads: ReadEntry[] }>()
                    for (const read of file.reads) {
                      const key = read.subagentSessionId ?? "root"
                      let g = groupMap.get(key)
                      if (!g) {
                        g = { agent: read.agent, prompt: read.prompt, reads: [] }
                        groupMap.set(key, g)
                        groups.push({ key, ...g })
                      }
                      g.reads.push(read)
                    }
                    return groups.map(group => (
                      <div key={group.key} className="space-y-1">
                        <div className="flex items-center gap-2 text-xs">
                          <Badge variant="outline" className="shrink-0">
                            {group.agent}
                          </Badge>
                          {group.prompt && (
                            <span className="text-muted-foreground truncate max-w-[30ch]" title={group.prompt}>
                              &ldquo;{group.prompt.length > 30 ? group.prompt.slice(0, 30) + "..." : group.prompt}&rdquo;
                            </span>
                          )}
                        </div>
                        <div className="ml-4 space-y-0.5">
                          {group.reads.map((read, idx) => (
                            <div key={idx} className="flex items-center gap-2 text-xs">
                              <span
                                className={onNavigateToTurn ? "font-mono text-blue-500 cursor-pointer hover:underline" : "font-mono text-muted-foreground"}
                                onClick={(e) => {
                                  e.stopPropagation()
                                  onNavigateToTurn?.(read.turnId)
                                }}
                              >
                                #{read.turnIndex}
                              </span>
                              {read.llmOutput && (
                                <span className="text-muted-foreground truncate max-w-[40ch]" title={read.llmOutput}>
                                  {read.llmOutput.length > 40 ? read.llmOutput.slice(0, 40) + "..." : read.llmOutput}
                                </span>
                              )}
                              <span className="text-muted-foreground shrink-0">
                                {read.range.type === "full"
                                  ? read.range.start > 0
                                    ? `[full from line ${read.range.start}]`
                                    : "[full read]"
                                  : `[lines ${read.range.start}-${read.range.end})`}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))
                  })()}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
