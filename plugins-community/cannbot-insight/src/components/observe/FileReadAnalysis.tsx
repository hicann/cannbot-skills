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

      <div className="flex gap-2">
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
      </div>

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
              <div className="flex items-center gap-2 mb-1">
                <span className="text-sm font-medium truncate flex-1">{file.displayPath}</span>
                <Badge variant="secondary">{file.totalReads} reads</Badge>
                {file.overlappingReads > 0 && (
                  <Badge variant="destructive">{file.overlappingReads} overlap</Badge>
                )}
              </div>
              <div className="flex gap-0.5 h-3 rounded-full overflow-hidden bg-muted">
                <div
                  className={file.overlappingReads > 0 ? "bg-blue-400 h-full rounded-full" : "bg-blue-500 h-full rounded-full"}
                  style={{
                    width: `${(file.totalReads / maxReads) * 100}%`,
                    minWidth: "2px",
                  }}
                />
              </div>
            </div>

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
