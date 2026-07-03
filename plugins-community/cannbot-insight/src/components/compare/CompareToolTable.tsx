"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface ToolStat {
  count: number
  successCount: number
}

interface CompareToolTableProps {
  toolStatsA: Record<string, ToolStat>
  toolStatsB: Record<string, ToolStat>
}

function formatDiff(diff: number): string {
  if (diff === 0) return "0"
  if (diff > 0) return `+${diff}`
  return `${diff}`
}

function diffVariant(diff: number): "green" | "red" | "gray" {
  if (diff < 0) return "green"
  if (diff > 0) return "red"
  return "gray"
}

export function CompareToolTable({ toolStatsA, toolStatsB }: CompareToolTableProps) {
  const allTools = Array.from(new Set([...Object.keys(toolStatsA), ...Object.keys(toolStatsB)])).sort()

  if (allTools.length === 0) {
    return (
      <Card size="sm">
        <CardHeader><CardTitle>Tool Call 对比</CardTitle></CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">No tool call data available.</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card size="sm">
      <CardHeader><CardTitle>Tool Call 对比</CardTitle></CardHeader>
      <CardContent>
        <div className="border rounded-lg">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="text-xs">Tool Name</TableHead>
                <TableHead className="text-xs text-center">Session A Count</TableHead>
                <TableHead className="text-xs text-center">Session A Success</TableHead>
                <TableHead className="text-xs text-center">Session B Count</TableHead>
                <TableHead className="text-xs text-center">Session B Success</TableHead>
                <TableHead className="text-xs text-center">Diff</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {allTools.map(toolName => {
                const a = toolStatsA[toolName] ?? { count: 0, successCount: 0 }
                const b = toolStatsB[toolName] ?? { count: 0, successCount: 0 }
                const diff = b.count - a.count

                return (
                  <TableRow key={toolName}>
                    <TableCell className="text-xs font-medium">{toolName}</TableCell>
                    <TableCell className="text-xs text-center tabular-nums">{a.count}</TableCell>
                    <TableCell className="text-xs text-center tabular-nums">
                      {a.count > 0 ? `${((a.successCount / a.count) * 100).toFixed(0)}%` : "—"}
                    </TableCell>
                    <TableCell className="text-xs text-center tabular-nums">{b.count}</TableCell>
                    <TableCell className="text-xs text-center tabular-nums">
                      {b.count > 0 ? `${((b.successCount / b.count) * 100).toFixed(0)}%` : "—"}
                    </TableCell>
                    <TableCell className="text-xs text-center">
                      <Badge variant={diffVariant(diff)}>{formatDiff(diff)}</Badge>
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  )
}
