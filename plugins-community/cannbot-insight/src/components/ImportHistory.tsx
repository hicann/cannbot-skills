"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useState, useEffect } from "react"
import { BRAND_NAME, BRAND_SOURCE_TYPE } from '@/lib/branding'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { TrashIcon } from "lucide-react"

export interface ImportHistoryEntry {
  taskId: string
  importedAt: string
  turnCount: number
  status: "success" | "error"
  query: string | null
  filePath: string | null
  sourceType?: string
}

const STORAGE_KEY = "cannbot-import-history"
const MAX_ENTRIES = 20

export function saveImportHistory(entries: ImportHistoryEntry[]) {
  try {
    const existing: ImportHistoryEntry[] = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]")
    const merged = [...entries, ...existing].slice(0, MAX_ENTRIES)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(merged))
  } catch {}
}

export function getImportHistory(): ImportHistoryEntry[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]")
  } catch {
    return []
  }
}

function removeImportHistoryEntry(taskId: string) {
  try {
    const existing: ImportHistoryEntry[] = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]")
    const filtered = existing.filter(e => e.taskId !== taskId)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered))
  } catch {}
}

function clearImportHistory() {
  try {
    localStorage.removeItem(STORAGE_KEY)
  } catch {}
}

function formatTime(iso: string): string {
  if (!iso) return "—"
  try {
    const d = new Date(iso)
    const year = d.getFullYear()
    const month = String(d.getMonth() + 1)
    const day = String(d.getDate())
    const hour = String(d.getHours()).padStart(2, '0')
    const minute = String(d.getMinutes()).padStart(2, '0')
    return `${year}/${month}/${day} ${hour}:${minute}`
  } catch {
    return iso
  }
}

export function ImportHistory() {
  const [history, setHistory] = useState<ImportHistoryEntry[] | null>(null)
  const [deleting, setDeleting] = useState<string | null>(null)
  const [deletingAll, setDeletingAll] = useState(false)

  useEffect(() => {
    setHistory(getImportHistory())
  }, [])

  if (history === null) {
    return (
      <div className="space-y-3">
        <h2 className="text-lg font-semibold">Recent Imports</h2>
        <div className="h-8 rounded-md bg-muted/30 animate-pulse" />
      </div>
    )
  }

  if (history.length === 0) return null

  async function handleDeleteOne(taskId: string) {
    setDeleting(taskId)
    try {
      const res = await fetch("/api/ingest/delete-session", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ taskId }),
      })
      // Remove from localStorage regardless of API result (session may already be deleted from DB)
      removeImportHistoryEntry(taskId)
      setHistory(getImportHistory())
      if (res.ok) {
        window.location.reload()
      }
    } catch {
      // Still remove from localStorage even on network error
      removeImportHistoryEntry(taskId)
      setHistory(getImportHistory())
    }
    setDeleting(null)
  }

  async function handleDeleteAll() {
    setDeletingAll(true)
    try {
      const res = await fetch("/api/ingest/delete-session", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ deleteAll: true }),
      })
      if (res.ok) {
        clearImportHistory()
        setHistory([])
        window.location.reload()
      }
    } catch { /* ignore */ }
    setDeletingAll(false)
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Recent Imports ({history.length})</h2>        <Button
          variant="destructive"
          size="sm"
          disabled={deletingAll}
          onClick={handleDeleteAll}
        >
          {deletingAll ? "Deleting..." : "Delete All"}
        </Button>
      </div>
      <div className="border rounded-lg">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="text-xs">File Path</TableHead>
              <TableHead className="text-xs">Query</TableHead>
              <TableHead className="text-xs whitespace-nowrap">Import Time</TableHead>
              <TableHead className="text-xs">Type</TableHead>
              <TableHead className="text-xs">Status</TableHead>
              <TableHead className="w-[40px]"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {history.map((entry, i) => (
              <TableRow key={`${entry.taskId}-${i}`}>
                <TableCell className="max-w-[400px] truncate text-xs font-mono text-muted-foreground" title={entry.filePath ?? ""}>{entry.filePath ?? "—"}</TableCell>
                <TableCell className="max-w-[260px] truncate text-xs" title={entry.query ?? entry.taskId}>{entry.query ?? entry.taskId}</TableCell>
                <TableCell className="text-xs text-muted-foreground whitespace-nowrap">{formatTime(entry.importedAt)}</TableCell>
                <TableCell className="text-xs">
                  {entry.sourceType === "opencode-db" ? (
                    <Badge variant="blue">OpenCode</Badge>
                  ) : entry.sourceType === "claude-jsonl" ? (
                    <Badge variant="orange">Claude</Badge>
                  ) : entry.sourceType === BRAND_SOURCE_TYPE ? (
                    <Badge variant="purple">{BRAND_NAME}</Badge>
                  ) : "—"}
                </TableCell>
                <TableCell className="text-xs">
                  {entry.status === "success" ? (
                    <Badge variant="green">OK</Badge>
                  ) : (
                    <Badge variant="red">error</Badge>
                  )}
                </TableCell>
                <TableCell className="px-2 text-right">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="size-6 text-muted-foreground hover:text-red-600"
                    disabled={deleting === entry.taskId}
                    onClick={() => handleDeleteOne(entry.taskId)}
                    title="Delete from Insight"
                  >
                    {deleting === entry.taskId ? <span className="text-xs">…</span> : <TrashIcon className="size-3.5" />}
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}
