// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

'use client';

import { useState } from 'react'
import Link from 'next/link'
import { BRAND_NAME, BRAND_SOURCE_TYPE } from '@/lib/branding'
import { useRouter } from 'next/navigation';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { UploadIcon, LoaderIcon, CheckIcon, XIcon, TrashIcon, DownloadIcon, CheckCircleIcon } from 'lucide-react';
import { toast } from 'sonner';

interface SessionListItem {
  sessionId: string;
  taskId: string;
  query: string | null;
  framework: string | null;
  startTime: string;
  endTime: string | null;
  totalTokens: number;
  totalCost: number;
  totalLatencyMs: number;
  totalToolCallCount: number;
  totalSkillLoadCount: number;
  totalSubagentCount: number;
  model: string | null;
  user: string | null;
  sourcePath: string | null;
}

interface SessionListProps {
  items: SessionListItem[];
  total: number;
  page: number;
  pageSize: number;
}

type UploadStatus = 'idle' | 'uploading' | 'done' | 'error';
type ExportStatus = 'idle' | 'exporting' | 'done' | 'error';

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function formatCost(n: number): string {
  return `$${n.toFixed(2)}`;
}

function formatDuration(ms: number): string {
  if (ms <= 0) return '—';
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms < 3600000) return `${(ms / 60000).toFixed(1)}min`;
  return `${(ms / 3600000).toFixed(1)}h`;
}

function formatTime(iso: string | null): string {
  if (!iso) return '—';
  const d = new Date(iso);
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  const hour = String(d.getHours()).padStart(2, '0');
  const minute = String(d.getMinutes()).padStart(2, '0');
  return `${month}/${day} ${hour}:${minute}`;
}

function buildUploadDescription(item: SessionListItem): string {
  const query = item.query ?? ''
  const user = item.user ?? ''
  const model = item.model ?? ''
  const timeRange = `${formatTime(item.startTime)} → ${formatTime(item.endTime)}`
  const framework = item.framework ?? ''
  const sourcePath = item.sourcePath ?? ''
  return `提交人:${user ? ' ' + user : ''}
内容描述:${query ? ' ' + query : ''}
问题说明:
日志路径:${sourcePath ? ' ' + sourcePath : ' taskId ' + item.taskId + ', 框架 ' + framework}
备注: 模型 ${model}, 时间 ${timeRange}`
}

export function SessionList({ items, total, page, pageSize }: SessionListProps) {
  const router = useRouter()
  const totalPages = Math.ceil(total / pageSize);
  const hasNext = page < totalPages;
  const hasPrev = page > 1;
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())

  const [uploadStatus, setUploadStatus] = useState<Record<string, UploadStatus>>({})
  const [exportStatus, setExportStatus] = useState<Record<string, ExportStatus>>({})
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)
  const [uploadDialogItem, setUploadDialogItem] = useState<SessionListItem | null>(null)
  const [uploadDescription, setUploadDescription] = useState('')

  function handleToggle(sessionId: string) {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(sessionId)) {
        next.delete(sessionId)
      } else if (next.size < 2) {
        next.add(sessionId)
      }
      return next
    })
  }

  async function handleDeleteAll() {
    try {
      const res = await fetch('/api/ingest/delete-session', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ deleteAll: true }),
      })
      if (res.ok) window.location.reload()
    } catch {}
  }

  async function handleDeleteOne(item: SessionListItem) {
    try {
      const res = await fetch('/api/ingest/delete-session', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ taskId: item.taskId, framework: item.framework ?? 'unknown' }),
      })
      if (res.ok) window.location.reload()
    } catch {}
  }

  async function handleUpload(item: SessionListItem, description: string) {
    const key = item.sessionId
    setUploadStatus(prev => ({ ...prev, [key]: 'uploading' }))
    setUploadError(null)
    setUploadDialogItem(null)
    try {
      const res = await fetch('/api/ingest/upload-session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          taskId: item.taskId,
          framework: item.framework ?? 'unknown',
          description,
        }),
      })
      const data = await res.json()
      if (!res.ok) {
        setUploadStatus(prev => ({ ...prev, [key]: 'error' }))
        setUploadError(data.error ?? 'Upload failed')
        return
      }
      setUploadStatus(prev => ({ ...prev, [key]: 'done' }))
    } catch {
      setUploadStatus(prev => ({ ...prev, [key]: 'error' }))
      setUploadError('Upload failed: network error')
    }
  }

  async function handleExport(item: SessionListItem) {
    const key = item.sessionId
    setExportStatus(prev => ({ ...prev, [key]: 'exporting' }))
    try {
      const res = await fetch('/api/ingest/export-session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ taskId: item.taskId }),
      })
      if (!res.ok) {
        const err = await res.json()
        setExportStatus(prev => ({ ...prev, [key]: 'error' }))
        toast.error('Export failed', { description: err.error ?? 'Unknown error' })
        return
      }
      const blob = await res.blob()
      const defaultName = `cannbot_session_${item.taskId}.db`
      if (typeof window.showSaveFilePicker === 'function') {
        try {
          const handle = await window.showSaveFilePicker({
            suggestedName: defaultName,
            types: [{ description: 'SQLite Database', accept: { 'application/x-sqlite3': ['.db'] } }],
          })
          const writable = await handle.createWritable()
          await writable.write(blob)
          await writable.close()
          setExportStatus(prev => ({ ...prev, [key]: 'done' }))
          toast.success('Database exported', {
            description: `Saved to ${handle.name}.`,
            icon: <CheckCircleIcon className="size-4" />,
            duration: 5000,
          })
          return
        } catch (e: unknown) {
          if (e instanceof DOMException && e.name === 'AbortError') {
            setExportStatus(prev => ({ ...prev, [key]: 'idle' }))
            return
          }
        }
      }
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = defaultName
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      setTimeout(() => URL.revokeObjectURL(url), 10000)
      setExportStatus(prev => ({ ...prev, [key]: 'done' }))
      toast.success('Database exported', {
        description: `${defaultName} has been downloaded.`,
        icon: <CheckCircleIcon className="size-4" />,
        duration: 5000,
      })
    } catch {
      setExportStatus(prev => ({ ...prev, [key]: 'error' }))
      toast.error('Export failed', { description: 'Network error' })
    }
  }

  const selectedArr = Array.from(selectedIds)

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Sessions ({total})</h2>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">
            {selectedIds.size === 0 && "Select 2 sessions to compare"}
            {selectedIds.size === 1 && "1 selected — select 1 more"}
          </span>
          {selectedIds.size === 2 && (
            <Button
              variant="default"
              size="sm"
              className="bg-blue-600 hover:bg-blue-700 text-white"
              onClick={() => {
                router.push(`/compare?ids=${selectedArr[0]},${selectedArr[1]}`)
              }}
            >
              Compare Selected
            </Button>
          )}
          {total > 0 && (
            <Button variant="destructive" size="sm" onClick={() => setDeleteDialogOpen(true)}>
              Delete All
            </Button>
          )}
        </div>
      </div>

      {uploadError && (
        <span className="text-xs text-red-500">{uploadError}</span>
      )}

      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete All Sessions</DialogTitle>
            <DialogDescription className="text-base">
              将从 Insight 数据库中移除全部 {total} 个 session 的分析数据。原始会话文件不受影响，可随时重新导入。
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteDialogOpen(false)}>取消</Button>
            <Button variant="destructive" onClick={() => { setDeleteDialogOpen(false); handleDeleteAll() }}>确认删除全部</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <Dialog open={confirmDeleteId !== null} onOpenChange={(open) => { if (!open) setConfirmDeleteId(null) }}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Session</DialogTitle>
            <DialogDescription className="text-base">
              将从 Insight 数据库中移除该 session 的分析数据。原始会话文件不受影响，可随时重新导入。
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmDeleteId(null)}>取消</Button>
            <Button variant="destructive" onClick={() => {
              const item = items.find(i => i.sessionId === confirmDeleteId)
              setConfirmDeleteId(null)
              if (item) handleDeleteOne(item)
            }}>确认删除</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <Dialog open={uploadDialogItem !== null} onOpenChange={(open) => { if (!open) setUploadDialogItem(null) }}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>上传到 CANNBay</DialogTitle>
          </DialogHeader>
          <textarea
            className="w-full min-h-[240px] text-sm p-3 border rounded-md bg-muted/30 resize-y"
            value={uploadDescription}
            onChange={(e) => setUploadDescription(e.target.value)}
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setUploadDialogItem(null)}>取消</Button>
            <Button
              variant="default"
              className="bg-blue-600 hover:bg-blue-700 text-white"
              onClick={() => { if (uploadDialogItem) handleUpload(uploadDialogItem, uploadDescription) }}
              disabled={!uploadDescription.trim()}
            >
              确认上传
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <div className="border rounded-lg">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="text-xs w-16"></TableHead>
              <TableHead className="text-xs">Query</TableHead>
              <TableHead className="text-xs">Type</TableHead>
              <TableHead className="text-xs whitespace-nowrap">Start → End</TableHead>
              <TableHead className="text-xs text-left whitespace-nowrap">Model</TableHead>
              <TableHead className="text-xs">Tokens</TableHead>
              <TableHead className="text-xs">Cost</TableHead>
              <TableHead className="text-xs">Duration</TableHead>
              <TableHead className="text-xs">Tool Calls</TableHead>
              <TableHead className="text-xs">Subagents</TableHead>
              <TableHead className="text-xs w-8"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.length === 0 && (
              <TableRow>
                <TableCell colSpan={11} className="text-center text-muted-foreground py-8">
                  No sessions found. Import a session to get started.
                </TableCell>
              </TableRow>
            )}
            {items.map((item) => {
              const uStatus = uploadStatus[item.sessionId] ?? 'idle'
              const eStatus = exportStatus[item.sessionId] ?? 'idle'
              return (
                <TableRow key={item.sessionId} className={selectedIds.has(item.sessionId) ? 'bg-blue-500/10' : selectedIds.size >= 2 ? 'opacity-50' : ''}>
                  <TableCell className="w-8 px-2">
                    <Checkbox
                      checked={selectedIds.has(item.sessionId)}
                      onCheckedChange={() => handleToggle(item.sessionId)}
                      disabled={selectedIds.size >= 2 && !selectedIds.has(item.sessionId)}
                    />
                  </TableCell>
                  <TableCell className="max-w-[300px] truncate text-xs">
                    <Link
                      href={`/session/${item.taskId}?framework=${item.framework ?? 'unknown'}`}
                      className="text-primary hover:underline"
                    >
                      {item.query ?? item.taskId}
                    </Link>
                  </TableCell>
                  <TableCell className="text-xs">
                    {item.framework === 'opencode' ? (
                      <Badge variant="blue">OpenCode</Badge>
                    ) : item.framework === 'claude-code' ? (
                      <Badge variant="orange">Claude</Badge>
                    ) : item.framework === BRAND_SOURCE_TYPE ? (
                      <Badge variant="purple">{BRAND_NAME}</Badge>
                    ) : '—'}
                  </TableCell>
                  <TableCell className="text-xs whitespace-nowrap">
                    {formatTime(item.startTime)} → {formatTime(item.endTime)}
                  </TableCell>
                  <TableCell className="text-xs whitespace-nowrap pl-0">{item.model ?? '—'}</TableCell>
                  <TableCell className="text-xs tabular-nums">{formatTokens(item.totalTokens)}</TableCell>
                  <TableCell className="text-xs tabular-nums">{formatCost(item.totalCost)}</TableCell>
                  <TableCell className="text-xs tabular-nums">{formatDuration(item.totalLatencyMs)}</TableCell>
                  <TableCell className="text-xs tabular-nums">{item.totalToolCallCount}</TableCell>
                  <TableCell className="text-xs tabular-nums">{item.totalSubagentCount}</TableCell>
                  <TableCell className="px-2">
                    <div className="flex items-center gap-2">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="size-6 text-muted-foreground hover:text-red-600"
                        onClick={() => setConfirmDeleteId(item.sessionId)}
                        title="Delete session"
                      >
                        <TrashIcon className="size-3.5" />
                      </Button>
                      <Button
                        variant="outline"
                        size="icon"
                        className="size-6"
                        onClick={() => handleExport(item)}
                        disabled={eStatus === 'exporting'}
                        title="Export DB"
                      >
                        {eStatus === 'idle' && <DownloadIcon className="size-3.5" />}
                        {eStatus === 'exporting' && <LoaderIcon className="size-3.5 animate-spin" />}
                        {eStatus === 'done' && <CheckCircleIcon className="size-3.5 text-green-600" />}
                        {eStatus === 'error' && <XIcon className="size-3.5 text-red-500" />}
                      </Button>
                      <Button
                        variant="outline"
                        size="icon"
                        className="size-6"
                        onClick={() => { setUploadDescription(buildUploadDescription(item)); setUploadDialogItem(item) }}
                        disabled={uStatus === 'uploading'}
                        title="Upload to CANNBay"
                      >
                        {uStatus === 'idle' && <UploadIcon className="size-3.5" />}
                        {uStatus === 'uploading' && <LoaderIcon className="size-3.5 animate-spin" />}
                        {uStatus === 'done' && <CheckIcon className="size-3.5 text-green-600" />}
                        {uStatus === 'error' && <XIcon className="size-3.5 text-red-500" />}
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-4">
          <p className="text-sm text-muted-foreground">
            Page {page} of {totalPages} ({total} sessions)
          </p>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={!hasPrev}
              onClick={() => {
                const url = new URL(window.location.href);
                url.searchParams.set('page', String(page - 1));
                window.location.href = url.toString();
              }}
            >
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={!hasNext}
              onClick={() => {
                const url = new URL(window.location.href);
                url.searchParams.set('page', String(page + 1));
                window.location.href = url.toString();
              }}
            >
              Next
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
