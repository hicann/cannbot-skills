"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"
import { WorkflowTreeView } from "@/components/observe/WorkflowTreeView"
import type { WorkflowTree } from "@/lib/ingest/phase-split"
import type { AIProviderConfig } from "@/lib/ai/analyzer"

import { BRAND_SLUG } from "@/lib/branding"

const STORAGE_KEY = `${BRAND_SLUG}-ai-provider`

interface WorkflowAIViewProps {
  taskId: string
  turnsCount: number
  bridgesCount: number
  bridges: Array<{
    bridgeId: string
    dispatchContent: string | null
    dispatchTimestamp: string | null
    responseContent: string | null
    responseTimestamp: string | null
    subagentName: string | null
    subagentType: string | null
    status: string
    subagentTokens: number
    subagentLatencyMs: number
  }>
  turns: Array<{
    turnIndex: number
    subagentSessionId: string | null
    isSubagent: boolean
    role: string
  }>
  result: WorkflowTree | null
  isAnalyzing: boolean
  error: string | null
  onAnalyze: (provider: AIProviderConfig) => void
  onClearResult: () => void
}

function loadProviderConfig(): AIProviderConfig | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    return JSON.parse(raw) as AIProviderConfig
  } catch {
    return null
  }
}

function saveProviderConfig(config: AIProviderConfig) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(config))
}

function clearProviderConfig() {
  localStorage.removeItem(STORAGE_KEY)
}

function ConfigPanel({
  onSave,
  onTest,
  testResult,
}: {
  onSave: (config: AIProviderConfig) => void
  onTest: (config: AIProviderConfig) => void
  testResult: { success: boolean; message: string } | null
}) {
  const saved = loadProviderConfig()
  const [baseUrl, setBaseUrl] = useState(saved?.baseUrl ?? "https://dashscope.aliyuncs.com/compatible-mode/v1")
  const [apiKey, setApiKey] = useState(saved?.apiKey ?? "")
  const [model, setModel] = useState(saved?.model ?? "qwen3.7-max")

  const config: AIProviderConfig = { baseUrl, apiKey, model }
  const isAnthropicPath = baseUrl.includes("/apps/anthropic")

  return (
    <Card>
      <CardContent className="py-3 space-y-3">
        <div className="flex items-center gap-2 text-sm font-medium">
          <span>⚠️ AI 阶段划分需要配置 LLM API（仅支持 OpenAI 兼容模式）</span>
          <Badge variant="purple">(beta)</Badge>
        </div>

        <p className="text-xs text-muted-foreground">
          AI 根据对话语义自动划分工作流阶段，比正则解析更准确。结果用 WorkflowTreeView 展示。
        </p>

        {isAnthropicPath && (
          <div className="text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-500/10 px-2 py-1.5 rounded">
            ❌ /apps/anthropic 是 Anthropic Messages 格式，不支持。请改用 /compatible-mode/v1 的 OpenAI 兼容地址
          </div>
        )}

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground w-16 shrink-0">Base URL</span>
            <Input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} className="h-7 text-xs" placeholder="https://..." />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground w-16 shrink-0">API Key</span>
            <Input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} className="h-7 text-xs" placeholder="sk-..." />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground w-16 shrink-0">Model</span>
            <Input value={model} onChange={(e) => setModel(e.target.value)} className="h-7 text-xs" placeholder="qwen3.7-max" />
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button size="sm" className="text-xs" onClick={() => onSave(config)} disabled={!apiKey}>
            保存配置
          </Button>
          <Button size="sm" variant="outline" className="text-xs" onClick={() => onTest(config)} disabled={!apiKey}>
            测试连接
          </Button>
          {testResult && (
            <span className={cn("text-xs", testResult.success ? "text-green-600" : "text-red-600")}>
              {testResult.message}
            </span>
          )}
        </div>

        <p className="text-xs text-muted-foreground">配置保存在浏览器本地，不会上传到服务器</p>
      </CardContent>
    </Card>
  )
}

function ReadyPanel({
  turnsCount,
  bridgesCount,
  onAnalyze,
  onClearConfig,
}: {
  turnsCount: number
  bridgesCount: number
  onAnalyze: () => void
  onClearConfig: () => void
}) {
  return (
    <Card>
      <CardContent className="py-3 space-y-2">
        <div className="flex items-center gap-2">
          <Badge variant="purple">✦ AI Ready</Badge>
          <span className="text-sm font-medium">配置已完成</span>
        </div>
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span>源数据: {turnsCount} turns + {bridgesCount} bridges</span>
          <span>估算: ~25K input tokens, ~2K output tokens</span>
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" className="text-xs" onClick={onAnalyze}>
            🤖 AI 划分阶段
          </Button>
          <Button size="sm" variant="ghost" className="text-xs" onClick={onClearConfig}>
            清除配置
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

export function WorkflowAIView({
  taskId,
  turnsCount,
  bridgesCount,
  bridges,
  turns,
  result,
  isAnalyzing,
  error,
  onAnalyze,
  onClearResult,
}: WorkflowAIViewProps) {
  const [provider, setProvider] = useState<AIProviderConfig | null>(loadProviderConfig())
  const [showConfig, setShowConfig] = useState(!provider)
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null)

  function handleSave(config: AIProviderConfig) {
    saveProviderConfig(config)
    setProvider(config)
    setShowConfig(false)
    setTestResult(null)
  }

  function handleTest(config: AIProviderConfig) {
    setTestResult(null)
    fetch("/api/ai/test-provider", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ baseUrl: config.baseUrl, apiKey: config.apiKey }),
      signal: AbortSignal.timeout(15000),
    })
      .then(res => res.json())
      .then(data => setTestResult({ success: data.success, message: data.message }))
      .catch(e => setTestResult({ success: false, message: `❌ ${e.message}` }))
  }

  function handleClearConfig() {
    clearProviderConfig()
    setProvider(null)
    setShowConfig(true)
    onClearResult()
    setTestResult(null)
  }

  function handleAnalyze() {
    if (provider) onAnalyze(provider)
  }

  const hasResult = result !== null

  return (
    <div className="flex flex-col h-full">
      {showConfig && (
        <div className="shrink-0 p-4">
          <ConfigPanel onSave={handleSave} onTest={handleTest} testResult={testResult} />
        </div>
      )}

      {!showConfig && !hasResult && !isAnalyzing && provider && (
        <div className="shrink-0 p-4">
          <ReadyPanel
            turnsCount={turnsCount}
            bridgesCount={bridgesCount}
            onAnalyze={handleAnalyze}
            onClearConfig={handleClearConfig}
          />
        </div>
      )}

      {isAnalyzing && (
        <div className="shrink-0 p-4">
          <Card>
            <CardContent className="py-3 flex items-center gap-3">
              <div className="animate-spin h-5 w-5 border-2 border-primary border-t-transparent rounded-full" />
              <span className="text-sm">AI 正在划分阶段（预计 30-120 秒，取决于数据量）...</span>
            </CardContent>
          </Card>
        </div>
      )}

      {error && !isAnalyzing && (
        <div className="shrink-0 p-4">
          <Card>
            <CardContent className="py-2 text-xs text-red-600 dark:text-red-400">
              ❌ {error}
            </CardContent>
          </Card>
        </div>
      )}

      {hasResult && result && !isAnalyzing && (
        <>
          <div className="shrink-0 p-4 flex items-center gap-2">
            <Badge variant="purple">✦ AI 阶段划分</Badge>
            <span className="text-xs text-muted-foreground">
              {result.summary.totalPhases} 阶段 · {result.summary.totalSteps} 步骤 · {result.summary.totalCheckpoints} 检查点
            </span>
            <Button size="sm" variant="outline" className="text-xs" onClick={handleAnalyze}>
              🤖 重新划分
            </Button>
            <Button size="sm" variant="ghost" className="text-xs" onClick={handleClearConfig}>
              清除配置
            </Button>
          </div>
          <div className="flex-1 min-h-0 overflow-y-auto">
            <WorkflowTreeView
              workflow={result}
              bridges={bridges}
              turns={turns}
              taskId={taskId}
              onViewTurnsInteraction={() => {}}
            />
          </div>
        </>
      )}
    </div>
  )
}
