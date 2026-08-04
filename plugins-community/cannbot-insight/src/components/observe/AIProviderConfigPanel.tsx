"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"
import type { AIProviderConfig } from "@/lib/ai/analyzer"
import {
  loadProviderConfig,
  saveProviderConfig,
  clearProviderConfig,
} from "@/lib/ai-provider-config"

interface Props {
  onConfigChange?: (config: AIProviderConfig | null) => void
  compact?: boolean
}

export function AIProviderConfigPanel({ onConfigChange, compact }: Props) {
  const saved = loadProviderConfig()
  const [baseUrl, setBaseUrl] = useState(saved?.baseUrl ?? "https://dashscope.aliyuncs.com/compatible-mode/v1")
  const [apiKey, setApiKey] = useState(saved?.apiKey ?? "")
  const [model, setModel] = useState(saved?.model ?? "qwen3.7-max")
  const [savedFlag, setSavedFlag] = useState(!!saved)
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null)
  const [testing, setTesting] = useState(false)

  const config: AIProviderConfig = { baseUrl, apiKey, model }
  const isAnthropicPath = baseUrl.includes("/apps/anthropic")

  function handleSave() {
    saveProviderConfig(config)
    setSavedFlag(true)
    setTestResult(null)
    onConfigChange?.(config)
  }

  function handleClear() {
    clearProviderConfig()
    setSavedFlag(false)
    setTestResult(null)
    onConfigChange?.(null)
  }

  function handleTest() {
    setTesting(true)
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
      .finally(() => setTesting(false))
  }

  return (
    <Card>
      <CardContent className={compact ? "py-3 space-y-3" : "p-4 space-y-3"}>
        <div className="flex items-center gap-2 text-sm font-medium">
          <span>LLM API 配置（OpenAI 兼容）</span>
          {savedFlag && <span className="text-xs text-green-600">✓ 已保存</span>}
        </div>

        {isAnthropicPath && (
          <div className="text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-500/10 px-2 py-1.5 rounded">
            ❌ /apps/anthropic 是 Anthropic Messages 格式，不支持。请改用 /compatible-mode/v1 的 OpenAI 兼容地址
          </div>
        )}

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground w-16 shrink-0">Base URL</span>
            <Input value={baseUrl} onChange={(e) => { setBaseUrl(e.target.value); setSavedFlag(false) }} className="h-7 text-xs" placeholder="https://..." />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground w-16 shrink-0">API Key</span>
            <Input type="password" value={apiKey} onChange={(e) => { setApiKey(e.target.value); setSavedFlag(false) }} className="h-7 text-xs" placeholder="sk-..." />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground w-16 shrink-0">Model</span>
            <Input value={model} onChange={(e) => { setModel(e.target.value); setSavedFlag(false) }} className="h-7 text-xs" placeholder="qwen3.7-max" />
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button size="sm" className="text-xs" onClick={handleSave} disabled={!apiKey}>
            保存配置
          </Button>
          <Button size="sm" variant="outline" className="text-xs" onClick={handleTest} disabled={!apiKey || testing}>
            {testing ? "测试中…" : "测试连接"}
          </Button>
          {savedFlag && (
            <Button size="sm" variant="ghost" className="text-xs" onClick={handleClear}>
              清除
            </Button>
          )}
          {testResult && (
            <span className={cn("text-xs", testResult.success ? "text-green-600" : "text-red-600")}>
              {testResult.message}
            </span>
          )}
        </div>

        <p className="text-xs text-muted-foreground">配置保存在浏览器本地。Audit 一键审计报告功能需使用此配置。</p>
      </CardContent>
    </Card>
  )
}
