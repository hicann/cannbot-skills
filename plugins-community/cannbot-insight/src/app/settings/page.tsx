"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent } from "@/components/ui/card"
import { ArrowLeftIcon } from "lucide-react"
import { BRAND_NAME, BAY_NAME } from "@/lib/branding"
import { VERSION_DISPLAY } from "@/lib/version"
import { AIProviderConfigPanel } from "@/components/observe/AIProviderConfigPanel"
import {
  loadCannbayConfig,
  saveCannbayConfig,
  clearCannbayConfig,
  type CannbayConfig,
} from "@/lib/cannbay-config"

export default function SettingsPage() {
  const saved = loadCannbayConfig()
  const [address, setAddress] = useState(saved?.address ?? "")
  const [account, setAccount] = useState(saved?.account ?? "")
  const [password, setPassword] = useState(saved?.password ?? "")
  const [savedFlag, setSavedFlag] = useState(!!saved)

  const config: CannbayConfig = { address, account, password }

  function handleSave() {
    saveCannbayConfig(config)
    setSavedFlag(true)
  }

  function handleClear() {
    clearCannbayConfig()
    setAddress("")
    setAccount("")
    setPassword("")
    setSavedFlag(false)
  }

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <main className="flex w-full max-w-3xl flex-col gap-6 px-6 py-8 mx-auto">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold tracking-tight text-foreground">设置</h1>
            <span className="text-xs text-muted-foreground">{BRAND_NAME} {VERSION_DISPLAY}</span>
          </div>
          <Link href="/" className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md border text-sm font-medium hover:bg-accent/30 transition-colors">
            <ArrowLeftIcon className="size-4" />
            返回
          </Link>
        </div>

        <div className="space-y-2">
          <h2 className="text-sm font-semibold">LLM API 配置</h2>
          <p className="text-xs text-muted-foreground">
            用于 Audit 一键审计报告与 AI workflow 阶段划分。仅支持 OpenAI 兼容端点（如 /compatible-mode/v1）。配置存于浏览器本地，两 tab 共享。
          </p>
        </div>
        <AIProviderConfigPanel compact />

        <div className="h-px bg-border" />

        <div className="space-y-2">
          <h2 className="text-sm font-semibold">{BAY_NAME} 数据之湖</h2>
          <p className="text-xs text-muted-foreground">
            {BAY_NAME} 数据之湖的连接地址与账户凭据。当前仅保存留后用，暂未接入功能。
          </p>
        </div>
        <Card>
          <CardContent className="py-3 space-y-3">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-16 shrink-0">地址</span>
                <Input value={address} onChange={(e) => { setAddress(e.target.value); setSavedFlag(false) }} className="h-7 text-xs" placeholder="https://cannbot.example.com" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-16 shrink-0">账户</span>
                <Input value={account} onChange={(e) => { setAccount(e.target.value); setSavedFlag(false) }} className="h-7 text-xs" placeholder="用户名" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-16 shrink-0">密码</span>
                <Input type="password" value={password} onChange={(e) => { setPassword(e.target.value); setSavedFlag(false) }} className="h-7 text-xs" placeholder="••••••" />
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button size="sm" className="text-xs" onClick={handleSave} disabled={!address}>
                保存配置
              </Button>
              {savedFlag && (
                <Button size="sm" variant="ghost" className="text-xs" onClick={handleClear}>
                  清除
                </Button>
              )}
              {savedFlag && <span className="text-xs text-green-600">✓ 已保存</span>}
            </div>
            <p className="text-xs text-muted-foreground">配置保存在浏览器本地，不上传服务器。</p>
          </CardContent>
        </Card>
      </main>
    </div>
  )
}
