// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import type { AIProviderConfig } from "@/lib/ai/analyzer"
import { BRAND_SLUG } from "@/lib/branding"

export const AI_PROVIDER_STORAGE_KEY = `${BRAND_SLUG}-ai-provider`

export function loadProviderConfig(): AIProviderConfig | null {
  if (typeof window === "undefined") return null
  try {
    const raw = localStorage.getItem(AI_PROVIDER_STORAGE_KEY)
    if (!raw) return null
    return JSON.parse(raw) as AIProviderConfig
  } catch {
    return null
  }
}

export function saveProviderConfig(config: AIProviderConfig): void {
  if (typeof window === "undefined") return
  localStorage.setItem(AI_PROVIDER_STORAGE_KEY, JSON.stringify(config))
  emit()
}

export function clearProviderConfig(): void {
  if (typeof window === "undefined") return
  localStorage.removeItem(AI_PROVIDER_STORAGE_KEY)
  emit()
}

// Reactive store so consumers re-render when config changes.
const listeners = new Set<() => void>()
function emit() {
  for (const l of listeners) l()
}

export function subscribeProviderConfig(cb: () => void): () => void {
  listeners.add(cb)
  return () => { listeners.delete(cb) }
}

export function getProviderConfigSnapshot(): string | null {
  if (typeof window === "undefined") return null
  return localStorage.getItem(AI_PROVIDER_STORAGE_KEY)
}

export function getProviderConfigServerSnapshot(): string | null {
  return null
}
