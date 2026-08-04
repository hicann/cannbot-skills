// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { BRAND_SLUG } from "@/lib/branding"

export type AgentVersion = "v1" | "v2" | "v3" | "v4"

const STORAGE_KEY = `${BRAND_SLUG}-agent-version`

const listeners = new Set<() => void>()
function emit() {
  for (const l of listeners) l()
}

export function subscribeAgentVersion(cb: () => void): () => void {
  listeners.add(cb)
  return () => { listeners.delete(cb) }
}

export function getAgentVersionSnapshot(): string {
  if (typeof window === "undefined") return "v1"
  return localStorage.getItem(STORAGE_KEY) ?? "v1"
}

export function getAgentVersionServerSnapshot(): string {
  return "v1"
}

export function saveAgentVersion(v: AgentVersion): void {
  if (typeof window === "undefined") return
  localStorage.setItem(STORAGE_KEY, v)
  emit()
}
