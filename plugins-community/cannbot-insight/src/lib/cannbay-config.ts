// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { BRAND_SLUG } from "@/lib/branding"

export const CANNBAY_CONFIG_STORAGE_KEY = `${BRAND_SLUG}-cannbay-config`

export interface CannbayConfig {
  address: string
  account: string
  password: string
}

export function loadCannbayConfig(): CannbayConfig | null {
  if (typeof window === "undefined") return null
  try {
    const raw = localStorage.getItem(CANNBAY_CONFIG_STORAGE_KEY)
    if (!raw) return null
    return JSON.parse(raw) as CannbayConfig
  } catch {
    return null
  }
}

export function saveCannbayConfig(config: CannbayConfig): void {
  if (typeof window === "undefined") return
  localStorage.setItem(CANNBAY_CONFIG_STORAGE_KEY, JSON.stringify(config))
}

export function clearCannbayConfig(): void {
  if (typeof window === "undefined") return
  localStorage.removeItem(CANNBAY_CONFIG_STORAGE_KEY)
}
