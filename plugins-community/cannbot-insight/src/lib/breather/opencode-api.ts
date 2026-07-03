// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

interface OpenCodeSession {
  id: string
  title: string | null
  parentID: string | null
  projectID: string
  directory: string | null
  location: {
    directory: string | null
    workspaceID: string | null
  } | null
  agent: string | null
  model: {
    id: string
    providerID: string
    variant: string | null
  } | null
  cost: number
  tokens: {
    input: number
    output: number
    reasoning: number
    cache: { read: number; write: number }
  }
  time: {
    created: number
    updated: number
    archived: number | null
  }
}

interface V1MessageInfo {
  id: string
  role: string
  modelID: string | null
  providerID: string | null
  agent: string | null
  finish: string | null
  tokens: {
    total: number
    input: number
    output: number
    reasoning: number
    cache: { read: number; write: number }
  } | null
  cost: number | null
  time: {
    created: number
    completed: number | null
  } | null
}

export interface V1Message {
  info: V1MessageInfo
  parts: any[]
}

interface ModelInfo {
  id: string
  providerID: string
  limit: {
    context: number
    input: number | null
    output: number
  }
}

export class OpenCodeClient {
  private baseUrl: string

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl.replace(/\/+$/, "")
  }

  private async request(path: string, options?: RequestInit): Promise<Response> {
    return fetch(`${this.baseUrl}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    })
  }

  private async safeJson<T>(res: Response, context: string): Promise<T | null> {
    try {
      return await res.json() as T
    } catch (err) {
      console.error(`[OpenCodeClient] ${context}: JSON parse failed`, err)
      return null
    }
  }

  async health(): Promise<boolean> {
    try {
      const res = await this.request("/api/health")
      return res.ok
    } catch {
      return false
    }
  }

  async listSessions(): Promise<OpenCodeSession[]> {
    const res = await this.request("/session?limit=500")
    if (!res.ok) return []
    const data = await this.safeJson<any>(res, "listSessions")
    if (!data) return []
    if (Array.isArray(data)) return data as OpenCodeSession[]
    return (data.data ?? data.items ?? []) as OpenCodeSession[]
  }

  async getSession(id: string): Promise<OpenCodeSession | null> {
    const res = await this.request(`/session/${encodeURIComponent(id)}`)
    if (!res.ok) return null
    const data = await this.safeJson<any>(res, "getSession")
    if (!data) return null
    return (data.data ?? data) as OpenCodeSession
  }

  async getMessages(sessionId: string, limit?: number): Promise<V1Message[]> {
    const query = limit ? `?limit=${limit}` : ""
    const res = await this.request(`/session/${encodeURIComponent(sessionId)}/message${query}`)
    if (!res.ok) return []
    const data = await this.safeJson<any>(res, "getMessages")
    if (!data) return []
    if (Array.isArray(data)) return data as V1Message[]
    if (data.data && Array.isArray(data.data)) return data.data as V1Message[]
    return []
  }

  async getModels(): Promise<ModelInfo[]> {
    const res = await this.request("/api/model")
    if (!res.ok) return []
    const data = await this.safeJson<any>(res, "getModels")
    if (!data) return []
    return (data.data ?? data ?? []) as ModelInfo[]
  }

  async getContextLimit(modelId: string): Promise<number | null> {
    const models = await this.getModels()
    const model = models.find(m => m.id === modelId)
    return model?.limit?.context ?? null
  }
}
