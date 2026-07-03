// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { ApiError, NetworkError } from './errors';
import type {
  ApiSessionListResponse,
  ApiSessionDetailResponse,
  ApiGlobalStatsResponse,
  ApiSessionStatsResponse,
  ApiExecutionsResponse,
  ApiTurnsResponse,
  ApiTurnDetailResponse,
  ApiSearchResponse,
  WorkflowTree,
  ApiBridgesResponse,
  ApiImportResponse,
  ApiImportableSessionsResponse,
  ApiDeleteResponse,
  ApiAnalyzeWorkflowResponse,
  ApiTestProviderResponse,
  ApiAIProviderConfig,
  ApiUploadResponse,
} from './types';
import fs from 'node:fs';
import path from 'node:path';

export interface ClientConfig {
  baseUrl: string;
  timeout: number;
  retries: number;
  retryDelay: number;
  authToken?: string;
}

export class InsightClient {
  private config: ClientConfig;

  constructor(baseUrl: string = 'http://localhost:21025', config?: Partial<ClientConfig>) {
    this.config = {
      baseUrl,
      timeout: 15000,
      retries: 2,
      retryDelay: 1000,
      ...config,
    };
  }

  getConfig(): ClientConfig {
    return { ...this.config };
  }

  private async request<T>(
    method: 'GET' | 'POST' | 'DELETE',
    apiPath: string,
    params?: Record<string, string | number | boolean>,
    body?: Record<string, unknown>,
    timeoutOverride?: number,
  ): Promise<T> {
    const url = new URL(apiPath, this.config.baseUrl);
    if (params && method === 'GET') {
      for (const [key, val] of Object.entries(params)) {
        url.searchParams.set(key, String(val));
      }
    }

    const requestTimeout = timeoutOverride ?? this.config.timeout;

    let lastError: Error | null = null;
    for (let attempt = 0; attempt <= this.config.retries; attempt++) {
      try {
        const init: RequestInit = {
          method,
          headers: {
            'Content-Type': 'application/json',
            ...(this.config.authToken ? { Authorization: `Bearer ${this.config.authToken}` } : {}),
          },
          signal: AbortSignal.timeout(requestTimeout),
        };
        if (body && (method === 'POST' || method === 'DELETE')) {
          init.body = JSON.stringify(body);
        }

        const res = await fetch(url.toString(), init);
        if (!res.ok) {
          const errorBody = await res.text();
          let errorMessage: string;
          try {
            const parsed = JSON.parse(errorBody);
            errorMessage = parsed.error ?? errorBody;
          } catch {
            errorMessage = errorBody;
          }
          if (res.status >= 400 && res.status < 500) {
            throw new ApiError(res.status, errorMessage, false);
          }
          throw new ApiError(res.status, errorMessage, true);
        }

        return await res.json() as T;
      } catch (err) {
        lastError = err instanceof ApiError
          ? err
          : new NetworkError(err instanceof Error ? err.message : String(err));

        if (err instanceof ApiError && !err.retryable) throw err;
        if (attempt < this.config.retries) {
          await new Promise(r => setTimeout(r, this.config.retryDelay * (attempt + 1)));
        }
      }
    }
    throw lastError ?? new NetworkError('Unknown error');
  }

  async listSessions(opts?: {
    page?: number;
    pageSize?: number;
    isSubagent?: boolean;
    user?: string;
  }): Promise<ApiSessionListResponse> {
    const params: Record<string, string | number | boolean> = {};
    if (opts?.page) params.page = opts.page;
    if (opts?.pageSize) params.pageSize = opts.pageSize;
    if (opts?.isSubagent !== undefined) params.isSubagent = opts.isSubagent;
    if (opts?.user) params.user = opts.user;
    return this.request<ApiSessionListResponse>('GET', '/api/observe/data', params);
  }

  async getSession(taskId: string): Promise<ApiSessionDetailResponse> {
    return this.request<ApiSessionDetailResponse>('GET', '/api/observe/session', { taskId });
  }

  async getStats(taskId?: string): Promise<ApiGlobalStatsResponse | ApiSessionStatsResponse> {
    const params: Record<string, string> = {};
    if (taskId) params.taskId = taskId;
    return this.request<ApiGlobalStatsResponse | ApiSessionStatsResponse>('GET', '/api/observe/stats', params);
  }

  async getExecutions(taskId: string): Promise<ApiExecutionsResponse> {
    return this.request<ApiExecutionsResponse>('GET', '/api/observe/executions', { taskId });
  }

  async getTurns(taskId: string, opts?: { isSubagent?: boolean; role?: string }): Promise<ApiTurnsResponse> {
    const params: Record<string, string | boolean> = { taskId };
    if (opts?.isSubagent !== undefined) params.isSubagent = opts.isSubagent;
    if (opts?.role) params.role = opts.role;
    return this.request<ApiTurnsResponse>('GET', '/api/observe/session/turns', params);
  }

  async getTurnDetail(turnId: string): Promise<ApiTurnDetailResponse> {
    return this.request<ApiTurnDetailResponse>('GET', `/api/observe/session/turns/${turnId}`);
  }

  async searchTurns(taskId: string, keyword: string): Promise<ApiSearchResponse> {
    return this.request<ApiSearchResponse>('GET', '/api/observe/session/turns/search', { taskId, keyword });
  }

  async getWorkflow(taskId: string): Promise<WorkflowTree> {
    return this.request<WorkflowTree>('GET', '/api/observe/session/workflow', { taskId });
  }

  async getBridges(taskId: string): Promise<ApiBridgesResponse> {
    return this.request<ApiBridgesResponse>('GET', '/api/observe/session/bridges', { taskId });
  }

  async importSession(source: string, filePath: string, sessionId: string): Promise<ApiImportResponse> {
    return this.request<ApiImportResponse>('POST', '/api/ingest/import-file', undefined, {
      source,
      filePath,
      sessionId,
    }, 60000);
  }

  async listImportableSessions(source: string, filePath: string): Promise<ApiImportableSessionsResponse> {
    return this.request<ApiImportableSessionsResponse>('POST', '/api/ingest/import-file/sessions', undefined, {
      source,
      filePath,
    });
  }

  async deleteSession(taskId?: string, deleteAll?: boolean): Promise<ApiDeleteResponse> {
    return this.request<ApiDeleteResponse>('DELETE', '/api/ingest/delete-session', undefined, {
      ...(taskId ? { taskId } : {}),
      ...(deleteAll ? { deleteAll } : {}),
    });
  }

  async analyzeWorkflow(taskId: string, provider: ApiAIProviderConfig): Promise<ApiAnalyzeWorkflowResponse> {
    return this.request<ApiAnalyzeWorkflowResponse>('POST', '/api/ai/analyze-workflow', undefined, {
      taskId,
      provider,
    });
  }

  async testProvider(baseUrl: string, apiKey: string): Promise<ApiTestProviderResponse> {
    return this.request<ApiTestProviderResponse>('POST', '/api/ai/test-provider', undefined, {
      baseUrl,
      apiKey,
    });
  }

  async exportSession(taskId: string, outputPath: string): Promise<{ size: number }> {
    const url = new URL('/api/ingest/export-session', this.config.baseUrl);
    const init: RequestInit = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(this.config.authToken ? { Authorization: `Bearer ${this.config.authToken}` } : {}),
      },
      body: JSON.stringify({ taskId }),
      signal: AbortSignal.timeout(60000),
    };

    const res = await fetch(url.toString(), init);
    if (!res.ok) {
      const errorBody = await res.text();
      let errorMessage: string;
      try {
        const parsed = JSON.parse(errorBody);
        errorMessage = parsed.error ?? errorBody;
      } catch {
        errorMessage = errorBody;
      }
      throw new ApiError(res.status, errorMessage, res.status >= 500);
    }

    const arrayBuffer = await res.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    fs.writeFileSync(outputPath, buffer);

    return { size: buffer.length };
  }

  async uploadSession(taskId: string, description: string, framework?: string): Promise<ApiUploadResponse> {
    return this.request<ApiUploadResponse>('POST', '/api/ingest/upload-session', undefined, {
      taskId,
      framework: framework ?? 'unknown',
      description,
    }, 120000);
  }
}
