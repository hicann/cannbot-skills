// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

export type Protocol = 'anthropic' | 'openai';

export interface AnthropicUsage {
  input_tokens?: number;
  output_tokens?: number;
  cache_read_input_tokens?: number;
  cache_creation_input_tokens?: number;
}

export interface OpenAIUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  prompt_tokens_details?: {
    cached_tokens?: number;
  };
}

export interface AnthropicContentBlock {
  type: string;
  text?: string;
  thinking?: string;
  id?: string;
  name?: string;
  input?: Record<string, unknown>;
  tool_use_id?: string;
  content?: string | Record<string, unknown> | unknown[];
}

export interface OpenAIContentBlock {
  type: string;
  text?: string;
  id?: string;
  function?: { name?: string; arguments?: string };
  tool_call_id?: string;
  content?: string | unknown[];
}

export interface AnthropicMessage {
  role: string;
  content?: string | AnthropicContentBlock[];
}

export interface OpenAIMessage {
  role: string;
  content?: string | OpenAIContentBlock[] | null;
}

export interface ProxyRecord {
  sid: string;
  protocol: Protocol;
  receivedAt: number;
  completedAt: number;
  latencyMs: number;
  // Time to first token (streaming only); null for non-stream responses.
  ttftMs: number | null;
  request: {
    path: string;
    model: string | null;
    body: unknown;
  };
  response: {
    status: number;
    model: string | null;
    stop_reason: string | null;
    content: AnthropicContentBlock[] | unknown;
    usage: AnthropicUsage | OpenAIUsage | null;
  };
  // opencode sends `x-session-id`/`x-session-affinity` headers (value: ses_xxx).
  // Subagents are child sessions with a DIFFERENT x-session-id, so this is the
  // reliable wire-level signal for subagent routing (claude relies on system
  // prompt text cc_is_subagent instead). Null for protocols that don't set it.
  xSessionId?: string | null;
  // opencode child sessions declare their parent via the x-parent-session-id
  // request header (verified on opencode 1.17.9) — deterministic subagent
  // routing with the parent's real session id. null for main/claude requests.
  parentSessionId?: string | null;
}
