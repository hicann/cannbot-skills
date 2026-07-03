// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

export interface SessionListItem {
  id: string;
  createdAt: string;
  firstQuery: string | null;
  turnCount: number;
  modelName: string | null;
  version?: string | null;
}

export interface RawInteraction {
  role: string;
  content: string | null;
  timestamp: string;
  timeInfo: {
    created: number;
    completed?: number;
  } | null;
  agent: string | null;
  subagent_name: string | null;
  subagent_session_id: string | null;
  subagent_type: string | null;
  tool_calls: ToolCallInfo[] | null;
  usage: TokenUsage | null;
  model: string | null;
  modelID: string | null;
  providerID: string | null;
  latency: number | null;
  finish_reason: string | null;
}

export interface ToolCallInfo {
  toolCallId: string;
  toolName: string;
  argsJson: string | null;
  resultJson: string | null;
  state: string;
}

export interface TokenUsage {
  total: number;
  input: number;
  output: number;
  reasoning: number;
  cacheRead: number;
  cacheWrite: number;
  cost: number;
  inputMessagesTokens: number;
}
