// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

export interface ApiSessionListItem {
  sessionId: string;
  taskId: string;
  query: string | null;
  startTime: string | null;
  endTime: string | null;
  totalTokens: number | null;
  totalCost: number | null;
  totalLatencyMs: number | null;
  totalToolCallCount: number | null;
  totalSkillLoadCount: number | null;
  totalSubagentCount: number | null;
  model: string | null;
  user: string | null;
  framework: string | null;
}

export interface ApiSessionListResponse {
  items: ApiSessionListItem[];
  total: number;
  page: number;
}

export interface ApiAgentItem {
  executionId: string;
  agentName: string | null;
  agentSessionId: string | null;
  isSubagent: boolean;
  parentExecutionId: string | null;
  tokens: number | null;
  maxSingleCallTokens: number | null;
  cost: number | null;
  toolCallCount: number | null;
  skillLoadCount: number | null;
  model: string | null;
  createdAt: string;
  latencyMs: number | null;
  firstPrompt: string | null;
}

export interface ApiSkillSummary {
  skillName: string;
  version: string | null;
  invocationCount: number;
}

export interface ApiSessionDetailResponse {
  sessionId: string;
  taskId: string;
  label: string | null;
  query: string | null;
  framework: string | null;
  startTime: string | null;
  endTime: string | null;
  totalTokens: number | null;
  totalInputTokens: number | null;
  totalOutputTokens: number | null;
  totalReasoningTokens: number | null;
  totalCacheReadTokens: number | null;
  totalCacheWriteTokens: number | null;
  totalCost: number | null;
  totalLatencyMs: number | null;
  totalToolCallCount: number | null;
  totalLlmCallCount: number | null;
  totalSkillLoadCount: number | null;
  totalSubagentCount: number | null;
  model: string | null;
  user: string | null;
  sourcePath: string | null;
  agents: ApiAgentItem[];
  skills: ApiSkillSummary[];
}

export interface ApiGlobalStatsResponse {
  totalSessions: number;
  totalTokens: number;
  totalCost: number;
  totalLatencyMs: number;
  avgLatencyMs: number;
}

export interface ApiSessionStatsResponse {
  taskId: string;
  totalTokens: number | null;
  totalInputTokens: number | null;
  totalOutputTokens: number | null;
  totalReasoningTokens: number | null;
  totalCacheReadTokens: number | null;
  totalCacheWriteTokens: number | null;
  totalCost: number | null;
  totalLatencyMs: number | null;
  totalToolCallCount: number | null;
  totalSkillLoadCount: number | null;
  totalSubagentCount: number | null;
  totalLlmCallCount: number | null;
}

export interface ApiExecutionItem {
  executionId: string;
  sessionId: string;
  agentName: string | null;
  agentSessionId: string | null;
  isSubagent: boolean;
  subagentType: string | null;
  subagentName: string | null;
  parentExecutionId: string | null;
  rootExecutionId: string | null;
  depth: number | null;
  tokens: number | null;
  inputTokens: number | null;
  outputTokens: number | null;
  reasoningTokens: number | null;
  cost: number | null;
  latencyMs: number | null;
  toolCallCount: number | null;
  toolCallErrorCount: number | null;
  llmCallCount: number | null;
  skillLoadCount: number | null;
  skillInvokeCount: number | null;
  model: string | null;
  createdAt: string;
  skills: ApiExecutionSkillInfo[];
}

export interface ApiExecutionSkillInfo {
  skillName: string;
  skillVersion: string | null;
  isPrimary: boolean | null;
}

export interface ApiExecutionsResponse {
  items: ApiExecutionItem[];
  root: ApiExecutionItem[];
  subagents: ApiExecutionItem[];
  totalExecutions: number;
  subagentCount: number;
}

export interface ApiToolCallBrief {
  toolCallId: string | null;
  toolName: string | null;
  state: string | null;
  durationMs: number | null;
}

export interface ApiSkillEventBrief {
  skillName: string | null;
  eventType: string | null;
  success: boolean | null;
}

export interface ApiTurnItem {
  turnId: string;
  turnIndex: number;
  role: string;
  contentSummary: string | null;
  agentName: string | null;
  isSubagent: boolean;
  subagentName: string | null;
  subagentSessionId: string | null;
  parentExecutionId: string | null;
  totalTokens: number | null;
  inputTokens: number | null;
  outputTokens: number | null;
  reasoningTokens: number | null;
  cacheReadTokens: number | null;
  cacheWriteTokens: number | null;
  inputMessagesCount: number | null;
  inputMessagesTokens: number | null;
  contextWindowPct: number | null;
  latencyMs: number | null;
  createdAt: string;
  completedAt: string | null;
  model: string | null;
  finishReason: string | null;
  toolCalls: ApiToolCallBrief[];
  skillEvents: ApiSkillEventBrief[];
}

export interface ApiTurnsResponse {
  items: ApiTurnItem[];
  total: number;
}

export interface ApiToolCallDetail {
  id: string;
  toolCallId: string | null;
  toolName: string | null;
  argsJson: string | null;
  resultJson: string | null;
  state: string | null;
  errorType: string | null;
  errorMessage: string | null;
  startedAt: string | null;
  completedAt: string | null;
  durationMs: number | null;
  dispatchBridgeId: string | null;
  isSkillRelated: boolean | null;
}

export interface ApiSkillEventDetail {
  id: string;
  skillName: string | null;
  skillVersion: string | null;
  eventType: string | null;
  success: boolean | null;
  errorMessage: string | null;
  argsJson: string | null;
  startedAt: string | null;
  completedAt: string | null;
  durationMs: number | null;
}

export interface ApiTurnDetailResponse {
  turnId: string;
  sessionId: string;
  turnIndex: number;
  role: string;
  content: string | null;
  contentJson: string | null;
  contentSummary: string | null;
  inputMessagesJson: string | null;
  inputMessagesCount: number | null;
  inputMessagesTokens: number | null;
  contextWindowPct: number | null;
  agentName: string | null;
  subagentName: string | null;
  subagentSessionId: string | null;
  isSubagent: boolean;
  totalTokens: number | null;
  inputTokens: number | null;
  outputTokens: number | null;
  reasoningTokens: number | null;
  cacheReadTokens: number | null;
  cacheWriteTokens: number | null;
  latencyMs: number | null;
  ttftMs: number | null;
  createdAt: string;
  completedAt: string | null;
  model: string | null;
  modelId: string | null;
  providerId: string | null;
  finishReason: string | null;
  toolCalls: ApiToolCallDetail[];
  skillEvents: ApiSkillEventDetail[];
}

export interface ApiSearchResult {
  turnId: string;
  turnIndex: number;
  role: string;
  agentName: string | null;
  isSubagent: boolean;
  subagentName: string | null;
  subagentSessionId: string | null;
  contentSummary: string | null;
  matchContext: string;
  matchField: 'content' | 'contentSummary';
  createdAt: string;
  hasDispatchBridge: boolean;
}

export interface ApiSearchResponse {
  items: ApiSearchResult[];
  total: number;
}

export interface ApiBridgeItem {
  bridgeId: string;
  dispatchExecutionId: string | null;
  dispatchTurnId: string | null;
  dispatchToolCallId: string | null;
  dispatchContent: string | null;
  dispatchTimestamp: string | null;
  responseExecutionId: string | null;
  responseTurnId: string | null;
  responseContent: string | null;
  responseTimestamp: string | null;
  subagentSessionId: string | null;
  subagentType: string | null;
  subagentName: string | null;
  status: string;
  subagentTokens: number;
  subagentLatencyMs: number;
}

export interface ApiBridgesResponse {
  items: ApiBridgeItem[];
  total: number;
}

export interface WorkflowStepNode {
  type: 'step';
  stepIndex: number;
  stepName: string;
  stepLabel: string;
  iterationIndex: number | null;
  iterationName: string | null;
  startTime: string | null;
  endTime: string | null;
  durationMs: number;
  totalTokens: number;
  totalCost: number;
  toolCallCount: number;
  bridgeId: string | null;
  subagentSessionId: string | null;
  subagentType: string | null;
  subagentName: string | null;
  status: string;
  parallelGroupId: string | null;
  triggerTurnId: string | null;
}

export interface WorkflowCheckpointNode {
  type: 'checkpoint';
  checkpointIndex: number;
  checkpointType: 'block' | 'info';
  checkpointLabel: string;
  requestedAt: string | null;
  approvedAt: string | null;
  waitTimeMs: number;
  triggerTurnId: string | null;
  responseTurnId: string | null;
}

export interface WorkflowParallelGroupNode {
  type: 'parallel-group';
  groupId: string;
  label: string;
  steps: WorkflowStepNode[];
  totalDurationMs: number;
  totalTokens: number;
}

export type WorkflowChildNode = WorkflowStepNode | WorkflowCheckpointNode | WorkflowParallelGroupNode;

export interface WorkflowPhaseNode {
  phaseIndex: number;
  phaseName: string;
  fullLabel: string;
  startTime: string | null;
  endTime: string | null;
  durationMs: number;
  activeTimeMs: number;
  waitTimeMs: number;
  totalTokens: number;
  totalCost: number;
  toolCallCount: number;
  subagentCount: number;
  triggerTurnId: string | null;
  children: WorkflowChildNode[];
}

export interface WorkflowSummary {
  totalPhases: number;
  totalSteps: number;
  totalCheckpoints: number;
  totalActiveTimeMs: number;
  totalWaitTimeMs: number;
  activeTimePct: number;
  iterations: number;
}

export interface WorkflowTree {
  phases: WorkflowPhaseNode[];
  summary: WorkflowSummary;
}

export interface ApiImportableSession {
  id: string;
  createdAt: string;
  firstQuery: string | null;
  turnCount: number;
  model: string | null;
}

export interface ApiImportResponse {
  sessionId: string;
  imported: boolean | number;
}

export interface ApiImportableSessionsResponse {
  sessions: ApiImportableSession[];
}

export interface ApiDeleteResponse {
  deleted: number;
  taskId?: string;
}

export interface ApiUploadResponse {
  success: boolean;
  filename: string;
}

export interface ApiAIProviderConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
}

export interface ApiAnalyzeWorkflowResponse {
  result: WorkflowTree;
}

export interface ApiTestProviderResponse {
  success: boolean;
  message: string;
}
