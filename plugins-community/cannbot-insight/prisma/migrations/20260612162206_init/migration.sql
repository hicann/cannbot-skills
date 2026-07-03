-- Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
-- This program is free software, you can redistribute it and/or modify it under the terms and conditions of
-- CANN Open Software License Agreement Version 2.0 (the "License").
-- Please refer to the License for details. You may not use this file except in compliance with the License.
-- THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
-- INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
-- See LICENSE in the root of the software repository for the full text of the License.

-- CreateTable
CREATE TABLE "Session" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "taskId" TEXT NOT NULL,
    "label" TEXT,
    "query" TEXT,
    "framework" TEXT,
    "model" TEXT,
    "startTime" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "endTime" DATETIME,
    "totalTokens" INTEGER NOT NULL DEFAULT 0,
    "totalInputTokens" INTEGER NOT NULL DEFAULT 0,
    "totalOutputTokens" INTEGER NOT NULL DEFAULT 0,
    "totalReasoningTokens" INTEGER NOT NULL DEFAULT 0,
    "totalCacheReadTokens" INTEGER NOT NULL DEFAULT 0,
    "totalCacheWriteTokens" INTEGER NOT NULL DEFAULT 0,
    "totalCost" REAL NOT NULL DEFAULT 0,
    "totalLatencyMs" INTEGER NOT NULL DEFAULT 0,
    "totalToolCallCount" INTEGER NOT NULL DEFAULT 0,
    "totalLlmCallCount" INTEGER NOT NULL DEFAULT 0,
    "totalSkillLoadCount" INTEGER NOT NULL DEFAULT 0,
    "totalSubagentCount" INTEGER NOT NULL DEFAULT 0,
    "rootExecutionId" TEXT,
    "user" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL
);

-- CreateTable
CREATE TABLE "Turn" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "sessionId" TEXT NOT NULL,
    "turnIndex" INTEGER NOT NULL,
    "role" TEXT NOT NULL,
    "content" TEXT,
    "contentJson" TEXT,
    "contentSummary" TEXT,
    "inputMessagesJson" TEXT,
    "inputMessagesCount" INTEGER NOT NULL DEFAULT 0,
    "inputMessagesTokens" INTEGER NOT NULL DEFAULT 0,
    "contextWindowPct" REAL,
    "agentName" TEXT,
    "subagentName" TEXT,
    "subagentSessionId" TEXT,
    "totalTokens" INTEGER NOT NULL DEFAULT 0,
    "inputTokens" INTEGER NOT NULL DEFAULT 0,
    "outputTokens" INTEGER NOT NULL DEFAULT 0,
    "reasoningTokens" INTEGER NOT NULL DEFAULT 0,
    "cacheReadTokens" INTEGER NOT NULL DEFAULT 0,
    "cacheWriteTokens" INTEGER NOT NULL DEFAULT 0,
    "createdAt_ts" DATETIME,
    "completedAt" DATETIME,
    "latencyMs" INTEGER NOT NULL DEFAULT 0,
    "ttftMs" INTEGER,
    "model" TEXT,
    "modelId" TEXT,
    "providerId" TEXT,
    "temperature" REAL,
    "maxTokens" INTEGER,
    "finishReason" TEXT,
    "isSubagent" BOOLEAN NOT NULL DEFAULT false,
    "parentExecutionId" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "Turn_sessionId_fkey" FOREIGN KEY ("sessionId") REFERENCES "Session" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "ToolCall" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "turnId" TEXT NOT NULL,
    "toolCallId" TEXT NOT NULL,
    "toolName" TEXT NOT NULL,
    "argsJson" TEXT,
    "resultJson" TEXT,
    "state" TEXT NOT NULL DEFAULT 'ok',
    "errorType" TEXT,
    "errorMessage" TEXT,
    "startedAt" DATETIME,
    "completedAt" DATETIME,
    "durationMs" INTEGER NOT NULL DEFAULT 0,
    "dispatchBridgeId" TEXT,
    "isSkillRelated" BOOLEAN NOT NULL DEFAULT false,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "ToolCall_turnId_fkey" FOREIGN KEY ("turnId") REFERENCES "Turn" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "SkillEvent" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "turnId" TEXT NOT NULL,
    "skillName" TEXT NOT NULL,
    "skillVersion" INTEGER,
    "eventType" TEXT NOT NULL,
    "success" BOOLEAN NOT NULL DEFAULT true,
    "errorMessage" TEXT,
    "argsJson" TEXT,
    "startedAt" DATETIME,
    "completedAt" DATETIME,
    "durationMs" INTEGER NOT NULL DEFAULT 0,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "SkillEvent_turnId_fkey" FOREIGN KEY ("turnId") REFERENCES "Turn" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "Execution" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "sessionId" TEXT NOT NULL,
    "agentName" TEXT,
    "agentSessionId" TEXT,
    "isSubagent" BOOLEAN NOT NULL DEFAULT false,
    "subagentType" TEXT,
    "subagentName" TEXT,
    "parentExecutionId" TEXT,
    "rootExecutionId" TEXT,
    "depth" INTEGER NOT NULL DEFAULT 0,
    "tokens" INTEGER NOT NULL DEFAULT 0,
    "inputTokens" INTEGER NOT NULL DEFAULT 0,
    "outputTokens" INTEGER NOT NULL DEFAULT 0,
    "reasoningTokens" INTEGER NOT NULL DEFAULT 0,
    "cacheReadInputTokens" INTEGER NOT NULL DEFAULT 0,
    "cacheCreationInputTokens" INTEGER NOT NULL DEFAULT 0,
    "maxSingleCallTokens" INTEGER NOT NULL DEFAULT 0,
    "cost" REAL NOT NULL DEFAULT 0,
    "latencyMs" INTEGER NOT NULL DEFAULT 0,
    "toolCallCount" INTEGER NOT NULL DEFAULT 0,
    "toolCallErrorCount" INTEGER NOT NULL DEFAULT 0,
    "llmCallCount" INTEGER NOT NULL DEFAULT 0,
    "skillLoadCount" INTEGER NOT NULL DEFAULT 0,
    "skillInvokeCount" INTEGER NOT NULL DEFAULT 0,
    "finalResult" TEXT,
    "model" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "Execution_sessionId_fkey" FOREIGN KEY ("sessionId") REFERENCES "Session" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "ExecutionSkill" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "executionId" TEXT NOT NULL,
    "skillName" TEXT NOT NULL,
    "skillVersion" INTEGER,
    "isPrimary" BOOLEAN NOT NULL DEFAULT false,
    "user" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "ExecutionSkill_executionId_fkey" FOREIGN KEY ("executionId") REFERENCES "Execution" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "SessionSkill" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "sessionId" TEXT NOT NULL,
    "skillName" TEXT NOT NULL,
    "skillVersion" INTEGER,
    "invocationCount" INTEGER NOT NULL DEFAULT 0,
    "user" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "SessionSkill_sessionId_fkey" FOREIGN KEY ("sessionId") REFERENCES "Session" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "InteractionBridge" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "sessionId" TEXT NOT NULL,
    "dispatchExecutionId" TEXT NOT NULL,
    "dispatchTurnId" TEXT,
    "dispatchToolCallId" TEXT,
    "dispatchContent" TEXT,
    "dispatchTimestamp" DATETIME,
    "responseExecutionId" TEXT,
    "responseTurnId" TEXT,
    "responseContent" TEXT,
    "responseTimestamp" DATETIME,
    "subagentSessionId" TEXT,
    "subagentType" TEXT,
    "subagentName" TEXT,
    "status" TEXT NOT NULL DEFAULT 'dispatched',
    "subagentTokens" INTEGER NOT NULL DEFAULT 0,
    "subagentLatencyMs" INTEGER NOT NULL DEFAULT 0,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL,
    CONSTRAINT "InteractionBridge_sessionId_fkey" FOREIGN KEY ("sessionId") REFERENCES "Session" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateIndex
CREATE UNIQUE INDEX "Session_taskId_key" ON "Session"("taskId");

-- CreateIndex
CREATE INDEX "Turn_sessionId_turnIndex_idx" ON "Turn"("sessionId", "turnIndex");

-- CreateIndex
CREATE INDEX "Turn_sessionId_isSubagent_idx" ON "Turn"("sessionId", "isSubagent");

-- CreateIndex
CREATE INDEX "Turn_subagentSessionId_idx" ON "Turn"("subagentSessionId");

-- CreateIndex
CREATE INDEX "Turn_agentName_idx" ON "Turn"("agentName");

-- CreateIndex
CREATE INDEX "ToolCall_turnId_idx" ON "ToolCall"("turnId");

-- CreateIndex
CREATE INDEX "ToolCall_toolName_idx" ON "ToolCall"("toolName");

-- CreateIndex
CREATE INDEX "ToolCall_toolName_state_idx" ON "ToolCall"("toolName", "state");

-- CreateIndex
CREATE INDEX "SkillEvent_turnId_idx" ON "SkillEvent"("turnId");

-- CreateIndex
CREATE INDEX "SkillEvent_skillName_idx" ON "SkillEvent"("skillName");

-- CreateIndex
CREATE INDEX "SkillEvent_skillName_eventType_idx" ON "SkillEvent"("skillName", "eventType");

-- CreateIndex
CREATE INDEX "Execution_sessionId_idx" ON "Execution"("sessionId");

-- CreateIndex
CREATE INDEX "Execution_parentExecutionId_idx" ON "Execution"("parentExecutionId");

-- CreateIndex
CREATE INDEX "Execution_rootExecutionId_idx" ON "Execution"("rootExecutionId");

-- CreateIndex
CREATE INDEX "Execution_isSubagent_idx" ON "Execution"("isSubagent");

-- CreateIndex
CREATE INDEX "Execution_agentSessionId_idx" ON "Execution"("agentSessionId");

-- CreateIndex
CREATE INDEX "ExecutionSkill_skillName_skillVersion_idx" ON "ExecutionSkill"("skillName", "skillVersion");

-- CreateIndex
CREATE INDEX "ExecutionSkill_executionId_idx" ON "ExecutionSkill"("executionId");

-- CreateIndex
CREATE INDEX "SessionSkill_skillName_idx" ON "SessionSkill"("skillName");

-- CreateIndex
CREATE UNIQUE INDEX "SessionSkill_sessionId_skillName_key" ON "SessionSkill"("sessionId", "skillName");

-- CreateIndex
CREATE INDEX "InteractionBridge_sessionId_idx" ON "InteractionBridge"("sessionId");

-- CreateIndex
CREATE INDEX "InteractionBridge_dispatchExecutionId_idx" ON "InteractionBridge"("dispatchExecutionId");

-- CreateIndex
CREATE INDEX "InteractionBridge_responseExecutionId_idx" ON "InteractionBridge"("responseExecutionId");

-- CreateIndex
CREATE INDEX "InteractionBridge_subagentSessionId_idx" ON "InteractionBridge"("subagentSessionId");

-- CreateIndex
CREATE INDEX "InteractionBridge_status_idx" ON "InteractionBridge"("status");
