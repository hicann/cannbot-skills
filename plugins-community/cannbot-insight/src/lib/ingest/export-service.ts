// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { PrismaClient } from '@prisma/client';
import Database from 'better-sqlite3';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

const DDL = `
CREATE TABLE "Session" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "taskId" TEXT NOT NULL,
    "label" TEXT,
    "query" TEXT,
    "framework" TEXT NOT NULL DEFAULT 'unknown',
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
    "sourcePath" TEXT,
    "parentId" TEXT,
    "version" TEXT,
    "directory" TEXT,
    "summaryAdditions" INTEGER NOT NULL DEFAULT 0,
    "summaryDeletions" INTEGER NOT NULL DEFAULT 0,
    "summaryFiles" INTEGER NOT NULL DEFAULT 0,
    "user" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL
);

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

CREATE UNIQUE INDEX "Session_taskId_framework_key" ON "Session"("taskId", "framework");
CREATE INDEX "Turn_sessionId_turnIndex_idx" ON "Turn"("sessionId", "turnIndex");
CREATE INDEX "Turn_sessionId_isSubagent_idx" ON "Turn"("sessionId", "isSubagent");
CREATE INDEX "Turn_subagentSessionId_idx" ON "Turn"("subagentSessionId");
CREATE INDEX "Turn_agentName_idx" ON "Turn"("agentName");
CREATE INDEX "ToolCall_turnId_idx" ON "ToolCall"("turnId");
CREATE INDEX "ToolCall_toolName_idx" ON "ToolCall"("toolName");
CREATE INDEX "ToolCall_toolName_state_idx" ON "ToolCall"("toolName", "state");
CREATE INDEX "SkillEvent_turnId_idx" ON "SkillEvent"("turnId");
CREATE INDEX "SkillEvent_skillName_idx" ON "SkillEvent"("skillName");
CREATE INDEX "SkillEvent_skillName_eventType_idx" ON "SkillEvent"("skillName", "eventType");
CREATE INDEX "Execution_sessionId_idx" ON "Execution"("sessionId");
CREATE INDEX "Execution_parentExecutionId_idx" ON "Execution"("parentExecutionId");
CREATE INDEX "Execution_rootExecutionId_idx" ON "Execution"("rootExecutionId");
CREATE INDEX "Execution_isSubagent_idx" ON "Execution"("isSubagent");
CREATE INDEX "Execution_agentSessionId_idx" ON "Execution"("agentSessionId");
CREATE INDEX "ExecutionSkill_skillName_skillVersion_idx" ON "ExecutionSkill"("skillName", "skillVersion");
CREATE INDEX "ExecutionSkill_executionId_idx" ON "ExecutionSkill"("executionId");
CREATE INDEX "SessionSkill_skillName_idx" ON "SessionSkill"("skillName");
CREATE UNIQUE INDEX "SessionSkill_sessionId_skillName_key" ON "SessionSkill"("sessionId", "skillName");
CREATE INDEX "InteractionBridge_sessionId_idx" ON "InteractionBridge"("sessionId");
CREATE INDEX "InteractionBridge_dispatchExecutionId_idx" ON "InteractionBridge"("dispatchExecutionId");
CREATE INDEX "InteractionBridge_responseExecutionId_idx" ON "InteractionBridge"("responseExecutionId");
CREATE INDEX "InteractionBridge_subagentSessionId_idx" ON "InteractionBridge"("subagentSessionId");
CREATE INDEX "InteractionBridge_status_idx" ON "InteractionBridge"("status");
`;

function toISO(d: Date | null): string | null {
  return d ? d.toISOString() : null;
}

export async function exportSession(
  taskId: string,
  outputPath?: string,
  prisma?: PrismaClient,
  framework?: string,
): Promise<string> {
  const client = prisma ?? new PrismaClient();

  try {
    const where: Record<string, string> = { taskId };
    if (framework) where.framework = framework;

    const session = await client.session.findFirst({
      where,
    });
    if (!session) {
      throw new Error(`Session not found: "${taskId}"`);
    }

    const outPath = outputPath ?? path.join(os.tmpdir(), `cannbot_session_${taskId}.db`);
    if (fs.existsSync(outPath)) {
      fs.unlinkSync(outPath);
    }

    const db = new Database(outPath);
    db.exec(DDL);

    const insertSession = db.prepare(`
      INSERT INTO "Session" (
        "id", "taskId", "label", "query", "framework", "model",
        "startTime", "endTime",
        "totalTokens", "totalInputTokens", "totalOutputTokens", "totalReasoningTokens",
        "totalCacheReadTokens", "totalCacheWriteTokens", "totalCost", "totalLatencyMs",
        "totalToolCallCount", "totalLlmCallCount", "totalSkillLoadCount", "totalSubagentCount",
        "rootExecutionId", "sourcePath", "parentId", "version", "directory",
        "summaryAdditions", "summaryDeletions", "summaryFiles",
        "user", "createdAt", "updatedAt"
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    const insertTurn = db.prepare(`
      INSERT INTO "Turn" (
        "id", "sessionId", "turnIndex", "role", "content", "contentJson", "contentSummary",
        "inputMessagesJson", "inputMessagesCount", "inputMessagesTokens", "contextWindowPct",
        "agentName", "subagentName", "subagentSessionId",
        "totalTokens", "inputTokens", "outputTokens", "reasoningTokens", "cacheReadTokens", "cacheWriteTokens",
        "createdAt_ts", "completedAt", "latencyMs", "ttftMs",
        "model", "modelId", "providerId", "temperature", "maxTokens", "finishReason",
        "isSubagent", "parentExecutionId", "createdAt"
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    const insertToolCall = db.prepare(`
      INSERT INTO "ToolCall" (
        "id", "turnId", "toolCallId", "toolName", "argsJson", "resultJson",
        "state", "errorType", "errorMessage",
        "startedAt", "completedAt", "durationMs", "dispatchBridgeId", "isSkillRelated", "createdAt"
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    const insertSkillEvent = db.prepare(`
      INSERT INTO "SkillEvent" (
        "id", "turnId", "skillName", "skillVersion", "eventType", "success",
        "errorMessage", "argsJson", "startedAt", "completedAt", "durationMs", "createdAt"
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    const insertExecution = db.prepare(`
      INSERT INTO "Execution" (
        "id", "sessionId", "agentName", "agentSessionId", "isSubagent", "subagentType", "subagentName",
        "parentExecutionId", "rootExecutionId", "depth",
        "tokens", "inputTokens", "outputTokens", "reasoningTokens",
        "cacheReadInputTokens", "cacheCreationInputTokens", "maxSingleCallTokens",
        "cost", "latencyMs", "toolCallCount", "toolCallErrorCount",
        "llmCallCount", "skillLoadCount", "skillInvokeCount",
        "finalResult", "model", "createdAt"
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    const insertExecutionSkill = db.prepare(`
      INSERT INTO "ExecutionSkill" (
        "id", "executionId", "skillName", "skillVersion", "isPrimary", "user", "createdAt"
      ) VALUES (?, ?, ?, ?, ?, ?, ?)
    `);

    const insertSessionSkill = db.prepare(`
      INSERT INTO "SessionSkill" (
        "id", "sessionId", "skillName", "skillVersion", "invocationCount", "user", "createdAt"
      ) VALUES (?, ?, ?, ?, ?, ?, ?)
    `);

    const insertBridge = db.prepare(`
      INSERT INTO "InteractionBridge" (
        "id", "sessionId", "dispatchExecutionId", "dispatchTurnId", "dispatchToolCallId",
        "dispatchContent", "dispatchTimestamp",
        "responseExecutionId", "responseTurnId", "responseContent", "responseTimestamp",
        "subagentSessionId", "subagentType", "subagentName", "status",
        "subagentTokens", "subagentLatencyMs", "createdAt", "updatedAt"
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    // Pre-fetch all related data

    // Pre-fetch all related data
    const turns = await client.turn.findMany({ where: { sessionId: session.id } });
    const turnIds = turns.map(t => t.id);
    const toolCalls = turnIds.length > 0
      ? await client.toolCall.findMany({ where: { turnId: { in: turnIds } } })
      : [];
    const skillEvents = turnIds.length > 0
      ? await client.skillEvent.findMany({ where: { turnId: { in: turnIds } } })
      : [];
    const executions = await client.execution.findMany({ where: { sessionId: session.id } });
    const executionIds = executions.map(e => e.id);
    const executionSkills = executionIds.length > 0
      ? await client.executionSkill.findMany({ where: { executionId: { in: executionIds } } })
      : [];
    const sessionSkills = await client.sessionSkill.findMany({ where: { sessionId: session.id } });
    const bridges = await client.interactionBridge.findMany({ where: { sessionId: session.id } });

    // Write all data in a single transaction
    db.transaction(() => {
      insertSession.run(
        session.id, session.taskId, session.label, session.query, session.framework, session.model,
        toISO(session.startTime), toISO(session.endTime),
        session.totalTokens, session.totalInputTokens, session.totalOutputTokens, session.totalReasoningTokens,
        session.totalCacheReadTokens, session.totalCacheWriteTokens, session.totalCost, session.totalLatencyMs,
        session.totalToolCallCount, session.totalLlmCallCount, session.totalSkillLoadCount, session.totalSubagentCount,
        session.rootExecutionId, session.sourcePath,
        session.parentId, session.version, session.directory,
        session.summaryAdditions, session.summaryDeletions, session.summaryFiles,
        session.user,
        toISO(session.createdAt), toISO(session.updatedAt),
      );

      for (const t of turns) {
        insertTurn.run(
          t.id, t.sessionId, t.turnIndex, t.role, t.content, t.contentJson, t.contentSummary,
          t.inputMessagesJson, t.inputMessagesCount, t.inputMessagesTokens, t.contextWindowPct,
          t.agentName, t.subagentName, t.subagentSessionId,
          t.totalTokens, t.inputTokens, t.outputTokens, t.reasoningTokens, t.cacheReadTokens, t.cacheWriteTokens,
          toISO(t.createdAt_ts), toISO(t.completedAt), t.latencyMs, t.ttftMs,
          t.model, t.modelId, t.providerId, t.temperature, t.maxTokens, t.finishReason,
          t.isSubagent ? 1 : 0, t.parentExecutionId, toISO(t.createdAt),
        );
      }

      for (const tc of toolCalls) {
        insertToolCall.run(
          tc.id, tc.turnId, tc.toolCallId, tc.toolName, tc.argsJson, tc.resultJson,
          tc.state, tc.errorType, tc.errorMessage,
          toISO(tc.startedAt), toISO(tc.completedAt), tc.durationMs, tc.dispatchBridgeId,
          tc.isSkillRelated ? 1 : 0, toISO(tc.createdAt),
        );
      }

      for (const se of skillEvents) {
        insertSkillEvent.run(
          se.id, se.turnId, se.skillName, se.skillVersion, se.eventType,
          se.success ? 1 : 0, se.errorMessage, se.argsJson,
          toISO(se.startedAt), toISO(se.completedAt), se.durationMs, toISO(se.createdAt),
        );
      }

      for (const e of executions) {
        insertExecution.run(
          e.id, e.sessionId, e.agentName, e.agentSessionId,
          e.isSubagent ? 1 : 0, e.subagentType, e.subagentName,
          e.parentExecutionId, e.rootExecutionId, e.depth,
          e.tokens, e.inputTokens, e.outputTokens, e.reasoningTokens,
          e.cacheReadInputTokens, e.cacheCreationInputTokens, e.maxSingleCallTokens,
          e.cost, e.latencyMs, e.toolCallCount, e.toolCallErrorCount,
          e.llmCallCount, e.skillLoadCount, e.skillInvokeCount,
          e.finalResult, e.model, toISO(e.createdAt),
        );
      }

      for (const es of executionSkills) {
        insertExecutionSkill.run(
          es.id, es.executionId, es.skillName, es.skillVersion,
          es.isPrimary ? 1 : 0, es.user, toISO(es.createdAt),
        );
      }

      for (const ss of sessionSkills) {
        insertSessionSkill.run(
          ss.id, ss.sessionId, ss.skillName, ss.skillVersion,
          ss.invocationCount, ss.user, toISO(ss.createdAt),
        );
      }

      for (const b of bridges) {
        insertBridge.run(
          b.id, b.sessionId, b.dispatchExecutionId, b.dispatchTurnId, b.dispatchToolCallId,
          b.dispatchContent, toISO(b.dispatchTimestamp),
          b.responseExecutionId, b.responseTurnId, b.responseContent, toISO(b.responseTimestamp),
          b.subagentSessionId, b.subagentType, b.subagentName, b.status,
          b.subagentTokens, b.subagentLatencyMs, toISO(b.createdAt), toISO(b.updatedAt),
        );
      }
    })();

    db.close();
    console.log(`[export] session ${taskId} → ${outPath} (${turns.length} turns, ${toolCalls.length} toolCalls, ${skillEvents.length} skillEvents, ${executions.length} executions, ${bridges.length} bridges)`);
    return outPath;
  } finally {
    if (!prisma) await client.$disconnect();
  }
}
