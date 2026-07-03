-- Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
-- This program is free software, you can redistribute it and/or modify it under the terms and conditions of
-- CANN Open Software License Agreement Version 2.0 (the "License").
-- Please refer to the License for details. You may not use this file except in compliance with the License.
-- THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
-- INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
-- See LICENSE in the root of the software repository for the full text of the License.

-- RedefineTables
PRAGMA defer_foreign_keys=ON;
PRAGMA foreign_keys=OFF;
CREATE TABLE "new_Session" (
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
INSERT INTO "new_Session" ("createdAt", "endTime", "framework", "id", "label", "model", "query", "rootExecutionId", "sourcePath", "startTime", "taskId", "totalCacheReadTokens", "totalCacheWriteTokens", "totalCost", "totalInputTokens", "totalLatencyMs", "totalLlmCallCount", "totalOutputTokens", "totalReasoningTokens", "totalSkillLoadCount", "totalSubagentCount", "totalTokens", "totalToolCallCount", "updatedAt", "user") SELECT "createdAt", "endTime", "framework", "id", "label", "model", "query", "rootExecutionId", "sourcePath", "startTime", "taskId", "totalCacheReadTokens", "totalCacheWriteTokens", "totalCost", "totalInputTokens", "totalLatencyMs", "totalLlmCallCount", "totalOutputTokens", "totalReasoningTokens", "totalSkillLoadCount", "totalSubagentCount", "totalTokens", "totalToolCallCount", "updatedAt", "user" FROM "Session";
DROP TABLE "Session";
ALTER TABLE "new_Session" RENAME TO "Session";
CREATE UNIQUE INDEX "Session_taskId_framework_key" ON "Session"("taskId", "framework");
PRAGMA foreign_keys=ON;
PRAGMA defer_foreign_keys=OFF;
