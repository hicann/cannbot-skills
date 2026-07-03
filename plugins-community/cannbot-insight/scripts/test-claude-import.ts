// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { PrismaClient } from "@prisma/client";
import { importSession } from "../src/lib/ingest/data-service";
import { readSession, listSessions } from "../src/lib/ingest/adapters/claude-jsonl";

const prisma = new PrismaClient();

const TEST_DIR = "/home/gxh/code/logs/claude-code-sessions/projects/-home-wangrui-code-ops-math";
const TEST_SESSION_ID = "7d546f46-7406-43c3-8ac8-912f31fde3a6";
const TEST_FILE = `${TEST_DIR}/${TEST_SESSION_ID}.jsonl`;

async function main() {
  console.log("=== Test 1: readSession with file path ===");
  const fileResult = readSession(TEST_FILE, TEST_SESSION_ID);
  console.log(`  Lines: ${fileResult.length}`);
  console.log(`  Roles: ${fileResult.map(r => r.role).slice(0, 5).join(", ")}...`);

  console.log("\n=== Test 2: readSession with directory path ===");
  const dirResult = readSession(TEST_DIR, TEST_SESSION_ID);
  console.log(`  Lines: ${dirResult.length}`);
  console.log(`  Same as file result: ${dirResult.length === fileResult.length}`);

  console.log("\n=== Test 3: listSessions ===");
  const sessions = listSessions(TEST_DIR);
  console.log(`  Sessions found: ${sessions.length}`);
  if (sessions.length > 0) {
    console.log(`  First session: ${sessions[0].id} (${sessions[0].turnCount} turns)`);
  }

  console.log("\n=== Test 4: full importSession ===");
  try {
    const result = await importSession(
      TEST_DIR,
      TEST_SESSION_ID,
      prisma,
      TEST_FILE,
      "claude-jsonl"
    );
    console.log(`  Result: sessionId=${result.sessionId}, imported=${result.imported}`);
  } catch (e) {
    console.error("  IMPORT ERROR:", e);
    process.exit(1);
  }
}

main()
  .catch(e => { console.error("FATAL ERROR:", e); process.exit(1); })
  .finally(() => prisma.$disconnect());
