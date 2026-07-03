// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { listSessions } from '../src/lib/ingest/adapters/opencode-db.ts';
import { importSession } from '../src/lib/ingest/data-service.ts';
import { PrismaClient } from '@prisma/client';
import * as readline from 'readline/promises';
import { stdin as input, stdout as output } from 'process';

function parseArgs(argv: string[]): { path: string | null } {
  let path: string | null = null;
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--path' && argv[i + 1]) {
      path = argv[i + 1];
      i++;
    }
  }
  return { path };
}

async function main() {
  const { path } = parseArgs(process.argv.slice(2));

  if (!path) {
    console.error('Usage: npx tsx scripts/import-opencode.ts --path /path/to/opencode.db');
    process.exit(1);
  }

  console.log(`Reading sessions from: ${path}`);
  const sessions = listSessions(path);

  if (sessions.length === 0) {
    console.error('No sessions found in the database.');
    process.exit(1);
  }

  console.log('\nAvailable sessions:');
  console.log('  #  | ID                                   | Time                 | Turns | Model             | First Query');
  console.log('-----|--------------------------------------|----------------------|-------|-------------------|-------------');
  for (let i = 0; i < sessions.length; i++) {
    const s = sessions[i];
    const time = s.createdAt.substring(0, 19).replace('T', ' ');
    const query = (s.firstQuery ?? '').substring(0, 40);
    const model = (s.modelName ?? 'unknown').substring(0, 17);
    console.log(`  ${String(i).padStart(2)} | ${s.id.padEnd(36)} | ${time.padEnd(20)} | ${String(s.turnCount).padStart(5)} | ${model.padEnd(17)} | ${query}`);
  }

  const rl = readline.createInterface({ input, output });
  const answer = await rl.question('\nSelect session # to import (or "q" to quit): ');
  rl.close();

  if (answer.toLowerCase() === 'q') {
    console.log('Exiting.');
    process.exit(0);
  }

  const index = parseInt(answer, 10);
  if (isNaN(index) || index < 0 || index >= sessions.length) {
    console.error(`Invalid selection: ${answer}`);
    process.exit(1);
  }

  const selected = sessions[index];
  console.log(`\nImporting session: ${selected.id} (${selected.turnCount} turns)...`);

  const prisma = new PrismaClient();
  try {
    const result = await importSession(path, selected.id, prisma);

    if (result.imported) {
      console.log(`\nImport successful!`);
      console.log(`  Session ID: ${result.sessionId}`);
      console.log(`  Task ID: ${selected.id}`);

      const session = await prisma.session.findFirst({ where: { id: result.sessionId } });
      if (session) {
        console.log(`  Model: ${session.model}`);
        console.log(`  Total Tokens: ${session.totalTokens}`);
        console.log(`  Total Cost: ${session.totalCost}`);
        console.log(`  Total Tool Calls: ${session.totalToolCallCount}`);
        console.log(`  Total LLM Calls: ${session.totalLlmCallCount}`);

        const turnCount = await prisma.turn.count({ where: { sessionId: session.id } });
        const tcCount = await prisma.toolCall.count({ where: { turnId: { in: await prisma.turn.findMany({ where: { sessionId: session.id } }).then(ts => ts.map(t => t.id)) } } });
        const bridgeCount = await prisma.interactionBridge.count({ where: { sessionId: session.id } });
        const execCount = await prisma.execution.count({ where: { sessionId: session.id } });

        console.log(`  Turns created: ${turnCount}`);
        console.log(`  ToolCalls created: ${tcCount}`);
        console.log(`  Bridges created: ${bridgeCount}`);
        console.log(`  Executions created: ${execCount}`);
      }
    } else {
      console.log(`\nSession already exists (dedup). No new rows created.`);
      console.log(`  Existing Session ID: ${result.sessionId}`);
    }
  } finally {
    await prisma.$disconnect();
  }
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
