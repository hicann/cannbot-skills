// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { beforeAll, afterAll } from "vitest";
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

beforeAll(async () => {
  // Probe native addons that vitest loads in-worker. A `node` upgrade leaves
  // the compiled binary on a stale ABI → "Module did not self-register" /
  // NODE_MODULE_VERSION mismatch, which surfaces as a dozen cryptic test
  // failures. Fail fast with one actionable message instead. (`npm run test`
  // auto-rebuilds via pretest; this guards bare `npx vitest` runs.)
  try {
    const mod = await import("better-sqlite3");
    const Database = (mod as { default?: unknown }).default ?? mod;
    const db = new (Database as new (s: string) => { close: () => void })(":memory:");
    db.close();
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (/NODE_MODULE_VERSION|did not self-register|was compiled against a different Node\.js version/i.test(msg)) {
      throw new Error(
        `Native addon better-sqlite3 is stale (Node ABI mismatch after a Node.js upgrade).\n` +
        `Run: npm rebuild better-sqlite3   (or: ./start.sh -u)\n` +
        `Original error: ${msg}`
      );
    }
    throw e;
  }
  await prisma.$connect();
});

afterAll(async () => {
  await prisma.$disconnect();
});

export { prisma };
