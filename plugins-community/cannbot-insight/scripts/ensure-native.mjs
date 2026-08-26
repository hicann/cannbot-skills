// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Pre-test guard: detect stale native addons (compiled for an older Node ABI
// after a Node.js upgrade) and rebuild them automatically. Without this, a
// `node` bump silently breaks every test that touches better-sqlite3 — they
// fail with "Module did not self-register" / "NODE_MODULE_VERSION mismatch",
// which is hard to diagnose from the test output. Run via the `pretest` hook.
//
// Fast path (binary fresh): one smoke-test load per module, ~tens of ms.
// Slow path (binary stale): `npm rebuild <mod>`, then exit so the test
// process reloads the fresh binary.

import { execSync } from 'node:child_process';

const NATIVES = [
  {
    name: 'better-sqlite3',
    // smoke test: open an in-memory db, run a trivial query, close.
    probe: async (mod) => {
      const Database = mod.default ?? mod;
      const db = new Database(':memory:');
      db.prepare('select 1 as x').get();
      db.close();
    },
  },
];

const STALE = /NODE_MODULE_VERSION|did not self-register|was compiled against a different Node\.js version/i;

let rebuilt = false;
for (const { name, probe } of NATIVES) {
  try {
    const mod = await import(name);
    await probe(mod);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (STALE.test(msg)) {
      console.warn(`[ensure-native] ${name} binary is stale (Node ABI mismatch), rebuilding...`);
      execSync(`npm rebuild ${name}`, { stdio: 'inherit' });
      rebuilt = true;
    } else {
      // A real bug (not an ABI mismatch) — surface it, don't hide.
      console.error(`[ensure-native] ${name} failed to load for an unexpected reason:`);
      console.error(e);
      process.exit(1);
    }
  }
}

if (rebuilt) {
  console.warn('[ensure-native] native addon(s) rebuilt. The test process will load the fresh binary.');
}
