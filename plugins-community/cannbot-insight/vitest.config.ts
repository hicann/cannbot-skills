// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  test: {
    environment: "node",
    exclude: ["tests/**/*.bench.ts", "tests/**/*.bench.tsx", "node_modules", "dist", ".next"],
    // Two decoupled projects: insight (Prisma-backed integration tests, needs the
    // @ alias + react plugin for .tsx) + proxy (self-contained JSON-output tests,
    // NO Prisma setup, NO cannbot-insight imports, no alias needed). `npm run test`
    // runs both; `npx vitest run --project proxy` runs proxy only.
    projects: [
      {
        plugins: [react()],
        resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
        test: {
          name: "insight",
          include: ["tests/**/*.test.ts", "tests/**/*.test.tsx"],
          setupFiles: ["./tests/setup.ts"],
        },
      },
      {
        test: {
          name: "proxy",
          include: ["proxy/tests/**/*.test.ts"],
        },
      },
    ],
  },
  bench: {
    include: ["tests/**/*.bench.ts", "tests/**/*.bench.tsx"],
    exclude: ["node_modules", "dist", ".next"],
  },
});
