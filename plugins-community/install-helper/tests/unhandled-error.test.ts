// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

describe("unhandled rejection surfaces the error message", () => {
  it("unhandledRejection prints the error before exiting (source code verification)", async () => {
    // Thrown validation errors (e.g. an unsupported -t value) used to die
    // with exit 1 and no output at all — the handler must print the error
    // message before process.exit(1).
    const src = readFileSync(join(__dirname, "..", "src", "index.ts"), "utf-8");
    const handler = src.match(
      /process\.on\("unhandledRejection"[\s\S]*?\}\);/
    )?.[0];
    expect(handler).toBeDefined();
    expect(handler).toMatch(/console\.error|logger\.error/);
    const printIndex = handler!.search(/console\.error|logger\.error/);
    const exitIndex = handler!.indexOf("process.exit(1)");
    expect(printIndex).toBeGreaterThan(-1);
    expect(exitIndex).toBeGreaterThan(printIndex);
  });
});
