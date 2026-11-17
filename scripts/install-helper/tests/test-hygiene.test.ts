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
import { readFileSync, readdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/** APIs that persist install records, skill records or config.yaml under
 * getCannbotConfigDir() → homedir(). Any test file calling one of these
 * MUST sandbox the write target, or parallel test files race on the shared
 * stores (flaky assertions) and the real ~/.cannbot gets polluted. */
const PERSISTING_APIS = [
  "writeRecord(",
  "deleteRecord(",
  "addSkillsToRecord(",
  "removeSkillsFromRecord(",
  "writeSkillRecord(",
  "installPlugin(",
  "installSkill(",
  "uninstallSkills(",
  "uninstallCommand(",
  "updateCommand(",
  "initCommand(",
  "delegateInstall(",
  "ensureCannbotInstaller(",
  "addInstalledPlugin(",
  "removeInstalledPlugin(",
  "updateConfig(",
  "writeConfig(",
];

/** Accepted sandbox patterns:
 *  - `process.env.HOME =`  — in-process HOME sandbox (the repo convention)
 *  - `HOME:`               — spawned CLI env carries a sandboxed HOME
 *  - `vi.mock("...paths.js"` — paths module fully mocked (no real HOME use) */
const SANDBOX_PATTERN = /env\.HOME\s*=|HOME:|vi\.mock\([^\n]*paths\.js/;

describe("test hygiene — persisting APIs are HOME-sandboxed", () => {
  it("every test file calling persisting APIs sets up a HOME sandbox or mocks paths", () => {
    const offenders: string[] = [];
    for (const f of readdirSync(__dirname)) {
      if (!f.endsWith(".test.ts") || f === "test-hygiene.test.ts") continue;
      const src = readFileSync(join(__dirname, f), "utf-8");
      const calls = PERSISTING_APIS.filter((api) => src.includes(api));
      if (calls.length > 0 && !SANDBOX_PATTERN.test(src)) {
        offenders.push(`${f} (calls: ${calls.join(", ")})`);
      }
    }
    expect(
      offenders,
      "test files calling persisting APIs without a HOME sandbox race on shared stores and pollute the real ~/.cannbot — add `process.env.HOME = <tmpdir>` setup/teardown (see record.test.ts) or mock utils/paths.js"
    ).toEqual([]);
  });
});
