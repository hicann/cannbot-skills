// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdirSync, writeFileSync, existsSync, rmSync, readFileSync } from "fs";
import { join, dirname } from "path";
import { tmpdir } from "os";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let testDir: string;

function setup() {
  testDir = join(tmpdir(), `ih-upd-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
}

function teardown() {
  rmSync(testDir, { recursive: true, force: true });
}

describe("update", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("update.ts uses scanInstalled for plugin discovery (source verification)", async () => {
    const src = readFileSync(join(__dirname, "..", "src", "commands", "update.ts"), "utf-8");
    expect(src).toContain("scanInstalled");
    expect(src).not.toContain("readAllManifests(configRoot)");
  });

  it("update.ts continue on invalid plugin name instead of return (M4)", async () => {
    const src = readFileSync(join(__dirname, "..", "src", "commands", "update.ts"), "utf-8");
    const errorBlockMatch = src.match(/if \(!plugin\) \{[\s\S]*?\n\s*\}/);
    expect(errorBlockMatch).toBeTruthy();
    expect(errorBlockMatch![0]).toContain("continue");
    expect(errorBlockMatch![0]).not.toContain("return;");
  });

  it("update.ts handles multiple install targets (tool+level per plugin)", async () => {
    const src = readFileSync(join(__dirname, "..", "src", "commands", "update.ts"), "utf-8");
    expect(src).toContain("UpdateTarget");
    expect(src).toContain("installed.filter");
    expect(src).toContain("target.tool");
    expect(src).toContain("target.level");
  });

  it("update.ts passes yes option to installPlugin (H2)", async () => {
    const src = readFileSync(join(__dirname, "..", "src", "commands", "update.ts"), "utf-8");
    expect(src).toContain("yes?: boolean");
    expect(src).toContain("yes: options.yes");
  });

  it("update.ts skips uninstalled plugins with warn instead of installing (M2)", async () => {
    const src = readFileSync(join(__dirname, "..", "src", "commands", "update.ts"), "utf-8");
    expect(src).toContain("update_skipped");
    expect(src).toContain("error_not_installed");
  });

  it("update.ts wraps ensureRepoAndScan in try/catch (H3)", async () => {
    const src = readFileSync(join(__dirname, "..", "src", "commands", "update.ts"), "utf-8");
    expect(src).toContain("try {");
    expect(src).toContain("repo_clone_failed");
  });

  it("update.ts calls ensureRepoAndScan before scanInstalled (H1 dynamic plugin discovery)", async () => {
    const src = readFileSync(join(__dirname, "..", "src", "commands", "update.ts"), "utf-8");
    const ensureCallPos = src.indexOf("repoManager.ensureRepoAndScan()");
    const scanCallPos = src.indexOf("const installed = scanInstalled()");
    expect(ensureCallPos).toBeGreaterThan(-1);
    expect(scanCallPos).toBeGreaterThan(-1);
    expect(ensureCallPos).toBeLessThan(scanCallPos);
  });

  it("update.ts does not import selectToolWithDetection (removed redundant call)", async () => {
    const src = readFileSync(join(__dirname, "..", "src", "commands", "update.ts"), "utf-8");
    expect(src).not.toContain("selectToolWithDetection");
  });
});
