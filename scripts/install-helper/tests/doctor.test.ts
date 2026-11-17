// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { mkdirSync, writeFileSync, existsSync, rmSync, readFileSync } from "fs";
import { join, dirname } from "path";
import { tmpdir } from "os";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let testDir: string;

function setup() {
  testDir = join(tmpdir(), `ih-doc-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
}

function teardown() {
  rmSync(testDir, { recursive: true, force: true });
}

describe("doctor", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("doctor.ts uses scanInstalled for multi-configRoot checking (M1)", async () => {
    const src = readFileSync(join(__dirname, "..", "src", "commands", "doctor.ts"), "utf-8");
    expect(src).toContain("scanInstalled");
    expect(src).toContain("checkedRoots");
    expect(src).toContain("configRootsToCheck");
  });

  it("doctor.ts falls back to detected tool when no plugins installed", async () => {
    const src = readFileSync(join(__dirname, "..", "src", "commands", "doctor.ts"), "utf-8");
    expect(src).toContain("installed.length > 0");
    expect(src).toContain("detectedTools[0]?.name");
  });

  it("doctor.ts checks config file per installed tool+level", async () => {
    const src = readFileSync(join(__dirname, "..", "src", "commands", "doctor.ts"), "utf-8");
    expect(src).toContain("for (const { configRoot, tool, level } of configRootsToCheck)");
    expect(src).toContain("getConfigFileName(tool)");
  });

  it("renders dual-source plugins as distinguishable status lines (badge guard)", async () => {
    // Dual-source plugins (MIGRATED tombstone + frozen plugins.d copy, e.g.
    // ops-registry-invoke) must not render as two identical status lines —
    // the cannbot-source line carries the [cannbot] badge (display fix).
    const { doctorCommand } = await import("../src/commands/doctor.js");

    const savedHome = process.env.HOME;
    process.env.HOME = testDir; // sandbox: no real installs, no real ~/.cannbot
    const lines_out: string[] = [];
    const spy = vi.spyOn(console, "log").mockImplementation((...args: unknown[]) => {
      lines_out.push(args.map(String).join(" "));
    });
    try {
      await doctorCommand();
    } finally {
      spy.mockRestore();
      process.env.HOME = savedHome;
    }

    const lines = lines_out.filter((l) => l.includes("ops-registry-invoke"));
    expect(lines.length).toBe(2);
    expect(lines.some((l) => l.includes("[cannbot]"))).toBe(true);
    expect(lines.some((l) => !l.includes("[cannbot]"))).toBe(true);
  });
});
