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
import { mkdirSync, rmSync } from "fs";
import { join, dirname } from "path";
import { tmpdir } from "os";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let testDir: string;
let savedHome: string | undefined;

function setup() {
  testDir = join(tmpdir(), `ih-disp-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
  // Sandbox HOME: display reads the language config under ~/.cannbot — pin
  // it to an empty sandbox so the output is deterministic.
  savedHome = process.env.HOME;
  process.env.HOME = testDir;
}

function teardown() {
  rmSync(testDir, { recursive: true, force: true });
  process.env.HOME = savedHome;
}

describe("display — install summary docs line", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("avoids the empty-prefix docs path for migrated tombstones (dir: \"\")", async () => {
    // A tombstone plugin (catlass-op-generator) carries dir: "" — the docs
    // line previously rendered the unreachable "/quickstart.md". It must
    // fall back to a plugin-id-relative hint instead.
    const { printEnhancedSummary } = await import("../src/ui/display.js");

    const lines: string[] = [];
    const spy = vi.spyOn(console, "log").mockImplementation((...args: unknown[]) => {
      lines.push(args.map(String).join(" "));
    });
    try {
      printEnhancedSummary(
        [
          {
            pluginId: "catlass-op-generator",
            displayName: "Catlass 算子开发",
            success: true,
            skillsCount: 6,
            agentsCount: 0,
          },
        ],
        "opencode",
        join(testDir, ".opencode")
      );
    } finally {
      spy.mockRestore();
    }

    const docsLine = lines.find((l) => l.includes("quickstart.md"));
    expect(docsLine).toBeDefined();
    expect(docsLine).toContain("catlass-op-generator/quickstart.md");
    expect(docsLine).not.toMatch(/:\s*\/quickstart\.md/);
  });

  it("keeps the repo-relative docs path for local-source plugins", async () => {
    const { printEnhancedSummary } = await import("../src/ui/display.js");

    const lines: string[] = [];
    const spy = vi.spyOn(console, "log").mockImplementation((...args: unknown[]) => {
      lines.push(args.map(String).join(" "));
    });
    try {
      printEnhancedSummary(
        [
          {
            pluginId: "torch-compile",
            displayName: "torch.compile 图模式",
            success: true,
            skillsCount: 7,
            agentsCount: 1,
            source: "skills",
          },
        ],
        "opencode",
        join(testDir, ".opencode")
      );
    } finally {
      spy.mockRestore();
    }

    const docsLine = lines.find((l) => l.includes("quickstart.md"));
    expect(docsLine).toBeDefined();
    expect(docsLine).toContain("plugins-official/torch-compile/quickstart.md");
  });
});
