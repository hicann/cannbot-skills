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
import { mkdirSync, rmSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

let testDir: string;
let savedEnv: Record<string, string | undefined>;

function setup() {
  testDir = join(tmpdir(), `ih-dsh-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
  savedEnv = { HOME: process.env.HOME, DSH_HOME: process.env.DSH_HOME };
  process.env.HOME = testDir;
  delete process.env.DSH_HOME;
}

function teardown() {
  rmSync(testDir, { recursive: true, force: true });
  for (const [key, value] of Object.entries(savedEnv)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
}

describe("dsh as a first-class install target", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("validateTool accepts dsh", async () => {
    const { validateTool } = await import("../src/utils/paths.js");
    expect(validateTool("dsh")).toBe("dsh");
  });

  it("project layout lands in .dsh (DSH project-dsh root)", async () => {
    const { getConfigRoot } = await import("../src/utils/paths.js");
    expect(getConfigRoot("dsh", "project", join(testDir, "proj"))).toBe(
      join(testDir, "proj", ".dsh")
    );
  });

  it("global layout defaults to ~/.dsh and honors DSH_HOME", async () => {
    const { getConfigRoot } = await import("../src/utils/paths.js");
    expect(getConfigRoot("dsh", "global")).toBe(join(testDir, ".dsh"));
    process.env.DSH_HOME = join(testDir, "custom-dsh-home");
    try {
      expect(getConfigRoot("dsh", "global")).toBe(join(testDir, "custom-dsh-home"));
    } finally {
      delete process.env.DSH_HOME;
    }
  });

  it("skills root is <root>/skills (project .dsh/skills, global $DSH_HOME/skills)", async () => {
    const { getSkillsRoot } = await import("../src/utils/paths.js");
    expect(getSkillsRoot("dsh", "project", join(testDir, "proj"))).toBe(
      join(testDir, "proj", ".dsh", "skills")
    );
    expect(getSkillsRoot("dsh", "global")).toBe(join(testDir, ".dsh", "skills"));
  });

  it("instructions file for dsh is AGENTS.md", async () => {
    const { getConfigFileName } = await import("../src/utils/paths.js");
    expect(getConfigFileName("dsh")).toBe("AGENTS.md");
  });

  it("cannbot registry static layout prior covers dsh so status/uninstall resolve", async () => {
    const { CANNBOT_TOOL_LAYOUTS } = await import("../src/core/cannbot-registry.js");
    expect(CANNBOT_TOOL_LAYOUTS.dsh).toEqual({
      configDirs: [".dsh"],
      defaultConfigDir: ".dsh",
    });
  });

  it("tool display name renders DSH", async () => {
    const { getToolDisplayName } = await import("../src/core/detector.js");
    expect(getToolDisplayName("dsh")).toBe("DSH");
  });

  it("detectTools discovers dsh via DSH_HOME", async () => {
    const dshHome = join(testDir, "detect-me");
    mkdirSync(dshHome, { recursive: true });
    process.env.DSH_HOME = dshHome;
    try {
      const { detectTools } = await import("../src/core/detector.js");
      const found = await detectTools();
      const dsh = found.find((t) => t.name === "dsh");
      expect(dsh).toBeDefined();
      expect(dsh!.path).toBe(dshHome);
    } finally {
      delete process.env.DSH_HOME;
    }
  });
});
