// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { mkdirSync, rmSync, writeFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { createFakeInstaller } from "./helpers/fake-cannbot.js";
import { seedMainlineCache, seedSkillsRepo } from "./helpers/seed-mainline-cache.js";

let testDir: string;
let sandboxHome: string;
let savedEnv: Record<string, string | undefined>;

function setup() {
  testDir = join(tmpdir(), `ih-info-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
  sandboxHome = join(testDir, "home");
  mkdirSync(join(sandboxHome, ".cannbot"), { recursive: true });
  // pre-seed a fresh mainline cache: discovery must not attempt a real
  // git clone from unit tests
  seedMainlineCache(sandboxHome);
  seedSkillsRepo(sandboxHome);
  savedEnv = {
    HOME: process.env.HOME,
    CANNBOT_INSTALLER_PATH: process.env.CANNBOT_INSTALLER_PATH,
  };
  process.env.HOME = sandboxHome;
  delete process.env.CANNBOT_INSTALLER_PATH;
}

function teardown() {
  for (const [key, value] of Object.entries(savedEnv)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  rmSync(testDir, { recursive: true, force: true });
}

describe("info", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("resolves cannbot-only plugins via installer discovery", async () => {
    // info shares the registry with list/install/init/update, all of which
    // run discovery first. Cannbot-only plugins (no embedded entry, no
    // tombstone) must resolve through discovery — otherwise `info <id>`
    // reports not-found while `list` shows the very same plugin.
    const installerRoot = join(testDir, "installer");
    createFakeInstaller(installerRoot, "fake-cb-plugin");
    process.env.CANNBOT_INSTALLER_PATH = installerRoot;

    const lines: string[] = [];
    const spy = vi.spyOn(console, "log").mockImplementation((...args: unknown[]) => {
      lines.push(args.map(String).join(" "));
    });
    try {
      const { infoCommand } = await import("../src/commands/info.js");
      await infoCommand("fake-cb-plugin");
    } finally {
      spy.mockRestore();
    }

    const out = lines.join("\n");
    // not-found path prints the info_not_found marker in either locale
    expect(out).not.toContain("未找到插件");
    expect(out).not.toContain("Plugin not found");
    // found path renders the cannbot badge and the plugin id
    expect(out).toContain("[cannbot]");
    expect(out).toContain("fake-cb-plugin");
  });

  it("resolves community plugins via the repository scan", async () => {
    // community plugins (autoresearch, catlass-dsl-generator, ...) are
    // discovered by mergeDynamicPlugins in ensureRepoAndScan — info must
    // trigger the scan, not just cannbot discovery
    const installerRoot = join(testDir, "installer-community");
    createFakeInstaller(installerRoot, "fake-cb-plugin");
    process.env.CANNBOT_INSTALLER_PATH = installerRoot;
    // plant a community plugin in the seeded skills repo
    const communityPlugin = join(sandboxHome, ".cannbot", "repo", "plugins-community", "autoresearch");
    mkdirSync(communityPlugin, { recursive: true });
    writeFileSync(join(communityPlugin, "init.sh"), "#!/bin/bash\n");
    mkdirSync(join(communityPlugin, ".claude-plugin"), { recursive: true });
    writeFileSync(
      join(communityPlugin, ".claude-plugin", "plugin.json"),
      JSON.stringify({ name: "autoresearch", description: "AutoResearch 算子自动优化工作流" })
    );

    const lines: string[] = [];
    const spy = vi.spyOn(console, "log").mockImplementation((...args: unknown[]) => {
      lines.push(args.map(String).join(" "));
    });
    try {
      const { infoCommand } = await import("../src/commands/info.js");
      await infoCommand("autoresearch");
    } finally {
      spy.mockRestore();
    }

    const out = lines.join("\n");
    expect(out).not.toContain("未找到插件");
    expect(out).toContain("autoresearch");
    expect(out).toContain("AutoResearch");
  });
});
