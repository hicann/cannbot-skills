// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

// Regression tests for the dual-source entry-resolution bug: installPlugin()
// used to re-resolve the plugin by bare id via getPluginById() (cannbot-first,
// source-blind), so a caller-routed legacy entry (--source skills / wizard
// skills choice / legacy update record) for a plugin that also exists in the
// cannbot registry was silently swapped for the cannbot entry, whose absolute
// dir and sentinel script ("cannbot") violate the legacy script-installer
// contract -> double-joined repoPath + error_script_not_found.

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdirSync, writeFileSync, rmSync } from "fs";
import { join, dirname } from "path";
import { tmpdir } from "os";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let testDir: string;
let savedHome: string | undefined;

function setup() {
  testDir = join(tmpdir(), `ih-dsr-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
  // installPlugin() bookkeeping (writeRecord) writes under ~/.cannbot — sandbox it
  savedHome = process.env.HOME;
  process.env.HOME = testDir;
}

async function teardown() {
  const { setCannbotPlugins } = await import("../src/core/registry.js");
  setCannbotPlugins([]);
  rmSync(testDir, { recursive: true, force: true });
  process.env.HOME = savedHome;
}

function writeBenignInitSh(dir: string): void {
  mkdirSync(dir, { recursive: true });
  // mentions tool tokens so scriptSupportsTool passes; exits 0
  writeFileSync(
    join(dir, "init.sh"),
    "#!/bin/bash\n# tools: opencode claude codex trae\nexit 0\n"
  );
}

describe("installer dual-source entry resolution", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("honors the caller-resolved legacy entry when a same-name cannbot entry exists", async () => {
    const repoPath = join(testDir, "repo");
    writeBenignInitSh(join(repoPath, "plugins-official", "ops-direct-invoke"));

    // absolute cannbot package layout WITHOUT a `cannbot` script (only init.sh)
    const cannbotDir = join(testDir, "cannbot-pkg", "dist", "plugins", "ops-direct-invoke");
    writeBenignInitSh(cannbotDir);

    const { installPlugin } = await import("../src/core/installer.js");
    const { setCannbotPlugins } = await import("../src/core/registry.js");

    setCannbotPlugins([
      {
        id: "ops-direct-invoke",
        dir: cannbotDir,
        displayName: "ops-direct-invoke",
        script: "cannbot",
        aliases: [],
        skills: 0,
        agents: 0,
        description: "",
        source: "cannbot",
      },
    ]);

    const target = join(testDir, "target");
    mkdirSync(target, { recursive: true });

    const result = await installPlugin({
      pluginId: "ops-direct-invoke",
      tool: "opencode",
      level: "project",
      repoPath,
      installPath: target,
      yes: true,
      plugin: {
        id: "ops-direct-invoke",
        dir: "plugins-official/ops-direct-invoke",
        displayName: "AscendC Kernel 直调",
        script: "init.sh",
        aliases: [],
        skills: 0,
        agents: 0,
        description: "",
      },
    } as Parameters<typeof installPlugin>[0]);

    expect(result.errors).toEqual([]);
    expect(result.success).toBe(true);
  });

  it("does not double-join repoPath when plugin.dir is absolute", async () => {
    const repoPath = join(testDir, "repo");
    mkdirSync(repoPath, { recursive: true });

    // an entry whose dir is already absolute (e.g. discovered from a managed
    // package) must resolve as-is, not repoPath + absoluteDir
    const absPluginDir = join(testDir, "abs-plugins", "demo-plugin");
    writeBenignInitSh(absPluginDir);

    const { installPlugin } = await import("../src/core/installer.js");

    const target = join(testDir, "target");
    mkdirSync(target, { recursive: true });

    const result = await installPlugin({
      pluginId: "demo-plugin",
      tool: "opencode",
      level: "project",
      repoPath,
      installPath: target,
      yes: true,
      plugin: {
        id: "demo-plugin",
        dir: absPluginDir,
        displayName: "demo-plugin",
        script: "init.sh",
        aliases: [],
        skills: 0,
        agents: 0,
        description: "",
      },
    } as Parameters<typeof installPlugin>[0]);

    expect(result.errors).toEqual([]);
    expect(result.success).toBe(true);
  });

  it("rejects cannbot-source entries with a routing error instead of a bogus script path", async () => {
    const repoPath = join(testDir, "repo");
    mkdirSync(repoPath, { recursive: true });

    const cannbotDir = join(testDir, "cannbot-pkg", "dist", "plugins", "ops-direct-invoke");
    writeBenignInitSh(cannbotDir);

    const { installPlugin } = await import("../src/core/installer.js");
    const { setCannbotPlugins } = await import("../src/core/registry.js");

    setCannbotPlugins([
      {
        id: "ops-direct-invoke",
        dir: cannbotDir,
        displayName: "ops-direct-invoke",
        script: "cannbot",
        aliases: [],
        skills: 0,
        agents: 0,
        description: "",
        source: "cannbot",
      },
    ]);

    // no caller-resolved entry: getPluginById fallback resolves the cannbot
    // entry — the legacy installer must refuse it with a routing error, not a
    // double-joined "script not found" path
    const result = await installPlugin({
      pluginId: "ops-direct-invoke",
      tool: "opencode",
      level: "project",
      repoPath,
      yes: true,
    });

    expect(result.success).toBe(false);
    expect(result.errors[0]).not.toContain(repoPath);
    expect(result.errors[0].toLowerCase()).toContain("cannbot");
  });
});
