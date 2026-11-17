// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------


import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdirSync, writeFileSync, rmSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { createFakeInstaller } from "./helpers/fake-cannbot.js";
import { seedMainlineCache } from "./helpers/seed-mainline-cache.js";

let testDir: string;
let sandboxHome: string;
let savedEnv: Record<string, string | undefined>;

function setup() {
  testDir = join(tmpdir(), `ih-manifestless-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
  sandboxHome = join(testDir, "home");
  mkdirSync(join(sandboxHome, ".cannbot"), { recursive: true });
  seedMainlineCache(sandboxHome);
  savedEnv = { HOME: process.env.HOME, CANNBOT_INSTALLER_PATH: process.env.CANNBOT_INSTALLER_PATH };
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

describe("manifest-less cannbot plugin discovery", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("discovers a cannbot repo directory without a manifest (layer 2 fallback)", async () => {
    // a plugin directory in the cannbot repo that has skills/ but no
    // .claude-plugin/plugin.json — the plugin IS in the cannbot repo, it
    // just lacks a manifest declaration. Discovery must still list it as
    // a cannbot entry (directory identity, derived counts), not let the
    // frozen plugins.d yml copy take over the display.
    const installerRoot = join(testDir, "installer");
    createFakeInstaller(installerRoot, "fake-cb-plugin");
    // add a manifest-less plugin with real skills
    const manifestless = join(installerRoot, "dist", "plugins", "ops-registry-invoke");
    mkdirSync(join(manifestless, "skills", "ops-requirement-analysis"), { recursive: true });
    writeFileSync(
      join(manifestless, "skills", "ops-requirement-analysis", "SKILL.md"),
      "---\nname: ops-requirement-analysis\ndescription: requirement analysis\n---\n# body\n"
    );
    mkdirSync(join(manifestless, "skills", "ops-registry-invoke"), { recursive: true });
    writeFileSync(
      join(manifestless, "skills", "ops-registry-invoke", "SKILL.md"),
      "---\nname: ops-registry-invoke\ndescription: registry invoke skill\n---\n# body\n"
    );
    mkdirSync(join(manifestless, "agents"), { recursive: true });
    writeFileSync(join(manifestless, "agents", "coder.md"), "---\nname: coder\n---\nbody\n");
    process.env.CANNBOT_INSTALLER_PATH = installerRoot;

    const { discoverCannbotPlugins } = await import("../src/core/cannbot-installer.js");
    const plugins = discoverCannbotPlugins();

    const found = plugins.find((p) => p.id === "ops-registry-invoke");
    expect(found).toBeDefined();
    expect(found?.displayName).toBe("ops-registry-invoke");
    expect(found?.skillsCount).toBe(2);
    expect(found?.agentsCount).toBe(1);
    expect(found?.description).toBe("");
  });

  it("skips directories that are not plugins (no skills/ and no agents/)", async () => {
    // test fixtures and incidental directories must not enter the list
    const installerRoot = join(testDir, "installer-fixture");
    createFakeInstaller(installerRoot, "fake-cb-plugin");
    const fixture = join(installerRoot, "dist", "plugins", "fixture-plugin");
    mkdirSync(fixture, { recursive: true });
    writeFileSync(join(fixture, "README.md"), "# test fixture\n");
    process.env.CANNBOT_INSTALLER_PATH = installerRoot;

    const { discoverCannbotPlugins } = await import("../src/core/cannbot-installer.js");
    const plugins = discoverCannbotPlugins();
    expect(plugins.find((p) => p.id === "fixture-plugin")).toBeUndefined();
  });
});
