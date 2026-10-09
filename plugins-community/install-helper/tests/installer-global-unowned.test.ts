// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { mkdirSync, writeFileSync, rmSync, readFileSync, existsSync, lstatSync, readdirSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

vi.mock("../src/ui/backup-prompts.js", () => ({
  showOverwriteWarning: vi.fn(async () => "overwrite" as const),
  showUnownedFileWarning: vi.fn(async () => "keep" as const),
  showRestorePrompt: vi.fn(async () => "none" as const),
}));

let testDir: string;
let originalHome: string | undefined;

function setup() {
  testDir = join(tmpdir(), `ih-glbl-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
  originalHome = process.env.HOME;
}

function teardown() {
  if (originalHome === undefined) delete process.env.HOME;
  else process.env.HOME = originalHome;
  rmSync(testDir, { recursive: true, force: true });
}

describe("global-level unowned config protection", () => {
  beforeEach(() => {
    setup();
    vi.clearAllMocks();
  });
  afterEach(teardown);

  const PLUGIN_ID = "zz-global-unowned";

  function makeFixture() {
    // Sandbox HOME so the global config root (~/.config/opencode) is isolated
    // from the real user environment.
    const home = join(testDir, "home");
    mkdirSync(home, { recursive: true });
    process.env.HOME = home;

    const globalConfigRoot = join(home, ".config", "opencode");
    mkdirSync(globalConfigRoot, { recursive: true });
    writeFileSync(join(globalConfigRoot, "AGENTS.md"), "# User global config");

    const repoDir = join(testDir, "repo");
    const pluginDir = join(repoDir, PLUGIN_ID);
    mkdirSync(join(pluginDir, "skills", "demo-skill"), { recursive: true });
    writeFileSync(join(pluginDir, "skills", "demo-skill", "SKILL.md"), "---\nname: demo-skill\n---\n");
    writeFileSync(join(pluginDir, "AGENTS.md"), "# Plugin agents");

    return { home, globalConfigRoot, repoDir };
  }

  function pluginEntry() {
    return {
      id: PLUGIN_ID,
      dir: PLUGIN_ID,
      displayName: "ZZ Global Unowned",
      script: "init.sh",
      aliases: [],
      skills: 1,
      agents: 0,
      description: "",
      configFile: "AGENTS.md",
      installSkills: [{ dir: `${PLUGIN_ID}/skills`, skills: ["demo-skill"] }],
      installAgents: [],
    };
  }

  it("--yes preserves an unowned global AGENTS.md without prompting", async () => {
    const { installPlugin } = await import("../src/core/installer.js");
    const { showUnownedFileWarning } = await import("../src/ui/backup-prompts.js");
    const { globalConfigRoot, repoDir } = makeFixture();

    const result = await installPlugin({
      pluginId: PLUGIN_ID,
      tool: "opencode",
      level: "global",
      repoPath: repoDir,
      yes: true,
      plugin: pluginEntry() as any,
    });

    expect(result.success).toBe(true);
    expect(showUnownedFileWarning).not.toHaveBeenCalled();
    expect(readFileSync(join(globalConfigRoot, "AGENTS.md"), "utf-8")).toBe("# User global config");
    expect(lstatSync(join(globalConfigRoot, "AGENTS.md")).isSymbolicLink()).toBe(false);
    // the rest of the plugin still installs into the global skills root
    expect(existsSync(join(globalConfigRoot, "skills", "demo-skill"))).toBe(true);
  });

  it("replace choice backs up beside the global config file", async () => {
    const { installPlugin } = await import("../src/core/installer.js");
    const { showUnownedFileWarning } = await import("../src/ui/backup-prompts.js");
    const { globalConfigRoot, repoDir } = makeFixture();
    vi.mocked(showUnownedFileWarning).mockResolvedValueOnce("replace");

    const result = await installPlugin({
      pluginId: PLUGIN_ID,
      tool: "opencode",
      level: "global",
      repoPath: repoDir,
      plugin: pluginEntry() as any,
    });

    expect(result.success).toBe(true);
    expect(showUnownedFileWarning).toHaveBeenCalledWith(
      join(globalConfigRoot, "AGENTS.md"),
      "ZZ Global Unowned"
    );
    expect(lstatSync(join(globalConfigRoot, "AGENTS.md")).isSymbolicLink()).toBe(true);
    const backups = readdirSync(globalConfigRoot).filter((f) => f.startsWith("AGENTS.md.cannbot-backup.unowned."));
    expect(backups.length).toBe(1);
    expect(readFileSync(join(globalConfigRoot, backups[0]), "utf-8")).toBe("# User global config");
  });

  it("cancel leaves the unowned global AGENTS.md untouched", async () => {
    const { installPlugin } = await import("../src/core/installer.js");
    const { showUnownedFileWarning } = await import("../src/ui/backup-prompts.js");
    const { globalConfigRoot, repoDir } = makeFixture();
    vi.mocked(showUnownedFileWarning).mockResolvedValueOnce("cancel");

    const result = await installPlugin({
      pluginId: PLUGIN_ID,
      tool: "opencode",
      level: "global",
      repoPath: repoDir,
      plugin: pluginEntry() as any,
    });

    expect(result.success).toBe(false);
    expect(readFileSync(join(globalConfigRoot, "AGENTS.md"), "utf-8")).toBe("# User global config");
    expect(existsSync(join(globalConfigRoot, "skills", "demo-skill"))).toBe(false);
  });
});
