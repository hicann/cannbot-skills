// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

// Qoder tool support — end-to-end install/uninstall coverage.
//
// Layout facts verified against qodercn 1.1.66 (docs.qoder.cn + live probes):
//   CLI:   project `.qoder/{skills,agents}`, project-root AGENTS.md memory,
//          user `~/.qoder-cn` (agents + AGENTS.md), user skills `~/.agents/skills`
//   IDE:   `.lingma/{skills,agents}`, user `~/.lingma`
// The layout is forced via CANNBOT_QODER_LAYOUT so tests do not depend on
// whether qodercn happens to be installed.

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { mkdirSync, writeFileSync, rmSync, readFileSync, existsSync, lstatSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

vi.mock("../src/ui/backup-prompts.js", () => ({
  showOverwriteWarning: vi.fn(async () => "overwrite" as const),
  showUnownedFileWarning: vi.fn(async () => "keep" as const),
  showRestorePrompt: vi.fn(async () => "none" as const),
}));

let testDir: string;
let savedHome: string | undefined;

function setup() {
  testDir = join(tmpdir(), `ih-qoder-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
  // Sandbox the install bookkeeping (records + config.yaml live under
  // getCannbotConfigDir() → HOME); keep the real ~/.cannbot untouched.
  savedHome = process.env.HOME;
  process.env.HOME = testDir;
}

function teardown() {
  rmSync(testDir, { recursive: true, force: true });
  process.env.HOME = savedHome;
}

function makeFixture() {
  const repoDir = join(testDir, "repo");
  const projectDir = join(testDir, "project");
  const pluginDir = join(repoDir, "zz-qoder-plugin");
  mkdirSync(join(pluginDir, "skills", "demo-skill"), { recursive: true });
  writeFileSync(join(pluginDir, "skills", "demo-skill", "SKILL.md"), "---\nname: demo-skill\n---\n");
  writeFileSync(join(pluginDir, "AGENTS.md"), "# Plugin agents");
  mkdirSync(join(projectDir), { recursive: true });
  writeFileSync(join(projectDir, "AGENTS.md"), "# User config");
  return { repoDir, projectDir, pluginDir };
}

function pluginEntry() {
  return {
    id: "zz-qoder-plugin",
    dir: "zz-qoder-plugin",
    displayName: "ZZ Qoder Plugin",
    script: "init.sh",
    aliases: [],
    skills: 1,
    agents: 1,
    description: "",
    configFile: "AGENTS.md",
    installSkills: [{ dir: "zz-qoder-plugin/skills", skills: ["demo-skill"] }],
    installAgents: ["demo-agent"],
  };
}

describe("qoder install (CLI layout)", () => {
  beforeEach(() => {
    setup();
    vi.clearAllMocks();
    process.env.CANNBOT_QODER_LAYOUT = "cli";
  });
  afterEach(() => {
    delete process.env.CANNBOT_QODER_LAYOUT;
    teardown();
  });

  it("installs skills, agents and project-root AGENTS.md into the qodercn layout", async () => {
    const { installPlugin } = await import("../src/core/installer.js");
    const { deleteRecord } = await import("../src/core/record.js");
    const { repoDir, projectDir, pluginDir } = makeFixture();
    mkdirSync(join(pluginDir, "agents"), { recursive: true });
    writeFileSync(join(pluginDir, "agents", "demo-agent.md"), "---\nname: demo-agent\ndescription: d\n---\n");

    try {
      const result = await installPlugin({
        pluginId: "zz-qoder-plugin",
        tool: "qoder",
        level: "project",
        repoPath: repoDir,
        installPath: projectDir,
        plugin: pluginEntry() as any,
      });

      expect(result.success).toBe(true);
      // skills land in .qoder/skills (qodercn project scan root)
      expect(existsSync(join(projectDir, ".qoder", "skills", "demo-skill"))).toBe(true);
      // agents land in .qoder/agents (qodercn project agent root) as real
      // copies — the qodercn agent loader does not follow symlinks
      expect(existsSync(join(projectDir, ".qoder", "agents", "demo-agent.md"))).toBe(true);
      expect(lstatSync(join(projectDir, ".qoder", "agents", "demo-agent.md")).isSymbolicLink()).toBe(false);
      expect(readFileSync(join(projectDir, ".qoder", "agents", "demo-agent.md"), "utf-8"))
        .toContain("name: demo-agent");
      // per-plugin manifest is install-helper bookkeeping inside .qoder
      expect(existsSync(join(projectDir, ".qoder", "zz-qoder-plugin-manifest.json"))).toBe(true);
    } finally {
      deleteRecord("zz-qoder-plugin");
    }
  });

  it("unowned project AGENTS.md is preserved (--yes) — qodercn memory target stays untouched", async () => {
    const { installPlugin } = await import("../src/core/installer.js");
    const { deleteRecord } = await import("../src/core/record.js");
    const { repoDir, projectDir } = makeFixture();

    try {
      const result = await installPlugin({
        pluginId: "zz-qoder-plugin",
        tool: "qoder",
        level: "project",
        repoPath: repoDir,
        installPath: projectDir,
        yes: true,
        plugin: pluginEntry() as any,
      });

      expect(result.success).toBe(true);
      expect(readFileSync(join(projectDir, "AGENTS.md"), "utf-8")).toBe("# User config");
      expect(lstatSync(join(projectDir, "AGENTS.md")).isSymbolicLink()).toBe(false);
    } finally {
      deleteRecord("zz-qoder-plugin");
    }
  });

  it("uninstall removes the .qoder payload and keeps the preserved AGENTS.md", async () => {
    const { installPlugin } = await import("../src/core/installer.js");
    const { uninstallCommand } = await import("../src/commands/uninstall.js");
    const { repoDir, projectDir } = makeFixture();
    const origCwd = process.cwd();

    try {
      await installPlugin({
        pluginId: "zz-qoder-plugin",
        tool: "qoder",
        level: "project",
        repoPath: repoDir,
        installPath: projectDir,
        yes: true,
        plugin: pluginEntry() as any,
      });
      expect(existsSync(join(projectDir, ".qoder", "skills", "demo-skill"))).toBe(true);

      process.chdir(projectDir);
      await uninstallCommand(["zz-qoder-plugin"], { tool: "qoder", level: "project", yes: true });

      expect(existsSync(join(projectDir, ".qoder", "skills", "demo-skill"))).toBe(false);
      expect(readFileSync(join(projectDir, "AGENTS.md"), "utf-8")).toBe("# User config");
      expect(lstatSync(join(projectDir, "AGENTS.md")).isSymbolicLink()).toBe(false);
    } finally {
      process.chdir(origCwd);
      const { deleteRecord } = await import("../src/core/record.js");
      deleteRecord("zz-qoder-plugin");
    }
  });
});

describe("qoder install (IDE fallback layout)", () => {
  beforeEach(() => {
    setup();
    vi.clearAllMocks();
    process.env.CANNBOT_QODER_LAYOUT = "ide";
  });
  afterEach(() => {
    delete process.env.CANNBOT_QODER_LAYOUT;
    teardown();
  });

  it("installs skills and agents into .lingma for IDE users", async () => {
    const { installPlugin } = await import("../src/core/installer.js");
    const { deleteRecord } = await import("../src/core/record.js");
    const { repoDir, projectDir, pluginDir } = makeFixture();
    mkdirSync(join(pluginDir, "agents"), { recursive: true });
    writeFileSync(join(pluginDir, "agents", "demo-agent.md"), "---\nname: demo-agent\ndescription: d\n---\n");

    try {
      const result = await installPlugin({
        pluginId: "zz-qoder-plugin",
        tool: "qoder",
        level: "project",
        repoPath: repoDir,
        installPath: projectDir,
        plugin: pluginEntry() as any,
      });

      expect(result.success).toBe(true);
      expect(existsSync(join(projectDir, ".lingma", "skills", "demo-skill"))).toBe(true);
      expect(existsSync(join(projectDir, ".lingma", "agents", "demo-agent.md"))).toBe(true);
      expect(lstatSync(join(projectDir, ".lingma", "agents", "demo-agent.md")).isSymbolicLink()).toBe(false);
      expect(existsSync(join(projectDir, ".qoder"))).toBe(false);
    } finally {
      deleteRecord("zz-qoder-plugin");
    }
  });
});
