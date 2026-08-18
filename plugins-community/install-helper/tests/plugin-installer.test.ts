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
import { mkdirSync, writeFileSync, existsSync, symlinkSync, readFileSync, rmSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

let testDir: string;

function setup() {
  testDir = join(tmpdir(), `ih-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
}

function teardown() {
  rmSync(testDir, { recursive: true, force: true });
}

describe("plugin-installer", () => {
  beforeEach(setup);
  afterEach(teardown);

  describe("installViaManifest", () => {
    it("installs skills as symlinks", async () => {
      const { installViaManifest } = await import("../src/core/plugin-installer.js");
      const repoPath = testDir;
      const pluginDir = join(testDir, "my-plugin");
      const skillsDir = join(pluginDir, "skills", "my-skill");
      const agentsDir = join(pluginDir, "agents");
      const configRoot = join(testDir, ".opencode");
      mkdirSync(skillsDir, { recursive: true });
      mkdirSync(agentsDir, { recursive: true });
      writeFileSync(join(skillsDir, "SKILL.md"), "---\nname: my-skill\n---\n# Content");
      writeFileSync(join(pluginDir, "AGENTS.md"), "# Agents");
      writeFileSync(join(agentsDir, "my-agent.md"), "# Agent");
      mkdirSync(configRoot, { recursive: true });

      const plugin = {
        id: "my-plugin",
        dir: "my-plugin",
        displayName: "My Plugin",
        script: "init.sh",
        aliases: [],
        skills: 0,
        agents: 0,
        description: "",
        configFile: "AGENTS.md",
        installSkills: [{ dir: "my-plugin/skills", skills: ["my-skill"] }],
        installAgents: ["my-agent"],
      };

      const result = await installViaManifest(plugin, repoPath, "opencode", "project", testDir);
      expect(result.success).toBe(true);
      expect(result.skillsCount).toBe(1);
      expect(result.agentsCount).toBe(1);
      expect(existsSync(join(configRoot, "skills", "my-skill"))).toBe(true);
      expect(existsSync(join(configRoot, "agents", "my-agent.md"))).toBe(true);
      expect(existsSync(join(testDir, "AGENTS.md"))).toBe(true);
    });

    it("skips missing skill source gracefully", async () => {
      const { installViaManifest } = await import("../src/core/plugin-installer.js");
      const plugin = {
        id: "my-plugin",
        dir: "nonexistent",
        displayName: "My Plugin",
        script: "init.sh",
        aliases: [],
        skills: 0,
        agents: 0,
        description: "",
        configFile: "AGENTS.md",
        installSkills: [{ dir: "nonexistent", skills: ["missing-skill"] }],
        installAgents: [],
      };

      const result = await installViaManifest(plugin, testDir, "opencode", "project", testDir);
      expect(result.success).toBe(true);
      expect(result.skillsCount).toBe(0);
    });

    it("respects configRootConfigLink=false for project level", async () => {
      const { installViaManifest } = await import("../src/core/plugin-installer.js");
      const pluginDir = join(testDir, "my-plugin");
      mkdirSync(join(pluginDir), { recursive: true });
      writeFileSync(join(pluginDir, "AGENTS.md"), "# Agents");
      const configRoot = join(testDir, ".opencode");

      const plugin = {
        id: "my-plugin",
        dir: "my-plugin",
        displayName: "My Plugin",
        script: "init.sh",
        aliases: [],
        skills: 0,
        agents: 0,
        description: "",
        configFile: "AGENTS.md",
        configRootConfigLink: false,
        installSkills: [],
        installAgents: [],
      };

      await installViaManifest(plugin, testDir, "opencode", "project", testDir);
      expect(existsSync(join(testDir, "AGENTS.md"))).toBe(true);
      expect(existsSync(join(configRoot, "AGENTS.md"))).toBe(false);
    });

    it("creates configRoot AGENTS.md when configRootConfigLink=true", async () => {
      const { installViaManifest } = await import("../src/core/plugin-installer.js");
      const pluginDir = join(testDir, "my-plugin");
      mkdirSync(join(pluginDir), { recursive: true });
      writeFileSync(join(pluginDir, "AGENTS.md"), "# Agents");
      const configRoot = join(testDir, ".opencode");

      const plugin = {
        id: "my-plugin",
        dir: "my-plugin",
        displayName: "My Plugin",
        script: "init.sh",
        aliases: [],
        skills: 0,
        agents: 0,
        description: "",
        configFile: "AGENTS.md",
        configRootConfigLink: true,
        installSkills: [],
        installAgents: [],
      };

      await installViaManifest(plugin, testDir, "opencode", "project", testDir);
      expect(existsSync(join(testDir, "AGENTS.md"))).toBe(true);
      expect(existsSync(join(configRoot, "AGENTS.md"))).toBe(true);
    });

    it("installs workflows symlink when plugin has workflows dir", async () => {
      const { installViaManifest } = await import("../src/core/plugin-installer.js");
      const pluginDir = join(testDir, "my-plugin");
      const workflowsDir = join(pluginDir, "workflows");
      mkdirSync(workflowsDir, { recursive: true });
      writeFileSync(join(workflowsDir, "task.md"), "# Task");
      writeFileSync(join(pluginDir, "AGENTS.md"), "# Agents");
      const configRoot = join(testDir, ".opencode");

      const plugin = {
        id: "my-plugin",
        dir: "my-plugin",
        displayName: "My Plugin",
        script: "init.sh",
        aliases: [],
        skills: 0,
        agents: 0,
        description: "",
        configFile: "AGENTS.md",
        installSkills: [],
        installAgents: [],
      };

      await installViaManifest(plugin, testDir, "opencode", "project", testDir);
      expect(existsSync(join(configRoot, "workflows"))).toBe(true);
    });

    it("skips workflows when plugin has no workflows dir", async () => {
      const { installViaManifest } = await import("../src/core/plugin-installer.js");
      const pluginDir = join(testDir, "my-plugin");
      mkdirSync(pluginDir, { recursive: true });
      writeFileSync(join(pluginDir, "AGENTS.md"), "# Agents");
      const configRoot = join(testDir, ".opencode");

      const plugin = {
        id: "my-plugin",
        dir: "my-plugin",
        displayName: "My Plugin",
        script: "init.sh",
        aliases: [],
        skills: 0,
        agents: 0,
        description: "",
        configFile: "AGENTS.md",
        installSkills: [],
        installAgents: [],
      };

      await installViaManifest(plugin, testDir, "opencode", "project", testDir);
      expect(existsSync(join(configRoot, "workflows"))).toBe(false);
    });

    it("writes cannbot-manifest.json with correct data", async () => {
      const { installViaManifest } = await import("../src/core/plugin-installer.js");
      const pluginDir = join(testDir, "my-plugin");
      mkdirSync(pluginDir, { recursive: true });
      writeFileSync(join(pluginDir, "AGENTS.md"), "# Agents");
      const configRoot = join(testDir, ".opencode");

      const plugin = {
        id: "my-plugin",
        dir: "my-plugin",
        displayName: "My Plugin",
        script: "init.sh",
        aliases: [],
        skills: 0,
        agents: 0,
        description: "",
        version: "1.0.0",
        configFile: "AGENTS.md",
        installSkills: [],
        installAgents: [],
      };

      const result = await installViaManifest(plugin, testDir, "opencode", "project", testDir);
      const manifestPath = join(configRoot, "my-plugin-manifest.json");
      expect(existsSync(manifestPath)).toBe(true);
      const manifest = JSON.parse(readFileSync(manifestPath, "utf-8"));
      expect(manifest.team).toBe("my-plugin");
      expect(manifest.version).toBe("1.0.0");
      expect(manifest.tool).toBe("opencode");
      expect(manifest.level).toBe("project");
    });

    it("handles skill rename (as field)", async () => {
      const { installViaManifest } = await import("../src/core/plugin-installer.js");
      const pluginDir = join(testDir, "my-plugin");
      const skillDir = join(pluginDir, "workflow");
      mkdirSync(skillDir, { recursive: true });
      writeFileSync(join(skillDir, "SKILL.md"), "---\nname: workflow\n---\n");
      writeFileSync(join(pluginDir, "AGENTS.md"), "# Agents");
      const configRoot = join(testDir, ".opencode");

      const plugin = {
        id: "my-plugin",
        dir: "my-plugin",
        displayName: "My Plugin",
        script: "init.sh",
        aliases: [],
        skills: 0,
        agents: 0,
        description: "",
        configFile: "AGENTS.md",
        installSkills: [{ dir: "my-plugin", skills: [{ name: "workflow", as: "workflow-renamed" }] }],
        installAgents: [],
      };

      const result = await installViaManifest(plugin, testDir, "opencode", "project", testDir);
      expect(result.success).toBe(true);
      expect(existsSync(join(configRoot, "skills", "workflow-renamed"))).toBe(true);
      expect(result.manifest?.installed_skills).toContain("workflow-renamed");
    });

    it("writes per-plugin manifest (no overwrite on multi-plugin install)", async () => {
      const { installViaManifest } = await import("../src/core/plugin-installer.js");
      const { getConfigRoot } = await import("../src/utils/paths.js");

      const pluginDir1 = join(testDir, "plugin1");
      const pluginDir2 = join(testDir, "plugin2");
      mkdirSync(join(pluginDir1, "skills", "skill-a"), { recursive: true });
      mkdirSync(join(pluginDir2, "skills", "skill-b"), { recursive: true });
      writeFileSync(join(pluginDir1, "skills", "skill-a", "SKILL.md"), "---\nname: skill-a\n---\n");
      writeFileSync(join(pluginDir2, "skills", "skill-b", "SKILL.md"), "---\nname: skill-b\n---\n");

      const origCwd = process.cwd();
      process.chdir(testDir);
      try {
        const configRoot = getConfigRoot("opencode", "project");
        mkdirSync(join(configRoot, "skills"), { recursive: true });
        mkdirSync(join(configRoot, "agents"), { recursive: true });

        const plugin1 = {
          id: "plugin-a",
          dir: "plugin1",
          displayName: "Plugin A",
          script: "init.sh",
          aliases: [],
          skills: 1,
          agents: 0,
          description: "",
          configFile: "AGENTS.md",
          installSkills: [{ dir: "plugin1/skills", skills: ["skill-a"] }],
          installAgents: [],
        };

        const plugin2 = {
          id: "plugin-b",
          dir: "plugin2",
          displayName: "Plugin B",
          script: "init.sh",
          aliases: [],
          skills: 1,
          agents: 0,
          description: "",
          configFile: "AGENTS.md",
          installSkills: [{ dir: "plugin2/skills", skills: ["skill-b"] }],
          installAgents: [],
        };

        await installViaManifest(plugin1 as any, testDir, "opencode", "project", testDir);
        await installViaManifest(plugin2 as any, testDir, "opencode", "project", testDir);

        expect(existsSync(join(configRoot, "plugin-a-manifest.json"))).toBe(true);
        expect(existsSync(join(configRoot, "plugin-b-manifest.json"))).toBe(true);
      } finally {
        process.chdir(origCwd);
      }
    });

    it("creates symlink even when git pull fails on existing external repo", async () => {
      const { installViaManifest } = await import("../src/core/plugin-installer.js");

      const pluginDir = join(testDir, "my-plugin");
      const extRepoDir = join(pluginDir, "ext-repo");
      mkdirSync(join(extRepoDir, ".git"), { recursive: true });
      writeFileSync(join(extRepoDir, "README.md"), "ext-repo content");
      writeFileSync(join(pluginDir, "AGENTS.md"), "# Agents");

      const plugin = {
        id: "my-plugin",
        dir: "my-plugin",
        displayName: "My Plugin",
        script: "init.sh",
        aliases: [],
        skills: 0,
        agents: 0,
        description: "",
        configFile: "AGENTS.md",
        installSkills: [],
        installAgents: [],
        externalRepos: [{ url: "https://example.com/ext.git", dir: "my-plugin/ext-repo" }],
      };

      const configRoot = join(testDir, ".opencode");
      mkdirSync(join(configRoot, "skills"), { recursive: true });
      mkdirSync(join(configRoot, "agents"), { recursive: true });

      await installViaManifest(plugin as any, testDir, "opencode", "project", testDir);

      // git pull fails (not a real repo), but dir exists → symlink is still created
      expect(existsSync(join(extRepoDir, "README.md"))).toBe(true);
      // project-level symlink must be created even when pull fails
      expect(existsSync(join(testDir, "ext-repo"))).toBe(true);
    });
  });
});
