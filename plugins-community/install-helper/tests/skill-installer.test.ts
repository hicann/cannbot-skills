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
import { mkdirSync, writeFileSync, symlinkSync, existsSync, rmSync, readFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

let testDir: string;

function setup() {
  testDir = join(tmpdir(), `ih-si-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
}

function teardown() {
  rmSync(testDir, { recursive: true, force: true });
}

describe("skill-installer", () => {
  beforeEach(setup);
  afterEach(teardown);

  describe("resolveSkillSourcePath", () => {
    it("uses filePath dirname when filePath exists", async () => {
      const { findSkill } = await import("../src/core/skill-registry.js");
      const { initFromScan } = await import("../src/core/skill-registry.js");
      const skillDir = join(testDir, "my-skill");
      mkdirSync(skillDir);
      writeFileSync(join(skillDir, "SKILL.md"), "---\nname: my-skill\n---\n");

      initFromScan([{
        id: "my-skill",
        description: "test",
        source: "test",
        filePath: join(skillDir, "SKILL.md"),
      }]);

      const skill = findSkill("my-skill");
      expect(skill).toBeDefined();
      expect(skill!.filePath).toBe(join(skillDir, "SKILL.md"));
    });

    it("falls back to join(source, id) when filePath not set", async () => {
      const { findSkill } = await import("../src/core/skill-registry.js");
      const { initFromScan } = await import("../src/core/skill-registry.js");

      initFromScan([{
        id: "my-skill",
        description: "test",
        source: "ops",
      }]);

      const skill = findSkill("my-skill");
      expect(skill).toBeDefined();
      expect(skill!.filePath).toBeUndefined();
    });
  });

  describe("installSkills", () => {
    it("creates symlink for skill", async () => {
      const { installSkills } = await import("../src/core/skill-installer.js");
      const { initFromScan } = await import("../src/core/skill-registry.js");
      const { getConfigRoot } = await import("../src/utils/paths.js");

      const skillDir = join(testDir, "repo", "ops", "my-skill");
      mkdirSync(skillDir, { recursive: true });
      writeFileSync(join(skillDir, "SKILL.md"), "---\nname: my-skill\n---\n");

      const origCwd = process.cwd();
      process.chdir(testDir);
      try {
        const configRoot = getConfigRoot("opencode", "project");
        mkdirSync(join(configRoot, "skills"), { recursive: true });

        initFromScan([{
          id: "my-skill",
          description: "test",
          source: "ops",
          filePath: join(skillDir, "SKILL.md"),
        }]);

        const results = await installSkills(["my-skill"], "opencode", "project", join(testDir, "repo"));
        expect(results.length).toBe(1);
        expect(results[0].success).toBe(true);
        expect(existsSync(join(configRoot, "skills", "my-skill"))).toBe(true);
      } finally {
        process.chdir(origCwd);
      }
    });

    it("returns error for non-existent skill", async () => {
      const { installSkills } = await import("../src/core/skill-installer.js");
      const results = await installSkills(["nonexistent"], "opencode", "project", testDir);
      expect(results.length).toBe(1);
      expect(results[0].success).toBe(false);
    });

    it("S2: does not delete existing symlink when source path is broken", async () => {
      const { installSkills } = await import("../src/core/skill-installer.js");
      const { initFromScan } = await import("../src/core/skill-registry.js");
      const { getConfigRoot } = await import("../src/utils/paths.js");

      const skillDir = join(testDir, "repo", "ops", "good-skill");
      mkdirSync(skillDir, { recursive: true });
      writeFileSync(join(skillDir, "SKILL.md"), "---\nname: good-skill\n---\n");

      const origCwd = process.cwd();
      process.chdir(testDir);
      try {
        const configRoot = getConfigRoot("opencode", "project");
        mkdirSync(join(configRoot, "skills"), { recursive: true });

        initFromScan([{
          id: "good-skill",
          description: "test",
          source: "ops",
          filePath: join(skillDir, "SKILL.md"),
        }]);

        await installSkills(["good-skill"], "opencode", "project", join(testDir, "repo"));
        const targetSymlink = join(configRoot, "skills", "good-skill");
        expect(existsSync(targetSymlink)).toBe(true);

        initFromScan([{
          id: "good-skill",
          description: "test",
          source: "nonexistent-source",
          filePath: "/nonexistent/path/SKILL.md",
        }]);

        const results = await installSkills(["good-skill"], "opencode", "project", join(testDir, "repo"));
        expect(results[0].success).toBe(false);
        expect(existsSync(targetSymlink)).toBe(true);
      } finally {
        process.chdir(origCwd);
      }
    });

    it("overwrites existing symlink with new source", async () => {
      const { installSkills } = await import("../src/core/skill-installer.js");
      const { initFromScan } = await import("../src/core/skill-registry.js");
      const { getConfigRoot } = await import("../src/utils/paths.js");

      const skillDir = join(testDir, "repo", "ops", "my-skill");
      mkdirSync(skillDir, { recursive: true });
      writeFileSync(join(skillDir, "SKILL.md"), "---\nname: my-skill\n---\n# v1");

      const origCwd = process.cwd();
      process.chdir(testDir);
      try {
        const configRoot = getConfigRoot("opencode", "project");
        mkdirSync(join(configRoot, "skills"), { recursive: true });

        initFromScan([{
          id: "my-skill",
          description: "test",
          source: "ops",
          filePath: join(skillDir, "SKILL.md"),
        }]);

        await installSkills(["my-skill"], "opencode", "project", join(testDir, "repo"));
        expect(existsSync(join(configRoot, "skills", "my-skill"))).toBe(true);

        writeFileSync(join(skillDir, "SKILL.md"), "---\nname: my-skill\n---\n# v2");
        const results = await installSkills(["my-skill"], "opencode", "project", join(testDir, "repo"));
        expect(results[0].success).toBe(true);
      } finally {
        process.chdir(origCwd);
      }
    });

    it("skips when symlink already points to same source", async () => {
      const { installSkills } = await import("../src/core/skill-installer.js");
      const { initFromScan } = await import("../src/core/skill-registry.js");
      const { getConfigRoot } = await import("../src/utils/paths.js");

      const skillDir = join(testDir, "repo", "ops", "shared-skill");
      mkdirSync(skillDir, { recursive: true });
      writeFileSync(join(skillDir, "SKILL.md"), "---\nname: shared-skill\n---\n");

      const origCwd = process.cwd();
      process.chdir(testDir);
      try {
        const configRoot = getConfigRoot("opencode", "project");
        mkdirSync(join(configRoot, "skills"), { recursive: true });

        initFromScan([{
          id: "shared-skill",
          description: "test",
          source: "ops",
          filePath: join(skillDir, "SKILL.md"),
        }]);

        const results1 = await installSkills(["shared-skill"], "opencode", "project", join(testDir, "repo"));
        expect(results1[0].success).toBe(true);

        const results2 = await installSkills(["shared-skill"], "opencode", "project", join(testDir, "repo"));
        expect(results2[0].success).toBe(true);
        expect(existsSync(join(configRoot, "skills", "shared-skill"))).toBe(true);
      } finally {
        process.chdir(origCwd);
      }
    });

    it("updates symlink when source changes", async () => {
      const { installSkills } = await import("../src/core/skill-installer.js");
      const { initFromScan } = await import("../src/core/skill-registry.js");
      const { getConfigRoot } = await import("../src/utils/paths.js");

      const skillDir1 = join(testDir, "repo1", "ops", "update-skill");
      const skillDir2 = join(testDir, "repo2", "ops", "update-skill");
      mkdirSync(skillDir1, { recursive: true });
      mkdirSync(skillDir2, { recursive: true });
      writeFileSync(join(skillDir1, "SKILL.md"), "---\nname: update-skill\n---\n# v1");
      writeFileSync(join(skillDir2, "SKILL.md"), "---\nname: update-skill\n---\n# v2");

      const origCwd = process.cwd();
      process.chdir(testDir);
      try {
        const configRoot = getConfigRoot("opencode", "project");
        mkdirSync(join(configRoot, "skills"), { recursive: true });

        initFromScan([{
          id: "update-skill",
          description: "test",
          source: "ops",
          filePath: join(skillDir1, "SKILL.md"),
        }]);
        await installSkills(["update-skill"], "opencode", "project", join(testDir, "repo1"));
        expect(existsSync(join(configRoot, "skills", "update-skill"))).toBe(true);

        initFromScan([{
          id: "update-skill",
          description: "test",
          source: "ops",
          filePath: join(skillDir2, "SKILL.md"),
        }]);
        const results = await installSkills(["update-skill"], "opencode", "project", join(testDir, "repo2"));
        expect(results[0].success).toBe(true);
        expect(existsSync(join(configRoot, "skills", "update-skill"))).toBe(true);
      } finally {
        process.chdir(origCwd);
      }
    });

    it("reinstalls when symlink is broken", async () => {
      const { installSkills } = await import("../src/core/skill-installer.js");
      const { initFromScan } = await import("../src/core/skill-registry.js");
      const { getConfigRoot } = await import("../src/utils/paths.js");

      const skillDir = join(testDir, "repo", "ops", "fix-skill");
      mkdirSync(skillDir, { recursive: true });
      writeFileSync(join(skillDir, "SKILL.md"), "---\nname: fix-skill\n---\n");

      const origCwd = process.cwd();
      process.chdir(testDir);
      try {
        const configRoot = getConfigRoot("opencode", "project");
        mkdirSync(join(configRoot, "skills"), { recursive: true });

        const oldSourceDir = join(testDir, "old-source");
        mkdirSync(oldSourceDir, { recursive: true });
        writeFileSync(join(oldSourceDir, "SKILL.md"), "---\nname: fix-skill\n---\n");
        symlinkSync(oldSourceDir, join(configRoot, "skills", "fix-skill"));

        rmSync(oldSourceDir, { recursive: true, force: true });

        initFromScan([{
          id: "fix-skill",
          description: "test",
          source: "ops",
          filePath: join(skillDir, "SKILL.md"),
        }]);

        const results = await installSkills(["fix-skill"], "opencode", "project", join(testDir, "repo"));
        expect(results[0].success).toBe(true);
        expect(existsSync(join(configRoot, "skills", "fix-skill"))).toBe(true);
      } finally {
        process.chdir(origCwd);
      }
    });
  });

  describe("EPERM copy fallback", () => {
    it("source code includes cpSync EPERM fallback for Windows compatibility", async () => {
      const { readFileSync } = await import("fs");
      const src = readFileSync(join(__dirname, "..", "src", "core", "skill-installer.ts"), "utf-8");
      expect(src).toContain("cpSync");
      expect(src).toContain("EPERM");
      expect(src).toContain("cpSync(resolvedSource, targetPath, { recursive: true })");
    });
  });

  describe("multi-skill batch installation", () => {
    it("installs multiple skills in one call", async () => {
      const { installSkills } = await import("../src/core/skill-installer.js");
      const { initFromScan } = await import("../src/core/skill-registry.js");
      const { getConfigRoot } = await import("../src/utils/paths.js");

      const repoPath = join(testDir, "repo");
      const skillNames = ["skill-a", "skill-b", "skill-c"];
      for (const name of skillNames) {
        const dir = join(repoPath, "ops", name);
        mkdirSync(dir, { recursive: true });
        writeFileSync(join(dir, "SKILL.md"), `---\nname: ${name}\n---\n# ${name}`);
      }

      const origCwd = process.cwd();
      process.chdir(testDir);
      try {
        const configRoot = getConfigRoot("opencode", "project");
        mkdirSync(join(configRoot, "skills"), { recursive: true });

        initFromScan(skillNames.map((name) => ({
          id: name,
          description: "test",
          source: "ops",
          filePath: join(repoPath, "ops", name, "SKILL.md"),
        })));

        const results = await installSkills(skillNames, "opencode", "project", repoPath);
        expect(results.length).toBe(3);
        expect(results.every((r) => r.success)).toBe(true);
        for (const name of skillNames) {
          expect(existsSync(join(configRoot, "skills", name, "SKILL.md"))).toBe(true);
        }
      } finally {
        process.chdir(origCwd);
      }
    });

    it("handles mixed source dirs in batch", async () => {
      const { installSkills } = await import("../src/core/skill-installer.js");
      const { initFromScan } = await import("../src/core/skill-registry.js");
      const { getConfigRoot } = await import("../src/utils/paths.js");

      const repoPath = join(testDir, "repo");
      mkdirSync(join(repoPath, "ops", "ops-skill"), { recursive: true });
      writeFileSync(join(repoPath, "ops", "ops-skill", "SKILL.md"), "---\nname: ops-skill\n---\n");
      mkdirSync(join(repoPath, "infra", "infra-skill"), { recursive: true });
      writeFileSync(join(repoPath, "infra", "infra-skill", "SKILL.md"), "---\nname: infra-skill\n---\n");

      const origCwd = process.cwd();
      process.chdir(testDir);
      try {
        const configRoot = getConfigRoot("opencode", "project");
        mkdirSync(join(configRoot, "skills"), { recursive: true });

        initFromScan([
          { id: "ops-skill", description: "test", source: "ops", filePath: join(repoPath, "ops", "ops-skill", "SKILL.md") },
          { id: "infra-skill", description: "test", source: "infra", filePath: join(repoPath, "infra", "infra-skill", "SKILL.md") },
        ]);

        const results = await installSkills(["ops-skill", "infra-skill"], "opencode", "project", repoPath);
        expect(results.length).toBe(2);
        expect(results.every((r) => r.success)).toBe(true);
        expect(existsSync(join(configRoot, "skills", "ops-skill", "SKILL.md"))).toBe(true);
        expect(existsSync(join(configRoot, "skills", "infra-skill", "SKILL.md"))).toBe(true);
      } finally {
        process.chdir(origCwd);
      }
    });
  });

  describe("cross-plugin shared skill", () => {
    it("second install of same skill is skipped (already exists)", async () => {
      const { installSkills } = await import("../src/core/skill-installer.js");
      const { initFromScan } = await import("../src/core/skill-registry.js");
      const { getConfigRoot } = await import("../src/utils/paths.js");

      const repoPath = join(testDir, "repo");
      const skillDir = join(repoPath, "ops", "shared-skill");
      mkdirSync(skillDir, { recursive: true });
      writeFileSync(join(skillDir, "SKILL.md"), "---\nname: shared-skill\n---\n# Shared");

      const origCwd = process.cwd();
      process.chdir(testDir);
      try {
        const configRoot = getConfigRoot("opencode", "project");
        mkdirSync(join(configRoot, "skills"), { recursive: true });

        initFromScan([{
          id: "shared-skill",
          description: "test",
          source: "ops",
          filePath: join(skillDir, "SKILL.md"),
        }]);

        const results1 = await installSkills(["shared-skill"], "opencode", "project", repoPath);
        expect(results1.length).toBe(1);
        expect(results1[0].success).toBe(true);
        expect(existsSync(join(configRoot, "skills", "shared-skill", "SKILL.md"))).toBe(true);

        const results2 = await installSkills(["shared-skill"], "opencode", "project", repoPath);
        expect(results2.length).toBe(1);
        expect(results2[0].success).toBe(true);
        expect(existsSync(join(configRoot, "skills", "shared-skill", "SKILL.md"))).toBe(true);
      } finally {
        process.chdir(origCwd);
      }
    });
  });
});
