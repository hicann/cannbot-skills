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
import { mkdirSync, writeFileSync, existsSync, rmSync, readFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

let testDir: string;

function setup() {
  testDir = join(tmpdir(), `ih-rec-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
}

function teardown() {
  rmSync(testDir, { recursive: true, force: true });
}

describe("record", () => {
  beforeEach(setup);
  afterEach(teardown);

  describe("writeRecord / readRecord", () => {
    it("writes and reads record correctly", async () => {
      const { writeRecord, readRecord } = await import("../src/core/record.js");
      const record = {
        pluginId: "test-plugin",
        displayName: "Test Plugin",
        tool: "opencode" as const,
        level: "project" as const,
        installPath: testDir,
        configRoot: join(testDir, ".opencode"),
        installTime: new Date().toISOString(),
        files: [join(testDir, "file1"), join(testDir, "file2")],
        directories: [join(testDir, "dir1")],
      };
      writeRecord(record);
      const read = readRecord("test-plugin");
      expect(read).not.toBeNull();
      expect(read!.pluginId).toBe("test-plugin");
      expect(read!.files).toEqual(record.files);
      expect(read!.directories).toEqual(record.directories);
    });

    it("returns null for non-existent record", async () => {
      const { readRecord } = await import("../src/core/record.js");
      const read = readRecord("nonexistent-plugin");
      expect(read).toBeNull();
    });

    it("returns null for corrupted record", async () => {
      const { readRecord, getRecordPath } = await import("../src/core/record.js");
      const recordPath = getRecordPath("corrupt-plugin");
      mkdirSync(join(recordPath, ".."), { recursive: true });
      writeFileSync(recordPath, "{invalid json}", "utf-8");
      const read = readRecord("corrupt-plugin");
      expect(read).toBeNull();
    });
  });

  describe("deleteRecord", () => {
    it("deletes existing record", async () => {
      const { writeRecord, deleteRecord, readRecord } = await import("../src/core/record.js");
      const record = {
        pluginId: "del-plugin",
        displayName: "Del",
        tool: "opencode" as const,
        level: "project" as const,
        installPath: testDir,
        configRoot: testDir,
        installTime: "",
        files: [],
        directories: [],
      };
      writeRecord(record);
      expect(readRecord("del-plugin")).not.toBeNull();
      deleteRecord("del-plugin");
      expect(readRecord("del-plugin")).toBeNull();
    });
  });

  describe("scanInstalledFiles", () => {
    it("records skills and agents from manifest", async () => {
      const { scanInstalledFiles } = await import("../src/core/record.js");
      const configRoot = join(testDir, ".opencode");
      const skillsDir = join(configRoot, "skills");
      const agentsDir = join(configRoot, "agents");
      mkdirSync(skillsDir, { recursive: true });
      mkdirSync(agentsDir, { recursive: true });
      mkdirSync(join(skillsDir, "skill-a"));
      writeFileSync(join(agentsDir, "agent-a.md"), "# Agent");

      const manifest = {
        brand: "CANNBot",
        version: "1.0.0",
        team: "test-plugin",
        level: "project",
        tool: "opencode",
        installed_skills: ["skill-a"],
        installed_agents: ["agent-a"],
        brand_dir: configRoot,
        install_time: "",
      };

      const record = scanInstalledFiles(
        "test-plugin", "Test", "opencode", "project",
        testDir, configRoot, manifest
      );
      expect(record.files.some(f => f.includes("skill-a"))).toBe(true);
      expect(record.files.some(f => f.includes("agent-a.md"))).toBe(true);
      expect(record.directories).toContain(skillsDir);
      expect(record.directories).toContain(agentsDir);
    });

    it("records external repo names dynamically", async () => {
      const { scanInstalledFiles } = await import("../src/core/record.js");
      const configRoot = join(testDir, ".opencode");
      mkdirSync(configRoot, { recursive: true });
      const repoLink = join(testDir, "my-custom-repo");
      const { symlinkSync } = await import("fs");
      symlinkSync(configRoot, repoLink);

      const record = scanInstalledFiles(
        "test-plugin", "Test", "opencode", "project",
        testDir, configRoot, null, ["my-custom-repo"]
      );
      expect(record.files.some(f => f.includes("my-custom-repo"))).toBe(true);
    });

    it("falls back to hardcoded repos when no externalRepoNames", async () => {
      const { scanInstalledFiles } = await import("../src/core/record.js");
      const configRoot = join(testDir, ".opencode");
      mkdirSync(configRoot, { recursive: true });
      const repoLink = join(testDir, "asc-devkit");
      const { symlinkSync } = await import("fs");
      symlinkSync(configRoot, repoLink);

      const record = scanInstalledFiles(
        "test-plugin", "Test", "opencode", "project",
        testDir, configRoot, null
      );
      expect(record.files.some(f => f.includes("asc-devkit"))).toBe(true);
    });
  });

  describe("skill records", () => {
    it("adds and reads skill records", async () => {
      const { addSkillsToRecord, readSkillRecord, getRecordPath } = await import("../src/core/record.js");
      mkdirSync(join(getRecordPath("dummy"), ".."), { recursive: true });
      addSkillsToRecord(["skill-x", "skill-y"], "opencode", "project", testDir);
      const record = readSkillRecord();
      expect(record.opencode.project[testDir].skills).toContain("skill-x");
      expect(record.opencode.project[testDir].skills).toContain("skill-y");
    });

    it("does not duplicate skills on re-add", async () => {
      const { addSkillsToRecord, readSkillRecord, getRecordPath } = await import("../src/core/record.js");
      mkdirSync(join(getRecordPath("dummy"), ".."), { recursive: true });
      addSkillsToRecord(["skill-x"], "opencode", "project", testDir);
      addSkillsToRecord(["skill-x"], "opencode", "project", testDir);
      const record = readSkillRecord();
      const count = record.opencode.project[testDir].skills.filter((s: string) => s === "skill-x").length;
      expect(count).toBe(1);
    });
  });

  describe("batch tracking", () => {
    it("creates a batch entry on add", async () => {
      const { addSkillsToRecord, readSkillRecord, getRecordPath } = await import("../src/core/record.js");
      mkdirSync(join(getRecordPath("dummy"), ".."), { recursive: true });
      addSkillsToRecord(["skill-a", "skill-b"], "opencode", "project", testDir);
      const record = readSkillRecord();
      const entry = record.opencode.project[testDir];
      expect(entry.batches).toBeDefined();
      expect(entry.batches!.length).toBe(1);
      expect(entry.batches![0].skills).toContain("skill-a");
      expect(entry.batches![0].skills).toContain("skill-b");
      expect(entry.batches![0].batchId).toMatch(/^batch-/);
      expect(entry.batches![0].installedAt).toBeTruthy();
    });

    it("creates separate batches for separate install calls", async () => {
      const { addSkillsToRecord, readSkillRecord, getRecordPath } = await import("../src/core/record.js");
      mkdirSync(join(getRecordPath("dummy"), ".."), { recursive: true });
      addSkillsToRecord(["skill-a"], "opencode", "project", testDir);
      addSkillsToRecord(["skill-b"], "opencode", "project", testDir);
      const record = readSkillRecord();
      const entry = record.opencode.project[testDir];
      expect(entry.batches!.length).toBe(2);
      expect(entry.batches![0].skills).toEqual(["skill-a"]);
      expect(entry.batches![1].skills).toEqual(["skill-b"]);
    });

    it("getLastBatchSkills returns last batch skills", async () => {
      const { addSkillsToRecord, getLastBatchSkills, getRecordPath } = await import("../src/core/record.js");
      const { getConfigRoot } = await import("../src/utils/paths.js");
      mkdirSync(join(getRecordPath("dummy"), ".."), { recursive: true });
      const configRoot = getConfigRoot("opencode", "project", testDir);
      const skillsDir = join(configRoot, "skills");
      try {
        mkdirSync(join(skillsDir, "skill-a"), { recursive: true });
        mkdirSync(join(skillsDir, "skill-b"), { recursive: true });
        mkdirSync(join(skillsDir, "skill-c"), { recursive: true });
        addSkillsToRecord(["skill-a"], "opencode", "project", testDir);
        addSkillsToRecord(["skill-b", "skill-c"], "opencode", "project", testDir);
        const lastBatch = getLastBatchSkills("opencode", "project", testDir);
        expect(lastBatch).not.toBeNull();
        expect(lastBatch).toEqual(["skill-b", "skill-c"]);
      } finally {
        rmSync(configRoot, { recursive: true, force: true });
      }
    });

    it("getLastBatchSkills returns null when no batches", async () => {
      const { getLastBatchSkills, getRecordPath } = await import("../src/core/record.js");
      mkdirSync(join(getRecordPath("dummy"), ".."), { recursive: true });
      const lastBatch = getLastBatchSkills("opencode", "project", testDir);
      expect(lastBatch).toBeNull();
    });

    it("removeSkillsFromRecord cleans up batch entries", async () => {
      const { addSkillsToRecord, removeSkillsFromRecord, readSkillRecord, getRecordPath } = await import("../src/core/record.js");
      mkdirSync(join(getRecordPath("dummy"), ".."), { recursive: true });
      addSkillsToRecord(["skill-a", "skill-b"], "opencode", "project", testDir);
      addSkillsToRecord(["skill-c"], "opencode", "project", testDir);
      removeSkillsFromRecord(["skill-a"], "opencode", "project", testDir);
      const record = readSkillRecord();
      const entry = record.opencode.project[testDir];
      expect(entry.skills).not.toContain("skill-a");
      expect(entry.batches![0].skills).not.toContain("skill-a");
      expect(entry.batches![0].skills).toContain("skill-b");
    });

    it("removes empty batch entries after cleanup", async () => {
      const { addSkillsToRecord, removeSkillsFromRecord, readSkillRecord, getRecordPath } = await import("../src/core/record.js");
      mkdirSync(join(getRecordPath("dummy"), ".."), { recursive: true });
      const thisTestDir = join(tmpdir(), `ih-rec-batch-${Date.now()}-${Math.random().toString(36).slice(2)}`);
      mkdirSync(thisTestDir, { recursive: true });
      addSkillsToRecord(["skill-a"], "opencode", "project", thisTestDir);
      addSkillsToRecord(["skill-b"], "opencode", "project", thisTestDir);
      removeSkillsFromRecord(["skill-a"], "opencode", "project", thisTestDir);
      const record = readSkillRecord();
      const entry = record.opencode.project[thisTestDir];
      expect(entry.batches!.length).toBe(1);
      expect(entry.batches![0].skills).toEqual(["skill-b"]);
      rmSync(thisTestDir, { recursive: true, force: true });
    });

    it("backward compatible with old records without batches field", async () => {
      const { readSkillRecord, writeSkillRecord, getLastBatchSkills, getRecordPath } = await import("../src/core/record.js");
      mkdirSync(join(getRecordPath("dummy"), ".."), { recursive: true });
      const oldRecord = {
        opencode: {
          project: {
            [testDir]: {
              skills: ["skill-old"],
              installTime: "2026-01-01T00:00:00.000Z",
            },
          },
        },
      };
      writeSkillRecord(oldRecord);
      const record = readSkillRecord();
      expect(record.opencode.project[testDir].batches).toBeUndefined();
      const lastBatch = getLastBatchSkills("opencode", "project", testDir);
      expect(lastBatch).toBeNull();
    });
  });
});
