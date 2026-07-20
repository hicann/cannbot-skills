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
import { mkdirSync, writeFileSync, copyFileSync, readdirSync, existsSync, rmSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

let testDir: string;

function setup() {
  testDir = join(tmpdir(), `ih-bk-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
}

function teardown() {
  rmSync(testDir, { recursive: true, force: true });
}

describe("backup", () => {
  beforeEach(setup);
  afterEach(teardown);

  describe("createBackup", () => {
    it("creates backup file when AGENTS.md exists", async () => {
      const { createBackup } = await import("../src/core/backup.js");
      const agentsFile = join(testDir, "AGENTS.md");
      writeFileSync(agentsFile, "# Original config");
      const result = createBackup(testDir, "opencode", "plugin-a", "Plugin A");
      expect(result).not.toBeNull();
      expect(existsSync(result!.filePath)).toBe(true);
      expect(result!.pluginId).toBe("plugin-a");
    });

    it("returns null when AGENTS.md does not exist", async () => {
      const { createBackup } = await import("../src/core/backup.js");
      const result = createBackup(testDir, "opencode", "plugin-a", "Plugin A");
      expect(result).toBeNull();
    });
  });

  describe("findBackups", () => {
    it("finds existing backup files", async () => {
      const { findBackups } = await import("../src/core/backup.js");
      const agentsFile = join(testDir, "AGENTS.md");
      writeFileSync(agentsFile, "# Config");
      const backupFile = join(testDir, "AGENTS.md.cannbot-backup.plugin-a.20260101-120000");
      copyFileSync(agentsFile, backupFile);

      const backups = findBackups(testDir);
      expect(backups.length).toBe(1);
      expect(backups[0].pluginId).toBe("plugin-a");
      expect(backups[0].backupTime).toBe("20260101-120000");
    });

    it("returns empty array when no backups exist", async () => {
      const { findBackups } = await import("../src/core/backup.js");
      const backups = findBackups(testDir);
      expect(backups).toEqual([]);
    });

    it("parses pluginId with dots correctly", async () => {
      const { findBackups } = await import("../src/core/backup.js");
      const agentsFile = join(testDir, "AGENTS.md");
      writeFileSync(agentsFile, "# Config");
      const backupFile = join(testDir, "AGENTS.md.cannbot-backup.plugin.with.dots.20260101-120000");
      copyFileSync(agentsFile, backupFile);

      const backups = findBackups(testDir);
      expect(backups.length).toBe(1);
      expect(backups[0].pluginId).toBe("plugin.with.dots");
    });
  });

  describe("restoreBackup", () => {
    it("restores backup to AGENTS.md", async () => {
      const { restoreBackup } = await import("../src/core/backup.js");
      const agentsFile = join(testDir, "AGENTS.md");
      const backupFile = join(testDir, "backup.bak");
      writeFileSync(backupFile, "# Original");
      writeFileSync(agentsFile, "# Modified");

      const result = restoreBackup(backupFile, testDir, "opencode");
      expect(result).toBe(true);
      expect(readFileSync(agentsFile, "utf-8")).toBe("# Original");
    });

    it("returns false when backup file does not exist", async () => {
      const { restoreBackup } = await import("../src/core/backup.js");
      const result = restoreBackup(join(testDir, "nonexistent.bak"), testDir, "opencode");
      expect(result).toBe(false);
    });
  });

  describe("deleteBackup", () => {
    it("deletes existing backup file", async () => {
      const { deleteBackup } = await import("../src/core/backup.js");
      const backupFile = join(testDir, "backup.bak");
      writeFileSync(backupFile, "# Backup");
      deleteBackup(backupFile);
      expect(existsSync(backupFile)).toBe(false);
    });
  });

  describe("detectCurrentPlugin", () => {
    it("reads per-plugin manifest (not just legacy cannbot-manifest.json)", async () => {
      await import("../src/core/registry.js");
      const { detectCurrentPlugin } = await import("../src/core/backup.js");
      const configRoot = join(testDir, ".opencode");
      mkdirSync(configRoot, { recursive: true });

      writeFileSync(
        join(configRoot, "ops-direct-invoke-manifest.json"),
        JSON.stringify({
          brand: "CANNBot",
          version: "1.0.0",
          team: "ops-direct-invoke",
          level: "project",
          tool: "opencode",
          installed_skills: [],
          installed_agents: [],
          brand_dir: configRoot,
          install_time: "2026-01-01T00:00:00Z",
        })
      );

      const result = detectCurrentPlugin(configRoot, "opencode");
      expect(result).not.toBeNull();
      expect(result!.pluginId).toBe("ops-direct-invoke");
    });

    it("checks installPath for agents file when configRootConfigLink is false", async () => {
      await import("../src/core/registry.js");
      const { detectCurrentPlugin } = await import("../src/core/backup.js");
      const { writeRecord } = await import("../src/core/record.js");
      const installPath = join(testDir, "project-dir");
      const configRoot = join(installPath, ".opencode");
      mkdirSync(configRoot, { recursive: true });

      writeFileSync(join(installPath, "AGENTS.md"), "# Agents content");

      writeRecord({
        pluginId: "ops-direct-invoke",
        displayName: "AscendC Kernel 直调",
        tool: "opencode",
        level: "project",
        installPath,
        configRoot,
        files: [],
        skills: [],
        agents: [],
      } as any);

      const result = detectCurrentPlugin(configRoot, "opencode", installPath);
      expect(result).not.toBeNull();
    });

    it("returns null when no manifest and no agents file in configRoot or installPath", async () => {
      const { detectCurrentPlugin } = await import("../src/core/backup.js");
      const configRoot = join(testDir, ".opencode");
      mkdirSync(configRoot, { recursive: true });

      const result = detectCurrentPlugin(configRoot, "opencode");
      expect(result).toBeNull();
    });

    it("returns plugin with latest install_time when multiple manifests exist", async () => {
      await import("../src/core/registry.js");
      const { detectCurrentPlugin } = await import("../src/core/backup.js");
      const configRoot = join(testDir, ".opencode-multi");
      mkdirSync(configRoot, { recursive: true });

      const manifestA = {
        brand: "CANNBot",
        version: "1.0.0",
        team: "ops-direct-invoke",
        level: "project",
        tool: "opencode",
        installed_skills: [],
        installed_agents: [],
        brand_dir: configRoot,
        install_time: "2026-01-01T00:00:00Z",
      };
      const manifestB = {
        brand: "CANNBot",
        version: "1.0.0",
        team: "torch-compile",
        level: "project",
        tool: "opencode",
        installed_skills: [],
        installed_agents: [],
        brand_dir: configRoot,
        install_time: "2026-07-17T12:00:00Z",
      };

      const { writeFileSync: wf } = require("fs");
      wf(join(configRoot, "ops-direct-invoke-manifest.json"), JSON.stringify(manifestA));
      wf(join(configRoot, "torch-compile-manifest.json"), JSON.stringify(manifestB));

      const result = detectCurrentPlugin(configRoot, "opencode");
      expect(result).not.toBeNull();
      expect(result!.pluginId).toBe("torch-compile");
    });
  });
});

function readFileSync(path: string, encoding: string): string {
  const { readFileSync: rs } = require("fs");
  return rs(path, encoding);
}
