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
});

function readFileSync(path: string, encoding: string): string {
  const { readFileSync: rs } = require("fs");
  return rs(path, encoding);
}
