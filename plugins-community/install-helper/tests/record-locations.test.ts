// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

// Regression tests for location-keyed install records and the E2E-found
// display/uninstall fixes:
//  - records: one file per plugin holding one record per install location
//    (tool x level x installPath) so an install elsewhere never clobbers the
//    bookkeeping of an earlier location (uninstall would delete the wrong
//    project's files otherwise)
//  - uninstall must act on the record's own configRoot (not cwd-derived)
//  - install summary must resolve plugin entries source-scoped (--source
//    skills docs hint must not point at the cannbot package)

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { mkdirSync, writeFileSync, rmSync, readFileSync } from "fs";
import { join, dirname, resolve } from "path";
import { tmpdir } from "os";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let testDir: string;
let savedHome: string | undefined;
let savedCwd: string;

function setup() {
  testDir = join(tmpdir(), `ih-loc-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
  savedHome = process.env.HOME;
  savedCwd = process.cwd();
  process.env.HOME = testDir;
}

async function teardown() {
  const { setCannbotPlugins } = await import("../src/core/registry.js");
  setCannbotPlugins([]);
  process.chdir(savedCwd);
  process.env.HOME = savedHome;
  rmSync(testDir, { recursive: true, force: true });
}

function makeRecord(overrides: Partial<Record<string, unknown>> & { pluginId: string; installPath: string }) {
  return {
    displayName: overrides.pluginId,
    tool: "opencode" as const,
    level: "project" as const,
    configRoot: join(overrides.installPath, ".opencode"),
    installTime: new Date().toISOString(),
    files: [],
    directories: [],
    ...overrides,
  };
}

describe("location-keyed install records", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("keeps records for multiple locations of the same plugin", async () => {
    const { writeRecord, readRecords } = await import("../src/core/record.js");
    const projectA = join(testDir, "project-a");
    const projectB = join(testDir, "project-b");

    writeRecord(makeRecord({ pluginId: "multi-plugin", installPath: projectA }) as any);
    writeRecord(makeRecord({ pluginId: "multi-plugin", installPath: projectB }) as any);

    const all = readRecords("multi-plugin");
    expect(all).toHaveLength(2);
    expect(all.map((r) => r.installPath).sort()).toEqual([projectA, projectB].sort());
  });

  it("readRecord prefers the record matching the current working directory", async () => {
    const { writeRecord, readRecord } = await import("../src/core/record.js");
    const projectA = join(testDir, "project-a");
    const projectB = join(testDir, "project-b");
    mkdirSync(projectB, { recursive: true });

    writeRecord(makeRecord({ pluginId: "cwd-plugin", installPath: projectA, displayName: "A" }) as any);
    writeRecord(makeRecord({ pluginId: "cwd-plugin", installPath: projectB, displayName: "B" }) as any);

    process.chdir(projectB);
    const record = readRecord("cwd-plugin");
    expect(record?.installPath).toBe(projectB);
  });

  it("readRecord falls back to the global record, then the latest record", async () => {
    const { writeRecord, readRecord } = await import("../src/core/record.js");

    writeRecord(makeRecord({
      pluginId: "fallback-plugin",
      installPath: join(testDir, "project-x"),
      displayName: "older-project",
      installTime: "2026-01-01T00:00:00.000Z",
    }) as any);
    writeRecord(makeRecord({
      pluginId: "fallback-plugin",
      installPath: testDir,
      level: "global",
      displayName: "global",
      installTime: "2026-01-02T00:00:00.000Z",
    }) as any);

    // cwd is a temp dir unrelated to any record: global wins over the older project record
    const record = readRecord("fallback-plugin");
    expect(record?.level).toBe("global");
  });

  it("deleteRecord removes only the matching location and prunes the empty file", async () => {
    const { writeRecord, deleteRecord, readRecords, getRecordPath } = await import("../src/core/record.js");
    const { existsSync } = await import("fs");
    const projectA = join(testDir, "project-a");
    const projectB = join(testDir, "project-b");

    writeRecord(makeRecord({ pluginId: "del-loc-plugin", installPath: projectA }) as any);
    writeRecord(makeRecord({ pluginId: "del-loc-plugin", installPath: projectB }) as any);

    deleteRecord("del-loc-plugin", { tool: "opencode", level: "project", installPath: projectA });
    const remaining = readRecords("del-loc-plugin");
    expect(remaining).toHaveLength(1);
    expect(remaining[0].installPath).toBe(projectB);

    deleteRecord("del-loc-plugin", { tool: "opencode", level: "project", installPath: projectB });
    expect(readRecords("del-loc-plugin")).toHaveLength(0);
    expect(existsSync(getRecordPath("del-loc-plugin"))).toBe(false);
  });

  it("migrates the v1 single-record file format on read", async () => {
    const { getRecordPath, readRecord } = await import("../src/core/record.js");
    const installsDir = join(getRecordPath("dummy"), "..");
    mkdirSync(installsDir, { recursive: true });
    const v1 = {
      pluginId: "v1-plugin",
      displayName: "V1",
      tool: "opencode",
      level: "project",
      installPath: join(testDir, "v1-project"),
      configRoot: join(testDir, "v1-project", ".opencode"),
      installTime: "2026-01-01T00:00:00.000Z",
      files: [join(testDir, "v1-project", "file1")],
      directories: [],
    };
    writeFileSync(getRecordPath("v1-plugin"), JSON.stringify(v1), "utf-8");

    const read = readRecord("v1-plugin");
    expect(read).not.toBeNull();
    expect(read!.pluginId).toBe("v1-plugin");
    expect(read!.files).toEqual(v1.files);
  });
});

describe("install summary source scoping (display layer)", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("printEnhancedSummary points --source skills docs at the legacy entry", async () => {
    const { setCannbotPlugins } = await import("../src/core/registry.js");
    const cannbotPkgDir = join(testDir, "cannbot-pkg", "dist", "plugins", "ops-direct-invoke");
    mkdirSync(cannbotPkgDir, { recursive: true });
    setCannbotPlugins([
      {
        id: "ops-direct-invoke",
        dir: cannbotPkgDir,
        displayName: "ops-direct-invoke",
        script: "cannbot",
        aliases: [],
        skills: 0,
        agents: 0,
        description: "",
        source: "cannbot",
      },
    ]);

    const { printEnhancedSummary } = await import("../src/ui/display.js");
    const logs: string[] = [];
    const spy = vi.spyOn(console, "log").mockImplementation((...args: unknown[]) => {
      logs.push(args.map(String).join(" "));
    });
    try {
      printEnhancedSummary(
        [
          {
            pluginId: "ops-direct-invoke",
            displayName: "AscendC Kernel 直调",
            success: true,
            skillsCount: 45,
            agentsCount: 6,
            source: "skills",
          },
        ],
        "opencode",
        join(testDir, "target", ".opencode")
      );
    } finally {
      spy.mockRestore();
    }

    const output = logs.join("\n");
    expect(output).toContain("plugins-official/ops-direct-invoke/quickstart.md");
    expect(output).not.toContain(cannbotPkgDir);
  });

  it("printEnhancedSummary keeps cannbot-sourced docs pointing at the cannbot package", async () => {
    const { setCannbotPlugins } = await import("../src/core/registry.js");
    const cannbotPkgDir = join(testDir, "cannbot-pkg", "dist", "plugins", "ops-direct-invoke");
    mkdirSync(cannbotPkgDir, { recursive: true });
    setCannbotPlugins([
      {
        id: "ops-direct-invoke",
        dir: cannbotPkgDir,
        displayName: "ops-direct-invoke",
        script: "cannbot",
        aliases: [],
        skills: 0,
        agents: 0,
        description: "",
        source: "cannbot",
      },
    ]);

    const { printEnhancedSummary } = await import("../src/ui/display.js");
    const logs: string[] = [];
    const spy = vi.spyOn(console, "log").mockImplementation((...args: unknown[]) => {
      logs.push(args.map(String).join(" "));
    });
    try {
      printEnhancedSummary(
        [
          {
            pluginId: "ops-direct-invoke",
            displayName: "ops-direct-invoke",
            success: true,
            skillsCount: 31,
            agentsCount: 3,
            source: "cannbot",
          },
        ],
        "opencode",
        join(testDir, "target", ".opencode")
      );
    } finally {
      spy.mockRestore();
    }

    const output = logs.join("\n");
    expect(output).toContain(cannbotPkgDir);
  });
});

describe("uninstall / exit-code wiring (source verification)", () => {
  it("uninstall acts on the record's own configRoot and deletes only that location", async () => {
    const src = readFileSync(join(__dirname, "..", "src", "commands", "uninstall.ts"), "utf-8");
    expect(src).toContain("const configRoot = record.configRoot");
    expect(src).toContain("deleteRecord(plugin.id, {");
  });

  it("install command sets a non-zero exit code on failures", async () => {
    const src = readFileSync(join(__dirname, "..", "src", "commands", "install.ts"), "utf-8");
    expect(src).toContain("process.exitCode = 1");
    // the summary must carry the command-line source (user intent): legacy
    // entries have no `source` field, so plugin?.source alone would lose the
    // --source skills scope at the display layer
    expect(src).toContain("source: source ?? plugin?.source");
  });
});
