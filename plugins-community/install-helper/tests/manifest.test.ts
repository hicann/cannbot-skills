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
import { mkdirSync, writeFileSync, rmSync, existsSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

let testDir: string;

function setup() {
  testDir = join(tmpdir(), `ih-mf-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
}

function teardown() {
  rmSync(testDir, { recursive: true, force: true });
}

describe("manifest", () => {
  beforeEach(setup);
  afterEach(teardown);

  describe("readAllManifests", () => {
    it("reads legacy cannbot-manifest.json", async () => {
      const { readAllManifests } = await import("../src/core/manifest.js");
      const { getConfigRoot, getManifestPath } = await import("../src/utils/paths.js");

      const origCwd = process.cwd();
      process.chdir(testDir);
      try {
        const configRoot = getConfigRoot("opencode", "project");
        mkdirSync(configRoot, { recursive: true });

        const manifestPath = getManifestPath(configRoot);
        writeFileSync(manifestPath, JSON.stringify({
          brand: "CANNBot",
          version: "1.0.0",
          team: "test-plugin",
          level: "project",
          tool: "opencode",
          installed_skills: ["test-skill"],
          installed_agents: ["test-agent.md"],
          brand_dir: configRoot,
          install_time: "",
        }));

        const manifests = readAllManifests(configRoot);
        expect(manifests.length).toBeGreaterThanOrEqual(1);
        expect(manifests.some((m) => m.team === "test-plugin")).toBe(true);
      } finally {
        process.chdir(origCwd);
      }
    });

    it("reads per-plugin manifest files", async () => {
      const { readAllManifests } = await import("../src/core/manifest.js");
      const { getAllPlugins } = await import("../src/core/registry.js");
      const { getConfigRoot } = await import("../src/utils/paths.js");

      const origCwd = process.cwd();
      process.chdir(testDir);
      try {
        const configRoot = getConfigRoot("opencode", "project");
        mkdirSync(configRoot, { recursive: true });

        const plugins = getAllPlugins();
        if (plugins.length === 0) return;

        const testPlugin = plugins[0];
        writeFileSync(join(configRoot, `${testPlugin.id}-manifest.json`), JSON.stringify({
          brand: "CANNBot",
          version: "1.0.0",
          team: testPlugin.id,
          level: "project",
          tool: "opencode",
          installed_skills: [],
          installed_agents: [],
          brand_dir: configRoot,
          install_time: "",
        }));

        const manifests = readAllManifests(configRoot);
        expect(manifests.some((m) => m.team === testPlugin.id)).toBe(true);
      } finally {
        process.chdir(origCwd);
      }
    });
  });

  describe("writeCannbotManifest", () => {
    it("writes per-plugin manifest file", async () => {
      const { writeCannbotManifest } = await import("../src/core/plugin-installer.js");
      const { getConfigRoot } = await import("../src/utils/paths.js");

      const origCwd = process.cwd();
      process.chdir(testDir);
      try {
        const configRoot = getConfigRoot("opencode", "project");
        mkdirSync(configRoot, { recursive: true });

        const manifest = writeCannbotManifest(configRoot, "test-plugin", "1.0.0", "project", "opencode", ["skill-a"], ["agent-a.md"]);
        expect(manifest.team).toBe("test-plugin");

        const perPluginPath = join(configRoot, "test-plugin-manifest.json");
        expect(existsSync(perPluginPath)).toBe(true);
      } finally {
        process.chdir(origCwd);
      }
    });

    it("does not overwrite other plugin manifests", async () => {
      const { writeCannbotManifest } = await import("../src/core/plugin-installer.js");
      const { getConfigRoot } = await import("../src/utils/paths.js");

      const origCwd = process.cwd();
      process.chdir(testDir);
      try {
        const configRoot = getConfigRoot("opencode", "project");
        mkdirSync(configRoot, { recursive: true });

        writeCannbotManifest(configRoot, "plugin-a", "1.0.0", "project", "opencode", ["skill-a"], []);
        writeCannbotManifest(configRoot, "plugin-b", "1.0.0", "project", "opencode", ["skill-b"], []);

        expect(existsSync(join(configRoot, "plugin-a-manifest.json"))).toBe(true);
        expect(existsSync(join(configRoot, "plugin-b-manifest.json"))).toBe(true);
      } finally {
        process.chdir(origCwd);
      }
    });
  });
});
