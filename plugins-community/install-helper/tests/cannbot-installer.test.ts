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
import { mkdirSync, writeFileSync, existsSync, rmSync } from "fs";
import { join, dirname } from "path";
import { tmpdir } from "os";
import { fileURLToPath } from "url";
import { createFakeInstaller } from "./helpers/fake-cannbot.js";
import { seedMainlineCache } from "./helpers/seed-mainline-cache.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let testDir: string;
let savedEnv: Record<string, string | undefined>;

function setup() {
  testDir = join(tmpdir(), `ih-cbi-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(testDir, { recursive: true });
  savedEnv = { HOME: process.env.HOME };
  // Sandbox HOME: the real home may carry a managed installer cache
  // (~/.cannbot/installers/cannbot) from interactive use, which must not
  // leak into the "no installer available" assertions below.
  process.env.HOME = testDir;
  for (const key of ["CANNBOT_INSTALLER_PATH", "CANNBOT_INSTALLER_CHANNEL"]) {
    savedEnv[key] = process.env[key];
    delete process.env[key];
  }
  // pre-seed a fresh mainline cache: discovery must not attempt a real
  // git clone from unit tests
  seedMainlineCache(testDir);
}

function teardown() {
  rmSync(testDir, { recursive: true, force: true });
  for (const [key, value] of Object.entries(savedEnv)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
}

describe("cannbot-installer", () => {
  beforeEach(setup);
  afterEach(teardown);

  it("loadCannbotInstallerConfig reads repository.yaml defaults", async () => {
    const { loadCannbotInstallerConfig } = await import("../src/core/cannbot-installer.js");
    const config = loadCannbotInstallerConfig();
    expect(config.package).toBe("@cannbot-plugin/cannbot");
    expect(config.channel).toBe("latest");
    expect(config.ttlHours).toBeGreaterThan(0);
  });

  it("CANNBOT_INSTALLER_CHANNEL overrides the channel", async () => {
    process.env.CANNBOT_INSTALLER_CHANNEL = "beta";
    const { loadCannbotInstallerConfig } = await import("../src/core/cannbot-installer.js");
    expect(loadCannbotInstallerConfig().channel).toBe("beta");
  });

  it("discoverCannbotPlugins returns [] when no installer is available", async () => {
    const { discoverCannbotPlugins, isCannbotInstallerReady } = await import("../src/core/cannbot-installer.js");
    expect(isCannbotInstallerReady()).toBe(false);
    expect(discoverCannbotPlugins()).toEqual([]);
  });

  it("discovers plugins from a CANNBOT_INSTALLER_PATH layout", async () => {
    createFakeInstaller(testDir, "fake-cb-plugin");
    process.env.CANNBOT_INSTALLER_PATH = testDir;

    const { discoverCannbotPlugins, isCannbotInstallerReady, getCannbotInstallerVersion } = await import(
      "../src/core/cannbot-installer.js"
    );

    expect(isCannbotInstallerReady()).toBe(true);
    expect(getCannbotInstallerVersion()).toBe("9.9.9");

    const plugins = discoverCannbotPlugins();
    expect(plugins).toHaveLength(1);
    expect(plugins[0].id).toBe("fake-cb-plugin");
    expect(plugins[0].version).toBe("3.1.0");
    expect(plugins[0].skillsCount).toBe(2);
    expect(plugins[0].agentsCount).toBe(1);
    expect(plugins[0].description).toContain("fake cannbot plugin");
  });

  it("discovers plugins from the post-rename plugins-official checkout layout", async () => {
    // cannbot mainline renamed plugins/ -> plugins-official/ (e140ca6):
    // discovery must also probe the new layout when CANNBOT_INSTALLER_PATH
    // points at a repository checkout without a dist/ assembly.
    const installerRoot = join(testDir, "checkout");
    createFakeInstaller(installerRoot, "fake-cb-plugin");
    // move dist/plugins -> plugins-official to simulate the renamed checkout
    const { cpSync, rmSync, mkdirSync: mk } = await import("fs");
    mk(join(installerRoot, "plugins-official"), { recursive: true });
    cpSync(
      join(installerRoot, "dist", "plugins", "fake-cb-plugin"),
      join(installerRoot, "plugins-official", "fake-cb-plugin"),
      { recursive: true }
    );
    rmSync(join(installerRoot, "dist"), { recursive: true, force: true });
    // bin must exist for readiness
    mk(join(installerRoot, "bin"), { recursive: true });
    writeFileSync(join(installerRoot, "bin", "cannbot.js"), "#!/usr/bin/env node\n");
    process.env.CANNBOT_INSTALLER_PATH = installerRoot;

    const { discoverCannbotPlugins } = await import("../src/core/cannbot-installer.js");
    const plugins = discoverCannbotPlugins();
    expect(plugins).toHaveLength(1);
    expect(plugins[0].id).toBe("fake-cb-plugin");
  });

  it("discovers plugins from the plugins-community checkout layout", async () => {
    // every plugin the cannbot installer ships is installable — community
    // layouts (checkout or dist) must be probed just like official ones
    const installerRoot = join(testDir, "community-checkout");
    createFakeInstaller(installerRoot, "fake-cb-plugin");
    const { cpSync, rmSync, mkdirSync: mk } = await import("fs");
    mk(join(installerRoot, "plugins-community"), { recursive: true });
    cpSync(
      join(installerRoot, "dist", "plugins", "fake-cb-plugin"),
      join(installerRoot, "plugins-community", "fake-cb-plugin"),
      { recursive: true }
    );
    rmSync(join(installerRoot, "dist"), { recursive: true, force: true });
    mk(join(installerRoot, "bin"), { recursive: true });
    writeFileSync(join(installerRoot, "bin", "cannbot.js"), "#!/usr/bin/env node\n");
    process.env.CANNBOT_INSTALLER_PATH = installerRoot;

    const { discoverCannbotPlugins } = await import("../src/core/cannbot-installer.js");
    const plugins = discoverCannbotPlugins();
    expect(plugins).toHaveLength(1);
    expect(plugins[0].id).toBe("fake-cb-plugin");
  });

  it("ensureCannbotInstaller trusts a valid CANNBOT_INSTALLER_PATH without npm", async () => {
    createFakeInstaller(testDir, "fake-cb-plugin");
    process.env.CANNBOT_INSTALLER_PATH = testDir;

    const { ensureCannbotInstaller } = await import("../src/core/cannbot-installer.js");
    const root = await ensureCannbotInstaller();
    expect(root).toBe(testDir);
  });

  it("ensureCannbotInstaller rejects an invalid CANNBOT_INSTALLER_PATH", async () => {
    process.env.CANNBOT_INSTALLER_PATH = join(testDir, "does-not-exist");
    const { ensureCannbotInstaller } = await import("../src/core/cannbot-installer.js");
    await expect(ensureCannbotInstaller()).rejects.toThrow();
  });

  it("ensureCannbotDiscovery merges cannbot entries into the registry with cannbot-first precedence", async () => {
    createFakeInstaller(testDir, "fake-cb-plugin");
    process.env.CANNBOT_INSTALLER_PATH = testDir;

    const { ensureCannbotDiscovery } = await import("../src/core/cannbot-installer.js");
    const { findPlugin, getAllPlugins, getPluginById } = await import("../src/core/registry.js");

    const ok = await ensureCannbotDiscovery();
    expect(ok).toBe(true);

    const merged = getAllPlugins();
    expect(merged.some((p) => p.id === "fake-cb-plugin" && p.source === "cannbot")).toBe(true);

    const byId = getPluginById("fake-cb-plugin");
    expect(byId?.source).toBe("cannbot");
    expect(byId?.skills).toBe(2);

    // findPlugin resolves the cannbot entry (fast path without legacy lookup)
    const found = findPlugin("fake-cb-plugin");
    expect(found?.source).toBe("cannbot");
  });

  it("discovery carries the manifest displayName and falls back to the plugin id", async () => {
    createFakeInstaller(testDir, "fake-cb-plugin");
    createFakeInstaller(testDir, "named-cb-plugin", { displayName: "PyPTO-Pro 算子开发" });
    process.env.CANNBOT_INSTALLER_PATH = testDir;

    const { ensureCannbotDiscovery } = await import("../src/core/cannbot-installer.js");
    const { getAllPlugins } = await import("../src/core/registry.js");

    await ensureCannbotDiscovery();

    const merged = getAllPlugins();
    // manifest-declared display name flows straight through (self-adapting:
    // upstream adding displayName to plugin.json needs no install-helper
    // release — same principle as the delegated tool-set derivation)
    const named = merged.find((p) => p.id === "named-cb-plugin");
    expect(named?.displayName).toBe("PyPTO-Pro 算子开发");
    // manifests without the field keep the id fallback
    const unnamed = merged.find((p) => p.id === "fake-cb-plugin");
    expect(unnamed?.displayName).toBe("fake-cb-plugin");
  });

  it("findPlugin resolves same-name legacy ids via --source skills semantics", async () => {
    createFakeInstaller(testDir, "torch-compile");
    process.env.CANNBOT_INSTALLER_PATH = testDir;

    const { ensureCannbotDiscovery } = await import("../src/core/cannbot-installer.js");
    const { findPlugin, setCannbotPlugins } = await import("../src/core/registry.js");

    await ensureCannbotDiscovery();

    // cannbot wins for the same name
    const cannbotFirst = findPlugin("torch-compile");
    expect(cannbotFirst?.source).toBe("cannbot");

    // explicit skills source falls back to the legacy registry entry
    const legacy = findPlugin("torch-compile", { source: "skills" });
    expect(legacy?.source).toBeUndefined();
    expect(legacy?.dir).toContain("plugins-official");

    setCannbotPlugins([]);
  });

  it("ensureCannbotDiscovery degrades to legacy-only when the installer cannot be ensured", async () => {
    // invalid override path -> ensureCannbotInstaller throws -> discovery
    // catches and returns false without touching the registry
    process.env.CANNBOT_INSTALLER_PATH = join(testDir, "not-a-real-installer");

    const { ensureCannbotDiscovery } = await import("../src/core/cannbot-installer.js");
    const { findPlugin, setCannbotPlugins } = await import("../src/core/registry.js");

    setCannbotPlugins([]);
    const ok = await ensureCannbotDiscovery();
    expect(ok).toBe(false);

    const plugin = findPlugin("torch-compile");
    expect(plugin?.source).toBeUndefined();
    expect(plugin?.dir).toContain("plugins-official");
  });

  it("discoverMainlinePlugins reads the cannbot mainline repo layout", async () => {
    const { discoverMainlinePlugins, ensureCannbotRepoCache, getCannbotRepoCachePath, ensureCannbotDiscovery } = await import(
      "../src/core/cannbot-installer.js"
    );
    const { findPlugin, setCannbotPlugins } = await import("../src/core/registry.js");
    const { createFakeInstaller } = await import("./helpers/fake-cannbot.js");

    // Fake a cannbot mainline repo layout under the discovery stamp TTL:
    // the cache clone is skipped by pre-seeding the cache directory.
    const repoPath = getCannbotRepoCachePath();
    const official = join(repoPath, "plugins-official", "mainline-fixture");
    mkdirSync(join(official, ".claude-plugin"), { recursive: true });
    writeFileSync(
      join(official, ".claude-plugin", "plugin.json"),
      JSON.stringify({ name: "mainline-fixture", description: "mainline plugin", version: "9.9.9", skills: [], agents: [] })
    );
    writeFileSync(join(official, "plugin-sources.json"), JSON.stringify({ skills: ["a", "b"] }));
    mkdirSync(join(repoPath, "plugins-community", "community-fixture"), { recursive: true });

    ensureCannbotRepoCache(); // TTL probe only — seeded cache stays as-is
    const entries = discoverMainlinePlugins(repoPath);
    expect(entries.map((e) => e.id)).toContain("mainline-fixture");
    expect(entries.find((e) => e.id === "mainline-fixture")!.version).toBe("9.9.9");
    expect(entries.find((e) => e.id === "mainline-fixture")!.skillsCount).toBe(2);
    // a directory without .claude-plugin/plugin.json is not a plugin
    expect(entries.map((e) => e.id)).not.toContain("community-fixture");

    // and the merged discovery routes the mainline-only plugin to cannbot
    setCannbotPlugins([]);
    process.env.CANNBOT_INSTALLER_PATH = createFakeInstaller(testDir, "fake-cb-plugin");
    const ok = await ensureCannbotDiscovery();
    expect(ok).toBe(true);
    const found = findPlugin("mainline-fixture");
    expect(found?.source).toBe("cannbot");
    expect(found?.version).toBe("9.9.9");
  });
});
