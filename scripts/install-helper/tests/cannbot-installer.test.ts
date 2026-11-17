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
import { mkdirSync, writeFileSync, existsSync, rmSync, readdirSync, readFileSync } from "fs";
import { join, dirname } from "path";
import { tmpdir } from "os";
import { fileURLToPath } from "url";
import { execaSync } from "execa";
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
  for (const key of ["CANNBOT_INSTALLER_PATH", "CANNBOT_INSTALLER_CHANNEL", "CANNBOT_MAINLINE_URL"]) {
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

  it("getCannbotRepoUrl honors CANNBOT_MAINLINE_URL for firewalled HTTPS environments", async () => {
    const { getCannbotRepoUrl } = await import("../src/core/cannbot-installer.js");
    // default: no override → canonical mainline URL
    expect(getCannbotRepoUrl()).toBe("https://gitcode.com/cann/cannbot.git");
    // firewall blocks HTTPS git but SSH works → SSH override wins, read at
    // call time (not captured at module load)
    process.env.CANNBOT_MAINLINE_URL = "git@gitcode.com:cann/cannbot.git";
    try {
      expect(getCannbotRepoUrl()).toBe("git@gitcode.com:cann/cannbot.git");
    } finally {
      delete process.env.CANNBOT_MAINLINE_URL;
    }
    // cleared again → back to default
    expect(getCannbotRepoUrl()).toBe("https://gitcode.com/cann/cannbot.git");
  });

  it("cleanStaleInstallerTempDirs removes npm's leftover rename targets", async () => {
    // npm's package replacement renames the old dir to ".<name>-r<random>"
    // first; a crashed refresh leaves that copy behind and every later
    // refresh dies with ENOTEMPTY (deterministic, not network) — the
    // cleaner removes exactly those, keeping the real package and
    // unrelated dot-entries intact.
    const { cleanStaleInstallerTempDirs } = await import("../src/core/cannbot-installer.js");
    const scoped = join(testDir, ".cannbot", "installers", "cannbot", "node_modules", "@cannbot-plugin");
    mkdirSync(join(scoped, "cannbot", "bin"), { recursive: true });
    writeFileSync(join(scoped, "cannbot", "bin", "cannbot.js"), "// real\n");
    for (const stale of [".cannbot-rEynCl1b", ".cannbot-r0000002"]) {
      mkdirSync(join(scoped, stale, "bin"), { recursive: true });
      writeFileSync(join(scoped, stale, "bin", "cannbot.js"), "// stale copy\n");
    }
    mkdirSync(join(scoped, ".unrelated"), { recursive: true });

    cleanStaleInstallerTempDirs("@cannbot-plugin/cannbot", join(testDir, ".cannbot", "installers", "cannbot"));

    const entries = readdirSync(scoped).sort();
    expect(entries).toEqual([".unrelated", "cannbot"]);
    expect(readFileSync(join(scoped, "cannbot", "bin", "cannbot.js"), "utf-8")).toBe("// real\n");
  });

  it("npmInstallInstaller cleans temp dirs up front and retries once on ENOTEMPTY (source code verification)", async () => {
    const src = readFileSync(join(__dirname, "..", "src", "core", "cannbot-installer.ts"), "utf-8");
    expect(src).toMatch(/cleanStaleInstallerTempDirs\(.*\);/);
    expect(src).toMatch(/ENOTEMPTY/);
    const cleanIndex = src.indexOf("cleanStaleInstallerTempDirs");
    const loopIndex = src.indexOf("for (const registry of NPM_MIRRORS)");
    expect(loopIndex).toBeGreaterThan(cleanIndex);
  });

  it("TTL refresh re-points the cached origin at CANNBOT_MAINLINE_URL (firewall after HTTPS cache)", async () => {
    // Scenario from PR #1671 review: cache cloned via one URL, override set
    // later — the stale-cache refresh must honor the override, not keep
    // pulling the origin baked in at clone time. Real local git, no network.
    const git = (cwd: string, args: string[]) =>
      execaSync("git", ["-c", "user.name=test", "-c", "user.email=test@test", ...args], { cwd });

    // origin-a: first source of the cache
    const originA = join(testDir, "origin-a");
    mkdirSync(originA, { recursive: true });
    git(originA, ["init", "--quiet"]);
    mkdirSync(join(originA, "plugins-official"), { recursive: true });
    writeFileSync(join(originA, "plugins-official", "marker-a.txt"), "a");
    git(originA, ["add", "-A"]);
    git(originA, ["commit", "--quiet", "-m", "a"]);

    // origin-b: same history + one newer commit (shares history so pull works)
    const originB = join(testDir, "origin-b");
    git(testDir, ["clone", "--quiet", originA, originB]);
    writeFileSync(join(originB, "plugins-official", "marker-b.txt"), "b");
    git(originB, ["add", "-A"]);
    git(originB, ["commit", "--quiet", "-m", "b"]);

    // drop the seeded fake cache so our real clone path runs from origin-a
    rmSync(join(testDir, ".cannbot", "repo-cannbot"), { recursive: true, force: true });
    process.env.CANNBOT_MAINLINE_URL = originA;
    const { ensureCannbotRepoCache } = await import("../src/core/cannbot-installer.js");
    const cachePath = await ensureCannbotRepoCache();
    expect(cachePath).toBeTruthy();
    expect(existsSync(join(cachePath!, "plugins-official", "marker-a.txt"))).toBe(true);

    // age the stamp past the TTL → next call takes the refresh branch
    writeFileSync(join(cachePath!, ".cannbot-repo-stamp"), "0");

    // firewall move: switch the override to origin-b and refresh
    process.env.CANNBOT_MAINLINE_URL = originB;
    try {
      const cachePath2 = await ensureCannbotRepoCache();
      expect(cachePath2).toBe(cachePath);
      const remoteUrl = execaSync("git", ["remote", "get-url", "origin"], { cwd: cachePath! }).stdout.trim();
      expect(remoteUrl).toBe(originB);
      expect(existsSync(join(cachePath!, "plugins-official", "marker-b.txt"))).toBe(true);
    } finally {
      delete process.env.CANNBOT_MAINLINE_URL;
    }
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
