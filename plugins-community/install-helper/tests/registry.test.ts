// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------
import { describe, it, expect } from "vitest";
import { writeFileSync, mkdirSync, existsSync, rmSync, readFileSync } from "fs";
import { join, dirname } from "path";
import { tmpdir } from "os";
import { fileURLToPath } from "url";
import { createFakeInstaller } from "./helpers/fake-cannbot.js";
import { seedMainlineCache } from "./helpers/seed-mainline-cache.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

describe("registry", () => {
  it("getAllPlugins returns plugins from plugins.d", async () => {
    const pluginsDir = join(__dirname, "..", "plugins.d");
    if (!existsSync(pluginsDir)) {
      console.warn("plugins.d not found, skipping registry tests");
      return;
    }
    const { getAllPlugins, findPlugin, getPluginById } = await import("../src/core/registry.js");
    const { MIGRATED_PLUGINS } = await import("../src/core/migrated-plugins.js");
    const plugins = getAllPlugins();
    expect(plugins.length).toBeGreaterThan(0);

    // Structural invariants describe plugins.d entries. Migrated tombstones
    // have no local tree left in this repo and legitimately carry dir: "".
    const migratedIds = new Set(MIGRATED_PLUGINS.map((p) => p.id));
    for (const p of plugins) {
      if (migratedIds.has(p.id)) continue;
      expect(p.id).toBeTruthy();
      expect(p.dir).toBeTruthy();
      expect(p.displayName).toBeTruthy();
      expect(typeof p.script).toBe("string");
      expect(Array.isArray(p.aliases)).toBe(true);
    }

    const sorted = [...plugins].sort((a, b) =>
      a.displayName < b.displayName ? -1 : a.displayName > b.displayName ? 1 : 0
    );
    expect(plugins.map((p) => p.id)).toEqual(sorted.map((p) => p.id));
  });

  it("findPlugin finds by exact id", async () => {
    const { findPlugin } = await import("../src/core/registry.js");
    const plugin = findPlugin("ops-direct-invoke");
    expect(plugin).toBeDefined();
    expect(plugin!.id).toBe("ops-direct-invoke");
    expect(plugin!.source).toBe("cannbot");
  });

  it("findPlugin finds by alias", async () => {
    const { findPlugin } = await import("../src/core/registry.js");
    const plugin = findPlugin("flash");
    expect(plugin).toBeDefined();
    expect(plugin!.id).toBe("ops-direct-invoke-flash");
    expect(plugin!.source).toBe("cannbot");
  });

  it("findPlugin finds by prefix", async () => {
    const { findPlugin } = await import("../src/core/registry.js");
    const plugin = findPlugin("ops-direct-invoke-fl");
    expect(plugin).toBeDefined();
    expect(plugin!.id).toBe("ops-direct-invoke-flash");
    expect(plugin!.source).toBe("cannbot");
  });

  it("findPlugin returns undefined for non-existent", async () => {
    const { findPlugin } = await import("../src/core/registry.js");
    const plugin = findPlugin("non-existent-plugin");
    expect(plugin).toBeUndefined();
  });

  it("getPluginById returns plugin by exact id", async () => {
    const { getPluginById } = await import("../src/core/registry.js");
    const plugin = getPluginById("torch-compile");
    expect(plugin).toBeDefined();
    expect(plugin!.id).toBe("torch-compile");
  });

  it("getPluginById returns undefined for non-existent", async () => {
    const { getPluginById } = await import("../src/core/registry.js");
    const plugin = getPluginById("non-existent");
    expect(plugin).toBeUndefined();
  });

  it("visible tombstone rows show the repository identity, not legacy names", async () => {
    // tombstones are cannbot-delegated rows: the repository id is the
    // identity users map onto the repo (ops-registry-invoke shows as
    // "ops-registry-invoke [cannbot]", not the legacy "AscendC 算子注册调用")
    const { setCannbotPlugins, getAllPlugins, findPlugin } = await import("../src/core/registry.js");
    const { dedupeForWizardSelection } = await import("../src/ui/wizard.js");

    setCannbotPlugins([]); // offline: no discovered cannbot entries
    const row = dedupeForWizardSelection(getAllPlugins()).find((p) => p.id === "ops-registry-invoke");
    expect(row).toBeDefined();
    expect(row?.source).toBe("cannbot");
    expect(row?.displayName).toBe("ops-registry-invoke");

    // resolution paths (status/uninstall of legacy installs) agree
    const found = findPlugin("ops-registry-invoke");
    expect(found?.displayName).toBe("ops-registry-invoke");
  });

  it("hides uncovered tombstones from the install list once discovery is authoritative", async () => {
    // online: discovery produced live entries — an uncovered tombstone means
    // upstream genuinely does not ship the plugin (e.g. ops-registry-invoke
    // pending its manifest in the cannbot repo); its row would be a dead
    // install target, so it stays out of the install list
    const { setCannbotPlugins, getAllPlugins } = await import("../src/core/registry.js");

    setCannbotPlugins([
      {
        id: "some-live-plugin",
        dir: "",
        displayName: "some-live-plugin",
        script: "cannbot",
        aliases: [],
        skills: 0,
        agents: 0,
        description: "",
        version: "1.0.0",
        source: "cannbot",
      },
    ]);
    // the cannbot tombstone (dead delegation target) is out of the list;
    // the frozen local copy (skills source, locally installable) may remain
    expect(getAllPlugins().some((p) => p.id === "ops-registry-invoke" && p.source === "cannbot")).toBe(false);

    // offline fallback preserved: with no discovered entries the tombstone
    // is the only pointer to the migrated plugin
    setCannbotPlugins([]);
    expect(getAllPlugins().some((p) => p.id === "ops-registry-invoke" && p.source === "cannbot")).toBe(true);
  });

  it("suppresses a tombstone whose cannbot identity was renamed (alias coverage)", async () => {    // the cannbot repo renamed collaborative-agent-kernel-evolution's
    // manifest name to "cake"; the tombstone (legacy id, alias "cake") must
    // collapse into the live cannbot entry — one wizard row, and the legacy
    // id routes to the entry the cannbot installer actually knows
    const savedHome = process.env.HOME;
    const savedInstallerPath = process.env.CANNBOT_INSTALLER_PATH;
    const sandboxHome = join(tmpdir(), `ih-reg-rename-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    mkdirSync(join(sandboxHome, ".cannbot"), { recursive: true });
    seedMainlineCache(sandboxHome);
    process.env.HOME = sandboxHome;
    try {
      const installerRoot = join(sandboxHome, "installer");
      createFakeInstaller(installerRoot, "fake-cake-plugin");
      const manifestPath = join(installerRoot, "dist", "plugins", "fake-cake-plugin", ".claude-plugin", "plugin.json");
      const manifest = JSON.parse(readFileSync(manifestPath, "utf-8"));
      manifest.name = "cake";
      writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
      process.env.CANNBOT_INSTALLER_PATH = installerRoot;

      const { ensureCannbotDiscovery } = await import("../src/core/cannbot-installer.js");
      const { getAllPlugins, findPlugin } = await import("../src/core/registry.js");
      await ensureCannbotDiscovery();

      const rows = getAllPlugins().filter(
        (p) => p.id === "cake" || p.id === "collaborative-agent-kernel-evolution"
      );
      expect(rows).toHaveLength(1);
      expect(rows[0].id).toBe("cake");
      expect(rows[0].source).toBe("cannbot");

      const found = findPlugin("collaborative-agent-kernel-evolution");
      expect(found?.id).toBe("cake");
      expect(found?.source).toBe("cannbot");
    } finally {
      process.env.HOME = savedHome;
      if (savedInstallerPath === undefined) delete process.env.CANNBOT_INSTALLER_PATH;
      else process.env.CANNBOT_INSTALLER_PATH = savedInstallerPath;
      rmSync(sandboxHome, { recursive: true, force: true });
    }
  });
});

describe("mergeDynamicPlugins scope (official + community)", () => {
  it("registers plugins from both official and community dirs", async () => {
    // requirement: every plugin the repository ships is installable —
    // the scan scope is not limited to plugins-official
    const { mergeDynamicPlugins, findPlugin } = await import("../src/core/registry.js");
    const { mkdirSync, writeFileSync, rmSync } = await import("fs");
    const { join } = await import("path");
    const { tmpdir } = await import("os");

    const repo = join(tmpdir(), `ih-reg-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    try {
      const official = join(repo, "plugins-official", "zz-fake-official-plugin");
      mkdirSync(official, { recursive: true });
      writeFileSync(join(official, "init.sh"), "#!/bin/bash\n");
      const community = join(repo, "plugins-community", "zz-fake-community-plugin");
      mkdirSync(community, { recursive: true });
      writeFileSync(join(community, "init.sh"), "#!/bin/bash\n");

      mergeDynamicPlugins(repo);

      expect(findPlugin("zz-fake-official-plugin")).toBeDefined();
      expect(findPlugin("zz-fake-community-plugin")).toBeDefined();
    } finally {
      rmSync(repo, { recursive: true, force: true });
    }
  });

  it("skips community plugins without an install mechanism (init.sh)", async () => {
    const { mergeDynamicPlugins, findPlugin } = await import("../src/core/registry.js");
    const { mkdirSync, rmSync } = await import("fs");
    const { join } = await import("path");
    const { tmpdir } = await import("os");

    const repo = join(tmpdir(), `ih-reg-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    try {
      const bare = join(repo, "plugins-community", "zz-bare-community-plugin");
      mkdirSync(bare, { recursive: true });

      mergeDynamicPlugins(repo);

      expect(findPlugin("zz-bare-community-plugin")).toBeUndefined();
    } finally {
      rmSync(repo, { recursive: true, force: true });
    }
  });

  it("enriches the entry from the plugin's own manifest (repo content wins)", async () => {
    const { mergeDynamicPlugins, findPlugin } = await import("../src/core/registry.js");
    const { mkdirSync, writeFileSync, rmSync } = await import("fs");
    const { join } = await import("path");
    const { tmpdir } = await import("os");

    const repo = join(tmpdir(), `ih-reg-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    try {
      const community = join(repo, "plugins-community", "zz-manifested-plugin");
      mkdirSync(join(community, ".claude-plugin"), { recursive: true });
      writeFileSync(join(community, "init.sh"), "#!/bin/bash\n");
      writeFileSync(
        join(community, ".claude-plugin", "plugin.json"),
        JSON.stringify({
          name: "zz-manifested-plugin",
          description: "社区插件：由 manifest 提供的仓库描述",
          skills: ["./skills/one"],
          agents: ["./agents/a.md", "./agents/b.md"],
        })
      );

      mergeDynamicPlugins(repo);

      const found = findPlugin("zz-manifested-plugin");
      expect(found).toBeDefined();
      // identity stays the repository dir name; description and counts
      // come from the repository-authored manifest
      expect(found?.description).toBe("社区插件：由 manifest 提供的仓库描述");
      expect(found?.skills).toBe(1);
      expect(found?.agents).toBe(2);
    } finally {
      rmSync(repo, { recursive: true, force: true });
    }
  });
});
