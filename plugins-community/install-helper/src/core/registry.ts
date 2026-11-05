// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { readdirSync, existsSync, readFileSync } from "fs";
import { join } from "path";
import type { PluginEntry, PluginSource } from "../types/index.js";
import { isDirectory } from "../utils/fs-helpers.js";
import { getPluginDirs } from "./scanner.js";
import { MIGRATED_PLUGINS } from "./migrated-plugins.js";
import embeddedPlugins from "../embedded-plugins.json" with { type: "json" };

function loadPlugins(): PluginEntry[] {
  const plugins = [...embeddedPlugins] as PluginEntry[];
  plugins.sort((a, b) =>
    a.displayName < b.displayName ? -1 : a.displayName > b.displayName ? 1 : 0
  );
  return plugins;
}

export let PLUGIN_REGISTRY: PluginEntry[] = loadPlugins();

// cannbot-repo plugins discovered from the managed @cannbot-plugin/cannbot
// package (dist/plugins manifests). Kept separate from the embedded legacy
// snapshot: cannbot entries are resolved at runtime and never baked into
// embedded-plugins.json (no release coupling with the cannbot repo).
let CANNBOT_PLUGINS: PluginEntry[] = [];

export function setCannbotPlugins(entries: PluginEntry[]): void {
  CANNBOT_PLUGINS = entries.map((entry) => ({ ...entry, source: "cannbot" as PluginSource }));
}

export function getCannbotPlugins(): PluginEntry[] {
  return [...CANNBOT_PLUGINS];
}

/** Reads a plugin's self-maintained manifest for display metadata
 *  (description / skills / agents counts). Returns null when absent. */
function readDynamicManifest(pluginPath: string): {
  description?: unknown;
  skills?: unknown;
  agents?: unknown;
} | null {
  try {
    return JSON.parse(readFileSync(join(pluginPath, ".claude-plugin", "plugin.json"), "utf-8"));
  } catch {
    return null;
  }
}

export function mergeDynamicPlugins(repoPath: string): void {
  if (!existsSync(repoPath)) return;

  const existingIds = new Set(PLUGIN_REGISTRY.map((p) => p.id));
  const dynamic: PluginEntry[] = [];

  // Plugin discovery scope follows repository.yaml pluginDirs — official and
  // community alike: every plugin the repository ships is installable.
  // init.sh is the install mechanism; the plugin's own manifest enriches the
  // display metadata (repo content is the display authority).
  for (const pluginDir of getPluginDirs()) {
    const parent = join(repoPath, pluginDir);
    if (!existsSync(parent)) continue;

    let entries: string[] = [];
    try {
      entries = readdirSync(parent);
    } catch {
      continue;
    }

    for (const entry of entries) {
      const pluginPath = join(parent, entry);
      if (!isDirectory(pluginPath)) continue;
      if (entry === "install-helper") continue;

      const initSh = join(pluginPath, "init.sh");
      if (!existsSync(initSh)) continue;

      const pluginId = entry;
      if (existingIds.has(pluginId)) continue;

      const manifest = readDynamicManifest(pluginPath);
      dynamic.push({
        id: pluginId,
        dir: `${pluginDir}/${entry}`,
        displayName: entry,
        script: "init.sh",
        aliases: [],
        skills: Array.isArray(manifest?.skills) ? manifest.skills.length : 0,
        agents: Array.isArray(manifest?.agents) ? manifest.agents.length : 0,
        description: typeof manifest?.description === "string" ? manifest.description : "",
      });
      existingIds.add(pluginId);
    }
  }

  if (dynamic.length > 0) {
    PLUGIN_REGISTRY = [...PLUGIN_REGISTRY, ...dynamic].sort((a, b) =>
      a.displayName < b.displayName ? -1 : a.displayName > b.displayName ? 1 : 0
    );
  }
}

/**
 * Resolves a plugin by id, alias or unique prefix. Resolution is cannbot-first
 * (the cannbot repo is authoritative for migrated plugins); migrated
 * tombstones (MIGRATED_PLUGINS) participate only in the cannbot scope, so
 * `--source skills` never resolves them — they have no legacy tree left in
 * this repository. Snapshot entries (CANNBOT_PLUGINS) win over tombstones
 * for the same id once the installer npm release catches up with the
 * migration. Tombstone exact/alias matching runs before any prefix matching
 * so a historical alias (e.g. "triton") is not captured by a prefix match on
 * a different cannbot plugin (e.g. triton-auto-evolve).
 */
export function findPlugin(
  query: string,
  options?: { source?: PluginSource }
): PluginEntry | undefined {
  const normalized = query.toLowerCase().trim();
  const wantCannbot = options?.source !== "skills";
  const tombstone = MIGRATED_PLUGINS.find(
    (p) => p.id === normalized || p.id === query.trim() || p.aliases.some((a) => a === normalized)
  );

  // cannbot-source plugins win for same-name ids (migration policy: the new
  // generation in the cannbot repo is authoritative; the legacy version can
  // still be requested explicitly via --source skills).
  if (wantCannbot) {
    const cannbotExact = CANNBOT_PLUGINS.find(
      (p) => p.id === normalized || p.id === query.trim()
    );
    if (cannbotExact) return cannbotExact;

    const cannbotAlias = CANNBOT_PLUGINS.find((p) =>
      p.aliases.some((a) => a === normalized)
    );
    if (cannbotAlias) return cannbotAlias;

    // Migrated tombstones: exact/alias resolve before any prefix matching so
    // the historical alias (e.g. "triton") is not captured by a prefix match
    // on a different cannbot plugin (e.g. triton-auto-evolve). When the
    // tombstone's live cannbot identity is discovered (same id, or renamed
    // manifest name matching a tombstone alias, e.g. cake), route to that
    // entry — the delegation must use the name the installer knows.
    if (tombstone) {
      const covered = CANNBOT_PLUGINS.find((p) => p.id === tombstone.id || tombstone.aliases.includes(p.id));
      // the bare tombstone (offline) still carries the repository identity
      return covered || { ...tombstone, displayName: tombstone.id };
    }
  }

  const exactMatch = PLUGIN_REGISTRY.find(
    (p) => p.id === normalized || p.id === query.trim()
  );
  if (exactMatch) return exactMatch;

  const aliasMatch = PLUGIN_REGISTRY.find((p) =>
    p.aliases.some((a) => a === normalized)
  );
  if (aliasMatch) return aliasMatch;

  // A removed plugin must not resolve to a different local plugin by prefix
  // during fallback.
  if (!wantCannbot && tombstone) return undefined;

  if (wantCannbot) {
    const cannbotPrefixes = CANNBOT_PLUGINS.filter((p) => p.id.startsWith(normalized));
    if (cannbotPrefixes.length === 1) return cannbotPrefixes[0];

    const tombstonePrefixes = MIGRATED_PLUGINS.filter((p) => p.id.startsWith(normalized));
    if (tombstonePrefixes.length === 1) return tombstonePrefixes[0];
  }

  const prefixMatches = PLUGIN_REGISTRY.filter((p) => p.id.startsWith(normalized));
  if (prefixMatches.length === 1) return prefixMatches[0];

  return undefined;
}

/**
 * All resolvable plugins across sources: cannbot snapshot entries, the
 * embedded legacy registry, and migrated tombstones. A tombstone whose id
 * the installer snapshot already lists is deduplicated (the snapshot entry
 * is authoritative once the npm release catches up with the migration).
 * Dual-source plugins (a migrated id whose frozen local copy still exists in
 * plugins.d) intentionally appear once per source: the tombstone bridges
 * cannbot discovery while the frozen copy stays installable via
 * --source skills (see migrated-plugins.test.ts).
 */
export function getAllPlugins(): PluginEntry[] {
  // Tombstone visibility rule: once cannbot discovery is authoritative
  // (live entries exist), the discovered set IS the offer — a tombstone left
  // uncovered means upstream genuinely does not ship the plugin
  // (mid-migration, removed), so its row would be a dead install target and
  // stays out of the install list. Offline (no discovered entries),
  // tombstones remain the fallback rows. Visible tombstone rows carry the
  // repository identity (the id) — legacy friendly names must not leak into
  // the wizard/list surfaces (repo-name-first display rule).
  const tombstones =
    CANNBOT_PLUGINS.length > 0
      ? []
      : MIGRATED_PLUGINS.map((t) => ({ ...t, displayName: t.id }));
  return [...CANNBOT_PLUGINS, ...PLUGIN_REGISTRY, ...tombstones].sort((a, b) =>
    a.displayName < b.displayName ? -1 : a.displayName > b.displayName ? 1 : 0
  );
}

/**
 * Exact-id lookup across sources: cannbot snapshot first, then the embedded
 * legacy registry, then migrated tombstones (so legacy install records of
 * migrated plugins keep resolving against the tombstone entry).
 */
export function getPluginById(id: string): PluginEntry | undefined {
  const cannbot = CANNBOT_PLUGINS.find((p) => p.id === id);
  if (cannbot) return cannbot;
  const legacy = PLUGIN_REGISTRY.find((p) => p.id === id);
  if (legacy) return legacy;
  const tombstone = MIGRATED_PLUGINS.find((p) => p.id === id);
  // tombstone rows display the repository identity (repo-name-first)
  return tombstone ? { ...tombstone, displayName: tombstone.id } : undefined;
}
