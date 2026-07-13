// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { readdirSync, readFileSync, existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { parse as parseYaml } from "yaml";
import type { PluginEntry } from "../types/index.js";
import { logger } from "../utils/logger.js";
import { t } from "../utils/i18n.js";
import { isDirectory } from "../utils/fs-helpers.js";
import embeddedPlugins from "../embedded-plugins.json" with { type: "json" };

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

function findPluginsDir(): string {
  const candidates = [
    join(__dirname, "plugins.d"),
    join(__dirname, "..", "plugins.d"),
    join(__dirname, "..", "..", "plugins.d"),
  ];
  for (const dir of candidates) {
    if (existsSync(dir)) return dir;
  }
  return candidates[0];
}

function loadPluginsFromYaml(): PluginEntry[] {
  const pluginsDir = findPluginsDir();
  const defaultsPath = join(pluginsDir, "_defaults.yml");

  let defaults: Record<string, unknown> = {};
  try {
    defaults = parseYaml(readFileSync(defaultsPath, "utf-8")) || {};
  } catch {
  }

  const plugins: PluginEntry[] = [];
  try {
    for (const file of readdirSync(pluginsDir)) {
      if (file.startsWith("_") || !file.endsWith(".yml")) continue;
      try {
        const content = parseYaml(readFileSync(join(pluginsDir, file), "utf-8"));
        if (!content || !content.id) continue;
        plugins.push({
          id: content.id,
          dir: content.dir,
          displayName: content.displayName || content.id,
          script: content.script || defaults.script || "init.sh",
          aliases: content.aliases || [],
          skills: content.skills ?? defaults.skills ?? 0,
          agents: content.agents ?? defaults.agents ?? 0,
          description: content.description || "",
          version: content.version,
          configFile: content.configFile,
          configRootConfigLink: content.configRootConfigLink,
          installSkills: content.installSkills,
          installAgents: content.installAgents,
          externalRepos: content.externalRepos,
        });
      } catch {
        logger.warn(t("plugin_yaml_error").replace("{file}", file));
      }
    }
  } catch {
  }

  if (plugins.length === 0) {
    return [...embeddedPlugins] as PluginEntry[];
  }

  plugins.sort((a, b) =>
    a.displayName < b.displayName ? -1 : a.displayName > b.displayName ? 1 : 0
  );
  return plugins;
}

export let PLUGIN_REGISTRY: PluginEntry[] = loadPluginsFromYaml();

const PLUGIN_DIRS = ["plugins-official", "plugins-community"];

export function mergeDynamicPlugins(repoPath: string): void {
  if (!existsSync(repoPath)) return;

  const existingIds = new Set(PLUGIN_REGISTRY.map((p) => p.id));
  const dynamic: PluginEntry[] = [];

  for (const pluginDir of PLUGIN_DIRS) {
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

      dynamic.push({
        id: pluginId,
        dir: `${pluginDir}/${entry}`,
        displayName: entry,
        script: "init.sh",
        aliases: [],
        skills: 0,
        agents: 0,
        description: "",
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

export function findPlugin(query: string): PluginEntry | undefined {
  const normalized = query.toLowerCase().trim();

  const exactMatch = PLUGIN_REGISTRY.find(
    (p) => p.id === normalized || p.id === query.trim()
  );
  if (exactMatch) return exactMatch;

  const aliasMatch = PLUGIN_REGISTRY.find((p) =>
    p.aliases.some((a) => a === normalized)
  );
  if (aliasMatch) return aliasMatch;

  const prefixMatch = PLUGIN_REGISTRY.find((p) => p.id.startsWith(normalized));
  if (prefixMatch) return prefixMatch;

  return undefined;
}

export function getAllPlugins(): PluginEntry[] {
  return PLUGIN_REGISTRY;
}

export function getPluginById(id: string): PluginEntry | undefined {
  return PLUGIN_REGISTRY.find((p) => p.id === id);
}
