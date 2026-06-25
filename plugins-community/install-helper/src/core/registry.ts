// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { readdirSync, readFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { parse as parseYaml } from "yaml";
import type { PluginEntry } from "../types/index.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

function loadPluginsFromYaml(): PluginEntry[] {
  const pluginsDir = join(__dirname, "plugins.d");
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
        });
      } catch {
      }
    }
  } catch {
  }
  return plugins;
}

export const PLUGIN_REGISTRY: PluginEntry[] = loadPluginsFromYaml();

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
