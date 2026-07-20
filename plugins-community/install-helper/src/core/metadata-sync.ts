// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { existsSync, readFileSync, statSync } from "fs";
import { join } from "path";
import type { PluginEntry, PluginManifestSkillEntry, PluginManifestSkillSource } from "../types/index.js";
import { PLUGIN_REGISTRY } from "./registry.js";
import { getScanDirs } from "./scanner.js";

interface PluginJson {
  name?: string;
  version?: string;
  description?: string;
  agents?: string[];
}

function readPluginJson(pluginDir: string): PluginJson | null {
  const pjPath = join(pluginDir, ".claude-plugin", "plugin.json");
  try {
    return JSON.parse(readFileSync(pjPath, "utf-8")) as PluginJson;
  } catch {
    return null;
  }
}

function parseIncludedSkills(pluginDir: string): string[] {
  const initPath = join(pluginDir, "init.sh");
  try {
    const text = readFileSync(initPath, "utf-8");
    const m = text.match(/INCLUDED_SKILLS="([^"]*)"/);
    if (!m) return [];
    return m[1].split(/\s+/).filter(Boolean);
  } catch {
    return [];
  }
}

function findSkillSourceDir(skillName: string, repoRoot: string, scanDirs: string[]): string | null {
  for (const dir of scanDirs) {
    const candidate = join(repoRoot, dir, skillName);
    if (existsSync(candidate) && statSync(candidate).isDirectory()) {
      return dir;
    }
  }
  return null;
}

function collectDeclaredSkillNames(
  installSkills: PluginManifestSkillSource[] | undefined
): Set<string> {
  const set = new Set<string>();
  if (!installSkills) return set;
  for (const src of installSkills) {
    for (const sk of src.skills) {
      const name = typeof sk === "string" ? sk : sk.name;
      const as = typeof sk === "string" ? sk : (sk.as || sk.name);
      set.add(name);
      if (as !== name) set.add(as);
    }
  }
  return set;
}

function resolveAgentsFromPluginJson(pj: PluginJson | null): string[] {
  if (!pj || !Array.isArray(pj.agents)) return [];
  return pj.agents
    .map((a) => a.replace(/^\.\/agents\//, "").replace(/\.md$/, ""))
    .filter(Boolean);
}

function countSkills(installSkills: PluginManifestSkillSource[] | undefined): number {
  if (!installSkills) return 0;
  let n = 0;
  for (const src of installSkills) n += src.skills.length;
  return n;
}

/**
 * Merge init.sh INCLUDED_SKILLS into existing installSkills.
 * Existing skills (including `as` aliases) are preserved; init.sh-only skills
 * are appended to the matching source bucket (or a new bucket is created).
 */
function mergeInitSkills(
  existing: PluginManifestSkillSource[] | undefined,
  pluginDir: string,
  repoRoot: string,
  scanDirs: string[]
): PluginManifestSkillSource[] {
  const base = existing || [];
  const seen = collectDeclaredSkillNames(base);
  const initSkills = parseIncludedSkills(pluginDir);
  if (initSkills.length === 0) return base;

  const merged: PluginManifestSkillSource[] = base.map((s) => ({ ...s, skills: [...s.skills] }));

  for (const skill of initSkills) {
    if (seen.has(skill)) continue;
    const srcDir = findSkillSourceDir(skill, repoRoot, scanDirs);
    if (!srcDir) continue;
    let bucket = merged.find((b) => b.dir === srcDir);
    if (!bucket) {
      bucket = { dir: srcDir, skills: [] };
      merged.push(bucket);
    }
    bucket.skills.push(skill);
    seen.add(skill);
  }
  return merged;
}

/**
 * Enrich PLUGIN_REGISTRY in-place by reading the latest plugin.json and
 * init.sh INCLUDED_SKILLS from the freshly-pulled repository.
 *
 * Fields updated at runtime:
 *  - version            ← plugin.json version
 *  - description        ← plugin.json description (only if currently empty)
 *  - displayName        ← plugin.json name (only for dynamically discovered plugins with bare dir name)
 *  - installAgents      ← plugin.json agents (only if currently empty)
 *  - installSkills      ← existing + init.sh INCLUDED_SKILLS supplement
 *  - skills / agents    ← recomputed counts
 *
 * Fields NOT updated (require npm republish): aliases, externalRepos,
 * configRootConfigLink, configFile — these rarely change.
 */
export function enrichPluginMetadata(repoPath: string): void {
  const scanDirs = getScanDirs();

  for (const plugin of PLUGIN_REGISTRY) {
    const pluginDir = join(repoPath, plugin.dir);
    if (!existsSync(pluginDir)) continue;

    const pj = readPluginJson(pluginDir);

    // version: plugin.json overrides embedded hardcoded value
    if (pj && pj.version) {
      plugin.version = pj.version;
    }

    // description: fill from plugin.json if currently empty
    if ((!plugin.description || plugin.description === "") && pj && pj.description) {
      plugin.description = pj.description;
    }

    // displayName: for dynamically discovered plugins, upgrade from bare dir name
    if (pj && pj.name && (plugin.displayName === plugin.id)) {
      plugin.displayName = pj.name;
    }

    // installAgents: fill from plugin.json if currently empty
    if ((!plugin.installAgents || plugin.installAgents.length === 0)) {
      const agents = resolveAgentsFromPluginJson(pj);
      if (agents.length > 0) {
        plugin.installAgents = agents;
      }
    }

    // installSkills: merge init.sh INCLUDED_SKILLS into existing
    const merged = mergeInitSkills(plugin.installSkills, pluginDir, repoPath, scanDirs);
    plugin.installSkills = merged;

    // recompute counts
    plugin.skills = countSkills(merged);
    plugin.agents = plugin.installAgents ? plugin.installAgents.length : 0;
  }
}
