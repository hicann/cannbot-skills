// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { existsSync, readFileSync } from "fs";
import { join } from "path";
import type {
  CannbotManifest,
  InstalledPlugin,
  AITool,
  InstallLevel,
} from "../types/index.js";
import { getConfigRoot, getManifestPath, getAgentsFileName, VALID_TOOLS } from "../utils/paths.js";
import { getAllPlugins } from "./registry.js";
import { isSymlink } from "../utils/fs-helpers.js";

export function readManifest(configRoot: string): CannbotManifest | null {
  const manifestPath = getManifestPath(configRoot);
  if (existsSync(manifestPath)) {
    try {
      const content = readFileSync(manifestPath, "utf-8");
      return JSON.parse(content) as CannbotManifest;
    } catch {
      // fall through
    }
  }
  return null;
}

export function readAllManifests(configRoot: string): CannbotManifest[] {
  const manifests: CannbotManifest[] = [];
  const plugins = getAllPlugins();

  const standardPath = getManifestPath(configRoot);
  if (existsSync(standardPath)) {
    try {
      const content = readFileSync(standardPath, "utf-8");
      manifests.push(JSON.parse(content) as CannbotManifest);
    } catch {
      // ignore
    }
  }

  for (const plugin of plugins) {
    const pluginManifestPath = join(configRoot, `${plugin.id}-manifest.json`);
    if (existsSync(pluginManifestPath)) {
      try {
        const content = readFileSync(pluginManifestPath, "utf-8");
        manifests.push(JSON.parse(content) as CannbotManifest);
      } catch {
        // ignore
      }
    }
  }

  return manifests;
}

export function scanInstalled(): InstalledPlugin[] {
  const plugins = getAllPlugins();
  const installed: InstalledPlugin[] = [];
  const tools: AITool[] = VALID_TOOLS;
  const levels: InstallLevel[] = ["project", "global"];

  for (const tool of tools) {
    for (const level of levels) {
      const configRoot = getConfigRoot(tool, level);
      const manifests = readAllManifests(configRoot);

      for (const manifest of manifests) {
        const plugin = plugins.find((p) => p.id === manifest.team);
        if (!plugin) continue;

        // 验证实际文件是否存在
        if (!verifyPluginFilesExist(configRoot, manifest, tool, level)) {
          continue;
        }

        const alreadyAdded = installed.some(
          (p) => p.id === plugin.id && p.tool === tool && p.level === level
        );
        if (alreadyAdded) continue;

        installed.push({
          id: plugin.id,
          displayName: plugin.displayName,
          tool,
          level,
          skillsCount: manifest.installed_skills?.length || 0,
          agentsCount: manifest.installed_agents?.length || 0,
          installTime: manifest.install_time,
          configRoot,
        });
      }
    }
  }

  return installed;
}

function verifyPluginFilesExist(configRoot: string, manifest: CannbotManifest, tool: AITool, level: InstallLevel): boolean {
  const skillsDir = join(configRoot, "skills");
  const agentsDir = join(configRoot, "agents");
  
  let hasAnyComponent = false;
  
  // 验证 skills
  if (manifest.installed_skills && manifest.installed_skills.length > 0) {
    const hasAnySkill = manifest.installed_skills.some((skillId) => {
      const skillPath = join(skillsDir, skillId);
      return existsSync(skillPath) || isSymlink(skillPath);
    });
    if (hasAnySkill) hasAnyComponent = true;
  }
  
  // 验证 agents
  if (manifest.installed_agents && manifest.installed_agents.length > 0) {
    const hasAnyAgent = manifest.installed_agents.some((agentId) => {
      const agentPath = join(agentsDir, agentId);
      const agentPathMd = join(agentsDir, agentId + ".md");
      return existsSync(agentPath) || isSymlink(agentPath) ||
             existsSync(agentPathMd) || isSymlink(agentPathMd);
    });
    if (hasAnyAgent) hasAnyComponent = true;
  }
  
  // 验证配置文件（AGENTS.md / CLAUDE.md）
  const agentsFileName = getAgentsFileName(tool);
  const configFilePath = level === "project"
    ? join(configRoot, "..", agentsFileName)
    : join(configRoot, agentsFileName);
  if (existsSync(configFilePath) || isSymlink(configFilePath)) hasAnyComponent = true;
  
  // 验证 per-plugin manifest 文件
  const pluginManifestPath = join(configRoot, `${manifest.team}-manifest.json`);
  if (existsSync(pluginManifestPath)) hasAnyComponent = true;
  
  return hasAnyComponent;
}
