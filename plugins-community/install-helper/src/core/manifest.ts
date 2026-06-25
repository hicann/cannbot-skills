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
import { getConfigRoot, getManifestPath } from "../utils/paths.js";
import { getAllPlugins } from "./registry.js";

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
  const tools: AITool[] = ["opencode", "claude", "trae", "cursor", "copilot"];
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
  
  // 验证 skills
  if (manifest.installed_skills && manifest.installed_skills.length > 0) {
    const hasAnySkill = manifest.installed_skills.some((skillId) => {
      const skillPath = join(skillsDir, skillId);
      return existsSync(skillPath);
    });
    if (!hasAnySkill) return false;
  }
  
  // 验证 agents
  if (manifest.installed_agents && manifest.installed_agents.length > 0) {
    const hasAnyAgent = manifest.installed_agents.some((agentId) => {
      const agentPath = join(agentsDir, agentId);
      return existsSync(agentPath);
    });
    if (!hasAnyAgent) return false;
  }
  
  // 验证 AGENTS.md
  // For opencode project level, AGENTS.md is in project root, not in configRoot
  let agentsMdPath: string;
  if (tool === "opencode" && level === "project") {
    // For opencode project level, AGENTS.md is in project root
    agentsMdPath = join(configRoot, "..", "AGENTS.md");
  } else {
    agentsMdPath = join(configRoot, "AGENTS.md");
  }
  if (!existsSync(agentsMdPath)) return false;
  
  return true;
}

export function isPluginInstalled(
  pluginId: string,
  tool?: AITool,
  level?: InstallLevel
): boolean {
  const installed = scanInstalled();
  return installed.some(
    (p) =>
      p.id === pluginId &&
      (!tool || p.tool === tool) &&
      (!level || p.level === level)
  );
}

export function getInstalledPlugin(
  pluginId: string,
  tool?: AITool,
  level?: InstallLevel
): InstalledPlugin | undefined {
  const installed = scanInstalled();
  return installed.find(
    (p) =>
      p.id === pluginId &&
      (!tool || p.tool === tool) &&
      (!level || p.level === level)
  );
}
