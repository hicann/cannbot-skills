// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { existsSync, mkdirSync, readFileSync, writeFileSync, lstatSync, unlinkSync } from "fs";
import { join } from "path";
import { getCannbotConfigDir, getConfigRoot } from "../utils/paths.js";
import type { AITool, InstallLevel, CannbotManifest } from "../types/index.js";

export interface InstallRecord {
  pluginId: string;
  displayName: string;
  tool: AITool;
  level: InstallLevel;
  installPath: string;
  configRoot: string;
  installTime: string;
  files: string[];
  directories: string[];
  backup?: {
    filePath: string;
    fromPluginId: string;
    fromPluginName: string;
    backupTime: string;
  };
}

function getInstallsDir(): string {
  return join(getCannbotConfigDir(), "installs");
}

export function getRecordPath(pluginId: string): string {
  return join(getInstallsDir(), `${pluginId}.json`);
}

export function readRecord(pluginId: string): InstallRecord | null {
  const recordPath = getRecordPath(pluginId);
  if (!existsSync(recordPath)) {
    return null;
  }

  try {
    const content = readFileSync(recordPath, "utf-8");
    return JSON.parse(content) as InstallRecord;
  } catch {
    return null;
  }
}

export function writeRecord(record: InstallRecord): void {
  const installsDir = getInstallsDir();
  if (!existsSync(installsDir)) {
    mkdirSync(installsDir, { recursive: true });
  }

  const recordPath = getRecordPath(record.pluginId);
  writeFileSync(recordPath, JSON.stringify(record, null, 2), "utf-8");
}

export function deleteRecord(pluginId: string): void {
  const recordPath = getRecordPath(pluginId);
  if (existsSync(recordPath)) {
    unlinkSync(recordPath);
  }
}

export function scanInstalledFiles(
  pluginId: string,
  displayName: string,
  tool: AITool,
  level: InstallLevel,
  installPath: string,
  configRoot: string,
  manifest: CannbotManifest | null
): InstallRecord {
  const files: string[] = [];
  const directories: string[] = [];

  if (manifest) {
    const skillsDir = join(configRoot, "skills");
    if (existsSync(skillsDir)) {
      directories.push(skillsDir);
      for (const skillName of manifest.installed_skills || []) {
        const skillPath = join(skillsDir, skillName);
        if (existsSync(skillPath) || isSymlink(skillPath)) {
          files.push(skillPath);
        }
      }
    }

    const agentsDir = join(configRoot, "agents");
    if (existsSync(agentsDir)) {
      directories.push(agentsDir);
      for (const agentName of manifest.installed_agents || []) {
        const agentPath = join(agentsDir, agentName);
        const agentPathMd = join(agentsDir, agentName + ".md");
        if (existsSync(agentPath) || isSymlink(agentPath)) {
          files.push(agentPath);
        } else if (existsSync(agentPathMd) || isSymlink(agentPathMd)) {
          files.push(agentPathMd);
        }
      }
    }
  }

  const workflowsLink = join(configRoot, "workflows");
  if (isSymlink(workflowsLink)) {
    files.push(workflowsLink);
  }

  const manifestPath = join(configRoot, "cannbot-manifest.json");
  if (existsSync(manifestPath)) {
    files.push(manifestPath);
  }

  const pluginManifestPath = join(configRoot, `${pluginId}-manifest.json`);
  if (existsSync(pluginManifestPath)) {
    files.push(pluginManifestPath);
  }

  const configFileName = tool === "claude" ? "CLAUDE.md" : "AGENTS.md";
  const configFilePath = level === "project"
    ? join(installPath, configFileName)
    : join(configRoot, configFileName);
  if (existsSync(configFilePath)) {
    files.push(configFilePath);
  }

  const repoLinks = ["asc-devkit", "pypto", "tilelang-ascend", "cann-recipes-infer", "cann-samples"];
  for (const repoName of repoLinks) {
    const repoLinkPath = join(installPath, repoName);
    if (isSymlink(repoLinkPath)) {
      files.push(repoLinkPath);
    }
    const repoLinkInConfig = join(configRoot, repoName);
    if (isSymlink(repoLinkInConfig)) {
      files.push(repoLinkInConfig);
    }
  }

  return {
    pluginId,
    displayName,
    tool,
    level,
    installPath,
    configRoot,
    installTime: new Date().toISOString(),
    files,
    directories,
  };
}

function isSymlink(path: string): boolean {
  try {
    return lstatSync(path).isSymbolicLink();
  } catch {
    return false;
  }
}

// === Skill-level install records ===

export interface SkillInstallEntry {
  skills: string[];
  installTime: string;
}

export interface SkillInstallRecord {
  [tool: string]: {
    [level: string]: {
      [installPath: string]: SkillInstallEntry;
    };
  };
}

function getSkillRecordPath(): string {
  return join(getInstallsDir(), "skills.json");
}

export function readSkillRecord(): SkillInstallRecord {
  const recordPath = getSkillRecordPath();
  if (!existsSync(recordPath)) {
    return {};
  }
  try {
    const content = readFileSync(recordPath, "utf-8");
    return JSON.parse(content) as SkillInstallRecord;
  } catch {
    return {};
  }
}

export function writeSkillRecord(record: SkillInstallRecord): void {
  const installsDir = getInstallsDir();
  if (!existsSync(installsDir)) {
    mkdirSync(installsDir, { recursive: true });
  }
  const recordPath = getSkillRecordPath();
  writeFileSync(recordPath, JSON.stringify(record, null, 2), "utf-8");
}

export function addSkillsToRecord(
  skillIds: string[],
  tool: AITool,
  level: InstallLevel,
  installPath: string
): void {
  const record = readSkillRecord();
  if (!record[tool]) record[tool] = {};
  if (!record[tool][level]) record[tool][level] = {};
  if (!record[tool][level][installPath]) {
    record[tool][level][installPath] = { skills: [], installTime: "" };
  }
  const entry = record[tool][level][installPath];
  for (const id of skillIds) {
    if (!entry.skills.includes(id)) {
      entry.skills.push(id);
    }
  }
  entry.installTime = new Date().toISOString();
  writeSkillRecord(record);
}

export function removeSkillsFromRecord(
  skillIds: string[],
  tool: AITool,
  level: InstallLevel,
  installPath: string
): void {
  const record = readSkillRecord();
  if (!record[tool]?.[level]?.[installPath]) return;
  const entry = record[tool][level][installPath];
  entry.skills = entry.skills.filter((id) => !skillIds.includes(id));
  if (entry.skills.length === 0) {
    delete record[tool][level][installPath];
    if (Object.keys(record[tool][level]).length === 0) {
      delete record[tool][level];
      if (Object.keys(record[tool]).length === 0) {
        delete record[tool];
      }
    }
  }
  writeSkillRecord(record);
}

export function getInstalledSkills(
  tool: AITool,
  level: InstallLevel,
  installPath: string
): string[] {
  const record = readSkillRecord();
  const recordedSkills = record[tool]?.[level]?.[installPath]?.skills || [];
  
  // 验证实际文件是否存在
  const configRoot = getConfigRoot(tool, level);
  const skillsDir = join(configRoot, "skills");
  
  return recordedSkills.filter((skillId) => {
    const skillPath = join(skillsDir, skillId);
    return existsSync(skillPath) || isSymlink(skillPath);
  });
}
