// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------
import { existsSync, mkdirSync, symlinkSync, realpathSync, readlinkSync, copyFileSync, cpSync, writeFileSync, rmSync } from "fs";
import { join, basename } from "path";
import { execa } from "execa";
import type { AITool, InstallLevel, PluginEntry, PluginManifestExternalRepo, CannbotManifest } from "../types/index.js";
import { getConfigRoot, getAgentsFileName } from "../utils/paths.js";
import { isSymlink, removePath } from "../utils/fs-helpers.js";
import { logger } from "../utils/logger.js";
import { t } from "../utils/i18n.js";

export interface ManifestInstallResult {
  success: boolean;
  skillsCount: number;
  agentsCount: number;
  errors: string[];
  manifest: CannbotManifest | null;
}

function installLink(source: string, target: string): "symlink" | "copy" | "skipped" | "failed" {
  try {
    const resolved = realpathSync(source);

    if (isSymlink(target)) {
      try {
        const currentTarget = readlinkSync(target);
        const normCurrent = currentTarget.replace(/\\/g, "/").toLowerCase();
        const normSource = resolved.replace(/\\/g, "/").toLowerCase();
        if (normCurrent === normSource) {
          return "skipped";
        }
      } catch {
        return "skipped";
      }
    }

    if (existsSync(target) || isSymlink(target)) {
      removePath(target);
    }
    try {
      symlinkSync(resolved, target);
      return "symlink";
    } catch (e: any) {
      if (e.code === "EPERM") {
        cpSync(resolved, target, { recursive: true });
        return "copy";
      }
      throw e;
    }
  } catch {
    return "failed";
  }
}

function installConfigFile(
  pluginDir: string,
  configFile: string,
  tool: AITool,
  level: InstallLevel,
  configRoot: string,
  installPath: string,
  configRootConfigLink?: boolean
): void {
  const sourcePath = join(pluginDir, configFile);
  if (!existsSync(sourcePath)) return;

  const agentsFileName = getAgentsFileName(tool);
  const isProjectLevel = level === "project";

  let primaryTarget: string;
  if (isProjectLevel) {
    primaryTarget = join(installPath, agentsFileName);
  } else {
    primaryTarget = join(configRoot, agentsFileName);
  }

  try {
    const resolvedSource = realpathSync(sourcePath);

    if (existsSync(primaryTarget) || isSymlink(primaryTarget)) {
      removePath(primaryTarget);
    }

    try {
      symlinkSync(resolvedSource, primaryTarget);
    } catch (e: any) {
      if (e.code === "EPERM") {
        copyFileSync(sourcePath, primaryTarget);
      } else {
        throw e;
      }
    }
  } catch (e: any) {
    logger.warn(t("install_config_failed").replace("{error}", e.message));
  }

  if (isProjectLevel && primaryTarget !== join(configRoot, agentsFileName)) {
    if (configRootConfigLink === false) return;
    const configRootTarget = join(configRoot, agentsFileName);
    if (existsSync(configRootTarget) || isSymlink(configRootTarget)) {
      removePath(configRootTarget);
    }
    try {
      symlinkSync(realpathSync(sourcePath), configRootTarget);
    } catch (e: any) {
      if (e.code === "EPERM") {
        copyFileSync(sourcePath, configRootTarget);
      }
    }
  }
}

async function installExternalRepos(
  repos: PluginManifestExternalRepo[] | undefined,
  repoPath: string,
  configRoot: string,
  installPath: string,
  level: InstallLevel
): Promise<void> {
  if (!repos || repos.length === 0) return;

  for (const repo of repos) {
    const targetDir = join(repoPath, repo.dir);
    const gitDir = join(targetDir, ".git");

    if (existsSync(gitDir)) {
      try {
        await execa("git", ["pull", "--quiet"], { cwd: targetDir, timeout: 120000 });
      } catch {
        logger.warn(t("external_repo_update_failed").replace("{dir}", repo.dir));
        // 不 continue — 目录已存在且可用，继续创建 symlink（与 init.sh 行为一致）
      }
    } else if (!existsSync(targetDir)) {
      const args = ["clone"];
      if (repo.depth) {
        args.push("--depth", String(repo.depth));
      }
      if (repo.recursive) {
        args.push("--recursive");
      }
      args.push(repo.url, targetDir);

      try {
        await execa("git", args, { timeout: 120000 });
      } catch {
        logger.warn(t("install_clone_failed").replace("{url}", repo.url).replace("{dir}", repo.dir));
        try { rmSync(targetDir, { recursive: true, force: true }); } catch {}
        continue;
      }
    }

    const linkName = basename(repo.dir);
    const resolved = realpathSync(targetDir);
    const shouldConfigRootLink = repo.configRootLink !== false && (level === "global" || repo.configRootLink === true);

    if (shouldConfigRootLink) {
      const configRootLink = join(configRoot, linkName);
      try {
        if (existsSync(configRootLink) || isSymlink(configRootLink)) {
          removePath(configRootLink);
        }
        try {
          symlinkSync(resolved, configRootLink);
        } catch (e: any) {
          if (e.code === "EPERM") {
            cpSync(resolved, configRootLink, { recursive: true });
          }
        }
      } catch {
        logger.warn(t("install_link_config_failed").replace("{name}", linkName));
      }
    }

    if (level === "project") {
      const projectLink = join(installPath, linkName);
      try {
        if (existsSync(projectLink) || isSymlink(projectLink)) {
          removePath(projectLink);
        }
        try {
          symlinkSync(resolved, projectLink);
        } catch (e: any) {
          if (e.code === "EPERM") {
            cpSync(resolved, projectLink, { recursive: true });
          }
        }
      } catch {
        logger.warn(t("install_link_project_failed").replace("{name}", linkName));
      }
    }
  }
}

function installWorkflows(pluginDir: string, configRoot: string): void {
  const workflowsDir = join(pluginDir, "workflows");
  if (!existsSync(workflowsDir)) return;

  const targetPath = join(configRoot, "workflows");
  try {
    const resolved = realpathSync(workflowsDir);
    if (existsSync(targetPath) || isSymlink(targetPath)) {
      removePath(targetPath);
    }
    try {
      symlinkSync(resolved, targetPath);
    } catch (e: any) {
      if (e.code === "EPERM") {
        cpSync(resolved, targetPath, { recursive: true });
      }
    }
  } catch {
    logger.warn(t("install_link_workflows_failed"));
  }
}

export function writeCannbotManifest(
  configRoot: string,
  pluginId: string,
  version: string,
  level: InstallLevel,
  tool: AITool,
  skills: string[],
  agents: string[]
): CannbotManifest {
  const manifest: CannbotManifest = {
    brand: "CANNBot",
    version,
    team: pluginId,
    level,
    tool,
    installed_skills: skills,
    installed_agents: agents,
    brand_dir: configRoot,
    install_time: new Date().toISOString(),
  };

  const manifestPath = join(configRoot, `${pluginId}-manifest.json`);
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2), "utf-8");
  return manifest;
}

export async function installViaManifest(
  plugin: PluginEntry,
  repoPath: string,
  tool: AITool,
  level: InstallLevel,
  installPath?: string
): Promise<ManifestInstallResult> {
  const cwd = installPath || process.cwd();
  const configRoot = getConfigRoot(tool, level, installPath);
  const pluginDir = join(repoPath, plugin.dir);

  try {
    const skillsDir = join(configRoot, "skills");
    const agentsDir = join(configRoot, "agents");
    mkdirSync(skillsDir, { recursive: true });
    mkdirSync(agentsDir, { recursive: true });

    const installedSkills: string[] = [];
    const installedAgents: string[] = [];

    for (const source of plugin.installSkills || []) {
      const sourceDir = join(repoPath, source.dir);
      if (!existsSync(sourceDir)) {
        logger.warn(t("install_skill_source_not_found").replace("{dir}", source.dir));
        continue;
      }

      for (const skillEntry of source.skills) {
        const skillName = typeof skillEntry === "string" ? skillEntry : skillEntry.name;
        const skillAlias = typeof skillEntry === "string" ? skillName : (skillEntry.as || skillName);
        const skillPath = join(sourceDir, skillName);
        if (!existsSync(skillPath)) {
          logger.warn(t("install_skill_not_found").replace("{name}", skillName).replace("{dir}", source.dir));
          continue;
        }

        const targetPath = join(skillsDir, skillAlias);
        const result = installLink(skillPath, targetPath);
        if (result !== "failed") {
          installedSkills.push(skillAlias);
        } else {
          logger.warn(t("install_skill_failed").replace("{name}", skillAlias));
        }
      }
    }

    for (const agentName of plugin.installAgents || []) {
      const agentFile = agentName.endsWith(".md") ? agentName : `${agentName}.md`;
      const agentPath = join(pluginDir, "agents", agentFile);
      if (!existsSync(agentPath)) {
        logger.warn(t("install_agent_not_found").replace("{name}", agentFile));
        continue;
      }

      const targetPath = join(agentsDir, agentFile);
      const result = installLink(agentPath, targetPath);
      if (result !== "failed") {
        installedAgents.push(agentFile);
      } else {
        logger.warn(t("install_agent_failed").replace("{name}", agentFile));
      }
    }

    installConfigFile(pluginDir, plugin.configFile || "AGENTS.md", tool, level, configRoot, cwd, plugin.configRootConfigLink);

    installWorkflows(pluginDir, configRoot);

    await installExternalRepos(plugin.externalRepos, repoPath, configRoot, cwd, level);

    const cannbotManifest = writeCannbotManifest(
      configRoot,
      plugin.id,
      plugin.version || "1.0.0",
      level,
      tool,
      installedSkills,
      installedAgents
    );

    return {
      success: true,
      skillsCount: installedSkills.length,
      agentsCount: installedAgents.length,
      errors: [],
      manifest: cannbotManifest,
    };
  } catch (error) {
    return {
      success: false,
      skillsCount: 0,
      agentsCount: 0,
      errors: [error instanceof Error ? error.message : t("error_unknown")],
      manifest: null,
    };
  }
}
