// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { existsSync } from "fs";
import { join } from "path";
import { execa } from "execa";
import type { AITool, InstallLevel, InstallOptions, InstallResult, BackupInfo } from "../types/index.js";
import { getPluginById } from "./registry.js";
import { readManifest } from "./manifest.js";
import { getConfigRoot } from "../utils/paths.js";
import { scanInstalledFiles, writeRecord } from "./record.js";
import { detectCurrentPlugin, createBackup, getAgentsFileName } from "./backup.js";
import { showOverwriteWarning } from "../ui/backup-prompts.js";
import { t } from "../utils/i18n.js";
import { logger } from "../utils/logger.js";

export async function installPlugin(
  opts: InstallOptions
): Promise<InstallResult> {
  const plugin = getPluginById(opts.pluginId);
  if (!plugin) {
    return {
      success: false,
      pluginId: opts.pluginId,
      skillsCount: 0,
      agentsCount: 0,
      errors: [`Plugin not found: ${opts.pluginId}`],
      warnings: [],
    };
  }

  const scriptPath = join(opts.repoPath, plugin.dir, plugin.script);
  if (!existsSync(scriptPath)) {
    return {
      success: false,
      pluginId: opts.pluginId,
      skillsCount: 0,
      agentsCount: 0,
      errors: [`Script not found: ${scriptPath}`],
      warnings: [],
    };
  }

  const args = [opts.level, opts.tool];
  if (opts.installPath) {
    args.push(opts.installPath);
  }

  const cwd = opts.installPath || process.cwd();

  const configRoot = getConfigRoot(opts.tool, opts.level, opts.installPath);
  const agentsFile = join(configRoot, getAgentsFileName(opts.tool));
  let backupInfo: BackupInfo | null = null;

  if (existsSync(agentsFile)) {
    const currentPlugin = detectCurrentPlugin(configRoot, opts.tool);

    if (currentPlugin && currentPlugin.pluginId !== opts.pluginId) {
      let choice: "overwrite" | "cancel" = "overwrite";
      
      if (!opts.yes) {
        choice = await showOverwriteWarning(
          currentPlugin.pluginName,
          plugin.displayName
        );
      }

      if (choice === "cancel") {
        logger.info(t("backup_cancel"));
        return {
          success: false,
          pluginId: opts.pluginId,
          skillsCount: 0,
          agentsCount: 0,
          errors: [t("backup_cancel")],
          warnings: [],
        };
      }

      backupInfo = createBackup(
        configRoot,
        opts.tool,
        currentPlugin.pluginId,
        currentPlugin.pluginName
      );

      if (backupInfo) {
        logger.success(`${t("backup_created")}: ${backupInfo.filePath}`);
      }
    }
  }

  try {
    const execOptions: any = {
      cwd,
      timeout: 300000,
      stdio: "pipe",
      input: opts.yes ? "y\n" : undefined,
    };
    
    await execa("bash", [scriptPath, ...args], execOptions);

    const manifest = readManifest(configRoot);

    let skillsCount = 0;
    let agentsCount = 0;

    if (manifest) {
      skillsCount = manifest.installed_skills?.length || 0;
      agentsCount = manifest.installed_agents?.length || 0;
    } else {
      skillsCount = plugin.skills;
      agentsCount = plugin.agents;
    }

    try {
      const record = scanInstalledFiles(
        opts.pluginId,
        plugin.displayName,
        opts.tool,
        opts.level,
        cwd,
        configRoot,
        manifest
      );
      
      if (backupInfo) {
        record.backup = {
          filePath: backupInfo.filePath,
          fromPluginId: backupInfo.pluginId,
          fromPluginName: backupInfo.pluginName,
          backupTime: backupInfo.backupTime,
        };
      }
      
      writeRecord(record);
    } catch {
      // Record writing is best-effort, don't fail install
    }

    return {
      success: true,
      pluginId: opts.pluginId,
      skillsCount,
      agentsCount,
      errors: [],
      warnings: [],
    };
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";
    return {
      success: false,
      pluginId: opts.pluginId,
      skillsCount: 0,
      agentsCount: 0,
      errors: [errorMessage],
      warnings: [],
    };
  }
}

export async function installPlugins(
  pluginIds: string[],
  tool: AITool,
  level: InstallLevel,
  repoPath: string,
  installPath?: string,
  yes?: boolean
): Promise<InstallResult[]> {
  const results: InstallResult[] = [];

  for (const pluginId of pluginIds) {
    const result = await installPlugin({
      pluginId,
      tool,
      level,
      repoPath,
      installPath,
      yes,
    });
    results.push(result);
  }

  return results;
}
