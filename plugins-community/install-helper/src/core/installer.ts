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
import { join, basename } from "path";
import { execa, execaSync } from "execa";
import type { AITool, InstallLevel, InstallOptions, InstallResult, BackupInfo } from "../types/index.js";
import { getPluginById } from "./registry.js";
import { readManifest } from "./manifest.js";
import { getConfigRoot } from "../utils/paths.js";
import { scanInstalledFiles, writeRecord } from "./record.js";
import { detectCurrentPlugin, createBackup, getAgentsFileName } from "./backup.js";
import { showOverwriteWarning } from "../ui/backup-prompts.js";
import { installViaManifest } from "./plugin-installer.js";
import { t } from "../utils/i18n.js";
import { logger } from "../utils/logger.js";

function findShell(): string | null {
  const candidates = ["bash", "sh"];
  for (const cmd of candidates) {
    try {
      const detectCmd = process.platform === "win32" ? "where" : "which";
      execaSync(detectCmd, [cmd], { timeout: 3000 });
      return cmd;
    } catch {}
  }
  return null;
}

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
      errors: [t("error_plugin_not_found") + ": " + opts.pluginId],
      warnings: [],
    };
  }

  const pluginDir = join(opts.repoPath, plugin.dir);
  const cwd = opts.installPath || process.cwd();
  const configRoot = getConfigRoot(opts.tool, opts.level, opts.installPath);
  const agentsFile = join(configRoot, getAgentsFileName(opts.tool));
  let backupInfo: BackupInfo | null = null;

  if (existsSync(agentsFile)) {
    const currentPlugin = detectCurrentPlugin(configRoot, opts.tool, opts.installPath);

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
    }

    if (currentPlugin) {
      backupInfo = createBackup(
        configRoot,
        opts.tool,
        currentPlugin.pluginId,
        currentPlugin.pluginName
      );

      if (!backupInfo) {
        logger.error(t("backup_failed"));
        return {
          success: false,
          pluginId: opts.pluginId,
          skillsCount: 0,
          agentsCount: 0,
          errors: [t("backup_failed")],
          warnings: [],
        };
      }

      logger.success(`${t("backup_created")}: ${backupInfo.filePath}`);
    }
  }

  if (plugin.installSkills && plugin.installSkills.length > 0) {
    const result = await installViaManifest(
      plugin,
      opts.repoPath,
      opts.tool,
      opts.level,
      opts.installPath
    );

    if (result.success) {
      try {
        const record = scanInstalledFiles(
          opts.pluginId,
          plugin.displayName,
          opts.tool,
          opts.level,
          cwd,
          configRoot,
          result.manifest,
          plugin.externalRepos?.map(r => basename(r.dir))
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
        logger.warn(t("record_write_failed"));
      }

      return {
        success: true,
        pluginId: opts.pluginId,
        skillsCount: result.skillsCount,
        agentsCount: result.agentsCount,
        errors: result.errors,
        warnings: [],
      };
    } else {
      return {
        success: false,
        pluginId: opts.pluginId,
        skillsCount: 0,
        agentsCount: 0,
        errors: result.errors,
        warnings: [],
      };
    }
  }

  const scriptPath = join(pluginDir, plugin.script);
  if (!existsSync(scriptPath)) {
    return {
      success: false,
      pluginId: opts.pluginId,
      skillsCount: 0,
      agentsCount: 0,
      errors: [t("error_script_not_found").replace("{path}", scriptPath)],
      warnings: [],
    };
  }

  const args: string[] = [opts.level, opts.tool];
  if (opts.installPath) {
    args.push(opts.installPath);
  }

  try {
    const shell = findShell();
    if (!shell) {
      return {
        success: false,
        pluginId: opts.pluginId,
        skillsCount: 0,
        agentsCount: 0,
        errors: [t("error_no_shell")],
        warnings: [],
      };
    }
    const execOptions: any = {
      cwd,
      timeout: 300000,
      stdio: "pipe",
      input: opts.yes ? "y\n" : undefined,
    };
    
    await execa(shell, [scriptPath, ...args], execOptions);

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
        manifest,
        plugin.externalRepos?.map(r => basename(r.dir))
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
      logger.warn(t("record_write_failed"));
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
      error instanceof Error ? error.message : t("error_unknown");
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
