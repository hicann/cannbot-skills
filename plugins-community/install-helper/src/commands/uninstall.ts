// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { existsSync, unlinkSync, rmdirSync, readdirSync, lstatSync } from "fs";
import chalk from "chalk";
import { findPlugin } from "../core/registry.js";
import { findSkill } from "../core/skill-registry.js";
import { uninstallSkills } from "../core/skill-installer.js";
import { readRecord, deleteRecord } from "../core/record.js";
import { removeInstalledPlugin } from "../utils/config.js";
import { logger } from "../utils/logger.js";
import { t } from "../utils/i18n.js";
import type { AITool, InstallLevel } from "../types/index.js";
import { findBackups, restoreBackup, deleteBackup, getAgentsFileName } from "../core/backup.js";
import { showRestorePrompt } from "../ui/backup-prompts.js";
import { getConfigRoot, validateTool, validateLevel } from "../utils/paths.js";

export async function uninstallCommand(
  names: string[],
  options: { tool?: string; level?: string }
): Promise<void> {
  const tool: AITool = options.tool ? validateTool(options.tool) : "opencode";
  const level: InstallLevel = options.level ? validateLevel(options.level) : "project";

  // Classify names into plugins and skills
  const plugins: string[] = [];
  const skills: string[] = [];

  for (const name of names) {
    const plugin = findPlugin(name);
    if (plugin) {
      plugins.push(plugin.id);
      continue;
    }
    const skill = findSkill(name);
    if (skill) {
      skills.push(skill.id);
      continue;
    }
    logger.error(`${t("uninstall_not_found")}: ${name}`);
  }

  // Uninstall plugins
  for (const pluginId of plugins) {
    const plugin = findPlugin(pluginId);
    if (!plugin) continue;

    const record = readRecord(plugin.id);
    if (!record) {
      logger.warn(`${plugin.displayName} ${t("uninstall_no_record")}`);
      logger.info(`${t("uninstall_no_record_hint")}`);
      continue;
    }

    logger.info(`${t("uninstall_in_progress")} ${plugin.displayName}...`);

    let removedFiles = 0;
    let removedDirs = 0;

    for (const filePath of record.files) {
      try {
        if (existsSync(filePath) || isSymlink(filePath)) {
          unlinkSync(filePath);
          const name = filePath.split("/").pop() || filePath;
          logger.step(`  ${t("uninstall_remove_label")}: ${name}`);
          removedFiles++;
        }
      } catch {
        // ignore
      }
    }

    const configRoot = getConfigRoot(record.tool, record.level);
    const sortedDirs = [...record.directories].sort((a, b) => b.length - a.length);
    for (const dirPath of sortedDirs) {
      try {
        // Skip configRoot - it's the AI tool's config directory, not created by us
        if (dirPath === configRoot) {
          continue;
        }
        if (existsSync(dirPath)) {
          const entries = readdirSync(dirPath);
          if (entries.length === 0) {
            rmdirSync(dirPath);
            const name = dirPath.split("/").pop() || dirPath;
            logger.step(`  ${t("uninstall_clean_empty_dir")}: ${name}/`);
            removedDirs++;
          }
        }
      } catch {
        // ignore
      }
    }

    deleteRecord(plugin.id);
    removeInstalledPlugin(plugin.id);

    const backups = findBackups(configRoot);
    const otherBackups = backups.filter((b) => b.pluginId !== pluginId);

    if (otherBackups.length > 0) {
      const choice = await showRestorePrompt(otherBackups);
      if (choice !== "none") {
        const restored = restoreBackup(choice, configRoot, record.tool);
        if (restored) {
          const backupInfo = otherBackups.find((b) => b.filePath === choice);
          logger.success(`${t("backup_restore_success")}: ${backupInfo?.pluginName || ""}`);
          deleteBackup(choice);
        }
      }
    }

    const currentBackups = backups.filter((b) => b.pluginId === pluginId);
    for (const backup of currentBackups) {
      deleteBackup(backup.filePath);
    }

    const totalRemoved = removedFiles + removedDirs;
    if (totalRemoved > 0) {
      logger.success(`${t("uninstall_removed")} ${plugin.displayName}（${t("uninstall_remove_file_format").replace("{count}", String(removedFiles))}，${t("uninstall_remove_dir_format").replace("{count}", String(removedDirs))}）`);
    } else {
      logger.warn(`${plugin.displayName}${t("uninstall_files_gone")}`);
    }
  }

  // Uninstall skills
  if (skills.length > 0) {
    logger.info(`${t("uninstall_skills_in_progress")} ${t("skill_count_format").replace("{count}", String(skills.length))}...`);

    const results = await uninstallSkills(skills, tool, level);

    let successCount = 0;
    let failCount = 0;

    for (const result of results) {
      if (result.success) {
        successCount++;
      } else {
        failCount++;
      }
    }

    logger.blank();
    logger.success(`${t("skill_uninstall_done")}: ${chalk.green(t("result_success_format").replace("{count}", String(successCount)))}, ${failCount > 0 ? chalk.red(t("result_failed_format").replace("{count}", String(failCount))) : chalk.dim(t("result_failed_format").replace("{count}", String(failCount)))}`);
    logger.blank();
  }
}

function isSymlink(path: string): boolean {
  try {
    return lstatSync(path).isSymbolicLink();
  } catch {
    return false;
  }
}
