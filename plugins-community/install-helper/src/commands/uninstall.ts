// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { existsSync, unlinkSync, rmdirSync, readdirSync, readFileSync } from "fs";
import { resolve, basename, sep, join } from "path";
import chalk from "chalk";
import { select, checkbox, Separator, confirm } from "@inquirer/prompts";
import { findPlugin } from "../core/registry.js";
import { findSkill } from "../core/skill-registry.js";
import { uninstallSkills, interactiveSkillUnselect } from "../core/skill-installer.js";
import { readRecord, deleteRecord, getInstalledSkills, getLastBatchSkills, scanInstalledFiles } from "../core/record.js";
import { removeInstalledPlugin } from "../utils/config.js";
import { isSymlink } from "../utils/fs-helpers.js";
import { logger, printBoxTitle, showOperationHints } from "../utils/logger.js";
import { t } from "../utils/i18n.js";
import type { AITool, InstallLevel, CannbotManifest } from "../types/index.js";
import { findBackups, restoreBackup, deleteBackup } from "../core/backup.js";
import { showRestorePrompt } from "../ui/backup-prompts.js";
import { getConfigRoot, validateTool, validateLevel, VALID_TOOLS } from "../utils/paths.js";
import { scanInstalled } from "../core/manifest.js";
import { selectTheme, checkboxTheme } from "../ui/theme.js";
import { selectToolWithDetection } from "../ui/wizard.js";

export async function uninstallCommand(
  names: string[],
  options: { tool?: string; level?: string; yes?: boolean; all?: boolean; recent?: boolean }
): Promise<void> {
  const userSpecifiedTool = !!options.tool;
  const userSpecifiedLevel = !!options.level;
  const tool: AITool = options.tool ? validateTool(options.tool) : "opencode";
  const level: InstallLevel = options.level ? validateLevel(options.level) : "project";

  // --all mode: uninstall all installed skills + plugins
  if (options.all) {
    await uninstallAll(tool, level, options.yes || false);
    return;
  }

  // --recent mode: uninstall last batch of skills
  if (options.recent) {
    await uninstallRecent(tool, level, options.yes || false);
    return;
  }

  // Interactive mode: no names provided
  if (names.length === 0) {
    await uninstallInteractive(tool, level, userSpecifiedTool, userSpecifiedLevel);
    return;
  }

  // Name-driven mode (existing behavior)
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

  await uninstallPlugins(plugins, options.yes || false);
  await uninstallSkillsByName(skills, tool, level);
}

async function uninstallAll(tool: AITool, level: InstallLevel, yes: boolean): Promise<void> {
  const installPath = level === "project" ? process.cwd() : getConfigRoot(tool, level);
  const installedSkills = getInstalledSkills(tool, level, installPath, !yes);
  const installedPlugins = scanInstalled().filter((p) => p.tool === tool && p.level === level);

  const totalItems = installedSkills.length + installedPlugins.length;
  if (totalItems === 0) {
    logger.warn(t("uninstall_no_installed"));
    return;
  }

  logger.info(t("uninstall_all_summary")
    .replace("{skills}", String(installedSkills.length))
    .replace("{plugins}", String(installedPlugins.length)));
  logger.blank();

  for (const s of installedSkills) {
    logger.step(`  • ${chalk.cyan(s)}`);
  }
  for (const p of installedPlugins) {
    logger.step(`  • ${chalk.cyan(p.displayName)} ${chalk.dim(`(${p.id})`)}`);
  }
  logger.blank();

  if (!yes) {
    const confirmed = await confirm({
      message: t("uninstall_confirm_prompt").replace("{count}", chalk.bold(String(totalItems))),
      default: false,
    });
    if (!confirmed) {
      logger.info(t("init_cancelled"));
      return;
    }
  }

  // Uninstall skills
  if (installedSkills.length > 0) {
    logger.info(`${t("uninstall_skills_in_progress")} ${t("skill_count_format").replace("{count}", String(installedSkills.length))}...`);
    await uninstallSkills(installedSkills, tool, level);
  }

  // Uninstall plugins
  if (installedPlugins.length > 0) {
    logger.info(`${t("uninstall_plugin_batch_progress").replace("{count}", String(installedPlugins.length))}...`);
    for (const inst of installedPlugins) {
      await uninstallPluginById(inst.id, true);
    }
  }

  logger.blank();
  logger.success(t("uninstall_all_done"));
  logger.blank();
}

async function uninstallRecent(tool: AITool, level: InstallLevel, yes: boolean): Promise<void> {
  const installPath = level === "project" ? process.cwd() : getConfigRoot(tool, level);
  const lastBatch = getLastBatchSkills(tool, level, installPath);

  if (!lastBatch || lastBatch.length === 0) {
    logger.warn(t("uninstall_recent_no_history"));
    return;
  }

  logger.info(t("uninstall_recent_progress").replace("{count}", chalk.bold(String(lastBatch.length))));
  logger.blank();
  for (const s of lastBatch) {
    logger.step(`  • ${chalk.cyan(s)}`);
  }
  logger.blank();

  if (!yes) {
    const confirmed = await confirm({
      message: t("uninstall_confirm_prompt").replace("{count}", chalk.bold(String(lastBatch.length))),
      default: false,
    });
    if (!confirmed) {
      logger.info(t("init_cancelled"));
      return;
    }
  }

  logger.info(`${t("uninstall_skills_in_progress")} ${t("skill_count_format").replace("{count}", String(lastBatch.length))}...`);
  await uninstallSkills(lastBatch, tool, level);
  logger.blank();
  logger.success(t("skill_uninstall_done"));
  logger.blank();
}

async function uninstallInteractive(
  defaultTool: AITool,
  defaultLevel: InstallLevel,
  userSpecifiedTool: boolean,
  userSpecifiedLevel: boolean
): Promise<void> {
  let tool = defaultTool;
  let level = defaultLevel;

  // Scan all tool×level combinations for installed content
  const allInstalledPlugins = scanInstalled();
  const toolsWithPlugins = new Set(allInstalledPlugins.map((p) => `${p.tool}:${p.level}`));

  // Check which tool×level combos have installed skills
  const toolsWithSkills = new Set<string>();
  for (const toolName of ["opencode", "claude", "trae", "cursor", "copilot", "codearts"] as AITool[]) {
    for (const lvl of ["project", "global"] as InstallLevel[]) {
      const ip = lvl === "project" ? process.cwd() : getConfigRoot(toolName, lvl);
      const skills = getInstalledSkills(toolName, lvl, ip);
      if (skills.length > 0) {
        toolsWithSkills.add(`${toolName}:${lvl}`);
      }
    }
  }

  const hasSkills = toolsWithSkills.size > 0;
  const hasPlugins = toolsWithPlugins.size > 0;

  if (!hasSkills && !hasPlugins) {
    logger.warn(t("uninstall_no_installed"));
    return;
  }

  // Step: select type (skip if only one type has content)
  let uninstallType: string;
  if (hasSkills && hasPlugins) {
    printBoxTitle(t("uninstall_interactive_title"));
    const skillCount = [...toolsWithSkills].reduce((sum, key) => {
      const [toolName, lvl] = key.split(":") as [AITool, InstallLevel];
      const ip = lvl === "project" ? process.cwd() : getConfigRoot(toolName, lvl);
      return sum + getInstalledSkills(toolName, lvl, ip).length;
    }, 0);
    const pluginCount = allInstalledPlugins.length;
    const typeChoices: Array<{ name: string; value: string } | Separator> = [
      { name: `> ${t("uninstall_type_skill")} (${skillCount})`, value: "skill" },
      { name: `> ${t("uninstall_type_plugin")} (${pluginCount})`, value: "plugin" },
      new Separator("──────────────"),
      { name: "x  " + t("wizard_cancel"), value: "cancel" },
    ];
    showOperationHints();
    const choice = await select({
      message: t("uninstall_select_type"),
      choices: typeChoices,
      loop: false,
      theme: selectTheme,
    });
    if (choice === "cancel") {
      logger.info(t("init_cancelled"));
      return;
    }
    uninstallType = choice;
  } else if (hasSkills) {
    uninstallType = "skill";
  } else {
    uninstallType = "plugin";
  }

  // Find which tools have content for the selected type
  const relevantKeys = uninstallType === "skill" ? toolsWithSkills : toolsWithPlugins;
  const toolsWithContent = new Set<AITool>();
  for (const key of relevantKeys) {
    toolsWithContent.add(key.split(":")[0] as AITool);
  }

  // Select tool (skip if only one tool has content, or respect --tool if user specified it)
  if (toolsWithContent.size === 1) {
    tool = [...toolsWithContent][0];
  } else if (userSpecifiedTool && toolsWithContent.has(defaultTool)) {
    tool = defaultTool;
  } else {
    const detected = await selectToolWithDetection();
    if (detected === "back" || detected === "cancel") {
      logger.info(t("init_cancelled"));
      return;
    }
    tool = detected;
  }

  // Find which levels have content for the selected tool
  const levelsWithContent = new Set<InstallLevel>();
  for (const key of relevantKeys) {
    const [toolName, lvl] = key.split(":") as [AITool, InstallLevel];
    if (toolName === tool) {
      levelsWithContent.add(lvl);
    }
  }

  // Select level (skip if only one level has content, or respect --level if user specified it)
  if (levelsWithContent.size === 1) {
    level = [...levelsWithContent][0];
  } else if (userSpecifiedLevel && levelsWithContent.has(defaultLevel)) {
    level = defaultLevel;
  } else {
    const { stepLevel } = await import("../ui/wizard.js");
    const levelResult = await stepLevel();
    if (levelResult === "__back__" || levelResult === "__cancel__") {
      logger.info(t("init_cancelled"));
      return;
    }
    level = levelResult as InstallLevel;
  }

  if (uninstallType === "skill") {
    await interactiveSkillUninstall(tool, level);
  } else {
    await interactivePluginUninstall(tool, level);
  }
}

async function interactiveSkillUninstall(tool: AITool, level: InstallLevel): Promise<void> {
  const result = await interactiveSkillUnselect(tool, level);

  if (result === "back" || result === "cancel") {
    if (result === "cancel") logger.info(t("init_cancelled"));
    return;
  }

  const { toUninstall } = result;
  logger.info(`${t("uninstall_skills_in_progress")} ${t("skill_count_format").replace("{count}", String(toUninstall.length))}...`);
  await uninstallSkills(toUninstall, tool, level);
  logger.blank();
  logger.success(t("skill_uninstall_done"));
  logger.blank();
}

async function interactivePluginUninstall(tool: AITool, level: InstallLevel): Promise<void> {
  const installed = scanInstalled().filter((p) => p.tool === tool && p.level === level);

  if (installed.length === 0) {
    logger.warn(t("uninstall_no_installed_plugins"));
    return;
  }

  const pluginChoices: Array<{ name: string; value: string; checked: boolean; description?: string } | Separator> = [
    ...installed.map((inst) => ({
      name: `${chalk.cyan(inst.displayName)} ${chalk.dim(`(${inst.id})`)}`,
      value: inst.id,
      checked: false,
      description: `${inst.skillsCount} skills, ${inst.agentsCount} agents`,
    })),
  ];

  printBoxTitle(t("uninstall_interactive_title"));
  logger.info(chalk.dim(`  ${t("uninstall_checkbox_hint_select")}`));
  logger.blank();
  showOperationHints(true);

  const selected = await checkbox({
    message: t("uninstall_select_plugins"),
    choices: pluginChoices,
    loop: false,
    instructions: false,
    theme: checkboxTheme,
    pageSize: 15,
  });

  const toUninstall = installed.filter((inst) => selected.includes(inst.id));

  if (toUninstall.length === 0) {
    logger.info(t("uninstall_nothing_to_uninstall"));
    return;
  }

  const confirmed = await confirm({
    message: t("uninstall_confirm_prompt").replace("{count}", chalk.bold(String(toUninstall.length))),
    default: false,
  });

  if (!confirmed) {
    logger.info(t("init_cancelled"));
    return;
  }

  logger.info(`${t("uninstall_plugin_batch_progress").replace("{count}", String(toUninstall.length))}...`);
  for (const inst of toUninstall) {
    await uninstallPluginById(inst.id, true);
  }
  logger.blank();
  logger.success(t("uninstall_all_done"));
  logger.blank();
}

async function uninstallPlugins(pluginIds: string[], batchMode: boolean): Promise<void> {
  for (const pluginId of pluginIds) {
    await uninstallPluginById(pluginId, batchMode);
  }
}

async function uninstallPluginById(pluginId: string, batchMode: boolean): Promise<void> {
  const plugin = findPlugin(pluginId);
  if (!plugin) return;

  let record = readRecord(plugin.id);

  if (!record) {
    // Fallback: scan all tool×level for manifest-based cleanup
    for (const toolName of VALID_TOOLS) {
      for (const lvl of ["project", "global"] as InstallLevel[]) {
        const configRoot = getConfigRoot(toolName, lvl);
        const manifestPath = join(configRoot, `${pluginId}-manifest.json`);
        if (!existsSync(manifestPath)) continue;

        let manifest: CannbotManifest | null = null;
        try {
          manifest = JSON.parse(readFileSync(manifestPath, "utf-8")) as CannbotManifest;
        } catch {
          continue;
        }

        const installPath = lvl === "project" ? process.cwd() : configRoot;
        record = scanInstalledFiles(
          pluginId, plugin.displayName, toolName, lvl,
          installPath, configRoot, manifest,
          plugin.externalRepos?.map((r) => basename(r.dir)),
          plugin.configRootConfigLink
        );
        break;
      }
      if (record) break;
    }
  }

  if (!record) {
    logger.warn(`${plugin.displayName} ${t("uninstall_no_record")}`);
    logger.info(`${t("uninstall_no_record_hint")}`);
    return;
  }

  logger.info(`${t("uninstall_in_progress")} ${plugin.displayName}...`);

  const configRoot = getConfigRoot(record.tool, record.level);
  const allowedBases = [resolve(configRoot), resolve(record.installPath)];
  const isSafePath = (p: string): boolean => {
    const resolved = resolve(p);
    return allowedBases.some(base => resolved === base || resolved.startsWith(base + sep));
  };

  let removedFiles = 0;
  let removedDirs = 0;

  // Add configRoot-level config file if not already in record (backward compat)
  const configFileName = record.tool === "claude" ? "CLAUDE.md" : "AGENTS.md";
  const configRootConfigPath = join(configRoot, configFileName);
  const installPathConfigPath = join(record.installPath, configFileName);
  const extraFiles: string[] = [];
  if (record.level === "project" &&
      configRootConfigPath !== installPathConfigPath &&
      !record.files.includes(configRootConfigPath) &&
      (existsSync(configRootConfigPath) || isSymlink(configRootConfigPath))) {
    extraFiles.push(configRootConfigPath);
  }

  const allFiles = [...record.files, ...extraFiles];

  for (const filePath of allFiles) {
    if (!isSafePath(filePath)) continue;
    try {
      if (existsSync(filePath) || isSymlink(filePath)) {
        unlinkSync(filePath);
        const name = basename(filePath);
        logger.step(`  ${t("uninstall_remove_label")}: ${name}`);
        removedFiles++;
      }
    } catch {
      // ignore
    }
  }

  const sortedDirs = [...record.directories].filter(isSafePath).sort((a, b) => b.length - a.length);
  for (const dirPath of sortedDirs) {
    try {
      if (resolve(dirPath) === resolve(configRoot)) {
        continue;
      }
      if (existsSync(dirPath)) {
        const entries = readdirSync(dirPath);
        if (entries.length === 0) {
          rmdirSync(dirPath);
          const name = basename(dirPath);
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

  if (!batchMode && otherBackups.length > 0) {
    const choice = await showRestorePrompt(otherBackups);
    if (choice !== "none") {
      const restored = restoreBackup(choice, configRoot, record.tool);
      if (restored) {
        const backupInfo = otherBackups.find((b) => b.filePath === choice);
        logger.success(`${t("backup_restore_success")}: ${backupInfo?.pluginName || ""}`);
        deleteBackup(choice);
      }
    }
  } else if (batchMode && otherBackups.length > 0) {
    logger.info(t("uninstall_plugin_batch_skip_restore"));
  }

  const currentBackups = backups.filter((b) => b.pluginId === pluginId);
  for (const backup of currentBackups) {
    deleteBackup(backup.filePath);
  }

  const totalRemoved = removedFiles + removedDirs;
  if (totalRemoved > 0) {
    logger.success(t("uninstall_summary_format").replace("{name}", plugin.displayName).replace("{files}", String(removedFiles)).replace("{dirs}", String(removedDirs)));
  } else {
    logger.warn(`${plugin.displayName}${t("uninstall_files_gone")}`);
  }
}

async function uninstallSkillsByName(skills: string[], tool: AITool, level: InstallLevel): Promise<void> {
  if (skills.length === 0) return;

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
