// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { createRepositoryManager } from "../core/repository.js";
import { installPlugin } from "../core/installer.js";
import { findPlugin, getAllPlugins } from "../core/registry.js";
import { findSkill, getAllSkills } from "../core/skill-registry.js";
import { installSkills, interactiveSkillSelect, listAllSkills } from "../core/skill-installer.js";
import { selectToolWithDetection } from "../ui/wizard.js";
import { readAllManifests } from "../core/manifest.js";
import { printInstallSummary, printEnhancedSummary } from "../ui/display.js";
import { logger, createSpinner } from "../utils/logger.js";
import { t } from "../utils/i18n.js";
import { join } from "path";
import { addInstalledPlugin } from "../utils/config.js";
import { getConfigRoot, validateTool, validateLevel } from "../utils/paths.js";
import { confirm } from "@inquirer/prompts";
import chalk from "chalk";
import type { AITool, InstallLevel } from "../types/index.js";

export async function installCommand(
  names: string[],
  options: { tool?: string; level?: string; yes?: boolean; all?: boolean; list?: boolean }
): Promise<void> {
  // Handle --list flag
  if (options.list) {
    try {
      const repoManager = createRepositoryManager();
      await repoManager.ensureRepoAndScan();
    } catch {
    }
    listAllSkills();
    return;
  }

  // Handle --all flag: install all available skills
  if (options.all) {
    const scanSpinner = createSpinner(t("loading_skills_list"));
    scanSpinner.start();
    try {
      const scanRepoManager = createRepositoryManager();
      await scanRepoManager.ensureRepoAndScan();
      scanSpinner.succeed(t("loading_skills_list_complete"));
    } catch {
      scanSpinner.warn(t("loading_skills_list_failed"));
    }

    const allSkills = getAllSkills();
    const skillIds = allSkills.map((s) => s.id);

    if (skillIds.length === 0) {
      logger.error(t("install_no_skill_selected"));
      return;
    }

    let tool: AITool;
    if (options.tool) {
      tool = validateTool(options.tool);
    } else {
      const detected = await selectToolWithDetection();
      if (detected === "back" || detected === "cancel") {
        logger.info(t("init_cancelled"));
        return;
      }
      tool = detected;
    }

    const level: InstallLevel = options.level ? validateLevel(options.level) : "project";

    const repoManager = createRepositoryManager();
    const spinner = createSpinner(t("loading_plugin_data"));
    spinner.start();
    let repoPath: string;
    try {
      repoPath = await repoManager.ensureRepo();
      spinner.succeed(t("loading_plugin_data_complete"));
    } catch (error) {
      spinner.fail(t("repo_clone_failed")
        .replace("{error}", error instanceof Error ? error.message : t("error_unknown"))
        .replace("{url}", "")
        .replace("{dir}", ""));
      return;
    }

    logger.info(`${t("skill_all_install_progress").replace("{count}", chalk.bold(String(skillIds.length)))}`);

    const results = await installSkills(skillIds, tool, level, repoPath);

    let successCount = 0;
    let failCount = 0;
    const total = results.length;
    for (let i = 0; i < results.length; i++) {
      const result = results[i];
      const progress = `[${i + 1}/${total}]`;
      if (result.success) { logger.success(`${progress} ${result.skillId}`); successCount++; }
      else { logger.error(`${progress} ${result.skillId}: ${result.error}`); failCount++; }
    }

    const configRoot = getConfigRoot(tool, level);
    logger.blank();
    logger.success(`${t("skill_install_done")}: ${chalk.green(t("result_success_format").replace("{count}", String(successCount)))}, ${failCount > 0 ? chalk.red(t("result_failed_format").replace("{count}", String(failCount))) : chalk.dim(t("result_failed_format").replace("{count}", String(failCount)))}`);
    logger.blank();
    logger.info(`${t("install_to")}: ${chalk.cyan(join(configRoot, "skills"))}`);
    logger.info(`${t("start_to_use").replace("{tool}", chalk.green(tool))}`);
    logger.blank();
    return;
  }

  // If no names provided, enter interactive skill selection
  if (names.length === 0) {
    // Scan repo for dynamic skill discovery
    const scanSpinner = createSpinner(t("loading_skills_list"));
    scanSpinner.start();
    try {
      const scanRepoManager = createRepositoryManager();
      await scanRepoManager.ensureRepoAndScan();
      scanSpinner.succeed(t("loading_skills_list_complete"));
    } catch {
      scanSpinner.warn(t("loading_skills_list_failed"));
    }

    let tool: AITool;
    if (options.tool) {
      tool = validateTool(options.tool);
    } else {
      const detected = await selectToolWithDetection();
      if (detected === "back" || detected === "cancel") {
        logger.info(t("init_cancelled"));
        return;
      }
      tool = detected;
    }

    const level: InstallLevel = options.level ? validateLevel(options.level) : "project";

    let step = 1;
    while (true) {
      switch (step) {
        case 1: {
          const selectedSkills = await interactiveSkillSelect(tool, level);
          if (selectedSkills === "back") {
            const redetected = await selectToolWithDetection();
            if (redetected === "back" || redetected === "cancel") {
              logger.info(t("init_cancelled"));
              return;
            }
            tool = redetected;
            break;
          }
          if (selectedSkills === "cancel") {
            logger.info(t("init_cancelled"));
            return;
          }
          if (selectedSkills.length === 0) {
            logger.info(t("install_no_skill_selected"));
            return;
          }
          names = selectedSkills;

          const repoManager = createRepositoryManager();
          const spinner = createSpinner(t("loading_plugin_data"));
          spinner.start();
          let repoPath: string;
          try {
            repoPath = await repoManager.ensureRepo();
            spinner.succeed(t("loading_plugin_data_complete"));
          } catch (error) {
            spinner.fail(t("repo_clone_failed")
              .replace("{error}", error instanceof Error ? error.message : t("error_unknown"))
              .replace("{url}", "")
              .replace("{dir}", ""));
            return;
          }

          logger.info(`${t("skill_install_progress")} ${t("skill_count_format").replace("{count}", chalk.bold(String(names.length)))}...`);
          const results = await installSkills(names, tool, level, repoPath);

          let successCount = 0;
          let failCount = 0;
          const total = results.length;
          for (let i = 0; i < results.length; i++) {
            const result = results[i];
            const progress = `[${i + 1}/${total}]`;
            if (result.success) { logger.success(`${progress} ${result.skillId}`); successCount++; }
            else { logger.error(`${progress} ${result.skillId}: ${result.error}`); failCount++; }
          }

          const configRoot = getConfigRoot(tool, level);
          logger.blank();
          logger.success(`${t("skill_install_done")}: ${chalk.green(t("result_success_format").replace("{count}", String(successCount)))}, ${failCount > 0 ? chalk.red(t("result_failed_format").replace("{count}", String(failCount))) : chalk.dim(t("result_failed_format").replace("{count}", String(failCount)))}`);
          logger.blank();
          logger.info(`${t("install_to")}: ${chalk.cyan(join(configRoot, "skills"))}`);
          logger.info(`${t("start_to_use").replace("{tool}", chalk.green(tool))}`);
          logger.blank();
          return;
        }
      }
    }
  }

  // Scan repo for dynamic skill/plugin discovery
  const scanSpinner = createSpinner(t("loading_skills_list"));
  scanSpinner.start();
  try {
    const scanRepoManager = createRepositoryManager();
    await scanRepoManager.ensureRepoAndScan();
    scanSpinner.succeed(t("loading_skills_list_complete"));
  } catch {
    scanSpinner.warn(t("loading_skills_list_failed"));
  }

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
    // Not found
    logger.error(`${t("error_plugin_not_found")}: ${name}`);
    const allPlugins = getAllPlugins();
    logger.info(t("available_plugins") + ":");
    for (const p of allPlugins) {
      logger.step(`  ${p.id} (${p.aliases.join(", ")})`);
    }
    logger.info(`${t("available_skills")}: ${chalk.cyan("install-helper install --list")}`);
    return;
  }

  let tool: AITool;
  if (options.tool) {
    tool = validateTool(options.tool);
  } else {
    const detected = await selectToolWithDetection();
    if (detected === "back" || detected === "cancel") {
      logger.info(t("init_cancelled"));
      return;
    }
    tool = detected;
  }

  const level: InstallLevel = options.level ? validateLevel(options.level) : "project";

  // Install plugins
  if (plugins.length > 0) {
    const configRoot = getConfigRoot(tool, level);
    const manifests = readAllManifests(configRoot);
    const installedSet = new Set(manifests.map((m) => m.team));
    const pluginsToInstall: string[] = [];
    const skippedPlugins: string[] = [];

    for (const pluginId of plugins) {
      if (installedSet.has(pluginId) && !options.yes) {
        const plugin = findPlugin(pluginId);
        const displayName = plugin?.displayName || pluginId;
        logger.warn(`${displayName} [${t("install_already_installed")}]`);
        const shouldReinstall = await confirm({
          message: `${t("install_reinstall_confirm")}?`,
          default: false,
        });
        if (shouldReinstall) {
          pluginsToInstall.push(pluginId);
        } else {
          skippedPlugins.push(pluginId);
          logger.info(`${displayName} ${t("install_skip")}`);
        }
      } else {
        pluginsToInstall.push(pluginId);
      }
    }

    if (pluginsToInstall.length > 0) {
      const repoManager = createRepositoryManager();
      const spinner = createSpinner(t("loading_plugin_data"));
      spinner.start();

      let repoPath: string;
      try {
        repoPath = await repoManager.ensureRepo();
        spinner.succeed(t("loading_plugin_data_complete"));
      } catch (error) {
        spinner.fail(t("repo_clone_failed")
          .replace("{error}", error instanceof Error ? error.message : t("error_unknown"))
          .replace("{url}", "")
          .replace("{dir}", ""));
        return;
      }

      const allPlugins = getAllPlugins();
      const results = await installPlugins(
        pluginsToInstall,
        tool,
        level,
        repoPath,
        undefined,
        options.yes
      );

      const summary = results.map((result) => {
        const plugin = allPlugins.find((p) => p.id === result.pluginId);
        return {
          pluginId: result.pluginId,
          displayName: plugin?.displayName || result.pluginId,
          success: result.success,
          skillsCount: result.skillsCount,
          agentsCount: result.agentsCount,
        };
      });

      printInstallSummary(summary);
      printEnhancedSummary(summary, tool, configRoot);
    }
  }

  // Install skills
  if (skills.length > 0) {
    const repoManager = createRepositoryManager();
    const spinner = createSpinner(t("loading_plugin_data"));
    spinner.start();

    let repoPath: string;
    try {
      repoPath = await repoManager.ensureRepo();
      spinner.succeed(t("loading_plugin_data_complete"));
    } catch (error) {
      spinner.fail(t("repo_clone_failed")
        .replace("{error}", error instanceof Error ? error.message : t("error_unknown"))
        .replace("{url}", "")
        .replace("{dir}", ""));
      return;
    }

    logger.info(`${t("skill_install_progress")} ${t("skill_count_format").replace("{count}", chalk.bold(String(skills.length)))}...`);

    const results = await installSkills(skills, tool, level, repoPath);

    let successCount = 0;
    let failCount = 0;
    const total = results.length;

    for (let i = 0; i < results.length; i++) {
      const result = results[i];
      const progress = `[${i + 1}/${total}]`;
      if (result.success) {
        logger.success(`${progress} ${result.skillId}`);
        successCount++;
      } else {
        logger.error(`${progress} ${result.skillId}: ${result.error}`);
        failCount++;
      }
    }

    logger.blank();
    logger.success(`${t("skill_install_done")}: ${chalk.green(t("result_success_format").replace("{count}", String(successCount)))}, ${failCount > 0 ? chalk.red(t("result_failed_format").replace("{count}", String(failCount))) : chalk.dim(t("result_failed_format").replace("{count}", String(failCount)))}`);
    logger.blank();
    logger.info(`${t("install_to")}: ${chalk.cyan(join(getConfigRoot(tool, level), "skills"))}`);
    logger.info(`${t("start_to_use").replace("{tool}", chalk.green(tool))}`);
    logger.blank();
  }
}

async function installPlugins(
  pluginIds: string[],
  tool: AITool,
  level: InstallLevel,
  repoPath: string,
  installPath?: string,
  yes?: boolean
) {
  const results = [];
  const total = pluginIds.length;

  for (let i = 0; i < pluginIds.length; i++) {
    const pluginId = pluginIds[i];
    const progress = `[${i + 1}/${total}]`;
    const spinner = createSpinner(`${progress} ${t("install_progress")} ${pluginId}...`);
    spinner.start();

    const result = await installPlugin({
      pluginId,
      tool,
      level,
      repoPath,
      installPath,
      yes,
    });

    if (result.success) {
      spinner.succeed(
        `${progress} ${pluginId} — ${result.skillsCount} skills, ${result.agentsCount} agents`
      );
      addInstalledPlugin(pluginId);
    } else {
      spinner.fail(`${progress} ${pluginId} — ${result.errors.join(", ")}`);
    }

    results.push(result);
  }
  return results;
}
