// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { select, Separator } from "@inquirer/prompts";
import chalk from "chalk";
import { runWizard, selectToolWithDetection, stepLevel } from "../ui/wizard.js";
import { createRepositoryManager } from "../core/repository.js";
import { installPlugin } from "../core/installer.js";
import { getAllPlugins } from "../core/registry.js";
import { installSkills, interactiveSkillSelect } from "../core/skill-installer.js";
import { printInstallSummary, printEnhancedSummary } from "../ui/display.js";
import { logger, createSpinner, printBanner, printBoxTitle, showOperationHints } from "../utils/logger.js";
import { t } from "../utils/i18n.js";
import { existsSync, readFileSync } from "fs";
import { join } from "path";
import { addInstalledPlugin } from "../utils/config.js";
import { getConfigRoot } from "../utils/paths.js";
import type { AITool, InstallLevel } from "../types/index.js";
import { selectTheme } from "../ui/theme.js";
import { BACK, CANCEL } from "../utils/constants.js";

export async function initCommand(): Promise<void> {
  const cwd = process.cwd();
  if (
    existsSync(join(cwd, "package.json")) &&
    (() => {
      try {
        const pkg = JSON.parse(readFileSync(join(cwd, "package.json"), "utf-8"));
        return pkg.name === "@cannbot-ai/install-helper";
      } catch { return false; }
    })()
  ) {
    logger.warn(t("init_pkg_dir_warn"));
    logger.info(t("init_pkg_dir_hint"));
    logger.info(t("init_pkg_dir_example"));
    return;
  }

  printBanner(t("wizard_title"));

  const spinner = createSpinner(t("loading_skills_list"));
  spinner.start();
  try {
    const repoManager = createRepositoryManager();
    await repoManager.ensureRepoAndScan();
    spinner.succeed(t("loading_skills_list_complete"));
  } catch {
    spinner.warn(t("loading_skills_list_failed"));
  }

  let step = 0;
  let mode: "plugin" | "skill" = "plugin";

  while (true) {
    switch (step) {
      case 0: {
        const result = await selectMode();
        if (result === null) return;
        mode = result;
        step = 1;
        break;
      }
      case 1: {
        if (mode === "plugin") {
          const result = await pluginInstallFlow();
          if (result === "back") { step = 0; break; }
          return;
        } else {
          const result = await skillInstallFlow();
          if (result === "back") { step = 0; break; }
          return;
        }
      }
    }
  }
}

async function selectMode(): Promise<"plugin" | "skill" | null> {
  printBoxTitle(t("wizard_select_mode_title"));

  const choices: Array<{ name: string; value: string } | Separator> = [
    {
      name: `> ${t("wizard_mode_plugin")} — ${t("wizard_mode_plugin_desc")}`,
      value: "plugin",
    },
    {
      name: `> ${t("wizard_mode_skill")} — ${t("wizard_mode_skill_desc")}`,
      value: "skill",
    },
    new Separator("──────────────"),
    { name: "x  " + t("wizard_cancel"), value: CANCEL },
  ];
  showOperationHints();
  const result = await select({
    message: t("wizard_select_mode"),
    choices,
    loop: false,
    theme: selectTheme,
  });
  if (result === CANCEL) return null;
  return result as "plugin" | "skill";
}

async function pluginInstallFlow(): Promise<"done" | "back"> {
  const answers = await runWizard();

  if (!answers.confirmed) {
    if (answers.back) return "back";
    logger.info(t("init_cancelled"));
    return "done";
  }

  const repoManager = createRepositoryManager();
  const spinner = createSpinner(t("loading_plugin_data"));
  spinner.start();

  const repoPath = await repoManager.ensureRepo();
  spinner.succeed(t("loading_plugin_data_complete"));

  const allPlugins = getAllPlugins();
  const selectedPlugins = answers.plugins.map((id) =>
    allPlugins.find((p) => p.id === id)
  );

  const total = answers.plugins.length;
  const results = [];

  for (let i = 0; i < answers.plugins.length; i++) {
    const pluginId = answers.plugins[i];
    const plugin = selectedPlugins[i];
    const displayName = plugin?.displayName || pluginId;
    const progress = `[${i + 1}/${total}]`;

    const pluginSpinner = createSpinner(`${progress} ${t("install_progress")} ${displayName}...`);
    pluginSpinner.start();

    const result = await installPlugin({
      pluginId,
      tool: answers.tool,
      level: answers.level,
      repoPath,
    });

    if (result.success) {
      pluginSpinner.succeed(
        `${progress} ${displayName} — ${result.skillsCount} skills, ${result.agentsCount} agents`
      );
      addInstalledPlugin(pluginId);
    } else {
      pluginSpinner.fail(
        `${progress} ${displayName} — ${result.errors.join(", ")}`
      );
    }

    results.push(result);
  }

  const summary = results.map((result, index) => ({
    pluginId: result.pluginId,
    displayName: selectedPlugins[index]?.displayName || result.pluginId,
    success: result.success,
    skillsCount: result.skillsCount,
    agentsCount: result.agentsCount,
  }));

  printInstallSummary(summary);

  const configRoot = getConfigRoot(answers.tool, answers.level);
  printEnhancedSummary(summary, answers.tool, configRoot);
  return "done";
}

async function skillInstallFlow(): Promise<"done" | "back"> {
  let step = 0;
  let tool: AITool = "opencode";
  let level: InstallLevel = "project";

  while (true) {
    switch (step) {
      case 0: {
        const result = await selectToolWithDetection();
        if (result === "back") return "back";
        if (result === "cancel") {
          logger.info(t("init_cancelled"));
          return "done";
        }
        tool = result;
        step = 1;
        break;
      }
      case 1: {
        const result = await stepLevel();
        if (result === BACK) { step = 0; break; }
        if (result === CANCEL) {
          logger.info(t("init_cancelled"));
          return "done";
        }
        level = result;
        step = 2;
        break;
      }
      case 2: {
        const selectedSkills = await interactiveSkillSelect(tool, level);
        if (selectedSkills === "back") { step = 1; break; }
        if (selectedSkills === "cancel") {
          logger.info(t("init_cancelled"));
          return "done";
        }
        if (selectedSkills.length === 0) {
          logger.info(t("install_no_skill_selected"));
          return "done";
        }

        const repoManager = createRepositoryManager();
        const spinner = createSpinner(t("loading_plugin_data"));
        spinner.start();

        const repoPath = await repoManager.ensureRepo();
        spinner.succeed(t("loading_plugin_data_complete"));

        logger.info(`${t("skill_install_progress")} ${t("skill_count_format").replace("{count}", chalk.bold(String(selectedSkills.length)))}...`);

        const results = await installSkills(selectedSkills, tool, level, repoPath);

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

        const configRoot = getConfigRoot(tool, level);
        logger.blank();
        logger.success(`${t("skill_install_done")}: ${chalk.green(t("result_success_format").replace("{count}", String(successCount)))}, ${failCount > 0 ? chalk.red(t("result_failed_format").replace("{count}", String(failCount))) : chalk.dim(t("result_failed_format").replace("{count}", String(failCount)))}`);
        logger.blank();
        logger.info(`${t("install_to")}: ${chalk.cyan(join(configRoot, "skills"))}`);
        logger.info(`${t("start_to_use").replace("{tool}", chalk.green(tool))}`);
        logger.blank();
        return "done";
      }
    }
  }
}
