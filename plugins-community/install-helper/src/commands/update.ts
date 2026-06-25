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
import { selectToolWithDetection } from "../ui/wizard.js";
import { readAllManifests } from "../core/manifest.js";
import { printInstallSummary } from "../ui/display.js";
import { logger, createSpinner } from "../utils/logger.js";
import { t } from "../utils/i18n.js";
import { getConfigRoot, validateTool, validateLevel } from "../utils/paths.js";
import chalk from "chalk";
import type { AITool, InstallLevel } from "../types/index.js";

export async function updateCommand(
  pluginNames: string[],
  options: { tool?: string; level?: string }
): Promise<void> {
  let tool: AITool;
  if (options.tool) {
    tool = validateTool(options.tool);
  } else {
    const detected = await selectToolWithDetection();
    if (detected === "back" || detected === "cancel") return;
    tool = detected;
  }

  const level: InstallLevel = options.level ? validateLevel(options.level) : "project";

  let pluginsToUpdate: string[] = [];

  if (pluginNames.length === 0) {
    const configRoot = getConfigRoot(tool, level);
    const manifests = readAllManifests(configRoot);
    pluginsToUpdate = manifests.map((m) => m.team);
    if (pluginsToUpdate.length === 0) {
      logger.info(t("update_no_plugins"));
      logger.info(t("update_install_hint").replace("{cmd}", chalk.cyan("install-helper install <plugin>")));
      return;
    }
  } else {
    for (const name of pluginNames) {
      const plugin = findPlugin(name);
      if (!plugin) {
        logger.error(`${t("error_plugin_not_found")}: ${name}`);
        return;
      }
      pluginsToUpdate.push(plugin.id);
    }
  }

  const repoManager = createRepositoryManager();
  const updateSpinner = createSpinner(t("update_updating") + "...");
  updateSpinner.start();

  await repoManager.updateRepo();
  const repoPath = repoManager.getRepoPath();
  await repoManager.ensureRepoAndScan();
  updateSpinner.succeed(t("install_repo_ready"));

  const allPlugins = getAllPlugins();
  const results = [];
  const total = pluginsToUpdate.length;

  for (let i = 0; i < pluginsToUpdate.length; i++) {
    const pluginId = pluginsToUpdate[i];
    const plugin = allPlugins.find((p) => p.id === pluginId);
    const displayName = plugin?.displayName || pluginId;
    const progress = `[${i + 1}/${total}]`;

    const pluginSpinner = createSpinner(`${progress} ${t("update_updating")} ${displayName}...`);
    pluginSpinner.start();

    const result = await installPlugin({
      pluginId,
      tool,
      level,
      repoPath,
    });

    if (result.success) {
      pluginSpinner.succeed(
        `${progress} ${displayName} — ${result.skillsCount} skills, ${result.agentsCount} agents`
      );
    } else {
      pluginSpinner.fail(
        `${progress} ${displayName} — ${result.errors.join(", ")}`
      );
    }

    results.push(result);
  }

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
  logger.success(t("update_done"));
}
