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
import { scanInstalled } from "../core/manifest.js";
import { printInstallSummary } from "../ui/display.js";
import { logger, createSpinner } from "../utils/logger.js";
import { t } from "../utils/i18n.js";
import { validateTool, validateLevel } from "../utils/paths.js";
import chalk from "chalk";
import type { AITool, InstallLevel } from "../types/index.js";

interface UpdateTarget {
  pluginId: string;
  tool: AITool;
  level: InstallLevel;
}

export async function updateCommand(
  pluginNames: string[],
  options: { tool?: string; level?: string; yes?: boolean }
): Promise<void> {
  // Phase 1: ensureRepoAndScan (discover dynamic plugins + enrich metadata)
  const repoManager = createRepositoryManager();
  const updateSpinner = createSpinner(t("update_updating") + "...");
  updateSpinner.start();

  let repoPath: string;
  try {
    repoPath = await repoManager.ensureRepoAndScan();
    updateSpinner.succeed(t("install_repo_ready"));
  } catch (error) {
    updateSpinner.fail(t("repo_clone_failed")
      .replace("{error}", error instanceof Error ? error.message : t("error_unknown"))
      .replace("{url}", "")
      .replace("{dir}", ""));
    return;
  }

  // Phase 2: scanInstalled (now includes dynamically discovered plugins)
  const installed = scanInstalled();

  let targets: UpdateTarget[] = [];

  if (pluginNames.length === 0) {
    if (installed.length === 0) {
      logger.info(t("update_no_plugins"));
      logger.info(t("update_install_hint").replace("{cmd}", chalk.cyan("install-helper install <plugin>")));
      return;
    }
    targets = installed.map((p) => ({ pluginId: p.id, tool: p.tool, level: p.level }));
  } else {
    let defaultTool: AITool | undefined;
    let defaultLevel: InstallLevel | undefined;

    if (options.tool) {
      defaultTool = validateTool(options.tool);
    }
    if (options.level) {
      defaultLevel = validateLevel(options.level);
    }

    for (const name of pluginNames) {
      const plugin = findPlugin(name);
      if (!plugin) {
        logger.error(`${t("error_plugin_not_found")}: ${name}`);
        continue;
      }

      const installedForPlugin = installed.filter((p) => p.id === plugin.id);

      if (installedForPlugin.length > 0) {
        for (const inst of installedForPlugin) {
          if (defaultTool && inst.tool !== defaultTool) continue;
          if (defaultLevel && inst.level !== defaultLevel) continue;
          targets.push({ pluginId: plugin.id, tool: inst.tool, level: inst.level });
        }
      } else {
        logger.warn(`${plugin.displayName} ${t("error_not_installed")}, ${t("update_skipped")}`);
      }
    }
  }

  if (targets.length === 0) {
    logger.info(t("update_no_plugins"));
    return;
  }

  // Phase 3: reinstall each target
  const allPlugins = getAllPlugins();
  const results = [];
  const total = targets.length;

  for (let i = 0; i < targets.length; i++) {
    const target = targets[i];
    const plugin = allPlugins.find((p) => p.id === target.pluginId);
    const displayName = plugin?.displayName || target.pluginId;
    const progress = `[${i + 1}/${total}]`;

    const pluginSpinner = createSpinner(`${progress} ${t("update_updating")} ${displayName}...`);
    pluginSpinner.start();

    const result = await installPlugin({
      pluginId: target.pluginId,
      tool: target.tool,
      level: target.level,
      repoPath,
      yes: options.yes,
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
