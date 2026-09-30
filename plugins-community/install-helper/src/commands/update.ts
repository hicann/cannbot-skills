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
import { installPluginRouted } from "../core/cannbot-delegate.js";
import { ensureCannbotDiscovery, ensureCannbotInstaller } from "../core/cannbot-installer.js";
import { scanCannbotInstalled } from "../core/cannbot-registry.js";
import { findPlugin } from "../core/registry.js";
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
  source: "cannbot" | "skills";
}

export async function updateCommand(
  pluginNames: string[],
  options: { tool?: string; level?: string; yes?: boolean }
): Promise<void> {
  // Phase 1: ensureRepoAndScan (discover dynamic plugins + enrich metadata)
  // + cannbot dual-source discovery
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
  await ensureCannbotDiscovery();

  // Phase 2: scanInstalled (legacy manifests) + cannbot registry records
  const installed = scanInstalled();
  const cannbotInstalled = scanCannbotInstalled();

  let targets: UpdateTarget[] = [];

  if (pluginNames.length === 0) {
    if (installed.length === 0 && cannbotInstalled.length === 0) {
      logger.info(t("update_no_plugins"));
      logger.info(t("update_install_hint").replace("{cmd}", chalk.cyan("install-helper install <plugin>")));
      return;
    }
    targets = [
      ...installed.map((p) => ({ pluginId: p.id, tool: p.tool, level: p.level, source: "skills" as const })),
      ...cannbotInstalled.map((e) => ({ pluginId: e.id, tool: e.tool as AITool, level: "project" as InstallLevel, source: "cannbot" as const })),
    ];
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
      // cannbot registry records win for same-name ids (migration policy);
      // legacy resolution must be source-scoped so a same-name cannbot
      // entry never shadows an installed legacy plugin
      const cannbotMatches = cannbotInstalled.filter((e) => e.id === name);
      const legacyPlugin = findPlugin(name, { source: "skills" });
      const anyPlugin = findPlugin(name);

      if (cannbotMatches.length === 0 && !legacyPlugin && !anyPlugin) {
        logger.error(`${t("error_plugin_not_found")}: ${name}`);
        continue;
      }

      const legacyMatches = cannbotMatches.length === 0 && legacyPlugin
        ? installed.filter((p) => p.id === legacyPlugin.id)
        : [];

      const matches = [
        ...cannbotMatches.map((e) => ({ pluginId: e.id, tool: e.tool as AITool, level: "project" as InstallLevel, source: "cannbot" as const })),
        ...legacyMatches.map((p) => ({ pluginId: p.id, tool: p.tool, level: p.level, source: "skills" as const })),
      ];

      if (matches.length > 0) {
        for (const match of matches) {
          if (defaultTool && match.tool !== defaultTool) continue;
          if (defaultLevel && match.level !== defaultLevel) continue;
          targets.push(match);
        }
      } else {
        const displayName = cannbotMatches.length
          ? cannbotMatches[0].id
          : legacyPlugin?.displayName || anyPlugin?.displayName || name;
        logger.warn(`${displayName} ${t("error_not_installed")}, ${t("update_skipped")}`);
      }
    }
  }

  if (targets.length === 0) {
    logger.info(t("update_no_plugins"));
    return;
  }

  // Refresh the managed cannbot installer when cannbot targets are present
  // (update implies pulling the latest bundled plugins).
  if (targets.some((target) => target.source === "cannbot")) {
    try {
      await ensureCannbotInstaller({ force: true });
    } catch (error) {
      logger.warn(
        t("cannbot_installer_failed").replace(
          "{error}",
          error instanceof Error ? error.message : t("error_unknown")
        )
      );
    }
  }

  // Phase 3: reinstall each target (source-scoped lookup so same-name
  // cannbot entries never override a legacy target's metadata or routing)
  const results = [];
  const total = targets.length;

  for (let i = 0; i < targets.length; i++) {
    const target = targets[i];
    const plugin = findPlugin(target.pluginId, { source: target.source });
    const displayName = plugin?.displayName || target.pluginId;
    const progress = `[${i + 1}/${total}]`;

    const pluginSpinner = createSpinner(`${progress} ${t("update_updating")} ${displayName}...`);
    pluginSpinner.start();

    const result = await installPluginRouted({
      pluginId: target.pluginId,
      tool: target.tool,
      level: target.level,
      repoPath,
      yes: options.yes,
      plugin,
      source: target.source,
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
    for (const warning of result.warnings || []) {
      logger.warn(warning);
    }

    results.push(result);
  }

  const summary = results.map((result, index) => {
    const source = targets[index]?.source;
    const plugin = source ? findPlugin(result.pluginId, { source }) : undefined;
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
