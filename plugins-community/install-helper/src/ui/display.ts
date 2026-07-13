// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import chalk from "chalk";
import Table from "cli-table3";
import type { PluginEntry, InstalledPlugin, AITool } from "../types/index.js";
import { t } from "../utils/i18n.js";
import { getToolDisplayName } from "../core/detector.js";
import { findPlugin } from "../core/registry.js";

export function printPluginList(
  plugins: PluginEntry[],
  installed: Map<string, InstalledPlugin>
): void {
  const table = new Table({
    head: [
      chalk.cyan(t("list_id")),
      chalk.cyan(t("list_name")),
      chalk.cyan(t("list_status")),
      chalk.cyan(t("list_description")),
    ],
    style: { head: [], border: [] },
    colWidths: [6, 32, 8, 50],
    wordWrap: true,
  });

  plugins.forEach((plugin, index) => {
    const isInstalled = installed.has(plugin.id);
    const status = isInstalled
      ? chalk.green(`✓ ${t("status_installed")}`)
      : chalk.dim(`—`);

    table.push([
      String(index + 1),
      plugin.displayName,
      status,
      plugin.description || chalk.dim("—"),
    ]);
  });

  console.log();
  const countSuffix = t("count_suffix");
  console.log(chalk.bold(`  ${t("list_title")} (${plugins.length}${countSuffix ? " " + countSuffix : ""})`));
  console.log(table.toString());
  console.log();
}

export function printInstallSummary(
  results: Array<{
    pluginId: string;
    displayName: string;
    success: boolean;
    skillsCount: number;
    agentsCount: number;
  }>
): void {
  const successCount = results.filter((r) => r.success).length;
  const totalCount = results.length;

  console.log();
  console.log(
    chalk.bold(
      `  ${t("install_done")}! ${successCount}/${totalCount} ${t("install_success")}`
    )
  );
  console.log();

  for (const result of results) {
    if (result.success) {
      console.log(
        chalk.green("  ✓") +
          ` ${result.displayName} (${result.skillsCount} skills, ${result.agentsCount} agents)`
      );
    } else {
      console.log(chalk.red("  ✗") + ` ${result.displayName}`);
    }
  }
  console.log();
}

export function printEnhancedSummary(
  results: Array<{
    pluginId: string;
    displayName: string;
    success: boolean;
    skillsCount: number;
    agentsCount: number;
  }>,
  tool: AITool,
  configRoot: string
): void {
  const successResults = results.filter((r) => r.success);
  const totalSkills = successResults.reduce((sum, r) => sum + r.skillsCount, 0);
  const totalAgents = successResults.reduce((sum, r) => sum + r.agentsCount, 0);

  if (successResults.length === 0) {
    return;
  }

  const toolName = getToolDisplayName(tool);
  const countSuffix = t("count_suffix");
  const lines = [
    chalk.bold(`  ${t("install_done")}!`),
    "",
    `  ${chalk.dim(t("install_installed_to") + ":")} ${configRoot}`,
    `  ${chalk.dim(t("install_skills_count") + ":")} ${totalSkills}${countSuffix ? " " + countSuffix : ""}  ${chalk.dim("|")}  ${chalk.dim(t("install_agents_count") + ":")} ${totalAgents}${countSuffix ? " " + countSuffix : ""}`,
    "",
    chalk.bold(`  ${t("install_enhanced_next_steps")}:`),
    `    ${chalk.cyan("1.")} ${t("install_enhanced_launch")}: ${chalk.green(toolName.toLowerCase())}`,
    `    ${chalk.cyan("2.")} ${t("install_enhanced_try")}: ${chalk.green(t("try_prompt"))}`,
    `    ${chalk.cyan("3.")} ${t("install_enhanced_more")}: ${chalk.green("install-helper list")}`,
    `    ${chalk.cyan("4.")} ${t("install_enhanced_check")}: ${chalk.green("install-helper doctor")}`,
    "",
  ];

  if (successResults.length === 1) {
    const plugin = findPlugin(successResults[0].pluginId);
    const docsPath = plugin ? `${plugin.dir}/quickstart.md` : `${successResults[0].pluginId}/quickstart.md`;
    lines.push(
      `  ${chalk.dim(t("install_enhanced_docs"))}: ${chalk.dim(docsPath)}`
    );
  }

  console.log();
  for (const line of lines) {
    console.log(line);
  }
  console.log();
}
