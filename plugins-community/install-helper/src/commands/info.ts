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
import { existsSync, readdirSync } from "fs";
import { join } from "path";
import { findPlugin } from "../core/registry.js";
import { createRepositoryManager } from "../core/repository.js";
import { scanInstalled } from "../core/manifest.js";
import { t } from "../utils/i18n.js";

export async function infoCommand(pluginName: string): Promise<void> {
  const plugin = findPlugin(pluginName);
  if (!plugin) {
    console.log(chalk.red(`  ${t("info_not_found")}: ${pluginName}`));
    return;
  }

  const installed = scanInstalled();
  const inst = installed.find((p) => p.id === plugin.id);
  const isInstalled = !!inst;
  const statusText = isInstalled
    ? chalk.green(`✓ ${t("status_installed")}`)
    : chalk.dim(`— ${t("status_not_installed")}`);

  console.log();
  console.log(chalk.bold(`  ${plugin.displayName}`));
  console.log(chalk.dim("  " + "─".repeat(40)));
  console.log();
  console.log(`  ${chalk.dim(t("info_description") + ":")} ${plugin.description}`);
  console.log(`  ${chalk.dim("ID:")} ${plugin.id}`);
  console.log(`  ${chalk.dim(t("info_status") + ":")} ${statusText}`);
  console.log(`  ${chalk.dim(t("info_skills") + ":")} ${inst ? inst.skillsCount : plugin.skills}`);
  console.log(`  ${chalk.dim(t("info_agents") + ":")} ${inst ? inst.agentsCount : plugin.agents}`);
  console.log(`  ${chalk.dim(t("info_aliases") + ":")} ${plugin.aliases.join(", ")}`);
  console.log();

  try {
    const repoManager = createRepositoryManager();
    const repoPath = await repoManager.ensureRepo();
    const quickstartPath = join(repoPath, plugin.dir, "quickstart.md");

    if (existsSync(quickstartPath)) {
      console.log(`  ${chalk.dim(t("info_quickstart") + ":")} ${quickstartPath}`);
    }

    const agentsDir = join(repoPath, plugin.dir, "agents");
    if (existsSync(agentsDir)) {
      const agentFiles = readdirSync(agentsDir).filter((f: string) => f.endsWith(".md"));
      if (agentFiles.length > 0) {
        console.log();
        console.log(`  ${chalk.bold("Agents:")}`);
        for (const agent of agentFiles) {
          console.log(`    • ${agent.replace(".md", "")}`);
        }
      }
    }
  } catch {
    // ignore if repo not available
  }

  console.log();
}
