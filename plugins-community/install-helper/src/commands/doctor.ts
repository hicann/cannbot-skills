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
import { existsSync, readdirSync, lstatSync, unlinkSync, mkdirSync } from "fs";
import { join } from "path";
import { detectTools, getToolDisplayName, getAllTools } from "../core/detector.js";
import { getAllPlugins } from "../core/registry.js";
import { scanInstalled } from "../core/manifest.js";
import { getConfigRoot, getSkillsDir, getAgentsDir, getConfigFileName } from "../utils/paths.js";
import { logger } from "../utils/logger.js";
import { t } from "../utils/i18n.js";
import type { AITool } from "../types/index.js";

export async function doctorCommand(options: { fix?: boolean } = {}): Promise<void> {
  console.log();
  console.log(chalk.bold(`  ${t("doctor_title")}`));
  console.log(chalk.dim("  " + "─".repeat(46)));

  let warnings = 0;
  let fixes = 0;

  console.log();
  console.log(chalk.bold(`  ${t("doctor_tools")}`));
  const detectedTools = await detectTools();
  const allTools = getAllTools();

  for (const tool of allTools) {
    const detected = detectedTools.find((d) => d.name === tool);
    if (detected) {
      console.log(
        chalk.green("  ✓") +
          ` ${getToolDisplayName(tool)}${detected.version ? ` v${detected.version}` : ""}`
      );
    } else {
      console.log(chalk.dim("  —") + ` ${getToolDisplayName(tool)} — ${t("doctor_not_installed")}`);
    }
  }

  console.log();
  console.log(chalk.bold(`  ${t("doctor_plugins")}`));
  const plugins = getAllPlugins();
  const installed = scanInstalled();
  const installedMap = new Map(installed.map((p) => [p.id, p]));

  for (const plugin of plugins) {
    const inst = installedMap.get(plugin.id);
    if (inst) {
      console.log(
        chalk.green("  ✓") +
          ` ${plugin.id.padEnd(30)} ${inst.skillsCount} skills, ${inst.agentsCount} agents`
      );
    } else {
      console.log(chalk.dim("  —") + ` ${plugin.id.padEnd(30)} — ${t("doctor_not_installed")}`);
    }
  }

  console.log();
  console.log(chalk.bold(`  ${t("doctor_links")}`));

  const primaryTool = detectedTools[0]?.name || "opencode";
  const configRoot = getConfigRoot(primaryTool, "project");

  const skillsDir = getSkillsDir(configRoot);
  const agentsDir = getAgentsDir(configRoot);

  if (existsSync(skillsDir)) {
    const brokenLinks = checkBrokenLinks(skillsDir);
    if (brokenLinks === 0) {
      const count = readdirSync(skillsDir).length;
      console.log(chalk.green("  ✓") + ` ${skillsDir} — ${t("doctor_links_valid").replace("{count}", String(count))}`);
    } else {
      console.log(chalk.yellow("  ⚠") + ` ${skillsDir} — ${t("doctor_broken_links").replace("{count}", String(brokenLinks))}`);
      warnings++;
      if (options.fix) {
        const fixed = fixBrokenLinks(skillsDir);
        fixes += fixed;
        console.log(chalk.green("  ✓") + ` ${t("doctor_fix_cleaning")}: ${t("doctor_fixed").replace("{count}", String(fixed))}`);
      }
    }
  } else {
    console.log(chalk.dim("  —") + ` ${skillsDir} — ${t("doctor_not_exist")}`);
    if (options.fix) {
      mkdirSync(skillsDir, { recursive: true });
      fixes++;
      console.log(chalk.green("  ✓") + ` ${t("doctor_fix_rebuilding")}: ${t("doctor_created").replace("{path}", skillsDir)}`);
    }
  }

  if (existsSync(agentsDir)) {
    const brokenLinks = checkBrokenLinks(agentsDir);
    if (brokenLinks === 0) {
      const count = readdirSync(agentsDir).length;
      console.log(chalk.green("  ✓") + ` ${agentsDir} — ${t("doctor_links_valid").replace("{count}", String(count))}`);
    } else {
      console.log(chalk.yellow("  ⚠") + ` ${agentsDir} — ${t("doctor_broken_links").replace("{count}", String(brokenLinks))}`);
      warnings++;
      if (options.fix) {
        const fixed = fixBrokenLinks(agentsDir);
        fixes += fixed;
        console.log(chalk.green("  ✓") + ` ${t("doctor_fix_cleaning")}: ${t("doctor_fixed").replace("{count}", String(fixed))}`);
      }
    }
  } else {
    console.log(chalk.dim("  —") + ` ${agentsDir} — ${t("doctor_not_exist")}`);
    if (options.fix) {
      mkdirSync(agentsDir, { recursive: true });
      fixes++;
      console.log(chalk.green("  ✓") + ` ${t("doctor_fix_rebuilding")}: ${t("doctor_created").replace("{path}", agentsDir)}`);
    }
  }

  console.log();
  console.log(chalk.bold(`  ${t("doctor_config")}`));
  const configFile = getConfigFileName(primaryTool);
  const configPath = join(process.cwd(), configFile);
  if (existsSync(configPath)) {
    console.log(chalk.green("  ✓") + ` ${configFile} ${t("doctor_config_exists")}`);
  } else {
    console.log(chalk.dim("  —") + ` ${configFile} ${t("doctor_config_not_exist")}`);
  }

  console.log();
  console.log(chalk.dim("  " + "─".repeat(46)));
  if (options.fix && fixes > 0) {
    console.log(
      `  ${t("doctor_result")}: ${t("doctor_result_with_fix").replace("{warnings}", String(warnings)).replace("{fixes}", String(fixes))}`
    );
  } else {
    console.log(
      `  ${t("doctor_result")}: ${t("doctor_result_without_fix").replace("{warnings}", String(warnings))}`
    );
  }
  console.log();
}

function checkBrokenLinks(dir: string): number {
  let broken = 0;
  try {
    const entries = readdirSync(dir);
    for (const entry of entries) {
      const fullPath = join(dir, entry);
      try {
        const stats = lstatSync(fullPath);
        if (stats.isSymbolicLink()) {
          if (!existsSync(fullPath)) {
            broken++;
          }
        }
      } catch {
        broken++;
      }
    }
  } catch {
    // ignore
  }
  return broken;
}

function fixBrokenLinks(dir: string): number {
  let fixed = 0;
  try {
    const entries = readdirSync(dir);
    for (const entry of entries) {
      const fullPath = join(dir, entry);
      try {
        const stats = lstatSync(fullPath);
        if (stats.isSymbolicLink()) {
          if (!existsSync(fullPath)) {
            unlinkSync(fullPath);
            fixed++;
          }
        }
      } catch {
        // ignore
      }
    }
  } catch {
    // ignore
  }
  return fixed;
}
