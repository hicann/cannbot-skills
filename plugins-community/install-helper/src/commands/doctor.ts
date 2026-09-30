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
import { existsSync, readdirSync, lstatSync, unlinkSync, mkdirSync, readFileSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import { detectTools, getToolDisplayName, getAllTools } from "../core/detector.js";
import { getAllPlugins } from "../core/registry.js";
import { scanInstalled } from "../core/manifest.js";
import {
  hasNodeRuntime,
  isCannbotInstallerReady,
  getCannbotInstallerVersion,
  loadCannbotInstallerConfig,
} from "../core/cannbot-installer.js";
import {
  scanCannbotInstalled,
  countMarkerBlocks,
  findOrphanedDelegatedRecords,
} from "../core/cannbot-registry.js";
import { getConfigRoot, getSkillsRoot, getAgentsDir, getConfigFileName } from "../utils/paths.js";
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

  const checkedRoots = new Set<string>();
  const configRootsToCheck: { configRoot: string; tool: AITool; level: "project" | "global" }[] = [];

  if (installed.length > 0) {
    for (const inst of installed) {
      if (!checkedRoots.has(inst.configRoot)) {
        checkedRoots.add(inst.configRoot);
        configRootsToCheck.push({ configRoot: inst.configRoot, tool: inst.tool, level: inst.level });
      }
    }
  } else {
    const primaryTool = detectedTools[0]?.name || "opencode";
    const configRoot = getConfigRoot(primaryTool, "project");
    configRootsToCheck.push({ configRoot, tool: primaryTool, level: "project" });
  }

  for (const { configRoot, tool, level } of configRootsToCheck) {
    const skillsDir = getSkillsRoot(tool, level);
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
  }

  console.log();
  console.log(chalk.bold(`  ${t("doctor_config")}`));
  for (const { configRoot, tool, level } of configRootsToCheck) {
    const configFile = getConfigFileName(tool);
    const configPath = level === "project"
      ? join(configRoot, "..", configFile)
      : join(configRoot, configFile);
    if (existsSync(configPath)) {
      console.log(chalk.green("  ✓") + ` ${configPath} ${t("doctor_config_exists")}`);
    } else {
      console.log(chalk.dim("  —") + ` ${configPath} ${t("doctor_config_not_exist")}`);
    }
  }

  // cannbot integration checks (Node runtime, managed installer, registry,
  // marker blocks, recorded skills, delegated-record consistency)
  const cannbotResult = doctorCannbotSection(options);
  warnings += cannbotResult.warnings;
  fixes += cannbotResult.fixes;

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

function doctorCannbotSection(options: { fix?: boolean }): { warnings: number; fixes: number } {
  let warnings = 0;
  let fixes = 0;

  console.log();
  console.log(chalk.bold(`  ${t("doctor_cannbot")}`));

  // 1. Node.js runtime (required to delegate to the cannbot installer)
  if (hasNodeRuntime()) {
    console.log(chalk.green("  ✓") + ` ${t("doctor_cannbot_node_ok")}`);
  } else {
    console.log(chalk.yellow("  ⚠") + ` ${t("doctor_cannbot_node_missing")}`);
    warnings++;
  }

  // 2. managed cannbot installer package
  const installerConfig = loadCannbotInstallerConfig();
  if (isCannbotInstallerReady()) {
    const version = getCannbotInstallerVersion() || "?";
    console.log(
      chalk.green("  ✓") +
        ` ${installerConfig.package} v${version} ${chalk.dim(`(${installerConfig.channel})`)}`
    );
  } else {
    console.log(
      chalk.dim("  —") + ` ${installerConfig.package} — ${t("doctor_not_installed")}`
    );
  }

  // 3-5. cannbot registries in the current project
  const cannbotInstalled = scanCannbotInstalled();
  if (cannbotInstalled.length === 0) {
    console.log(chalk.dim("  —") + ` cannbot-plugin.json — ${t("doctor_not_exist")}`);
  } else {
    for (const entry of cannbotInstalled) {
      console.log(
        chalk.green("  ✓") +
          ` ${entry.id} ${chalk.dim(`(${entry.tool}, ${entry.skillsCount} skills, ${entry.agentsCount} agents)`)}`
      );

      // 5. recorded skills actually exist on disk
      const missing = (entry.record.skills || []).filter((rel) => {
        const p = join(entry.target, rel);
        return !existsSync(p) && !lstatTryIsSymlink(p);
      });
      if (missing.length > 0) {
        console.log(
          chalk.yellow("  ⚠") +
            ` ${entry.id}: ${t("doctor_cannbot_missing_skills").replace("{count}", String(missing.length))}`
        );
        warnings++;
      }
    }
  }

  // 4. marker block pairing in the project instructions file
  for (const fileName of ["AGENTS.md", "CLAUDE.md"]) {
    const filePath = join(process.cwd(), fileName);
    if (!existsSync(filePath)) continue;
    try {
      const content = readFileSync(filePath, "utf-8");
      for (const entry of cannbotInstalled) {
        const blocks = countMarkerBlocks(content, entry.id);
        if (blocks > 1) {
          console.log(
            chalk.yellow("  ⚠") +
              ` ${fileName}: ${t("doctor_cannbot_marker_duplicate").replace("{plugin}", entry.id)}`
          );
          warnings++;
        }
      }
      const starts = (content.match(/<!-- cannbot:[\w.-]+:start -->/g) || []).length;
      const ends = (content.match(/<!-- cannbot:[\w.-]+:end -->/g) || []).length;
      if (starts !== ends) {
        console.log(
          chalk.yellow("  ⚠") +
            ` ${fileName}: ${t("doctor_cannbot_marker_unpaired").replace("{start}", String(starts)).replace("{end}", String(ends))}`
        );
        warnings++;
      }
    } catch {
      // unreadable file — skip
    }
  }

  // 6. delegated-record ↔ registry consistency
  const orphans = findOrphanedDelegatedRecords();
  for (const pluginId of orphans) {
    console.log(
      chalk.yellow("  ⚠") +
        ` ${t("doctor_cannbot_orphan_record").replace("{plugin}", pluginId)}`
    );
    warnings++;
    if (options.fix) {
      try {
        unlinkSync(join(homedir(), ".cannbot", "installs", `${pluginId}.json`));
        fixes++;
        console.log(chalk.green("  ✓") + ` ${t("doctor_fixed").replace("{count}", "1")}: ${pluginId}`);
      } catch {
        // ignore
      }
    }
  }

  return { warnings, fixes };
}

function lstatTryIsSymlink(path: string): boolean {
  try {
    return lstatSync(path).isSymbolicLink();
  } catch {
    return false;
  }
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
