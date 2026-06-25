// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { existsSync, mkdirSync, symlinkSync, unlinkSync, readdirSync, rmdirSync, realpathSync, lstatSync } from "fs";
import { join } from "path";
import { select, checkbox, Separator } from "@inquirer/prompts";
import chalk from "chalk";
import Table from "cli-table3";
import type { AITool, InstallLevel } from "../types/index.js";
import { getConfigRoot } from "../utils/paths.js";
import { logger, createSpinner, printBoxTitle, showOperationHints } from "../utils/logger.js";
import { t } from "../utils/i18n.js";
import { getAllCategories, findSkill, getAllSkills } from "./skill-registry.js";
import { addSkillsToRecord, removeSkillsFromRecord, getInstalledSkills } from "./record.js";
import { selectTheme, checkboxTheme } from "../ui/theme.js";

export interface SkillInstallResult {
  skillId: string;
  success: boolean;
  error?: string;
}

export interface SkillUninstallResult {
  skillId: string;
  success: boolean;
  error?: string;
}

export async function installSkills(
  skillIds: string[],
  tool: AITool,
  level: InstallLevel,
  repoPath: string
): Promise<SkillInstallResult[]> {
  const configRoot = getConfigRoot(tool, level);
  const skillsDir = join(configRoot, "skills");
  const installPath = level === "project" ? process.cwd() : configRoot;

  if (!existsSync(skillsDir)) {
    mkdirSync(skillsDir, { recursive: true });
  }

  const results: SkillInstallResult[] = [];
  const installedIds: string[] = [];

  for (const skillId of skillIds) {
    const skill = findSkill(skillId);
    if (!skill) {
      results.push({ skillId, success: false, error: "Skill not found in registry" });
      continue;
    }

    const sourcePath = join(repoPath, skill.source, skill.id);
    const targetPath = join(skillsDir, skillId);

    if (!existsSync(sourcePath)) {
      results.push({ skillId, success: false, error: `Source not found: ${sourcePath}` });
      continue;
    }

    try {
      if (existsSync(targetPath) || isSymlink(targetPath)) {
        unlinkSync(targetPath);
      }
      symlinkSync(realpathSync(sourcePath), targetPath);
      results.push({ skillId, success: true });
      installedIds.push(skillId);
    } catch (error) {
      results.push({ skillId, success: false, error: error instanceof Error ? error.message : "Unknown error" });
    }
  }

  if (installedIds.length > 0) {
    addSkillsToRecord(installedIds, tool, level, installPath);
  }

  return results;
}

export async function uninstallSkills(
  skillIds: string[],
  tool: AITool,
  level: InstallLevel
): Promise<SkillUninstallResult[]> {
  const configRoot = getConfigRoot(tool, level);
  const skillsDir = join(configRoot, "skills");
  const installPath = level === "project" ? process.cwd() : configRoot;

  const results: SkillUninstallResult[] = [];
  const removedIds: string[] = [];

  for (const skillId of skillIds) {
    const targetPath = join(skillsDir, skillId);

    try {
      if (existsSync(targetPath) || isSymlink(targetPath)) {
        unlinkSync(targetPath);
        results.push({ skillId, success: true });
        removedIds.push(skillId);
        logger.step(`  ${t("uninstall_remove_label")}: ${skillId}`);
      } else {
        results.push({ skillId, success: false, error: "Not installed" });
        logger.warn(`${skillId} ${t("uninstall_skill_not_installed")}`);
      }
    } catch (error) {
      results.push({ skillId, success: false, error: error instanceof Error ? error.message : "Unknown error" });
    }
  }

  if (removedIds.length > 0) {
    removeSkillsFromRecord(removedIds, tool, level, installPath);
  }

  // Clean up empty skills directory
  if (existsSync(skillsDir)) {
    try {
      const entries = readdirSync(skillsDir);
      if (entries.length === 0) {
        rmdirSync(skillsDir);
        logger.step(`  ${t("uninstall_clean_empty_dir")}: skills/`);
      }
    } catch {
      // ignore
    }
  }

  return results;
}

export async function interactiveSkillSelect(
  tool: AITool,
  level: InstallLevel
): Promise<string[] | "back" | "cancel"> {
  const BACK = "__back__";
  const CANCEL = "__cancel__";
  const categories = getAllCategories();

  let step = 0;
  let selectedCategoryId = "";

  while (true) {
    switch (step) {
      case 0: {
        printBoxTitle(t("wizard_step_3_skill_title"));

        const categoryChoices: Array<{ name: string; value: string } | Separator> = [
          ...categories.map((cat) => ({
            name: `> ${cat.name} (${cat.skills.length} skills)`,
            value: cat.id,
          })),
          new Separator("──────────────"),
          { name: "<- " + t("wizard_back"), value: BACK },
          { name: "x  " + t("wizard_cancel"), value: CANCEL },
        ];

        showOperationHints();
        selectedCategoryId = await select({
          message: t("skill_select_category"),
          choices: categoryChoices,
          loop: false,
          theme: selectTheme,
          pageSize: 15,
        });

        if (selectedCategoryId === BACK) return "back";
        if (selectedCategoryId === CANCEL) return "cancel";
        step = 1;
        break;
      }
      case 1: {
        const category = categories.find((c) => c.id === selectedCategoryId);
        if (!category) return [];

        printBoxTitle(t("wizard_step_4_skill_title").replace("{category}", category.name));

        const installPath = level === "project" ? process.cwd() : getConfigRoot(tool, level);
        const installedSkills = getInstalledSkills(tool, level, installPath);
        const installedSet = new Set(installedSkills);

        const skillChoices: Array<{ name: string; value: string; checked: boolean } | Separator> = category.skills.map((skill) => {
          const isInstalled = installedSet.has(skill.id);
          const suffix  = isInstalled ? ` [${t("skill_already_installed")}]` : "";
          return {
            name: `${skill.id}${suffix} — ${skill.description}`,
            value: skill.id,
            checked: false,
          };
        });
        showOperationHints(true);
        const selectedSkills = await checkbox({
          message: t("skill_select_items"),
          choices: skillChoices,
          loop: false,
          instructions: false,
          theme: checkboxTheme,
          pageSize: 15,
        });

        const skillIds = selectedSkills;

        // 如果什么都没勾选，显示操作菜单
        if (skillIds.length === 0) {
          const action = await select({
            message: t("wizard_no_selection"),
            choices: [
              new Separator("──────────────"),
              { name: "<- " + t("wizard_back_to_reselect"), value: "back" },
              { name: "x  " + t("wizard_cancel"), value: "cancel" },
            ],
            loop: false,
            theme: selectTheme,
          });
          if (action === "back") { step = 0; break; }
          return "cancel";
        }

        return skillIds;
      }
    }
  }
}

export function listAllSkills(): void {
  const categories = getAllCategories();

  console.log();
  console.log(chalk.bold(`  ${t("skill_list_title")}`));
  console.log(chalk.dim("  " + "─".repeat(60)));

  for (const category of categories) {
    console.log();
    console.log(chalk.bold(`  ${category.name}`) + chalk.dim(` (${category.skills.length} skills)`));

    const table = new Table({
      style: { head: [], border: [] },
      colWidths: [35, 45],
      wordWrap: true,
    });

    for (const skill of category.skills) {
      table.push([
        chalk.cyan(skill.id),
        skill.description,
      ]);
    }

    console.log(table.toString());
  }

  console.log();
  console.log(chalk.dim(`  ${t("skill_total_count").replace("{count}", String(getAllSkills().length))}`));
  console.log();
}

function isSymlink(path: string): boolean {
  try {
    return lstatSync(path).isSymbolicLink();
  } catch {
    return false;
  }
}
