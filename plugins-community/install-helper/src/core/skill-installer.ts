// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { existsSync, mkdirSync, symlinkSync, unlinkSync, readdirSync, rmSync, realpathSync, readlinkSync } from "fs";
import { join, dirname, isAbsolute } from "path";
import { select, checkbox, Separator } from "@inquirer/prompts";
import chalk from "chalk";
import Table from "cli-table3";
import type { AITool, InstallLevel } from "../types/index.js";
import { getConfigRoot } from "../utils/paths.js";
import { isSymlink } from "../utils/fs-helpers.js";
import { BACK, CANCEL } from "../utils/constants.js";
import { logger, printBoxTitle, showOperationHints } from "../utils/logger.js";
import { t } from "../utils/i18n.js";
import { getAllCategories, findSkill, getAllSkills } from "./skill-registry.js";
import { addSkillsToRecord, removeSkillsFromRecord, getInstalledSkills } from "./record.js";
import { selectTheme, checkboxTheme } from "../ui/theme.js";

interface SkillInstallResult {
  skillId: string;
  success: boolean;
  error?: string;
}

interface SkillUninstallResult {
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
      results.push({ skillId, success: false, error: t("error_skill_not_found") });
      continue;
    }

    const sourcePath = resolveSkillSourcePath(skill, repoPath);
    const targetPath = join(skillsDir, skillId);

    if (!existsSync(sourcePath)) {
      results.push({ skillId, success: false, error: t("error_source_not_found").replace("{path}", sourcePath) });
      continue;
    }

    try {
      const resolvedSource = realpathSync(sourcePath);

      if (isSymlink(targetPath)) {
        try {
          const currentTarget = readlinkSync(targetPath);
          const normCurrent = currentTarget.replace(/\\/g, "/").toLowerCase();
          const normSource = resolvedSource.replace(/\\/g, "/").toLowerCase();
          if (normCurrent === normSource) {
            results.push({ skillId, success: true });
            installedIds.push(skillId);
            continue;
          }
        } catch {
          results.push({ skillId, success: true });
          installedIds.push(skillId);
          continue;
        }
      }

      if (existsSync(targetPath) || isSymlink(targetPath)) {
        unlinkSync(targetPath);
      }
      symlinkSync(resolvedSource, targetPath);
      results.push({ skillId, success: true });
      installedIds.push(skillId);
    } catch (error) {
      let msg = error instanceof Error ? error.message : t("error_unknown");
      if (process.platform === "win32" && msg.includes("EPERM")) {
        msg = t("error_symlink_windows");
      }
      results.push({ skillId, success: false, error: msg });
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
        results.push({ skillId, success: false, error: t("error_not_installed") });
        logger.warn(`${skillId} ${t("uninstall_skill_not_installed")}`);
      }
    } catch (error) {
      results.push({ skillId, success: false, error: error instanceof Error ? error.message : t("error_unknown") });
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
        rmSync(skillsDir, { recursive: true });
        logger.step(`  ${t("uninstall_clean_empty_dir")}: skills/`);
      }
    } catch {
      // ignore
    }
  }

  return results;
}

const SELECT_ALL = "__select_all__";
const SELECT_ALL_CATEGORY = "__select_all_category__";
const CONTINUE_SELECT = "__continue_select__";

export async function interactiveSkillSelect(
  tool: AITool,
  level: InstallLevel
): Promise<string[] | "back" | "cancel"> {
  const categories = getAllCategories();
  const allSkillIds = getAllSkills().map((s) => s.id);

  let accumulated: string[] = [];
  let step = 0;
  let selectedCategoryId = "";

  while (true) {
    switch (step) {
      case 0: {
        const titleSuffix = accumulated.length > 0
          ? `${t("wizard_step_3_skill_title")}（${t("skill_count_format").replace("{count}", String(accumulated.length))}）`
          : t("wizard_step_3_skill_title");
        printBoxTitle(titleSuffix);

        const categoryChoices: Array<{ name: string; value: string } | Separator> = [
          {
            name: `>> ${t("skill_select_all").replace("{count}", String(allSkillIds.length))}`,
            value: SELECT_ALL,
          },
          new Separator("──────────────"),
          ...categories.map((cat) => ({
            name: `> ${cat.name} (${cat.skills.length} skills)`,
            value: cat.id,
          })),
          new Separator("──────────────"),
          ...(accumulated.length > 0
            ? [{ name: `>> ${t("skill_confirm_install_count").replace("{count}", String(accumulated.length))}`, value: "install" } as { name: string; value: string }]
            : []),
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
        if (selectedCategoryId === SELECT_ALL) {
          return allSkillIds;
        }
        if (selectedCategoryId === "install") {
          return accumulated;
        }
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
        const categorySkillIds = new Set(category.skills.map((s) => s.id));
        const accumulatedInCategory = new Set(accumulated.filter((id) => categorySkillIds.has(id)));

        const skillChoices: Array<{ name: string; value: string; checked: boolean; description?: string } | Separator> = [
          {
            name: `>> ${t("skill_select_all_category")}`,
            value: SELECT_ALL_CATEGORY,
            checked: accumulatedInCategory.size === category.skills.length,
            description: t("skill_select_all_category"),
          },
          new Separator("──────────────"),
          ...category.skills.map((skill) => {
            const isInstalled = installedSet.has(skill.id);
            const isAccumulated = accumulatedInCategory.has(skill.id);
            const suffix = isInstalled ? ` [${t("skill_already_installed")}]` : (isAccumulated ? " [+]" : "");
            return {
              name: `${chalk.cyan(skill.id)}${suffix}`,
              value: skill.id,
              checked: isAccumulated,
              description: skill.description,
            };
          }),
        ];
        showOperationHints(true);
        const selectedSkills = await checkbox({
          message: t("skill_select_items"),
          choices: skillChoices,
          loop: false,
          instructions: false,
          theme: checkboxTheme,
          pageSize: 15,
        });

        let skillIds: string[];
        if (selectedSkills.includes(SELECT_ALL_CATEGORY)) {
          skillIds = category.skills.map((s) => s.id);
        } else {
          skillIds = selectedSkills.filter((id) => id !== SELECT_ALL_CATEGORY);
        }

        if (skillIds.length === 0) {
          accumulated = accumulated.filter((id) => !categorySkillIds.has(id));
          const action = await select({
            message: accumulated.length > 0
              ? t("skill_continue_or_install").replace("{count}", String(accumulated.length))
              : t("wizard_no_selection"),
            choices: accumulated.length > 0
              ? [
                  { name: ">> " + t("skill_confirm_install_count").replace("{count}", String(accumulated.length)), value: "install" },
                  { name: "<- " + t("wizard_back_to_reselect"), value: "back" },
                  { name: "x  " + t("wizard_cancel"), value: "cancel" },
                ]
              : [
                  new Separator("──────────────"),
                  { name: "<- " + t("wizard_back_to_reselect"), value: "back" },
                  { name: "x  " + t("wizard_cancel"), value: "cancel" },
                ],
            loop: false,
            theme: selectTheme,
          });
          if (action === "install") return accumulated;
          if (action === "back") { step = 0; break; }
          return "cancel";
        }

        accumulated = accumulated.filter((id) => !categorySkillIds.has(id));
        for (const id of skillIds) {
          if (!accumulated.includes(id)) {
            accumulated.push(id);
          }
        }

        const action = await select({
          message: t("skill_continue_or_install").replace("{count}", String(accumulated.length)),
          choices: [
            { name: ">> " + t("skill_confirm_install_count").replace("{count}", String(accumulated.length)), value: "install" },
            { name: "> " + t("skill_continue_select"), value: CONTINUE_SELECT },
            new Separator("──────────────"),
            { name: "x  " + t("wizard_cancel"), value: "cancel" },
          ],
          loop: false,
          theme: selectTheme,
        });

        if (action === "install") {
          return accumulated;
        }
        if (action === CONTINUE_SELECT) {
          step = 0;
          break;
        }
        return "cancel";
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

function resolveSkillSourcePath(skill: { filePath?: string; source: string; id: string }, repoPath: string): string {
  if (skill.filePath) {
    const absPath = isAbsolute(skill.filePath) ? skill.filePath : join(repoPath, skill.filePath);
    if (existsSync(absPath)) {
      return dirname(absPath);
    }
  }
  return join(repoPath, skill.source, skill.id);
}
