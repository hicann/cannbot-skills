// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { select, checkbox, Separator } from "@inquirer/prompts";
import chalk from "chalk";
import type { AITool, InstallLevel, WizardAnswers } from "../types/index.js";
import { getAllPlugins } from "../core/registry.js";
import { detectTools, getToolDisplayName } from "../core/detector.js";
import { readAllManifests } from "../core/manifest.js";
import { getConfigRoot } from "../utils/paths.js";
import { t } from "../utils/i18n.js";
import { logger, createSpinner, printBoxTitle, showOperationHints } from "../utils/logger.js";
import { readConfig, updateConfig } from "../utils/config.js";
import { selectTheme, checkboxTheme } from "./theme.js";

const BACK = "__back__";
const CANCEL = "__cancel__";

export async function selectToolWithDetection(): Promise<AITool | "back" | "cancel"> {
  const config = readConfig();
  logger.blank();

  const spinner = createSpinner(t("wizard_detect"));
  spinner.start();
  const detectedTools = await detectTools();
  spinner.stop();

  const result = await stepTool(detectedTools, config.lastTool);
  if (result === BACK) return "back";
  if (result === CANCEL) return "cancel";
  return result as AITool;
}

export async function runWizard(): Promise<WizardAnswers> {
  const config = readConfig();
  logger.blank();

  const spinner = createSpinner(t("wizard_detect"));
  spinner.start();
  const detectedTools = await detectTools();
  spinner.stop();

  let selectedTool: AITool | undefined;
  let level: InstallLevel = "project";
  let plugins: string[] = [];

  let step = 0;

  while (true) {
    switch (step) {
      case 0: {
        const result = await stepTool(detectedTools, config.lastTool);
        if (result === BACK) {
          return { language: config.language || "zh_CN", tool: "opencode", level: "project", plugins: [], confirmed: false, back: true };
        }
        if (result === CANCEL) {
          return { language: config.language || "zh_CN", tool: "opencode", level: "project", plugins: [], confirmed: false };
        }
        selectedTool = result as AITool;
        step = 1;
        break;
      }
      case 1: {
        const result = await stepLevel(config.lastLevel);
        if (result === BACK) { step = 0; break; }
        if (result === CANCEL) {
          return { language: config.language || "zh_CN", tool: selectedTool!, level: "project", plugins: [], confirmed: false };
        }
        level = result as InstallLevel;
        step = 2;
        break;
      }
      case 2: {
        const result = await stepPlugins(selectedTool!, level);
        if (result === BACK) { step = 1; break; }
        if (result === CANCEL) {
          return { language: config.language || "zh_CN", tool: selectedTool!, level, plugins: [], confirmed: false };
        }
        plugins = result as string[];
        step = 3;
        break;
      }
      case 3: {
        const result = await stepConfirm(selectedTool!, level, plugins);
        if (result === BACK) { step = 2; break; }
        if (result === CANCEL || result === false) {
          return { language: config.language || "zh_CN", tool: selectedTool!, level, plugins: [], confirmed: false };
        }
        updateConfig({ lastTool: selectedTool, lastLevel: level });
        return { language: config.language || "zh_CN", tool: selectedTool!, level, plugins, confirmed: true };
      }
    }
  }
}

async function stepTool(detectedTools: Awaited<ReturnType<typeof detectTools>>, lastTool?: AITool): Promise<AITool | typeof BACK | typeof CANCEL> {
  printBoxTitle(t("wizard_step_1_title"));

  if (detectedTools.length === 0) {
    logger.error(t("error_tool_not_found"));
    const choices: Array<{ name: string; value: string } | Separator> = [
      ...["opencode", "claude", "trae", "cursor", "copilot"].map((tool) => ({
        name: `> ${getToolDisplayName(tool as AITool)}`,
        value: tool,
      })),
      new Separator("──────────────"),
      { name: "<- " + t("wizard_back"), value: BACK },
      { name: "x  " + t("wizard_cancel"), value: CANCEL },
    ];
    showOperationHints();
    const result = await select({ message: t("wizard_select_tool"), choices, default: lastTool, loop: false, theme: selectTheme });
    if (result === BACK) return BACK;
    if (result === CANCEL) return CANCEL;
    return result as AITool;
  }

  if (detectedTools.length === 1) {
    const tool = detectedTools[0];
    const isLastUsed = lastTool === tool.name;
    const suffix  = isLastUsed ? ` [${t("wizard_last_used")}]` : "";
    logger.success(
      `${t("wizard_detected_tool")}: ${getToolDisplayName(tool.name)}${tool.version ? ` (v${tool.version})` : ""}${suffix}`
    );
    const choices: Array<{ name: string; value: string } | Separator> = [
      { name: `> ${t("wizard_confirm_tool")} ${getToolDisplayName(tool.name)}`, value: tool.name },
      { name: `> ${t("wizard_select_tool")}`, value: "manual" },
      new Separator("──────────────"),
      { name: "<- " + t("wizard_back"), value: BACK },
      { name: "x  " + t("wizard_cancel"), value: CANCEL },
    ];
    showOperationHints();
    const result = await select({ message: t("wizard_select_tool"), choices, loop: false, theme: selectTheme });
    if (result === BACK) return BACK;
    if (result === CANCEL) return CANCEL;
    if (result === "manual") {
      return stepToolManual(lastTool);
    }
    return tool.name as AITool;
  }

  const choices: Array<{ name: string; value: string } | Separator> = detectedTools.map((tool) => {
    const isLastUsed = lastTool === tool.name;
    const suffix  = isLastUsed ? ` [${t("wizard_last_used")}]` : "";
    return {
      name: `> ${getToolDisplayName(tool.name)}${tool.version ? ` (v${tool.version})` : ""}${suffix}`,
      value: tool.name,
    };
  });
  choices.push(new Separator("──────────────"));
  choices.push({ name: "<- " + t("wizard_back"), value: BACK });
  choices.push({ name: "x  " + t("wizard_cancel"), value: CANCEL });

  showOperationHints();
  const result = await select({ message: t("wizard_select_tool"), choices, default: lastTool, loop: false, theme: selectTheme });
  if (result === BACK) return BACK;
  if (result === CANCEL) return CANCEL;
  return result as AITool;
}

async function stepToolManual(lastTool?: AITool): Promise<AITool | typeof BACK | typeof CANCEL> {
  const choices: Array<{ name: string; value: string } | Separator> = [
    ...["opencode", "claude", "trae", "cursor", "copilot"].map((tool) => ({
      name: `> ${getToolDisplayName(tool as AITool)}`,
      value: tool,
    })),
    new Separator("──────────────"),
    { name: "<- " + t("wizard_back"), value: BACK },
    { name: "x  " + t("wizard_cancel"), value: CANCEL },
  ];
  showOperationHints();
  const result = await select({ message: t("wizard_select_tool"), choices, default: lastTool, loop: false, theme: selectTheme });
  if (result === BACK) return BACK;
  if (result === CANCEL) return CANCEL;
  return result as AITool;
}

export async function stepLevel(lastLevel?: InstallLevel): Promise<InstallLevel | typeof BACK | typeof CANCEL> {
  printBoxTitle(t("wizard_step_2_title"));

  const choices: Array<{ name: string; value: string } | Separator> = [
    { name: `> project — ${t("install_level_project")}`, value: "project" },
    { name: `> global — ${t("install_level_global")}`, value: "global" },
    new Separator("──────────────"),
    { name: "<- " + t("wizard_back"), value: BACK },
    { name: "x  " + t("wizard_cancel"), value: CANCEL },
  ];
  showOperationHints();
  const result = await select({
    message: t("wizard_select_level"),
    choices,
    default: lastLevel || "project",
    loop: false,
    theme: selectTheme,
  });
  if (result === BACK) return BACK;
  if (result === CANCEL) return CANCEL;
  return result as InstallLevel;
}

async function stepPlugins(tool: AITool, level: InstallLevel): Promise<string[] | typeof BACK | typeof CANCEL> {
  printBoxTitle(t("wizard_step_3_title"));

  const plugins = getAllPlugins();
  const configRoot = getConfigRoot(tool, level);
  const manifests = readAllManifests(configRoot);
  const installedSet = new Set(manifests.map((m) => m.team));

  const choices: Array<{ name: string; value: string; checked: boolean } | Separator> = plugins.map((p) => {
    const isInstalled = installedSet.has(p.id);
    const suffix  = isInstalled ? ` [${t("wizard_already_installed")}]` : "";
    return {
      name: `${p.displayName}${suffix} — ${p.description}`,
      value: p.id,
      checked: false,
    };
  });
  showOperationHints(true);
  const selected = await checkbox({
    message: t("wizard_select_plugins"),
    choices,
    loop: false,
    instructions: false,
    theme: checkboxTheme,
    pageSize: 15,
  });

  const pluginIds = selected;

  // 如果什么都没勾选，显示操作菜单
  if (pluginIds.length === 0) {
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
    if (action === "back") return BACK;
    return CANCEL;
  }

  return pluginIds;
}

async function stepConfirm(
  tool: AITool,
  level: InstallLevel,
  plugins: string[]
): Promise<true | typeof BACK | typeof CANCEL> {
  printBoxTitle(t("wizard_step_4_title"));

  const allPlugins = getAllPlugins();
  const selectedPlugins = plugins.map((id) => allPlugins.find((p) => p.id === id));

  const displayPath = getConfigRoot(tool, level);
  const toolName = getToolDisplayName(tool);
  const levelText = level === "project" ? "project" : "global";
  logger.info(t("wizard_confirm_install_format").replace("{count}", chalk.bold(String(plugins.length))).replace("{tool}", chalk.green(toolName)).replace("{level}", chalk.cyan(levelText)));
  logger.info(`${t("wizard_install_path")}: ${chalk.cyan(displayPath)}`);
  for (const plugin of selectedPlugins) {
    if (plugin) {
      logger.step(`  • ${chalk.bold(plugin.displayName)}`);
    }
  }
  logger.blank();

  const choices: Array<{ name: string; value: string } | Separator> = [
    { name: "> " + t("wizard_confirm"), value: "confirm" },
    new Separator("──────────────"),
    { name: "<- " + t("wizard_back"), value: BACK },
    { name: "x  " + t("wizard_cancel"), value: CANCEL },
  ];
  showOperationHints();
  const result = await select({ message: t("wizard_confirm"), choices, loop: false, theme: selectTheme });
  if (result === BACK) return BACK;
  if (result === CANCEL) return CANCEL;
  return true;
}
