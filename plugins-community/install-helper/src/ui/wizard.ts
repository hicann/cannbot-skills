// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { select, Separator } from "@inquirer/prompts";
import chalk from "chalk";
import type { AITool, InstallLevel, PluginEntry, PluginSource, WizardAnswers } from "../types/index.js";
import { getAllPlugins } from "../core/registry.js";
import { detectTools, getToolDisplayName } from "../core/detector.js";
import { readAllManifests } from "../core/manifest.js";
import { getConfigRoot, VALID_TOOLS } from "../utils/paths.js";
import { t } from "../utils/i18n.js";
import { logger, createSpinner, printBoxTitle, showOperationHints } from "../utils/logger.js";
import { readConfig, updateConfig } from "../utils/config.js";
import { selectTheme, truncateForDisplay } from "./theme.js";
import { BACK, CANCEL } from "../utils/constants.js";

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

// Wizard choice values encode the plugin source so same-name cannbot and
// legacy entries stay distinguishable through selection, confirm and install.
export function encodePluginChoice(id: string, source?: PluginSource): string {
  return source === "cannbot" ? `cannbot::${id}` : `skills::${id}`;
}

/**
 * One entry per plugin id for the wizard's install-selection list.
 * getAllPlugins() intentionally lists dual-source plugins once per source
 * (frozen local copy + cannbot entry — the doctor/list badge design); in the
 * wizard a plugin that can be installed two ways is still one choice. The
 * cannbot entry wins wholesale (same routing as findPlugin's default
 * resolution): the row shows the repository's own name and description —
 * the cannbot manifest is the display authority, the frozen embedded
 * name/description never shadows it.
 */
export function dedupeForWizardSelection(plugins: PluginEntry[]): PluginEntry[] {
  const seen = new Map<string, PluginEntry>();
  for (const p of plugins) {
    const existing = seen.get(p.id);
    if (!existing) {
      seen.set(p.id, { ...p });
      continue;
    }
    // cannbot source wins the routing — and the display: repo name + repo
    // description, regardless of arrival order.
    if (p.source === "cannbot" && existing.source !== "cannbot") {
      seen.set(p.id, { ...p });
    }
  }
  // Re-sort by the WINNING entry's displayName — the pre-dedupe sort
  // positions entries by their source-specific names (e.g. a frozen yml
  // "AscendC 算子注册调用" sorts at 'A' while the winning cannbot entry
  // "ops-registry-invoke" belongs at 'o'); without re-sorting, the
  // replaced entry stays at the frozen name's position.
  return [...seen.values()].sort((a, b) =>
    a.displayName < b.displayName ? -1 : a.displayName > b.displayName ? 1 : 0
  );
}

export function decodePluginChoice(value: string): { id: string; source: PluginSource } {
  if (value.startsWith("cannbot::")) return { id: value.slice("cannbot::".length), source: "cannbot" };
  if (value.startsWith("skills::")) return { id: value.slice("skills::".length), source: "skills" };
  return { id: value, source: "skills" };
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
        if (result === CANCEL) {
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
      ...VALID_TOOLS.map((tool) => ({
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
    ...VALID_TOOLS.map((tool) => ({
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

/** Inline description budget (display columns) for wizard choice rows. The
 *  full description stays available on the inquirer focus line; the inline
 *  part only identifies the plugin at a glance. */
const CHOICE_DESCRIPTION_WIDTH = 40;

/** Builds the wizard choice label for a plugin: friendly name, source badge,
 *  installed marker and a width-aware truncated description. Pure function
 *  so display behavior is unit-testable without a TTY. */
export function pluginChoiceLabel(plugin: PluginEntry, installed: boolean): string {
  const badge = plugin.source === "cannbot" ? chalk.magenta(" [cannbot]") : "";
  const suffix = installed ? ` [${t("wizard_already_installed")}]` : "";
  const hint = plugin.description
    ? ` ${chalk.dim(`— ${truncateForDisplay(plugin.description, CHOICE_DESCRIPTION_WIDTH)}`)}`
    : "";
  return `> ${chalk.cyan(plugin.displayName)}${badge}${suffix}${hint}`;
}

async function stepPlugins(tool: AITool, level: InstallLevel): Promise<string[] | typeof BACK | typeof CANCEL> {
  printBoxTitle(t("wizard_step_3_title"));

  const plugins = dedupeForWizardSelection(getAllPlugins());
  const configRoot = getConfigRoot(tool, level);
  const manifests = readAllManifests(configRoot);
  const installedSet = new Set(manifests.map((m) => m.team));

  const choices: Array<{ name: string; value: string; description?: string } | Separator> = plugins.map((p) => {
    const isInstalled = installedSet.has(p.id);
    return {
      name: pluginChoiceLabel(p, isInstalled),
      value: encodePluginChoice(p.id, p.source),
      description: p.description,
    };
  });
  choices.push(new Separator("──────────────"));
  choices.push({ name: "<- " + t("wizard_back"), value: BACK });
  choices.push({ name: "x  " + t("wizard_cancel"), value: CANCEL });

  showOperationHints();
  const selected = await select({
    message: t("wizard_select_plugin"),
    choices,
    loop: false,
    theme: selectTheme,
    pageSize: 15,
  });

  if (selected === BACK) return BACK;
  if (selected === CANCEL) return CANCEL;
  return [selected];
}

async function stepConfirm(
  tool: AITool,
  level: InstallLevel,
  plugins: string[]
): Promise<true | typeof BACK | typeof CANCEL> {
  printBoxTitle(t("wizard_step_4_title"));

  const allPlugins = getAllPlugins();
  const selectedPlugins = plugins.map((value) => {
    const { id, source } = decodePluginChoice(value);
    return allPlugins.find((p) => p.id === id && (p.source ?? "skills") === source);
  });

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
