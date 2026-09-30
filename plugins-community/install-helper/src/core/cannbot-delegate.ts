// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

// Delegation protocol: cannbot-style plugins are installed by the cannbot
// repository's own installer (@cannbot-plugin/cannbot) so the resulting
// layout (.agents/skills, marker-block AGENTS.md, .cannbot/plugins assets,
// cannbot-plugin.json registry) is byte-for-byte identical to a direct
// `npx @cannbot-plugin/cannbot install`. install-helper only wraps discovery,
// boundaries, bookkeeping and (uninstall/list/status) around it.

import { existsSync, readFileSync, writeFileSync, copyFileSync, unlinkSync, readdirSync } from "fs";
import { join, resolve } from "path";
import { execa } from "execa";
import type { AITool, InstallLevel, InstallOptions, InstallResult, PluginEntry, PluginSource } from "../types/index.js";
import { discoverCannbotPlugins, ensureCannbotInstaller, getCannbotBinPath, hasNodeRuntime } from "./cannbot-installer.js";
import { findCannbotRegistryPath, readCannbotRegistry } from "./cannbot-registry.js";
import { writeRecord } from "./record.js";
import { installPlugin } from "./installer.js";
import { addInstalledPlugin } from "../utils/config.js";
import { getConfigFileName, getSkillsRoot } from "../utils/paths.js";
import { isSymlink } from "../utils/fs-helpers.js";
import { t } from "../utils/i18n.js";

export const DELEGATE_TOOLS: AITool[] = ["opencode", "claude", "trae", "codex"];

export interface DelegateInstallOptions {
  pluginId: string;
  displayName?: string;
  tool: AITool;
  level: InstallLevel;
  target?: string;
  yes?: boolean;
}

function boundaryErrors(tool: AITool, level: InstallLevel): string[] {
  const errors: string[] = [];
  if (!DELEGATE_TOOLS.includes(tool)) {
    errors.push(
      t("cannbot_tool_unsupported")
        .replace("{tool}", tool)
        .replace("{tools}", DELEGATE_TOOLS.join(", "))
    );
  }
  if (level !== "project") {
    errors.push(t("cannbot_level_unsupported"));
  }
  return errors;
}

/**
 * Migration-time precheck: legacy installs own AGENTS.md/CLAUDE.md as a
 * symlink to the plugin source. The cannbot installer merges marker blocks
 * into a real file, so a symlinked config would be replaced wholesale and
 * the legacy uninstall would later delete the merged file. Convert the
 * symlink to a real file (keeping content) before delegating.
 */
function migrateSymlinkedInstructions(target: string, tool: AITool): string[] {
  const warnings: string[] = [];
  const fileName = getConfigFileName(tool);
  const candidates = [join(target, fileName), join(target, "AGENTS.md"), join(target, "CLAUDE.md")];

  for (const configPath of candidates) {
    if (!isSymlink(configPath)) continue;
    try {
      const content = readFileSync(configPath, "utf-8");
      const backupPath = `${configPath}.cannbot-migration.${Date.now()}`;
      copyFileSync(configPath, backupPath);
      unlinkSync(configPath);
      writeFileSync(configPath, content);
      warnings.push(t("cannbot_migrated_symlink").replace("{file}", fileName));
    } catch {
      warnings.push(t("cannbot_migrate_symlink_failed").replace("{file}", fileName));
    }
    break;
  }

  return warnings;
}

/** Detects skills installed by a legacy plugin that the cannbot plugin also
 *  installs (same skill name in a different layout root) — scenario B. */
function detectDualSkillCopies(tool: AITool, target: string, skillNames: string[]): string[] {
  const duplicates: string[] = [];
  if (tool !== "opencode") return duplicates;

  const legacyRoot = getSkillsRoot("opencode", "project", target);
  if (!existsSync(legacyRoot)) return duplicates;
  let legacyEntries: string[] = [];
  try {
    legacyEntries = readdirSync(legacyRoot);
  } catch {
    return duplicates;
  }

  for (const name of skillNames) {
    if (legacyEntries.includes(name)) {
      duplicates.push(name);
    }
  }
  return duplicates;
}

function skillNameFromPath(rel: string): string {
  const parts = rel.split("/");
  return parts[parts.length - 1] || rel;
}

interface DelegateRunResult {
  ok: boolean;
  stdout: string;
  stderr: string;
  exitCode: number | undefined;
}

/**
 * Runs the cannbot installer CLI. Output is captured and re-emitted
 * verbatim (equivalent to stdio inherit) so the transcript stays identical
 * while still being inspectable for the rename-gap fallback below.
 */
async function runDelegate(
  args: string[],
  target: string
): Promise<DelegateRunResult> {
  try {
    const result = await execa("node", [getCannbotBinPath(), ...args], {
      cwd: target,
      timeout: 600000,
    });
    process.stdout.write(result.stdout || "");
    process.stderr.write(result.stderr || "");
    return { ok: true, stdout: result.stdout || "", stderr: result.stderr || "", exitCode: 0 };
  } catch (error) {
    const err = error as { stdout?: string; stderr?: string; exitCode?: number; message?: string };
    process.stdout.write(err.stdout || "");
    process.stderr.write(err.stderr || "");
    if (!err.stdout && !err.stderr && err.message) {
      process.stderr.write(`${err.message}\n`);
    }
    return {
      ok: false,
      stdout: err.stdout || "",
      stderr: err.stderr || "",
      exitCode: err.exitCode,
    };
  }
}

/**
 * The --bundled retry only applies to installers that still expose the flag
 * (pre-1.8.0, where installLatest followed the cannbot mainline). Installers
 * >= 1.8.0 dropped the flag because bundled is the default behavior — for
 * them a first-attempt "plugin not found" is a genuine failure and must be
 * passed through verbatim without a misleading unknown-option retry.
 */
function installerSupportsBundledFlag(): boolean {
  try {
    const binSource = readFileSync(getCannbotBinPath(), "utf-8");
    return binSource.includes('"--bundled"');
  } catch {
    return false;
  }
}

export async function delegateInstall(options: DelegateInstallOptions): Promise<InstallResult> {
  const { pluginId, tool, level } = options;
  const target = resolve(options.target || process.cwd());
  const displayName = options.displayName || pluginId;

  const result: InstallResult = {
    success: false,
    pluginId,
    skillsCount: 0,
    agentsCount: 0,
    errors: [],
    warnings: [],
  };

  // 1. boundary validation (tool matrix + level)
  result.errors.push(...boundaryErrors(tool, level));
  if (result.errors.length > 0) {
    return result;
  }

  // 2. Node.js runtime is required to run the cannbot installer
  if (!hasNodeRuntime()) {
    result.errors.push(t("cannbot_node_required"));
    return result;
  }

  // 3. ensure the managed installer package
  try {
    await ensureCannbotInstaller();
  } catch (error) {
    result.errors.push(
      t("cannbot_installer_failed").replace(
        "{error}",
        error instanceof Error ? error.message : t("error_unknown")
      )
    );
    return result;
  }

  // 4. the plugin must exist in the bundled release; if it is missing from a
  //    cached installer, force one refresh before giving up (handles newly
  //    published cannbot plugins)
  if (!discoverCannbotPlugins().some((p) => p.id === pluginId)) {
    try {
      await ensureCannbotInstaller({ force: true });
    } catch {
      // fall through to the not-found error below
    }
    if (!discoverCannbotPlugins().some((p) => p.id === pluginId)) {
      result.errors.push(t("cannbot_plugin_not_found").replace("{plugin}", pluginId));
      return result;
    }
  }

  // 5. instructions-file precheck (legacy symlink -> real file)
  result.warnings.push(...migrateSymlinkedInstructions(target, tool));

  // 6. delegate: node <installer>/bin/cannbot.js install <id> --tool <t> --target <dir>
  //    Default mirrors a direct `cannbot install`. Installers packaged before
  //    the plugins/ -> plugins-official/ rename (cannbot e140ca6) resolve
  //    official plugins against a fresh mainline checkout via their bundled
  //    sourcePluginDirectory and fail with "plugin not found". When exactly
  //    that failure occurs on a --bundled-capable installer, retry once with
  //    --bundled (install from the installer package's own dist snapshot) so
  //    users are not blocked on upstream's release cadence. Installers
  //    >= 1.8.0 install from the bundled snapshot by default and dropped the
  //    flag, so for them the first failure is final and passed through.
  let run = await runDelegate(["install", pluginId, "--tool", tool, "--target", target], target);

  // installLatest reports the failure via console.error (stderr)
  const hitRenameGap =
    !run.ok &&
    installerSupportsBundledFlag() &&
    (run.stdout.includes(`plugin not found: ${pluginId}`) ||
      run.stderr.includes(`plugin not found: ${pluginId}`));
  if (hitRenameGap) {
    result.warnings.push(t("cannbot_bundled_fallback"));
    run = await runDelegate(
      ["install", pluginId, "--tool", tool, "--target", target, "--bundled"],
      target
    );
  }

  if (!run.ok) {
    result.errors.push(
      t("cannbot_delegate_failed").replace(
        "{code}",
        run.exitCode !== undefined ? String(run.exitCode) : "?"
      )
    );
    return result;
  }

  // 7. read back the registry record the cannbot installer just wrote and
  //    mirror it into install-helper's own bookkeeping
  const registryRead = readCannbotRegistry(target, tool);
  const record = registryRead?.registry.plugins.find((r) => r?.plugin === pluginId);

  if (!record) {
    result.errors.push(t("cannbot_registry_missing"));
    return result;
  }

  result.skillsCount = Array.isArray(record.skills) ? record.skills.length : 0;
  result.agentsCount = Array.isArray(record.agents) ? record.agents.length : 0;

  const files: string[] = [];
  const directories: string[] = [];
  for (const rel of [...(record.skills || []), ...(record.agents || [])]) {
    files.push(join(target, rel));
  }
  if (record.assets) {
    directories.push(join(target, record.assets));
  }
  if (record.permissions) {
    directories.push(join(target, record.permissions));
  }

  const registryPath = findCannbotRegistryPath(target, tool);

  writeRecord({
    pluginId,
    displayName,
    tool,
    level: "project",
    installPath: target,
    configRoot: registryPath ? join(registryPath, "..") : target,
    installTime: new Date().toISOString(),
    files,
    directories,
    kind: "delegated",
  });
  addInstalledPlugin(pluginId);

  // 8. dual-copy warning (scenario B)
  const duplicates = detectDualSkillCopies(
    tool,
    target,
    (record.skills || []).map(skillNameFromPath)
  );
  if (duplicates.length > 0) {
    result.warnings.push(t("cannbot_dual_skill_copy").replace("{skills}", duplicates.join(", ")));
  }

  result.success = true;
  return result;
}

/**
 * Unified routing helper: cannbot-style plugins go through delegation,
 * everything else through the legacy installer. Used by install/update and
 * the interactive wizard so no call site needs to branch on its own.
 */
export async function installPluginRouted(
  options: InstallOptions & { plugin?: PluginEntry; source?: PluginSource }
): Promise<InstallResult> {
  // An explicit --source skills always forces the legacy installer, even
  // when the plugin metadata (resolved by an unscoped cannbot-first lookup
  // at a call site) points at the cannbot entry of a same-name plugin.
  const isCannbot =
    options.source === "cannbot" ||
    (options.source !== "skills" && options.plugin?.source === "cannbot");

  if (isCannbot) {
    return delegateInstall({
      pluginId: options.pluginId,
      displayName: options.plugin?.displayName,
      tool: options.tool,
      level: options.level,
      target: options.installPath || process.cwd(),
      yes: options.yes,
    });
  }

  return installPlugin(options);
}
