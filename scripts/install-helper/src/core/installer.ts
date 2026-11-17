// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { existsSync, readFileSync, realpathSync } from "fs";
import { join, basename, isAbsolute } from "path";
import { execa, execaSync } from "execa";
import type { AITool, InstallLevel, InstallOptions, InstallResult, BackupInfo } from "../types/index.js";
import { getPluginById } from "./registry.js";
import { readManifest } from "./manifest.js";
import { getConfigRoot } from "../utils/paths.js";
import { scanInstalledFiles, writeRecord } from "./record.js";
import { detectCurrentPlugin, createBackup, findRecordOwner, isPreservedTarget, UNOWNED_PLUGIN_ID } from "./backup.js";
import { isSymlink } from "../utils/fs-helpers.js";
import { showOverwriteWarning, showUnownedFileWarning } from "../ui/backup-prompts.js";
import { installViaManifest, configFileTargets } from "./plugin-installer.js";
import { t } from "../utils/i18n.js";
import { logger } from "../utils/logger.js";

function findShell(): string | null {
  const candidates = ["bash", "sh"];
  for (const cmd of candidates) {
    try {
      const detectCmd = process.platform === "win32" ? "where" : "which";
      execaSync(detectCmd, [cmd], { timeout: 3000 });
      return cmd;
    } catch {}
  }
  return null;
}

export function scriptSupportsTool(scriptPath: string, tool: AITool): boolean {
  try {
    const scriptContent = readFileSync(scriptPath, "utf-8");
    return new RegExp(`\\b${tool}\\b`).test(scriptContent);
  } catch {
    // script unreadable — let execa surface the failure
    return true;
  }
}

interface TargetOwner {
  kind: "same-plugin" | "other-plugin" | "unowned";
  pluginId: string;
  pluginName: string;
}

/**
 * Classifies who owns an existing configuration file that installation is
 * about to replace:
 *  - "same-plugin": the file is this plugin's own (reinstall) — no prompt;
 *  - "other-plugin": a recorded/manifested install of another plugin;
 *  - "unowned": a user-maintained or unknown-origin file — must be confirmed
 *    by the user before it is replaced.
 */
function classifyConfigTarget(
  targetPath: string,
  pluginConfigSource: string,
  pluginId: string,
  pluginName: string,
  configRoot: string,
  tool: AITool,
  installPath: string
): TargetOwner {
  // A symlink pointing at this plugin's own source file: reinstall.
  if (isSymlink(targetPath)) {
    try {
      if (realpathSync(targetPath) === realpathSync(pluginConfigSource)) {
        return { kind: "same-plugin", pluginId, pluginName };
      }
    } catch {
      // dangling/unreadable link — fall through to the record lookup
    }
  }

  // A previous install explicitly preserved this file for the user — it is
  // user-maintained, not plugin-owned (its record keeps it out of `files`),
  // so it must be re-confirmed instead of silently replaced by a reinstall
  // that otherwise looks like "same-plugin" via the leftover manifest.
  if (isPreservedTarget(targetPath)) {
    return { kind: "unowned", pluginId: UNOWNED_PLUGIN_ID, pluginName: UNOWNED_PLUGIN_ID };
  }

  // A recorded install owns the target (covers project-root files too).
  const recordOwner = findRecordOwner(targetPath);
  if (recordOwner) {
    return recordOwner.pluginId === pluginId
      ? { kind: "same-plugin", pluginId, pluginName }
      : { kind: "other-plugin", ...recordOwner };
  }

  // Config-root manifests: the legacy ownership signal.
  const currentPlugin = detectCurrentPlugin(configRoot, tool, installPath);
  if (currentPlugin) {
    return currentPlugin.pluginId === pluginId
      ? { kind: "same-plugin", pluginId, pluginName }
      : { kind: "other-plugin", ...currentPlugin };
  }

  return { kind: "unowned", pluginId: UNOWNED_PLUGIN_ID, pluginName: UNOWNED_PLUGIN_ID };
}

export async function installPlugin(
  opts: InstallOptions
): Promise<InstallResult> {
  // Prefer the caller-resolved entry (installPluginRouted passes the entry
  // it routed on — source-scoped). Re-resolving by bare id here would apply
  // cannbot-first precedence to a legacy-routed install of a same-name
  // plugin, silently swapping in the cannbot entry whose absolute dir and
  // sentinel script violate this script-installer's contract.
  const plugin = opts.plugin ?? getPluginById(opts.pluginId);
  if (!plugin) {
    return {
      success: false,
      pluginId: opts.pluginId,
      skillsCount: 0,
      agentsCount: 0,
      errors: [t("error_plugin_not_found") + ": " + opts.pluginId],
      warnings: [],
    };
  }

  // Guard: cannbot-source entries must go through installPluginRouted
  // delegation, never the legacy script path. Reached only via internal
  // routing errors — refuse with a clear message instead of a bogus
  // double-joined script path.
  if (plugin.source === "cannbot") {
    return {
      success: false,
      pluginId: opts.pluginId,
      skillsCount: 0,
      agentsCount: 0,
      errors: [t("error_cannbot_requires_delegation").replace("{plugin}", plugin.displayName)],
      warnings: [],
    };
  }

  // Legacy entries carry repo-relative dirs; cannbot discovery entries are
  // absolute (managed installer package) and must never be joined onto repoPath.
  const pluginDir = isAbsolute(plugin.dir) ? plugin.dir : join(opts.repoPath, plugin.dir);
  const cwd = opts.installPath || process.cwd();
  const configRoot = getConfigRoot(opts.tool, opts.level, opts.installPath);

  // --- Multi-target preflight: every configuration file this install will
  // replace is detected, confirmed and backed up before anything is touched.
  // At project level that includes the project-root instructions file, which
  // used to be replaced silently (no prompt, no backup) whenever the tool
  // config root held no previous install.
  const pluginConfigSource = join(pluginDir, plugin.configFile || "AGENTS.md");
  const existingTargets = configFileTargets(
    pluginDir,
    plugin.configFile || "AGENTS.md",
    opts.tool,
    opts.level,
    configRoot,
    cwd,
    plugin.configRootConfigLink
  ).filter((p) => existsSync(p) || isSymlink(p));

  const backups: BackupInfo[] = [];
  // Targets the user chose to keep (or that --yes preserves) — passed to the
  // manifest installer so it skips them. Legacy init.sh scripts cannot honor
  // this list (see issue #372); for them the preflight backup above is the
  // recovery path.
  const preservedTargets: string[] = [];
  const promptedPlugins = new Set<string>();

  for (const target of existingTargets) {
    const owner = classifyConfigTarget(
      target,
      pluginConfigSource,
      opts.pluginId,
      plugin.displayName,
      configRoot,
      opts.tool,
      cwd
    );

    if (owner.kind === "same-plugin") {
      // Reinstall of the same plugin: content identical, keep the safety
      // backup but never prompt (historical behavior).
      const backup = createBackup(target, owner.pluginId, owner.pluginName);
      if (backup) backups.push(backup);
      continue;
    }

    if (owner.kind === "other-plugin") {
      if (!promptedPlugins.has(owner.pluginId)) {
        promptedPlugins.add(owner.pluginId);
        let choice: "overwrite" | "cancel" = "overwrite";

        if (!opts.yes) {
          choice = await showOverwriteWarning(owner.pluginName, plugin.displayName);
        }

        if (choice === "cancel") {
          logger.info(t("backup_cancel"));
          return {
            success: false,
            pluginId: opts.pluginId,
            skillsCount: 0,
            agentsCount: 0,
            errors: [t("backup_cancel")],
            warnings: [],
          };
        }
      }

      const backup = createBackup(target, owner.pluginId, owner.pluginName);
      if (!backup) {
        logger.error(t("backup_failed"));
        return {
          success: false,
          pluginId: opts.pluginId,
          skillsCount: 0,
          agentsCount: 0,
          errors: [t("backup_failed")],
          warnings: [],
        };
      }
      backups.push(backup);
      logger.success(`${t("backup_created")}: ${backup.filePath}`);
      continue;
    }

    // Unowned file: user-maintained or unknown origin.
    if (opts.yes) {
      // Non-interactive: no explicit user confirmation exists — preserve it.
      preservedTargets.push(target);
      logger.warn(t("unowned_preserved_yes").replace("{file}", target));
      continue;
    }

    const choice = await showUnownedFileWarning(target, plugin.displayName);
    if (choice === "cancel") {
      logger.info(t("backup_cancel"));
      return {
        success: false,
        pluginId: opts.pluginId,
        skillsCount: 0,
        agentsCount: 0,
        errors: [t("backup_cancel")],
        warnings: [],
      };
    }

    if (choice === "keep") {
      preservedTargets.push(target);
      logger.info(t("unowned_preserved").replace("{file}", target));
      continue;
    }

    const backup = createBackup(target, UNOWNED_PLUGIN_ID, t("unowned_backup_label"));
    if (!backup) {
      logger.error(t("backup_failed"));
      return {
        success: false,
        pluginId: opts.pluginId,
        skillsCount: 0,
        agentsCount: 0,
        errors: [t("backup_failed")],
        warnings: [],
      };
    }
    backups.push(backup);
    logger.success(`${t("backup_created")}: ${backup.filePath}`);
  }

  if (plugin.installSkills && plugin.installSkills.length > 0) {
    const result = await installViaManifest(
      plugin,
      opts.repoPath,
      opts.tool,
      opts.level,
      opts.installPath,
      preservedTargets
    );

    if (result.success) {
      try {
        const record = scanInstalledFiles(
          opts.pluginId,
          plugin.displayName,
          opts.tool,
          opts.level,
          cwd,
          configRoot,
          result.manifest,
          plugin.externalRepos?.map(r => basename(r.dir)),
          plugin.configRootConfigLink,
          preservedTargets
        );

        if (backups.length > 0) {
          record.backups = backups.map((b) => ({
            filePath: b.filePath,
            originalPath: b.originalPath,
            fromPluginId: b.pluginId,
            fromPluginName: b.pluginName,
            backupTime: b.backupTime,
          }));
        }
        if (preservedTargets.length > 0) {
          record.preservedTargets = [...preservedTargets];
        }

        writeRecord(record);
      } catch {
        logger.warn(t("record_write_failed"));
      }

      return {
        success: true,
        pluginId: opts.pluginId,
        skillsCount: result.skillsCount,
        agentsCount: result.agentsCount,
        errors: result.errors,
        warnings: [],
      };
    } else {
      return {
        success: false,
        pluginId: opts.pluginId,
        skillsCount: 0,
        agentsCount: 0,
        errors: result.errors,
        warnings: [],
      };
    }
  }

  const scriptPath = join(pluginDir, plugin.script);
  if (!existsSync(scriptPath)) {
    return {
      success: false,
      pluginId: opts.pluginId,
      skillsCount: 0,
      agentsCount: 0,
      errors: [t("error_script_not_found").replace("{path}", scriptPath)],
      warnings: [],
    };
  }

  // Guard: the plugin's init.sh must actually support the target tool.
  // Scripts that predate a tool (e.g. most community init.sh without codex)
  // silently ignore the unknown arg and would mis-install into default
  // .opencode directories — block that with an explicit error.
  if (!scriptSupportsTool(scriptPath, opts.tool)) {
    return {
      success: false,
      pluginId: opts.pluginId,
      skillsCount: 0,
      agentsCount: 0,
      errors: [t("error_tool_not_supported").replace("{plugin}", plugin.displayName).replace("{tool}", opts.tool)],
      warnings: [],
    };
  }

  const args: string[] = [opts.level, opts.tool];
  if (opts.installPath) {
    args.push(opts.installPath);
  }

  try {
    const shell = findShell();
    if (!shell) {
      return {
        success: false,
        pluginId: opts.pluginId,
        skillsCount: 0,
        agentsCount: 0,
        errors: [t("error_no_shell")],
        warnings: [],
      };
    }
    const execOptions: any = {
      cwd,
      timeout: 300000,
      stdio: "pipe",
      input: opts.yes ? "y\n" : undefined,
    };
    
    await execa(shell, [scriptPath, ...args], execOptions);

    const manifest = readManifest(configRoot);

    let skillsCount = 0;
    let agentsCount = 0;

    if (manifest) {
      skillsCount = manifest.installed_skills?.length || 0;
      agentsCount = manifest.installed_agents?.length || 0;
    } else {
      skillsCount = plugin.skills;
      agentsCount = plugin.agents;
    }

    try {
      const record = scanInstalledFiles(
        opts.pluginId,
        plugin.displayName,
        opts.tool,
        opts.level,
        cwd,
        configRoot,
        manifest,
        plugin.externalRepos?.map(r => basename(r.dir)),
        plugin.configRootConfigLink,
        preservedTargets
      );

      if (backups.length > 0) {
        record.backups = backups.map((b) => ({
          filePath: b.filePath,
          originalPath: b.originalPath,
          fromPluginId: b.pluginId,
          fromPluginName: b.pluginName,
          backupTime: b.backupTime,
        }));
      }
      if (preservedTargets.length > 0) {
        record.preservedTargets = [...preservedTargets];
      }

      writeRecord(record);
    } catch {
      logger.warn(t("record_write_failed"));
    }

    return {
      success: true,
      pluginId: opts.pluginId,
      skillsCount,
      agentsCount,
      errors: [],
      warnings: [],
    };
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : t("error_unknown");
    return {
      success: false,
      pluginId: opts.pluginId,
      skillsCount: 0,
      agentsCount: 0,
      errors: [errorMessage],
      warnings: [],
    };
  }
}
