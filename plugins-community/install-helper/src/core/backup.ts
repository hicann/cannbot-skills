// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { existsSync, copyFileSync, readdirSync, unlinkSync } from "fs";
import { join, dirname, basename, resolve } from "path";
import type { AITool, BackupInfo } from "../types/index.js";
import { readAllManifests } from "./manifest.js";
import { readRecords } from "./record.js";
import { findPlugin } from "./registry.js";
import { getAgentsFileName, getCannbotConfigDir } from "../utils/paths.js";

export { getAgentsFileName };

/** Backup owner id used when the replaced file has no known plugin owner
 *  (a user-maintained configuration file). */
export const UNOWNED_PLUGIN_ID = "unowned";

export function detectCurrentPlugin(
  configRoot: string,
  tool: AITool,
  installPath?: string
): { pluginId: string; pluginName: string } | null {
  const manifests = readAllManifests(configRoot);
  if (manifests.length > 0) {
    const sorted = [...manifests].sort((a, b) =>
      (b.install_time || "").localeCompare(a.install_time || "")
    );
    const manifest = sorted[0];
    if (manifest.team) {
      const plugin = findPlugin(manifest.team);
      if (plugin) {
        return {
          pluginId: plugin.id,
          pluginName: plugin.displayName,
        };
      }
    }
  }

  const agentsFileName = getAgentsFileName(tool);
  const configRootAgentsFile = join(configRoot, agentsFileName);
  const installPathAgentsFile = installPath ? join(installPath, agentsFileName) : null;

  if (!existsSync(configRootAgentsFile) && !(installPathAgentsFile && existsSync(installPathAgentsFile))) {
    return null;
  }

  const allRecords = readAllRecords();
  for (const record of allRecords) {
    if (record.configRoot === configRoot && record.tool === tool) {
      const plugin = findPlugin(record.pluginId);
      if (plugin) {
        return {
          pluginId: plugin.id,
          pluginName: plugin.displayName,
        };
      }
    }
  }

  return null;
}

/**
 * Creates a backup of `targetPath` beside the file itself, so a replaced
 * configuration file (including a project-root AGENTS.md that lives outside
 * the tool config root) always keeps a recoverable copy next to its original
 * location. Returns null when the file does not exist.
 */
export function createBackup(
  targetPath: string,
  fromPluginId: string,
  fromPluginName: string
): BackupInfo | null {
  if (!existsSync(targetPath)) {
    return null;
  }

  const timestamp = formatTimestamp(new Date());
  const backupFileName = `${basename(targetPath)}.cannbot-backup.${fromPluginId}.${timestamp}`;
  const backupPath = join(dirname(targetPath), backupFileName);

  try {
    copyFileSync(targetPath, backupPath);
    return {
      filePath: backupPath,
      originalPath: targetPath,
      pluginId: fromPluginId,
      pluginName: fromPluginName,
      backupTime: timestamp,
    };
  } catch {
    return null;
  }
}

/** Finds the plugin whose install record owns `targetPath` (its recorded
 *  files list contains the path), if any. Works for both config-root and
 *  project-root configuration files. */
export function findRecordOwner(
  targetPath: string
): { pluginId: string; pluginName: string } | null {
  const resolved = resolve(targetPath);
  for (const record of readAllRecords()) {
    if ((record.files || []).some((f: string) => resolve(f) === resolved)) {
      const plugin = findPlugin(record.pluginId);
      return {
        pluginId: record.pluginId,
        pluginName: plugin?.displayName || record.displayName || record.pluginId,
      };
    }
  }
  return null;
}

/** True when a previous install explicitly preserved `targetPath` for the
 *  user (recorded in `preservedTargets` and deliberately kept out of the
 *  record's owned files). Such a file is user-maintained: it must be
 *  re-confirmed before any install replaces it — including reinstalls of the
 *  plugin that preserved it, whose leftover manifest would otherwise make
 *  the file look "same-plugin" owned. */
export function isPreservedTarget(targetPath: string): boolean {
  const resolved = resolve(targetPath);
  return readAllRecords().some((record) =>
    (record.preservedTargets || []).some((p: string) => resolve(p) === resolved)
  );
}

/**
 * Scans one or more directories for `*.cannbot-backup.*` files. The restore
 * target (`originalPath`) is derived from the backup file name, so backups
 * created beside project-root files restore to the project root, not to the
 * tool config root.
 */
export function findBackups(configRoot: string | string[]): BackupInfo[] {
  const dirs = Array.isArray(configRoot) ? configRoot : [configRoot];
  const backups: BackupInfo[] = [];

  for (const dir of dirs) {
    if (!existsSync(dir)) {
      continue;
    }

    const files = readdirSync(dir);

    for (const file of files) {
      if (file.includes(".cannbot-backup.")) {
        const parts = file.split(".cannbot-backup.");
        if (parts.length === 2) {
          const pluginIdAndTime = parts[1];
          const lastDotIndex = pluginIdAndTime.lastIndexOf(".");
          if (lastDotIndex > 0) {
            const pluginId = pluginIdAndTime.substring(0, lastDotIndex);
            const timestamp = pluginIdAndTime.substring(lastDotIndex + 1);
            const plugin = findPlugin(pluginId);

            backups.push({
              filePath: join(dir, file),
              originalPath: join(dir, parts[0]),
              pluginId,
              pluginName: plugin?.displayName || pluginId,
              backupTime: timestamp,
            });
          }
        }
      }
    }
  }

  return backups.sort((a, b) => b.backupTime.localeCompare(a.backupTime));
}

/**
 * Restores a backup file to `targetPath` (the original location recorded
 * when the backup was created).
 */
export function restoreBackup(
  backupPath: string,
  targetPath: string
): boolean {
  try {
    copyFileSync(backupPath, targetPath);
    return true;
  } catch {
    return false;
  }
}

export function deleteBackup(backupPath: string): boolean {
  try {
    if (existsSync(backupPath)) {
      unlinkSync(backupPath);
      return true;
    }
    return false;
  } catch {
    return false;
  }
}

function formatTimestamp(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  const hours = String(date.getHours()).padStart(2, "0");
  const minutes = String(date.getMinutes()).padStart(2, "0");
  const seconds = String(date.getSeconds()).padStart(2, "0");
  return `${year}${month}${day}-${hours}${minutes}${seconds}`;
}

function readAllRecords(): any[] {
  const records: any[] = [];

  // Scan the installs directory directly (not the plugin registry): records
  // exist for community plugins that are not part of the embedded registry,
  // and ownership detection must see those too.
  const installsDir = join(getCannbotConfigDir(), "installs");
  if (!existsSync(installsDir)) {
    return records;
  }
  for (const file of readdirSync(installsDir)) {
    if (!file.endsWith(".json") || file === "skills.json") continue;
    const pluginId = file.replace(/\.json$/, "");
    records.push(...readRecords(pluginId));
  }

  return records;
}
