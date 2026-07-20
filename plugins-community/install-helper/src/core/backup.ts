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
import { join } from "path";
import type { AITool, BackupInfo } from "../types/index.js";
import { readAllManifests } from "./manifest.js";
import { readRecord } from "./record.js";
import { findPlugin, getAllPlugins } from "./registry.js";
import { getAgentsFileName } from "../utils/paths.js";

export { getAgentsFileName };

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

export function createBackup(
  configRoot: string,
  tool: AITool,
  fromPluginId: string,
  fromPluginName: string
): BackupInfo | null {
  const agentsFile = join(configRoot, getAgentsFileName(tool));
  if (!existsSync(agentsFile)) {
    return null;
  }

  const timestamp = formatTimestamp(new Date());
  const backupFileName = `${getAgentsFileName(tool)}.cannbot-backup.${fromPluginId}.${timestamp}`;
  const backupPath = join(configRoot, backupFileName);

  try {
    copyFileSync(agentsFile, backupPath);
    return {
      filePath: backupPath,
      pluginId: fromPluginId,
      pluginName: fromPluginName,
      backupTime: timestamp,
    };
  } catch {
    return null;
  }
}

export function findBackups(configRoot: string): BackupInfo[] {
  if (!existsSync(configRoot)) {
    return [];
  }

  const files = readdirSync(configRoot);
  const backups: BackupInfo[] = [];

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
            filePath: join(configRoot, file),
            pluginId,
            pluginName: plugin?.displayName || pluginId,
            backupTime: timestamp,
          });
        }
      }
    }
  }

  return backups.sort((a, b) => b.backupTime.localeCompare(a.backupTime));
}

export function restoreBackup(
  backupPath: string,
  configRoot: string,
  tool: AITool
): boolean {
  try {
    const agentsFile = join(configRoot, getAgentsFileName(tool));
    copyFileSync(backupPath, agentsFile);
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
  const plugins = getAllPlugins();

  for (const plugin of plugins) {
    const record = readRecord(plugin.id);
    if (record) {
      records.push(record);
    }
  }

  return records;
}
