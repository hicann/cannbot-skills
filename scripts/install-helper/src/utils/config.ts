// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { existsSync, mkdirSync, readFileSync } from "fs";
import { join } from "path";
import { parse, stringify } from "yaml";
import type { AppConfig } from "../types/index.js";
import { atomicWriteFileSync } from "./fs.js";
import { logger } from "./logger.js";
import { t } from "./i18n.js";
import { getCannbotConfigDir } from "./paths.js";

export function getConfigDir(): string {
  return getCannbotConfigDir();
}

export function getConfigPath(): string {
  return join(getConfigDir(), "config.yaml");
}

export function readConfig(): AppConfig {
  const configPath = getConfigPath();
  if (!existsSync(configPath)) {
    return {
      language: "zh_CN",
      installedPlugins: [],
    };
  }

  try {
    const content = readFileSync(configPath, "utf-8");
    const parsed = parse(content) as Partial<AppConfig>;
    return {
      language: parsed.language || "zh_CN",
      lastTool: parsed.lastTool,
      lastLevel: parsed.lastLevel,
      repoPath: parsed.repoPath,
      installedPlugins: parsed.installedPlugins || [],
    };
  } catch {
    logger.warn(t("config_corrupted"));
    return {
      language: "zh_CN",
      installedPlugins: [],
    };
  }
}

export function writeConfig(config: AppConfig): void {
  const configDir = getConfigDir();
  if (!existsSync(configDir)) {
    mkdirSync(configDir, { recursive: true });
  }

  const configPath = getConfigPath();
  const content = stringify(config);
  atomicWriteFileSync(configPath, content);
}

export function updateConfig(updates: Partial<AppConfig>): AppConfig {
  const config = readConfig();
  const updated = { ...config, ...updates };
  writeConfig(updated);
  return updated;
}

export function addInstalledPlugin(pluginId: string): void {
  const config = readConfig();
  if (!config.installedPlugins.includes(pluginId)) {
    config.installedPlugins.push(pluginId);
    writeConfig(config);
  }
}

export function removeInstalledPlugin(pluginId: string): void {
  const config = readConfig();
  config.installedPlugins = config.installedPlugins.filter(
    (id) => id !== pluginId
  );
  writeConfig(config);
}
