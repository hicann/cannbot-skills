// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

export type AITool = "opencode" | "claude" | "trae" | "cursor" | "copilot" | "codearts";

export type InstallLevel = "project" | "global";

export type TraeVariant = "ide" | "plugin" | "cli" | "unknown";

export interface DetectedTool {
  name: AITool;
  version?: string;
  path: string;
}

export interface PluginManifestSkillEntry {
  name: string;
  as?: string;
}

export interface PluginManifestSkillSource {
  dir: string;
  skills: (string | PluginManifestSkillEntry)[];
}

export interface PluginManifestExternalRepo {
  url: string;
  dir: string;
  depth?: number;
  recursive?: boolean;
  configRootLink?: boolean;
}

export interface PluginEntry {
  id: string;
  dir: string;
  displayName: string;
  script: string;
  aliases: string[];
  skills: number;
  agents: number;
  description: string;
  version?: string;
  configFile?: string;
  configRootConfigLink?: boolean;
  installSkills?: PluginManifestSkillSource[];
  installAgents?: string[];
  externalRepos?: PluginManifestExternalRepo[];
}

export interface InstallOptions {
  pluginId: string;
  tool: AITool;
  level: InstallLevel;
  repoPath: string;
  installPath?: string;
  yes?: boolean;
}

export interface InstallResult {
  success: boolean;
  pluginId: string;
  skillsCount: number;
  agentsCount: number;
  errors: string[];
  warnings: string[];
}

export interface BackupInfo {
  filePath: string;
  pluginId: string;
  pluginName: string;
  backupTime: string;
}

export interface CannbotManifest {
  brand: string;
  version: string;
  team: string;
  level: string;
  tool: string;
  installed_skills: string[];
  installed_agents: string[];
  brand_dir: string;
  install_time: string;
}

export interface InstalledPlugin {
  id: string;
  displayName: string;
  tool: AITool;
  level: InstallLevel;
  skillsCount: number;
  agentsCount: number;
  installTime: string;
  configRoot: string;
}

export interface WizardAnswers {
  language: "zh_CN" | "en_US";
  tool: AITool;
  level: InstallLevel;
  plugins: string[];
  confirmed: boolean;
  back?: boolean;
}

export interface AppConfig {
  language: "zh_CN" | "en_US";
  lastTool?: AITool;
  lastLevel?: InstallLevel;
  repoPath?: string;
  installedPlugins: string[];
}
