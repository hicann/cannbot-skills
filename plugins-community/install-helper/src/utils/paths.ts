// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { homedir } from "os";
import { join } from "path";
import { existsSync } from "fs";
import type { AITool, InstallLevel, TraeVariant } from "../types/index.js";
import { t } from "./i18n.js";

export const VALID_TOOLS: AITool[] = ["opencode", "claude", "trae", "cursor", "copilot", "codearts"];
const VALID_LEVELS: InstallLevel[] = ["project", "global"];

export function validateTool(tool: string): AITool {
  if (!VALID_TOOLS.includes(tool as AITool)) {
    throw new Error(
      t("error_invalid_tool")
        .replace("{tool}", tool)
        .replace("{tools}", VALID_TOOLS.join(", "))
    );
  }
  return tool as AITool;
}

export function validateLevel(level: string): InstallLevel {
  if (!VALID_LEVELS.includes(level as InstallLevel)) {
    throw new Error(
      t("error_invalid_level")
        .replace("{level}", level)
        .replace("{levels}", VALID_LEVELS.join(", "))
    );
  }
  return level as InstallLevel;
}

export function detectTraeVariant(): TraeVariant {
  const home = homedir();
  if (existsSync(join(home, ".trae-cn"))) return "ide";
  if (existsSync(join(home, ".marscode"))) return "plugin";
  if (existsSync(join(home, ".traecli"))) return "cli";
  return "unknown";
}

export function getConfigRoot(
  tool: AITool,
  level: InstallLevel,
  base?: string
): string {
  const home = homedir();

  if (level === "global") {
    switch (tool) {
      case "opencode":
        return join(home, ".config", "opencode");
      case "claude":
        return join(home, ".claude");
      case "trae": {
        const variant = detectTraeVariant();
        switch (variant) {
          case "plugin":
            return join(home, ".marscode");
          case "cli":
            return join(home, ".traecli");
          default:
            return join(home, ".trae-cn");
        }
      }
      case "cursor":
        return join(home, ".cursor");
      case "copilot":
        return join(home, ".copilot");
      case "codearts":
        return join(home, ".codeartsdoer");
      default:
        throw new Error(
          t("error_invalid_tool")
            .replace("{tool}", tool)
            .replace("{tools}", VALID_TOOLS.join(", "))
        );
    }
  }

  const baseDir = base || process.cwd();
  switch (tool) {
    case "opencode":
      return join(baseDir, ".opencode");
    case "claude":
      return join(baseDir, ".claude");
    case "trae": {
      const variant = detectTraeVariant();
      switch (variant) {
        case "plugin":
          return join(baseDir, ".marscode");
        case "cli":
          return join(baseDir, ".traecli");
        default:
          return join(baseDir, ".trae");
      }
    }
    case "cursor":
      return join(baseDir, ".cursor");
    case "copilot":
      return join(baseDir, ".github");
    case "codearts":
      return join(baseDir, ".codeartsdoer");
    default:
      throw new Error(
        t("error_invalid_tool")
          .replace("{tool}", tool)
          .replace("{tools}", VALID_TOOLS.join(", "))
      );
  }
}

export function getConfigFileName(tool: AITool): string {
  return tool === "claude" ? "CLAUDE.md" : "AGENTS.md";
}

export function getAgentsFileName(tool: AITool): string {
  return getConfigFileName(tool);
}

export function getSkillsDir(configRoot: string): string {
  return join(configRoot, "skills");
}

export function getAgentsDir(configRoot: string): string {
  return join(configRoot, "agents");
}

export function getManifestPath(configRoot: string): string {
  return join(configRoot, "cannbot-manifest.json");
}

export function getCannbotConfigDir(): string {
  return join(homedir(), ".cannbot");
}

export function getCannbotRepoPath(): string {
  return join(getCannbotConfigDir(), "repo");
}
