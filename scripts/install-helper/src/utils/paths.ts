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
import { spawnSync } from "child_process";
import type { AITool, InstallLevel, TraeVariant } from "../types/index.js";
import { t } from "./i18n.js";

export const VALID_TOOLS: AITool[] = ["opencode", "claude", "trae", "cursor", "codex", "copilot", "codearts", "qoder", "dsh"];
const VALID_LEVELS: InstallLevel[] = ["project", "global"];

// Qoder has two client layouts with separate directories (verified against
// qodercn 1.1.66):
//   - Qoder CLI (qodercn): project `.qoder/` (skills + agents), user
//     `~/.qoder-cn/` (agents + settings + AGENTS.md memory), user skills at
//     `~/.agents/skills/` (shared agents-skills convention, like codex).
//     The CLI also scans project `.agents/skills/`.
//   - Qoder IDE (Lingma): `.lingma/` (skills + agents), user `~/.lingma/`.
// When the qodercn CLI is on PATH or `~/.qoder-cn` is present the CLI
// layout wins; otherwise fall back to the IDE layout.
// CANNBOT_QODER_LAYOUT=cli|ide forces a layout (tests and environments
// without the CLI on PATH).
let qoderCliLayoutCache: boolean | undefined;

export function isQoderCliLayout(): boolean {
  if (process.env.CANNBOT_QODER_LAYOUT === "cli") return true;
  if (process.env.CANNBOT_QODER_LAYOUT === "ide") return false;
  if (qoderCliLayoutCache === undefined) {
    let cli = false;
    try {
      const whichCmd = process.platform === "win32" ? "where" : "which";
      const result = spawnSync(whichCmd, ["qodercn"], { timeout: 5000 });
      cli = result.status === 0;
    } catch {
      cli = false;
    }
    if (!cli) {
      // ~/.qoder-cn is created by the qodercn CLI (the IDE uses ~/.lingma).
      // GUI / non-login shells often lack the CLI on PATH, so its config dir
      // is the remaining CLI evidence — detectQoder() accepts the same
      // signal. Without this, detection hits Qoder while installs silently
      // land in the IDE layout (.lingma) the CLI never reads.
      cli = existsSync(join(homedir(), ".qoder-cn"));
    }
    qoderCliLayoutCache = cli;
  }
  return qoderCliLayoutCache;
}

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
      case "codex":
        return join(home, ".codex");
      case "copilot":
        return join(home, ".copilot");
      case "codearts":
        return join(home, ".codeartsdoer");
      case "qoder":
        return isQoderCliLayout()
          ? (process.env.QODERCN_CONFIG_DIR || join(home, ".qoder-cn"))
          : join(home, ".lingma");
      case "dsh":
        // DSH_HOME is the official DeepSeek Harness home (defaults to
        // ~/.dsh); project installs land in the project-dsh root .dsh/.
        return process.env.DSH_HOME || join(home, ".dsh");
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
    case "codex":
      return join(baseDir, ".codex");
    case "copilot":
      return join(baseDir, ".github");
    case "codearts":
      return join(baseDir, ".codeartsdoer");
    case "qoder":
      return isQoderCliLayout() ? join(baseDir, ".qoder") : join(baseDir, ".lingma");
    case "dsh":
      return join(baseDir, ".dsh");
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

export function getSkillsRoot(tool: AITool, level: InstallLevel, base?: string): string {
  if (tool === "codex") {
    if (level === "global") {
      return join(homedir(), ".agents", "skills");
    }
    return join(base || process.cwd(), ".agents", "skills");
  }
  if (tool === "qoder" && isQoderCliLayout() && level === "global") {
    // qodercn discovers user-level skills in ~/.agents/skills (same shared
    // convention as codex), not under ~/.qoder-cn.
    return join(homedir(), ".agents", "skills");
  }
  return join(getConfigRoot(tool, level, base), "skills");
}

export function getAgentsFileName(tool: AITool): string {
  return getConfigFileName(tool);
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
