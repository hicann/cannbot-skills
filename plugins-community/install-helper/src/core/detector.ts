// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { execa } from "execa";
import { existsSync } from "fs";
import { homedir } from "os";
import { join } from "path";
import type { AITool, DetectedTool } from "../types/index.js";
import { detectTraeVariant, VALID_TOOLS } from "../utils/paths.js";

async function getCommandVersion(
  cmd: string,
  args: string[] = ["--version"]
): Promise<string | undefined> {
  try {
    const result = await execa(cmd, args, { timeout: 5000 });
    const output = result.stdout.trim();
    const match = output.match(/(\d+\.\d+\.\d+)/);
    return match ? match[1] : output.split("\n")[0];
  } catch {
    return undefined;
  }
}

async function getCommandPath(cmd: string): Promise<string | undefined> {
  try {
    const whichCmd = process.platform === "win32" ? "where" : "which";
    const result = await execa(whichCmd, [cmd], { timeout: 5000 });
    return result.stdout.trim().split("\n")[0];
  } catch {
    return undefined;
  }
}

async function detectOpenCode(): Promise<DetectedTool | undefined> {
  const path = await getCommandPath("opencode");
  if (!path) return undefined;
  const version = await getCommandVersion("opencode");
  return { name: "opencode", version, path };
}

async function detectClaude(): Promise<DetectedTool | undefined> {
  const path = await getCommandPath("claude");
  if (!path) return undefined;
  const version = await getCommandVersion("claude");
  return { name: "claude", version, path };
}

async function detectTrae(): Promise<DetectedTool | undefined> {
  const home = homedir();
  const variant = detectTraeVariant();

  let path: string | undefined;
  if (variant === "ide" && existsSync(join(home, ".trae-cn"))) {
    path = join(home, ".trae-cn");
  } else if (variant === "plugin" && existsSync(join(home, ".marscode"))) {
    path = join(home, ".marscode");
  } else if (variant === "cli" && existsSync(join(home, ".traecli"))) {
    path = join(home, ".traecli");
  } else if (variant === "unknown") {
    const cmdPath = await getCommandPath("trae");
    if (cmdPath) path = cmdPath;
  }

  if (!path) return undefined;
  return { name: "trae", version: variant, path };
}

async function detectCursor(): Promise<DetectedTool | undefined> {
  const path = await getCommandPath("cursor");
  if (path) {
    const version = await getCommandVersion("cursor");
    return { name: "cursor", version, path };
  }

  if (process.platform === "darwin") {
    const appPath = "/Applications/Cursor.app";
    if (existsSync(appPath)) {
      return { name: "cursor", path: appPath };
    }
  }

  return undefined;
}

async function detectCopilot(): Promise<DetectedTool | undefined> {
  const path = await getCommandPath("gh");
  if (!path) return undefined;

  try {
    await execa("gh", ["extension", "list"], { timeout: 5000 });
    return { name: "copilot", path };
  } catch {
    return undefined;
  }
}

async function detectCodeArts(): Promise<DetectedTool | undefined> {
  const home = homedir();
  const configDir = join(home, ".codeartsdoer");
  if (existsSync(configDir)) {
    return { name: "codearts", path: configDir };
  }
  return undefined;
}

export async function detectTools(): Promise<DetectedTool[]> {
  const detectors = [
    detectOpenCode,
    detectClaude,
    detectTrae,
    detectCursor,
    detectCopilot,
    detectCodeArts,
  ];

  const results = await Promise.allSettled(detectors.map((d) => d()));
  const tools: DetectedTool[] = [];

  for (const result of results) {
    if (result.status === "fulfilled" && result.value) {
      tools.push(result.value);
    }
  }

  return tools;
}

export function getToolDisplayName(tool: AITool): string {
  const names: Record<AITool, string> = {
    opencode: "OpenCode",
    claude: "Claude Code",
    trae: "Trae",
    cursor: "Cursor",
    copilot: "GitHub Copilot",
    codearts: "CodeArts",
  };
  return names[tool];
}

export function getAllTools(): AITool[] {
  return VALID_TOOLS;
}
