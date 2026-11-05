// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------

// Runtime derivation of the cannbot installer's declared tool support.
//
// install-helper is a view over the upstream installer (@cannbot-plugin/cannbot);
// the set of tools the delegation supports belongs to the installer and evolves
// with its releases (<= 1.9.x: TOOL_NAMES array, >= 1.10.0: TOOL_DIRECTORIES
// object). Deriving it here — instead of mirroring a hardcoded list — lets
// upstream add tools without an install-helper release. Resolution order:
//   1. contract: machine-readable `cannbot.toolDirectories` field in the
//      installed installer's package.json (future upstream contract; read
//      opportunistically, never assumed to exist)
//   2. directories: the `TOOL_DIRECTORIES = (Object.freeze)?{...}` object
//      literal in bin/cannbot.js (>= 1.10.0) — names + registry layouts
//   3. names: the `TOOL_NAMES = [...]` array literal in bin/cannbot.js
//      (<= 1.9.x) — names only
//   4. none: installer absent or declaration unparseable — callers fall back
//      to the legacy prior and are never worse than pre-derivation behavior.

import { existsSync, readFileSync } from "fs";
import { join } from "path";
import { getCannbotPackageRoot } from "./cannbot-installer.js";

export interface CannbotToolLayout {
  configDirs: string[];
  defaultConfigDir: string;
}

export type CannbotToolsSource = "contract" | "directories" | "names" | "absent" | "unparseable";

export interface DerivedCannbotTools {
  source: CannbotToolsSource;
  /** Declared tool names, or null when undeterminable. */
  tools: string[] | null;
  /** Declared registry layouts, or null when only names (or nothing) were derived. */
  layouts: Record<string, CannbotToolLayout> | null;
}

/** Legacy prior — the pre-derivation hardcoded delegation matrix. Used only
 *  when the installed installer's declaration cannot be determined, so
 *  behavior degrades to (never below) the historic default. */
export const LEGACY_DELEGATE_TOOLS: string[] = ["opencode", "claude", "trae", "codex"];

function parseStringList(raw: string): string[] {
  return raw
    .split(",")
    .map((entry) => entry.trim().replace(/^["']|["']$/g, ""))
    .filter(Boolean);
}

/** Extracts the body of the object literal opened at `startIndex` (which must
 *  point at the opening brace) via quote-aware brace matching. */
function extractObjectBody(source: string, startIndex: number): string | null {
  let depth = 0;
  let cursor = startIndex;
  while (cursor < source.length) {
    const ch = source[cursor];
    if (ch === '"' || ch === "'") {
      const quote = ch;
      cursor += 1;
      while (cursor < source.length && source[cursor] !== quote) {
        if (source[cursor] === "\\") cursor += 1;
        cursor += 1;
      }
      if (cursor >= source.length) return null;
      cursor += 1;
      continue;
    }
    if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) return source.slice(startIndex + 1, cursor);
    }
    cursor += 1;
  }
  return null;
}

function parseDirectoriesDeclaration(
  binSource: string
): { tools: string[]; layouts: Record<string, CannbotToolLayout> } | null {
  const header = binSource.match(/TOOL_DIRECTORIES\s*=\s*(?:Object\.freeze\(\s*)?\{/);
  if (!header || header.index === undefined) return null;
  // header[0] ends at the opening brace
  const body = extractObjectBody(binSource, header.index + header[0].length - 1);
  if (body === null) return null;

  const layouts: Record<string, CannbotToolLayout> = {};
  const entry = /(\w+)\s*:\s*\{([^{}]*)\}/g;
  let match: RegExpExecArray | null;
  while ((match = entry.exec(body)) !== null) {
    const inner = match[2];
    const configDirs = inner.match(/configDirs\s*:\s*\[([^\]]*)\]/);
    const defaultConfigDir = inner.match(/defaultConfigDir\s*:\s*["']([^"']*)["']/);
    if (!configDirs || !defaultConfigDir) continue;
    const dirs = parseStringList(configDirs[1]);
    if (dirs.length === 0 || !defaultConfigDir[1]) continue;
    layouts[match[1]] = { configDirs: dirs, defaultConfigDir: defaultConfigDir[1] };
  }
  const tools = Object.keys(layouts);
  if (tools.length === 0) return null;
  return { tools, layouts };
}

function parseNamesDeclaration(binSource: string): string[] | null {
  const match = binSource.match(/TOOL_NAMES\s*=\s*\[([^\]]*)\]/);
  if (!match) return null;
  const names = parseStringList(match[1]);
  return names.length > 0 ? names : null;
}

function readContractDeclaration(
  packageRoot: string
): { tools: string[]; layouts: Record<string, CannbotToolLayout> } | null {
  try {
    const pkg = JSON.parse(readFileSync(join(packageRoot, "package.json"), "utf-8"));
    const declared = pkg?.cannbot?.toolDirectories;
    if (!declared || typeof declared !== "object" || Array.isArray(declared)) return null;
    const layouts: Record<string, CannbotToolLayout> = {};
    for (const [name, value] of Object.entries(declared)) {
      if (!value || typeof value !== "object") continue;
      const configDirs = Array.isArray(value.configDirs)
        ? value.configDirs.filter((entry): entry is string => typeof entry === "string")
        : [];
      const defaultConfigDir = typeof value.defaultConfigDir === "string" ? value.defaultConfigDir : null;
      if (configDirs.length === 0 || !defaultConfigDir) continue;
      layouts[name] = { configDirs, defaultConfigDir };
    }
    const tools = Object.keys(layouts);
    if (tools.length === 0) return null;
    return { tools, layouts };
  } catch {
    return null;
  }
}

/** Derives the tool support declared by the *installed* cannbot installer.
 *  Synchronous on purpose: delegation boundary checks live on synchronous
 *  call paths (wizard preflight, update routing). */
export function deriveCannbotTools(): DerivedCannbotTools {
  const packageRoot = getCannbotPackageRoot();

  const contract = readContractDeclaration(packageRoot);
  if (contract) return { source: "contract", ...contract };

  const binPath = join(packageRoot, "bin", "cannbot.js");
  let binSource: string | null = null;
  try {
    if (existsSync(binPath)) binSource = readFileSync(binPath, "utf-8");
  } catch {
    binSource = null;
  }
  if (binSource === null) return { source: "absent", tools: null, layouts: null };

  const directories = parseDirectoriesDeclaration(binSource);
  if (directories) return { source: "directories", ...directories };

  const names = parseNamesDeclaration(binSource);
  if (names) return { source: "names", tools: names, layouts: null };

  return { source: "unparseable", tools: null, layouts: null };
}

/** Tool names gating delegation: the installed installer's declaration, or
 *  the legacy prior when the declaration cannot be determined. */
export function delegateBoundaryTools(): string[] {
  return deriveCannbotTools().tools ?? LEGACY_DELEGATE_TOOLS;
}
