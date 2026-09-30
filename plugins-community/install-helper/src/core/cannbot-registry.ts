// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

// Adapter for the cannbot installer's own install registry
// (<configDir>/cannbot-plugin.json, schemaVersion 2). install-helper reads
// it to implement list/status and a full uninstall for cannbot-style
// plugins (the upstream CLI only ships `install`). The registry file is the
// source of truth for cannbot-type installs — records written directly by
// the cannbot CLI (no install-helper delegated record) are handled the same
// way.

import { existsSync, readFileSync, readdirSync, rmSync, rmdirSync, unlinkSync, lstatSync, writeFileSync } from "fs";
import { join, resolve, sep, dirname } from "path";
import type { CannbotInstalledEntry, CannbotRegistryFile, CannbotRegistryRecord } from "../types/index.js";
import { deleteRecord, readRecords } from "./record.js";
import { removeInstalledPlugin } from "../utils/config.js";
import { getCannbotConfigDir } from "../utils/paths.js";
import { logger } from "../utils/logger.js";
import { t } from "../utils/i18n.js";

// Mirror of cannbot installer TOOL_DIRECTORIES (script/bin/cannbot.js) for
// the tools install-helper delegates to. configDirs are probed in order; the
// first existing one wins, falling back to defaultConfigDir.
export const CANNBOT_TOOL_LAYOUTS: Record<string, { configDirs: string[]; defaultConfigDir: string }> = {
  opencode: { configDirs: [".opencode"], defaultConfigDir: ".opencode" },
  codex: { configDirs: [".codex"], defaultConfigDir: ".codex" },
  claude: { configDirs: [".claude"], defaultConfigDir: ".claude" },
  trae: { configDirs: [".traecli", ".marscode", ".trae", ".trae-cn"], defaultConfigDir: ".trae" },
};

export const CANNBOT_SUPPORTED_TOOLS = Object.keys(CANNBOT_TOOL_LAYOUTS);

export const CANNBOT_REGISTRY_SCHEMA_VERSION = 2;
const REGISTRY_FILE_NAME = "cannbot-plugin.json";

function toolConfigDir(target: string, tool: string): string {
  const layout = CANNBOT_TOOL_LAYOUTS[tool];
  if (!layout) return join(target, `.${tool}`);
  return layout.configDirs
    .map((name) => join(target, name))
    .find((path) => existsSync(path)) ?? join(target, layout.defaultConfigDir);
}

export function findCannbotRegistryPath(target: string, tool: string): string | null {
  const configDir = toolConfigDir(target, tool);
  const registryPath = join(configDir, REGISTRY_FILE_NAME);
  return existsSync(registryPath) ? registryPath : null;
}

export interface CannbotRegistryReadResult {
  registryPath: string;
  registry: CannbotRegistryFile;
  schemaSupported: boolean;
}

export function readCannbotRegistry(target: string, tool: string): CannbotRegistryReadResult | null {
  const registryPath = findCannbotRegistryPath(target, tool);
  if (!registryPath) return null;

  let registry: CannbotRegistryFile;
  try {
    registry = JSON.parse(readFileSync(registryPath, "utf-8")) as CannbotRegistryFile;
  } catch {
    logger.warn(t("cannbot_registry_corrupted").replace("{file}", registryPath));
    return null;
  }

  if (typeof registry.schemaVersion !== "number" || !Array.isArray(registry.plugins)) {
    logger.warn(t("cannbot_registry_corrupted").replace("{file}", registryPath));
    return null;
  }

  return {
    registryPath,
    registry,
    schemaSupported: registry.schemaVersion === CANNBOT_REGISTRY_SCHEMA_VERSION,
  };
}

function toInstalledEntry(target: string, registryPath: string, record: CannbotRegistryRecord): CannbotInstalledEntry {
  return {
    id: record.plugin,
    version: record.pluginVersion || "",
    tool: record.tool,
    target,
    registryPath,
    skillsCount: Array.isArray(record.skills) ? record.skills.length : 0,
    agentsCount: Array.isArray(record.agents) ? record.agents.length : 0,
    record,
  };
}

/**
 * Scans for cannbot-style installs. cannbot plugins are project-level only,
 * so only `target` (defaults to cwd) is scanned across the supported tools.
 */
export function scanCannbotInstalled(target?: string): CannbotInstalledEntry[] {
  const root = resolve(target || process.cwd());
  const installed: CannbotInstalledEntry[] = [];

  for (const tool of CANNBOT_SUPPORTED_TOOLS) {
    const result = readCannbotRegistry(root, tool);
    if (!result) continue;
    for (const record of result.registry.plugins) {
      if (!record?.plugin) continue;
      installed.push(toInstalledEntry(root, result.registryPath, record));
    }
  }

  return installed;
}

export function findCannbotRecord(pluginId: string, target?: string): CannbotInstalledEntry | null {
  const entries = scanCannbotInstalled(target);
  return entries.find((e) => e.id === pluginId) || null;
}

export function removeMarkerBlock(content: string, pluginName: string): string {
  const start = `<!-- cannbot:${pluginName}:start -->`;
  const end = `<!-- cannbot:${pluginName}:end -->`;
  const pattern = new RegExp(
    `${start.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\n?[\\s\\S]*?${end.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\n?`,
    "g"
  );
  return content.replace(pattern, "");
}

export function countMarkerBlocks(content: string, pluginName: string): number {
  const start = `<!-- cannbot:${pluginName}:start -->`;
  return content.split(start).length - 1;
}

// --- uninstall ---

export interface CannbotUninstallResult {
  success: boolean;
  pluginId: string;
  removedFiles: number;
  removedDirs: number;
  errors: string[];
}

function removeIfSafe(pathToRemove: string, allowedBase: string): "removed" | "missing" | "skipped" {
  const resolved = resolve(pathToRemove);
  if (resolved !== allowedBase && !resolved.startsWith(allowedBase + sep)) {
    return "skipped";
  }
  try {
    const stat = lstatSync(resolved);
    if (stat.isDirectory()) {
      rmSync(resolved, { recursive: true, force: true });
    } else {
      unlinkSync(resolved);
    }
    return "removed";
  } catch {
    return "missing";
  }
}

function removeEmptyDirUpward(dirPath: string, stopAt: string): boolean {
  let current = resolve(dirPath);
  const stop = resolve(stopAt);
  let removedAny = false;
  while (current.startsWith(stop + sep)) {
    try {
      const entries = readdirSync(current);
      if (entries.length > 0) break;
      rmdirSync(current);
      removedAny = true;
    } catch {
      break;
    }
    current = dirname(current);
  }
  return removedAny;
}

function cleanHookSettings(hookSettingsPath: string, target: string, pluginId: string): void {
  // Remove hook entries that point into .cannbot/plugins/<pluginId>/ so a
  // shared settings.json (e.g. .claude/settings.json) keeps serving other
  // plugins. Best effort — malformed files are left untouched.
  const resolved = resolve(target, hookSettingsPath);
  if (!existsSync(resolved)) return;
  let settings: Record<string, unknown>;
  try {
    settings = JSON.parse(readFileSync(resolved, "utf-8"));
  } catch {
    return;
  }
  if (!settings || typeof settings !== "object" || !settings.hooks) return;

  const pluginRootFragment = `.cannbot/plugins/${pluginId}`;
  const cleanHooks = (hooks: unknown): unknown => {
    if (Array.isArray(hooks)) {
      return hooks.filter((entry) => !JSON.stringify(entry).includes(pluginRootFragment));
    }
    if (hooks && typeof hooks === "object") {
      const result: Record<string, unknown> = {};
      for (const [key, value] of Object.entries(hooks as Record<string, unknown>)) {
        const cleaned = cleanHooks(value);
        if (Array.isArray(cleaned) ? cleaned.length > 0 : cleaned !== undefined) {
          result[key] = cleaned;
        }
      }
      return result;
    }
    return hooks;
  };

  try {
    const cleaned = cleanHooks(settings.hooks);
    if (Array.isArray(cleaned) ? cleaned.length === 0 : Object.keys(cleaned as object).length === 0) {
      delete settings.hooks;
    } else {
      settings.hooks = cleaned;
    }
    writeFileSync(resolved, `${JSON.stringify(settings, null, 2)}\n`);
  } catch {
    // leave the file untouched on failure
  }
}

function cleanCodexNativePlugin(record: CannbotRegistryRecord, pluginId: string): void {
  const native = record.nativePlugin;
  if (!native || typeof native !== "object") return;

  if (native.pluginRoot) {
    try {
      rmSync(native.pluginRoot, { recursive: true, force: true });
      logger.step(`  ${t("uninstall_remove_label")}: ${native.pluginRoot}`);
    } catch {
      // best effort
    }
  }

  if (native.marketplacePath && existsSync(native.marketplacePath)) {
    try {
      const marketplace = JSON.parse(readFileSync(native.marketplacePath, "utf-8"));
      if (Array.isArray(marketplace?.plugins)) {
        marketplace.plugins = marketplace.plugins.filter(
          (p: { name?: string }) => !(p && (p.name === pluginId || p.name === native.marketplaceName))
        );
        if (marketplace.plugins.length === 0) {
          unlinkSync(native.marketplacePath);
        } else {
          writeFileSync(native.marketplacePath, `${JSON.stringify(marketplace, null, 2)}\n`);
        }
      }
    } catch {
      // best effort
    }
  }
}

/**
 * The upstream installer registers its npm plugin with the opencode CLI
 * (`opencode plugin @cannbot-plugin/cannbot@<version>`), which writes the
 * spec into <target>/.opencode/opencode.json. Strip that entry so uninstall
 * leaves no residue; the file is removed once nothing remains in it. Only
 * called when no sibling cannbot plugin still needs the registration.
 */
function cleanOpenCodePluginRegistration(root: string): void {
  const configPath = join(root, ".opencode", "opencode.json");
  if (!existsSync(configPath)) return;
  let config: Record<string, unknown>;
  try {
    config = JSON.parse(readFileSync(configPath, "utf-8"));
  } catch {
    return;
  }
  if (!Array.isArray(config.plugin)) return;

  const CLEAN_PACKAGE_PREFIX = "@cannbot-plugin/cannbot";
  config.plugin = config.plugin.filter(
    (entry) => !(typeof entry === "string" && entry.startsWith(CLEAN_PACKAGE_PREFIX))
  );

  try {
    if (Array.isArray(config.plugin) && config.plugin.length === 0) {
      delete config.plugin;
    }
    if (Object.keys(config).length === 0) {
      unlinkSync(configPath);
      removeOpenCodePluginGitignore(dirname(configPath));
    } else {
      writeFileSync(configPath, `${JSON.stringify(config, null, 2)}\n`);
    }
  } catch {
    // leave the file untouched on failure
  }
}

/**
 * `opencode plugin add` also drops a generic .gitignore next to the
 * registration (node_modules/package.json/bun.lock/...). Remove it together
 * with the registration, but only when its content matches the opencode
 * plugin signature so user-authored files are never touched.
 */
function removeOpenCodePluginGitignore(configDir: string): void {
  const gitignorePath = join(configDir, ".gitignore");
  if (!existsSync(gitignorePath)) return;
  try {
    const content = readFileSync(gitignorePath, "utf-8");
    if (content.includes("bun.lock") && content.includes("node_modules")) {
      unlinkSync(gitignorePath);
    }
  } catch {
    // best effort
  }
}

/**
 * Full uninstall of a cannbot-style plugin driven by its registry record.
 * Steps: remove skills/agents, strip the AGENTS.md/CLAUDE.md marker block,
 * remove .cannbot/plugins/<id> assets, shared permissions/settings only when
 * no other cannbot plugin still references them, dependency clones, the
 * codex native plugin, then update the registry itself.
 */
export function uninstallCannbotPlugin(pluginId: string, target?: string): CannbotUninstallResult {
  const result: CannbotUninstallResult = {
    success: false,
    pluginId,
    removedFiles: 0,
    removedDirs: 0,
    errors: [],
  };

  const root = resolve(target || process.cwd());
  let read: CannbotRegistryReadResult | null = null;
  let record: CannbotRegistryRecord | undefined;

  for (const tool of CANNBOT_SUPPORTED_TOOLS) {
    const candidate = readCannbotRegistry(root, tool);
    if (!candidate) continue;
    const found = candidate.registry.plugins.find((r) => r?.plugin === pluginId);
    if (found) {
      read = candidate;
      record = found;
      break;
    }
  }

  if (!read || !record) {
    result.errors.push(t("cannbot_uninstall_no_record"));
    return result;
  }

  if (!read.schemaSupported) {
    result.errors.push(
      t("cannbot_registry_schema_unsupported").replace("{version}", String(read.registry.schemaVersion))
    );
    return result;
  }

  const registryPath = read.registryPath;
  const allowedBase = root;

  // 1. skills + agents (record paths are target-relative)
  for (const relPath of [...(record.skills || []), ...(record.agents || [])]) {
    const absolute = join(root, relPath);
    if (removeIfSafe(absolute, allowedBase) === "removed") {
      result.removedFiles++;
      removeEmptyDirUpward(dirname(absolute), allowedBase);
    }
  }

  // 2. instructions marker block (keep the file for other plugins)
  if (record.instructions) {
    const instructionsPath = join(root, record.instructions);
    const resolved = resolve(instructionsPath);
    if (resolved.startsWith(allowedBase + sep) && existsSync(resolved)) {
      try {
        const content = readFileSync(resolved, "utf-8");
        const cleaned = removeMarkerBlock(content, pluginId);
        if (cleaned.trim().length === 0) {
          unlinkSync(resolved);
          result.removedFiles++;
        } else if (cleaned !== content) {
          writeFileSync(resolved, cleaned);
        }
      } catch (error) {
        result.errors.push(error instanceof Error ? error.message : t("error_unknown"));
      }
    }
  }

  // 3. plugin assets (.cannbot/plugins/<id>)
  if (record.assets) {
    if (removeIfSafe(join(root, record.assets), allowedBase) === "removed") {
      result.removedDirs++;
      removeEmptyDirUpward(join(root, ".cannbot", "plugins"), allowedBase);
    }
  }

  // 4. shared runtime paths — remove only when no sibling plugin uses them
  const siblings = scanCannbotInstalled(root)
    .map((e) => e.record)
    .filter((r) => r.plugin !== pluginId);
  const siblingUses = (rel: string | null) =>
    siblings.some((r) => [r.permissions, r.settings, r.hookSettings].filter(Boolean).includes(rel as string));

  if (record.permissions && !siblingUses(record.permissions)) {
    if (removeIfSafe(join(root, record.permissions), allowedBase) === "removed") {
      result.removedDirs++;
    }
  }
  if (record.settings && !siblingUses(record.settings)) {
    if (removeIfSafe(join(root, record.settings), allowedBase) === "removed") {
      result.removedFiles++;
    }
  }

  // 5. hook settings: surgical removal of this plugin's hook entries
  if (record.hookSettings) {
    cleanHookSettings(record.hookSettings, root, pluginId);
  }

  // 6. dependency clones (.cannbot/dependencies/<id>/...)
  for (const relDep of record.dependencies || []) {
    if (removeIfSafe(join(root, relDep), allowedBase) === "removed") {
      result.removedDirs++;
    }
  }
  if ((record.dependencies || []).length > 0) {
    removeEmptyDirUpward(join(root, ".cannbot", "dependencies", pluginId), allowedBase);
  }

  // 7. codex native plugin (~/plugins/<id> + marketplace registration)
  cleanCodexNativePlugin(record, pluginId);
  if (record.nativePlugin && typeof record.nativePlugin === "object" && record.nativePlugin.package) {
    logger.warn(t("cannbot_uninstall_native_hint").replace("{spec}", String(record.nativePlugin.package)));
  }
  // opencode native plugin registration (opencode.json) — strip it only when
  // the last cannbot plugin for this target is being removed
  if (record.tool === "opencode" && siblings.length === 0) {
    cleanOpenCodePluginRegistration(root);
  }

  // 8. update the registry itself
  try {
    const registry = JSON.parse(readFileSync(registryPath, "utf-8")) as CannbotRegistryFile;
    registry.plugins = registry.plugins.filter((r) => r?.plugin !== pluginId);
    if (registry.plugins.length === 0) {
      unlinkSync(registryPath);
      removeEmptyDirUpward(dirname(registryPath), allowedBase);
    } else {
      writeFileSync(registryPath, `${JSON.stringify(registry, null, 2)}\n`);
    }
  } catch (error) {
    result.errors.push(error instanceof Error ? error.message : t("error_unknown"));
  }

  // 8b. clean layout roots the cannbot installer created but that carry no
  // recorded entries (e.g. an empty agents/ dir for a 0-agent plugin)
  if (CANNBOT_TOOL_LAYOUTS[record.tool]) {
    const configDir = dirname(registryPath);
    const layoutRoots = [join(configDir, "agents"), join(configDir, "skills")];
    if (record.tool === "opencode" || record.tool === "codex") {
      // opencode/codex share the .agents/skills root
      layoutRoots.push(join(root, ".agents", "skills"), join(root, ".agents"));
    }
    for (const dirPath of layoutRoots) {
      if (removeEmptyDirUpward(dirPath, allowedBase)) {
        result.removedDirs++;
      }
    }
  }

  // 9. install-helper side bookkeeping
  deleteRecord(pluginId);
  removeInstalledPlugin(pluginId);

  result.success = result.errors.length === 0;
  return result;
}

/**
 * Consistency check for doctor: delegated records in
 * ~/.cannbot/installs/<id>.json whose cannbot registry entry has vanished.
 * Each location record is checked against the registry at its own
 * install path (a record for another project is not an orphan just because
 * the current project's registry lacks it).
 */
export function findOrphanedDelegatedRecords(target?: string): string[] {
  const orphans: string[] = [];

  const installsDir = join(getCannbotConfigDir(), "installs");
  if (!existsSync(installsDir)) return orphans;

  for (const file of readdirSync(installsDir)) {
    if (!file.endsWith(".json") || file === "skills.json") continue;
    const pluginId = file.replace(/\.json$/, "");
    try {
      // readRecords transparently handles both the v2 location-keyed store
      // and the v1 single-record format
      const records = readRecords(pluginId);
      const delegated = records.filter((r) => r?.kind === "delegated");
      if (delegated.length === 0) continue;

      const isOrphan = delegated.some((rec) => {
        // Records without an install path (legacy fixtures) fall back to the
        // passed target, mirroring the pre-location behavior.
        const recTarget = rec.installPath || target;
        const installedThere = new Set(scanCannbotInstalled(recTarget).map((e) => e.id));
        return !installedThere.has(pluginId);
      });
      if (isOrphan) {
        orphans.push(pluginId);
      }
    } catch {
      // ignore malformed records
    }
  }
  return orphans;
}
