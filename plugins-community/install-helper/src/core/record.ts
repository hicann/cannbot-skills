// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { existsSync, mkdirSync, readFileSync, unlinkSync, readdirSync } from "fs";
import { join, resolve } from "path";
import { getCannbotConfigDir, getSkillsRoot } from "../utils/paths.js";
import { atomicWriteFileSync } from "../utils/fs.js";
import { isSymlink } from "../utils/fs-helpers.js";
import { logger } from "../utils/logger.js";
import { t } from "../utils/i18n.js";
import type { AITool, InstallLevel, CannbotManifest, SkillBatchRecord, BackupRecordEntry } from "../types/index.js";

export interface InstallRecord {
  pluginId: string;
  displayName: string;
  tool: AITool;
  level: InstallLevel;
  installPath: string;
  configRoot: string;
  installTime: string;
  files: string[];
  directories: string[];
  /** "delegated" = installed via the cannbot installer (@cannbot-plugin/cannbot);
   *  absent/legacy = installed by install-helper itself. */
  kind?: "legacy" | "delegated";
  /** Legacy single-backup field (pre multi-target backups). Kept for
   *  reading old records; new installs populate `backups` instead. */
  backup?: {
    filePath: string;
    fromPluginId: string;
    fromPluginName: string;
    backupTime: string;
  };
  /** Backups created during this install, one per replaced configuration
   *  file. `originalPath` is the restore target so uninstall can return
   *  project-level files to their original locations. */
  backups?: BackupRecordEntry[];
  /** Configuration paths the user chose to keep (never replaced by this
   *  install). Recorded so uninstall never removes them even though they
   *  sit where the plugin's own config file would live. */
  preservedTargets?: string[];
}

/** Location dimensions of an install: the same plugin can be installed at
 *  several locations (different projects, tools or levels) simultaneously —
 *  each keeps its own record so a later install elsewhere never clobbers the
 *  bookkeeping of an earlier one (uninstall would otherwise delete the wrong
 *  location's files). */
export interface RecordLocation {
  tool: AITool;
  level: InstallLevel;
  installPath: string;
}

interface RecordStore {
  version: 2;
  /** key: `${tool}:${level}:${resolved installPath}` */
  locations: Record<string, InstallRecord>;
}

const LEGACY_LOCATION_KEY = "__legacy__";

function getInstallsDir(): string {
  return join(getCannbotConfigDir(), "installs");
}

export function getRecordPath(pluginId: string): string {
  return join(getInstallsDir(), `${pluginId}.json`);
}

function locationKey(loc: RecordLocation): string {
  return `${loc.tool}:${loc.level}:${resolve(loc.installPath)}`;
}

/** Reads the record file, migrating the v1 single-record format (a plain
 *  InstallRecord object) to the v2 location-keyed store in memory. */
function readStore(pluginId: string): RecordStore | null {
  const recordPath = getRecordPath(pluginId);
  if (!existsSync(recordPath)) {
    return null;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(recordPath, "utf-8"));
  } catch {
    logger.warn(t("record_corrupted").replace("{file}", recordPath));
    return null;
  }

  const store = parsed as Partial<RecordStore> & Partial<InstallRecord>;
  if (store && typeof store === "object" && "locations" in store && store.locations) {
    return { version: 2, locations: store.locations as Record<string, InstallRecord> };
  }

  // v1: a single InstallRecord object
  const legacy = parsed as InstallRecord;
  if (legacy && typeof legacy === "object" && typeof legacy.pluginId === "string") {
    const key =
      legacy.tool && legacy.level && legacy.installPath
        ? locationKey(legacy)
        : LEGACY_LOCATION_KEY;
    return { version: 2, locations: { [key]: legacy } };
  }

  logger.warn(t("record_corrupted").replace("{file}", recordPath));
  return null;
}

function writeStore(pluginId: string, store: RecordStore): void {
  const installsDir = getInstallsDir();
  if (!existsSync(installsDir)) {
    mkdirSync(installsDir, { recursive: true });
  }
  atomicWriteFileSync(getRecordPath(pluginId), JSON.stringify(store, null, 2));
}

/** All location records for a plugin (any tool / level / install path). */
export function readRecords(pluginId: string): InstallRecord[] {
  const store = readStore(pluginId);
  return store ? Object.values(store.locations) : [];
}

const byLatest = (a: InstallRecord, b: InstallRecord) =>
  (b.installTime || "").localeCompare(a.installTime || "");

/**
 * The record uninstall/install flows should act on, preferring the one that
 * matches the current working directory:
 *   1. project-level record whose installPath is the cwd (latest first)
 *   2. global-level record (latest first)
 *   3. latest record overall (legacy compatibility: acts on the recorded
 *      location even when run from an unrelated directory)
 */
export function readRecord(pluginId: string): InstallRecord | null {
  const records = readRecords(pluginId);
  if (records.length === 0) {
    return null;
  }

  const cwd = resolve(process.cwd());
  const localMatch = records
    .filter((r) => r.level === "project" && resolve(r.installPath) === cwd)
    .sort(byLatest);
  if (localMatch.length > 0) return localMatch[0];

  const globalMatch = records.filter((r) => r.level === "global").sort(byLatest);
  if (globalMatch.length > 0) return globalMatch[0];

  return [...records].sort(byLatest)[0];
}

export function writeRecord(record: InstallRecord): void {
  const store = readStore(record.pluginId) ?? { version: 2, locations: {} };
  store.locations[locationKey(record)] = record;
  writeStore(record.pluginId, store);
}

/**
 * Removes record locations. With `where`, only the matching location is
 * removed (the file is pruned when the last location goes away); without it,
 * the whole record file is deleted (legacy whole-plugin semantics).
 */
export function deleteRecord(pluginId: string, where?: RecordLocation): void {
  const recordPath = getRecordPath(pluginId);
  if (!existsSync(recordPath)) {
    return;
  }

  if (!where) {
    unlinkSync(recordPath);
    return;
  }

  const store = readStore(pluginId);
  if (!store) {
    return;
  }
  delete store.locations[locationKey(where)];
  if (Object.keys(store.locations).length === 0) {
    unlinkSync(recordPath);
  } else {
    writeStore(pluginId, store);
  }
}

export function scanInstalledFiles(
  pluginId: string,
  displayName: string,
  tool: AITool,
  level: InstallLevel,
  installPath: string,
  configRoot: string,
  manifest: CannbotManifest | null,
  externalRepoNames?: string[],
  configRootConfigLink?: boolean,
  preservedTargets?: string[]
): InstallRecord {
  const files: string[] = [];
  const directories: string[] = [];

  if (manifest) {
    const skillsDir = getSkillsRoot(tool, level, installPath);
    if (existsSync(skillsDir)) {
      directories.push(skillsDir);
      for (const skillName of manifest.installed_skills || []) {
        const skillPath = join(skillsDir, skillName);
        if (existsSync(skillPath) || isSymlink(skillPath)) {
          files.push(skillPath);
        }
      }
    }

    const agentsDir = join(configRoot, "agents");
    if (existsSync(agentsDir)) {
      directories.push(agentsDir);
      for (const agentName of manifest.installed_agents || []) {
        const agentPath = join(agentsDir, agentName);
        const agentPathMd = join(agentsDir, agentName + ".md");
        if (existsSync(agentPath) || isSymlink(agentPath)) {
          files.push(agentPath);
        } else if (existsSync(agentPathMd) || isSymlink(agentPathMd)) {
          files.push(agentPathMd);
        }
      }
    }
  }

  const workflowsLink = join(configRoot, "workflows");
  if (isSymlink(workflowsLink)) {
    files.push(workflowsLink);
  }

  const manifestPath = join(configRoot, "cannbot-manifest.json");
  if (existsSync(manifestPath)) {
    files.push(manifestPath);
  }

  const pluginManifestPath = join(configRoot, `${pluginId}-manifest.json`);
  if (existsSync(pluginManifestPath)) {
    files.push(pluginManifestPath);
  }

  const configFileName = tool === "claude" ? "CLAUDE.md" : "AGENTS.md";
  const configFilePath = level === "project"
    ? join(installPath, configFileName)
    : join(configRoot, configFileName);
  // A preserved (user-kept) file at the config location is NOT owned by this
  // install — recording it would make uninstall delete the user's file.
  if (!preservedTargets?.includes(configFilePath) && (existsSync(configFilePath) || isSymlink(configFilePath))) {
    files.push(configFilePath);
  }

  if (level === "project" && configRootConfigLink !== false) {
    const configRootConfigPath = join(configRoot, configFileName);
    if (configRootConfigPath !== configFilePath &&
        !preservedTargets?.includes(configRootConfigPath) &&
        (existsSync(configRootConfigPath) || isSymlink(configRootConfigPath))) {
      files.push(configRootConfigPath);
    }
  }

  const repoLinks = externalRepoNames && externalRepoNames.length > 0
    ? externalRepoNames
    : ["asc-devkit", "pypto", "tilelang-ascend", "cann-recipes-infer", "cann-samples", "ops-tensor"];
  for (const repoName of repoLinks) {
    const repoLinkPath = join(installPath, repoName);
    if (isSymlink(repoLinkPath)) {
      files.push(repoLinkPath);
    }
    const repoLinkInConfig = join(configRoot, repoName);
    if (isSymlink(repoLinkInConfig)) {
      files.push(repoLinkInConfig);
    }
  }

  return {
    pluginId,
    displayName,
    tool,
    level,
    installPath,
    configRoot,
    installTime: new Date().toISOString(),
    files,
    directories,
  };
}

// === Skill-level install records ===

export interface SkillInstallEntry {
  skills: string[];
  installTime: string;
  batches?: SkillBatchRecord[];
}

export interface SkillInstallRecord {
  [tool: string]: {
    [level: string]: {
      [installPath: string]: SkillInstallEntry;
    };
  };
}

function getSkillRecordPath(): string {
  return join(getInstallsDir(), "skills.json");
}

export function readSkillRecord(): SkillInstallRecord {
  const recordPath = getSkillRecordPath();
  if (!existsSync(recordPath)) {
    return {};
  }
  try {
    const content = readFileSync(recordPath, "utf-8");
    return JSON.parse(content) as SkillInstallRecord;
  } catch {
    logger.warn(t("skill_record_corrupted"));
    return {};
  }
}

export function writeSkillRecord(record: SkillInstallRecord): void {
  const installsDir = getInstallsDir();
  if (!existsSync(installsDir)) {
    mkdirSync(installsDir, { recursive: true });
  }
  const recordPath = getSkillRecordPath();
  atomicWriteFileSync(recordPath, JSON.stringify(record, null, 2));
}

export function addSkillsToRecord(
  skillIds: string[],
  tool: AITool,
  level: InstallLevel,
  installPath: string
): void {
  const record = readSkillRecord();
  if (!record[tool]) record[tool] = {};
  if (!record[tool][level]) record[tool][level] = {};
  if (!record[tool][level][installPath]) {
    record[tool][level][installPath] = { skills: [], installTime: "" };
  }
  const entry = record[tool][level][installPath];
  const now = new Date().toISOString();
  const newSkills: string[] = [];
  for (const id of skillIds) {
    if (!entry.skills.includes(id)) {
      entry.skills.push(id);
      newSkills.push(id);
    }
  }
  entry.installTime = now;
  if (newSkills.length > 0) {
    if (!entry.batches) entry.batches = [];
    entry.batches.push({
      batchId: `batch-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      installedAt: now,
      skills: newSkills,
    });
  }
  writeSkillRecord(record);
}

export function removeSkillsFromRecord(
  skillIds: string[],
  tool: AITool,
  level: InstallLevel,
  installPath: string
): void {
  const record = readSkillRecord();
  if (!record[tool]?.[level]?.[installPath]) return;
  const entry = record[tool][level][installPath];
  entry.skills = entry.skills.filter((id) => !skillIds.includes(id));
  if (entry.batches) {
    for (const batch of entry.batches) {
      batch.skills = batch.skills.filter((id) => !skillIds.includes(id));
    }
    entry.batches = entry.batches.filter((b) => b.skills.length > 0);
    if (entry.batches.length === 0) {
      delete entry.batches;
    }
  }
  if (entry.skills.length === 0) {
    delete record[tool][level][installPath];
    if (Object.keys(record[tool][level]).length === 0) {
      delete record[tool][level];
      if (Object.keys(record[tool]).length === 0) {
        delete record[tool];
      }
    }
  }
  writeSkillRecord(record);
}

export function getInstalledSkills(
  tool: AITool,
  level: InstallLevel,
  installPath: string,
  allowFsScan: boolean = true
): string[] {
  const record = readSkillRecord();
  const recordedSkills = record[tool]?.[level]?.[installPath]?.skills || [];

  const skillsDir = getSkillsRoot(tool, level);

  const fromRecord = recordedSkills.filter((skillId) => {
    const skillPath = join(skillsDir, skillId);
    return existsSync(skillPath) || isSymlink(skillPath);
  });

  if (!allowFsScan) {
    return fromRecord;
  }

  const fromFs: string[] = [];
  if (existsSync(skillsDir)) {
    try {
      const entries = readdirSync(skillsDir);
      for (const entry of entries) {
        if (entry === "." || entry === "..") continue;
        const entryPath = join(skillsDir, entry);
        if (isSymlink(entryPath) && !fromRecord.includes(entry)) {
          fromFs.push(entry);
        }
      }
    } catch {
      // ignore
    }
  }

  return [...fromRecord, ...fromFs];
}

export function getLastBatchSkills(
  tool: AITool,
  level: InstallLevel,
  installPath: string
): string[] | null {
  const record = readSkillRecord();
  const entry = record[tool]?.[level]?.[installPath];
  if (!entry || !entry.batches || entry.batches.length === 0) {
    return null;
  }
  const lastBatch = entry.batches[entry.batches.length - 1];
  const skillsDir = getSkillsRoot(tool, level, installPath);
  return lastBatch.skills.filter((skillId) => {
    const skillPath = join(skillsDir, skillId);
    return existsSync(skillPath) || isSymlink(skillPath);
  });
}
