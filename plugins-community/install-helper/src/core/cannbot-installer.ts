// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

// Managed installer for the cannbot repository's unified installer
// (@cannbot-plugin/cannbot). install-helper delegates cannbot-style plugin
// installation to this package so both installers always produce identical
// on-disk layouts (single source of truth for the "resolve + materialize"
// stages of installation).
//
// Lifecycle mirrors RepositoryManager.ensureRepo():
//   - missing  -> npm install <package>@<channel> into the managed prefix
//   - TTL ok   -> reuse the cached copy (offline friendly)
//   - TTL gone -> refresh via npm install (same channel)

import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync, statSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { execa, execaSync } from "execa";
import { parse as parseYaml } from "yaml";
import type { CannbotInstallerConfig, CannbotPluginManifest } from "../types/index.js";
import { getCannbotConfigDir } from "../utils/paths.js";
import { isDirectory } from "../utils/fs-helpers.js";
import { logger } from "../utils/logger.js";
import { t } from "../utils/i18n.js";
import embeddedConfig from "../embedded-config.json" with { type: "json" };

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const DEFAULT_CONFIG: CannbotInstallerConfig = {
  package: "@cannbot-plugin/cannbot",
  channel: "latest",
  ttlHours: 24,
};

const NPM_MIRRORS = [
  undefined, // user's npm config (registry.npmjs.org by default)
  "https://registry.npmmirror.com",
];

export interface CannbotPluginEntry {
  id: string;
  displayName: string;
  version: string;
  description: string;
  skillsCount: number;
  agentsCount: number;
  dir: string;
}

export function loadCannbotInstallerConfig(): CannbotInstallerConfig {
  // Resolution order (mirrors scanner.loadScanConfig):
  //   1. repository.yaml on disk (dist/config/ or src/config/)
  //   2. embedded-config.json (bun-compiled native binary)
  //   3. hardcoded defaults (must not drift from repository.yaml)
  // CANNBOT_INSTALLER_CHANNEL always overrides the channel when set.
  let config: CannbotInstallerConfig = DEFAULT_CONFIG;

  const configCandidates = [
    join(__dirname, "config", "repository.yaml"),
    join(__dirname, "..", "config", "repository.yaml"),
  ];
  for (const configPath of configCandidates) {
    try {
      const parsed = parseYaml(readFileSync(configPath, "utf-8"));
      if (!parsed?.cannbotInstaller) continue;
      config = {
        package: parsed.cannbotInstaller.package || DEFAULT_CONFIG.package,
        channel: parsed.cannbotInstaller.channel || DEFAULT_CONFIG.channel,
        ttlHours: parsed.cannbotInstaller.ttlHours || DEFAULT_CONFIG.ttlHours,
      };
      break;
    } catch {
      // try next candidate
    }
  }

  if (config === DEFAULT_CONFIG) {
    const embedded = embeddedConfig as {
      cannbotInstaller?: { package?: string; channel?: string; ttlHours?: number };
    };
    if (embedded.cannbotInstaller) {
      config = {
        package: embedded.cannbotInstaller.package || DEFAULT_CONFIG.package,
        channel: embedded.cannbotInstaller.channel || DEFAULT_CONFIG.channel,
        ttlHours: embedded.cannbotInstaller.ttlHours || DEFAULT_CONFIG.ttlHours,
      };
    }
  }

  return {
    ...config,
    channel: process.env.CANNBOT_INSTALLER_CHANNEL || config.channel,
  };
}

export function getCannbotInstallerPrefix(): string {
  return join(getCannbotConfigDir(), "installers", "cannbot");
}

export function getCannbotPackageRoot(): string {
  const override = process.env.CANNBOT_INSTALLER_PATH;
  if (override) return override;
  const { package: packageName } = loadCannbotInstallerConfig();
  return join(getCannbotInstallerPrefix(), "node_modules", ...packageName.split("/"));
}

export function getCannbotBinPath(): string {
  return join(getCannbotPackageRoot(), "bin", "cannbot.js");
}

interface InstallerState {
  version: string;
  channel: string;
  installedAt: number;
}

function getStatePath(): string {
  return join(getCannbotInstallerPrefix(), "state.json");
}

function readState(): InstallerState | null {
  try {
    const state = JSON.parse(readFileSync(getStatePath(), "utf-8")) as InstallerState;
    if (!state || typeof state.channel !== "string" || !state.channel ||
        typeof state.installedAt !== "number" || !Number.isFinite(state.installedAt)) return null;
    return state;
  } catch {
    return null;
  }
}

function writeState(state: InstallerState): void {
  const prefix = getCannbotInstallerPrefix();
  if (!existsSync(prefix)) {
    mkdirSync(prefix, { recursive: true });
  }
  writeFileSync(getStatePath(), JSON.stringify(state, null, 2));
}

export function isCannbotInstallerReady(): boolean {
  return existsSync(getCannbotBinPath());
}

export function getCannbotInstallerVersion(): string | null {
  try {
    const pkg = JSON.parse(readFileSync(join(getCannbotPackageRoot(), "package.json"), "utf-8"));
    return pkg.version || null;
  } catch {
    return null;
  }
}

export function hasNodeRuntime(): boolean {
  // Delegation spawns `node`; probe it directly. process.versions.node is
  // unreliable here because bun-compiled binaries also set it.
  try {
    const result = execaSync("node", ["--version"], { timeout: 10000 });
    return result.exitCode === 0;
  } catch {
    return false;
  }
}

async function npmInstallInstaller(config: CannbotInstallerConfig, force: boolean): Promise<boolean> {
  const prefix = getCannbotInstallerPrefix();
  if (!existsSync(prefix)) {
    mkdirSync(prefix, { recursive: true });
  }

  const spec = `${config.package}@${config.channel}`;
  let lastError: unknown = null;
  for (const registry of NPM_MIRRORS) {
    const args = ["install", "--prefix", prefix, "--no-audit", "--no-fund", "--ignore-scripts", spec];
    if (registry) args.push("--registry", registry);
    try {
      await execa("npm", args, { timeout: 300000 });
      return true;
    } catch (error) {
      lastError = error;
      if (force || !isCannbotInstallerReady()) {
        // keep trying mirrors only when we have nothing cached yet
        continue;
      }
      break;
    }
  }
  if (isCannbotInstallerReady() && !force) {
    // refresh failed but a cached copy exists — degrade to the cache
    logger.warn(t("cannbot_installer_refresh_failed"));
    return false;
  }
  throw lastError instanceof Error
    ? lastError
    : new Error(t("error_unknown"));
}

/**
 * Ensures the cannbot installer package is available. Returns the package
 * root. Throws when the package cannot be installed and no cache exists.
 */
export async function ensureCannbotInstaller(options?: { force?: boolean }): Promise<string> {
  const config = loadCannbotInstallerConfig();
  const packageRoot = getCannbotPackageRoot();

  // Explicit override: trust it as-is (offline/test escape hatch). Never
  // npm-install into it.
  if (process.env.CANNBOT_INSTALLER_PATH) {
    if (!isCannbotInstallerReady()) {
      throw new Error(
        t("cannbot_installer_path_invalid").replace("{path}", process.env.CANNBOT_INSTALLER_PATH)
      );
    }
    return packageRoot;
  }

  const ready = isCannbotInstallerReady();
  const state = readState();
  const channelChanged = !!state && state.channel !== config.channel;

  if (!ready || options?.force || channelChanged) {
    await npmInstallInstaller(config, true);
    writeState({
      version: getCannbotInstallerVersion() || "unknown",
      channel: config.channel,
      installedAt: Date.now(),
    });
    return packageRoot;
  }

  const ttlMs = config.ttlHours * 60 * 60 * 1000;
  if (!state || Date.now() - state.installedAt > ttlMs) {
    try {
      const installed = await npmInstallInstaller(config, false);
      if (installed) {
        writeState({
          version: getCannbotInstallerVersion() || "unknown",
          channel: config.channel,
          installedAt: Date.now(),
        });
      }
    } catch {
      // already handled inside npmInstallInstaller (cache fallback / throw)
    }
  }

  return packageRoot;
}

function readManifest(pluginDir: string): CannbotPluginManifest | null {
  try {
    return JSON.parse(readFileSync(join(pluginDir, ".claude-plugin", "plugin.json"), "utf-8"));
  } catch {
    return null;
  }
}

/** Counts immediate subdirectories of a path (skills/ entries). */
function countSubdirs(path: string): number {
  try {
    return readdirSync(path).filter((entry) => isDirectory(join(path, entry))).length;
  } catch {
    return 0;
  }
}

/** Counts .md files in a directory (agents/ entries). */
function countAgentFiles(path: string): number {
  try {
    return readdirSync(path).filter((entry) => entry.endsWith(".md")).length;
  } catch {
    return 0;
  }
}

/** Whether a directory looks like a plugin (has at least a skills/ or
 *  agents/ subdirectory) — filters test fixtures and incidental dirs. */
function looksLikePlugin(pluginDir: string): boolean {
  return existsSync(join(pluginDir, "skills")) || existsSync(join(pluginDir, "agents"));
}

/** Display name as declared by the plugin's own manifest; falls back to the
 *  manifest name (the plugin id) when the field is absent — the display
 *  surface tracks the repository declaration, no local copy to drift. */
function manifestDisplayName(manifest: CannbotPluginManifest): string {
  return typeof manifest.displayName === "string" && manifest.displayName.trim() ? manifest.displayName : manifest.name;
}

/**
 * Discovers cannbot-style plugins bundled inside the managed installer
 * package. Layouts probed (in order):
 *   - dist/plugins            — npm package layout (assemble output; stable
 *                               across the upstream plugins-official rename)
 *   - dist/plugins-official   — defensive: future npm layout change
 *   - plugins / plugins-official / plugins-community — repository checkout
 *                               layouts, for CANNBOT_INSTALLER_PATH pointing
 *                               at a cannbot repo clone (pre/post the
 *                               e140ca6 rename)
 * The npm package is self-contained: skills are copied into dist at pack
 * time, so no cannbot repository checkout or vendor/cannbot-skills
 * submodule is needed. Returns [] when the installer package is not
 * available (graceful degradation for offline environments).
 */
export function discoverCannbotPlugins(): CannbotPluginEntry[] {  if (!isCannbotInstallerReady()) {
    return [];
  }

  const packageRoot = getCannbotPackageRoot();
  const entries: CannbotPluginEntry[] = [];
  // Every layout the installer package may present: the flattened npm dist
  // (dist/plugins), split dist variants and repository checkouts — official
  // and community alike. No dir scope filter: whatever the cannbot installer
  // ships is discoverable and installable.
  const parents = [
    join(packageRoot, "dist", "plugins"),
    join(packageRoot, "dist", "plugins-official"),
    join(packageRoot, "dist", "plugins-community"),
    join(packageRoot, "plugins"),
    join(packageRoot, "plugins-official"),
    join(packageRoot, "plugins-community"),
  ];

  for (const parent of parents) {
    if (!existsSync(parent)) continue;
    let names: string[] = [];
    try {
      names = readdirSync(parent);
    } catch {
      continue;
    }
    for (const name of names) {
      const pluginDir = join(parent, name);
      if (!isDirectory(pluginDir)) continue;
      const manifest = readManifest(pluginDir);
      if (!manifest || typeof manifest.name !== "string") {
        // Layer 2: directory is a cannbot plugin but lacks a manifest
        // declaration (e.g. ops-registry-invoke mid-migration upstream).
        // Derive minimal identity from the directory structure — the entry
        // still shows as [cannbot] with the repository directory name, and
        // the frozen plugins.d yml copy no longer shadows it in dedupe.
        if (looksLikePlugin(pluginDir)) {
          if (entries.some((e) => e.id === name)) continue;
          entries.push({
            id: name,
            displayName: name,
            version: "",
            description: "",
            skillsCount: countSubdirs(join(pluginDir, "skills")),
            agentsCount: countAgentFiles(join(pluginDir, "agents")),
            dir: pluginDir,
          });
        }
        continue;
      }
      if (entries.some((e) => e.id === manifest.name)) continue;
      entries.push({
        id: manifest.name,
        displayName: manifestDisplayName(manifest),
        version: manifest.version || "",
        description: manifest.description || "",
        skillsCount: Array.isArray(manifest.skills) ? manifest.skills.length : 0,
        agentsCount: Array.isArray(manifest.agents) ? manifest.agents.length : 0,
        dir: join(parent, name),
      });
    }
  }

  return entries;
}

/**
 * Bootstrap for the dual-source registry: ensures the managed installer
 * (best effort, never throws — offline environments degrade to the legacy
 * source only) and merges cannbot plugin entries into the plugin registry
 * so findPlugin resolves them with cannbot-first precedence.
 */
// ---------------------------------------------------------------------------
// cannbot mainline repo cache — the npm bootstrap installer installs by
// following the cannbot git mainline at install time (the npm package is
// only a bootstrap). The npm dist snapshot therefore lags new plugins that
// exist upstream; the mainline cache closes that discovery gap: any plugin
// in plugins-official/ or plugins-community/ (with .claude-plugin/
// plugin.json) becomes discoverable and routable to the delegated install.
// ---------------------------------------------------------------------------

const CANNBOT_REPO_URL = "https://gitcode.com/cann/cannbot.git";
const CANNBOT_REPO_TTL_MS = 24 * 60 * 60 * 1000;

export function getCannbotRepoCachePath(): string {
  return join(getCannbotConfigDir(), "repo-cannbot");
}

function readCannbotRepoStamp(): number {
  try {
    const stamp = readFileSync(join(getCannbotRepoCachePath(), ".cannbot-repo-stamp"), "utf-8");
    return Number(stamp) || 0;
  } catch {
    return 0;
  }
}

function writeCannbotRepoStamp(): void {
  try {
    writeFileSync(join(getCannbotRepoCachePath(), ".cannbot-repo-stamp"), String(Date.now()));
  } catch {
    // stamp is best-effort only
  }
}

function isValidCannbotRepo(path: string): boolean {
  return existsSync(join(path, "plugins-official")) || existsSync(join(path, "plugins-community"));
}

/** Ensures the shallow cannbot mainline cache exists and is TTL-fresh.
 * Returns the cache path, or null when the cache is unavailable (offline
 * first run, clone failure on a stale cache kept as-is). */
export async function ensureCannbotRepoCache(): Promise<string | null> {
  const cachePath = getCannbotRepoCachePath();
  const stamp = readCannbotRepoStamp();
  const fresh = Date.now() - stamp < CANNBOT_REPO_TTL_MS;

  if (isValidCannbotRepo(cachePath) && fresh) return cachePath;

  try {
    if (!isValidCannbotRepo(cachePath)) {
      if (existsSync(cachePath)) {
        return null; // unexpected layout — do not touch user data
      }
      mkdirSync(dirname(cachePath), { recursive: true });
      await execa(
        "git",
        ["clone", "--depth", "1", "--single-branch", CANNBOT_REPO_URL, cachePath],
        { timeout: 600000 }
      );
      writeCannbotRepoStamp();
      return cachePath;
    }

    // stale but valid: try to refresh, keep serving the stale cache on failure
    try {
      await execa("git", ["pull", "--quiet"], { cwd: cachePath, timeout: 300000 });
      writeCannbotRepoStamp();
    } catch {
      // keep the stale cache — installation still follows the mainline
    }
    return cachePath;
  } catch {
    return null;
  }
}

function readMainlinePluginVersion(pluginDir: string): string {
  try {
    const manifest = JSON.parse(
      readFileSync(join(pluginDir, ".claude-plugin", "plugin.json"), "utf-8")
    ) as { version?: string };
    return (manifest.version || "").trim();
  } catch {
    return "";
  }
}

function countMainlineSkills(pluginDir: string): number {
  try {
    const sources = JSON.parse(
      readFileSync(join(pluginDir, "plugin-sources.json"), "utf-8")
    ) as { skills?: string[] };
    return Array.isArray(sources.skills) ? sources.skills.length : 0;
  } catch {
    return 0;
  }
}

/** Discovers plugins from the cached cannbot mainline checkout. Mirrors the
 * upstream assemble-plugins rule: a directory with .claude-plugin/plugin.json
 * is a plugin. */
export function discoverMainlinePlugins(repoPath: string): CannbotPluginEntry[] {
  const entries: CannbotPluginEntry[] = [];
  for (const parent of ["plugins-official", "plugins-community"]) {
    const parentPath = join(repoPath, parent);
    if (!existsSync(parentPath)) continue;
    let names: string[] = [];
    try {
      names = readdirSync(parentPath);
    } catch {
      continue;
    }
    for (const name of names) {
      const pluginDir = join(parentPath, name);
      if (!isDirectory(pluginDir)) continue;
      const manifest = readManifest(pluginDir);
      if (!manifest || typeof manifest.name !== "string") {
        // Layer 2: manifest-less cannbot plugin (same as snapshot discovery)
        if (looksLikePlugin(pluginDir)) {
          entries.push({
            id: name,
            displayName: name,
            version: "",
            description: "",
            skillsCount: countSubdirs(join(pluginDir, "skills")),
            agentsCount: countAgentFiles(join(pluginDir, "agents")),
            dir: pluginDir,
          });
        }
        continue;
      }
      entries.push({
        id: manifest.name,
        displayName: manifestDisplayName(manifest),
        version: readMainlinePluginVersion(pluginDir),
        description: manifest.description || name,
        skillsCount: countMainlineSkills(pluginDir),
        agentsCount: 0,
        dir: pluginDir,
      });
    }
  }
  return entries;
}

export async function ensureCannbotDiscovery(): Promise<boolean> {
  try {
    await ensureCannbotInstaller();
  } catch {
    return false;
  }

  // Snapshot first (npm dist), then overlay the mainline cache entries —
  // mainline wins for the same id (freshest), and snapshot-only entries
  // are kept. When the cache is unavailable only the snapshot is used.
  const byId = new Map<string, CannbotPluginEntry>();
  for (const entry of discoverCannbotPlugins()) byId.set(entry.id, entry);

  const repoPath = await ensureCannbotRepoCache();
  if (repoPath) {
    for (const entry of discoverMainlinePlugins(repoPath)) byId.set(entry.id, entry);
  }

  const { setCannbotPlugins } = await import("./registry.js");
  setCannbotPlugins(
    [...byId.values()].map((entry) => ({
      id: entry.id,
      dir: entry.dir,
      displayName: entry.displayName,
      script: "cannbot",
      aliases: [],
      skills: entry.skillsCount,
      agents: entry.agentsCount,
      description: entry.description,
      version: entry.version,
      source: "cannbot" as const,
    }))
  );
  return true;
}
