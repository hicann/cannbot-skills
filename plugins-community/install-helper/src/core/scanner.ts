// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { existsSync, readdirSync, readFileSync, statSync, writeFileSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { parse as parseYaml } from "yaml";
import { getCannbotConfigDir } from "../utils/paths.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export interface ScannedSkill {
  id: string;
  description: string;
  source: string;
  filePath: string;
}

export interface ScanCache {
  skills: ScannedSkill[];
  repoCommit: string;
  timestamp: number;
}

const EXCLUDE_DIRS = new Set([
  "node_modules",
  ".git",
  "asc-devkit",
  "cann-samples",
  "tilelang-ascend",
  ".agents",
  ".opencode",
  ".claude",
  ".claude-plugin",
  "dist",
  "build",
  "references",
  "hooks",
  "operators",
  "workflows",
  "tests",
  "docs",
  "scripts",
]);

function loadScanConfig(): { skillDirs: string[]; pluginDirs: string[] } {
  const configPath = join(__dirname, "config", "repository.yaml");
  try {
    const config = parseYaml(readFileSync(configPath, "utf-8"));
    return {
      skillDirs: config.scanDirs || ["ops", "model", "graph", "infra", "ops-lab"],
      pluginDirs: config.pluginDirs || ["plugins-official", "plugins-community"],
    };
  } catch {
    return {
      skillDirs: ["ops", "model", "graph", "infra", "ops-lab"],
      pluginDirs: ["plugins-official", "plugins-community"],
    };
  }
}

const scanConfig = loadScanConfig();
const SKILL_SCAN_DIRS = scanConfig.skillDirs;
const PLUGIN_SCAN_DIRS = scanConfig.pluginDirs;

export function scanSkills(repoPath: string): ScannedSkill[] {
  const skills: ScannedSkill[] = [];
  const seen = new Set<string>();

  for (const dir of SKILL_SCAN_DIRS) {
    const fullPath = join(repoPath, dir);
    if (!existsSync(fullPath)) continue;
    scanDirectory(fullPath, repoPath, dir, skills, seen);
  }

  for (const pluginDir of PLUGIN_SCAN_DIRS) {
    const pluginParent = join(repoPath, pluginDir);
    if (!existsSync(pluginParent)) continue;
    
    for (const plugin of safeReaddir(pluginParent)) {
      const pluginPath = join(pluginParent, plugin);
      if (!isDirectory(pluginPath)) continue;

      const skillsDir = join(pluginPath, "skills");
      if (existsSync(skillsDir)) {
        scanDirectory(skillsDir, repoPath, `${pluginDir}/${plugin}/skills`, skills, seen);
      }

      const skillDir = join(pluginPath, "skill");
      if (existsSync(skillDir)) {
        scanDirectory(skillDir, repoPath, `${pluginDir}/${plugin}/skill`, skills, seen);
      }
    }
  }

  return skills;
}

export function parseFrontmatter(filePath: string): { name: string; description: string } | null {
  try {
    const content = readFileSync(filePath, "utf-8");
    const lines = content.split("\n");
    
    if (lines.length < 3) return null;
    if (lines[0].trim() !== "---") return null;

    let endIndex = -1;
    for (let i = 1; i < lines.length; i++) {
      if (lines[i].trim() === "---") {
        endIndex = i;
        break;
      }
    }

    if (endIndex === -1) return null;

    const yamlBlock = lines.slice(1, endIndex).map(l => l.replace(/\r$/, "")).join("\n");
    const parsed = parseYaml(yamlBlock);

    if (!parsed || typeof parsed !== "object") return null;
    if (!parsed.name || typeof parsed.name !== "string") return null;

    return {
      name: parsed.name.trim(),
      description: (parsed.description || "").trim(),
    };
  } catch {
    return null;
  }
}

export function getCurrentCommit(repoPath: string): string {
  try {
    const headFile = join(repoPath, ".git", "HEAD");
    if (!existsSync(headFile)) return "unknown";
    
    const headContent = readFileSync(headFile, "utf-8").trim();
    if (headContent.startsWith("ref: ")) {
      const refPath = join(repoPath, ".git", headContent.slice(5));
      if (existsSync(refPath)) {
        return readFileSync(refPath, "utf-8").trim();
      }
      // Fallback to packed-refs for shallow clones
      const packedRefsPath = join(repoPath, ".git", "packed-refs");
      if (existsSync(packedRefsPath)) {
        const packedRefs = readFileSync(packedRefsPath, "utf-8");
        const refName = headContent.slice(5);
        const lines = packedRefs.split("\n");
        for (const line of lines) {
          if (line.endsWith(refName)) {
            return line.split(" ")[0];
          }
        }
      }
    }
    return headContent;
  } catch {
    return "unknown";
  }
}

export function readScanCache(): ScanCache | null {
  try {
    const cachePath = getCachePath();
    if (!existsSync(cachePath)) return null;
    
    const content = readFileSync(cachePath, "utf-8");
    const cache = JSON.parse(content) as ScanCache;
    
    if (!cache.skills || !cache.timestamp) return null;
    
    const age = Date.now() - cache.timestamp;
    if (age > 24 * 60 * 60 * 1000) return null;
    
    return cache;
  } catch {
    return null;
  }
}

export function writeScanCache(cache: ScanCache): void {
  try {
    const cachePath = getCachePath();
    const cacheDir = dirname(cachePath);
    if (!existsSync(cacheDir)) {
      mkdirSync(cacheDir, { recursive: true });
    }
    writeFileSync(cachePath, JSON.stringify(cache, null, 2));
  } catch {
  }
}

function getCachePath(): string {
  return join(getCannbotConfigDir(), "scan-cache.json");
}

function scanDirectory(
  dirPath: string,
  repoPath: string,
  sourcePrefix: string,
  skills: ScannedSkill[],
  seen: Set<string>
): void {
  // Check if dirPath itself contains SKILL.md (for skill/ directories)
  const selfSkillMd = join(dirPath, "SKILL.md");
  if (existsSync(selfSkillMd)) {
    const frontmatter = parseFrontmatter(selfSkillMd);
    if (frontmatter && !seen.has(frontmatter.name)) {
      seen.add(frontmatter.name);
      skills.push({
        id: frontmatter.name,
        description: frontmatter.description,
        source: sourcePrefix,
        filePath: selfSkillMd,
      });
    }
    return; // Don't recurse into subdirectories
  }

  for (const entry of safeReaddir(dirPath)) {
    const fullPath = join(dirPath, entry);
    
    if (EXCLUDE_DIRS.has(entry)) continue;
    
    if (!isDirectory(fullPath)) continue;

    const skillMd = join(fullPath, "SKILL.md");
    if (existsSync(skillMd)) {
      const frontmatter = parseFrontmatter(skillMd);
      if (frontmatter && !seen.has(frontmatter.name)) {
        seen.add(frontmatter.name);
        skills.push({
          id: frontmatter.name,
          description: frontmatter.description,
          source: sourcePrefix,
          filePath: skillMd,
        });
      }
      continue;
    }

    scanDirectory(fullPath, repoPath, sourcePrefix, skills, seen);
  }
}

function safeReaddir(dir: string): string[] {
  try {
    return readdirSync(dir);
  } catch {
    return [];
  }
}

function isDirectory(path: string): boolean {
  try {
    return statSync(path).isDirectory();
  } catch {
    return false;
  }
}
