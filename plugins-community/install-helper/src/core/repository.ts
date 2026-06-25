// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { existsSync, mkdirSync, writeFileSync, readFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { execa } from "execa";
import { parse as parseYaml } from "yaml";
import { getCannbotConfigDir, getCannbotRepoPath } from "../utils/paths.js";
import {
  scanSkills,
  getCurrentCommit,
  readScanCache,
  writeScanCache,
} from "./scanner.js";
import { initFromScan } from "./skill-registry.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

function loadRepoUrl(): string {
  if (process.env.CANNBOT_REPO_URL) {
    return process.env.CANNBOT_REPO_URL;
  }
  const configPath = join(__dirname, "config", "repository.yaml");
  try {
    const config = parseYaml(readFileSync(configPath, "utf-8"));
    return config.repository?.url || "https://gitcode.com/cann/cannbot-skills.git";
  } catch {
    return "https://gitcode.com/cann/cannbot-skills.git";
  }
}

const REPO_URL = loadRepoUrl();

export class RepositoryManager {
  private repoPath: string | undefined;

  constructor(customPath?: string) {
    this.repoPath = customPath;
  }

  async ensureRepo(): Promise<string> {
    if (this.repoPath && this.isValidRepo(this.repoPath)) {
      return this.repoPath;
    }

    const cwd = process.cwd();
    if (this.isValidRepo(cwd)) {
      this.repoPath = cwd;
      return cwd;
    }

    const envPath = process.env.CANNBOT_REPO_PATH;
    if (envPath && this.isValidRepo(envPath)) {
      this.repoPath = envPath;
      return envPath;
    }

    const cachedPath = getCannbotRepoPath();
    if (this.isValidRepo(cachedPath)) {
      this.repoPath = cachedPath;
      return cachedPath;
    }

    return await this.cloneRepo();
  }

  async ensureRepoAndScan(): Promise<string> {
    const repoPath = await this.ensureRepo();

    const cache = readScanCache();
    if (cache && cache.repoCommit === getCurrentCommit(repoPath)) {
      initFromScan(cache.skills);
      return repoPath;
    }

    const skills = scanSkills(repoPath);
    initFromScan(skills);

    writeScanCache({
      skills,
      repoCommit: getCurrentCommit(repoPath),
      timestamp: Date.now(),
    });

    return repoPath;
  }

  async updateRepo(): Promise<void> {
    const repoPath = await this.ensureRepo();
    try {
      await execa("git", ["pull", "--quiet"], { cwd: repoPath, timeout: 30000 });
    } catch {
      // Ignore update failures
    }
  }

  getRepoPath(): string {
    if (!this.repoPath) {
      throw new Error("Repository not initialized. Call ensureRepo() first.");
    }
    return this.repoPath;
  }

  private isValidRepo(path: string): boolean {
    if (!existsSync(path)) return false;
    const gitDir = join(path, ".git");
    const pluginsDir = join(path, "plugins-official");
    return existsSync(gitDir) && existsSync(pluginsDir);
  }

  private async cloneRepo(): Promise<string> {
    const configDir = getCannbotConfigDir();
    if (!existsSync(configDir)) {
      mkdirSync(configDir, { recursive: true });
    }

    const targetPath = getCannbotRepoPath();
    if (existsSync(targetPath)) {
      if (this.isValidRepo(targetPath)) {
        this.repoPath = targetPath;
        return targetPath;
      }
    }

    try {
      await execa(
        "git",
        ["clone", "--depth", "1", REPO_URL, targetPath],
        { timeout: 120000 }
      );
      this.repoPath = targetPath;
      return targetPath;
    } catch (error) {
      throw new Error(
        `Failed to clone repository: ${error instanceof Error ? error.message : "Unknown error"}`
      );
    }
  }
}

export function createRepositoryManager(customPath?: string): RepositoryManager {
  return new RepositoryManager(customPath);
}
