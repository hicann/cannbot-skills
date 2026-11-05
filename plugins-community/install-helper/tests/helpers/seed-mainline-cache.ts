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

import { mkdirSync, writeFileSync } from "fs";
import { join } from "path";

/** Pre-seeds a valid, TTL-fresh cannbot mainline cache under the sandbox
 *  HOME so discovery never attempts a real `git clone` from unit tests
 *  (hermeticity — no network from the suite). */
export function seedMainlineCache(sandboxHome: string): void {
  const cache = join(sandboxHome, ".cannbot", "repo-cannbot");
  mkdirSync(join(cache, "plugins-official"), { recursive: true });
  writeFileSync(join(cache, ".cannbot-repo-stamp"), String(Date.now()));
}

/** Pre-seeds a valid skills-repo checkout under the sandbox HOME so the
 *  RepositoryManager's ensureRepoAndScan() hits the cache instead of
 *  attempting a real git clone. Requires .git/ and plugins-official/ for
 *  isValidRepo(), plus a fresh scan-cache for the commit-keyed hit. */
export function seedSkillsRepo(sandboxHome: string): void {
  const repo = join(sandboxHome, ".cannbot", "repo");
  mkdirSync(join(repo, ".git"), { recursive: true });
  mkdirSync(join(repo, "plugins-official"), { recursive: true });
  writeFileSync(join(repo, ".git", "HEAD"), "ref: refs/heads/master\n");
}
