// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import { getAllPlugins } from "../core/registry.js";
import { scanInstalled } from "../core/manifest.js";
import { printPluginList } from "../ui/display.js";
import { createRepositoryManager } from "../core/repository.js";

export async function listCommand(): Promise<void> {
  try {
    const repoManager = createRepositoryManager();
    await repoManager.ensureRepoAndScan();
  } catch {
  }

  const plugins = getAllPlugins();
  const installed = scanInstalled();
  const installedMap = new Map(installed.map((p) => [p.id, p]));

  printPluginList(plugins, installedMap);
}
