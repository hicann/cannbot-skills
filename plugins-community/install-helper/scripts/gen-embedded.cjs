// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

const fs = require("fs");
const path = require("path");
const yaml = require("yaml");

const pluginsDir = path.join(__dirname, "..", "plugins.d");
const defaultsPath = path.join(pluginsDir, "_defaults.yml");

let defaults = {};
try {
  defaults = yaml.parse(fs.readFileSync(defaultsPath, "utf-8")) || {};
} catch {}

const plugins = [];
for (const file of fs.readdirSync(pluginsDir)) {
  if (file.startsWith("_") || !file.endsWith(".yml")) continue;
  try {
    const content = yaml.parse(fs.readFileSync(path.join(pluginsDir, file), "utf-8"));
    if (!content || !content.id) continue;
    plugins.push({
      id: content.id,
      dir: content.dir,
      displayName: content.displayName || content.id,
      script: content.script || defaults.script || "init.sh",
      aliases: content.aliases || [],
      skills: content.skills ?? defaults.skills ?? 0,
      agents: content.agents ?? defaults.agents ?? 0,
      description: content.description || "",
      version: content.version,
      configFile: content.configFile,
      configRootConfigLink: content.configRootConfigLink,
      installSkills: content.installSkills,
      installAgents: content.installAgents,
      externalRepos: content.externalRepos,
    });
  } catch {}
}
plugins.sort((a, b) => (a.displayName < b.displayName ? -1 : a.displayName > b.displayName ? 1 : 0));

const outPath = path.join(__dirname, "..", "src", "embedded-plugins.json");
fs.writeFileSync(outPath, JSON.stringify(plugins, null, 2));
console.log(`Generated embedded-plugins.json with ${plugins.length} plugins`);
