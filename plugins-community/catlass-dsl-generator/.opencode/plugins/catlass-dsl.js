import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";


const PLUGIN_FILE = fileURLToPath(import.meta.url);
const PLUGIN_ROOT = path.resolve(path.dirname(PLUGIN_FILE), "../..");
const SUBAGENT_MAPPING_MARKER = "CATLASS_DSL_OPENCODE_SUBAGENT_MAPPING_V1";
const SUBAGENT_MAPPING = `[${SUBAGENT_MAPPING_MARKER}]
OpenCode 工具映射：\`Subagent (general-purpose)\` 表示调用 \`task\`，并设置 \`subagent_type: "general"\`。每次实现和 review 使用不同的 task 身份。`;


export function registerSkillsPath(
  config,
  pluginRoot = PLUGIN_ROOT,
  pathExists = fs.existsSync,
) {
  const skillsDir = path.resolve(pluginRoot, "skills");
  if (!pathExists(skillsDir)) {
    throw new Error(`CATLASS DSL skills directory not found: ${skillsDir}`);
  }

  config.skills ??= {};
  config.skills.paths ??= [];
  if (!Array.isArray(config.skills.paths)) {
    throw new TypeError("OpenCode config.skills.paths must be an array");
  }
  if (!config.skills.paths.every((entry) => typeof entry === "string")) {
    throw new TypeError("OpenCode config.skills.paths entries must be strings");
  }

  const normalizedPaths = config.skills.paths.map((entry) => path.resolve(entry));
  const existingIndex = normalizedPaths.indexOf(skillsDir);
  if (existingIndex === -1) {
    config.skills.paths.push(skillsDir);
  } else {
    config.skills.paths[existingIndex] = skillsDir;
  }
  return skillsDir;
}


export function injectSubagentMapping(output) {
  if (!output || !Array.isArray(output.messages) || output.messages.length === 0) {
    return false;
  }
  const firstUser = output.messages.find(
    (message) => message?.info?.role === "user" && Array.isArray(message.parts),
  );
  if (!firstUser || firstUser.parts.length === 0) return false;
  if (firstUser.parts.some(
    (part) => part?.type === "text" && part.text.includes(SUBAGENT_MAPPING_MARKER),
  )) return false;
  const reference = firstUser.parts[0];
  firstUser.parts.unshift({ ...reference, type: "text", text: SUBAGENT_MAPPING });
  return true;
}


export const CatlassDslPlugin = async () => ({
  config: async (config) => {
    registerSkillsPath(config);
  },
  "experimental.chat.messages.transform": async (_input, output) => {
    injectSubagentMapping(output);
  },
});
