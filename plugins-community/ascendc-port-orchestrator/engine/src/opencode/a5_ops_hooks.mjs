// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

// a5_ops opencode host-hook adapter.
//
// opencode exposes plugin hooks such as permission.ask and
// tool.execute.before/after. This file maps those events into the
// Claude-Code-style JSON payload consumed by the existing a5_ops hook scripts.
// The canonical check logic stays in Python.

import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const TOOL_MAP = new Map([
  ["agent", "Agent"],
  ["task", "Agent"],
  ["bash", "Bash"],
  ["shell", "Bash"],
  ["edit", "Edit"],
  ["write", "Write"],
  ["multiedit", "MultiEdit"],
  ["read", "Read"],
  ["grep", "Grep"],
  ["glob", "Glob"],
  ["webfetch", "WebFetch"],
]);

const AUTO_ALLOW_AFTER_GUARD = new Set([
  "Agent",
  "Bash",
  "Edit",
  "Write",
  "MultiEdit",
  "Read",
  "Grep",
  "Glob",
  "WebFetch",
]);
const KERNEL_AUTHOR_AGENTS = new Set(["aog-kernel-worker", "aog-kernel-optimizer"]);

function isKernelAuthor(payload) {
  return KERNEL_AUTHOR_AGENTS.has(payload.agent_type || "");
}

function titleToolName(raw) {
  const s = String(raw || "");
  const leaf = s.split(".").pop().split("__").pop().toLowerCase();
  return TOOL_MAP.get(leaf) || s;
}

function titlePermissionName(raw) {
  const s = String(raw || "");
  if (s === "external_directory") return "Read";
  return titleToolName(s);
}

function normalizeToolInput(toolName, args) {
  const input = args && typeof args === "object" ? { ...args } : {};
  if (toolName === "Bash" && input.command == null) {
    input.command = input.cmd || input.shell || "";
  }
  if ((toolName === "Edit" || toolName === "Write" || toolName === "MultiEdit") && input.file_path == null) {
    input.file_path = input.filePath || input.filepath || input.path || input.file || "";
  }
  if (toolName === "Read") {
    if (input.file_path == null) {
      input.file_path = input.filePath || input.filepath || input.path || input.file || input.pattern || "";
    }
    if (input.path == null && input.file_path != null) {
      input.path = input.file_path;
    }
  }
  if ((toolName === "Grep" || toolName === "Glob") && input.path == null && input.file_path != null) {
    input.path = input.file_path;
  }
  return input;
}

function toolArgs(input, output) {
  const candidates = [
    output && output.args,
    input && input.args,
    input && input.input,
    input && input.toolInput,
    input && input.parameters,
    input && input.metadata,
  ];
  for (const candidate of candidates) {
    if (candidate && typeof candidate === "object") {
      return candidate;
    }
  }
  return {};
}

function normalizePermissionInput(toolName, input) {
  const metadata = input && typeof input.metadata === "object" ? { ...input.metadata } : {};
  if (input && input.pattern != null && metadata.pattern == null) {
    metadata.pattern = input.pattern;
  }
  if (toolName === "Bash" && metadata.command == null) {
    metadata.command = metadata.cmd || metadata.shell || "";
  }
  if ((toolName === "Edit" || toolName === "Write" || toolName === "MultiEdit") && metadata.file_path == null) {
    metadata.file_path = metadata.filePath || metadata.filepath || metadata.path || metadata.file || "";
  }
  if ((toolName === "Read" || toolName === "Grep" || toolName === "Glob") && metadata.path == null) {
    metadata.path = metadata.filePath || metadata.filepath || metadata.file_path || metadata.parentDir || metadata.pattern || "";
  }
  return normalizeToolInput(toolName, metadata);
}

function runChecker(projectRoot, scriptRel, payload, timeoutMs) {
  const script = path.join(projectRoot, scriptRel);
  const env = {
    ...process.env,
    CLAUDE_PROJECT_DIR: projectRoot,
    AOG_HARNESS_BACKEND: "opencode",
  };
  const res = spawnSync("python3", [script], {
    input: JSON.stringify(payload),
    encoding: "utf8",
    timeout: timeoutMs,
    env,
  });
  if (res.error) {
    throw new Error(`[a5_ops opencode hook] ${scriptRel} failed: ${res.error.message}`);
  }
  if (res.status && res.status !== 0) {
    const stderr = (res.stderr || "").trim();
    const stdout = (res.stdout || "").trim();
    throw new Error(stderr || stdout || `[a5_ops opencode hook] ${scriptRel} exited ${res.status}`);
  }
}

function basename(p) {
  return String(p || "").split(/[\\/]/).pop();
}

function expectedPybindModuleName(filePath, workspace) {
  const fromWorkspace = String(workspace || process.env.ASCENDC_WORKSPACE || "");
  if (fromWorkspace) {
    return `_${path.basename(fromWorkspace)}_ext`;
  }
  const normalized = String(filePath || "").replace(/\\/g, "/");
  const match = normalized.match(/(?:^|\/)workspace\/([^/]+)\/kernel\/pybind11\.cpp$/);
  return match ? `_${match[1]}_ext` : "";
}

function generatedKernelPath(p) {
  const s = String(p || "");
  const b = basename(s);
  if (/(^|\/)(op_host|op_kernel)\//.test(s)) {
    return true;
  }
  if (["model_new_ascendc.py", "pybind11.cpp", "kernels.cpp", "kernel.h"].includes(b)) {
    return true;
  }
  return /(^|\/)kernel\/[^/]+\.(h|hpp|cpp|cc)$/.test(s);
}

function contentFromToolInput(input) {
  if (!input || typeof input !== "object") return "";
  const chunks = [];
  for (const key of ["content", "file_text", "new_string", "replacement"]) {
    if (typeof input[key] === "string") chunks.push(input[key]);
  }
  if (Array.isArray(input.edits)) {
    for (const edit of input.edits) {
      if (edit && typeof edit.new_string === "string") chunks.push(edit.new_string);
      if (edit && typeof edit.replacement === "string") chunks.push(edit.replacement);
    }
  }
  return chunks.join("\n");
}

function extractWorkspaceFromCommand(command) {
  const s = String(command || "");
  const match = s.match(/(?:^|\s)ASCENDC_WORKSPACE=(?:"([^"]+)"|'([^']+)'|(\S+))/);
  return match ? (match[1] || match[2] || match[3] || "") : "";
}

function findShortInitBufferLine(content) {
  for (const line of String(content || "").split(/\r?\n/)) {
    if (!/\bInitBuffer\s*\(/.test(line)) continue;
    const args = line.match(/\bInitBuffer\s*\(([^)]*)\)/);
    if (!args) continue;
    const commaCount = (args[1].match(/,/g) || []).length;
    if (commaCount < 2) return line.trim();
  }
  return "";
}

function findDynamicInitBufferLine(content) {
  for (const line of String(content || "").split(/\r?\n/)) {
    if (!/\bInitBuffer\s*\(/.test(line)) continue;
    const args = line.match(/\bInitBuffer\s*\(([^)]*)\)/);
    if (!args) continue;
    const parts = args[1].split(",").map((p) => p.trim());
    const bytesExpr = parts.length >= 3 ? parts.slice(2).join(",") : "";
    if (/\b(totalElems?|totalSize|blockSize|count|numel|shapeSize|nElems?)\b/i.test(bytesExpr)) {
      return line.trim();
    }
  }
  return "";
}

function hasAscendCKernelDefinition(content) {
  return /\bextern\s+"C"\s+__global__\s+__aicore__\s+void\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*\{/s.test(String(content || ""));
}

function hasAscendCKernelDeclarationOnly(content) {
  const s = String(content || "");
  return /\bextern\s+"C"\s+__global__\s+__aicore__\s+void\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*;/s.test(s)
    && !hasAscendCKernelDefinition(s);
}

function hasPybindKernelLaunch(content) {
  const s = String(content || "");
  return /\bACLRT_LAUNCH_KERNEL\s*\(/.test(s) || /\baclrtlaunch_[A-Za-z0-9_]+\s*\(/.test(s);
}

function findShortSetGlobalBufferLine(content) {
  for (const line of String(content || "").split(/\r?\n/)) {
    if (!/\bSetGlobalBuffer\s*\(/.test(line)) continue;
    const callText = line.slice(line.indexOf("SetGlobalBuffer"));
    if (!/,/.test(callText)) return line.trim();
  }
  return "";
}

function usesUnqualifiedAscendSymbolsWithoutNamespace(content) {
  const s = String(content || "");
  if (!/\b(?:TPipe|TQue|GlobalTensor|LocalTensor)\b/.test(s)) return false;
  if (/\busing\s+namespace\s+AscendC\s*;/.test(s)) return false;
  if (/\bAscendC::(?:TPipe|TQue|GlobalTensor|LocalTensor)\b/.test(s)) return false;
  return true;
}

function findNonAsciiLine(content) {
  for (const line of String(content || "").split(/\r?\n/)) {
    if (/[^\x00-\x7F]/.test(line)) return line.trim();
  }
  return "";
}

function findOverlappingAlignedBlockDataCopy(content) {
  const s = String(content || "");
  const remainderBlockSplit =
    /\bbase\s*=\s*\(?\s*total_?\s*\/\s*blockNum\s*\)?\s*;/.test(s)
    && /\bremainder\s*=\s*total_?\s*%\s*blockNum\s*;/.test(s)
    && /\bstart\s*=\s*blockIdx\s*\*\s*base\b/.test(s)
    && /\bcount\s*=\s*base\s*\+/.test(s);
  const ceilBlockSizeSplit =
    /\bblockSize\s*=\s*\(\s*total_?\s*\+\s*blockNum\s*-\s*1\s*\)\s*\/\s*blockNum\s*;/.test(s)
    && /\bstart\s*=\s*blockIdx\s*\*\s*blockSize\s*;/.test(s)
    && /\bend\s*=\s*\([^;]*start\s*\+\s*blockSize[^;]*\)\s*\?[^;]*total_?[^;]*:[^;]*start\s*\+\s*blockSize[^;]*;/.test(s)
    && /\bcount\s*=\s*end\s*-\s*start\s*;/.test(s);
  const scalarBlockSplit = remainderBlockSplit || ceilBlockSizeSplit;
  if (!scalarBlockSplit) return "";
  const alignedFromCount =
    /\balignedCount\s*=\s*\([^;]*\bcount\b[^;]*\+[^;]*(?:7|kFP32BlockElems\s*-\s*1)[^;]*\)[^;]*;/.test(s)
    || /\balignedCount\s*=.*\bAlign(?:Up|UP)?[^;]*\bcount\b/s.test(s)
    || /\btileLenAligned\s*=\s*\([^;]*\btileLen\b[^;]*\+[^;]*(?:7|kFP32BlockElems\s*-\s*1)[^;]*\)[^;]*;/.test(s);
  if (!alignedFromCount) return "";
  const copiesAlignedTileFromScalarStart =
    /\bDataCopy\s*\([^;]*\[\s*start\s*\+\s*offset\s*\][^;]*,\s*[^;]*\btileSize\b[^;]*\)\s*;/.test(s)
    || /\bDataCopy\s*\([^;]*,\s*[^;]*\[\s*start\s*\+\s*offset\s*\][^;]*,\s*[^;]*\btileSize\b[^;]*\)\s*;/.test(s)
    || /\bDataCopy\s*\([^;]*\[\s*start\s*\+\s*offset\s*\][^;]*,\s*[^;]*\btileLenAligned\b[^;]*\)\s*;/.test(s)
    || /\bDataCopy\s*\([^;]*,\s*[^;]*\[\s*start\s*\+\s*offset\s*\][^;]*,\s*[^;]*\btileLenAligned\b[^;]*\)\s*;/.test(s);
  return copiesAlignedTileFromScalarStart
    ? "scalar block split with count-rounded DataCopy at start+offset"
    : "";
}

function isProjectWideRecursiveGlob(pattern, searchPath, projectRoot) {
  const p = String(pattern || "");
  if (!p.includes("**/")) return false;
  const root = String(searchPath || "").trim();
  if (!root || root === "." || root === projectRoot) return true;
  const resolved = path.resolve(root);
  return resolved === projectRoot;
}

function activeWorkspaceRoot() {
  const ws = process.env.ASCENDC_WORKSPACE || process.env.CLAUDE_ACTIVE_WORKSPACE || "";
  return ws ? path.resolve(ws) : "";
}

function referencesOtherWorkspacePath(value, projectRoot) {
  const text = String(value || "");
  const active = activeWorkspaceRoot();
  const root = path.resolve(projectRoot);
  const workspaceRoot = path.join(root, "workspace");
  const patterns = [
    /(?:^|[\s'"`(])((?:\.\/)?workspace\/[^\s'"`)]+)/g,
    new RegExp(`(?:^|[\\s'"\\\`(])(${workspaceRoot.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\/[^\\s'"\\\`)]+)`, "g"),
  ];
  for (const re of patterns) {
    for (const match of text.matchAll(re)) {
      const raw = match[1] || "";
      const resolved = path.resolve(root, raw);
      if (!resolved.startsWith(workspaceRoot + path.sep)) continue;
      if (active && (resolved === active || resolved.startsWith(active + path.sep))) continue;
      return raw;
    }
  }
  return "";
}

function runInlineAccessGuard(projectRoot, payload) {
  if (!isKernelAuthor(payload)) return;
  const input = payload.tool_input || {};
  const tool = payload.tool_name;

  if (tool === "Glob") {
    const pattern = String(input.pattern || input.file_path || "");
    const searchPath = String(input.path || "");
    const otherWorkspace = referencesOtherWorkspacePath(`${searchPath} ${pattern}`, projectRoot);
    if (otherWorkspace) {
      throw new Error(`[a5_ops opencode hook] access guard blocked cross-workspace glob by aog-kernel-worker: ${otherWorkspace}`);
    }
    if (/(^|\/)output\//.test(pattern) || /(^|\/)output\//.test(searchPath)) {
      throw new Error("[a5_ops opencode hook] access guard blocked kernel-worker glob over output/ archives; use current workspace, KB, SDK headers, and source inputs");
    }
    if (isProjectWideRecursiveGlob(pattern, searchPath, projectRoot)) {
      throw new Error(`[a5_ops opencode hook] access guard blocked project-wide recursive glob '${pattern}' from aog-kernel-worker; scope Glob to ASCENDC_WORKSPACE or an explicit source directory`);
    }
    if (/\b(pass_[ab]_runner|verification\.json|tested_[A-Za-z0-9_]+|model_new_[A-Za-z0-9_]+\.py)\b/.test(pattern) && !searchPath) {
      throw new Error(`[a5_ops opencode hook] access guard blocked answer-bearing glob '${pattern}' without an explicit non-output search path`);
    }
  }

  if (tool === "Grep") {
    const searchPath = String(input.path || input.file_path || "");
    const otherWorkspace = referencesOtherWorkspacePath(searchPath, projectRoot);
    if (otherWorkspace) {
      throw new Error(`[a5_ops opencode hook] access guard blocked cross-workspace grep by aog-kernel-worker: ${otherWorkspace}`);
    }
    if (/(^|\/)output\//.test(searchPath)) {
      throw new Error("[a5_ops opencode hook] access guard blocked kernel-worker grep over output/ archives");
    }
  }

  if (tool === "Read") {
    const filePath = String(input.file_path || input.path || "");
    const otherWorkspace = referencesOtherWorkspacePath(filePath, projectRoot);
    if (otherWorkspace) {
      throw new Error(`[a5_ops opencode hook] access guard blocked cross-workspace read by aog-kernel-worker: ${otherWorkspace}`);
    }
  }

  if (tool === "Bash") {
    const command = String(input.command || "");
    const otherWorkspace = referencesOtherWorkspacePath(command, projectRoot);
    if (otherWorkspace) {
      throw new Error(`[a5_ops opencode hook] access guard blocked cross-workspace Bash access by aog-kernel-worker: ${otherWorkspace}`);
    }
    if (/(^|[\s'"`])(?:\.\/|\/[^\s'"`]*)?(op_host|op_kernel)(?:\/|[\s'"`]|$)/.test(command)) {
      throw new Error("[a5_ops opencode hook] access guard blocked op_host/op_kernel Bash access in direct-pybind kernel-worker mode");
    }
    if (/a5_exec\.py\b[\s\S]*\bdocker\s+exec\b/.test(command)) {
      throw new Error("[a5_ops opencode hook] runtime guard blocked nested docker exec through a5_exec.py; a5_exec.py already runs inside the configured A5 container");
    }
  }
}

function runBuildArtifactGuard(payload) {
  if (!isKernelAuthor(payload)) return;
  if (payload.tool_name !== "Bash") return;
  const command = String((payload.tool_input || {}).command || "");
  if (/\bdeploy_to_npu(?:_lane)?\.sh\b/.test(command) && /\|\s*(tail|head|grep|sed|awk)\b/.test(command)) {
    throw new Error("[a5_ops opencode hook] build guard blocked deploy: do not pipe deploy_to_npu*.sh output; pipes mask exit status and can hang post-build sync");
  }
  if (/\bdeploy_to_npu(?:_lane)?\.sh\b/.test(command) && /\bunpiped\b/.test(command)) {
    throw new Error("[a5_ops opencode hook] build guard blocked deploy: run deploy_to_npu*.sh with direct output; do not append unsupported marker words such as unpiped");
  }
  if (!/deploy_to_npu_lane\.sh\b/.test(command) || !/--build\b/.test(command)) return;

  const workspace = extractWorkspaceFromCommand(command) || process.env.ASCENDC_WORKSPACE || "";
  if (!workspace) {
    throw new Error("[a5_ops opencode hook] build guard blocked deploy: ASCENDC_WORKSPACE is required for op workspace validation");
  }
  const pybindPath = path.join(workspace, "kernel", "pybind11.cpp");
  const modelPath = path.join(workspace, "model_new_ascendc.py");
  if (!fs.existsSync(pybindPath)) {
    throw new Error(`[a5_ops opencode hook] build guard blocked deploy: missing ${pybindPath}; build_ascendc.py does not auto-generate pybind11.cpp`);
  }
  if (!fs.existsSync(modelPath)) {
    throw new Error(`[a5_ops opencode hook] build guard blocked deploy: missing ${modelPath}`);
  }

  const pybind = fs.readFileSync(pybindPath, "utf8");
  const model = fs.readFileSync(modelPath, "utf8");
  const kernelDir = path.join(workspace, "kernel");
  const kernelFiles = fs.readdirSync(kernelDir)
    .filter((name) => /\.(h|hpp|cpp|cc)$/.test(name))
    .map((name) => path.join(kernelDir, name));
  let kernelDefinitionSeen = false;
  for (const file of kernelFiles) {
    const name = path.basename(file);
    const content = fs.readFileSync(file, "utf8");
    if (/\bcoreCoord_t\b/.test(content)) {
      throw new Error(`[a5_ops opencode hook] build guard blocked deploy: ${file} uses unsupported coreCoord_t; use GetBlockIdx()/GetBlockNum() scalars`);
    }
    if (/\b(?:IN_QUE_NUM|OUT_QUE_NUM)\b/.test(content)) {
      throw new Error(`[a5_ops opencode hook] build guard blocked deploy: ${file} uses undefined IN_QUE_NUM/OUT_QUE_NUM queue constants`);
    }
    if (/\bWaitAllDone\s*\(/.test(content)) {
      throw new Error(`[a5_ops opencode hook] build guard blocked deploy: ${file} uses unsupported TQue::WaitAllDone()`);
    }
    if (/\bGlobalTensor\s*</.test(content) && !/\bSetGlobalBuffer\s*\(/.test(content)) {
      throw new Error(`[a5_ops opencode hook] build guard blocked deploy: ${file} declares GlobalTensor but never calls SetGlobalBuffer`);
    }
    if (/^\s*TQue<[^;]+>\s+g_[A-Za-z_][A-Za-z0-9_]*\s*;/m.test(content)) {
      throw new Error(`[a5_ops opencode hook] build guard blocked deploy: ${file} declares file-scope TQue queues; keep queues inside the kernel operator object`);
    }
    const shortGm = findShortSetGlobalBufferLine(content);
    if (shortGm) {
      throw new Error(`[a5_ops opencode hook] build guard blocked deploy: ${file} calls GlobalTensor::SetGlobalBuffer without an element-count argument: ${shortGm}`);
    }
    const shortInit = findShortInitBufferLine(content);
    if (shortInit) {
      throw new Error(`[a5_ops opencode hook] build guard blocked deploy: ${file} calls TPipe::InitBuffer without a byte-size argument: ${shortInit}`);
    }
    const dynamicInit = findDynamicInitBufferLine(content);
    if (dynamicInit) {
      throw new Error(`[a5_ops opencode hook] build guard blocked deploy: ${file} allocates UB buffer from dynamic full input size; use a fixed tile byte-size and loop over chunks: ${dynamicInit}`);
    }
    if (/\bpipe\s*\.\s*Barrier\s*\(/.test(content)) {
      throw new Error(`[a5_ops opencode hook] build guard blocked deploy: ${file} calls unsupported pipe.Barrier(); use TQue EnQue/DeQue or documented PipeBarrier APIs`);
    }
    if (usesUnqualifiedAscendSymbolsWithoutNamespace(content)) {
      throw new Error(`[a5_ops opencode hook] build guard blocked deploy: ${file} uses unqualified AscendC symbols without using namespace AscendC or AscendC:: qualification`);
    }
    if (/\.(cpp|cc)$/.test(name) && name !== "pybind11.cpp") {
      if (hasAscendCKernelDefinition(content)) {
        kernelDefinitionSeen = true;
      }
      if (hasAscendCKernelDeclarationOnly(content)) {
        throw new Error(`[a5_ops opencode hook] build guard blocked deploy: ${file} declares an AscendC kernel but does not define its body`);
      }
      const includes = [...content.matchAll(/#include\s+"([^"]+)"/g)].map((m) => m[1]);
      for (const includeName of includes) {
        if (includeName === "kernel_operator.h" || includeName.startsWith("aclrtlaunch_")) continue;
        if (!fs.existsSync(path.join(kernelDir, includeName))) {
          throw new Error(`[a5_ops opencode hook] build guard blocked deploy: ${file} includes missing local header ${includeName}`);
        }
      }
    }
  }
  if (!kernelDefinitionSeen) {
    throw new Error("[a5_ops opencode hook] build guard blocked deploy: kernel/*.cpp lacks an extern \"C\" __global__ __aicore__ kernel definition");
  }
  const moduleMatch = pybind.match(/PYBIND11_MODULE\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,/);
  if (!moduleMatch) {
    throw new Error(`[a5_ops opencode hook] build guard blocked deploy: ${pybindPath} lacks literal PYBIND11_MODULE(_<op>_ext, m)`);
  }
  const moduleName = moduleMatch[1];
  const expectedModule = expectedPybindModuleName(pybindPath, workspace);
  if (!moduleName.startsWith("_") || !moduleName.endsWith("_ext") || (expectedModule && moduleName !== expectedModule)) {
    throw new Error(`[a5_ops opencode hook] build guard blocked deploy: pybind module ${moduleName} must use exact _<op>_ext naming${expectedModule ? ` (${expectedModule})` : ""}`);
  }
  if (!model.includes("kernel") || !model.includes("build") || !/sys\.path\.(insert|append)\s*\(/.test(model)) {
    throw new Error(`[a5_ops opencode hook] build guard blocked deploy: ${modelPath} must add workspace/<op>/kernel/build to sys.path before importing the extension`);
  }
  if (/\bfrom\s+kernel\s+import\b/.test(model)) {
    throw new Error(`[a5_ops opencode hook] build guard blocked deploy: ${modelPath} must import ${moduleName} from kernel/build, not from package kernel`);
  }
  const escapedModule = moduleName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const importPattern = new RegExp(`\\bimport\\s+${escapedModule}\\b`);
  const fromImportPattern = new RegExp(`\\bfrom\\s+${escapedModule}\\s+import\\s+`);
  if (!importPattern.test(model) && !fromImportPattern.test(model)) {
    throw new Error(`[a5_ops opencode hook] build guard blocked deploy: ${modelPath} must import ${moduleName}, matching PYBIND11_MODULE`);
  }
  const callsModuleWrapper = /\.run_[A-Za-z0-9_]+\s*\(/.test(model)
    || (fromImportPattern.test(model) && /\brun_[A-Za-z0-9_]+\s*\(/.test(model));
  if (!callsModuleWrapper) {
    throw new Error(`[a5_ops opencode hook] build guard blocked deploy: ModelNew.forward must call the pybind run_<op> wrapper`);
  }
}

function runInlineGeneratedCodeGuard(payload) {
  if (!isKernelAuthor(payload)) return;
  if (!["Write", "Edit", "MultiEdit"].includes(payload.tool_name)) return;
  const input = payload.tool_input || {};
  const filePath = input.file_path || input.path || "";
  if (!generatedKernelPath(filePath)) return;
  const b = basename(filePath);
  if (/(^|\/)(op_host|op_kernel)\//.test(String(filePath))) {
    throw new Error(
      `[a5_ops opencode hook] generated-code guard blocked ${filePath}: direct pybind benchmark tasks must not create op_host/ or op_kernel/ scaffold`
    );
  }
  if (["kernel.h", "kernels.cpp", "pybind11.cpp"].includes(b) && !/(^|\/)kernel\//.test(String(filePath))) {
    throw new Error(
      `[a5_ops opencode hook] generated-code guard blocked ${filePath}: kernel sources must live under workspace/<op>/kernel/ for deploy sync`
    );
  }
  const content = contentFromToolInput(input);
  if (!content) return;

  const checks = [];
  if (b === "model_new_ascendc.py") {
    checks.push(
      [/return\s+[A-Za-z_]\w*\s*[-+*/]\s*[A-Za-z_]\w*/m, "model_new_ascendc.py host arithmetic fallback"],
      [/\btorch\.(add|sub|mul|div|matmul|mm|bmm|sum|mean|max|min|sort|topk|where)\s*\(/, "model_new_ascendc.py torch compute fallback"],
      [/torch\.nn\.functional\./, "model_new_ascendc.py torch functional fallback"],
      [/\bnumpy\b|\bnp\./, "model_new_ascendc.py numpy fallback"],
    );
  }
  if (["pybind11.cpp", "kernels.cpp"].includes(b) || /(^|\/)kernel\/[^/]+\.(cpp|cc)$/.test(String(filePath))) {
    checks.push(
      [/\baclrtLaunchKernel\s*\(/, "direct aclrtLaunchKernel call; use auto-generated aclrtlaunch_* wrapper via ACLRT_LAUNCH_KERNEL"],
      [/#include\s*<torch_npu\/csrc\/aten\/common\/ACLRT(?:Launch|Lauch)Kernel\.h>/, "non-portable torch_npu ACLRT macro header; use generated aclrtlaunch_<kernel>.h or an explicit extern aclrtlaunch_<kernel> stub"],
      [/#include\s*<pybind11\/strict_rcward\.h>/, "invalid pybind11 strict_rcward header"],
      [/#include\s*<torch\/npu\.h>/, "non-project torch/npu.h include; use torch_npu NPUStream header for current stream"],
      [/\bpy::object\b/, "pybind wrapper uses py::object; use torch::Tensor or at::Tensor for NPU tensors"],
      [/\bpy::tensor\b/, "pybind wrapper uses py::tensor; use torch::Tensor or at::Tensor for NPU tensors"],
      [/\bpy::array_t\b/, "CPU pybind array fallback"],
      [/\btorch::kCPU\b|\bc10::DeviceType::CPU\b|\.device\s*\(\s*torch::kCPU\s*\)/, "CPU tensor allocation/device in generated pybind; output must stay on NPU"],
      [/\b(?:static\s+)?uint32_t\s+run_[A-Za-z0-9_]*\s*\(/, "pybind run_<op> wrapper returns launch status instead of output tensor"],
      [/\bretistory\b|\blaunchRetistory\b|\bstatusistory\b/, "pybind launch-status check references an invented status variable"],
      [/reinterpret_cast\s*<\s*GM_ADDR\s*>\s*\(\s*&/, "host stack pointer passed as GM_ADDR tiling/workspace"],
      [/\b(?:uint64_t|int64_t|uint32_t|int32_t)\s+[A-Za-z_][A-Za-z0-9_]*\s*\[[^\]]+\]\s*=\s*\{[^;]*\}[\s\S]*reinterpret_cast\s*<\s*uint64_t\s*>\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\)/, "host stack array passed as GM tiling/workspace; create a NPU tensor and pass its data_ptr"],
      [/\baclrtlaunch_[A-Za-z0-9_]+\s*\([^;{)]*\buint64_t\b[^;{)]*\)\s*;/s, "aclrtlaunch_<kernel> host stub uses uint64_t GM addresses; generated stubs take void* tensor data_ptr arguments"],
      [/(\bkernel_module_t\b|\bKernelAddParams\b|\bKERNEL_STATUS_SUCCESS\b)/, "OPP/PR4778 op_kernel registration scaffold in direct pybind path"],
      [/std::vector<[^>]+>\s+\w+_h\b|for\s*\([^)]*\)\s*\{[^{}]*(out|result)[^{}]*=/s, "host-side compute loop in generated binding"],
      [/\btorch::(add|sub|mul|div|matmul|mm|bmm|sum|mean|max|min|sort|topk)\s*\(/, "torch C++ compute fallback"],
    );
    if (b === "pybind11.cpp") {
      checks.push(
        [/\b__gm__\b/, "pybind host wrapper must not use device-side __gm__ pointer qualifiers; use void*, uint8_t*, or ordinary host pointer types for launch stubs"],
      );
    }
    if (b === "kernels.cpp") {
      checks.push(
        [/\bPYBIND11_MODULE\s*\(/, "kernels.cpp must hold AscendC kernel/source glue, not a pybind module"],
        [hasAscendCKernelDeclarationOnly, "kernels.cpp declares an AscendC kernel but does not define its body"],
      );
    }
  }
  if (b === "pybind11.cpp" && /PYBIND11_MODULE\s*\(/.test(content)) {
    if (/\bpy::tensor\b/.test(content)) {
      throw new Error(`[a5_ops opencode hook] generated-code guard blocked ${filePath}: pybind wrapper uses py::tensor; use torch::Tensor or at::Tensor for NPU tensors`);
    }
    if (/\bpy::object\b/.test(content)) {
      throw new Error(`[a5_ops opencode hook] generated-code guard blocked ${filePath}: pybind wrapper uses py::object; use torch::Tensor or at::Tensor for NPU tensors`);
    }
    if (/#include\s*<torch\/pybind\.h>/.test(content)) {
      throw new Error(`[a5_ops opencode hook] generated-code guard blocked ${filePath}: invalid torch/pybind.h header; use torch/extension.h`);
    }
    const pybindChecks = [
      [/\bm\.def\s*\([^;]*&\s*ACLRT_LAUNCH_KERNEL\s*\(/s, "pybind exposes ACLRT_LAUNCH_KERNEL directly; wrap it in a run_<op> function"],
      [hasPybindKernelLaunch, "missing aclrtlaunch_<kernel> or ACLRT_LAUNCH_KERNEL launch in pybind wrapper"],
      [/\bgetCurrentNPUStream\s*\(/, "missing c10_npu::getCurrentNPUStream() stream handoff"],
      [/#include\s*[<"]torch_npu\/csrc\/core\/npu\/NPUStream\.h[>"]/, "missing torch_npu NPUStream header for c10_npu::getCurrentNPUStream()"],
      [/\b(?:torch|at)::empty(?:_like)?\s*\(/, "missing NPU output allocation before launch"],
      [/\bm\.def\s*\(\s*["']run_/, "pybind module must expose a run_<op> wrapper function"],
    ];
    const moduleMatch = content.match(/PYBIND11_MODULE\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,/);
    const expectedModule = expectedPybindModuleName(filePath, "");
    if (moduleMatch && expectedModule && moduleMatch[1] !== expectedModule) {
      throw new Error(`[a5_ops opencode hook] generated-code guard blocked ${filePath}: pybind module ${moduleMatch[1]} must be ${expectedModule}`);
    }
    for (const [pattern, reason] of pybindChecks) {
      const present = typeof pattern === "function" ? Boolean(pattern(content)) : pattern.test(content);
      const mustReject = reason.startsWith("pybind exposes") ? present : !present;
      if (mustReject) {
        throw new Error(`[a5_ops opencode hook] generated-code guard blocked ${filePath}: ${reason}`);
      }
    }
  }
  if (/\.(h|hpp|cpp|cc)$/.test(b) || /(^|\/)kernel\/[^/]+\.(h|hpp|cpp|cc)$/.test(String(filePath))) {
    checks.push(
      [/\bOPENVINO_HIDDEN\b/, "OpenVINO token in AscendC kernel"],
      [/\b__opencl__\b/, "OpenCL kernel qualifier in AscendC kernel"],
      [/\bKernelTensor\b/, "non-project KernelTensor API in AscendC kernel"],
      [/reinterpret_cast\s*<\s*__gm__\s+\w+\s*\*\s*>\s*\(\s*(offset|idx|index)\s*\)/, "fake GM pointer reconstructed from numeric offset instead of saved GM_ADDR base"],
      [/#include\s+" +ascendc\//, "malformed ascendc include path"],
      [findNonAsciiLine, "non-ASCII text in generated C/C++ source"],
      [/\/\/\s*\.\.\.|\/\*\s*\.\.\.|write process logic|TODO(?:\b|_)/i, "placeholder/TODO left in generated C/C++ source"],
      [/\b\w+\s*--\s*\)/, "post-decrement expression in tile/count calculation; use explicit arithmetic"],
      [/\b[A-Za-z_][A-Za-z0-9_]*(?:onge|istory|apse)\b/, "hallucinated identifier suffix in generated C/C++ source"],
      [/\bpipe_\s*\.\s*(?:EnQue|DeQue)\s*\(/, "TPipe has no EnQue/DeQue queue operations; use TQue::EnQue/DeQue on LocalTensor values"],
      [/\bGetTPipe\s*\(/, "unsupported GetTPipe() in generated kernel; keep a TPipe member inside the kernel operator object and call pipe_.InitBuffer(...)"],
      [/\bextern\s+"C"\s+__global__\s+__aicore__\s+void\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*__gm__\s+\w+\s*\*/s, "kernel entry parameters must use GM_ADDR; cast to __gm__ pointers inside the operator Init"],
      [/\bperBlock\s*=\s*\(\s*total_?\s*\+\s*blockNum\s*-\s*1\s*\)\s*\/\s*blockNum\s*;/, "unaligned per-block ceil division for fp32 DataCopy; use blockDim=8 for the simple smoke or round per-block/tile counts to 8 elements and handle tails explicitly"],
      [findOverlappingAlignedBlockDataCopy, "overlapping aligned DataCopy block partition; do not scalar-split total/blockNum and then round each block count to 8 at start+offset"],
      [/\b[A-Za-z_][A-Za-z0-9_]*\s*\.\s*DeQue\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\)/, "TQue::DeQue takes no LocalTensor argument; store `queue.DeQue<T>()` exactly once"],
      [/\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*EnQue\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*;\s*\1\s*\.\s*EnQue\s*\(\s*\2\s*\)\s*;/s, "same LocalTensor enqueued twice without an intervening DeQue"],
      [/\bDataCopy\s*\(\s*[A-Za-z_][A-Za-z0-9_]*Queue[A-Za-z0-9_]*\s*,/i, "DataCopy destination must be LocalTensor, not a TQue queue"],
      [/\bDataCopy\s*\([^,]+,\s*[A-Za-z_][A-Za-z0-9_]*Queue[A-Za-z0-9_]*\s*,/i, "DataCopy source must be LocalTensor/GlobalTensor, not a TQue queue"],
      [/\bFreeTensor\s*\(\s*[A-Za-z_][A-Za-z0-9_]*Queue[A-Za-z0-9_]*\s*\.\s*DeQue\s*</, "do not DeQue inside FreeTensor; store the DeQue result once and free that LocalTensor"],
      [/\bInitBuffer\s*\(\s*[A-Za-z_][A-Za-z0-9_]*Tbuf_\s*,\s*\d+\s*,/i, "TBuf used as a queue buffer in TPipe::InitBuffer; initialize TQue queues for the DataCopy/Add pipeline"],
      [/\bepilogue_len\b|\b(?:Add|Sub|Mul|Div|Muls|Adds)\s*\([^;]*,\s*[A-Za-z_][A-Za-z0-9_]*\s*[-+*/]\s*[A-Za-z_][A-Za-z0-9_]*\s*\)/, "vector intrinsic count uses invented tail arithmetic; pass the current tile count directly"],
      [/\busing\s+namespace\s+ascendc\b/, "lowercase ascendc namespace"],
      [/#ifndef\s+(?:__)?KERNEL_OPERATOR_H(?:__)?\b/, "kernel_operator.h header guard collision"],
      [/^\s*TQue<[^;]+>\s+g_[A-Za-z_][A-Za-z0-9_]*\s*;/m, "file-scope TQue queues; keep queues inside the kernel operator object"],
      [/\bpipe\s*\.\s*Barrier\s*\(/, "unsupported pipe.Barrier(); use TQue EnQue/DeQue or documented PipeBarrier APIs"],
      [findShortSetGlobalBufferLine, "GlobalTensor::SetGlobalBuffer without an element-count argument"],
      [usesUnqualifiedAscendSymbolsWithoutNamespace, "unqualified AscendC symbols without using namespace AscendC or AscendC:: qualification"],
      [findShortInitBufferLine, "TPipe::InitBuffer without a byte-size argument"],
      [findDynamicInitBufferLine, "TPipe::InitBuffer uses dynamic full-input size; use fixed tile byte-size and loop over chunks"],
    );
    if (/\.(h|hpp)$/.test(b)) {
      checks.push(
        [/\bextern\s+"C"\s+__global__\s+__aicore__\s+void\b/, "kernel entry definition/declaration belongs in kernels.cpp, not kernel.h"],
        [/}\s*\/\/\s*namespace\s+AscendC/, "do not close or define namespace AscendC in generated kernel.h; use `using namespace AscendC;` or explicit `AscendC::`"],
      );
    }
  }
  for (const [pattern, reason] of checks) {
    const matched = typeof pattern === "function" ? pattern(content) : pattern.test(content);
    if (matched) {
      throw new Error(`[a5_ops opencode hook] generated-code guard blocked ${filePath}: ${reason}`);
    }
  }
}

function runGuardSet(projectRoot, payload) {
  runInlineAccessGuard(projectRoot, payload);
  runBuildArtifactGuard(payload);
  runInlineGeneratedCodeGuard(payload);
  const tool = payload.tool_name;
  if (["Agent", "Edit", "Write", "MultiEdit", "Bash", "WebFetch"].includes(tool)) {
    runChecker(projectRoot, "src/scripts/workflow/workflow_critic.py", payload, 30000);
  }
  if (["Read", "Grep", "Glob", "Bash"].includes(tool)) {
    runChecker(projectRoot, "src/scripts/workflow/output_read_guard.py", payload, 10000);
  }
  if (tool.startsWith("mcp__plugin_discord_discord__")) {
    runChecker(projectRoot, "src/scripts/workflow/ship_claim_audit.py", payload, 5000);
  }
}

function hookPayload(hookEventName, input, output) {
  const toolName = titleToolName(input.tool || input.type || input.permission || output.tool);
  const toolInput = normalizeToolInput(toolName, toolArgs(input, output));
  return {
    hook_event_name: hookEventName,
    tool_name: toolName,
    tool_input: toolInput,
    session_id: input.sessionID,
    call_id: input.callID,
    agent_id: process.env.AOG_HOOK_AGENT_ID || "",
    agent_type: process.env.AOG_HOOK_AGENT_TYPE || "",
    cwd: process.cwd(),
  };
}

function permissionPayload(input) {
  const toolName = titlePermissionName(input.type || input.permission);
  return {
    hook_event_name: "PreToolUse",
    tool_name: toolName,
    tool_input: normalizePermissionInput(toolName, input),
    session_id: input.sessionID,
    call_id: input.callID,
    agent_id: process.env.AOG_HOOK_AGENT_ID || "",
    agent_type: process.env.AOG_HOOK_AGENT_TYPE || "",
    cwd: process.cwd(),
  };
}

export async function A5OpsHooksPlugin(ctx, options = {}) {
  const projectRoot = path.resolve(String(options.projectRoot || process.env.AOG_PROJECT_ROOT || ctx.directory));
  return {
    async "permission.ask"(input, output) {
      try {
        const payload = permissionPayload(input);
        runGuardSet(projectRoot, payload);
        if (AUTO_ALLOW_AFTER_GUARD.has(payload.tool_name)) {
          output.status = "allow";
        }
      } catch (err) {
        output.status = "deny";
        output.message = err instanceof Error ? err.message : String(err);
      }
    },

    async "tool.execute.before"(input, output) {
      const payload = hookPayload("PreToolUse", input, output);
      runGuardSet(projectRoot, payload);
    },

    async "tool.execute.after"(input, output) {
      const payload = hookPayload("PostToolUse", input, { args: output.args || {} });
      if (payload.tool_name === "Agent") {
        runChecker(projectRoot, "src/scripts/workflow/workflow_critic.py", payload, 30000);
      }
    },
  };
}

export default A5OpsHooksPlugin;
