// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import type { Plugin } from "@opencode-ai/plugin";
import path from "node:path";

type HookOutput = {
  hookSpecificOutput?: {
    additionalContext?: string;
    decision?: "allow" | "block";
    reason?: string;
  };
};

function appendMessage(output: { output?: string }, message: string): void {
  output.output = `${output.output ?? ""}\n\n${message}`.trim();
}

function formatPluginError(scope: string, error: unknown): string {
  const detail = error instanceof Error ? error.message : String(error);
  return `[pypto-op-lint plugin-error] ${scope} 自动检查失败：${detail}`;
}

function parseHookOutput(raw: string): { additionalContext: string; decision: "allow" | "block"; reason: string } {
  if (!raw.trim()) return { additionalContext: "", decision: "allow", reason: "" };

  try {
    const parsed = JSON.parse(raw) as HookOutput;
    return {
      additionalContext: parsed.hookSpecificOutput?.additionalContext ?? "",
      decision: parsed.hookSpecificOutput?.decision ?? "allow",
      reason: parsed.hookSpecificOutput?.reason ?? "",
    };
  } catch (error) {
    return {
      additionalContext: formatPluginError("post-edit 输出解析", error),
      decision: "allow",
      reason: "",
    };
  }
}

function resolvePath(baseDir: string, filePath: string): string {
  if (!filePath) return "";
  return path.isAbsolute(filePath) ? filePath : path.resolve(baseDir, filePath);
}

function isStateWriteCommand(command: string): boolean {
  if (!command.includes(".orchestrator_state.json")) return false;
  // Split compound commands (&&, ||, ;) and only inspect segments that mention the state file.
  const segments = command.split(/&&|\|\||;/).map(s => s.trim());
  const stateSegs = segments.filter(s => s.includes(".orchestrator_state.json"));
  // If all state-file segments start with a read-only verb, allow the command.
  const readOnly = /^(cat|ls|stat|test|head|tail|grep|wc|file|find|read)\b/;
  if (stateSegs.length > 0 && stateSegs.every(s => readOnly.test(s.replace(/^sudo\s+/, "")))) {
    return false;
  }
  // Otherwise treat as a write attempt.
  return true;
}

// Operator artifact file matcher: <op>_impl.py / <op>_golden.py / test_<op>.py.
// Module variants are also covered (e.g., demo_module1_impl.py, demo_module1_golden.py,
// test_demo_module1.py). Used to detect bash bypass of the post-edit hook —
// Coder must Write/Edit these via OpenCode's tool layer so lint fires.
const OP_ARTIFACT_RE = /\b\w+_(?:impl|golden)\.py\b|\btest_\w+\.py\b/;

function isBashWriteToOpArtifact(command: string): boolean {
  if (!OP_ARTIFACT_RE.test(command)) return false;

  // Strip quoted strings so e.g. `echo "no _impl.py here"` doesn't false-trip later
  // matchers. We DO keep the artifact name visible to the redirect / arg-style
  // detectors below by leaving real shell tokens intact.
  // (Implementation note: we don't fully parse the shell; we look at structural
  // signals that strongly imply a write to an artifact file.)

  // Signal 1: redirect (`>` or `>>`) whose right-hand side is an artifact filename.
  //   Examples: `cat > foo_impl.py <<EOF`, `echo x > test_foo.py`, `printf … >> bar_golden.py`
  if (/>>?\s*\S*(?:\w+_(?:impl|golden)\.py|test_\w+\.py)/.test(command)) {
    return true;
  }

  // Signal 2a: `python -c` is special-cased at WHOLE-COMMAND level.
  //
  // `python -c` is ambiguous — it can read (exec(open(...).read())), import
  // (importlib.spec_from_file_location), or write (open(..., 'w'),
  // .write_text(), etc.). Only block when the body contains an unambiguous
  // write indicator; read / exec / import patterns must remain allowed
  // (verifier and debugger agents routinely invoke them).
  //
  // The check runs against the WHOLE command rather than per-segment
  // because shell-style separators `;`, `&&`, `|` etc. are also Python's
  // statement separator inside the `-c` argument. Splitting on `;` would
  // shred a multi-statement `python -c "from pathlib import Path; Path(...).write_text(...)"`
  // into pieces where neither half carries enough evidence.
  if (/\bpython3?\b\s+(?:-\w+\s+)*-c\b/.test(command)) {
    const writeIndicators = [
      /\.write\w*\(/,                       // .write(, .write_text(, .write_bytes(, .writelines(
      /\.truncate\(/,                       // file.truncate()
      /open\([^)]*['"][wax]b?\+?['"]/,      // open(..., 'w' / 'wb' / 'a' / 'a+' / 'x' / ...)
      /Path\([^)]*\)\.unlink\(/,            // pathlib delete
      /shutil\.(copy\w*|move|rmtree)\(/,    // shutil writes/destructive
      /os\.(remove|unlink|rename)\(/,       // os destructive
    ];
    if (writeIndicators.some((p) => p.test(command))) return true;
    // No write indicator anywhere in the command → python -c is reading
    // / exec'ing / importing the artifact. Allow (fall through to the
    // segment check below for non-python parts of the same command).
  }

  // Signal 2b: write-style shell commands whose argument is an artifact filename.
  //   tee, cp, mv, rm, install, ln — destructive
  //   sed -i — in-place edit
  // Split on shell separators so we only inspect segments that actually mention an
  // artifact. python -c segments are intentionally skipped here (handled above).
  const segments = command.split(/&&|\|\||;|\|/).map((s) => s.trim());
  for (const raw of segments) {
    if (!OP_ARTIFACT_RE.test(raw)) continue;
    const seg = raw.replace(/^sudo\s+/, "");
    if (/^python3?\b.*\s-c\b/.test(seg)) continue;   // handled at whole-command level
    if (/^(tee|cp|mv|rm|install|ln)\b/.test(seg)) return true;
    if (/^sed\b.*-i\b/.test(seg)) return true;
  }
  return false;
}

export const PyptoOpLintPlugin: Plugin = async (input) => {
  const $ = input.$;
  const client = input.client;
  const baseDir = input.directory || input.worktree || process.cwd();
  const lintScript = new URL(
    "../hooks/pypto-op-lint/pypto_op_lint.py",
    import.meta.url,
  ).pathname;

  async function execHookJson(hook: string, payload: unknown): Promise<string> {
    return await $`python3 ${lintScript} --hook ${hook}`
      .env({
        PYPTO_OP_LINT_HOOK_INPUT: JSON.stringify(payload),
      })
      .quiet()
      .text();
  }

  // ─────────────────────────────────────────────────────────────────────
  // Tool name normalization
  //
  // OpenCode SDKs and forks differ on the casing they pass to lifecycle
  // hooks: some emit lowercase ("write" / "edit"), others PascalCase
  // ("Write" / "Edit") to match the user-facing tool API. We normalize
  // by lowercasing once and then compare against a single canonical set.
  // The same applies to extracting the edited file path: OpenCode uses
  // `args.file_path`, Claude Code's plugin shape uses `args.filePath`,
  // and a few helpers pass `args.path`. We try all three.
  // ─────────────────────────────────────────────────────────────────────
  const WRITE_TOOLS = new Set(["write", "edit", "multiedit"]);
  const BASH_TOOLS = new Set(["bash", "shell"]);
  function normTool(t: unknown): string {
    return typeof t === "string"
      ? t.toLowerCase()
      : typeof (t as { name?: unknown })?.name === "string"
        ? String((t as { name?: unknown }).name).toLowerCase()
        : "";
  }
  function extractFilePath(args: any): string {
    return String(
      args?.file_path ?? args?.filePath ?? args?.path ?? args?.target_file ?? ""
    );
  }
  // One-time diagnostic so that if the hook NEVER fires, the absence of
  // this line in the host editor log is itself the signal. If the hook
  // fires but on tools we don't care about, the tool name will appear
  // here so the canonical set above can be extended.
  // Set PYPTO_OP_LINT_TRACE=1 to enable in the host editor environment.
  const trace = process.env.PYPTO_OP_LINT_TRACE === "1";
  function tdebug(label: string, payload: Record<string, unknown>): void {
    if (!trace) return;
    try {
      // eslint-disable-next-line no-console
      console.error(`[pypto-op-lint:trace] ${label}`, JSON.stringify(payload));
    } catch {
      // ignore stringify errors
    }
  }

  // tool.execute.* hooks do not carry agent identity. Remember the latest
  // chat agent by session and activate this workflow only for the classic
  // PyPTO team. This positive allowlist prevents unrelated agents from
  // inheriting classic PyPTO lint behavior.
  const LINT_RUN_AGENTS = new Set([
    "pypto-op-orchestrator",
    "pypto-op-planner",
    "pypto-op-mathematician",
    "pypto-op-architect",
    "pypto-op-designer",
    "pypto-op-coder",
    "pypto-op-verifier",
    "pypto-op-debugger",
    "pypto-op-optimizer",
  ]);
  const agentBySession = new Map<string, string>();
  function rememberAgent(input: { sessionID?: unknown; agent?: unknown }): void {
    if (typeof input.sessionID !== "string" || typeof input.agent !== "string") return;
    agentBySession.set(input.sessionID, input.agent);
  }
  function shouldRunLint(input: { sessionID?: unknown }, toolOutput?: unknown): boolean {
    const sessionID = typeof input.sessionID === "string" ? input.sessionID : "";
    const agent = sessionID ? agentBySession.get(sessionID) : undefined;
    const serialized = JSON.stringify([input, toolOutput]).replaceAll("\\\\", "/");
    const targetsClassicRoot = serialized.includes("custom/");
    // AGENTS.md is loaded by OpenCode's built-in `build` primary. For that
    // identity (and for events without identity), require positive path
    // evidence so unrelated edits do not trigger classic lint.
    // Named classic subagents remain guarded even when a command uses a
    // temporary or basename-only artifact path.
    const primaryOrUnknown = agent === undefined || agent === "build";
    const run = primaryOrUnknown ? targetsClassicRoot : LINT_RUN_AGENTS.has(agent);
    tdebug("gate", { sessionID, agent, run });
    return run;
  }

  return {
    "chat.message": async (chatInput, _output) => {
      rememberAgent(chatInput);
    },
    "chat.params": async (chatInput, _output) => {
      rememberAgent(chatInput);
    },
    "tool.execute.after": async (rawArgs, output) => {
      if (!shouldRunLint(rawArgs)) return;
      const { tool, args } = rawArgs as { tool: unknown; args: any };
      const toolName = normTool(tool);
      const filePath = extractFilePath(args);
      tdebug("after.fired", { rawTool: tool, toolName, filePath });
      if (
        WRITE_TOOLS.has(toolName) &&
        (filePath.endsWith("_impl.py") ||
         filePath.endsWith("_golden.py") ||
         /test_\w+\.py$/.test(filePath))
      ) {
        tdebug("after.match", { toolName, filePath });
        try {
          const raw = await execHookJson("post-edit", {
            tool_input: { file_path: filePath },
          });
          const parsed = parseHookOutput(raw);
          if (parsed.additionalContext) appendMessage(output, parsed.additionalContext);
          if (parsed.decision === "block") {
            throw new Error(parsed.reason || "[pypto-op-lint] 产物写入后门禁未通过");
          }
        } catch (error) {
          // post-edit 为强约束：违规时应阻断本次工具调用
          if (error instanceof Error) {
            throw error;
          }
          throw new Error(formatPluginError("post-edit", error));
        }
        return;
      }

      if (
        BASH_TOOLS.has(toolName) &&
        /python3?\s+.*test_\w+\.py/.test(args?.command ?? "")
      ) {
        try {
          // OpenCode bash tool result 的输出在 output 字段（stdout+stderr 合并），
          // 没有单独的 stderr/exit_code。逐字段安全提取，避免映射错误导致 marker 丢失。
          const result = output as Record<string, unknown>;
          const combinedOutput = String(result.output ?? result.stdout ?? result.text ?? "");
          const stderrText = String(result.stderr ?? "");
          const exitCode = typeof result.exit_code === "number" ? result.exit_code
            : typeof result.code === "number" ? result.code
            : typeof result.exitCode === "number" ? result.exitCode
            : 0;
          const raw = await execHookJson("post-bash", {
            tool_input: { command: args?.command ?? "" },
            tool_result: {
              stdout: combinedOutput,
              stderr: stderrText,
              exit_code: exitCode,
            },
          });
          const context = parseHookOutput(raw).additionalContext;
          if (context) appendMessage(output, context);
        } catch (error) {
          appendMessage(output, formatPluginError("post-bash", error));
        }
      }

      return;
    },

    "tool.execute.before": async (rawArgs, output) => {
      if (!shouldRunLint(rawArgs, output)) return;
      const { tool } = rawArgs as { tool: unknown };
      const toolName = normTool(tool);
      const command = String(output.args?.command ?? "");
      tdebug("before.fired", { rawTool: tool, toolName, hasCmd: !!command });
      // ── bash bypass of post-edit hook ──
      // PostToolUse fires on `tool.execute.after` for write/edit/multiedit only.
      // If the Coder writes a kernel artifact (_impl.py / _golden.py / test_*.py)
      // through bash heredoc, cp/mv/sed -i, tee, python -c, etc., the hook
      // never runs and lint never sees the violation. Reject these up front so
      // Coder is forced to use Write/Edit (which DO fire the hook).
      if (BASH_TOOLS.has(toolName) && isBashWriteToOpArtifact(command)) {
        throw new Error(
          "[pypto-op-lint] 禁止通过 bash/shell 写入算子产物文件 (_impl.py / _golden.py / test_*.py)。\n"
          + "原因: bash 写入不会触发 PostToolUse lint hook，违规不会以 in-band block 反馈给当前 agent，可能拖到 Verifier 阶段才暴露。\n"
          + "处理方式: 请使用 Write / Edit / MultiEdit 工具直接写入；这些工具会经 tool.execute.after 触发 post-edit hook，并把 S0/S1 violation 作为 decision: block 返回。\n"
          + "如果是查看文件 (cat/grep/diff/stat), 请改用对应只读命令; 这条规则只拦截写入。"
        );
      }
      if (BASH_TOOLS.has(toolName) && isStateWriteCommand(command)) {
        throw new Error(
          "[pypto-op-lint] 禁止通过 bash/shell 直接写入 .orchestrator_state.json。"
          + "请使用 state_transition 工具更新状态文件。",
        );
      }

      if (!WRITE_TOOLS.has(toolName)) return;

      const filePath = extractFilePath(output.args ?? {});
      const absPath = resolvePath(baseDir, filePath);
      if (!absPath) return;

      if (absPath.endsWith(`${path.sep}.orchestrator_state.json`)) {
        throw new Error(
          "[pypto-op-lint] 禁止直接修改 .orchestrator_state.json。"
          + "请使用 state_transition 工具进行阶段状态迁移。",
        );
      }

      if (!absPath.endsWith("_impl.py")) return;

      // pre-edit-backup: Stage 5 cleanup 编辑 impl 前 git auto-commit
      try {
        await execHookJson("pre-edit-backup", {
          tool_input: { file_path: absPath },
        });
      } catch (error) {
        await client.app.log({
          body: {
            service: "pypto-op-lint",
            level: "warn",
            message: formatPluginError("pre-edit-backup", error),
            extra: { tool, filePath },
          },
        });
      }
    },

  };
};
