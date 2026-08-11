// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, expect, test } from "bun:test";
import { $ } from "bun";
import { PyptoOpLintPlugin } from "../pypto-op-lint";

async function rememberAgent(
  plugin: Awaited<ReturnType<typeof PyptoOpLintPlugin>>,
  agent: string,
  sessionID = "test-session",
) {
  await plugin["chat.params"]?.(
    { sessionID, agent } as never,
    {} as never,
  );
}

// ─── tool.execute.before tests ───

test("blocks direct state write via write tool", async () => {
  const plugin = await PyptoOpLintPlugin({
    $,
    client: { app: { log: async () => {} } },
    directory: process.cwd(),
    worktree: process.cwd(),
    project: {},
  } as never);

  const before = plugin["tool.execute.before"];
  await expect(
    before?.(
      { tool: "write" } as never,
      { args: { file_path: "/tmp/custom/qat/.orchestrator_state.json", content: "{}" } } as never,
    ),
  ).rejects.toThrow("禁止直接修改 .orchestrator_state.json");
});

test("blocks direct state write via bash tool", async () => {
  const plugin = await PyptoOpLintPlugin({
    $,
    client: { app: { log: async () => {} } },
    directory: process.cwd(),
    worktree: process.cwd(),
    project: {},
  } as never);

  const before = plugin["tool.execute.before"];
  await expect(
    before?.(
      { tool: "bash" } as never,
      { args: { command: "echo '{}' > /tmp/custom/qat/.orchestrator_state.json" } } as never,
    ),
  ).rejects.toThrow("禁止通过 bash/shell 直接写入");
});

test("no watcher handler registered", async () => {
  const plugin = await PyptoOpLintPlugin({
    $,
    client: { app: { log: async () => {} } },
    directory: process.cwd(),
    worktree: process.cwd(),
    project: {},
  } as never);

  expect(plugin["file.watcher.updated"]).toBeUndefined();
});

test("skips lint for built-in agents without a classic target", async () => {
  const plugin = await PyptoOpLintPlugin({
    $,
    client: { app: { log: async () => {} } },
    directory: process.cwd(),
    worktree: process.cwd(),
    project: {},
  } as never);
  const before = plugin["tool.execute.before"];

  for (const agent of ["build", "plan", "general", "explore", "scout"]) {
    const sessionID = `session-${agent}`;
    await rememberAgent(plugin, agent, sessionID);
    await expect(
      before?.(
        { tool: "bash", sessionID } as never,
        { args: { command: "echo '{}' > /tmp/qat/.orchestrator_state.json" } } as never,
      ),
    ).resolves.toBeUndefined();
  }
});

test("build primary activates lint only for the custom output root", async () => {
  const plugin = await PyptoOpLintPlugin({
    $,
    client: { app: { log: async () => {} } },
    directory: process.cwd(),
    worktree: process.cwd(),
    project: {},
  } as never);
  await rememberAgent(plugin, "build", "build-primary");
  const before = plugin["tool.execute.before"];

  await expect(
    before?.(
      { tool: "bash", sessionID: "build-primary" } as never,
      { args: { command: "echo x > custom/foo/foo_impl.py" } } as never,
    ),
  ).rejects.toThrow("禁止通过 bash/shell 写入算子产物文件");

  await expect(
    before?.(
      { tool: "bash", sessionID: "build-primary" } as never,
      { args: { command: "echo x > custom/foo/test_foo.py" } } as never,
    ),
  ).rejects.toThrow("禁止通过 bash/shell 写入算子产物文件");
});

test("skips lint for PyPTO-Pro agents", async () => {
  const plugin = await PyptoOpLintPlugin({
    $,
    client: { app: { log: async () => {} } },
    directory: process.cwd(),
    worktree: process.cwd(),
    project: {},
  } as never);
  const before = plugin["tool.execute.before"];

  for (const agent of [
    "pypto-pro-op-orchestrator",
    "pypto-pro-op-planner",
    "pypto-pro-op-mathematician",
    "pypto-pro-op-architect",
    "pypto-pro-op-coder",
    "pypto-pro-op-verifier",
  ]) {
    const sessionID = `session-${agent}`;
    await rememberAgent(plugin, agent, sessionID);
    await expect(
      before?.(
        { tool: "bash", sessionID } as never,
        { args: { command: "echo '{}' > /tmp/qat/.orchestrator_state.json" } } as never,
      ),
    ).resolves.toBeUndefined();
  }
});

test("runs lint for PyPTO subagents", async () => {
  const plugin = await PyptoOpLintPlugin({
    $,
    client: { app: { log: async () => {} } },
    directory: process.cwd(),
    worktree: process.cwd(),
    project: {},
  } as never);
  await rememberAgent(plugin, "pypto-op-coder", "pypto-child");

  const before = plugin["tool.execute.before"];
  await expect(
    before?.(
      { tool: "bash", sessionID: "pypto-child" } as never,
      { args: { command: "cat > custom/foo/foo_impl.py <<EOF\nimport pypto\nEOF" } } as never,
    ),
  ).rejects.toThrow("禁止通过 bash/shell 写入算子产物文件");
});

test("agent skip state is isolated by session", async () => {
  const plugin = await PyptoOpLintPlugin({
    $,
    client: { app: { log: async () => {} } },
    directory: process.cwd(),
    worktree: process.cwd(),
    project: {},
  } as never);
  await rememberAgent(plugin, "build", "build-session");
  await rememberAgent(plugin, "pypto-op-orchestrator", "pypto-session");

  const before = plugin["tool.execute.before"];
  await expect(
    before?.(
      { tool: "bash", sessionID: "build-session" } as never,
      { args: { command: "echo '{}' > /tmp/qat/.orchestrator_state.json" } } as never,
    ),
  ).resolves.toBeUndefined();
  await expect(
    before?.(
      { tool: "bash", sessionID: "pypto-session" } as never,
      { args: { command: "echo '{}' > /tmp/qat/.orchestrator_state.json" } } as never,
    ),
  ).rejects.toThrow("禁止通过 bash/shell 直接写入");
});

test("allows readonly bash commands mentioning state file", async () => {
  const plugin = await PyptoOpLintPlugin({
    $,
    client: { app: { log: async () => {} } },
    directory: process.cwd(),
    worktree: process.cwd(),
    project: {},
  } as never);

  const before = plugin["tool.execute.before"];

  // cat、ls、stat、grep 等只读命令不应被拦截
  const readonlyCommands = [
    "cat /tmp/qat/.orchestrator_state.json",
    "ls -la /tmp/qat/.orchestrator_state.json",
    "stat /tmp/qat/.orchestrator_state.json",
    "grep current_stage /tmp/qat/.orchestrator_state.json",
    "test -f /tmp/qat/.orchestrator_state.json",
  ];

  for (const cmd of readonlyCommands) {
    // 只读命令不应抛出异常
    await expect(
      before?.(
        { tool: "bash" } as never,
        { args: { command: cmd } } as never,
      ),
    ).resolves.toBeUndefined();
  }
});

// ── bash bypass of post-edit hook on operator artifact files ──

test("blocks bash heredoc/redirect writing to *_impl.py", async () => {
  const plugin = await PyptoOpLintPlugin({
    $, client: { app: { log: async () => {} } },
    directory: process.cwd(), worktree: process.cwd(), project: {},
  } as never);
  await rememberAgent(plugin, "pypto-op-coder");
  const before = plugin["tool.execute.before"];

  const blockedWrites = [
    "cat > custom/foo/foo_impl.py <<EOF\nimport pypto\nEOF",
    "echo 'x' > foo_impl.py",
    "printf 'x' >> custom/foo/modules/foo_module1_impl.py",
    "tee custom/foo/foo_impl.py < src.py",
    "cp /tmp/draft.py custom/foo/foo_impl.py",
    "mv tmp.py foo_module12_impl.py",
    "rm foo_impl.py",
    "sed -i 's/old/new/' foo_impl.py",
    "cat > test_foo.py <<EOF\nEOF",
    "echo x > custom/foo/foo_golden.py",
    // python -c with explicit write indicators
    "python3 -c \"open('foo_impl.py','w').write(SRC)\"",
    "python3 -c \"with open('foo_golden.py','w') as f: f.write('x')\"",
    "python3 -c \"from pathlib import Path; Path('foo_impl.py').write_text(x)\"",
    "python3 -c \"open('foo_impl.py','a').write('x')\"",                // append
    "python3 -c \"open('foo_impl.py','wb').write(b'x')\"",              // wb
    "python3 -c \"import shutil; shutil.copy('src', 'foo_impl.py')\"",
    "python3 -c \"import os; os.remove('foo_impl.py')\"",
  ];

  for (const cmd of blockedWrites) {
    await expect(
      before?.(
        { tool: "bash", sessionID: "test-session" } as never,
        { args: { command: cmd } } as never,
      ),
    ).rejects.toThrow("禁止通过 bash/shell 写入算子产物文件");
  }
});

test("allows readonly bash commands on operator artifacts", async () => {
  const plugin = await PyptoOpLintPlugin({
    $, client: { app: { log: async () => {} } },
    directory: process.cwd(), worktree: process.cwd(), project: {},
  } as never);
  const before = plugin["tool.execute.before"];

  const readonlyCommands = [
    "cat custom/foo/foo_impl.py",
    "ls -la custom/foo/modules/foo_module1_impl.py",
    "stat foo_impl.py",
    "grep '@pypto.frontend.jit' foo_impl.py",
    "diff foo_impl.py bar_impl.py",
    "wc -l test_foo.py",
    "head -20 foo_golden.py",
    "find custom/foo -name 'foo_*.py'",
    "python3 foo_impl.py",                     // running, not writing
    "python3 -m pytest test_foo.py",           // running tests, not writing
    // python -c reading / importing / exec — must NOT be blocked
    "python3 -c \"exec(open('foo_golden.py').read())\"",
    "python3 -c \"print(open('foo_impl.py').read())\"",
    "python3 -c \"import importlib.util; spec = importlib.util.spec_from_file_location('g', 'foo_golden.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\"",
    "python3 -c \"from pathlib import Path; print(Path('foo_impl.py').read_text())\"",
    "python3 -c \"with open('foo_golden.py') as f: print(f.read())\"",   // default 'r'
    "python3 -c \"with open('foo_golden.py', 'r') as f: print(f.read())\"",  // explicit 'r'
  ];

  for (const cmd of readonlyCommands) {
    await expect(
      before?.(
        { tool: "bash" } as never,
        { args: { command: cmd } } as never,
      ),
    ).resolves.toBeUndefined();
  }
});

test("bash write blocker does not affect non-artifact files", async () => {
  const plugin = await PyptoOpLintPlugin({
    $, client: { app: { log: async () => {} } },
    directory: process.cwd(), worktree: process.cwd(), project: {},
  } as never);
  const before = plugin["tool.execute.before"];

  const allowed = [
    "echo x > /tmp/notes.txt",
    "cat > README.md <<EOF\nhi\nEOF",
    "cp DESIGN.md DESIGN.md.bak",
    "sed -i 's/foo/bar/' SPEC.md",
    "rm /tmp/scratch.py",
  ];

  for (const cmd of allowed) {
    await expect(
      before?.(
        { tool: "bash" } as never,
        { args: { command: cmd } } as never,
      ),
    ).resolves.toBeUndefined();
  }
});

// ── tool.execute.after (post-edit / post-bash) ──

function createPlugin(shellMock: unknown) {
  return PyptoOpLintPlugin({
    $: shellMock,
    client: { app: { log: async () => {} } },
    directory: process.cwd(),
    worktree: process.cwd(),
  } as never);
}

/** Helper: build a mock $ that returns the given stdout text. */
function mockShell(stdout: string) {
  const chain = {
    text: async () => stdout,
    quiet: () => chain,
    env: () => chain,
  };
  // Tagged-template invocation: $`...`
  const fn = () => chain;
  return fn as unknown as typeof $;
}

test("post-edit skips non-operator files silently", async () => {
  const plugin = await createPlugin(mockShell(""));
  const after = plugin["tool.execute.after"]!;
  const output: Record<string, unknown> = {};
  // A random file that is not *_impl.py / *_golden.py / test_*.py
  await after(
    { tool: "write", args: { file_path: "/tmp/README.md" } } as never,
    output as never,
  );
  // No output should be appended
  expect(output.output).toBeUndefined();
});

test("post-bash skips non-test commands", async () => {
  let called = false;
  const shell = (() => { called = true; return mockShell("")(""); }) as any;
  const plugin = await (await import("../pypto-op-lint")).PyptoOpLintPlugin({
    $: mockShell(""),
    client: { app: { log: async () => {} } },
    directory: process.cwd(),
    worktree: process.cwd(),
  } as never);

  const after = plugin["tool.execute.after"]!;
  const output: Record<string, unknown> = {};
  // A non-test bash command should not trigger post-bash hook
  await after(
    { tool: "bash", args: { command: "ls -la" } } as never,
    output as any,
  );
  expect(output.output).toBeUndefined();
});

test("post-edit appends lint feedback for impl file", async () => {
  const hookResponse = JSON.stringify({
    hookSpecificOutput: {
      hookEventName: "post-edit",
      decision: "allow",
      additionalContext: "[pypto-op-lint] OL01: kernel 缺少 jit 装饰器",
    },
  });
  const mockShellFn = mockShell(hookResponse);
  const plugin = await createPlugin(mockShellFn);

  const after = plugin["tool.execute.after"]!;
  const output: Record<string, unknown> = {};

  await after(
    { tool: "write", args: { file_path: "/workspace/custom/sinh/sinh_impl.py" } } as never,
    output as any,
  );
  expect(typeof output.output).toBe("string");
  expect((output.output as string)).toContain("OL01");
});

test("post-edit blocks on S0/S1 violation", async () => {
  const hookResponse = JSON.stringify({
    hookSpecificOutput: {
      hookEventName: "PostToolUse",
      decision: "block",
      reason: "[S0] 缺少 import pypto",
      additionalContext: "OL07 致命",
    },
  });

  const plugin = await createPlugin(mockShell(hookResponse));
  const after = plugin["tool.execute.after"];
  await expect(
    after?.(
      { tool: "write", args: { file_path: "/tmp/custom/sinh/sinh_impl.py" } } as never,
      {} as never,
    ),
  ).rejects.toThrow();
});
