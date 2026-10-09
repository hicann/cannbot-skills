// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

// Interactive (real-PTY) regression tests for the unowned-config-file
// protection. These drive the built CLI (`dist/index.js`) through a
// pseudo-terminal via util-linux `script`, so the @inquirer prompts render
// for real and keystrokes (Enter / arrow keys) are exercised end to end:
//
//   A. `install <plugin>` without --yes prompts for an unowned project
//      AGENTS.md; Enter (default) keeps the user file.
//   B. Choosing "replace" creates a backup beside the file, and an
//      interactive uninstall offers a restore that returns the original
//      content to its original path.
//   C. The full wizard (`install-helper` with no arguments) reaches the same
//      prompt through mode/tool/level/plugin/confirm and honors it.
//
// Skipped automatically when not on Linux or when `script` is unavailable.
// Each test runs with a sandboxed HOME and an offline plugin source
// (CANNBOT_REPO_PATH) plus a deliberately invalid CANNBOT_INSTALLER_PATH so
// cannbot discovery degrades instantly instead of hitting the network.

import { describe, it, expect, beforeAll } from "vitest";
import {
  spawn, spawnSync, execSync,
  type ChildProcess,
} from "child_process";
import { existsSync, lstatSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "fs";
import { join, resolve } from "path";
import { tmpdir } from "os";

const PKG_DIR = join(__dirname, "..");
const REPO_ROOT = resolve(join(PKG_DIR, "..", ".."));
const CLI_PATH = join(PKG_DIR, "dist", "index.js");
const SRC_DIR = join(PKG_DIR, "src");
const LOCALES_DIR = join(PKG_DIR, "src", "locales");

const ANSI_RE = /\x1b\[[0-9;?]*[A-Za-z]/g;

function stripAnsi(s: string): string {
  return s.replace(ANSI_RE, "");
}

/** Index (0-based) of the option line containing `needle` among rendered
 *  "> " option lines, or -1 when absent. `exclude` skips matching lines. */
function optionIndex(frame: string, needle: string, exclude?: string): number {
  let idx = 0;
  for (const raw of frame.split(/\r?\n/)) {
    const line = stripAnsi(raw);
    if (!line.includes("> ")) continue;
    if (line.includes(needle) && (!exclude || !line.includes(exclude))) return idx;
    idx++;
  }
  return -1;
}

function shellQuote(s: string): string {
  return `'${s.replace(/'/g, "'\\''")}'`;
}

interface PtyOptions {
  cwd: string;
  env: NodeJS.ProcessEnv;
}

/** A CLI session running under a real pseudo-terminal (util-linux `script`),
 *  with sequential `expect` semantics: each expect scans from the end of the
 *  previous match, so anchors can safely repeat across prompt frames. */
class PtySession {
  private child: ChildProcess;
  private buffer = "";
  private scanFrom = 0;
  private pending: {
    pattern: string | RegExp;
    resolve: (ctx: string) => void;
    reject: (err: Error) => void;
    timer: ReturnType<typeof setTimeout>;
  } | null = null;
  private exitCode: number | null = null;
  private exited = false;
  private exitWaiters: Array<(code: number | null) => void> = [];

  constructor(cmd: string, args: string[], opts: PtyOptions) {
    const shCmd = `stty rows 40 cols 120 2>/dev/null; exec ${[cmd, ...args].map(shellQuote).join(" ")}`;
    this.child = spawn("script", ["-qec", shCmd, "/dev/null"], {
      cwd: opts.cwd,
      env: opts.env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.child.stdout!.on("data", (chunk: Buffer) => {
      this.buffer += chunk.toString("utf-8");
      this.checkPending();
    });
    this.child.on("error", () => this.failPending("spawn error"));
    this.child.on("exit", (code) => {
      this.exited = true;
      this.exitCode = code;
      const waiters = this.exitWaiters.splice(0);
      for (const w of waiters) w(code);
      if (this.pending) this.failPending(`process exited (code ${code}) before matching`);
    });
  }

  private tail(): string {
    return stripAnsi(this.buffer.slice(-500));
  }

  private failPending(reason: string): void {
    const p = this.pending;
    if (!p) return;
    clearTimeout(p.timer);
    this.pending = null;
    p.reject(new Error(`${reason}: ${String(p.pattern)}; buffer tail: ${this.tail()}`));
  }

  private checkPending(): void {
    const p = this.pending;
    if (!p) return;
    const hay = this.buffer.slice(this.scanFrom);
    let matchLen: number;
    let idx: number;
    if (typeof p.pattern === "string") {
      idx = hay.indexOf(p.pattern);
      matchLen = p.pattern.length;
    } else {
      const m = p.pattern.exec(hay);
      idx = m === null ? -1 : m.index;
      matchLen = m === null ? 0 : m[0].length;
    }
    if (idx < 0) return;
    const ctx = hay.slice(0, idx + matchLen);
    this.scanFrom += idx + matchLen;
    clearTimeout(p.timer);
    this.pending = null;
    p.resolve(ctx);
  }

  expect(pattern: string | RegExp, timeoutMs = 60_000): Promise<string> {
    if (this.pending) return Promise.reject(new Error("another expect is already pending"));
    if (this.exited) {
      // The pattern may already be in the buffer; checkPending handles that.
    }
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        if (this.pending && this.pending.pattern === pattern) {
          this.pending = null;
          reject(new Error(`timeout waiting for ${String(pattern)}; buffer tail: ${this.tail()}`));
        }
      }, timeoutMs);
      this.pending = { pattern, resolve, reject, timer };
      this.checkPending();
    });
  }

  send(s: string): void {
    this.child.stdin!.write(s);
  }

  waitForExit(timeoutMs = 180_000): Promise<number | null> {
    if (this.exited) return Promise.resolve(this.exitCode);
    return new Promise((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error(`timeout waiting for process exit; buffer tail: ${this.tail()}`)),
        timeoutMs
      );
      this.exitWaiters.push((code) => {
        clearTimeout(timer);
        resolve(code);
      });
    });
  }

  close(): void {
    try { this.child.kill("SIGTERM"); } catch { /* already gone */ }
    try { this.child.stdin!.end(); } catch { /* already closed */ }
  }
}

function newestMtime(dir: string, filter: (name: string) => boolean): number {
  let newest = 0;
  let entries: ReturnType<typeof readdirSync>;
  try {
    entries = readdirSync(dir, { withFileTypes: true });
  } catch {
    return 0;
  }
  for (const e of entries) {
    const p = join(dir, e.name);
    if (e.isDirectory()) {
      newest = Math.max(newest, newestMtime(p, filter));
    } else if (filter(e.name)) {
      try {
        newest = Math.max(newest, statSync(p).mtimeMs);
      } catch { /* ignore */ }
    }
  }
  return newest;
}

function distNeedsBuild(): boolean {
  if (!existsSync(CLI_PATH)) return true;
  const distMtime = statSync(CLI_PATH).mtimeMs;
  const isSource = (name: string) => name.endsWith(".ts") || (name.endsWith(".json") && !name.startsWith("embedded-"));
  return (
    newestMtime(SRC_DIR, isSource) > distMtime ||
    newestMtime(LOCALES_DIR, (n) => n.endsWith(".json")) > distMtime
  );
}

interface TestProject {
  sandbox: string;
  project: string;
  env: NodeJS.ProcessEnv;
}

function makeProject(tag: string): TestProject {
  const sandbox = mkdtempSync(join(tmpdir(), `ih-pty-${tag}-`));
  const home = join(sandbox, "home");
  const project = join(sandbox, "project");
  mkdirSync(home, { recursive: true });
  mkdirSync(join(project, ".opencode"), { recursive: true });
  writeFileSync(join(project, "AGENTS.md"), "# User config\nKeep me safe.\n");
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    HOME: home,
    CANNBOT_REPO_PATH: REPO_ROOT,
    CANNBOT_INSTALLER_PATH: join(sandbox, "missing-installer"),
  };
  return { sandbox, project, env };
}

/** Navigate a rendered select to the option containing `needle` and press
 *  Enter. `anchor` frames the prompt; the context returned ends at the
 *  target option line, so `optionIndex` counts real rendered options. */
async function pickIndexed(
  session: PtySession,
  anchor: string,
  needle: string,
  exclude?: string,
  timeoutMs = 120_000
): Promise<string> {
  await session.expect(anchor, timeoutMs);
  const ctx = await session.expect(needle, 60_000);
  const idx = optionIndex(ctx, needle, exclude);
  if (idx < 0) {
    throw new Error(`option "${needle}" not found after "${anchor}":\n${stripAnsi(ctx).slice(-1200)}`);
  }
  for (let i = 0; i < idx; i++) session.send("\x1b[B");
  session.send("\r");
  return ctx;
}

/** Select the opencode tool whatever the detection branch renders
 *  (detected list, "confirm detected" shortcut, or the full manual list). */
async function pickOpenCode(session: PtySession): Promise<void> {
  await session.expect("选择 AI 编程工具", 120_000);
  // "x  退出" is the last line of every tool frame — matching it means the
  // whole choice list has rendered.
  const ctx = await session.expect("退出", 60_000);
  const idx = optionIndex(ctx, "OpenCode");
  if (idx < 0) {
    throw new Error(`OpenCode option not found in tool frame:\n${stripAnsi(ctx).slice(-1200)}`);
  }
  for (let i = 0; i < idx; i++) session.send("\x1b[B");
  session.send("\r");
}

const isLinux = process.platform === "linux";
const scriptAvailable = (() => {
  try {
    const r = spawnSync("script", ["--version"], { timeout: 10_000 });
    return !r.error;
  } catch {
    return false;
  }
})();

describe.skipIf(!isLinux || !scriptAvailable)("interactive install (real PTY)", () => {
  beforeAll(() => {
    if (!existsSync(join(SRC_DIR, "embedded-plugins.json"))) {
      execSync("node scripts/gen-embedded.cjs", { cwd: PKG_DIR, timeout: 120_000 });
    }
    if (distNeedsBuild()) {
      execSync("npx tsup", { cwd: PKG_DIR, timeout: 240_000, stdio: "pipe" });
    }
    expect(existsSync(CLI_PATH)).toBe(true);
  }, 300_000);

  it("A: install without --yes prompts for an unowned AGENTS.md; Enter keeps it", async () => {
    const { sandbox, project, env } = makeProject("a");
    const session = new PtySession(process.execPath, [
      CLI_PATH, "install", "ops-direct-invoke-flash", "--tool", "opencode", "--level", "project",
    ], { cwd: project, env });
    try {
      await session.expect("检测到非 CANNBot 安装的配置文件", 180_000);
      await session.expect("navigate", 60_000);
      session.send("\r"); // default = keep my file
      await session.expect("已保留现有文件", 60_000);
      expect(await session.waitForExit()).toBe(0);

      expect(readFileSync(join(project, "AGENTS.md"), "utf-8")).toBe("# User config\nKeep me safe.\n");
      expect(lstatSync(join(project, "AGENTS.md")).isSymbolicLink()).toBe(false);
      expect(existsSync(join(project, ".opencode", "skills", "ops-direct-invoke-flash"))).toBe(true);
      expect(existsSync(join(project, ".opencode", "AGENTS.md"))).toBe(true);
    } finally {
      session.close();
      rmSync(sandbox, { recursive: true, force: true });
    }
  }, 300_000);

  it("B: replace choice backs up beside the file; interactive uninstall restores it", async () => {
    const { sandbox, project, env } = makeProject("b");
    try {
      const install = new PtySession(process.execPath, [
        CLI_PATH, "install", "ops-direct-invoke-flash", "--tool", "opencode", "--level", "project",
      ], { cwd: project, env });
      try {
        await install.expect("检测到非 CANNBot 安装的配置文件", 180_000);
        await install.expect("navigate", 60_000);
        install.send("\x1b[B"); // Down -> 替换（自动备份）
        install.send("\r");
        await install.expect("已创建备份", 60_000);
        expect(await install.waitForExit()).toBe(0);
      } finally {
        install.close();
      }

      expect(lstatSync(join(project, "AGENTS.md")).isSymbolicLink()).toBe(true);
      const backups = readdirSync(project).filter((f) => f.startsWith("AGENTS.md.cannbot-backup.unowned."));
      expect(backups.length).toBe(1);
      expect(readFileSync(join(project, backups[0]), "utf-8")).toBe("# User config\nKeep me safe.\n");

      const uninstall = new PtySession(process.execPath, [
        CLI_PATH, "uninstall", "ops-direct-invoke-flash", "--tool", "opencode", "--level", "project",
      ], { cwd: project, env });
      try {
        await uninstall.expect("恢复用户文件", 120_000);
        await uninstall.expect("navigate", 60_000);
        uninstall.send("\r"); // restore the user file to its original path
        await uninstall.expect("已恢复配置", 60_000);
        expect(await uninstall.waitForExit()).toBe(0);
      } finally {
        uninstall.close();
      }

      expect(readFileSync(join(project, "AGENTS.md"), "utf-8")).toBe("# User config\nKeep me safe.\n");
      expect(lstatSync(join(project, "AGENTS.md")).isSymbolicLink()).toBe(false);
      expect(readdirSync(project).filter((f) => f.includes(".cannbot-backup.")).length).toBe(0);
      expect(existsSync(join(project, ".opencode", "skills", "ops-direct-invoke-flash"))).toBe(false);
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
    }
  }, 420_000);

  it("C: full wizard reaches and honors the unowned-file prompt", async () => {
    const { sandbox, project, env } = makeProject("c");
    const session = new PtySession(process.execPath, [CLI_PATH], { cwd: project, env });
    try {
      await pickIndexed(session, "选择安装类型", "安装 Plugin");
      await pickOpenCode(session);
      await pickIndexed(session, "选择安装位置", "project");
      const pluginCtx = await pickIndexed(session, "选择要安装的插件", "快速版", "[cannbot]");

      // confirm step: summary must reflect tool / level / plugin
      await session.expect("[4/4]", 60_000);
      const confirmCtx = await session.expect("❯", 60_000);
      const flat = stripAnsi(confirmCtx);
      expect(flat).toContain("project");
      expect(flat).toContain("快速版");
      expect(flat).toContain("OpenCode");
      expect(stripAnsi(pluginCtx)).toContain("快速版");
      session.send("\r"); // 确认安装

      await session.expect("检测到非 CANNBot 安装的配置文件", 180_000);
      await session.expect("navigate", 60_000);
      session.send("\r"); // keep my file
      await session.expect("已保留现有文件", 60_000);
      expect(await session.waitForExit()).toBe(0);

      expect(readFileSync(join(project, "AGENTS.md"), "utf-8")).toBe("# User config\nKeep me safe.\n");
      expect(lstatSync(join(project, "AGENTS.md")).isSymbolicLink()).toBe(false);
      expect(existsSync(join(project, ".opencode", "skills", "ops-direct-invoke-flash"))).toBe(true);
      expect(existsSync(join(project, ".opencode", "AGENTS.md"))).toBe(true);
    } finally {
      session.close();
      rmSync(sandbox, { recursive: true, force: true });
    }
  }, 420_000);
});
