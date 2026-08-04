// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi, beforeEach, afterEach, afterAll } from "vitest";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";

// Persistent-mirror route reads UPLOAD_CACHE_DIR at module load, so set it to
// an isolated temp dir and pretend it's an existing mirror (.git present)
// BEFORE importing the route. This makes ensureUploadMirror take the
// fetch-delta path (no real clone).
const MIRROR_DIR = path.join(os.tmpdir(), `cannbay-upload-test-${process.pid}`);
fs.rmSync(MIRROR_DIR, { recursive: true, force: true });
fs.mkdirSync(path.join(MIRROR_DIR, ".git"), { recursive: true });
process.env.CANNBAY_UPLOAD_CACHE_DIR = MIRROR_DIR;

const execMock = vi.fn(() => Buffer.alloc(0));
vi.mock("node:child_process", () => ({ execSync: execMock }));

vi.mock("@/lib/ingest/export-service", () => ({
  exportSession: async (_taskId: string, outputPath?: string) => {
    if (outputPath) fs.writeFileSync(outputPath, Buffer.from("dummy-db"));
    return outputPath ?? "";
  },
}));

const { POST, rotateArchiveIfNeeded } = await import("@/app/api/ingest/upload-session/route");

function makeRequest(body: unknown) {
  return new Request("http://localhost/api/ingest/upload-session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function resetMirror() {
  fs.rmSync(MIRROR_DIR, { recursive: true, force: true });
  fs.mkdirSync(path.join(MIRROR_DIR, ".git"), { recursive: true });
}

afterAll(() => {
  fs.rmSync(MIRROR_DIR, { recursive: true, force: true });
});

describe("upload-session route · persistent mirror", () => {
  beforeEach(() => {
    execMock.mockReset();
    execMock.mockImplementation(() => Buffer.alloc(0));
    resetMirror();
  });

  it("syncs mirror to master, commits, pushes via credentialed URL; no rotation for a single new file", async () => {
    const res = await POST(makeRequest({ taskId: "ses_t1", description: "d" }));
    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json.success).toBe(true);
    expect(json.archivedBranch).toBeNull();

    const calls = execMock.mock.calls.map((c) => c[0] as string);
    // mirror sync (fetch in ensureUploadMirror) + checkout to remote tip + clean
    expect(calls).toContainEqual("git fetch --quiet origin master");
    expect(calls).toContainEqual("git checkout -B master origin/master");
    expect(calls).toContainEqual("git clean -fdq");
    // commit + push via explicit credentialed URL (not origin)
    expect(calls.some((c) => c.startsWith("git commit -F"))).toBe(true);
    const pushCall = calls.find((c) => c.startsWith("git push") && c.includes("master"));
    expect(pushCall).toBeDefined();
    expect(pushCall).not.toContain("origin master"); // must be the credentialed URL, not `origin`
    // no rotation triggered for a single file
    expect(calls.some((c) => c.startsWith("git rm"))).toBe(false);
    expect(calls.some((c) => c.includes("archive-"))).toBe(false);
  });

  it("integrates concurrent remote push via fetch+merge before push (non-fast-forward)", async () => {
    // Model: `git push` fails unless `git merge` has run first (remote advanced).
    let merged = false;
    execMock.mockImplementation((cmd: string) => {
      if (cmd.startsWith("git merge") && cmd.includes("--allow-unrelated-histories")) {
        merged = true;
        return Buffer.alloc(0);
      }
      if (cmd.startsWith("git push") && cmd.includes("master")) {
        if (!merged) throw new Error("rejected (fetch first) — non-fast-forward");
        return Buffer.alloc(0);
      }
      return Buffer.alloc(0);
    });

    const res = await POST(makeRequest({ taskId: "ses_t2", description: "d" }));
    expect(res.status).toBe(200);
    const calls = execMock.mock.calls.map((c) => c[0] as string);
    expect(calls).toContainEqual(
      expect.stringContaining("git merge --allow-unrelated-histories --no-edit origin/master"),
    );
    expect(calls.some((c) => c.startsWith("git push") && c.includes("master"))).toBe(true);
    expect(calls.some((c) => c.includes("--force"))).toBe(false);
  });
});

describe("upload-session · archive rotation", () => {
  const RUN = Date.UTC(2026, 6, 23);
  let tmp: string;

  beforeEach(() => {
    execMock.mockReset();
    tmp = fs.mkdtempSync("cannbay-arch-");
    for (let i = 0; i < 25; i++) fs.writeFileSync(`${tmp}/s${i}.db`, Buffer.from("x"));
  });
  afterEach(() => {
    try { fs.rmSync(tmp, { recursive: true, force: true }); } catch {}
  });

  it("rotates the oldest 5 off master onto archive-YYYY-MM, pushes archive via credentialed URL", () => {
    execMock.mockImplementation((cmd: string) => {
      const m = cmd.match(/git log .*-- "s(\d+)\.db"/);
      if (m) return Buffer.from(new Date(RUN - Number(m[1]) * 86_400_000).toISOString());
      if (cmd.startsWith("git fetch origin")) throw new Error("no such ref");
      return Buffer.alloc(0);
    });

    const branch = rotateArchiveIfNeeded(tmp, RUN, "https://creds@gitcode.com/x.git");
    expect(branch).toBe("archive-2026-07");

    const calls = execMock.mock.calls.map((c) => c[0] as string);
    const rmCall = calls.find((c) => c.startsWith("git rm"));
    expect(rmCall).toBeDefined();
    for (let i = 20; i < 25; i++) expect(rmCall).toContain(`"s${i}.db"`);
    for (let i = 0; i < 20; i++) expect(rmCall).not.toContain(`"s${i}.db"`);
    // archive pushed via the explicit credentialed URL, not `origin`
    expect(calls).toContainEqual(`git push "https://creds@gitcode.com/x.git" "archive-2026-07"`);
    expect(calls.some((c) => c.includes("--force"))).toBe(false);
  });

  it("no rotation when master has <= 20 files", () => {
    for (let i = 0; i < 6; i++) fs.unlinkSync(`${tmp}/s${i}.db`);
    const branch = rotateArchiveIfNeeded(tmp, RUN, "https://creds@gitcode.com/x.git");
    expect(branch).toBeNull();
    const calls = execMock.mock.calls.map((c) => c[0] as string);
    expect(calls.some((c) => c.startsWith("git rm"))).toBe(false);
    expect(calls.some((c) => c.startsWith("git push"))).toBe(false);
  });
});
