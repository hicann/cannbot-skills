// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

// Behavioural proof that the opencode safety net ENFORCES, run at install time.
//
// The structural proof in init.sh only shows that opencode RESOLVES the injected config —
// agents exist, skills are visible. That says nothing about whether a forbidden tool call is
// actually refused, and "the files are all present" is exactly the reasoning that let a
// disarmed install pass before.
//
// This drives the real adapter, in the real JS runtime, through the same entry point opencode
// calls (`tool.execute.before`), and requires a DENY/ALLOW PAIR:
//
//   deny  — a kernel-worker reading another op's workspace must throw
//   allow — the same worker reading its OWN workspace must not
//
// The pair is the point. A deny-only probe is passed by a guard that refuses everything,
// including a hook wired to reject unconditionally because its door is broken — which looks
// identical to "armed" from the outside. Only the allow half distinguishes an enforcing net
// from a bricked one.
//
// What it does NOT prove: that opencode itself invokes the hook for a model-driven tool call.
// Nothing short of a real model turn shows that, and neither this probe nor Phase O0's — which
// drives the same entry point through node — can stand in for one. That link is covered by the
// model-driven end-to-end check in src/scripts/tests/test_opencode_e2e_live.py, which runs only
// when an operator points AOG_E2E_OPENCODE_MODEL at a configured model.
//
// Three outcomes, deliberately distinct — conflating the last two would either block installs
// on restricted filesystems or wave through a net that does not enforce:
//
//   exit 0, "OK"          the pair held: enforcing
//   exit 1, "FAIL: …"     the probe RAN and the net misbehaved — a defect, install must stop
//   exit 2, "SKIP: …"     the probe could not be SET UP here (no writable temp dir, no symlink
//                         permission). Says nothing about the net, so it must not be reported
//                         as a failure of it; Phase O0 still gates at runtime.

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const { A5OpsHooksPlugin } = await import(path.join(here, "a5_ops_hooks.mjs"));

let root = "";
let mine = "";
let other = "";

function cleanup() {
  if (!root) return;
  try {
    // Unlinks the `src` symlink rather than descending through it, so the engine tree it
    // points at is never touched. Verified against this Node before relying on it.
    fs.rmSync(root, { recursive: true, force: true });
  } catch {
    /* a leftover temp dir must not turn into a failed install */
  }
}

function fail(why) {
  cleanup();
  console.log(`FAIL: ${why}`);
  process.exit(1);
}

function skip(why) {
  cleanup();
  console.log(`SKIP: ${why}`);
  process.exit(2);
}

// Scaffolding only. A failure here is the machine saying "not from this filesystem" — a
// read-only or noexec temp dir, a sandbox without symlink permission — and reveals nothing
// about whether the guards enforce. Reporting it as a safety-net failure would block installs
// on those machines for a reason the operator cannot act on, so it exits SKIP and leaves the
// verdict to the runtime gate.
try {
  root = fs.mkdtempSync(path.join(os.tmpdir(), "aog-oc-probe-"));
  mine = path.join(root, "workspace", "probe_op", "kernel");
  other = path.join(root, "workspace", "other_op", "kernel");
  fs.mkdirSync(mine, { recursive: true });
  fs.mkdirSync(other, { recursive: true });
  fs.writeFileSync(path.join(mine, "kernels.cpp"), "// probe\n");
  fs.writeFileSync(path.join(other, "kernels.cpp"), "// probe\n");
  // Some guards shell out to checker scripts resolved RELATIVE to the project root
  // (`<root>/src/scripts/...`). A bare temp root therefore fails them for a reason that has
  // nothing to do with policy — which the allow half correctly reported as "the net refuses
  // everything". Linking the engine's real `src/` in keeps the probe hermetic (nothing is
  // written into the engine tree) while letting those checkers resolve as they do in a run.
  const engineRoot = path.resolve(here, "..", "..");
  fs.symlinkSync(path.join(engineRoot, "src"), path.join(root, "src"), "dir");
} catch (err) {
  skip(`could not build the probe workspace: ${err instanceof Error ? err.message : String(err)}`);
}

try {
  // The guards read the active workspace from the environment, the same way a dispatched
  // agent's environment is built by the backend.
  process.env.AOG_PROJECT_ROOT = root;
  process.env.ASCENDC_WORKSPACE = path.join(root, "workspace", "probe_op");
  process.env.CLAUDE_ACTIVE_WORKSPACE = process.env.ASCENDC_WORKSPACE;
  // Marks this as a plugin-dispatched run, which is what arms the default-deny path.
  process.env.AOG_HOOK_AGENT_TYPE = "aog-kernel-worker";

  const plugin = await A5OpsHooksPlugin({ directory: root }, { projectRoot: root });
  for (const hook of ["chat.params", "tool.execute.before"]) {
    if (typeof plugin[hook] !== "function") fail(`adapter exposes no ${hook} hook`);
  }
  // Identity arrives per LLM turn in a real session; the tool hooks carry none of their own.
  await plugin["chat.params"]({ sessionID: "probe", agent: "aog-kernel-worker" });

  const readAttempt = async (filePath) => {
    try {
      await plugin["tool.execute.before"](
        { tool: "read", sessionID: "probe", callID: "probe-call" },
        { args: { filePath } },
      );
      return null;
    } catch (err) {
      return err instanceof Error ? err.message : String(err);
    }
  };

  const denied = await readAttempt(path.join(other, "kernels.cpp"));
  if (!denied) fail("cross-workspace read was NOT refused — the safety net is not enforcing");

  const allowed = await readAttempt(path.join(mine, "kernels.cpp"));
  if (allowed) fail(`own-workspace read was refused, so the net refuses everything: ${allowed}`);

  cleanup();
  console.log("OK");
} catch (err) {
  // Past the scaffolding, an exception is the adapter itself misbehaving — a bad import, a
  // hook that throws where it should not. That IS a defect in what is being proven.
  fail(`the adapter raised outside the guarded calls: ${err instanceof Error ? err.message : String(err)}`);
}
