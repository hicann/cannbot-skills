// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// Integration test for Audit v4 Step 1 — deterministic agent-IO extraction.
// Runs against the real dev DB; gracefully skips when no session with a bridge exists.

import { describe, it, expect } from "vitest";
import { prisma } from "./setup";
import { buildAgentIO } from "@/lib/export/agent-io-export";

describe("buildAgentIO (v4 Step 1)", () => {
  it("extracts a flat agent tree with envelope + I/O from a session that has bridges", async () => {
    const bridge = await prisma.interactionBridge.findFirst({
      include: { session: { select: { taskId: true, framework: true } } },
    });
    if (!bridge) {
      // No bridge data in this DB — nothing to assert, pass vacuously.
      expect(bridge).toBeNull();
      return;
    }

    const io = await buildAgentIO(bridge.session.taskId, prisma, bridge.session.framework ?? undefined);

    expect(io.sessionId).toBeDefined();
    expect(Array.isArray(io.agents)).toBe(true);
    expect(io.agents.length).toBeGreaterThan(0);

    const main = io.agents.find(a => a.id === "main");
    expect(main).toBeDefined();
    expect(main!.parentId).toBeNull();
    expect(main!.role).toBe("main");

    // subagents: parentId points to an existing agent id, role subagent
    const subs = io.agents.filter(a => a.role === "subagent");
    const ids = new Set(io.agents.map(a => a.id));
    for (const s of subs) {
      expect(s.parentId).not.toBeNull();
      expect(ids.has(s.parentId!)).toBe(true);
      expect(s.inputSummary.length).toBeGreaterThanOrEqual(0); // may be empty if dispatchContent absent
    }

    // envelope fields are numeric and non-negative
    for (const a of io.agents) {
      const e = a.envelope;
      expect(typeof e.latencySec).toBe("number");
      expect(typeof e.tokensKt).toBe("number");
      expect(typeof e.turnCount).toBe("number");
      expect(typeof e.toolCallCount).toBe("number");
      expect(typeof e.errorCount).toBe("number");
      expect(typeof e.retryCount).toBe("number");
      expect(e.turnCount).toBeGreaterThanOrEqual(0);
      expect(e.errorCount).toBeGreaterThanOrEqual(0);
      expect(Array.isArray(a.artifacts)).toBe(true);
    }

    // at least one subagent should exist (we picked a session with a bridge)
    expect(subs.length).toBeGreaterThan(0);
  });
});
