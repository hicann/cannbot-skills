// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, test, expect } from "bun:test";
import { applyTransition, type OrchestratorState } from "../lib/state-transition-core";

const MAX_STAGE = 7;

function makeState(overrides: Partial<OrchestratorState> = {}): OrchestratorState {
  const stageStatus: Record<string, string> = {};
  const retryCount: Record<string, number> = {};
  for (let i = 1; i <= MAX_STAGE; i++) {
    stageStatus[String(i)] = "pending";
    retryCount[String(i)] = 0;
  }
  return {
    operator_name: "test_op",
    schema_version: "2.0",
    max_stage: MAX_STAGE,
    current_stage: 1,
    stage_status: stageStatus,
    stage_retry_count: retryCount,
    ...overrides,
  };
}

describe("init", () => {
  test("transitions from all-pending to stage 1 in_progress", () => {
    const state = makeState();
    const next = applyTransition(state, { action: "init" });
    expect(next.current_stage).toBe(1);
    expect(next.stage_status["1"]).toBe("in_progress");
    expect(next.schema_version).toBe("2.0");
  });

  test("rejects when a stage is already in_progress", () => {
    const state = makeState({
      stage_status: { ...makeState().stage_status, "1": "in_progress" },
    });
    expect(() => applyTransition(state, { action: "init" })).toThrow("already in_progress");
  });

  test("rejects target stage other than 1", () => {
    const state = makeState();
    expect(() => applyTransition(state, { action: "init", stage: 2 })).toThrow("stage 1");
  });

  test("clears residual stage5_phases, rollback_history and artifact_hashes", () => {
    const state = makeState({
      stage5_phases: {
        active_phase: "M1",
        max_cycles_per_phase: 10,
        phase_status: { M1: { status: "in_debug", cycles: 3 } },
      },
      rollback_history: [
        { from_stage: 5, to_stage: 3, reason: "stale", timestamp: "2026-01-01T00:00:00Z" },
      ],
      artifact_hashes: { spec_md: "stale-hash" },
    });
    const next = applyTransition(state, { action: "init" });
    expect(next.stage5_phases).toBeUndefined();
    expect(next.rollback_history).toBeUndefined();
    expect(next.artifact_hashes).toBeUndefined();
  });
});

describe("complete_stage", () => {
  test("advances to next stage", () => {
    const state = makeState({
      current_stage: 2,
      stage_status: { ...makeState().stage_status, "1": "completed", "2": "in_progress" },
    });
    const next = applyTransition(state, { action: "complete_stage", stage: 2 });
    expect(next.stage_status["2"]).toBe("completed");
    expect(next.stage_status["3"]).toBe("in_progress");
    expect(next.current_stage).toBe(3);
  });

  test("rejects wrong stage", () => {
    const state = makeState({
      current_stage: 2,
      stage_status: { ...makeState().stage_status, "1": "completed", "2": "in_progress" },
    });
    expect(() =>
      applyTransition(state, { action: "complete_stage", stage: 1 }),
    ).toThrow("current_stage");
  });

  test("rejects non-in_progress stage", () => {
    const state = makeState({
      current_stage: 2,
      stage_status: { ...makeState().stage_status, "1": "completed" },
    });
    expect(() =>
      applyTransition(state, { action: "complete_stage", stage: 2 }),
    ).toThrow("in_progress");
  });

  test("does not advance past max_stage", () => {
    const last = MAX_STAGE;
    const stageStatus: Record<string, string> = {};
    for (let i = 1; i <= MAX_STAGE; i++) {
      stageStatus[String(i)] = i < last ? "completed" : "in_progress";
    }
    const state = makeState({ current_stage: last, stage_status: stageStatus });
    const next = applyTransition(state, { action: "complete_stage", stage: last });
    expect(next.stage_status[String(last)]).toBe("completed");
    expect(next.current_stage).toBe(last);
  });

  test("rejects complete_stage(5) when a phase is not yet verified", () => {
    const state = makeState({
      current_stage: 5,
      stage_status: {
        ...makeState().stage_status,
        "1": "completed", "2": "completed", "3": "completed", "4": "completed", "5": "in_progress",
      },
      stage5_phases: {
        active_phase: "M2",
        max_cycles_per_phase: 10,
        phase_status: {
          M1: { status: "verified", cycles: 1 },
          M2: { status: "in_debug", cycles: 1, failure_category: "precision" },
        },
      },
    });
    expect(() =>
      applyTransition(state, { action: "complete_stage", stage: 5 }),
    ).toThrow("phase M2");
  });

  test("allows complete_stage(5) when all phases are verified", () => {
    const state = makeState({
      current_stage: 5,
      stage_status: {
        ...makeState().stage_status,
        "1": "completed", "2": "completed", "3": "completed", "4": "completed", "5": "in_progress",
      },
      stage5_phases: {
        active_phase: null,
        max_cycles_per_phase: 10,
        phase_status: {
          M1: { status: "verified", cycles: 1 },
          M2: { status: "verified", cycles: 2 },
        },
      },
    });
    const next = applyTransition(state, { action: "complete_stage", stage: 5 });
    expect(next.stage_status["5"]).toBe("completed");
    expect(next.stage_status["6"]).toBe("in_progress");
  });
});

describe("start_stage", () => {
  test("starts a failed stage for retry", () => {
    const state = makeState({
      current_stage: 2,
      stage_status: { ...makeState().stage_status, "1": "completed", "2": "failed" },
      stage_retry_count: { ...makeState().stage_retry_count, "2": 1 },
    });
    const next = applyTransition(state, { action: "start_stage", stage: 2 });
    expect(next.stage_status["2"]).toBe("in_progress");
    expect(next.current_stage).toBe(2);
  });

  test("rejects if previous stage not completed", () => {
    const state = makeState({ current_stage: 2 });
    expect(() =>
      applyTransition(state, { action: "start_stage", stage: 2 }),
    ).toThrow();
  });

  test("rejects starting a completed stage", () => {
    const state = makeState({
      stage_status: { ...makeState().stage_status, "1": "completed", "2": "completed" },
    });
    expect(() =>
      applyTransition(state, { action: "start_stage", stage: 2 }),
    ).toThrow("already completed");
  });
});

describe("fail_stage", () => {
  test("marks stage as failed and increments retry", () => {
    const state = makeState({
      current_stage: 2,
      stage_status: { ...makeState().stage_status, "1": "completed", "2": "in_progress" },
    });
    const result = applyTransition(state, { action: "fail_stage", stage: 2 });
    expect(result.stage_status["2"]).toBe("failed");
    expect(result.stage_retry_count["2"]).toBe(1);
  });

  test("rejects failing a stage other than current_stage", () => {
    const state = makeState({
      current_stage: 2,
      stage_status: { ...makeState().stage_status, "1": "completed", "2": "in_progress" },
    });
    expect(() =>
      applyTransition(state, { action: "fail_stage", stage: 1 }),
    ).toThrow("current_stage is 2");
  });

  test("rejects failing a stage that is not in_progress", () => {
    const state = makeState({
      current_stage: 2,
      stage_status: { ...makeState().stage_status, "1": "completed", "2": "completed" },
    });
    expect(() =>
      applyTransition(state, { action: "fail_stage", stage: 2 }),
    ).toThrow('status is "completed"');
  });
});

describe("Stage 5 phase loop", () => {
  function stage5Ready(): OrchestratorState {
    return makeState({
      current_stage: 5,
      stage_status: {
        ...makeState().stage_status,
        "1": "completed", "2": "completed", "3": "completed", "4": "completed", "5": "in_progress",
      },
    });
  }

  test("start_phase initialises the phase entry", () => {
    const next = applyTransition(stage5Ready(), { action: "start_phase", stage: 5, phase: "M1" });
    expect(next.stage5_phases?.active_phase).toBe("M1");
    expect(next.stage5_phases?.phase_status.M1.status).toBe("in_progress");
    expect(next.stage5_phases?.phase_status.M1.cycles).toBe(0);
  });

  test("fail_phase increments cycles and stores failure_category", () => {
    let s = applyTransition(stage5Ready(), { action: "start_phase", stage: 5, phase: "M1" });
    s = applyTransition(s, {
      action: "fail_phase",
      stage: 5,
      phase: "M1",
      failure_category: "precision",
      last_error: "max_diff=3.2e-4 vs atol=1e-6",
    });
    expect(s.stage5_phases?.phase_status.M1.status).toBe("in_debug");
    expect(s.stage5_phases?.phase_status.M1.cycles).toBe(1);
    expect(s.stage5_phases?.phase_status.M1.failure_category).toBe("precision");
    expect(s.stage5_phases?.phase_status.M1.last_error).toContain("max_diff");
  });

  test("10 consecutive fail_phase calls mark the phase blocked", () => {
    let s = applyTransition(stage5Ready(), { action: "start_phase", stage: 5, phase: "M1" });
    for (let i = 0; i < 10; i++) {
      s = applyTransition(s, {
        action: "fail_phase",
        stage: 5,
        phase: "M1",
        failure_category: "precision",
      });
    }
    expect(s.stage5_phases?.phase_status.M1.status).toBe("blocked");
    expect(s.stage5_phases?.phase_status.M1.cycles).toBe(10);
  });

  test("complete_phase marks phase verified and clears active_phase", () => {
    let s = applyTransition(stage5Ready(), { action: "start_phase", stage: 5, phase: "M1" });
    s = applyTransition(s, { action: "complete_phase", stage: 5, phase: "M1" });
    expect(s.stage5_phases?.phase_status.M1.status).toBe("verified");
    expect(s.stage5_phases?.phase_status.M1.verified_at).toBeDefined();
    expect(s.stage5_phases?.active_phase).toBeNull();
  });

  test("rejects starting a second phase while another is in_progress", () => {
    let s = applyTransition(stage5Ready(), { action: "start_phase", stage: 5, phase: "M1" });
    expect(() =>
      applyTransition(s, { action: "start_phase", stage: 5, phase: "M2" }),
    ).toThrow("already in_progress");
  });

  test("rejects fail_phase for a phase that was never started", () => {
    expect(() =>
      applyTransition(stage5Ready(), {
        action: "fail_phase",
        stage: 5,
        phase: "M99",
        failure_category: "precision",
      }),
    ).toThrow("phase not started");
  });

  test("rejects restarting a blocked phase", () => {
    let s = applyTransition(stage5Ready(), { action: "start_phase", stage: 5, phase: "M1" });
    for (let i = 0; i < 10; i++) {
      s = applyTransition(s, {
        action: "fail_phase",
        stage: 5,
        phase: "M1",
        failure_category: "precision",
      });
    }
    expect(s.stage5_phases?.phase_status.M1.status).toBe("blocked");
    expect(() =>
      applyTransition(s, { action: "start_phase", stage: 5, phase: "M1" }),
    ).toThrow("blocked");
  });

  test("rejects starting an already-verified phase", () => {
    let s = applyTransition(stage5Ready(), { action: "start_phase", stage: 5, phase: "M1" });
    s = applyTransition(s, { action: "complete_phase", stage: 5, phase: "M1" });
    expect(() =>
      applyTransition(s, { action: "start_phase", stage: 5, phase: "M1" }),
    ).toThrow("already verified");
  });

  test("phase actions are rejected outside stage 5", () => {
    expect(() =>
      applyTransition(makeState(), { action: "start_phase" as never, stage: 4 as never, phase: "M1" }),
    ).toThrow();
  });
});

describe("record_artifact_hash", () => {
  test("stores the named hash", () => {
    const next = applyTransition(makeState(), {
      action: "record_artifact_hash",
      name: "spec_md",
      hash: "abcdef1234",
    });
    expect(next.artifact_hashes?.spec_md).toBe("abcdef1234");
  });

  test("rejects empty name or hash", () => {
    expect(() =>
      applyTransition(makeState(), {
        action: "record_artifact_hash",
        name: "",
        hash: "x",
      }),
    ).toThrow("name");
    expect(() =>
      applyTransition(makeState(), {
        action: "record_artifact_hash",
        name: "spec_md",
        hash: "",
      }),
    ).toThrow("hash");
  });
});

describe("rollback_to_stage", () => {
  function stage5InDebug(): OrchestratorState {
    return makeState({
      current_stage: 5,
      stage_status: {
        ...makeState().stage_status,
        "1": "completed",
        "2": "completed",
        "3": "completed",
        "4": "completed",
        "5": "in_progress",
      },
      stage_retry_count: { ...makeState().stage_retry_count, "3": 0 },
      artifact_hashes: {
        spec_md: "spec-hash",
        design_md: "design-hash",
        module_interfaces_yaml: "yaml-hash",
      },
      stage5_phases: {
        active_phase: "M2",
        max_cycles_per_phase: 10,
        phase_status: {
          M1: { status: "verified", cycles: 1 },
          M2: { status: "in_debug", cycles: 3, failure_category: "fp32_unstable" },
        },
      },
    });
  }

  test("resets every stage after target to pending", () => {
    const next = applyTransition(stage5InDebug(), {
      action: "rollback_to_stage",
      target_stage: 3,
      reason: "fp32_unstable: log-sum-exp shift missing",
      failure_category: "fp32_unstable",
      failed_phase: "M2",
    });
    expect(next.current_stage).toBe(3);
    expect(next.stage_status["3"]).toBe("in_progress");
    expect(next.stage_status["4"]).toBe("pending");
    expect(next.stage_status["5"]).toBe("pending");
    expect(next.stage_status["6"]).toBe("pending");
  });

  test("increments retry_count on target stage", () => {
    const next = applyTransition(stage5InDebug(), {
      action: "rollback_to_stage",
      target_stage: 3,
      reason: "fp32_unstable",
    });
    expect(next.stage_retry_count["3"]).toBe(1);
  });

  test("wipes stage5_phases when rolling back to stage < 5", () => {
    const next = applyTransition(stage5InDebug(), {
      action: "rollback_to_stage",
      target_stage: 3,
      reason: "fp32_unstable",
    });
    expect(next.stage5_phases?.phase_status).toEqual({});
    expect(next.stage5_phases?.active_phase).toBeNull();
  });

  test("drops artifact hashes for stages after target", () => {
    const next = applyTransition(stage5InDebug(), {
      action: "rollback_to_stage",
      target_stage: 3,
      reason: "design rework",
    });
    // Stage 1 hashes survive (target=3, owning_stage=1 < 3)
    expect(next.artifact_hashes?.spec_md).toBe("spec-hash");
    // Stage 4+ hashes are dropped
    expect(next.artifact_hashes?.module_interfaces_yaml).toBeUndefined();
    // design_md (stage 3) is preserved because it equals target
    expect(next.artifact_hashes?.design_md).toBe("design-hash");
  });

  test("appends an entry to rollback_history", () => {
    const next = applyTransition(stage5InDebug(), {
      action: "rollback_to_stage",
      target_stage: 3,
      reason: "fp32_unstable in M2",
      failure_category: "fp32_unstable",
      failed_phase: "M2",
    });
    expect(next.rollback_history?.length).toBe(1);
    const entry = next.rollback_history![0];
    expect(entry.from_stage).toBe(5);
    expect(entry.from_phase).toBe("M2");
    expect(entry.to_stage).toBe(3);
    expect(entry.failure_category).toBe("fp32_unstable");
    expect(entry.reason).toContain("fp32_unstable");
    expect(entry.timestamp).toBeDefined();
  });

  test("rejects rollback to current or future stage", () => {
    const state = stage5InDebug();
    expect(() =>
      applyTransition(state, {
        action: "rollback_to_stage",
        target_stage: 5,
        reason: "x",
      }),
    ).toThrow("strictly less than");
    expect(() =>
      applyTransition(state, {
        action: "rollback_to_stage",
        target_stage: 6,
        reason: "x",
      }),
    ).toThrow();
  });

  test("rejects rollback without reason", () => {
    const state = stage5InDebug();
    expect(() =>
      applyTransition(state, {
        action: "rollback_to_stage",
        target_stage: 3,
        reason: "",
      }),
    ).toThrow("reason");
  });

  test("rollback within stage 5 (target=5 invalid since current_stage=5)", () => {
    // rolling back to stage 5 from current=5 is rejected (must be strictly less)
    expect(() =>
      applyTransition(stage5InDebug(), {
        action: "rollback_to_stage",
        target_stage: 5,
        reason: "x",
      }),
    ).toThrow();
  });

  test("rollback from stage 7 to stage 5 keeps stage5_phases (since target>=5)", () => {
    const stage7: OrchestratorState = makeState({
      current_stage: 7,
      stage_status: {
        ...makeState().stage_status,
        "1": "completed", "2": "completed", "3": "completed", "4": "completed",
        "5": "completed", "6": "completed", "7": "in_progress",
      },
      stage5_phases: {
        active_phase: null,
        max_cycles_per_phase: 10,
        phase_status: {
          M1: { status: "verified", cycles: 1 },
          M2: { status: "verified", cycles: 1 },
        },
      },
    });
    const next = applyTransition(stage7, {
      action: "rollback_to_stage",
      target_stage: 5,
      reason: "perf regression",
    });
    expect(next.current_stage).toBe(5);
    expect(next.stage_status["5"]).toBe("in_progress");
    expect(next.stage_status["7"]).toBe("pending");
    // Phase progress preserved (target_stage >= 5)
    expect(next.stage5_phases?.phase_status.M1.status).toBe("verified");
  });
});

describe("dynamic max_stage", () => {
  test("max_stage of 10 allows complete_stage(9) and (10)", () => {
    const stageStatus: Record<string, string> = {};
    const retryCount: Record<string, number> = {};
    for (let i = 1; i <= 10; i++) {
      stageStatus[String(i)] = i < 9 ? "completed" : i === 9 ? "in_progress" : "pending";
      retryCount[String(i)] = 0;
    }
    const state: OrchestratorState = {
      operator_name: "extended_op",
      schema_version: "2.0",
      max_stage: 10,
      current_stage: 9,
      stage_status: stageStatus,
      stage_retry_count: retryCount,
    };
    const after9 = applyTransition(state, { action: "complete_stage", stage: 9 });
    expect(after9.current_stage).toBe(10);
    expect(after9.stage_status["10"]).toBe("in_progress");
    const after10 = applyTransition(after9, { action: "complete_stage", stage: 10 });
    expect(after10.stage_status["10"]).toBe("completed");
    expect(after10.current_stage).toBe(10); // last stage, no advance
  });

  test("rejects stage > max_stage", () => {
    expect(() =>
      applyTransition(makeState(), { action: "fail_stage", stage: 8 }),
    ).toThrow("must be 1..7");
  });
});
