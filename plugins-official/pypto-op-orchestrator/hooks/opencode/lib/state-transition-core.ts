// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// state-transition-core.ts — schema v2.0
//
// Stages are 1..max_stage. max_stage is dynamic (default 7 for the
// PyPTO kernel workflow, but callers may extend).
//
// Stage 5 contains an inner per-Phase loop (Phase M_k). Phase progress
// is tracked under stage5_phases. Phase mutations are separate actions
// (start_phase, complete_phase, fail_phase) so that the per-Phase
// failure_category and cycles counter stay machine-readable.
//
// rollback_to_stage walks the workflow backwards: any stage strictly
// after target_stage is reset to pending, retry_count[target_stage] is
// incremented, stage5_phases is wiped (modules may be re-decomposed),
// and an entry is appended to rollback_history (append-only).
//
// All write actions are restricted to pypto-op-orchestrator at the
// plugin boundary; this file only enforces the state-machine invariants.

export type TransitionAction =
  | "init"
  | "start_stage"
  | "complete_stage"
  | "fail_stage"
  | "submit_design"
  | "start_phase"
  | "submit_for_verify"
  | "complete_phase"
  | "fail_phase"
  | "record_artifact_hash"
  | "rollback_to_stage";

export type StageStatus = "pending" | "in_progress" | "completed" | "failed";
export type PhaseStatus =
  | "pending"
  | "in_progress"
  | "awaiting_verify"
  | "in_debug"
  | "verified"
  | "blocked";

export type PhaseEntry = {
  status: PhaseStatus;
  cycles: number;
  failure_category?: string;
  failing_module_boundary?: string | null;
  last_error?: string;
  verified_at?: string;
};

export type Stage5Phases = {
  active_phase: string | null;
  phase_status: Record<string, PhaseEntry>;
  max_cycles_per_phase: number;
};

export type RollbackEntry = {
  from_stage: number;
  from_phase?: string;
  to_stage: number;
  reason: string;
  failure_category?: string;
  timestamp: string;
};

export type ArtifactHashes = Record<string, string>;

export type OrchestratorState = {
  operator_name?: string;
  schema_version?: string;
  max_stage: number;
  current_stage: number;
  stage_status: Record<string, StageStatus | string>;
  stage_retry_count: Record<string, number>;
  stage5_phases?: Stage5Phases;
  artifact_hashes?: ArtifactHashes;
  rollback_history?: RollbackEntry[];
  last_updated?: string;
  [key: string]: unknown;
};

export type TransitionInput =
  | { action: "init"; stage?: number; max_stage?: number }
  | { action: "start_stage"; stage: number; reason?: string }
  | { action: "complete_stage"; stage: number }
  | { action: "fail_stage"; stage: number; reason?: string }
  | { action: "submit_design"; stage: 4 }
  | { action: "start_phase"; stage: 5; phase: string }
  | { action: "submit_for_verify"; stage: 5; phase: string }
  | { action: "complete_phase"; stage: 5; phase: string }
  | {
      action: "fail_phase";
      stage: 5;
      phase: string;
      failure_category: string;
      failing_module_boundary?: string | null;
      last_error?: string;
    }
  | { action: "record_artifact_hash"; name: string; hash: string }
  | {
      action: "rollback_to_stage";
      target_stage: number;
      reason: string;
      failure_category?: string;
      failed_phase?: string;
    };

const DEFAULT_MAX_STAGE = 7;

function cloneState(prev: OrchestratorState): OrchestratorState {
  return JSON.parse(JSON.stringify(prev));
}

function ensureStageNumber(val: unknown, max: number): number {
  const n = Number(val);
  if (!Number.isFinite(n) || n < 1 || n > max) {
    throw new Error(`invalid stage: ${val} (must be 1..${max})`);
  }
  return Math.floor(n);
}

function emptyStageStatus(maxStage: number): Record<string, StageStatus> {
  const out: Record<string, StageStatus> = {};
  for (let i = 1; i <= maxStage; i++) out[String(i)] = "pending";
  return out;
}

function emptyRetryCount(maxStage: number): Record<string, number> {
  const out: Record<string, number> = {};
  for (let i = 1; i <= maxStage; i++) out[String(i)] = 0;
  return out;
}

function emptyStage5Phases(): Stage5Phases {
  return {
    active_phase: null,
    phase_status: {},
    max_cycles_per_phase: 10,
  };
}

export function applyTransition(
  prev: OrchestratorState,
  input: TransitionInput,
): OrchestratorState {
  const next = cloneState(prev);
  const maxStage = next.max_stage ?? DEFAULT_MAX_STAGE;
  next.max_stage = maxStage;

  // Ensure required maps exist before any branch reads them.
  if (!next.stage_status) next.stage_status = emptyStageStatus(maxStage);
  if (!next.stage_retry_count) next.stage_retry_count = emptyRetryCount(maxStage);

  const statusMap = next.stage_status;
  const retryMap = next.stage_retry_count;

  switch (input.action) {
    case "init": {
      const stage = ensureStageNumber(input.stage ?? 1, maxStage);
      if (stage !== 1) {
        throw new Error(`init action must target stage 1, got stage ${stage}`);
      }
      const hasInProgress = Object.values(statusMap).some((s) => s === "in_progress");
      if (hasInProgress) {
        throw new Error(`cannot init: a stage is already in_progress`);
      }
      next.current_stage = stage;
      statusMap[String(stage)] = "in_progress";
      next.schema_version = next.schema_version ?? "2.0";
      // Clear residual state from any previous run so a re-init cannot
      // leak stale stage 5 phases, rollback history, or artifact hashes.
      next.stage5_phases = undefined;
      next.rollback_history = undefined;
      next.artifact_hashes = undefined;
      break;
    }

    case "start_stage": {
      const stage = ensureStageNumber(input.stage, maxStage);
      const otherInProgress = Object.entries(statusMap).some(
        ([k, s]) => s === "in_progress" && k !== String(stage),
      );
      if (otherInProgress) {
        throw new Error(`cannot start stage ${stage}: another stage is already in_progress`);
      }
      const key = String(stage);
      if (statusMap[key] === "completed") {
        throw new Error(`cannot start stage ${stage}: already completed`);
      }
      if (stage > 1) {
        const prevKey = String(stage - 1);
        const prevStatus = statusMap[prevKey];
        if (prevStatus !== "completed") {
          throw new Error(
            `cannot start stage ${stage}: previous stage ${stage - 1} is "${prevStatus ?? "unknown"}", not "completed"`,
          );
        }
      }
      next.current_stage = stage;
      statusMap[key] = "in_progress";
      break;
    }

    case "complete_stage": {
      const stage = ensureStageNumber(input.stage, maxStage);
      if (stage !== prev.current_stage) {
        throw new Error(
          `cannot complete stage ${stage}: current_stage is ${prev.current_stage}`,
        );
      }
      const compKey = String(stage);
      if (statusMap[compKey] !== "in_progress") {
        throw new Error(
          `cannot complete stage ${stage}: status is "${statusMap[compKey]}", not "in_progress"`,
        );
      }
      // Stage 5 may not be completed unless every phase that has been
      // started is "verified". An empty phase map is allowed (the op may
      // not need any module decomposition — single-module trivial ops).
      if (stage === 5 && next.stage5_phases) {
        const phases = next.stage5_phases.phase_status;
        for (const [pname, entry] of Object.entries(phases)) {
          if (entry.status !== "verified") {
            throw new Error(
              `cannot complete stage 5: phase ${pname} status is "${entry.status}", expected "verified"`,
            );
          }
        }
      }
      statusMap[compKey] = "completed";
      // Auto-advance to next stage if it exists.
      const nextStage = stage + 1;
      const nextKey = String(nextStage);
      if (nextKey in statusMap) {
        next.current_stage = nextStage;
        statusMap[nextKey] = "in_progress";
      }
      break;
    }

    case "fail_stage": {
      const stage = ensureStageNumber(input.stage, maxStage);
      const failKey = String(stage);
      if (stage !== prev.current_stage) {
        throw new Error(
          `cannot fail_stage ${stage}: current_stage is ${prev.current_stage}`,
        );
      }
      if (statusMap[failKey] !== "in_progress") {
        throw new Error(
          `cannot fail_stage ${stage}: status is "${statusMap[failKey]}", not "in_progress"`,
        );
      }
      retryMap[failKey] = (Number(retryMap[failKey]) || 0) + 1;
      statusMap[failKey] = "failed";
      break;
    }

    case "start_phase": {
      if (input.stage !== 5) {
        throw new Error(`start_phase only valid for stage 5, got stage ${input.stage}`);
      }
      if (statusMap["5"] !== "in_progress") {
        throw new Error(`cannot start a phase: stage 5 is not in_progress`);
      }
      if (!next.stage5_phases) next.stage5_phases = emptyStage5Phases();
      const phases = next.stage5_phases.phase_status;
      const otherActive = Object.entries(phases).find(
        ([k, e]) => e.status === "in_progress" && k !== input.phase,
      );
      if (otherActive) {
        throw new Error(
          `cannot start phase ${input.phase}: phase ${otherActive[0]} is already in_progress`,
        );
      }
      const existing = phases[input.phase];
      if (existing && existing.status === "verified") {
        throw new Error(`cannot start phase ${input.phase}: already verified`);
      }
      if (existing && existing.status === "blocked") {
        throw new Error(
          `cannot start phase ${input.phase}: already blocked (cycles exhausted; escalate or rollback_to_stage)`,
        );
      }
      phases[input.phase] = existing
        ? { ...existing, status: "in_progress" }
        : { status: "in_progress", cycles: 0 };
      next.stage5_phases.active_phase = input.phase;
      break;
    }

    case "submit_design": {
      // Designer finished producing the Stage 4 design artifacts and is
      // handing Stage 4 off to the Verifier (scaffolding mode). Symmetric to
      // submit_for_verify (which sits between Coder and Verifier in Stage 5):
      // the plugin layer runs `--check-design-gate` synchronously as a side
      // effect of this transition; lint FAIL throws before any further work.
      // The gate scope is DESIGN.md only (OL12 + OL55 in code blocks);
      // module_interfaces.yaml is intentionally NOT lint-checked by OL55 —
      // it is a structural YAML, not executable code, so pypto.<attr>
      // existence checks do not apply.
      // On PASS, no state mutation is required — Stage 4 remains in_progress
      // until complete_stage(4) is invoked after the Verifier completes the
      // adversarial harness. This action's job is purely to gate the
      // Designer → Verifier handoff with a DESIGN.md lint check.
      if (input.stage !== 4) {
        throw new Error(`submit_design only valid for stage 4, got stage ${input.stage}`);
      }
      const stageKey = String(input.stage);
      const currentStatus = next.stage_status[stageKey];
      if (currentStatus !== "in_progress") {
        throw new Error(
          `cannot submit_design at stage 4: stage_status is "${currentStatus}" ` +
          `(expected "in_progress"; call start_stage(stage=4) first)`,
        );
      }
      // No state mutation — the lint gate runs in the plugin layer; on PASS
      // we simply allow the transition through so the orchestrator can
      // proceed to dispatch the Verifier for Stage 4 scaffolding.
      break;
    }

    case "submit_for_verify": {
      // Coder dispatched the impl and is handing the phase off to the verifier.
      // The plugin layer runs `--check-phase-gate` synchronously as a side
      // effect of this transition; lint FAIL throws before the state file is
      // mutated. On PASS the phase moves to `awaiting_verify`, signalling
      // "impl is lint-clean and ready for precision verification".
      if (input.stage !== 5) {
        throw new Error(`submit_for_verify only valid for stage 5, got stage ${input.stage}`);
      }
      if (!next.stage5_phases) {
        throw new Error(`cannot submit phase for verify: stage5_phases not initialised`);
      }
      const phases = next.stage5_phases.phase_status;
      const entry = phases[input.phase];
      if (!entry) {
        throw new Error(`cannot submit phase ${input.phase} for verify: not started`);
      }
      if (
        entry.status !== "in_progress" &&
        entry.status !== "in_debug"
      ) {
        throw new Error(
          `cannot submit phase ${input.phase} for verify: status is "${entry.status}" ` +
          `(expected "in_progress" or "in_debug")`,
        );
      }
      phases[input.phase] = { ...entry, status: "awaiting_verify" };
      next.stage5_phases.active_phase = input.phase;
      break;
    }

    case "complete_phase": {
      if (input.stage !== 5) {
        throw new Error(`complete_phase only valid for stage 5, got stage ${input.stage}`);
      }
      if (!next.stage5_phases) {
        throw new Error(`cannot complete phase: stage5_phases not initialised`);
      }
      const phases = next.stage5_phases.phase_status;
      const entry = phases[input.phase];
      if (!entry) {
        throw new Error(`cannot complete phase ${input.phase}: not started`);
      }
      // complete_phase is reachable from:
      //   - awaiting_verify (normal path after submit_for_verify + verifier PASS)
      //   - in_progress (legacy / direct completion without explicit submit step)
      //   - in_debug   (recovered from debug loop, verifier PASS without re-submit)
      if (
        entry.status !== "in_progress" &&
        entry.status !== "in_debug" &&
        entry.status !== "awaiting_verify"
      ) {
        throw new Error(
          `cannot complete phase ${input.phase}: status is "${entry.status}"`,
        );
      }
      phases[input.phase] = {
        ...entry,
        status: "verified",
        verified_at: new Date().toISOString(),
        // failure_category/failing_module_boundary/last_error are kept for audit
      };
      if (next.stage5_phases.active_phase === input.phase) {
        next.stage5_phases.active_phase = null;
      }
      break;
    }

    case "fail_phase": {
      if (input.stage !== 5) {
        throw new Error(`fail_phase only valid for stage 5, got stage ${input.stage}`);
      }
      if (!next.stage5_phases) next.stage5_phases = emptyStage5Phases();
      const phases = next.stage5_phases.phase_status;
      const entry = phases[input.phase];
      if (!entry) {
        throw new Error(`cannot fail_phase ${input.phase}: phase not started`);
      }
      const cycles = entry.cycles + 1;
      const blocked = cycles >= next.stage5_phases.max_cycles_per_phase;
      phases[input.phase] = {
        ...entry,
        status: blocked ? "blocked" : "in_debug",
        cycles,
        failure_category: input.failure_category,
        failing_module_boundary: input.failing_module_boundary ?? null,
        last_error: input.last_error,
      };
      // active_phase remains pointed at this phase for the next dispatch
      next.stage5_phases.active_phase = input.phase;
      break;
    }

    case "record_artifact_hash": {
      if (!input.name) throw new Error(`record_artifact_hash requires name`);
      if (!input.hash) throw new Error(`record_artifact_hash requires hash`);
      if (!next.artifact_hashes) next.artifact_hashes = {};
      next.artifact_hashes[input.name] = input.hash;
      break;
    }

    case "rollback_to_stage": {
      const target = ensureStageNumber(input.target_stage, maxStage);
      if (target >= prev.current_stage) {
        throw new Error(
          `rollback target stage ${target} must be strictly less than current_stage ${prev.current_stage}`,
        );
      }
      if (!input.reason || !input.reason.trim()) {
        throw new Error(`rollback_to_stage requires a non-empty reason`);
      }
      // Reset every stage strictly after target to pending.
      for (const key of Object.keys(statusMap)) {
        const k = Number(key);
        if (k > target) statusMap[key] = "pending";
      }
      // Increment retry count on target stage and re-enter it.
      retryMap[String(target)] = (Number(retryMap[String(target)]) || 0) + 1;
      statusMap[String(target)] = "in_progress";
      next.current_stage = target;
      // Wipe stage 5 phase progress — the module decomposition itself
      // may change after going back to architecture/design.
      if (target < 5) {
        next.stage5_phases = emptyStage5Phases();
      }
      // Drop artifact hashes that were computed at stages strictly after target.
      const HASH_STAGE: Record<string, number> = {
        spec_md: 1,
        api_report_md: 1,
        golden_py: 2,
        design_md: 3,
        module_interfaces_yaml: 4,
      };
      if (next.artifact_hashes) {
        for (const [name, owningStage] of Object.entries(HASH_STAGE)) {
          if (owningStage > target && name in next.artifact_hashes) {
            delete next.artifact_hashes[name];
          }
        }
      }
      // Append rollback history (append-only).
      if (!next.rollback_history) next.rollback_history = [];
      next.rollback_history.push({
        from_stage: prev.current_stage,
        from_phase: input.failed_phase,
        to_stage: target,
        reason: input.reason,
        failure_category: input.failure_category,
        timestamp: new Date().toISOString(),
      });
      break;
    }

    default: {
      const a = (input as { action?: string }).action ?? "unknown";
      throw new Error(`unsupported action: ${a}`);
    }
  }

  next.last_updated = new Date().toISOString();
  return next;
}
