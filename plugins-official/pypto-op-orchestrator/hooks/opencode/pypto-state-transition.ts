// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { type Plugin, tool } from "@opencode-ai/plugin";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import {
  applyTransition,
  type OrchestratorState,
  type TransitionAction,
  type TransitionInput,
} from "./lib/state-transition-core";

type GateFinding = {
  rule_id: string;
  severity: string;
  status: string;
  message: string;
  file: string;
};

type GateSummary = {
  warnCount: number;
  infoCount: number;
  /** FAIL findings included when gate blocks — gives the agent actionable detail. */
  failFindings: GateFinding[];
};

const ALLOWED_ACTIONS = new Set<TransitionAction>([
  "init",
  "start_stage",
  "complete_stage",
  "fail_stage",
  "submit_design",
  "start_phase",
  "submit_for_verify",
  "complete_phase",
  "fail_phase",
  "record_artifact_hash",
  "rollback_to_stage",
]);

/** The AGENTS.md primary runs as "build"; the legacy named primary remains compatible. */
const ALLOWED_AGENTS = new Set<string>([
  "build",
  "pypto-op-orchestrator",
]);

const DEFAULT_MAX_STAGE = 7;

function parseState(content: string): OrchestratorState {
  const parsed = JSON.parse(content);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("invalid orchestrator state file content");
  }
  return parsed as OrchestratorState;
}

function parseGateSummary(raw: string): GateSummary {
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const summary = parsed.summary;
    if (!summary || typeof summary !== "object" || Array.isArray(summary)) {
      return { warnCount: 0, infoCount: 0, failFindings: [] };
    }
    const warnCount = Number((summary as Record<string, unknown>).warn);
    const infoCount = Number((summary as Record<string, unknown>).info);

    // Extract FAIL findings for detailed error reporting
    const failFindings: GateFinding[] = [];
    const findings = parsed.findings;
    if (Array.isArray(findings)) {
      for (const f of findings) {
        if (f && typeof f === "object" && (f as Record<string, unknown>).status === "FAIL") {
          failFindings.push({
            rule_id: String((f as Record<string, unknown>).rule_id ?? ""),
            severity: String((f as Record<string, unknown>).severity ?? ""),
            status: "FAIL",
            message: String((f as Record<string, unknown>).message ?? ""),
            file: String((f as Record<string, unknown>).file ?? ""),
          });
        }
      }
    }

    return {
      warnCount: Number.isFinite(warnCount) ? warnCount : 0,
      infoCount: Number.isFinite(infoCount) ? infoCount : 0,
      failFindings,
    };
  } catch {
    return { warnCount: 0, infoCount: 0, failFindings: [] };
  }
}

function buildInitialState(opDir: string, maxStage: number): OrchestratorState {
  const stageStatus: Record<string, string> = {};
  const stageRetry: Record<string, number> = {};
  for (let i = 1; i <= maxStage; i++) {
    stageStatus[String(i)] = "pending";
    stageRetry[String(i)] = 0;
  }
  return {
    operator_name: path.basename(opDir),
    schema_version: "2.0",
    max_stage: maxStage,
    current_stage: 1,
    stage_status: stageStatus,
    stage_retry_count: stageRetry,
    artifact_hashes: {},
    rollback_history: [],
    last_updated: new Date().toISOString(),
  };
}

function readStateOrInit(statePath: string, maxStage: number): OrchestratorState {
  if (!fs.existsSync(statePath)) {
    return buildInitialState(path.dirname(statePath), maxStage);
  }
  const raw = fs.readFileSync(statePath, "utf8");
  return parseState(raw);
}

function writeStateAtomically(statePath: string, state: OrchestratorState): void {
  const tmpPath = `${statePath}.tmp`;
  fs.writeFileSync(tmpPath, `${JSON.stringify(state, null, 2)}\n`, "utf8");
  fs.renameSync(tmpPath, statePath);
}

function computeFileHash(filePath: string): string | null {
  if (!fs.existsSync(filePath)) return null;
  const content = fs.readFileSync(filePath, "utf8");
  return crypto.createHash("sha256").update(content).digest("hex");
}

/**
 * Build the TransitionInput payload for applyTransition() based on the
 * tool args. Only fields relevant to each action are forwarded.
 */
function buildTransitionInput(action: TransitionAction, args: Record<string, unknown>): TransitionInput {
  switch (action) {
    case "init":
      return {
        action,
        stage: args.stage !== undefined ? Number(args.stage) : 1,
        max_stage: args.max_stage !== undefined ? Number(args.max_stage) : undefined,
      };
    case "start_stage":
      return { action, stage: Number(args.stage), reason: args.reason as string | undefined };
    case "complete_stage":
      return { action, stage: Number(args.stage) };
    case "fail_stage":
      return { action, stage: Number(args.stage), reason: args.reason as string | undefined };
    case "submit_design":
      return { action, stage: 4 };
    case "start_phase":
      return { action, stage: 5, phase: String(args.phase ?? "") };
    case "submit_for_verify":
      return { action, stage: 5, phase: String(args.phase ?? "") };
    case "complete_phase":
      return { action, stage: 5, phase: String(args.phase ?? "") };
    case "fail_phase":
      return {
        action,
        stage: 5,
        phase: String(args.phase ?? ""),
        failure_category: String(args.failure_category ?? ""),
        failing_module_boundary: (args.failing_module_boundary as string | null | undefined) ?? null,
        last_error: args.last_error as string | undefined,
      };
    case "record_artifact_hash":
      return {
        action,
        name: String(args.name ?? ""),
        hash: String(args.hash ?? ""),
      };
    case "rollback_to_stage":
      return {
        action,
        target_stage: Number(args.target_stage),
        reason: String(args.reason ?? ""),
        failure_category: args.failure_category as string | undefined,
        failed_phase: args.failed_phase as string | undefined,
      };
    default: {
      throw new Error(`unsupported action: ${action}`);
    }
  }
}

export const PyptoStateTransitionPlugin: Plugin = async (input) => {
  const $ = input.$;
  const client = input.client;
  const baseDir = input.directory || input.worktree || process.cwd();
  const lintScript = new URL(
    "../hooks/pypto-op-lint/pypto_op_lint.py",
    import.meta.url,
  ).pathname;

  /**
   * Run the lint gate when:
   * - `complete_stage` is invoked — full Stage delivery gate
   *   (`--check-gate --stage <N>`, covers all impl/test/golden/gate rules for
   *   the stage including integrated artifacts).
   * - `submit_design` is invoked — Stage 4 design gate
   *   (`--check-design-gate`, covers ONLY OL12 + OL55 over DESIGN.md so the
   *   Designer's pseudo-code is lint-clean BEFORE the Verifier is dispatched
   *   for Stage 4 scaffolding; module_interfaces.yaml is intentionally NOT
   *   in scope because it is a structural YAML, not executable code).
   * - `complete_phase` / `submit_for_verify` is invoked — per-Phase M_k
   *   phase-scoped gate (`--check-phase-gate --phase M_k`, covers ONLY the
   *   phase's cumulative module impl file `modules/<op>_module<suffix>_impl.py`;
   *   integrated `<op>_impl.py` and `test_<op>.py` are excluded because they
   *   are Stage 5 cleanup artifacts produced after the inner loop finishes).
   *
   * Rationale (phase gate): invoking the full `--check-gate --stage 5` at
   * `complete_phase` surfaces OL01/OL07/OL08/OL18/etc. against placeholder
   * integrated artifacts and falsely blocks a phase that did its job.
   * Phase-scoping eliminates the false positive while still enforcing
   * impl-target rules on the new module file.
   *
   * Rationale (design gate): catching `pypto.empty`-style typos at Designer
   * exit (rather than at `complete_stage(4)` after the Verifier already
   * produced the adversarial harness) avoids wasting the Verifier's work
   * on a DESIGN.md whose pseudo-code references nonexistent PyPTO APIs.
   *
   * On block the orchestrator should re-dispatch the upstream agent
   * (Designer for submit_design, Coder for submit_for_verify/complete_phase)
   * with the failure details listed in fix_hints.
   */
  async function runGateIfNeeded(
    opDir: string,
    action: TransitionAction,
    stage: number | undefined,
    phase: string | undefined,
  ): Promise<GateSummary> {
    let lintCmd;
    if (action === "complete_stage" && stage !== undefined) {
      lintCmd = $`python3 ${lintScript} --check-gate --op-dir ${opDir} --stage ${stage}`;
    } else if (action === "submit_design") {
      // Stage 4 design gate: OL12 + OL55 on DESIGN.md only. Designer → Verifier
      // handoff; refuse if pseudo-code references nonexistent pypto.<attr>.
      lintCmd = $`python3 ${lintScript} --check-design-gate --op-dir ${opDir}`;
    } else if (
      (action === "complete_phase" || action === "submit_for_verify") &&
      phase !== undefined
    ) {
      // Phase-scoped gate: only the phase's cumulative module impl file is scanned.
      // Same command for both actions; the state machine layer decides which
      // status transition (awaiting_verify / verified) happens on success.
      lintCmd = $`python3 ${lintScript} --check-phase-gate --op-dir ${opDir} --phase ${phase}`;
    } else {
      return { warnCount: 0, infoCount: 0, failFindings: [] };
    }

    const result = await lintCmd.cwd(baseDir).quiet().nothrow();
    const raw = result.stdout.toString();
    const summary = parseGateSummary(raw);

    if (result.exitCode !== 0) {
      const details = summary.failFindings
        .map((f) => `  [${f.rule_id}][${f.severity}] ${f.message}${f.file ? ` (${f.file})` : ""}`)
        .join("\n");
      const scope = action === "complete_phase"
        ? `Phase ${phase ?? "?"} completion`
        : action === "submit_for_verify"
          ? `Phase ${phase ?? "?"} verify-handoff`
          : action === "submit_design"
            ? "Stage 4 design-handoff"
            : "Stage completion";
      const guidance = action === "complete_phase" || action === "submit_for_verify"
        ? "\n\nRecommended next step: re-dispatch pypto-op-coder for the same phase, " +
          "instruct it to fix the violations listed above, and call complete_phase (or " +
          "submit_for_verify) again."
        : action === "submit_design"
          ? "\n\nRecommended next step: re-dispatch pypto-op-designer with the violations " +
            "listed above (typically a `pypto.<attr>` typo in DESIGN.md pseudo-code). " +
            "After Designer fixes DESIGN.md, call submit_design again to re-gate before " +
            "the Verifier (Stage 4 scaffolding) is dispatched."
          : "";
      throw new Error(
        details
          ? `${scope} blocked by lint rule violations:\n${details}${guidance}`
          : `lint script returned exit code ${result.exitCode} with no parsable findings`,
      );
    }

    return summary;
  }

  return {
    tool: {
      state_transition: tool({
        description:
          "Safely transition .orchestrator_state.json (schema v2.0). " +
          "Stage actions: init (stage=1 only, first call), start_stage (set stage to in_progress for retry), " +
          "complete_stage (lint gate check + mark done + auto-advance), fail_stage (mark failed + increment retry). " +
          "Stage 4 design action: submit_design (Designer→Verifier handoff — runs design-scoped lint OL12+OL55 on DESIGN.md only, throws on FAIL so the Designer is re-dispatched BEFORE the Verifier wastes a cycle on a typo-bearing DESIGN.md; module_interfaces.yaml is intentionally not in scope). " +
          "Stage 5 phase actions (per-Phase M_k loop): start_phase, submit_for_verify (Coder→Verifier handoff — runs phase-scoped lint, moves status to awaiting_verify on success), complete_phase, fail_phase. " +
          "Other actions: record_artifact_hash (snapshot SPEC.md/DESIGN.md/etc. hashes), " +
          "rollback_to_stage (return to an earlier stage with reason and optional failure_category — wipes downstream stages and stage5_phases when target<5).",
        args: {
          opDir: tool.schema.string(),
          action: tool.schema.string(),
          // Stage actions
          stage: tool.schema.number().optional(),
          max_stage: tool.schema.number().optional(),
          reason: tool.schema.string().optional(),
          // Phase actions (stage=5 only)
          phase: tool.schema.string().optional(),
          failure_category: tool.schema.string().optional(),
          failing_module_boundary: tool.schema.string().optional(),
          last_error: tool.schema.string().optional(),
          // Artifact hash
          name: tool.schema.string().optional(),
          hash: tool.schema.string().optional(),
          // Rollback
          target_stage: tool.schema.number().optional(),
          failed_phase: tool.schema.string().optional(),
        },
        execute: async (args, context) => {
          // ── Permission check: AGENTS.md primary only ──
          const callerAgent = context?.agent ?? "";
          if (!ALLOWED_AGENTS.has(callerAgent)) {
            throw new Error(
              `permission denied: state_transition is restricted to the PyPTO classic primary, ` +
              `but was called by "${callerAgent || "(unknown)"}". ` +
              `Subagents must not modify .orchestrator_state.json; return stage results to the orchestrator instead.`,
            );
          }

          const action = String(args.action) as TransitionAction;
          if (!ALLOWED_ACTIONS.has(action)) {
            throw new Error(`unsupported action: ${args.action}`);
          }

          const opDir = path.isAbsolute(args.opDir)
            ? args.opDir
            : path.resolve(baseDir, args.opDir);
          const customRoot = path.resolve(baseDir, "custom");
          const relativeToCustomRoot = path.relative(customRoot, opDir);
          if (
            relativeToCustomRoot === "" ||
            relativeToCustomRoot === ".." ||
            relativeToCustomRoot.startsWith(`..${path.sep}`) ||
            path.isAbsolute(relativeToCustomRoot)
          ) {
            throw new Error(
              `permission denied: state_transition may only update an operator directory under ` +
              `${customRoot}, but received ${opDir}.`,
            );
          }
          const statePath = path.join(opDir, ".orchestrator_state.json");
          if (action === "init") {
            fs.mkdirSync(opDir, { recursive: true });
          }
          const desiredMaxStage = args.max_stage !== undefined ? Number(args.max_stage) : DEFAULT_MAX_STAGE;
          const prevState = readStateOrInit(statePath, desiredMaxStage);

          // Lint gate fires on:
          // - complete_stage  — full stage delivery rules.
          // - submit_design   — Stage 4 design gate (OL12 + OL55 on DESIGN.md).
          //   Designer → Verifier handoff; catches `pypto.empty`-style typos
          //   before Verifier wastes a cycle producing the adversarial harness.
          // - complete_phase  — phase-scoped final gate (Coder + Verifier done).
          // - submit_for_verify — phase-scoped gate at Coder→Verifier handoff.
          //   Same lint as complete_phase; the state transition that follows
          //   is in_progress → awaiting_verify (not → verified). This catches
          //   Coder's simple mistakes BEFORE Verifier runs and wastes a cycle.
          // Artifact / rollback / start_* / fail_* actions still skip the gate.
          let gateSummary: GateSummary = { warnCount: 0, infoCount: 0, failFindings: [] };
          if (
            action === "complete_stage" ||
            action === "submit_design" ||
            action === "complete_phase" ||
            action === "submit_for_verify"
          ) {
            try {
              gateSummary = await runGateIfNeeded(
                opDir,
                action,
                args.stage as number | undefined,
                args.phase as string | undefined,
              );
            } catch (error) {
              const detail = error instanceof Error ? error.message : String(error);
              const scope =
                action === "complete_phase"
                  ? `Phase ${args.phase ?? "?"} completion`
                  : action === "submit_for_verify"
                  ? `Phase ${args.phase ?? "?"} verify-handoff`
                  : action === "submit_design"
                  ? `Stage 4 design-handoff`
                  : `Stage completion`;
              throw new Error(
                `[pypto-op-lint] ${scope} blocked: action=${action}, ` +
                `stage=${args.stage ?? "-"}, phase=${args.phase ?? "-"}, op_dir=${opDir}\n${detail}`,
              );
            }
          }

          // ── SPEC.md freeze enforcement ──
          // complete_stage(1) auto-records the SPEC.md hash into artifact_hashes.spec_md.
          // From stage 3 onward, every complete_stage rejects if SPEC.md changed.
          // To legitimately re-edit SPEC.md, the orchestrator must call
          // rollback_to_stage(target_stage=1, reason=...).
          const specPath = path.join(opDir, "SPEC.md");
          if (action === "complete_stage" && Number(args.stage) === 1) {
            const hash = computeFileHash(specPath);
            if (hash) {
              prevState.artifact_hashes = prevState.artifact_hashes ?? {};
              prevState.artifact_hashes.spec_md = hash;
            }
          }
          if (action === "complete_stage" && Number(args.stage) >= 3) {
            const savedHash = prevState.artifact_hashes?.spec_md;
            if (typeof savedHash === "string") {
              const currentHash = computeFileHash(specPath);
              if (currentHash && currentHash !== savedHash) {
                throw new Error(
                  `SPEC.md freeze violation: SPEC.md was modified after Stage 1 completion. ` +
                  `current hash=${currentHash.slice(0, 12)}… recorded hash=${savedHash.slice(0, 12)}…. ` +
                  `To legitimately revise the spec, call rollback_to_stage(target_stage=1, reason=...).`,
                );
              }
            }
          }

          // Build and apply the transition
          const transitionInput = buildTransitionInput(action, args as Record<string, unknown>);
          const nextState = applyTransition(prevState, transitionInput);

          writeStateAtomically(statePath, nextState);

          // Audit log (non-fatal)
          try {
            await client.app.log({
              body: {
                service: "pypto-state-transition",
                level: "info",
                message: `[state_transition] ${action} completed`,
                extra: {
                  opDir,
                  action,
                  stage: args.stage,
                  phase: args.phase,
                  target_stage: args.target_stage,
                  reason: args.reason ?? "",
                  warnCount: gateSummary.warnCount,
                  infoCount: gateSummary.infoCount,
                },
              },
            });
          } catch {
            // no-op
          }

          return JSON.stringify({
            ok: true,
            action,
            stage: args.stage,
            phase: args.phase,
            target_stage: args.target_stage,
            current_stage: nextState.current_stage,
            stage5_active_phase: nextState.stage5_phases?.active_phase ?? null,
            statePath,
            warnCount: gateSummary.warnCount,
            infoCount: gateSummary.infoCount,
          });
        },
      }),
    },
  };
};
