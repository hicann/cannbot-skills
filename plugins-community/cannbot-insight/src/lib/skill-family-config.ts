// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import fs from "node:fs";
import path from "node:path";

const DEFAULT_ORCHESTRATOR_PATTERNS = ["workflow", "invoke-workflow"];
const DEFAULT_FAMILY_SEGMENT_COUNT = 2;
const DEFAULT_PHASE_GAP_TURNS = 40;
const VENDOR_PREFIXES = ["ascendc-", "ops-", "npu-", "cann-"];

export interface PhaseInfo {
  key: string;
  label: string;
  order: number;
}

interface SkillFamilyConfig {
  orchestratorPatterns?: string[];
  orchestratorLabel?: string;
  familySegmentCount?: number;
  phaseGapTurns?: number;
  familyLabels?: Record<string, string>;
  phases?: Record<string, { label?: string; order?: number }>;
}

let cachedConfig: SkillFamilyConfig | null = null;

function loadConfig(): SkillFamilyConfig {
  if (cachedConfig !== null) return cachedConfig;

  const configPath = path.join(process.cwd(), "skill-families.json");
  try {
    if (fs.existsSync(configPath)) {
      const raw = fs.readFileSync(configPath, "utf-8");
      cachedConfig = JSON.parse(raw) as SkillFamilyConfig;
      return cachedConfig;
    }
  } catch (e) {
    console.warn("Failed to load skill-families.json, using defaults:", e);
  }

  cachedConfig = {};
  return cachedConfig;
}

export function resetSkillFamilyConfigCache(): void {
  cachedConfig = null;
}

function patterns(): string[] {
  return loadConfig().orchestratorPatterns ?? DEFAULT_ORCHESTRATOR_PATTERNS;
}

function segmentCount(): number {
  const n = loadConfig().familySegmentCount ?? DEFAULT_FAMILY_SEGMENT_COUNT;
  return n > 0 ? n : DEFAULT_FAMILY_SEGMENT_COUNT;
}

export function getPhaseGapTurns(): number {
  const g = loadConfig().phaseGapTurns ?? DEFAULT_PHASE_GAP_TURNS;
  return g > 0 ? g : DEFAULT_PHASE_GAP_TURNS;
}

export function isOrchestratorSkill(skillName: string): boolean {
  const lower = skillName.toLowerCase();
  return patterns().some(p => lower.includes(p.toLowerCase()));
}

export function getPhaseInfo(key: string): PhaseInfo | null {
  const def = loadConfig().phases?.[key];
  if (!def) return null;
  return { key, label: def.label ?? key, order: def.order ?? 99 };
}

export function getAllPhaseKeys(): string[] {
  const phases = loadConfig().phases ?? {};
  return Object.entries(phases)
    .sort((a, b) => (a[1].order ?? 99) - (b[1].order ?? 99))
    .map(([k]) => k);
}

export function getFamilyKey(skillName: string): string {
  if (isOrchestratorSkill(skillName)) return "__orchestrator__";
  const segs = skillName.split("-");
  const n = segmentCount();
  return segs.length <= n ? skillName : segs.slice(0, n).join("-");
}

export function getFamilyLabel(familyKey: string): string {
  const labels = loadConfig().familyLabels ?? {};
  if (labels[familyKey]) return labels[familyKey];

  if (familyKey === "__orchestrator__") return "启动";

  let key = familyKey;
  for (const prefix of VENDOR_PREFIXES) {
    if (key.toLowerCase().startsWith(prefix)) {
      key = key.slice(prefix.length);
      break;
    }
  }
  return key || familyKey;
}
