// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// CANNBay retention policy: master holds at most the N newest session .db files;
// older ones rotate off master onto a date-bucketed archive branch. Upload and
// download only ever touch master, so capping it keeps clone/push bounded.

export const MAX_MASTER_SESSIONS = 20;

export interface FileWithDate {
  filename: string;
  /** Commit date (ms) of the commit that first added this file to master. */
  addedAtMs: number;
}

export interface ArchivePlan {
  archiveBranch: string;
  /** Filenames to move OFF master (oldest first), empty when no rotation needed. */
  toArchive: string[];
  /** Filenames that stay on master. */
  keepOnMaster: string[];
}

/** Archive branch is a monthly bucket keyed by the archive-run moment. */
export function archiveBranchName(runMs: number): string {
  const d = new Date(runMs);
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  return `archive-${d.getFullYear()}-${mm}`;
}

/**
 * Given the .db files currently on master (with their add-commit dates), decide
 * which rotate off when master is capped at `keepMax`. Oldest beyond the cap are
 * archived; the newest `keepMax` stay. No rotation when count <= keepMax.
 */
export function planArchive(files: FileWithDate[], keepMax: number, runMs: number): ArchivePlan {
  const sorted = [...files].sort((a, b) => a.addedAtMs - b.addedAtMs); // oldest first
  const overflow = Math.max(0, sorted.length - keepMax);
  return {
    archiveBranch: archiveBranchName(runMs),
    toArchive: sorted.slice(0, overflow).map((f) => f.filename),
    keepOnMaster: sorted.slice(overflow).map((f) => f.filename),
  };
}
