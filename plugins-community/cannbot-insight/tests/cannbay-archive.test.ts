// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from "vitest";
import { planArchive, archiveBranchName, MAX_MASTER_SESSIONS, type FileWithDate } from "@/lib/cannbay-archive";

const RUN = Date.UTC(2026, 6, 23, 0, 0, 0); // 2026-07-23

function f(name: string, daysAgo: number): FileWithDate {
  return { filename: name, addedAtMs: RUN - daysAgo * 86_400_000 };
}

describe("cannbay-archive · planArchive", () => {
  it("rotates nothing when count <= cap", () => {
    const plan = planArchive([f("a.db", 1), f("b.db", 2)], MAX_MASTER_SESSIONS, RUN);
    expect(plan.toArchive).toEqual([]);
    expect(plan.keepOnMaster).toHaveLength(2);
  });

  it("keeps newest 20, archives the oldest overflow", () => {
    // s0 = 1 day ago (newest), s24 = 25 days ago (oldest)
    const files = Array.from({ length: 25 }, (_, i) => f(`s${i}.db`, i + 1));
    const plan = planArchive(files, MAX_MASTER_SESSIONS, RUN);
    expect(plan.toArchive).toHaveLength(5);
    expect(plan.keepOnMaster).toHaveLength(20);
    // archived = the 5 with the largest daysAgo (s20..s24); kept = s0..s19
    const archSet = new Set(plan.toArchive);
    const keepSet = new Set(plan.keepOnMaster);
    expect(archSet.size).toBe(5);
    for (let i = 0; i < 25; i++) {
      const name = `s${i}.db`;
      if (i >= 20) expect(archSet.has(name)).toBe(true);
      else expect(keepSet.has(name)).toBe(true);
    }
    expect(archSet.size + keepSet.size).toBe(25);
  });

  it("archives exactly the overflow when cap is 1", () => {
    const plan = planArchive([f("old.db", 10), f("new.db", 1)], 1, RUN);
    expect(plan.toArchive).toEqual(["old.db"]);
    expect(plan.keepOnMaster).toEqual(["new.db"]);
  });

  it("branch name is monthly-bucketed from the run moment", () => {
    expect(archiveBranchName(RUN)).toBe("archive-2026-07");
    expect(planArchive([], MAX_MASTER_SESSIONS, RUN).archiveBranch).toBe("archive-2026-07");
    // same-month runs reuse the same branch; next month is a new bucket
    expect(archiveBranchName(Date.UTC(2026, 7, 1))).toBe("archive-2026-08");
  });
});
