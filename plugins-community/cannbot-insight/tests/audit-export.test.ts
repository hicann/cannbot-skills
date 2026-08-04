// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from "vitest"
import { buildAuditExportPayload } from "@/lib/audit-export"
import type { Analysis } from "@/components/observe/WorkflowFlowChart"

const analysis: Analysis = {
  sessionSummary: "demo",
  sessionMeta: {},
  flow: [],
  workflowLevelIssues: [],
  optimizationPriorities: [],
}

describe("audit-export payload", () => {
  it("builds pretty-printed JSON keyed off taskId", () => {
    const p = buildAuditExportPayload("ses_abc123", analysis)
    expect(p.defaultName).toBe("session-ses_abc123-analysis.json")
    expect(p.mime).toBe("application/json")
    // pretty-printed (2-space) and round-trips back into the same analysis
    expect(p.text).toContain('"sessionSummary": "demo"')
    expect(JSON.parse(p.text)).toEqual(analysis)
  })

  it("filename + content stay in sync with whatever analysis is passed", () => {
    const big = { ...analysis, flow: [{ id: "n1" } as never] }
    const p = buildAuditExportPayload("t-7", big)
    expect(p.defaultName).toBe("session-t-7-analysis.json")
    expect(JSON.parse(p.text)).toEqual(big)
  })
})
