// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, beforeAll, vi, afterAll } from "vitest"
import fs from "node:fs"
import path from "node:path"
import {
  validateSchema,
  analyzeTrajectory,
  AnalysisError,
  SchemaError,
} from "@/lib/ai/trajectory-analyzer"

const DEMO_PATH = path.resolve(__dirname, "../src/lib/workflow-demo-analysis.json")

describe("validateSchema", () => {
  it("demo JSON 缺 workflowMeta 抛 SchemaError", () => {
    const demo = JSON.parse(fs.readFileSync(DEMO_PATH, "utf-8"))
    expect(() => validateSchema(demo)).toThrow(SchemaError)
    expect(() => validateSchema(demo)).toThrow(/workflowMeta/)
  })

  it("顶层非对象抛", () => {
    expect(() => validateSchema("string")).toThrow(SchemaError)
  })
})

describe("analyzeTrajectory 有界循环（mock fetch）", () => {
  const tmpDir = path.resolve(__dirname, "data/tmp-traj-test")
  const tmpTrajectory = path.join(tmpDir, "log-fake.md")
  const fakeProvider = {
    baseUrl: "https://fake.example.com/v1",
    apiKey: "sk-fake",
    model: "fake-model",
  }

  beforeAll(() => {
    fs.mkdirSync(tmpDir, { recursive: true })
    fs.writeFileSync(
      tmpTrajectory,
      [
        '## §1 User',
        '',
        '"/ops-registry-invoke-glacier softplus_v2_grad"',
        '',
        '---',
        '',
        '## §2 Assistant',
        '',
        '*Skill: ops-registry-invoke-glacier (invoke) ✅',
        '',
        '**Tool: skill**',
        '',
        '**Output:**',
        '',
        '```',
        '<skill_content name="ops-registry-invoke-glacier">',
        '# fake skill',
        'fake body',
        '</skill_content>',
        '```',
        '',
        'PASS',
        '',
        '*Error: unknown*',
        '',
        '## Stats',
        '',
        '| Metric | Root | Subagent(s) | Total |',
        '|--------|------|-------------|-------|',
        '| Tokens | 1K | 1K | 2K |',
        '| Cost | $0 | $0 | $0 |',
        '| Turns | 2 | 0 | 2 |',
        '| Subagents | — | 0 | 0 |',
        '',
        '**Duration:** 1.0h | **Tokens:** 2K (in: 1K / out: 1K)',
      ].join("\n"),
      "utf-8",
    )
  })

  afterAll(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true })
  })

  it("round1 读请求 → round2 完整 JSON → 写文件", async () => {
    let callCount = 0
    const validAnalysis = {
      sessionSummary: "fake",
      workflowMeta: { skillName: "ops-registry-invoke-glacier", workflowType: "linear", phases: [], gates: [], requiredSkills: [], orderingRule: "strict-sequential", source: "test" },
      sessionMeta: { cpsExecuted: [], cpsMissing: [], phasesNotReached: [] },
      flow: [],
      skillQuality: [],
      workflowLevelIssues: [],
      optimizationPriorities: [],
    }

    const fetchSpy = vi.spyOn(global, "fetch").mockImplementation(async () => {
      callCount++
      const content = callCount === 1
        ? JSON.stringify({ read: { lines: [1, 5] } })
        : JSON.stringify(validAnalysis)
      return {
        ok: true,
        json: async () => ({ choices: [{ message: { content } }] }),
      } as never
    })

    try {
      const result = await analyzeTrajectory({
        trajectoryPath: tmpTrajectory,
        promptPath: path.resolve(__dirname, "../prompts/session-trajectory-analyse.md"),
        outputDir: tmpDir,
        provider: fakeProvider,
      })

      expect(fetchSpy).toHaveBeenCalledTimes(2)
      expect(result.rounds).toBe(2)
      expect(result.outputPath).toContain("log-fake-analysis.json")
      expect(fs.existsSync(result.outputPath)).toBe(true)
      const written = JSON.parse(fs.readFileSync(result.outputPath, "utf-8"))
      expect(written.sessionSummary).toBe("fake")
    } finally {
      fetchSpy.mockRestore()
    }
  })

  it("3 轮无有效 JSON 抛 AnalysisError", async () => {
    const fetchSpy = vi.spyOn(global, "fetch").mockImplementation(async () => ({
      ok: true,
      json: async () => ({ choices: [{ message: { content: "not json at all" } }] }),
    }) as never)

    try {
      await expect(
        analyzeTrajectory({
          trajectoryPath: tmpTrajectory,
          promptPath: path.resolve(__dirname, "../prompts/session-trajectory-analyse.md"),
          outputDir: tmpDir,
          provider: fakeProvider,
        }),
      ).rejects.toThrow(AnalysisError)

      expect(fetchSpy).toHaveBeenCalledTimes(3)
    } finally {
      fetchSpy.mockRestore()
    }
  })

  it("轨迹文件不存在抛 AnalysisError", async () => {
    await expect(
      analyzeTrajectory({
        trajectoryPath: "/nonexistent/path.md",
        provider: fakeProvider,
      }),
    ).rejects.toThrow(AnalysisError)
  })
})

describe("analyzeTrajectoryByTask（mock markdown-exporter + mock fetch）", () => {
  const tmpDir = path.resolve(__dirname, "data/tmp-traj-bytask")
  const fakeProvider = {
    baseUrl: "https://fake.example.com/v1",
    apiKey: "sk-fake",
    model: "fake-model",
  }
  const fakePrisma = {} as never

  beforeAll(() => {
    fs.mkdirSync(tmpDir, { recursive: true })
  })

  afterAll(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true })
  })

  it("内存生 MD → LLM 返回完整 JSON → 返回 analysis 内联", async () => {
    vi.resetModules()
    const exporterMod = await import("@/lib/export/markdown-exporter")
    vi.spyOn(exporterMod, "exportSessionToMarkdown").mockResolvedValue(
      [
        '## §1 User',
        '',
        '## §2 Assistant',
        '',
        '*Skill: ops-registry-invoke-glacier (invoke) ✅',
        '',
        '**Tool: skill**',
        '',
        '**Output:**',
        '',
        '<skill_content name="ops-registry-invoke-glacier">',
        '# fake',
        'fake body',
        '</skill_content>',
        '',
        'PASS',
        '',
        '**Duration:** 1.0h | **Tokens:** 2K',
        '',
        '## Stats',
        '',
        '| Metric | Root | Subagent(s) | Total |',
        '|--------|------|-------------|-------|',
        '| Turns | 2 | 0 | 2 |',
      ].join("\n"),
    )
    const analyzerMod = await import("@/lib/ai/trajectory-analyzer")

    const validAnalysis = {
      sessionSummary: "by-task fake",
      workflowMeta: { skillName: "ops-registry-invoke-glacier", workflowType: "linear", phases: [], gates: [], requiredSkills: [], orderingRule: "strict-sequential", source: "test" },
      sessionMeta: { cpsExecuted: [], cpsMissing: [], phasesNotReached: [] },
      flow: [],
      skillQuality: [],
      workflowLevelIssues: [],
      optimizationPriorities: [],
    }

    const fetchSpy = vi.spyOn(global, "fetch").mockImplementation(async () => ({
      ok: true,
      json: async () => ({ choices: [{ message: { content: JSON.stringify(validAnalysis) } }] }),
    }) as never)

    try {
      const result = await analyzerMod.analyzeTrajectoryByTask({
        taskId: "ses-fake-task-id",
        prisma: fakePrisma,
        promptPath: path.resolve(__dirname, "../prompts/session-trajectory-analyse.md"),
        outputDir: tmpDir,
        provider: fakeProvider,
      })

      expect(fetchSpy).toHaveBeenCalledTimes(1)
      expect(result.rounds).toBe(1)
      expect(result.outputPath).toContain("session-ses-fake-task-id-analysis.json")
      expect(fs.existsSync(result.outputPath)).toBe(true)
      expect(result.analysis).toMatchObject({ sessionSummary: "by-task fake" })
      vi.restoreAllMocks()
    } finally {
      vi.restoreAllMocks()
    }
  })
})
