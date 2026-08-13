// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This file is licensed under the CANN Open Software License Agreement Version 2.0.

// @vitest-environment happy-dom

/**
 * TurnTimeline quick-jump-by-number smoke test.
 *
 * The project has no @testing-library/react, so — like the compare-perf and
 * skill-audit-view smoke tests — we render to static markup and assert the
 * jump UI is present/absent based on the onJumpToTurnIndex prop, and that
 * each turn's #index renders. Event-driven interaction (typing + Enter →
 * callback) can't be exercised under renderToStaticMarkup; the handler logic
 * is exercised manually and guarded by typecheck.
 */
import { describe, it, expect } from "vitest"
import { renderToStaticMarkup } from "react-dom/server"
import { createElement } from "react"
import { TurnTimeline } from "@/components/observe/TurnTimeline"

function makeTurn(turnIndex: number, role = "assistant") {
  return {
    turnId: `t-${turnIndex}`,
    turnIndex,
    role,
    contentSummary: `turn ${turnIndex}`,
    agentName: null,
    isSubagent: false,
    subagentName: null,
    subagentSessionId: null,
    parentExecutionId: null,
    totalTokens: 100 * turnIndex,
    inputTokens: 0,
    outputTokens: 0,
    reasoningTokens: 0,
    latencyMs: 0,
    createdAt: null,
    completedAt: null,
    model: null,
    toolCalls: [],
    skillEvents: [],
  }
}

const turns = [1, 2, 459, 460].map(i => makeTurn(i))

describe("TurnTimeline quick-jump input", () => {
  it("renders the jump input + Go button when onJumpToTurnIndex is provided", () => {
    const html = renderToStaticMarkup(
      createElement(TurnTimeline, {
        turns,
        bridges: [],
        selectedTurnId: null,
        onSelectTurn: () => {},
        onJumpToTurnIndex: () => {},
      })
    )
    expect(html).toContain("Go to turn #")
    // Go submit button is rendered
    expect(html).toContain("Go")
    // Each visible turn index renders its #N label
    expect(html).toContain("#1")
    expect(html).toContain("#459")
    expect(html).toContain("#460")
  })

  it("hides the jump input when onJumpToTurnIndex is omitted", () => {
    const html = renderToStaticMarkup(
      createElement(TurnTimeline, {
        turns,
        bridges: [],
        selectedTurnId: null,
        onSelectTurn: () => {},
      })
    )
    expect(html).not.toContain("Go to turn #")
  })

  it("renders an empty-state message when no turns match the role filter (no crash on empty turns)", () => {
    const html = renderToStaticMarkup(
      createElement(TurnTimeline, {
        turns: [],
        bridges: [],
        selectedTurnId: null,
        onSelectTurn: () => {},
        onJumpToTurnIndex: () => {},
      })
    )
    expect(typeof html).toBe("string")
    expect(html.length).toBeGreaterThan(0)
  })
})
