// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This file is licensed under the CANN Open Software License Agreement Version 2.0.

// @vitest-environment happy-dom

/**
 * Trace → Turns keyword-highlight smoke test.
 *
 * Covers the "view turn" jump UX: the trace search keyword must follow the
 * user into the turns tab — LlmOutputView / ToolCallList auto-expand and
 * render the keyword highlighted (<mark>). No @testing-library/react, so we
 * use renderToStaticMarkup like the other render-smoke tests.
 */
import { describe, it, expect } from "vitest"
import { renderToStaticMarkup } from "react-dom/server"
import { createElement } from "react"
import { highlightKeyword } from "@/lib/shared/highlight"
import { LlmOutputView } from "@/components/observe/LlmOutputView"
import { ToolCallList } from "@/components/observe/ToolCallList"

describe("highlightKeyword (pure)", () => {
  it("returns the original text when keyword is empty", () => {
    expect(highlightKeyword("hello world", "")).toBe("hello world")
    expect(highlightKeyword("hello world", null)).toBe("hello world")
    expect(highlightKeyword("hello world", undefined)).toBe("hello world")
  })

  it("wraps matches in <mark> (case-insensitive)", () => {
    const out = renderToStaticMarkup(createElement("div", null, highlightKeyword("Find FOO foo here", "foo")))
    expect(out).toContain(">FOO</mark>")
    expect(out).toContain(">foo</mark>")
    expect(out).toContain("<mark")
    expect(out).toContain("Find")
    expect(out).toContain("here")
  })
})

describe("LlmOutputView highlight jump", () => {
  const baseProps = {
    content: "the quick brown fox jumps over the lazy dog",
    contentJson: null,
    contentSummary: null,
    outputTokens: 0,
    reasoningTokens: 0,
    role: "assistant",
  }

  it("auto-expands and renders the keyword highlighted when a content highlight is provided", () => {
    const html = renderToStaticMarkup(
      createElement(LlmOutputView, {
        ...baseProps,
        highlight: { keyword: "brown", matchField: "content" },
      })
    )
    // Content section is expanded (visible), and the keyword is marked.
    expect(html).toContain(">brown</mark>")
    expect(html).toContain("<mark")
    expect(html).toContain("quick")
    expect(html).toContain("fox")
  })

  it("does not highlight when no highlight is provided", () => {
    const html = renderToStaticMarkup(
      createElement(LlmOutputView, baseProps)
    )
    expect(html).not.toContain("<mark>")
  })
})

describe("ToolCallList highlight jump", () => {
  const toolCalls = [
    {
      id: "tc-1",
      toolCallId: "call_1",
      toolName: "Read",
      argsJson: '{"path":"/a/b.txt"}',
      resultJson: "the quick brown fox",
      state: "ok",
      errorType: null,
      errorMessage: null,
      durationMs: 10,
      isSkillRelated: false,
    },
    {
      id: "tc-2",
      toolCallId: "call_2",
      toolName: "Write",
      argsJson: '{"path":"/c.txt"}',
      resultJson: "no match here",
      state: "ok",
      errorType: null,
      errorMessage: null,
      durationMs: 5,
      isSkillRelated: false,
    },
  ]

  it("auto-expands the matching tool call and highlights the keyword in result", () => {
    const html = renderToStaticMarkup(
      createElement(ToolCallList, {
        toolCalls,
        highlight: { keyword: "brown", matchField: "toolResult", toolName: "Read" },
      })
    )
    expect(html).toContain(">brown</mark>")
    expect(html).toContain("<mark")
    // The matching tool (Read) is expanded — its args render too.
    expect(html).toContain("path")
  })

  it("does not highlight when no highlight is provided", () => {
    const html = renderToStaticMarkup(
      createElement(ToolCallList, { toolCalls })
    )
    expect(html).not.toContain("<mark>")
  })
})
