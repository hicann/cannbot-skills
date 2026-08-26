// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// @vitest-environment happy-dom

/**
 * Wire 输入 / LLM 输出可读渲染回归测试。
 *
 * 块数组消息（[{type:"text",text:...,cache_control:...}]）此前被
 * JSON.stringify 整坨直出（转义 \n、\"、cache_control 全部可见）；模型
 * 实际读到的是各 block 的文本内容。显示层应按 block 展开渲染：文本可读、
 * 协议标记（cache_control）消失、tool_use/tool_result 折叠为标记行。
 */
import { describe, it, expect } from "vitest"
import { renderToStaticMarkup } from "react-dom/server"
import { createElement } from "react"
import { TurnDetail } from "@/components/observe/TurnDetail"
import { LlmOutputView } from "@/components/observe/LlmOutputView"
import { BashEscapeView, parseBashEscape } from "@/components/observe/BashEscapeView"

const wireInputContentJson = JSON.stringify({
  wireInput: true,
  messages: [
    {
      role: "user",
      content: [
        { type: "text", text: "Available agent types for the Agent tool:\n- claude: Catch-all for any task.", cache_control: { type: "ephemeral" } },
      ],
    },
    {
      role: "assistant",
      content: [{ type: "tool_use", id: "tu_1", name: "Bash", input: { command: "ls" } }],
    },
    {
      role: "user",
      content: [{ type: "tool_result", tool_use_id: "tu_1", content: "file-a\nfile-b" }],
    },
  ],
})

const baseTurn = {
  turnId: "t-wire-1",
  turnIndex: 1,
  role: "user",
  content: null,
  contentSummary: null,
  inputMessagesJson: null,
  inputMessagesCount: 0,
  inputMessagesTokens: 0,
  contextWindowPct: null,
  agentName: null,
  subagentName: null,
  isSubagent: false,
  totalTokens: 0,
  inputTokens: 0,
  outputTokens: 0,
  reasoningTokens: 0,
  cacheReadTokens: 0,
  cacheWriteTokens: 0,
  latencyMs: 0,
  ttftMs: null,
  createdAt: "2026-08-18T03:31:02.367Z",
  completedAt: null,
  model: null,
  modelId: null,
  providerId: null,
  finishReason: null,
  toolCalls: [],
  skillEvents: [],
}

describe("TurnDetail wire 输入可读渲染", () => {
  it("块数组按 block 展开：文本可读、转义 JSON 与 cache_control 消失", () => {
    const html = renderToStaticMarkup(
      createElement(TurnDetail, { turn: { ...baseTurn, contentJson: wireInputContentJson } })
    )
    expect(html).toContain("Wire 输入")
    expect(html).toContain("Available agent types for the Agent tool:")
    expect(html).toContain("\n- claude: Catch-all")
    expect(html).not.toContain("\\n- claude")
    expect(html).not.toContain("cache_control")
    expect(html).not.toContain("\"type\":\"text\"")
  })

  it("tool_use / tool_result 渲染为折叠标记行", () => {
    const html = renderToStaticMarkup(
      createElement(TurnDetail, { turn: { ...baseTurn, contentJson: wireInputContentJson } })
    )
    expect(html).toContain("[tool_use Bash]")
    expect(html).toContain("[tool_result]")
    expect(html).toContain("file-a")
  })

  it("字符串 content 的 wire 消息保持原样", () => {
    const html = renderToStaticMarkup(
      createElement(TurnDetail, {
        turn: {
          ...baseTurn,
          contentJson: JSON.stringify({ wireInput: true, messages: [{ role: "user", content: "你用的什么模型？" }] }),
        },
      })
    )
    expect(html).toContain("你用的什么模型？")
  })
})

describe("shell 转义显示糖", () => {
  const wireBashJson = JSON.stringify({
    wireInput: true,
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "<bash-input>data</bash-input>" },
          { type: "text", text: "<bash-stdout></bash-stdout><bash-stderr>/bin/bash: line 1: data: command not found\n</bash-stderr>" },
        ],
      },
    ],
  })

  it("wire 消息整体为 shell 转义时渲染为终端样式，伪标签消失", () => {
    const html = renderToStaticMarkup(
      createElement(TurnDetail, { turn: { ...baseTurn, contentJson: wireBashJson } })
    )
    expect(html).toContain("shell")
    expect(html).toContain("$")
    expect(html).toContain("data")
    expect(html).toContain("command not found")
    expect(html).not.toContain("bash-input")
    expect(html).not.toContain("bash-stdout")
  })

  it("parseBashEscape 只匹配整条消息，混入其他文本回退普通渲染", () => {
    expect(parseBashEscape("<bash-input>data</bash-input>\n\n<bash-stdout>ok</bash-stdout>")).toEqual({
      command: "data",
      stdout: "ok",
      stderr: "",
    })
    expect(parseBashEscape("前置说明\n<bash-input>data</bash-input>")).toBeNull()
    expect(parseBashEscape("普通文本")).toBeNull()
  })

  it("BashEscapeView 渲染 stdout 与 stderr", () => {
    const html = renderToStaticMarkup(
      createElement(BashEscapeView, { esc: { command: "npm run test", stdout: "ok 5 passed", stderr: "" } })
    )
    expect(html).toContain("npm run test")
    expect(html).toContain("ok 5 passed")
  })
})

describe("LlmOutputView anthropic 块字段", () => {
  it("text 块读 .text 字段（此前误读 .content 落到 [object Object]）", () => {
    const html = renderToStaticMarkup(
      createElement(LlmOutputView, {
        content: null,
        contentJson: JSON.stringify([{ type: "text", text: "第一行\n第二行：回答内容" }]),
        contentSummary: null,
        outputTokens: 0,
        reasoningTokens: 0,
        role: "assistant",
        highlight: { keyword: "回答内容", matchField: "content" },
      })
    )
    expect(html).toContain("第一行")
    expect(html).toContain("回答内容")
    expect(html).not.toContain("[object Object]")
  })

  it("thinking 块读 .thinking 字段", () => {
    const html = renderToStaticMarkup(
      createElement(LlmOutputView, {
        content: null,
        contentJson: JSON.stringify([{ type: "thinking", thinking: "用户在问模型身份，需要简短回答。" }]),
        contentSummary: null,
        outputTokens: 0,
        reasoningTokens: 0,
        role: "assistant",
        highlight: { keyword: "模型身份", matchField: "content" },
      })
    )
    expect(html).toContain("用户在问")
    expect(html).toContain("需要简短回答")
    expect(html).not.toContain("[object Object]")
  })
})
