// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// @vitest-environment happy-dom

// LLM Input mirrors the WIRE: every message renders verbatim in its original
// position (reminder inside the user message, registry + skills inside the
// system message), and each piece of wire content appears EXACTLY ONCE. The
// only panels are wire top-level fields (System prompt, System tools), which
// are not part of any message. No extracted Memory/Skills panels — they were
// the duplication source. No dedupe hints or inner toggles.

import { describe, it, expect } from 'vitest';
import { createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react';
import { LlmContextView } from '@/components/observe/LlmContextView';

const SKILLS_TEXT = 'The following skills are available for use with the Skill tool:\n\n- cannbot-skill-review\n- dataviz: Use this skill whenever you are about to create ANY chart.';

const MEMORY_TEXT = '# claudeMd\nProject instructions for the fixture repo.';

const WIRE_USER_MSG = `<system-reminder>
As you answer the user's questions, you can use the following context:
${MEMORY_TEXT}

# currentDate
Today's date is 2026-08-14.
</system-reminder>

你用的什么模型`;

const WIRE_REGISTRY_MSG = `Available agent types for the Agent tool:
- claude: Catch-all for any task that doesn't match a more specific agent.

${SKILLS_TEXT}`;

function renderCollect(props: Record<string, unknown>): { html: () => string; container: HTMLElement } {
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => { root.render(createElement(LlmContextView, props as never)); });
  return { html: () => container.innerHTML, container };
}

function wireProps(overrides: Record<string, unknown> = {}) {
  return {
    inputMessagesJson: JSON.stringify([
      { role: 'user', content: WIRE_USER_MSG, tokenCount: 1800 },
      { role: 'system', content: WIRE_REGISTRY_MSG, tokenCount: 2000 },
    ]),
    inputMessagesCount: 2,
    inputMessagesTokens: 3800,
    contextWindowPct: 2.9,
    systemOverheadTokens: 0,
    systemPrompt: 'You are Claude Code.',
    fullContext: { tools: [], memoryFiles: MEMORY_TEXT, skills: SKILLS_TEXT },
    ...overrides,
  };
}

describe('LlmContextView wire-fidelity display', () => {
  // 消息体默认折叠（点击标题行展开）—— 内容断言前先展开目标消息
  function expandMessage(container: HTMLElement, tokenLabel: string) {
    const header = Array.from(container.querySelectorAll('[role="button"]'))
      .find(el => el.textContent?.includes(tokenLabel));
    expect(header, `header with ${tokenLabel} should exist`).toBeDefined();
    act(() => { header!.dispatchEvent(new MouseEvent('click', { bubbles: true })); });
  }

  it('user message renders the reminder as a separate sub-block above the prompt', () => {
    const { container } = renderCollect(wireProps());
    expandMessage(container, '1.8kt');
    const h = container.innerHTML;
    // sub-block has its own label and both parts remain present exactly once
    expect(h).toContain('system-reminder');
    expect(h.split('Project instructions for the fixture repo.').length - 1).toBe(1);
    expect(h.split('你用的什么模型').length - 1).toBe(1);
    // the reminder sub-block is visually separated (bordered block with its own header)
    const reminderBlock = Array.from(container.querySelectorAll('.border-purple-200'))
      .find(el => el.textContent?.includes('Project instructions'));
    expect(reminderBlock).toBeDefined();
    expect(reminderBlock!.textContent).toContain('system-reminder');
  });

  it('wire content appears EXACTLY ONCE, in its original message (no extracted panels)', () => {
    const { container } = renderCollect(wireProps());
    expandMessage(container, '1.8kt');
    expandMessage(container, '2.0kt');
    const h = container.innerHTML;
    // skills list: once, inside the expanded system message
    expect(h.split('The following skills are available').length - 1).toBe(1);
    // memory (claudeMd): once, inside the reminder of the user message
    expect(h.split('Project instructions for the fixture repo.').length - 1).toBe(1);
    // registry: once
    expect(h.split('Catch-all').length - 1).toBe(1);
    // real prompt: once
    expect(h.split('你用的什么模型').length - 1).toBe(1);
  });

  it('no Memory/Skills panels and no dedupe hint texts', () => {
    const { html } = renderCollect(wireProps());
    const h = html();
    expect(h).not.toContain('Memory files');
    expect(h).not.toContain('>Skills<');
    expect(h).not.toContain('与上方面板重复');
    expect(h).not.toContain('注入上下文');
    expect(h).not.toContain('skills 清单');
  });

  it('System prompt panel (wire top-level field) still renders, without duplicating messages', () => {
    const { container } = renderCollect(wireProps());
    const sysPanel = Array.from(container.querySelectorAll('[role="button"]'))
      .find(el => el.textContent === 'System6t · 20 chars▶' || (el.textContent?.includes('System') && el.textContent?.includes('20 chars')));
    expect(sysPanel).toBeDefined();
    act(() => { sysPanel!.dispatchEvent(new MouseEvent('click', { bubbles: true })); });
    // both occurrences are panel-internal (120-char preview line + <pre> body)
    expect(container.innerHTML.split('You are Claude Code.').length - 1).toBe(2);
  });

  it('message order matches the wire: user first, system second', () => {
    const { container } = renderCollect(wireProps());
    const badges = Array.from(container.querySelectorAll('[data-slot="badge"]'))
      .map(el => el.textContent);
    const roleBadges = badges.filter(t => t === 'user' || t === 'system' || t === 'assistant');
    expect(roleBadges).toEqual(['user', 'system']);
  });

  it('messages start collapsed; expanding reveals verbatim content, collapsing hides it', () => {
    const { container } = renderCollect(wireProps());
    const sysHeader = Array.from(container.querySelectorAll('[role="button"]'))
      .find(el => el.textContent?.includes('2.0kt'));
    // 默认折叠：内容不可见
    expect(container.innerHTML.split('The following skills are available').length - 1).toBe(0);
    act(() => { sysHeader!.dispatchEvent(new MouseEvent('click', { bubbles: true })); });
    expect(container.innerHTML.split('The following skills are available').length - 1).toBe(1);
    act(() => { sysHeader!.dispatchEvent(new MouseEvent('click', { bubbles: true })); });
    expect(container.innerHTML.split('The following skills are available').length - 1).toBe(0);
  });
});
