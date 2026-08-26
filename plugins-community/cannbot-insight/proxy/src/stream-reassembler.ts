// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import type { AnthropicContentBlock, AnthropicUsage, Protocol } from './types';

export interface ReassembledResponse {
  model: string | null;
  stop_reason: string | null;
  content: AnthropicContentBlock[];
  usage: AnthropicUsage | null;
  // Absolute ms (Date.now()) of the first streamed content token arriving.
  // Null for non-stream responses. Server computes ttftMs = firstTokenAt − receivedAt.
  firstTokenAt?: number | null;
}

export interface Reassembler {
  feed(chunk: string): void;
  result(): ReassembledResponse;
}

export function createReassembler(protocol: Protocol): Reassembler {
  return protocol === 'anthropic' ? new AnthropicReassembler() : new OpenAIReassembler();
}

interface AnthropicBlockState {
  type: string;
  text?: string;
  thinking?: string;
  id?: string;
  name?: string;
  inputPartial?: string;
  input?: Record<string, unknown>;
  tool_use_id?: string;
  content?: string | Record<string, unknown> | unknown[];
}

class AnthropicReassembler implements Reassembler {
  private blocks = new Map<number, AnthropicBlockState>();
  private model: string | null = null;
  private stopReason: string | null = null;
  private usage: AnthropicUsage = {};
  private buffer = '';
  private firstTokenAt: number | null = null;

  feed(chunk: string): void {
    this.buffer += chunk;
    let idx: number;
    while ((idx = this.buffer.indexOf('\n')) >= 0) {
      const line = this.buffer.slice(0, idx).replace(/\r$/, '');
      this.buffer = this.buffer.slice(idx + 1);
      this.processLine(line);
    }
  }

  private processLine(line: string): void {
    if (!line.startsWith('data:')) return;
    const data = line.slice(5).trim();
    if (!data || data === '[DONE]') return;
    let event: Record<string, unknown>;
    try {
      event = JSON.parse(data);
    } catch {
      return;
    }
    this.handleEvent(event);
  }

  private handleEvent(e: Record<string, unknown>): void {
    switch (e.type as string) {
      case 'message_start': {
        const msg = (e as { message?: Record<string, unknown> }).message;
        if (msg) {
          this.model = (msg.model as string) ?? this.model;
          const u = msg.usage as AnthropicUsage | undefined;
          if (u) {
            this.usage.input_tokens = u.input_tokens;
            this.usage.cache_read_input_tokens = u.cache_read_input_tokens;
            this.usage.cache_creation_input_tokens = u.cache_creation_input_tokens;
          }
        }
        break;
      }
      case 'content_block_start': {
        const start = e as { index: number; content_block: Record<string, unknown> };
        const cb = start.content_block ?? {};
        this.blocks.set(start.index, {
          type: (cb.type as string) ?? 'text',
          text: typeof cb.text === 'string' ? cb.text : '',
          thinking: typeof cb.thinking === 'string' ? cb.thinking : '',
          id: cb.id as string | undefined,
          name: cb.name as string | undefined,
          inputPartial: '',
          tool_use_id: cb.tool_use_id as string | undefined,
        });
        break;
      }
      case 'content_block_delta': {
        if (this.firstTokenAt === null) this.firstTokenAt = Date.now();
        const d = e as { index: number; delta: Record<string, unknown> };
        const block = this.blocks.get(d.index);
        if (!block) return;
        const delta = d.delta ?? {};
        const dtype = delta.type as string | undefined;
        if (dtype === 'text_delta' && typeof delta.text === 'string') {
          block.text = (block.text ?? '') + delta.text;
        } else if (dtype === 'thinking_delta' && typeof delta.thinking === 'string') {
          block.thinking = (block.thinking ?? '') + delta.thinking;
        } else if (dtype === 'input_json_delta' && typeof delta.partial_json === 'string') {
          block.inputPartial = (block.inputPartial ?? '') + delta.partial_json;
        }
        break;
      }
      case 'content_block_stop': {
        const block = this.blocks.get((e as { index: number }).index);
        if (block && block.inputPartial !== undefined && block.inputPartial !== '') {
          try {
            block.input = JSON.parse(block.inputPartial);
          } catch {
            block.input = { _raw: block.inputPartial };
          }
          delete block.inputPartial;
        }
        break;
      }
      case 'message_delta': {
        const delta = (e as { delta?: Record<string, unknown> }).delta;
        if (delta?.stop_reason) this.stopReason = delta.stop_reason as string;
        const u = (e as { usage?: AnthropicUsage }).usage;
        if (u && typeof u.output_tokens === 'number') this.usage.output_tokens = u.output_tokens;
        break;
      }
      case 'message_stop':
        break;
    }
  }

  result(): ReassembledResponse {
    const sorted = [...this.blocks.entries()].sort((a, b) => a[0] - b[0]);
    const content: AnthropicContentBlock[] = sorted.map(([, b]) => {
      const out: AnthropicContentBlock = { type: b.type };
      if (b.text) out.text = b.text;
      if (b.thinking) out.thinking = b.thinking;
      if (b.id) out.id = b.id;
      if (b.name) out.name = b.name;
      if (b.input !== undefined) out.input = b.input;
      if (b.tool_use_id) out.tool_use_id = b.tool_use_id;
      return out;
    });
    return { model: this.model, stop_reason: this.stopReason, content, usage: this.usage, firstTokenAt: this.firstTokenAt };
  }
}

class OpenAIReassembler implements Reassembler {
  private model: string | null = null;
  private stopReason: string | null = null;
  private text = '';
  private toolCalls: { id: string; name: string; args: string }[] = [];
  private usage: AnthropicUsage | null = null;
  private buffer = '';
  private firstTokenAt: number | null = null;

  feed(chunk: string): void {
    this.buffer += chunk;
    let idx: number;
    while ((idx = this.buffer.indexOf('\n')) >= 0) {
      const line = this.buffer.slice(0, idx).replace(/\r$/, '');
      this.buffer = this.buffer.slice(idx + 1);
      this.processLine(line);
    }
  }

  private processLine(line: string): void {
    if (!line.startsWith('data:')) return;
    const data = line.slice(5).trim();
    if (!data || data === '[DONE]') return;
    let event: Record<string, unknown>;
    try {
      event = JSON.parse(data);
    } catch {
      return;
    }
    if (typeof event.model === 'string') this.model = event.model;
    const choices = event.choices as Array<Record<string, unknown>> | undefined;
    if (choices && choices.length > 0) {
      const choice = choices[0];
      if (choice.finish_reason) this.stopReason = choice.finish_reason as string;
      const delta = choice.delta as Record<string, unknown> | undefined;
      if (delta) {
        if (delta.content || delta.tool_calls) {
          if (this.firstTokenAt === null) this.firstTokenAt = Date.now();
        }
        if (typeof delta.content === 'string') this.text += delta.content;
        const tcs = delta.tool_calls as Array<Record<string, unknown>> | undefined;
        if (tcs) {
          for (const tc of tcs) {
            const slot = typeof tc.index === 'number' ? (tc.index as number) : this.toolCalls.length;
            while (this.toolCalls.length <= slot) this.toolCalls.push({ id: '', name: '', args: '' });
            const fn = tc.function as Record<string, unknown> | undefined;
            if (fn) {
              // truthy checks: dashscope streams tool_calls across many chunks —
              // the FIRST carries id+name, subsequent chunks carry only arguments
              // and repeat id:"" / omit name. `typeof === 'string'` would let the
              // empty-string id overwrite the real one → tool_use.id="" → can't pair
              // with the tool_result's tool_use_id → resultJson null → tool results
              // (e.g. a subagent's read of agents/<name>.md) dropped during ingest.
              if (fn.name) this.toolCalls[slot].name = fn.name as string;
              if (typeof fn.arguments === 'string') this.toolCalls[slot].args += fn.arguments;
            }
            if (tc.id) this.toolCalls[slot].id = tc.id as string;
          }
        }
      }
    }
    const usage = event.usage as Record<string, unknown> | undefined;
    if (usage) {
      const mapped: AnthropicUsage = {
        input_tokens: typeof usage.prompt_tokens === 'number' ? usage.prompt_tokens : undefined,
        output_tokens: typeof usage.completion_tokens === 'number' ? usage.completion_tokens : undefined,
      };
      const details = usage.prompt_tokens_details as { cached_tokens?: number } | undefined;
      if (details && typeof details.cached_tokens === 'number') {
        mapped.cache_read_input_tokens = details.cached_tokens;
      }
      this.usage = mapped;
    }
  }

  result(): ReassembledResponse {
    const content: AnthropicContentBlock[] = [];
    if (this.text) content.push({ type: 'text', text: this.text });
    for (const tc of this.toolCalls) {
      let input: Record<string, unknown> = {};
      if (tc.args) {
        try {
          input = JSON.parse(tc.args);
        } catch {
          input = { _raw: tc.args };
        }
      }
      content.push({ type: 'tool_use', id: tc.id, name: tc.name, input });
    }
    return { model: this.model, stop_reason: this.stopReason, content, usage: this.usage, firstTokenAt: this.firstTokenAt };
  }
}
