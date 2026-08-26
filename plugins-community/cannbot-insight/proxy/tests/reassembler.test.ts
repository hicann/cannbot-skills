// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import { createReassembler } from '../src/stream-reassembler.ts';

function sse(events: object[]): string {
  return events.map(e => `data: ${JSON.stringify(e)}\n\n`).join('') + 'data: [DONE]\n\n';
}

describe('stream-reassembler: Anthropic', () => {
  it('accumulates text deltas into a text block', () => {
    const r = createReassembler('anthropic');
    r.feed(sse([
      { type: 'message_start', message: { model: 'claude-opus-x', usage: { input_tokens: 10, cache_read_input_tokens: 2 } } },
      { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Hello ' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'world' } },
      { type: 'content_block_stop', index: 0 },
      { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 3 } },
      { type: 'message_stop' },
    ]));
    const res = r.result();
    expect(res.model).toBe('claude-opus-x');
    expect(res.stop_reason).toBe('end_turn');
    expect(res.content.length).toBe(1);
    expect(res.content[0].text).toBe('Hello world');
    expect(res.usage?.input_tokens).toBe(10);
    expect(res.usage?.cache_read_input_tokens).toBe(2);
    expect(res.usage?.output_tokens).toBe(3);
  });

  it('reassembles tool_use input_json_delta into parsed input', () => {
    const r = createReassembler('anthropic');
    r.feed(sse([
      { type: 'content_block_start', index: 0, content_block: { type: 'tool_use', id: 't1', name: 'Read' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '{"file_path":"/x' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '.ts"}' } },
      { type: 'content_block_stop', index: 0 },
      { type: 'message_stop' },
    ]));
    const res = r.result();
    expect(res.content.length).toBe(1);
    expect(res.content[0].type).toBe('tool_use');
    expect(res.content[0].name).toBe('Read');
    expect(res.content[0].input).toEqual({ file_path: '/x.ts' });
  });
});

describe('stream-reassembler: OpenAI', () => {
  it('accumulates content deltas and tool_calls', () => {
    const r = createReassembler('openai');
    r.feed(sse([
      { model: 'gpt-4-x', choices: [{ delta: { role: 'assistant', content: 'Hi ' }, finish_reason: null }] },
      { choices: [{ delta: { content: 'there' }, finish_reason: null }] },
      { choices: [{ delta: { tool_calls: [{ index: 0, id: 'call_1', function: { name: 'read', arguments: '{"p' } }] }, finish_reason: null }] },
      { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: 'ath":42}' } }] }, finish_reason: null }] },
      { choices: [{ delta: {}, finish_reason: 'tool_calls' }] },
      { usage: { prompt_tokens: 8, completion_tokens: 4, total_tokens: 12, prompt_tokens_details: { cached_tokens: 3 } } },
    ]));
    const res = r.result();
    expect(res.model).toBe('gpt-4-x');
    expect(res.stop_reason).toBe('tool_calls');
    const text = res.content.find(c => c.type === 'text');
    expect(text?.text).toBe('Hi there');
    const tc = res.content.find(c => c.type === 'tool_use');
    expect(tc?.id).toBe('call_1');
    expect(tc?.name).toBe('read');
    expect(tc?.input).toEqual({ path: 42 });
    expect(res.usage?.input_tokens).toBe(8);
    expect(res.usage?.output_tokens).toBe(4);
    expect(res.usage?.cache_read_input_tokens).toBe(3);
  });

  it('preserves the tool_call id when later chunks repeat id:"" (dashscope streaming)', () => {
    // dashscope streams a tool_call across many chunks: the FIRST carries the
    // real id+name, subsequent chunks carry only arguments and repeat id:"".
    // The reassembler must NOT let the empty id overwrite the real one — else
    // tool_use.id="" can't pair with the tool_result's tool_use_id and the result
    // (e.g. a subagent's read of agents/<name>.md) is dropped during ingest.
    const r = createReassembler('openai');
    r.feed(sse([
      { choices: [{ delta: { tool_calls: [{ index: 0, id: 'call_abc', type: 'function', function: { name: 'read', arguments: '{"' } }] }, finish_reason: null }] },
      { choices: [{ delta: { tool_calls: [{ index: 0, id: '', type: 'function', function: { arguments: 'path":"foo.txt"}' } }] }, finish_reason: null }] },
      { choices: [{ delta: {}, finish_reason: 'tool_calls' }] },
    ]));
    const tc = r.result().content.find(c => c.type === 'tool_use');
    expect(tc?.id).toBe('call_abc');
    expect(tc?.name).toBe('read');
    expect(tc?.input).toEqual({ path: 'foo.txt' });
  });
});

describe('stream-reassembler: ttft (firstTokenAt)', () => {
  it('anthropic: firstTokenAt set on first content_block_delta, null without deltas', () => {
    const r = createReassembler('anthropic');
    r.feed(sse([
      { type: 'message_start', message: { model: 'm', usage: { input_tokens: 5 } } },
      { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'H' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'i' } },
      { type: 'content_block_stop', index: 0 },
      { type: 'message_stop' },
    ]));
    expect(r.result().firstTokenAt).toBeTypeOf('number');
    // message with no content deltas → null
    const r2 = createReassembler('anthropic');
    r2.feed(sse([{ type: 'message_start', message: { model: 'm' } }, { type: 'message_stop' }]));
    expect(r2.result().firstTokenAt).toBeNull();
  });

  it('openai: firstTokenAt set on first delta.content', () => {
    const r = createReassembler('openai');
    r.feed(sse([
      { choices: [{ delta: { content: 'H' }, finish_reason: null }] },
      { choices: [{ delta: { content: 'i' }, finish_reason: null }] },
      { choices: [{ delta: {}, finish_reason: 'stop' }] },
    ]));
    expect(r.result().firstTokenAt).toBeTypeOf('number');
    // no content → null
    const r2 = createReassembler('openai');
    r2.feed(sse([{ choices: [{ delta: {}, finish_reason: 'stop' }] }]));
    expect(r2.result().firstTokenAt).toBeNull();
  });
});
