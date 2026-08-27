// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 落盘清洗测试：结构层（敏感键名全等命中）+ 字符串层（厂家键特征 / Bearer /
// env 回显 / 内嵌 JSON / URL query）。重点双面：
// ①各厂家键形态全部打码；②token 计数类字段、tool 入参的裸 key、普通文本零误伤。
// 最后走 redactRecord + claude-emitter 的完整数据流，断言落盘 jsonl 无任何明文键。

import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { maskSecret, redactString, redactInPlace, redactRecord } from '../src/redactor.ts';
import { emit as claudeEmit } from '../src/claude-emitter.ts';
import { sessionFilePath } from '../src/writer.ts';
import type { ProxyRecord } from '../src/types.ts';

const ANTHROPIC_KEY = 'sk-ant-api03-AbCdEf1234567890aBcDeF';
const OPENAI_KEY = 'sk-proj-AbCdEf1234567890aBcDeF1234567890';
const GEMINI_KEY = 'AIzaSyA1234567890abcdefghijklmnopqrs';

describe('maskSecret', () => {
  it('长键保留首尾 4 位，短键全打码', () => {
    expect(maskSecret('abcdefghijklmnop')).toBe('abcd…mnop');
    expect(maskSecret('short')).toBe('****');
    expect(maskSecret('12345678')).toBe('****');
  });
});

describe('结构层：敏感键名全等命中', () => {
  it('api_key/apiKey/api-key/authorization/x-api-key/x-goog-api-key/cookie 全变体打码', () => {
    const body = {
      api_key: ANTHROPIC_KEY,
      apiKey: ANTHROPIC_KEY,
      'api-key': ANTHROPIC_KEY,
      authorization: `Bearer ${OPENAI_KEY}`,
      'x-api-key': ANTHROPIC_KEY,
      'X-Goog-Api-Key': GEMINI_KEY,
      cookie: 'session=abc; token=xyz1234567890',
      nested: [{ access_token: 'ghp_AbCdEf1234567890123', deep: { client_secret: 'AbCdEf1234567890abcd' } }],
    };
    redactInPlace(body);
    const flat = JSON.stringify(body);
    expect(flat).not.toContain(ANTHROPIC_KEY);
    expect(flat).not.toContain(OPENAI_KEY);
    expect(flat).not.toContain(GEMINI_KEY);
    expect(body.api_key).toBe(maskSecret(ANTHROPIC_KEY));
    expect(body.authorization).toBe(`Bearer ${maskSecret(OPENAI_KEY)}`);
  });

  it('token 计数类字段与 tool 入参的裸 key 不误伤', () => {
    const body = {
      model: 'glm-4.7',
      max_tokens: 4096,
      usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
      messages: [{ role: 'user', content: 'normal text', tool_use_id: 'tu_123', token: 'keepme-not-a-secret' }],
      key: 'a-normal-map-key',
    };
    const before = JSON.stringify(body);
    expect(redactInPlace(body)).toBe(false);
    expect(JSON.stringify(body)).toBe(before);
  });
});

describe('字符串层：内容回显打码', () => {
  it('Bearer 凭据：保留 scheme 只打码凭据', () => {
    expect(redactString(`curl -H "Authorization: Bearer ${OPENAI_KEY}" https://api.example.com`))
      .toBe(`curl -H "Authorization: Bearer ${maskSecret(OPENAI_KEY)}" https://api.example.com`);
  });

  it('env 赋值回显：ANTHROPIC_API_KEY=… / "DASHSCOPE_API_KEY":"…"', () => {
    expect(redactString(`ANTHROPIC_API_KEY=${ANTHROPIC_KEY}`)).toBe(`ANTHROPIC_API_KEY=${maskSecret(ANTHROPIC_KEY)}`);
    expect(redactString(`"DASHSCOPE_API_KEY": "sk-AbCdEf1234567890aBcDeF12345678"`))
      .toBe(`"DASHSCOPE_API_KEY": "${maskSecret('sk-AbCdEf1234567890aBcDeF12345678')}"`);
    // JSON 键名收尾引号在冒号前（KEY":"value"），且值无 sk- 前缀（bigmodel
    // 形态 hex.xxx）——曾因此漏清洗 cpx 启动行里的 --settings
    const bigmodel = 'e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2e2.FakeFakeFakeFake';
    expect(redactString(`--settings {"env":{"ANTHROPIC_BASE_URL":"http://127.0.0.1:39755","ANTHROPIC_API_KEY":"${bigmodel}","ANTHROPIC_MODEL":"glm-5.2"}} -p hi`))
      .toBe(`--settings {"env":{"ANTHROPIC_BASE_URL":"http://127.0.0.1:39755","ANTHROPIC_API_KEY":"${maskSecret(bigmodel)}","ANTHROPIC_MODEL":"glm-5.2"}} -p hi`);
  });

  it('内嵌 JSON 字段：cat settings.json 的输出', () => {
    const out = redactString(`{"api_key": "${GEMINI_KEY}", "model": "gemini"}`);
    expect(out).not.toContain(GEMINI_KEY);
    expect(out).toContain('"model": "gemini"');
  });

  it('URL query 键：?key= / &api-key=', () => {
    const out = redactString(`GET https://gen.googleapis.com/v1/models?key=${GEMINI_KEY}&alt=json`);
    expect(out).not.toContain(GEMINI_KEY);
    expect(out).toContain('&alt=json');
    expect(redactString('/v1/chat/completions?api-key=AbCdEf1234567890abcd')).not.toContain('AbCdEf1234567890abcd');
  });

  it('厂家前缀键：sk-ant-/sk-or-/sk-proj-/gsk_/AIza/xai- 单独出现在正文', () => {
    for (const key of [ANTHROPIC_KEY, 'sk-or-v1-AbCdEf1234567890aBcDeF', OPENAI_KEY, 'gsk_AbCdEf1234567890abcdef1234', GEMINI_KEY, 'xai-AbCdEf1234567890aBcDeF']) {
      expect(redactString(`模型回复里提到了 ${key} 请注意`)).not.toContain(key);
    }
  });

  it('国内厂家：dashscope LTAI / bigmodel 32hex.16 形态', () => {
    expect(redactString('LTAIAbCdEf1234567890aBcD')).not.toContain('LTAIAbCdEf1234567890aBcD');
    expect(redactString('key 是 e87a3a4a564e42fe836f3b45cc7921d2.nGzpciug0GMprKEM 请轮换')).not.toContain('nGzpciug0GMprKEM');
  });

  it('泛 sk- 前缀（DeepSeek/Moonshot 32hex 类）', () => {
    const deepseek = 'sk-AbCdEf1234567890aBcDeF12345678';
    expect(redactString(`key: ${deepseek}`)).not.toContain(deepseek);
  });

  it('普通文本零误伤', () => {
    const clean = [
      '请帮我分析 task-1234567890123456789012345678 这个任务',
      'max_tokens=4096, temperature=0.7',
      'GL-4 是一个模型编号',
      'sk-short 这个太短不像键',
      '{"key": "normal-value", "model": "claude-sonnet-4"}',
      'Bearer 票据这个词本身没有凭据',
    ];
    for (const text of clean) {
      expect(redactString(text)).toBe(text);
    }
  });
});

describe('redactRecord + 落盘数据流', () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'cpx-redact-'));
  process.env.CANNBOT_PROXY_DIR = tmpDir;
  const sid = 'redact-flow-it-1';
  const mainFile = sessionFilePath(sid);

  it('record 全表面清洗后经 emitter 落盘，jsonl 无任何明文键', () => {
    const rec: ProxyRecord = {
      sid,
      protocol: 'anthropic',
      receivedAt: Date.now(),
      completedAt: Date.now(),
      latencyMs: 100,
      ttftMs: null,
      request: {
        path: `/v1/messages?key=${GEMINI_KEY}`,
        model: 'glm-4.7',
        body: {
          model: 'glm-4.7',
          max_tokens: 1024,
          system: [{ type: 'text', text: `You are Claude. cc_version=2.1.234.467;` }],
          messages: [
            { role: 'user', content: '帮我看看环境变量' },
            { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'tu_1', content: `ANTHROPIC_API_KEY=${ANTHROPIC_KEY}\nOPENAI_API_KEY="${OPENAI_KEY}"` }] },
          ],
        },
      },
      response: {
        status: 200,
        model: 'glm-4.7',
        stop_reason: 'end_turn',
        content: [{ type: 'text', text: `你的配置里有 ${ANTHROPIC_KEY}，建议轮换` }],
        usage: { input_tokens: 10, output_tokens: 5 },
      },
      xSessionId: null,
      parentSessionId: null,
      userAgent: 'claude-cli/2.1.234',
    };
    redactRecord(rec);
    claudeEmit(rec);

    const lines = fs.readFileSync(mainFile, 'utf-8').split('\n').filter(l => l.trim()).map(l => JSON.parse(l));
    expect(lines.length).toBeGreaterThan(0);
    const raw = JSON.stringify(lines);
    expect(raw).not.toContain(ANTHROPIC_KEY);
    expect(raw).not.toContain(OPENAI_KEY);
    expect(raw).not.toContain(GEMINI_KEY);
    // 清洗不影响正常字段：版本号、usage、system 提示原文保留
    expect(raw).toContain('cc_version=2.1.234.467');
    expect(raw).toContain('"input_tokens":10');
    expect(raw).toContain('You are Claude.');
    // 打码痕迹可辨（排障可判断哪家键被清洗）
    expect(raw).toContain(maskSecret(ANTHROPIC_KEY));
  });

  it('干净 record 清洗后内容不变', () => {
    const rec: ProxyRecord = {
      sid: 'redact-clean-1',
      protocol: 'anthropic',
      receivedAt: 1, completedAt: 2, latencyMs: 1, ttftMs: null,
      request: { path: '/v1/messages', model: 'm', body: { model: 'm', max_tokens: 8, messages: [{ role: 'user', content: '你好' }] } },
      response: { status: 200, model: 'm', stop_reason: 'end_turn', content: [{ type: 'text', text: '在' }], usage: { input_tokens: 1, output_tokens: 1 } },
    };
    const before = JSON.stringify(rec);
    redactRecord(rec);
    expect(JSON.stringify(rec)).toBe(before);
  });
});
