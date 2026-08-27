// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// cannbay2 上传治理：唯一强制关口，所有待传文件一律视为不可信输入。
// redactor 核心自 proxy/src/redactor.ts 移植（该模块零内部依赖，两份实现由
// tests/cannbay2-governance.test.ts 的 parity 用例锁定字节一致，漂移即失败）。
// 之下的治理层（redactLine/scanResidue/governText）为 insight 侧独有：
//   ① 清洗：jsonl 逐行 parse → 结构层键名 + 字符串层厂家特征打码（写副本，不改原件）
//   ② 复检：对清洗产物独立重扫，任何残留命中 → 熔断拒绝上传（fail-closed）
export function maskSecret(v: string): string {
  if (v.length <= 8) return '****';
  return `${v.slice(0, 4)}…${v.slice(-4)}`;
}

// 结构层键名名单：全等比较（大小写不敏感），snake/kebab/camel 变体逐个列入。
// 刻意不含裸 "key"/"token"——tool 入参里大量合法字段（max_tokens、
// input_tokens、tool_use_id 等以精确匹配保证零误伤）。
const SENSITIVE_KEYS = new Set([
  'authorization', 'proxy-authorization',
  'x-api-key', 'api-key', 'apikey',
  'api_key', 'apiKey',
  'x-goog-api-key', 'goog-api-key', 'x-goog-apikey',
  'cookie', 'set-cookie',
  'access_token', 'access-token', 'accesstoken', 'accessToken',
  'refresh_token', 'refresh-token', 'refreshToken',
  'client_secret', 'client-secret', 'clientSecret',
  'secret_key', 'secret-key', 'secretkey', 'secretKey',
  'password', 'passwd',
]);

function isSensitiveKey(key: string): boolean {
  return SENSITIVE_KEYS.has(key.toLowerCase());
}

// authorization 值形如 "Bearer <cred>"：保留 scheme 只打码凭据部分。
function maskAuthValue(v: string): string {
  const m = v.match(/^(Bearer|Basic|Token)\s+(\S+)$/i);
  if (m) return `${m[1]} ${maskSecret(m[2])}`;
  return maskSecret(v);
}

// 字符串层正则组：按厂家键特征 + 常见回显形态打码。全部带长度阈值，
// 未命中的普通文本保持原引用（性能：大 body 只在命中处新建字符串）。
const STRING_RULES: Array<{ re: RegExp; build: (m: RegExpMatchArray) => string }> = [
  {
    // Bearer/Basic 凭据（配置 dump、错误消息、curl 回显）
    re: /\b(Bearer|Basic)\s+([A-Za-z0-9._~+/=-]{16,})/g,
    build: m => `${m[1]} ${maskSecret(m[2])}`,
  },
  {
    // env 赋值回显：ANTHROPIC_API_KEY=xxx / "DASHSCOPE_API_KEY":"xxx"。
    // JSON 形态的键名收尾引号在冒号之前（KEY":"value），名字后必须先吃掉
    // 这个引号再匹配分隔符，否则只命中裸 env 形式
    re: /\b([A-Z][A-Z0-9_]{1,30}(?:_API)?_?(?:API_?KEY|TOKEN|SECRET|PASSWORD))\b(["']?)(\s*[=:]\s*)(["']?)([A-Za-z0-9._~+/=-]{12,})\4/g,
    build: m => `${m[1]}${m[2]}${m[3]}${m[4]}${maskSecret(m[5])}${m[4]}`,
  },
  {
    // 内嵌 JSON 字段："api_key": "xxx"（cat settings.json / 配置打印）
    re: /("(?:api[_-]?key|apiKey|authorization|access[_-]?token|refresh[_-]?token|client[_-]?secret|x-api-key)"\s*:\s*")([^"]{12,})(")/gi,
    build: m => `${m[1]}${maskSecret(m[2])}${m[3]}`,
  },
  {
    // URL query 键：?key=AIza… / &api-key=…（含 request.path 与文本里的 URL）
    re: /([?&](?:api[_-]?key|key|access_token|token)=)([A-Za-z0-9._~%-]{8,})/gi,
    build: m => `${m[1]}${maskSecret(m[2])}`,
  },
  {
    // 厂家前缀键：anthropic / openrouter / openai project / groq / google / xai /
    // aliyun dashscope (LTAI) / zhipu bigmodel (GL- 与 32hex.16 两种形态)
    re: /\b(sk-ant-[A-Za-z0-9_-]{8,}|sk-or-[A-Za-z0-9_-]{8,}|sk-proj-[A-Za-z0-9_-]{8,}|sk-svcacct-[A-Za-z0-9_-]{8,}|gsk_[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_-]{30,}|xai-[A-Za-z0-9_-]{16,}|GL-[A-Za-z0-9_-]{28,}|LTAI[A-Za-z0-9]{12,}|[0-9a-f]{32}\.[A-Za-z0-9]{10,20})/g,
    build: m => maskSecret(m[1]),
  },
  {
    // 泛 sk- 前缀（DeepSeek/Moonshot/OpenAI legacy：sk- + 32hex 类长串）
    re: /\bsk-[A-Za-z0-9_-]{28,}/g,
    build: m => maskSecret(m[0]),
  },
];

export function redactString(text: string): string {
  let out = text;
  for (const rule of STRING_RULES) {
    out = out.replace(rule.re, (...args) => {
      const groups = args.slice(0, -2) as RegExpMatchArray;
      return rule.build(groups);
    });
  }
  return out;
}

// 深度遍历并原地清洗任意 JSON 值；返回是否有改动。字符串未命中时保留原引用。
export function redactInPlace(value: unknown): boolean {
  let changed = false;
  if (Array.isArray(value)) {
    for (let i = 0; i < value.length; i++) {
      changed = redactElement(value, i, value[i]) || changed;
    }
  } else if (value && typeof value === 'object') {
    for (const k of Object.keys(value as Record<string, unknown>)) {
      changed = redactElement(value as Record<string, unknown>, k, (value as Record<string, unknown>)[k]) || changed;
    }
  }
  return changed;

  function redactElement(container: Record<string, unknown> | unknown[], key: string | number, v: unknown): boolean {
    if (typeof v === 'string') {
      if (typeof key === 'string' && isSensitiveKey(key)) {
        const masked = maskAuthValue(v);
        if (masked !== v) {
          (container as Record<string, unknown>)[key] = masked;
          return true;
        }
        return false;
      }
      const cleaned = redactString(v);
      if (cleaned !== v) {
        (container as Record<string, unknown>)[key] = cleaned;
        return true;
      }
      return false;
    }
    return redactInPlace(v);
  }
}

// ── 治理层（insight 侧独有）────────────────────────────────────────────

export interface GovernanceResult {
  /** 治理后的文件文本（jsonl 保持逐行结构） */
  output: string;
  /** 复检残留（非空 = 熔断） */
  residue: string[];
}

// 单条 jsonl 行：能 parse 则结构层+字符串层双重清洗后回序列化（保持一行），
// 不能 parse 的行整行过字符串层（不丢内容）。
function redactLine(line: string): string {
  const trimmed = line.trim();
  if (!trimmed) return line;
  try {
    const parsed: unknown = JSON.parse(trimmed);
    redactInPlace(parsed);
    return JSON.stringify(parsed);
  } catch {
    return redactString(line);
  }
}

function looksMasked(v: string): boolean {
  return v === '****' || v.includes('…');
}

// 独立复检：与清洗分离的第二遍扫描。正则组逐条重测（match 全量、无状态），
// 结构层检查敏感键的值是否仍为明文。清洗后掩码形态（前4…后4/****）不会再
// 命中，任何命中都是清洗未能覆盖的形态 → 熔断。
function scanResidue(text: string): string[] {
  const hits: string[] = [];
  for (const rule of STRING_RULES) {
    const matches = text.match(rule.re);
    if (matches) hits.push(...matches.slice(0, 3).map(s => s.slice(0, 40)));
  }
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const parsed: unknown = JSON.parse(trimmed);
      walk(parsed);
    } catch { /* 非 JSON 行已由正则组覆盖 */ }
  }
  return hits;

  function walk(v: unknown): void {
    if (Array.isArray(v)) { for (const e of v) walk(e); return; }
    if (v && typeof v === 'object') {
      for (const [k, val] of Object.entries(v as Record<string, unknown>)) {
        if (typeof val === 'string' && isSensitiveKey(k) && val && !looksMasked(val)) {
          hits.push(`${k}=${val.slice(0, 40)}`);
        } else {
          walk(val);
        }
      }
    }
  }
}

// 治理一份文件文本：清洗 → 复检。调用方负责把 output 写入 staging 副本；
// residue 非空时必须熔断，不得上传。
export function governText(text: string): GovernanceResult {
  const output = text.split('\n').map(redactLine).join('\n');
  return { output, residue: scanResidue(output) };
}
