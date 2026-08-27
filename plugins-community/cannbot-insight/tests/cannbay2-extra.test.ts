// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// cannbay2 v2 IT 扩展（与 cannbay2.test.ts 互补）：镜像自愈（fetch 失败 →
// 删库重建）、治理形态全覆盖 + parity、重复上传覆盖语义、并发上传串行
// （withWriteLock mutex）、100MB 超限熔断。独立 bare 仓 + cache，taskId
// 用 'cc-extra-' 前缀避免与 cannbay2.test.ts 冲突。env 必须在 cannbay2 模块
// 加载前设置（mirror.ts 在模块顶层读取 env）。
import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { execSync } from 'node:child_process';
import { PrismaClient } from '@prisma/client';
import { importSession } from '../src/lib/ingest/data-service.ts';

function git(cmd: string, cwd: string): void {
  execSync(cmd, { cwd, stdio: 'pipe', timeout: 60_000 });
}

// ── 独立 bare 远端 + cache（与 cannbay2.test.ts 隔离，同 worker env 互不影响）──
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'cannbay2-extra-it-'));
const BARE = path.join(TMP, 'remote.git');
const SEED = path.join(TMP, 'seed');
const CACHE = path.join(TMP, 'cache');
const FIXTURES = path.join(TMP, 'fixtures');

execSync(`git init --bare -b main "${BARE}"`, { stdio: 'pipe' });
git('git config uploadpack.allowFilter true', BARE);
git('git config uploadpack.allowAnySHA1InWant true', BARE);
execSync(`git clone --quiet "${BARE}" "${SEED}"`, { stdio: 'pipe' });
fs.writeFileSync(path.join(SEED, 'README.md'), '# cannbay2 extra\n');
git('git add .', SEED);
git('git -c user.name=seed -c user.email=seed@t commit -q -m init', SEED);
git('git push -q origin main', SEED);

process.env.CANNBAY2_REMOTE_URL = BARE;
process.env.CANNBAY2_PUSH_URL = BARE;
process.env.CANNBAY2_CACHE_DIR = CACHE;

const prisma = new PrismaClient();

const cannbay2 = await import('../src/lib/cannbay2');
const mirror = await import('../src/lib/cannbay2/mirror');
const governance = await import('../src/lib/cannbay2/governance');
const proxyRedactor = await import('../proxy/src/redactor');

const EXTRA_SIDS: string[] = [];

function writeJsonlFixture(sid: string, lines: string[]): string {
  fs.mkdirSync(FIXTURES, { recursive: true });
  const f = path.join(FIXTURES, `${sid}.jsonl`);
  fs.writeFileSync(f, lines.join('\n') + '\n');
  return f;
}

function turnLine(turnIndex: number, role: 'user' | 'assistant', content: string, model = 'test-model'): string {
  return JSON.stringify({
    type: role,
    message: {
      id: `msg_${sid()}_${turnIndex}`,
      role,
      content,
      model,
      usage: { input_tokens: 10, output_tokens: 5 },
    },
    timestamp: new Date(Date.now() + turnIndex * 1000).toISOString(),
  });
}

// 占位 sid 用来给 message.id 注入唯一性（避免不同 fixture 的 msg_id 撞车影响 dedup）
function sid(): string { return Math.random().toString(36).slice(2, 8); }

beforeAll(async () => { /* setup 在模块顶层完成 */ });
afterAll(async () => {
  for (const taskId of EXTRA_SIDS) {
    try { await prisma.session.deleteMany({ where: { taskId } }); } catch { /* ignore */ }
  }
  await prisma.$disconnect();
  fs.rmSync(TMP, { recursive: true, force: true });
});

// ─────────────────────────────────────────────────────────────────────
describe('cannbay2 镜像自愈：fetch 失败 → 删库重建', () => {
  it('ensureMirror：fetch 抛错 → rmSync → cloneMirror 重建 → 仍可 list', () => {
    const dir = mirror.cannbay2CacheDir();
    mirror.ensureMirror();
    expect(fs.existsSync(path.join(dir, '.git'))).toBe(true);

    // 破坏 origin 让 fetch 抛错（指向不存在的远端）
    const cfgPath = path.join(dir, '.git', 'config');
    const cfg = fs.readFileSync(cfgPath, 'utf8').replace(/url = .*/g, 'url = /nonexistent-broken-remote.git');
    fs.writeFileSync(cfgPath, cfg);

    // ensureMirror: fetch 失败 catch → rmSync(.git) → cloneMirror 用 CANNBAY2_PULL_URL(=BARE) 重建
    mirror.ensureMirror();
    expect(fs.existsSync(path.join(dir, '.git'))).toBe(true);

    const sessions = cannbay2.listCannbay2Sessions();
    expect(Array.isArray(sessions)).toBe(true);
  });

  it('revParseMain：非镜像目录 → null（catch 分支容错）', () => {
    expect(mirror.revParseMain(path.join(os.tmpdir(), 'nonexistent-mirror-xyz-456'))).toBe(null);
  });
});

// ─────────────────────────────────────────────────────────────────────
describe('cannbay2 治理形态全覆盖 + parity（与 proxy/src/redactor.ts 字节一致）', () => {
  // 厂家前缀键：覆盖 governance STRING_RULES 第 5 条全部 9 种前缀
  const KEY_FORMS = [
    'sk-ant-api03-AAAA1111BBBB2222CCCC3333DDDD4444EEEE5', // anthropic
    'sk-or-aaaabbbbcccc1111222233334444',                  // openrouter
    'sk-proj-aaaabbbbcccc1111222233334444',                // openai project
    'sk-svcacct-aaaabbbbcccc1111222233334444',             // openai service account
    'gsk_' + 'a'.repeat(25),                                // groq (gsk_ + 20+)
    'AIza' + 'A'.repeat(35),                                // google (AIza + 30+)
    'xai-1234567890abcdef1',                                // xai (xai- + 16+)
    'GL-' + 'C'.repeat(30),                                 // zhipu bigmodel (GL- + 28+)
    'LTAIABCD1234EFGH5678IJKL',                             // aliyun dashscope (LTAI + 12+)
    'sk-' + 'a'.repeat(40),                                 // DeepSeek/Moonshot/OpenAI legacy (sk- + 28+)
    'abcdef0123456789abcdef0123456789.abcdefghijklmnop',   // zhipu bigmodel 32hex.16-20
    'Bearer AAAABBBBCCCCDDDD12345678',                      // Bearer 凭据
  ];

  it('每个厂家前缀键：清洗后无明文 + 复检零残留 + parity', () => {
    for (const key of KEY_FORMS) {
      const text = `配置 dump: ${key}`;
      const { output, residue } = governance.governText(JSON.stringify({ content: text }));
      expect(residue, `residue should be empty for ${key}`).toEqual([]);
      expect(output, `output should not contain ${key}`).not.toContain(key);
      // parity：两份实现字节一致
      expect(governance.redactString(text)).toBe(proxyRedactor.redactString(text));
    }
  });

  it('env 回显形态：ANTHROPIC_API_KEY=xxx 清洗打码 + parity', () => {
    const text = 'ANTHROPIC_API_KEY=sk-ant-api03-ZZZZ9999YYYY8888XXXX7777';
    const { output, residue } = governance.governText(text);
    expect(residue).toEqual([]);
    expect(output).not.toContain('sk-ant-api03-ZZZZ9999');
    expect(output).toContain('…');
    expect(governance.redactString(text)).toBe(proxyRedactor.redactString(text));
  });

  it('JSON 字段形态："api_key":"xxx" 清洗打码 + parity', () => {
    const text = '{"api_key":"0123456789abcdef0123456789abcdef.abcdef123456"}';
    const { output, residue } = governance.governText(text);
    expect(residue).toEqual([]);
    expect(output).not.toContain('0123456789abcdef0123456789abcdef');
    expect(governance.redactString(text)).toBe(proxyRedactor.redactString(text));
  });

  it('URL query 形态：?key=AIza... 清洗打码 + parity', () => {
    const text = 'https://example.com/v1?key=AIzaSyA' + 'B'.repeat(30);
    const { output, residue } = governance.governText(text);
    expect(residue).toEqual([]);
    expect(output).not.toContain('AIzaSyA' + 'B'.repeat(30));
    expect(governance.redactString(text)).toBe(proxyRedactor.redactString(text));
  });

  it('结构层键名：authorization / api_key / password 深度遍历打码 + parity', () => {
    const deep = {
      authorization: 'Bearer secretcredential123456',
      nested: { api_key: 'plainvalue1234567890', password: 'mypass12345678' },
      list: ['sk-ant-api03-ZZZZ9999YYYY8888XXXX7777'],
    };
    const a = structuredClone(deep);
    const b = structuredClone(deep);
    expect(governance.redactInPlace(a)).toBe(true);
    expect(proxyRedactor.redactInPlace(b)).toBe(true);
    expect(a).toEqual(b); // parity
    expect(a.nested.api_key).not.toBe('plainvalue1234567890');
    expect(a.nested.password).not.toBe('mypass12345678');
    expect(a.authorization).not.toBe('Bearer secretcredential123456');
  });

  it('零误伤基准：max_tokens / tool_use_id / input_tokens 不被打码', () => {
    const text = 'max_tokens=4096, tool_use_id=toolu_1, input_tokens=100';
    const { output, residue } = governance.governText(text);
    expect(residue).toEqual([]);
    expect(output).toEqual(text);
  });
});

// ─────────────────────────────────────────────────────────────────────
describe('cannbay2 重复上传覆盖语义：改内容后重传同 sid → 新 commit 替换文件夹', () => {
  const SID_OVR = 'cc-extra-overwrite';
  EXTRA_SIDS.push(SID_OVR);

  it('v1 上传 → 改 fixture → v2 重传 → list description 更新、远端内容是 v2、commit 推进', async () => {
    const tag = sid();
    const f = writeJsonlFixture(SID_OVR, [
      turnLine(0, 'user', `version 1 query ${tag}`, undefined),
      turnLine(1, 'assistant', `version 1 response ${tag}`),
    ]);
    await importSession(f, SID_OVR, prisma, f, 'claude-jsonl');

    const up1 = await cannbay2.uploadCannbay2Session(prisma, SID_OVR, '提交人: t\n内容描述: v1');
    expect(up1.unchanged).toBe(false);

    const list1 = cannbay2.listCannbay2Sessions().find(s => s.sid === SID_OVR);
    expect(list1).toBeDefined();
    expect(list1!.description).toBe('v1');
    const commit1 = execSync('git rev-parse main', { cwd: BARE, stdio: 'pipe' }).toString().trim();

    // 改 fixture 内容为 v2（覆盖）— 直接覆写文件
    fs.writeFileSync(f, [
      turnLine(0, 'user', `version 2 query ${tag}`, undefined),
      turnLine(1, 'assistant', `version 2 response ${tag}`),
    ].join('\n') + '\n');
    await importSession(f, SID_OVR, prisma, f, 'claude-jsonl'); // merge dedup 更新 turns

    const up2 = await cannbay2.uploadCannbay2Session(prisma, SID_OVR, '提交人: t\n内容描述: v2');
    expect(up2.unchanged).toBe(false); // 内容真变了 → 新 commit

    const list2 = cannbay2.listCannbay2Sessions().find(s => s.sid === SID_OVR);
    expect(list2!.description).toBe('v2');
    const commit2 = execSync('git rev-parse main', { cwd: BARE, stdio: 'pipe' }).toString().trim();
    expect(commit2).not.toBe(commit1); // 远端推进了新 commit（覆盖更新语义）

    // 物化回读：远端内容是 v2，v1 已被替换
    const mainJsonl = mirror.materializeSession(SID_OVR);
    const text = fs.readFileSync(mainJsonl, 'utf8');
    expect(text).toContain(`version 2 query ${tag}`);
    expect(text).not.toContain(`version 1 query ${tag}`);
  });
});

// ─────────────────────────────────────────────────────────────────────
describe('cannbay2 并发上传串行（withWriteLock mutex：sparse/commit/push 互不串台）', () => {
  const SID_A = 'cc-extra-conc-a';
  const SID_B = 'cc-extra-conc-b';
  EXTRA_SIDS.push(SID_A, SID_B);

  it('两个不同 sid 并发 Promise.all → 都成功、互不串台、git log 两个 commit', async () => {
    const tagA = sid();
    const tagB = sid();
    const fA = writeJsonlFixture(SID_A, [
      turnLine(0, 'user', `concurrent A query ${tagA}`, undefined),
      turnLine(1, 'assistant', `concurrent A response ${tagA}`),
    ]);
    const fB = writeJsonlFixture(SID_B, [
      turnLine(0, 'user', `concurrent B query ${tagB}`, undefined),
      turnLine(1, 'assistant', `concurrent B response ${tagB}`),
    ]);
    await importSession(fA, SID_A, prisma, fA, 'claude-jsonl');
    await importSession(fB, SID_B, prisma, fB, 'claude-jsonl');

    const [upA, upB] = await Promise.all([
      cannbay2.uploadCannbay2Session(prisma, SID_A, '提交人: t\n内容描述: A'),
      cannbay2.uploadCannbay2Session(prisma, SID_B, '提交人: t\n内容描述: B'),
    ]);
    expect(upA.unchanged).toBe(false);
    expect(upB.unchanged).toBe(false);

    // 两个 sid 都在远端列表
    const sessions = cannbay2.listCannbay2Sessions();
    expect(sessions.some(s => s.sid === SID_A)).toBe(true);
    expect(sessions.some(s => s.sid === SID_B)).toBe(true);

    // 物化互不串台：A 文件夹只有 A 内容，B 文件夹只有 B 内容
    const textA = fs.readFileSync(mirror.materializeSession(SID_A), 'utf8');
    const textB = fs.readFileSync(mirror.materializeSession(SID_B), 'utf8');
    expect(textA).toContain(`concurrent A query ${tagA}`);
    expect(textB).toContain(`concurrent B query ${tagB}`);
    expect(textA).not.toContain(`concurrent B query ${tagB}`);
    expect(textB).not.toContain(`concurrent A query ${tagA}`);

    // git log 两个新 commit（按约定提交消息含"内容描述: A/B"）
    const log = execSync('git log --pretty=format:%s main', { cwd: BARE, stdio: 'pipe' }).toString();
    expect(log).toContain('内容描述: A');
    expect(log).toContain('内容描述: B');
  });
});

// ─────────────────────────────────────────────────────────────────────
describe('cannbay2 100MB 超限熔断（MAX_SESSION_BYTES）', () => {
  const SID_BIG = 'cc-extra-oversize';
  EXTRA_SIDS.push(SID_BIG);

  it('fixture >100MB → GovernanceError（size limit）、远端无新 commit', async () => {
    // 写 105MB padding 文件（Buffer.alloc 直接写盘，不占 JS 字符串内存）。
    // 直接 prisma.session.create 创建元数据（跳过 importSession 的 105MB
    // adapter 解析 —— stageSessionFiles 走 sourcePath 直传分支只需
    // session.sourcePath + framework 在 DB 里）。
    // spy governance.governText 跳过实际清洗 + scanResidue —— 本用例专测
    // MAX_SESSION_BYTES 阈值熔断路径，治理逻辑在 cannbay2.test.ts 已覆盖。
    const bigFixture = path.join(FIXTURES, `${SID_BIG}.jsonl`);
    fs.mkdirSync(FIXTURES, { recursive: true });
    const fd = fs.openSync(bigFixture, 'w');
    fs.writeSync(fd, '{"type":"user","message":{"role":"user","content":"');
    const padding = Buffer.alloc(105 * 1024 * 1024, 0x61); // 'a' × 105MB
    fs.writeSync(fd, padding);
    fs.writeSync(fd, '"},"timestamp":"2026-08-20T00:00:00.000Z"}\n');
    fs.closeSync(fd);

    await prisma.session.create({
      data: { taskId: SID_BIG, framework: 'claude-code', sourcePath: bigFixture },
    });

    const before = execSync('git rev-parse main', { cwd: BARE, stdio: 'pipe' }).toString().trim();

    const spy = vi.spyOn(governance, 'governText').mockImplementation(() => ({ output: '', residue: [] }));
    try {
      await expect(cannbay2.uploadCannbay2Session(prisma, SID_BIG, '提交人: t\n内容描述: oversize'))
        .rejects.toThrow(/exceeds single-file size limit|100MB/);
    } finally {
      spy.mockRestore();
    }

    const after = execSync('git rev-parse main', { cwd: BARE, stdio: 'pipe' }).toString().trim();
    expect(after).toBe(before); // 熔断后远端无新 commit
  }, 60000);
});
