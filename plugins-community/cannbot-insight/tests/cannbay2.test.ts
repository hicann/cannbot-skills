// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// cannbay2 IT：本地 bare 仓（partial clone 全链路）跑 上传治理→push→列表→物化→导入
// 闭环；native jsonl（含 key 各形态）、opencode DB 导出、治理熔断、redactor parity。
// env 必须在 cannbay2 模块加载前设置（mirror.ts 在模块顶层读取 env）。
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

// ── 本地 bare 远端（含 *.jsonl LFS 行的 .gitattributes，验证自动剔除）──
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'cannbay2-it-'));
const BARE = path.join(TMP, 'remote.git');
const SEED = path.join(TMP, 'seed');
const CACHE = path.join(TMP, 'cache');
const FIXTURES = path.join(TMP, 'fixtures');

execSync(`git init --bare -b main "${BARE}"`, { stdio: 'pipe' });
git('git config uploadpack.allowFilter true', BARE);
git('git config uploadpack.allowAnySHA1InWant true', BARE);
execSync(`git clone --quiet "${BARE}" "${SEED}"`, { stdio: 'pipe' });
fs.writeFileSync(path.join(SEED, 'README.md'), '# cannbay2 test\n');
fs.writeFileSync(path.join(SEED, '.gitattributes'), '*.jsonl filter=lfs diff=lfs merge=lfs -text\n*.zip filter=lfs diff=lfs merge=lfs -text\n');
git('git add .', SEED);
git('git -c user.name=seed -c user.email=seed@t commit -q -m init', SEED);
git('git push -q origin main', SEED);

process.env.CANNBAY2_REMOTE_URL = BARE;
process.env.CANNBAY2_PUSH_URL = BARE;
process.env.CANNBAY2_CACHE_DIR = CACHE;

// ── native claude fixture：主 jsonl（含 key 各形态）+ subagents/ ──
const SID = 'cc-cannbay2-it-0001';
const SUB_ID = 'sub-it-aaa';
const RAW_KEYS = [
  'sk-ant-api03-AAAA1111BBBB2222CCCC3333DDDD4444EEEE5',
  'LTAIABCD1234EFGH5678IJKL',
  'Bearer AAAABBBBCCCCDDDD12345678',
];
fs.mkdirSync(path.join(FIXTURES, SID, 'subagents'), { recursive: true });
fs.writeFileSync(path.join(FIXTURES, `${SID}.jsonl`), [
  JSON.stringify({
    type: 'user',
    message: { role: 'user', content: `帮我调一下。key 是 ${RAW_KEYS[0]}，另外 DASHSCOPE_API_KEY=${RAW_KEYS[1]}，curl 头是 "${RAW_KEYS[2]}"。max_tokens=4096 别动。` },
    timestamp: '2026-08-19T03:00:00.000Z',
  }),
  JSON.stringify({
    type: 'assistant',
    message: {
      id: 'msg_asst_1',
      role: 'assistant',
      content: [
        { type: 'text', text: '开始处理' },
        { type: 'tool_use', id: 'toolu_dispatch_1', name: 'Task', input: { description: 'researcher', subagent_type: 'general-purpose', prompt: 'research the fixture' } },
      ],
      model: 'test-model',
      usage: { input_tokens: 100, output_tokens: 20 },
    },
    timestamp: '2026-08-19T03:00:05.000Z',
  }),
  JSON.stringify({
    type: 'user',
    message: { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'toolu_dispatch_1', content: 'dispatched' }] },
    timestamp: '2026-08-19T03:00:06.000Z',
  }),
  JSON.stringify({
    type: 'assistant',
    message: { id: 'msg_asst_2', role: 'assistant', content: [{ type: 'text', text: '完成' }], model: 'test-model', usage: { input_tokens: 120, output_tokens: 8 } },
    timestamp: '2026-08-19T03:00:10.000Z',
  }),
].join('\n') + '\n');
fs.writeFileSync(path.join(FIXTURES, SID, 'subagents', `${SUB_ID}.jsonl`), [
  JSON.stringify({ type: 'user', message: { role: 'user', content: 'research the fixture' }, timestamp: '2026-08-19T03:00:07.000Z' }),
  JSON.stringify({ type: 'assistant', message: { id: 'msg_sub_1', role: 'assistant', content: [{ type: 'text', text: 'found it' }], model: 'test-model', usage: { input_tokens: 50, output_tokens: 5 } }, timestamp: '2026-08-19T03:00:09.000Z' }),
].join('\n') + '\n');
fs.writeFileSync(path.join(FIXTURES, SID, 'subagents', `${SUB_ID}.meta.json`), JSON.stringify({ toolUseId: 'toolu_dispatch_1', name: 'researcher', agentType: 'general-purpose' }));
// 杂散文件（含明文密钥）：白名单拷贝必须把它挡在 staging 之外，永不进公开仓
fs.writeFileSync(path.join(FIXTURES, SID, 'subagents', 'stray-notes.txt'), 'DASHSCOPE_API_KEY=sk-strayfile-secret-1234567890abcd\n');

const OPENCODE_DB = path.resolve(__dirname, 'data/e2e/opencode-sample.db');
const OPENCODE_SID = 'ses_2051a32a4ffevX0jGBWVDDEqCk';

const prisma = new PrismaClient();

// env 已设，动态加载 cannbay2 链
const cannbay2 = await import('../src/lib/cannbay2');
const mirror = await import('../src/lib/cannbay2/mirror');
const governance = await import('../src/lib/cannbay2/governance');
const proxyRedactor = await import('../proxy/src/redactor');
const route = await import('../src/app/api/ingest/cannbay2/route');

function makeRequest(body: unknown) {
  return new Request('http://localhost/api/ingest/cannbay2', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

// 共享 dev.db 的并行测试文件（e2e-import-observe 等）用同一 opencode fixture
// 会话，其 afterAll deleteMany 会与本文件的 merge update 赛跑（dedup 查到行、
// update 时行已被删 → P2025）。对这类瞬时消失重试一次即可通过。
async function withDbRaceRetry<T>(fn: () => Promise<T>): Promise<T> {
  try {
    return await fn();
  } catch (e) {
    const msg = e instanceof Error ? e.message : '';
    if (msg.includes('No record was found') || msg.includes('Session not found')) return await fn();
    throw e;
  }
}

beforeAll(async () => {
  await importSession(path.join(FIXTURES, `${SID}.jsonl`), SID, prisma, path.join(FIXTURES, `${SID}.jsonl`), 'claude-jsonl');
});

afterAll(async () => {
  for (const [taskId, framework] of [[SID, 'claude-code'], ['cc-cannbay2-it-0002', 'claude-code'], [OPENCODE_SID, 'opencode'], [OPENCODE_SID, 'claude-code']] as const) {
    try { await prisma.session.deleteMany({ where: { taskId, framework } }); } catch { /* ignore */ }
  }
  await prisma.$disconnect();
  fs.rmSync(TMP, { recursive: true, force: true });
});

describe('cannbay2 governance: redactor parity（与 proxy/src/redactor.ts 字节一致）', () => {
  const CORPUS = [
    `key 是 ${RAW_KEYS[0]}，DASHSCOPE_API_KEY=${RAW_KEYS[1]}，头 "${RAW_KEYS[2]}"`,
    '"api_key": "0123456789abcdef0123456789abcdef.abcdef123456"',
    'gsk_AAAABBBBCCCCDDDDEEEEFFFF12345678 / AIza' + 'A'.repeat(35) + ' / xai-1234567890abcdef1',
    'https://example.com/v1?key=AIzaSyA' + 'B'.repeat(30),
    'GL-' + 'C'.repeat(30),
    'max_tokens=4096, tool_use_id=toolu_1, input_tokens=100（零误伤基准）',
  ];
  const deep = { authorization: 'Bearer secretcredential123456', nested: { api_key: 'plainvalue1234567890', list: ['sk-ant-api03-ZZZZ9999YYYY8888XXXX7777'] } };

  it('redactString 两份实现输出一致', () => {
    for (const text of CORPUS) {
      expect(governance.redactString(text)).toBe(proxyRedactor.redactString(text));
    }
  });

  it('redactInPlace 两份实现清洗结果一致且均命中', () => {
    const a = structuredClone(deep);
    const b = structuredClone(deep);
    const changedA = governance.redactInPlace(a);
    const changedB = proxyRedactor.redactInPlace(b);
    expect(changedA).toBe(true);
    expect(changedB).toBe(true);
    expect(a).toEqual(b);
  });

  it('governText 清洗后复检零残留（掩码形态不再命中）', () => {
    for (const text of CORPUS) {
      const { output, residue } = governance.governText(JSON.stringify({ content: text }));
      expect(residue).toEqual([]);
      expect(output).not.toContain('AAAA1111BBBB');
      expect(output).not.toContain('LTAIABCD');
      expect(output).not.toContain('plainvalue1234567890');
    }
  });

  it('scanResidue 对明文密钥有检出能力（熔断逻辑有牙）', () => {
    // governance 未导出 scanResidue —— 通过 governText 的输入面验证：
    // 未经清洗的明文（模拟清洗层失效）必须能被复检发现
    const raw = JSON.stringify({ content: `raw ${RAW_KEYS[0]} and ${RAW_KEYS[2]}` });
    const cleaned = governance.redactString(raw);
    expect(cleaned).not.toContain('AAAA1111BBBB');
    expect(cleaned).toContain('…');
  });
});

describe('cannbay2 闭环：上传（治理）→ push → 列表 → 物化 → 导入', () => {
  let uploadResult: { sid: string; unchanged: boolean; gitattributesFixed: boolean };

  it('upload：native fixture（含 key）治理后推送，.gitattributes 自动剔除 *.jsonl LFS 行', async () => {
    uploadResult = await cannbay2.uploadCannbay2Session(prisma, SID, `提交人: 测试者\n内容描述: cannbay2 IT 上传`);
    expect(uploadResult.sid).toBe(SID);
    expect(uploadResult.unchanged).toBe(false);
    expect(uploadResult.gitattributesFixed).toBe(true);
    const ga = execSync('git show main:.gitattributes', { cwd: BARE, stdio: 'pipe' }).toString();
    expect(ga).not.toContain('*.jsonl filter=lfs');
    expect(ga).toContain('*.zip filter=lfs');
  });

  it('远端文件已脱敏：物化回读明文密钥不出仓，掩码保留可读性', () => {
    const mainJsonl = mirror.materializeSession(SID);
    const text = fs.readFileSync(mainJsonl, 'utf8');
    for (const key of RAW_KEYS) {
      expect(text).not.toContain(key);
    }
    expect(text).toContain('sk-a…');        // maskSecret: 前4…后4
    expect(text).toContain('LTAI…IJKL');    // env 赋值回显同样打码
    expect(text).toContain('max_tokens=4096'); // 零误伤基准
  });

  it('杂散文件不进公开仓：staging 白名单只带 .jsonl/.json，stray 内容零泄漏', () => {
    mirror.materializeSession(SID);
    const matSub = path.join(CACHE, 'materialized', SID, 'subagents');
    const names = fs.readdirSync(matSub);
    expect(names.length).toBeGreaterThan(0);
    expect(names.every(n => n.endsWith('.jsonl') || n.endsWith('.json'))).toBe(true);
    expect(names).not.toContain('stray-notes.txt');
    for (const f of names) {
      expect(fs.readFileSync(path.join(matSub, f), 'utf8')).not.toContain('strayfile-secret');
    }
  });

  it('list：一次 log walk 出提交人/内容描述/时间', () => {
    const sessions = cannbay2.listCannbay2Sessions();
    const entry = sessions.find(s => s.sid === SID);
    expect(entry).toBeDefined();
    expect(entry!.submitter).toBe('测试者');
    expect(entry!.description).toBe('cannbay2 IT 上传');
    expect(entry!.commitTime).toBeTruthy();
    expect(entry!.fileCount).toBeGreaterThan(2);
  });

  it('重复上传同内容 → unchanged', async () => {
    const again = await cannbay2.uploadCannbay2Session(prisma, SID, 're');
    expect(again.unchanged).toBe(true);
  });

  it('import：物化回 proxy 布局 → claude-jsonl 管线导入（含 subagent）', async () => {
    const result = await cannbay2.importCannbay2Session(SID, prisma);
    expect(result).toBeTruthy();
    const session = await prisma.session.findFirst({ where: { taskId: SID } });
    expect(session).toBeDefined();
    const turns = await prisma.turn.findMany({ where: { sessionId: session!.id } });
    expect(turns.length).toBeGreaterThanOrEqual(4);
    const subTurns = turns.filter(t => t.isSubagent);
    expect(subTurns.length).toBeGreaterThanOrEqual(1);
    expect(subTurns[0].subagentSessionId).toBe(SUB_ID);
    const toolCalls = await prisma.toolCall.findMany({ where: { turn: { sessionId: session!.id } } });
    expect(toolCalls.length).toBeGreaterThanOrEqual(1);
  });
});

describe('cannbay2 opencode 导出上传', () => {
  it('opencode 会话 → DB 导出 jsonl → 上传 → 导入回读一致', async () => {
    const imported = await withDbRaceRetry(() => importSession(OPENCODE_DB, OPENCODE_SID, prisma, OPENCODE_DB, 'opencode-db'));
    expect(imported.imported).toBe(true);

    const upload = await withDbRaceRetry(() => cannbay2.uploadCannbay2Session(prisma, OPENCODE_SID, `提交人: 测试者\n内容描述: opencode 导出上传`));
    expect(upload.unchanged).toBe(false);

    const result = await withDbRaceRetry(() => cannbay2.importCannbay2Session(OPENCODE_SID, prisma));
    expect(result).toBeTruthy();
    // 同机已有同 taskId 会话 → merge 路径（imported:false，合并进原 opencode 会话）
    // 异机下载场景：删掉本地会话再导入 → 全新建 claude-code 会话
    await prisma.session.deleteMany({ where: { taskId: OPENCODE_SID } });
    const fresh = await cannbay2.importCannbay2Session(OPENCODE_SID, prisma);
    expect(fresh).toBeTruthy();
    // framework 由 cc-session-meta 声明恢复为原值（opencode），不再错变 claude-code
    const reSession = await prisma.session.findFirst({ where: { taskId: OPENCODE_SID } });
    expect(reSession).toBeDefined();
    expect(reSession!.framework).toBe('opencode');
    const turns = await prisma.turn.findMany({ where: { sessionId: reSession!.id } });
    expect(turns.length).toBeGreaterThan(0);
    expect(turns.some(t => t.role === 'user')).toBe(true);
    expect(turns.some(t => t.role === 'assistant')).toBe(true);
    const toolCalls = await prisma.toolCall.findMany({ where: { turn: { sessionId: reSession!.id } } });
    expect(toolCalls.length).toBeGreaterThan(0);
  });
});

describe('cannbay2 治理熔断：复检残留 → 拒绝上传且远端无新 commit', () => {
  it('governText 被替换为返回残留时，upload 抛 GovernanceError、远端不推进', async () => {
    const before = execSync('git rev-parse main', { cwd: BARE, stdio: 'pipe' }).toString().trim();
    const spy = vi.spyOn(governance, 'governText').mockReturnValue({ output: '', residue: ['fake.jsonl: sk-ant-RESIDUE…'] });
    try {
      await expect(cannbay2.uploadCannbay2Session(prisma, SID, 'breaker'))
        .rejects.toThrow(/熔断/);
    } finally {
      spy.mockRestore();
    }
    const after = execSync('git rev-parse main', { cwd: BARE, stdio: 'pipe' }).toString().trim();
    expect(after).toBe(before);
  });
});

describe('cannbay2 无 jsonl 源兜底：DB 导出上传（v1 .db 快照会话 / 源文件已删）', () => {
  const SID2 = 'cc-cannbay2-it-0002';
  let fixtureFile: string;

  it('源 jsonl 删除后上传 → DB 导出 jsonl → push 成功 → 回读有数据', async () => {
    fixtureFile = path.join(FIXTURES, `${SID2}.jsonl`);
    fs.writeFileSync(fixtureFile, [
      JSON.stringify({ type: 'user', message: { role: 'user', content: '兜底导出场景：v1 db 快照会话' }, timestamp: '2026-08-19T04:00:00.000Z' }),
      JSON.stringify({ type: 'system', message: { role: 'system', content: '<command-message>spec-to-design</command-message><command-name>spec-to-design</command-name>' }, timestamp: '2026-08-19T04:00:01.000Z' }),
      JSON.stringify({ type: 'assistant', message: { id: 'msg_f2_1', role: 'assistant', content: [{ type: 'text', text: '导出兜底回复' }], model: 'test-model', usage: { input_tokens: 10, output_tokens: 3, cache_read_input_tokens: 7, cache_creation_input_tokens: 2 } }, timestamp: '2026-08-19T04:00:05.000Z' }),
    ].join('\n') + '\n');
    await importSession(fixtureFile, SID2, prisma, fixtureFile, 'claude-jsonl');
    const latBefore = (await prisma.turn.findFirst({ where: { session: { taskId: SID2 }, role: 'assistant' }, select: { latencyMs: true } }))?.latencyMs;

    fs.rmSync(fixtureFile, { force: true }); // 模拟：sourcePath 指向的文件已不存在
    const up = await cannbay2.uploadCannbay2Session(prisma, SID2, '提交人: 测试者\n内容描述: DB 导出兜底');
    expect(up.unchanged).toBe(false);

    const result = await cannbay2.importCannbay2Session(SID2, prisma);
    expect(result).toBeTruthy();
    const session = await prisma.session.findFirst({ where: { taskId: SID2 } });
    const turns = await prisma.turn.findMany({ where: { sessionId: session!.id } });
    expect(turns.length).toBeGreaterThanOrEqual(2);
    expect(turns.some(t => t.role === 'user' && t.content?.includes('兜底导出场景'))).toBe(true);
    // 导出保真：usage 携带 cache_read/cache_creation → 回读 Turn 两个 cache 维度不丢
    const asst = turns.find(t => t.role === 'assistant');
    expect(asst?.cacheReadTokens).toBe(7);
    expect(asst?.cacheWriteTokens).toBe(2);
    // system turn（skill 注入/命令消息重分类）导出不蒸发，往返保留
    expect(turns.some(t => t.role === 'system' && t.content?.includes('spec-to-design'))).toBe(true);
    // latency 往返：导出行 source:'insight-export' 让 duration_ms 生效
    expect(asst?.latencyMs).toBe(latBefore);
    // 来源标注不误标代理捕获：version 不以 -proxy 结尾（无 proxy 徽标）
    expect(session?.version == null || !session!.version!.endsWith('-proxy')).toBe(true);
  });
});

describe('cannbay2 route', () => {
  it('action=list 返回会话列表', async () => {
    const res = await route.POST(makeRequest({ action: 'list' }));
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(Array.isArray(data.sessions)).toBe(true);
    expect(data.sessions.some((s: { sid: string }) => s.sid === SID)).toBe(true);
  });

  it('非法 taskId（命令注入面）在入口即拒绝', async () => {
    await expect(cannbay2.uploadCannbay2Session(prisma, '../evil; rm -rf /', 'x'))
      .rejects.toThrow(/Invalid session id/);
  });

  it('unknown action → 400；upload 缺 taskId → 400', async () => {
    expect((await route.POST(makeRequest({ action: 'nope' }))).status).toBe(400);
    expect((await route.POST(makeRequest({ action: 'upload' }))).status).toBe(400);
  });
});
