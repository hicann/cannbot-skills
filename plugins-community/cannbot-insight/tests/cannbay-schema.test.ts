// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// x_cannbay schema IT（docs/cannbay-schema-spec.md）：
// ① 形状一致锁 —— 声明的 schema ∈ 注册表且 version=1
// ② 双写一致锁 —— legacy 顶层字段 ≡ x_cannbay.data（同值）
// ③ 列清单对账锁（数据全不能少）—— 手工构造全列富 DB 会话 → 导出 → 上传 →
//    删库 → 再导入 → 逐列多重集比对 0 丢失 0 损造
// ④ 混排容错锁 —— native 行 / legacy-proxy 行 / 纯 x_cannbay 行同文件互不污染
// env 必须在 cannbay2 模块加载前设置（mirror.ts 模块顶层读 env）。
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { execSync } from 'node:child_process';
import { PrismaClient } from '@prisma/client';
import { importSession } from '../src/lib/ingest/data-service.ts';

function git(cmd: string, cwd: string): void {
  execSync(cmd, { cwd, stdio: 'pipe', timeout: 60_000 });
}

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'cannbay-schema-it-'));
const BARE = path.join(TMP, 'remote.git');
const SEED = path.join(TMP, 'seed');
const CACHE = path.join(TMP, 'cache');
const STAGE = path.join(TMP, 'stage');

execSync(`git init --bare -b main "${BARE}"`, { stdio: 'pipe' });
git('git config uploadpack.allowFilter true', BARE);
git('git config uploadpack.allowAnySHA1InWant true', BARE);
execSync(`git clone --quiet "${BARE}" "${SEED}"`, { stdio: 'pipe' });
fs.writeFileSync(path.join(SEED, 'README.md'), '# schema test\n');
git('git add .', SEED);
git('git -c user.name=seed -c user.email=seed@t commit -q -m init', SEED);
git('git push -q origin main', SEED);

process.env.CANNBAY2_REMOTE_URL = BARE;
process.env.CANNBAY2_PUSH_URL = BARE;
process.env.CANNBAY2_CACHE_DIR = CACHE;

const prisma = new PrismaClient();
const cannbay2 = await import('../src/lib/cannbay2');
const mirror = await import('../src/lib/cannbay2/mirror');
const exportMod = await import('../src/lib/cannbay2/export');

const SID = 'schema-rt-0001';
const SUB_ID = 'sub-schema-1';
const TS0 = new Date('2026-08-19T05:00:00.000Z');

// ── ③ 的列清单（对账范围 = 这些直载列；id/turnIndex/聚合/bridge 为重推导列）──
interface TurnSnapshot {
  role: string; content: string | null; model: string | null;
  inputTokens: number; outputTokens: number; reasoningTokens: number;
  cacheReadTokens: number; cacheWriteTokens: number;
  latencyMs: number; ttftMs: number | null; finishReason: string | null;
  temperature: number | null; maxTokens: number | null;
  modelId: string | null; providerId: string | null;
  isSubagent: boolean; subagentSessionId: string | null;
  createdAt_ts: string | null;
}
interface ToolCallSnapshot {
  toolCallId: string; toolName: string; argsJson: string | null; resultJson: string | null;
  state: string; errorType: string | null; errorMessage: string | null;
  durationMs: number; startedAt: string | null;
}

function normJson(s: string | null): string | null {
  if (s == null) return null;
  // 字符串结果两侧引号约定不同（seed 带引号 / 管线裸值）——信息等价，统一剥壳
  try { const p: unknown = JSON.parse(s); return typeof p === 'string' ? p : JSON.stringify(p); } catch { return s; }
}

async function snapTurns(sessionId: string): Promise<TurnSnapshot[]> {
  const rows = await prisma.turn.findMany({ where: { sessionId }, orderBy: { turnIndex: 'asc' } });
  return rows.map(t => ({
    role: t.role, content: t.content, model: t.model,
    inputTokens: t.inputTokens, outputTokens: t.outputTokens, reasoningTokens: t.reasoningTokens,
    cacheReadTokens: t.cacheReadTokens, cacheWriteTokens: t.cacheWriteTokens,
    // 非 assistant 轮的 latency 是时间线推导值（下一轮时间差），非直载列 → 归零不参与对账
    latencyMs: t.role === 'assistant' ? t.latencyMs : 0,
    ttftMs: t.ttftMs, finishReason: t.finishReason,
    temperature: t.temperature, maxTokens: t.maxTokens,
    modelId: t.modelId, providerId: t.providerId,
    isSubagent: t.isSubagent, subagentSessionId: t.subagentSessionId,
    createdAt_ts: t.createdAt_ts ? t.createdAt_ts.toISOString() : null,
  }));
}

async function snapToolCalls(sessionId: string): Promise<ToolCallSnapshot[]> {
  const rows = await prisma.toolCall.findMany({ where: { turn: { sessionId } } });
  return rows.map(tc => ({
    toolCallId: tc.toolCallId, toolName: tc.toolName,
    argsJson: normJson(tc.argsJson), resultJson: normJson(tc.resultJson),
    state: tc.state, errorType: tc.errorType, errorMessage: tc.errorMessage,
    durationMs: tc.durationMs, startedAt: tc.startedAt ? tc.startedAt.toISOString() : null,
  }));
}

// 多重集比对（与位置无关）：排序序列化后全等 + 差异定位报第一条
function expectMultisetEqual<T>(before: T[], after: T[], label: string): void {
  const key = (x: T) => JSON.stringify(x);
  const b = [...before].map(key).sort();
  const a = [...after].map(key).sort();
  const lost = b.filter(k => !a.includes(k));
  const fab = a.filter(k => !b.includes(k));
  expect(lost, `${label}：丢失 ${lost.length} 项 → ${lost.slice(0, 3).join(' ; ')}`).toEqual([]);
  expect(fab, `${label}：多出(损造) ${fab.length} 项 → ${fab.slice(0, 3).join(' ; ')}`).toEqual([]);
}

// ── 富 fixture：每个直载列都给非默认值 ──
async function seedRichSession(): Promise<void> {
  await prisma.session.deleteMany({ where: { taskId: SID } });
  const session = await prisma.session.create({
    data: { taskId: SID, framework: 'opencode', version: 'test-cc-2.1.100' },
  });
  const mk = (data: Record<string, unknown>) => prisma.turn.create({
    data: { sessionId: session.id, ...data },
  });
  await mk({ turnIndex: 0, role: 'user', content: '帮我查一下构建失败原因', createdAt_ts: TS0 });
  await mk({
    turnIndex: 1, role: 'assistant', content: '先看日志', model: 'test-model-x',
    inputTokens: 100, outputTokens: 20, reasoningTokens: 77, cacheReadTokens: 11, cacheWriteTokens: 7,
    totalTokens: 215, latencyMs: 4321, ttftMs: 321, finishReason: 'end_turn',
    temperature: 0.7, maxTokens: 4096, modelId: 'mid-abc', providerId: 'prov-zzz',
    createdAt_ts: new Date(TS0.getTime() + 5_000),
  });
  await mk({ turnIndex: 2, role: 'system', content: 'Available agent types for the Agent tool: Explore', createdAt_ts: new Date(TS0.getTime() + 6_000) });
  const t3 = await mk({
    turnIndex: 3, role: 'assistant', content: '派一个子代理并跑诊断', model: 'test-model-x',
    inputTokens: 200, outputTokens: 30, cacheReadTokens: 20, totalTokens: 250,
    latencyMs: 8888, finishReason: 'tool_use', createdAt_ts: new Date(TS0.getTime() + 10_000),
  });
  await prisma.toolCall.createMany({
    data: [
      { turnId: t3.id, toolCallId: 'tc_schema_dispatch', toolName: 'Agent', argsJson: JSON.stringify({ prompt: 'find the failing test', description: 'finder', subagent_type: 'Explore' }), resultJson: JSON.stringify('dispatched'), state: 'ok', durationMs: 15, startedAt: new Date(TS0.getTime() + 10_100) },
      { turnId: t3.id, toolCallId: 'tc_schema_bash', toolName: 'Bash', argsJson: JSON.stringify({ command: 'npm test' }), resultJson: JSON.stringify({ code: 1, msg: 'boom' }), state: 'error', errorType: 'ExecutionError', errorMessage: 'exit 1', durationMs: 99, startedAt: new Date(TS0.getTime() + 10_200) },
    ],
  });
  await prisma.interactionBridge.create({
    data: {
      sessionId: session.id, dispatchExecutionId: 'exec-main', dispatchTurnId: t3.id,
      dispatchToolCallId: 'tc_schema_dispatch', subagentSessionId: SUB_ID,
      subagentType: 'Explore', subagentName: 'finder',
    },
  });
  await mk({
    turnIndex: 0, role: 'user', content: 'find the failing test', isSubagent: true, subagentSessionId: SUB_ID,
    createdAt_ts: new Date(TS0.getTime() + 7_000),
  });
  await mk({
    turnIndex: 1, role: 'assistant', content: '找到了：cannbay-schema.test.ts', model: 'test-model-x',
    isSubagent: true, subagentSessionId: SUB_ID, inputTokens: 60, outputTokens: 9, totalTokens: 69,
    latencyMs: 555, finishReason: 'end_turn', createdAt_ts: new Date(TS0.getTime() + 9_000),
  });
}

beforeAll(async () => {
  await seedRichSession();
});

afterAll(async () => {
  await prisma.session.deleteMany({ where: { taskId: SID } });
  await prisma.session.deleteMany({ where: { taskId: 'schema-mixed-0002' } });
  await prisma.$disconnect();
  fs.rmSync(TMP, { recursive: true, force: true });
});

describe('cannbay schema：列清单对账 round-trip（数据全不能少）', () => {
  let beforeTurns: TurnSnapshot[];
  let beforeToolCalls: ToolCallSnapshot[];
  let stagingDir: string;

  it('导出 → ① 形状锁 + ② 双写锁', async () => {
    const s = await prisma.session.findFirst({ where: { taskId: SID } });
    expect(s).toBeTruthy();
    beforeTurns = await snapTurns(s!.id);
    beforeToolCalls = await snapToolCalls(s!.id);
    expect(beforeTurns.length).toBe(6);

    stagingDir = path.join(STAGE, 'run1');
    const written = await exportMod.exportSessionToClaudeJsonl(prisma, SID, stagingDir);
    expect(written.some(w => w.endsWith(`${SID}.jsonl`))).toBe(true);
    expect(written.some(w => w.endsWith(`${SID}.meta.json`))).toBe(true);
    expect(written.some(w => w.endsWith(`subagents/${SUB_ID}.jsonl`))).toBe(true);
    expect(written.some(w => w.endsWith(`subagents/${SUB_ID}.meta.json`))).toBe(true);

    // ① 形状一致：所有声明 ∈ 注册表
    const LINE_SCHEMAS = new Set(['cc-db-turn', 'cc-wire-round', 'cc-wire-input']);
    const META_SCHEMAS = new Set(['cc-session-meta', 'cc-subagent-meta']);
    let assistantDecls = 0;
    for (const rel of written.filter(w => w.endsWith('.jsonl'))) {
      const file = path.join(stagingDir, rel);
      for (const ln of fs.readFileSync(file, 'utf8').split('\n')) {
        if (!ln.trim()) continue;
        const line = JSON.parse(ln);
        if (!line.x_cannbay) continue;
        expect(LINE_SCHEMAS.has(line.x_cannbay.schema), `未知行 schema: ${line.x_cannbay.schema}`).toBe(true);
        expect(line.x_cannbay.version).toBe(1);
        if (line.x_cannbay.schema === 'cc-db-turn') {
          assistantDecls++;
          // ② 双写一致：legacy ≡ x_cannbay
          expect(line.duration_ms).toBe(line.x_cannbay.data.latencyMs);
          expect(line.stopReason ?? undefined).toBe(line.x_cannbay.data.stopReason ?? undefined);
        }
      }
    }
    for (const rel of written.filter(w => w.endsWith('.meta.json'))) {
      const meta = JSON.parse(fs.readFileSync(path.join(stagingDir, rel), 'utf8'));
      expect(META_SCHEMAS.has(meta.x_cannbay.schema)).toBe(true);
      expect(meta.x_cannbay.version).toBe(1);
    }
    expect(assistantDecls).toBe(3); // 主 2 + 子 1
  });

  it('上传 → 删库 → 再导入', async () => {
    const up = await mirror.uploadFolder(stagingDir, SID, 'schema round-trip IT');
    expect(up.unchanged).toBe(false);
    await prisma.session.deleteMany({ where: { taskId: SID } });
    expect(await prisma.session.findFirst({ where: { taskId: SID } })).toBeNull();
    const result = await cannbay2.importCannbay2Session(SID, prisma);
    expect(result).toBeTruthy();
  });

  it('③ 逐列对账：0 丢失 0 损造', async () => {
    const s = await prisma.session.findFirst({ where: { taskId: SID } });
    expect(s).toBeTruthy();
    // Session 直载列：framework/version 由 cc-session-meta 恢复（不再错变 claude-code）
    expect(s!.framework).toBe('opencode');
    expect(s!.version).toBe('test-cc-2.1.100');

    const afterTurns = await snapTurns(s!.id);
    expect(afterTurns.length).toBe(beforeTurns.length);
    // 角色序列：主/子分流后各自顺序一致（导入先主后子，turnIndex 不跨流可比）
    expect(afterTurns.filter(t => !t.isSubagent).map(t => t.role))
      .toEqual(beforeTurns.filter(t => !t.isSubagent).map(t => t.role));
    expect(afterTurns.filter(t => t.isSubagent).map(t => t.role))
      .toEqual(beforeTurns.filter(t => t.isSubagent).map(t => t.role));
    // 直载列多重集全等
    expectMultisetEqual(beforeTurns, afterTurns, 'Turn 直载列');

    const afterToolCalls = await snapToolCalls(s!.id);
    expectMultisetEqual(beforeToolCalls, afterToolCalls, 'ToolCall 直载列');

    // bridge 重推导：subagent 会话 + 派发关联恢复
    const bridges = await prisma.interactionBridge.findMany({ where: { sessionId: s!.id } });
    expect(bridges.some(b => b.subagentSessionId === SUB_ID)).toBe(true);
  });
});

describe('cannbay schema：⑤ cpx 捕获治理上传 → 再导入 —— meta 过治理且不覆盖 proxy 徽标', () => {
  it('producer:cpx 的 meta 过治理不改写，导入后徽标保留', async () => {
    const CPX_SID = 'schema-cpx-0003';
    const srcDir = path.join(TMP, 'cpx-src');
    fs.mkdirSync(srcDir, { recursive: true });
    const srcFile = path.join(srcDir, `${CPX_SID}.jsonl`);
    fs.writeFileSync(srcFile, [
      JSON.stringify({ type: 'user', message: { role: 'user', content: '跑一下' }, timestamp: '2026-08-19T07:00:00.000Z', source: 'claude-proxy' }),
      JSON.stringify({
        type: 'assistant',
        message: { id: 'm_cpx', role: 'assistant', content: [{ type: 'text', text: '完成' }], model: 'm', usage: { input_tokens: 5, output_tokens: 2 } },
        timestamp: '2026-08-19T07:00:03.000Z', source: 'claude-proxy', duration_ms: 3000,
        x_cannbay: { schema: 'cc-wire-round', version: 1, data: { roundIndex: 0, protocol: 'anthropic', latencyMs: 3000, ccVersion: '2.1.234.467', status: 200 } },
      }),
    ].join('\n') + '\n');
    fs.writeFileSync(srcFile.replace(/\.jsonl$/, '.meta.json'), JSON.stringify({
      x_cannbay: { schema: 'cc-session-meta', version: 1, data: { producer: 'cpx', framework: 'claude-code', protocol: 'anthropic', ccVersion: '2.1.234.467', sid: CPX_SID } },
    }, null, 2));
    await prisma.session.deleteMany({ where: { taskId: CPX_SID } });
    await prisma.session.create({ data: { taskId: CPX_SID, framework: 'claude-code', sourcePath: srcFile } });
    // 完整治理上传路径（staging 含 <sid>.meta.json，governStagedSession 递归扫描）
    const up = await cannbay2.uploadCannbay2Session(prisma, CPX_SID, 'cpx badge IT');
    expect(up.unchanged).toBe(false);
    await prisma.session.deleteMany({ where: { taskId: CPX_SID } });
    await cannbay2.importCannbay2Session(CPX_SID, prisma);
    const s = await prisma.session.findFirst({ where: { taskId: CPX_SID } });
    expect(s).toBeTruthy();
    // 徽标：version 以 -proxy 结尾（meta 的纯 ccVersion 2.1.234.467 不得覆盖）
    expect(s!.version?.endsWith('-proxy')).toBe(true);
    expect(s!.version).toContain('2.1.234.467');
    await prisma.session.deleteMany({ where: { taskId: CPX_SID } });
  });
});

describe('cannbay schema：④ 混排容错（native / legacy-proxy / 纯 x_cannbay 同文件）', () => {
  it('三通道各行其是、互不污染', async () => {
    const MIXED_SID = 'schema-mixed-0002';
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'schema-mixed-'));
    const mainFile = path.join(dir, `${MIXED_SID}.jsonl`);
    fs.writeFileSync(mainFile, [
      JSON.stringify({ type: 'user', message: { role: 'user', content: 'native 轮' }, timestamp: '2026-08-19T06:00:00.000Z' }),
      // native assistant：无任何声明 → 时间戳差算 latency
      JSON.stringify({ type: 'assistant', message: { id: 'm_native', role: 'assistant', content: [{ type: 'text', text: 'native 回复' }], model: 'm', usage: { input_tokens: 1, output_tokens: 1 } }, timestamp: '2026-08-19T06:00:10.000Z' }),
      JSON.stringify({ type: 'user', message: { role: 'user', content: 'legacy 轮' }, timestamp: '2026-08-19T06:01:00.000Z' }),
      // legacy assistant：source + duration_ms
      JSON.stringify({ type: 'assistant', message: { id: 'm_legacy', role: 'assistant', content: [{ type: 'text', text: 'legacy 回复' }], model: 'm', usage: { input_tokens: 1, output_tokens: 1 } }, timestamp: '2026-08-19T06:01:05.000Z', source: 'claude-proxy', duration_ms: 1500 }),
      JSON.stringify({ type: 'user', message: { role: 'user', content: 'x_cannbay 轮' }, timestamp: '2026-08-19T06:02:00.000Z' }),
      // 纯 x_cannbay assistant（收敛后的目标形态）：无 source/duration_ms
      JSON.stringify({
        type: 'assistant',
        message: { id: 'm_xb', role: 'assistant', content: [{ type: 'text', text: 'xb 回复' }], model: 'm', usage: { input_tokens: 1, output_tokens: 1 } },
        timestamp: '2026-08-19T06:02:08.000Z',
        x_cannbay: { schema: 'cc-wire-round', version: 1, data: { roundIndex: 0, protocol: 'anthropic', latencyMs: 2222, stopReason: 'end_turn' } },
      }),
    ].join('\n') + '\n');

    await importSession(mainFile, MIXED_SID, prisma, mainFile, 'claude-jsonl');
    const s = await prisma.session.findFirst({ where: { taskId: MIXED_SID } });
    expect(s).toBeTruthy();
    const turns = await prisma.turn.findMany({ where: { sessionId: s!.id, role: 'assistant' }, orderBy: { turnIndex: 'asc' } });
    expect(turns.length).toBe(3);
    const [native, legacy, xb] = turns;
    expect(native.latencyMs).toBe(0);                   // native 单行无 latency 来源（这正是 duration_ms/x_cannbay 通道存在的理由）
    expect(legacy.latencyMs).toBe(1500);               // duration_ms 通道
    expect(xb.latencyMs).toBe(2222);                    // x_cannbay 通道（未受时间戳差污染）
    expect(xb.finishReason).toBe('end_turn');
    fs.rmSync(dir, { recursive: true, force: true });
  });
});

// ⑥ opencode-producer 形状锁 + version 门禁（P0-1/P0-2 落地验证）
// opencode-emitter 此前完全脱约（零 x_cannbay），本测试锁住双轨补齐 +
// 读方 version 门禁（未知版本整块跳过 → 回退 legacy）。
describe('cannbay schema：⑥ opencode-producer 形状 + version 门禁', () => {
  it('opencode 捕获形状：cc-wire-round(cc-wire-round/openai) + cc-wire-input + cc-session-meta + cc-subagent-meta 双写', async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'oc-shape-'));
    const SID = 'oc-shape';
    const mainFile = path.join(dir, `${SID}.jsonl`);
    const subDir = path.join(dir, SID, 'subagents');
    fs.mkdirSync(subDir, { recursive: true });
    fs.writeFileSync(mainFile, [
      JSON.stringify({ type: 'user', message: { role: 'user', content: '开始' }, timestamp: '...', source: 'opencode-proxy',
        x_cannbay: { schema: 'cc-wire-input', version: 1, data: { roundIndex: 0, kind: 'user' } } }),
      JSON.stringify({ type: 'assistant', message: { role: 'assistant', id: 'a1', content: [{ type: 'text', text: 'ok' }], model: 'glm-5.2', usage: { input_tokens: 1, output_tokens: 1 } }, timestamp: '...', source: 'opencode-proxy',
        duration_ms: 800, stopReason: 'stop', ttftMs: 50, system: 'You are opencode', tools: [{ name: 'bash', description: '' }],
        x_cannbay: { schema: 'cc-wire-round', version: 1, data: { roundIndex: 0, protocol: 'openai', latencyMs: 800, ttftMs: 50, stopReason: 'stop', status: 200, requestParams: { temperature: 0.7, maxTokens: 4096, model: 'glm-5.2' }, system: 'You are opencode', tools: [{ name: 'bash', description: '' }] } } }),
    ].join('\n') + '\n');
    fs.writeFileSync(path.join(dir, `${SID}.meta.json`), JSON.stringify({ x_cannbay: { schema: 'cc-session-meta', version: 1, data: { producer: 'cpx', framework: 'opencode', protocol: 'openai', sid: SID } } }));
    fs.writeFileSync(path.join(subDir, 'sub-a.jsonl'), JSON.stringify({ type: 'user', message: { role: 'user', content: '子任务' }, source: 'opencode-proxy', x_cannbay: { schema: 'cc-wire-input', version: 1, data: { roundIndex: 0, kind: 'user' } } }) + '\n');
    fs.writeFileSync(path.join(subDir, 'sub-a.meta.json'), JSON.stringify({ toolUseId: 't1', name: '任务', agentType: 'developer', x_cannbay: { schema: 'cc-subagent-meta', version: 1, data: { toolUseId: 't1', name: '任务', agentType: 'developer', subagentSessionId: 'sub-a' } } }));

    await importSession(mainFile, SID, prisma, mainFile, 'claude-jsonl');
    const s = await prisma.session.findFirst({ where: { taskId: SID } });
    expect(s?.framework).toBe('opencode');   // cc-session-meta 权威源生效
    expect(s?.version).toBe('opencode-proxy');
    const turns = await prisma.turn.findMany({ where: { sessionId: s!.id, role: 'assistant' }, orderBy: { turnIndex: 'asc' } });
    expect(turns.length).toBe(1);
    expect(turns[0].latencyMs).toBe(800);    // x_cannbay.latencyMs 通道（双写一致）
    expect(turns[0].finishReason).toBe('stop');
    fs.rmSync(dir, { recursive: true, force: true });
    if (s) await prisma.session.delete({ where: { id: s.id } }).catch(() => {});
  });

  it('version 门禁：未知 version 整块跳过 → 回退 legacy 通道', async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'oc-vgt-'));
    const SID = 'oc-vgt';
    const mainFile = path.join(dir, `${SID}.jsonl`);
    fs.writeFileSync(mainFile, [
      JSON.stringify({ type: 'user', message: { role: 'user', content: 'x' }, timestamp: '...', source: 'opencode-proxy' }),
      // v99 声明：读方应整块跳过 x_cannbay，回退 duration_ms=1500 (legacy)
      JSON.stringify({ type: 'assistant', message: { role: 'assistant', id: 'a1', content: [{ type: 'text', text: 'r' }], model: 'glm-5.2', usage: { input_tokens: 1, output_tokens: 1 } }, timestamp: '...', source: 'opencode-proxy',
        duration_ms: 1500,
        x_cannbay: { schema: 'cc-wire-round', version: 99, data: { roundIndex: 0, latencyMs: 9999, stopReason: 'SHOULD-NOT-LEAK' } } }),
    ].join('\n') + '\n');
    await importSession(mainFile, SID, prisma, mainFile, 'claude-jsonl');
    const s = await prisma.session.findFirst({ where: { taskId: SID } });
    const turns = await prisma.turn.findMany({ where: { sessionId: s!.id, role: 'assistant' } });
    expect(turns[0].latencyMs).toBe(1500);               // legacy duration_ms 兜底生效
    expect(turns[0].finishReason).not.toBe('SHOULD-NOT-LEAK'); // v99 数据没泄漏
    fs.rmSync(dir, { recursive: true, force: true });
    if (s) await prisma.session.delete({ where: { id: s.id } }).catch(() => {});
  });
});
