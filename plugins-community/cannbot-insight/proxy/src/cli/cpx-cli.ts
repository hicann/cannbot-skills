// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { sessionFilePath, sidsFilePath } from '../writer.ts';
import { redactString } from '../redactor.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const START_SH = path.join(REPO_ROOT, 'start.sh');
const PROXY_SERVER = path.join(REPO_ROOT, 'proxy', 'src', 'server.ts');
const PROXY_DIR = process.env.CANNBOT_PROXY_DIR ?? path.join(os.homedir(), '.cannbot-insight', 'proxy');
const CPX_CONFIG_FILE = path.join(path.dirname(PROXY_DIR), 'cpx-config.json');

interface CpxConfig {
  /** 注入压缩：记录 jsonl 时把重复的注册表/skills 注入替换为 [已压缩] 标记。默认关闭。 */
  dedupInjection: boolean;
}

function loadCpxConfig(): CpxConfig {
  try {
    const raw = JSON.parse(fs.readFileSync(CPX_CONFIG_FILE, 'utf-8'));
    return { dedupInjection: raw.dedupInjection === true };
  } catch {
    return { dedupInjection: false };
  }
}

function saveCpxConfig(cfg: CpxConfig): void {
  fs.mkdirSync(path.dirname(CPX_CONFIG_FILE), { recursive: true });
  fs.writeFileSync(CPX_CONFIG_FILE, JSON.stringify(cfg, null, 2) + '\n');
}

function runConfig(args: string[]): void {
  const cfg = loadCpxConfig();
  if (args[0] === 'dedup' && (args[1] === 'on' || args[1] === 'off')) {
    cfg.dedupInjection = args[1] === 'on';
    saveCpxConfig(cfg);
    console.log(`dedup injection : ${cfg.dedupInjection ? 'on' : 'off'}  (已写入 ${CPX_CONFIG_FILE})`);
    return;
  }
  if (args.length === 0) {
    console.log(`config file     : ${CPX_CONFIG_FILE}`);
    console.log(`dedup injection : ${cfg.dedupInjection ? 'on' : 'off'}   (cpx config dedup on|off 切换，热生效；压缩仅作用于捕获记录，不改动转发)`);
    return;
  }
  console.error('用法: cpx config [dedup on|off]');
  process.exit(2);
}
const CLAUDE_SETTINGS = path.join(os.homedir(), '.claude', 'settings.json');
const INSIGHT_PORT = 21025;
const INSIGHT_BASE = `http://localhost:${INSIGHT_PORT}`;

const USAGE = `cpx — agent↔模型明文捕获代理编排器

用法：
  cpx <agent-cmd> [args...]              例: cpx claude
  cpx --agent <claude|opencode|openai|generic> -- <cmd> [args]
  cpx status [--kill [--all]]           查看状态；--kill 清理无 sid 的孤儿 proxy，--all 连活跃会话一起清
  cpx config [dedup on|off]             查看/配置；dedup=注入压缩（默认 off，热生效，只压缩捕获记录不改转发）
  cpx --help | -h                        本帮助

在原有 claude / opencode 命令前加 cpx，其他使用完全无变化。agent 退出后自动导入 cannbot-insight 并开浏览器。`;

type AgentProfile = 'claude' | 'opencode' | 'openai' | 'generic';

function log(msg: string): void { console.error(`[cpx-cli] ${msg}`); }

interface ClaudeSettings {
  upstream: string;
  apiKey: string | null;
  model: string | null;
}

function readClaudeSettings(): ClaudeSettings {
  const fallback: ClaudeSettings = {
    upstream: 'https://api.anthropic.com',
    apiKey: process.env.ANTHROPIC_API_KEY ?? null,
    model: process.env.ANTHROPIC_MODEL ?? null,
  };
  try {
    const raw = fs.readFileSync(CLAUDE_SETTINGS, 'utf-8');
    const env = (JSON.parse(raw).env ?? {}) as Record<string, string>;
    return {
      upstream: env.ANTHROPIC_BASE_URL ?? fallback.upstream,
      apiKey: env.ANTHROPIC_API_KEY ?? fallback.apiKey,
      model: env.ANTHROPIC_MODEL ?? fallback.model,
    };
  } catch {
    return fallback;
  }
}

function parseArgs(argv: string[]): { profile: AgentProfile; agentCmd: string; agentArgs: string[] } {
  const args = argv.slice(2);
  let profile: AgentProfile | null = null;
  let rest: string[] = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--agent' && args[i + 1]) { profile = args[i + 1] as AgentProfile; i++; continue; }
    rest.push(args[i]);
  }
  if (rest[0] === '--') rest = rest.slice(1);
  const agentCmd = rest[0] ?? '';
  const agentArgs = rest.slice(1);
  if (!profile) {
    const base = path.basename(agentCmd);
    if (base === 'claude' || base === 'claude-code') profile = 'claude';
    else if (base === 'opencode') profile = 'opencode';
    else profile = 'generic';
  }
  return { profile: profile!, agentCmd, agentArgs };
}

async function waitReady(url: string, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(url, { signal: AbortSignal.timeout(2000) });
      if (r.ok || r.status < 500) return true;
    } catch { /* keep polling */ }
    await new Promise(r => setTimeout(r, 500));
  }
  return false;
}

async function ensureInsight(): Promise<void> {
  if (await waitReady(`${INSIGHT_BASE}/api/observe/data?pageSize=1`, 2000)) {
    log(`cannbot-insight already running at ${INSIGHT_BASE}`);
    return;
  }
  log(`starting cannbot-insight via start.sh -k ...`);
  const child = spawn('bash', [START_SH, '-k'], {
    detached: true, stdio: 'ignore', cwd: REPO_ROOT,
    env: { ...process.env, NEXT_PUBLIC_SHOW_ADVANCED_TABS: process.env.NEXT_PUBLIC_SHOW_ADVANCED_TABS ?? '' },
  });
  child.unref();
  if (!(await waitReady(`${INSIGHT_BASE}/api/observe/data?pageSize=1`, 90000))) {
    throw new Error(`cannbot-insight did not become ready at ${INSIGHT_BASE} within 90s`);
  }
  log(`cannbot-insight ready at ${INSIGHT_BASE}`);
}

interface SessionProxy {
  port: number;
  child: ChildProcess;
  stop: () => void;
}

function startSessionProxy(sid: string, anthropicUpstream: string, openaiUpstream: string, providerUpstreams: Record<string, string>): Promise<SessionProxy> {
  fs.mkdirSync(PROXY_DIR, { recursive: true });
  return new Promise((resolve, reject) => {
    const child = spawn('npx', ['tsx', PROXY_SERVER], {
      stdio: ['ignore', 'pipe', 'pipe'],
      cwd: REPO_ROOT,
      detached: true,
      env: {
        ...process.env,
        CANNBOT_PROXY_PORT: '0',
        CANNBOT_PROXY_SESSION_ID: sid,
        CANNBOT_PROXY_ANTHROPIC_UPSTREAM: anthropicUpstream,
        CANNBOT_PROXY_OPENAI_UPSTREAM: openaiUpstream,
        CANNBOT_PROXY_PROVIDER_UPSTREAMS: JSON.stringify(providerUpstreams),
      },
    });
    let port = 0;
    const onPort = (p: number) => {
      if (port || !p) return;
      port = p;
      const stop = () => { try { child.kill('SIGTERM') ; } catch { /* */ } };
      resolve({ port, child, stop });
    };
    child.stdout?.on('data', (d: Buffer) => {
      const m = d.toString().match(/http:\/\/127\.0\.0\.1:(\d+)/);
      if (m) onPort(parseInt(m[1]));
    });
    child.stderr?.on('data', (d: Buffer) => process.stderr.write(d));
    const timer = setTimeout(() => {
      if (!port) reject(new Error('cannbot-proxy did not report a port within 30s'));
    }, 30000);
    child.on('exit', () => { clearTimeout(timer); if (!port) reject(new Error('cannbot-proxy exited before reporting a port')); });
  });
}

interface LaunchPlan {
  env: Record<string, string>;
  extraArgs: string[];
}

// Parse providerId from `-m <provider>/<model>` (opencode's model format) in
// agentArgs. Used to inject baseURL override for the right provider via
// OPENCODE_CONFIG_CONTENT. Returns null if not found (caller then resolves all
// providers from auth.json instead — see resolveOpencodeProviderIds).
function parseOpencodeProvider(agentArgs: string[]): string | null {
  for (let i = 0; i < agentArgs.length; i++) {
    if ((agentArgs[i] === '-m' || agentArgs[i] === '--model') && agentArgs[i + 1]) {
      const m = agentArgs[i + 1];
      const slash = m.indexOf('/');
      return slash > 0 ? m.slice(0, slash) : null;
    }
    // `-m=provider/model` form
    const eq = agentArgs[i].match(/^-m=(.+)$/);
    if (eq) {
      const m = eq[1];
      const slash = m.indexOf('/');
      return slash > 0 ? m.slice(0, slash) : null;
    }
  }
  return null;
}

// opencode stores provider credentials in auth.json (XDG data dir). Its keys
// are the provider ids opencode knows (alibaba-cn, zhipuai-coding-plan, ...).
// opencode's provider baseURLs are BUILT-IN to the binary — not in auth.json,
// not in opencode.json — so the only way to redirect opencode to the proxy is
// OPENCODE_CONFIG_CONTENT, and cpx must know which provider id(s) to override.
function opencodeDataDir(): string {
  const xdg = process.env.XDG_DATA_HOME;
  return xdg ? path.join(xdg, 'opencode') : path.join(os.homedir(), '.local', 'share', 'opencode');
}

// claude 原生转录：~/.claude/projects/<cwd-slug>/<session-id>.jsonl。捕获
// sid 即 claude 的真实 session id（header 路由），文件名精确同名；slug
// 规则 = cwd 非字母数字全部转 '-'（claude-code 约定）。候选不存在时全
// projects 目录扫一遍兜底（agent 内部切换过 cwd 的场景）。
function claudeNativeLogPath(sid: string): string | null {
  const base = path.join(os.homedir(), '.claude', 'projects');
  const slug = process.cwd().replace(/[^a-zA-Z0-9]/g, '-');
  const candidate = path.join(base, slug, `${sid}.jsonl`);
  if (fs.existsSync(candidate)) return candidate;
  try {
    for (const d of fs.readdirSync(base)) {
      const p = path.join(base, d, `${sid}.jsonl`);
      if (fs.existsSync(p)) return p;
    }
  } catch { /* */ }
  return null;
}

// opencode ≥1.17 会话统一存单一 sqlite（opencode.db），无 per-session
// 转录文件；运行日志在 log/ 下。
function opencodeNativePaths(): string {
  const dir = opencodeDataDir();
  return `${path.join(dir, 'opencode.db')} · ${path.join(dir, 'log', 'opencode.log')}`;
}

// Provider ids with stored credentials (i.e. the ones opencode can actually
// call). Used to inject baseURL override for every one of them when -m is not
// given — opencode picks its default at runtime and cpx can't know which.
function readOpencodeProviders(): string[] {
  try {
    const raw = fs.readFileSync(path.join(opencodeDataDir(), 'auth.json'), 'utf-8');
    const obj = JSON.parse(raw) as Record<string, unknown>;
    if (!obj || typeof obj !== 'object') return [];
    return Object.keys(obj).filter(k => obj[k] && typeof obj[k] === 'object');
  } catch {
    return [];
  }
}

// Resolve the opencode provider ids to override via OPENCODE_CONFIG_CONTENT:
// -m <provider>/<model> → just that one (precise); otherwise all of auth.json's
// providers (interactive `cpx opencode`). Empty = misconfiguration.
function resolveOpencodeProviderIds(agentArgs: string[]): string[] {
  const fromArg = parseOpencodeProvider(agentArgs);
  if (fromArg) return [fromArg];
  return readOpencodeProviders();
}

// Resolve the opencode binary path (follow symlinks to the real .exe). opencode
// is on PATH (cpx spawns it as `opencode`); we need the real file to scan for
// built-in provider baseURLs. Returns null if not resolvable.
function opencodeBinaryPath(): string | null {
  try {
    const r = spawnSync('sh', ['-c', 'command -v opencode'], { encoding: 'utf-8' });
    const p = (r.stdout || '').trim();
    if (!p || !fs.existsSync(p)) return null;
    return fs.realpathSync(p);
  } catch { return null; }
}

// Discover each provider's real upstream baseURL by scanning the opencode
// binary with `strings`. opencode's provider registry embeds, per provider:
//   id:"<pid>",env:[...],npm:"...",api:"<baseURL>"
// The api URL is the OpenAI-compatible base opencode would call WITHOUT the
// proxy. This is READ from the user's own installed binary (not fabricated),
// so cpx can forward to the correct upstream per provider — fully transparent,
// no CANNBOT_PROXY_OPENAI_UPSTREAM env needed for known providers. Returns
// {providerId: baseURL} for the requested ids that were found in the binary.
function discoverProviderUpstreams(providerIds: string[]): Record<string, string> {
  const bin = opencodeBinaryPath();
  if (!bin || providerIds.length === 0) return {};
  // One `strings` pass (~1s for a 167MB binary); grep -o extracts every
  // provider block as its own line, then we pair id↔api per line.
  const quotedBin = `'${bin.replace(/'/g, `'\\''`)}'`;
  const r = spawnSync('sh', ['-c',
    `strings -n 8 ${quotedBin} | grep -oaE 'id:"[^"]*",env:\\[[^]]*\\],npm:"[^"]*",api:"[^"]*"'`,
  ], { encoding: 'utf-8', maxBuffer: 64 * 1024 * 1024 });
  const all: Record<string, string> = {};
  for (const line of (r.stdout || '').split('\n')) {
    const idM = line.match(/id:"([^"]+)"/);
    const apiM = line.match(/api:"([^"]+)"/);
    if (idM && apiM) all[idM[1]] = apiM[1];
  }
  const out: Record<string, string> = {};
  for (const id of providerIds) if (all[id]) out[id] = all[id];
  return out;
}

function buildLaunch(profile: AgentProfile, proxyPort: number, claude: ClaudeSettings, agentArgs: string[] = []): LaunchPlan {
  // Per-session port: no sid encoded in URL path. The proxy pins sid via env,
  // so base_url is a clean host (survives SDK URL normalization).
  const proxyUrl = `http://127.0.0.1:${proxyPort}`;
  switch (profile) {
    case 'claude': {
      // Process-scoped override via `claude --settings`. Does NOT touch
      // ~/.claude/settings.json — when this process exits, the override is gone,
      // so a subsequent plain `claude` is unaffected even if cpx-cli crashes.
      const settingsEnv: Record<string, string> = {
        ANTHROPIC_BASE_URL: proxyUrl,
      };
      if (claude.apiKey) settingsEnv.ANTHROPIC_API_KEY = claude.apiKey;
      if (claude.model) settingsEnv.ANTHROPIC_MODEL = claude.model;
      return {
        env: {},
        extraArgs: ['--settings', JSON.stringify({ env: settingsEnv })],
      };
    }
    case 'opencode': {
      // Process-scoped override via OPENCODE_CONFIG_CONTENT (inline runtime
      // config, precedence tier 6 — above project/global config). Symmetric to
      // claude's `--settings`: only this process is affected, exit = clean,
      // ~/.config/opencode/ is never touched.
      //
      // opencode's provider baseURLs are BUILT-IN (e.g. alibaba-cn → dashscope),
      // NOT read from OPENAI_BASE_URL — verified via `opencode debug config`.
      // So we override baseURL per provider via OPENCODE_CONFIG_CONTENT.
      //
      // Per-provider path-prefix routing: baseURL is set to
      // http://<proxy>/<providerId>/ (trailing slash). opencode/ai-sdk appends
      // the model path AFTER the baseURL's path (verified: yields
      // POST /<providerId>/chat/completions), so the prefix survives SDK URL
      // normalization and the proxy routes each request to that provider's real
      // upstream (discovered from the opencode binary). This is fully
      // transparent — no CANNBOT_PROXY_OPENAI_UPSTREAM needed — and supports
      // multiple providers with different upstreams in one session.
      // Caller validates the list is non-empty before starting the proxy.
      const providerIds = resolveOpencodeProviderIds(agentArgs);
      const provider: Record<string, { options: { baseURL: string } }> = {};
      for (const id of providerIds) provider[id] = { options: { baseURL: `${proxyUrl}/${id}/` } };
      return { env: { OPENCODE_CONFIG_CONTENT: JSON.stringify({ provider }) }, extraArgs: [] };
    }
    case 'openai':
    case 'generic':
      return { env: { OPENAI_BASE_URL: proxyUrl }, extraArgs: [] };
  }
}

function openBrowser(url: string): void {
  try {
    if (fs.existsSync('/proc/version') && fs.readFileSync('/proc/version', 'utf-8').toLowerCase().includes('microsoft')
        && fs.existsSync('/mnt/c/Windows/System32/cmd.exe')) {
      spawnSync('/mnt/c/Windows/System32/cmd.exe', ['/c', 'start', url], { stdio: 'ignore' });
    } else if (process.platform === 'darwin') {
      spawnSync('open', [url], { stdio: 'ignore' });
    } else {
      spawnSync('xdg-open', [url], { stdio: 'ignore' });
    }
  } catch { /* best-effort */ }
}

async function triggerIngest(sid: string, jsonlPath: string): Promise<{ imported: boolean; query: string | null } | null> {
  try {
    const r = await fetch(`${INSIGHT_BASE}/api/ingest/import-file`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ source: 'claude-jsonl', sessionId: sid, filePath: jsonlPath }),
      signal: AbortSignal.timeout(120000),
    });
    if (!r.ok) { log(`ingest failed: HTTP ${r.status} ${await r.text()}`); return null; }
    return (await r.json()) as { imported: boolean; query: string | null };
  } catch (e) {
    log(`ingest error: ${String(e)}`);
    return null;
  }
}

const PROXY_SERVER_REL = path.join('proxy', 'src', 'server.ts');
const CLK_TCK = 100;

interface ProxyProc {
  pid: number; ppid: number | null; sid: string | null;
  upstream: string | null; ageSec: number | null; port: number | null;
}

function readProcFile(pid: number, name: string): string | null {
  try { return fs.readFileSync(`/proc/${pid}/${name}`, 'utf-8'); } catch { return null; }
}

function parseEnviron(s: string): Record<string, string> {
  const e: Record<string, string> = {};
  for (const kv of s.split('\0')) {
    const i = kv.indexOf('=');
    if (i > 0) e[kv.slice(0, i)] = kv.slice(i + 1);
  }
  return e;
}

// pid → listening port on 127.0.0.1, via `ss` (best-effort; absent → empty map).
function readListenPorts(): Map<number, number> {
  const m = new Map<number, number>();
  let out = '';
  try { out = spawnSync('ss', ['-ltnp'], { encoding: 'utf-8' }).stdout ?? ''; } catch { return m; }
  if (!out) return m;
  for (const line of out.split('\n')) {
    const pm = line.match(/pid=(\d+)/);
    const ptm = line.match(/127\.0\.0\.1:(\d+)/);
    if (pm && ptm) {
      const pid = parseInt(pm[1]); const port = parseInt(ptm[1]);
      if (pid && port) m.set(pid, port);
    }
  }
  return m;
}

function findProxyProcs(): ProxyProc[] {
  let pids: number[] = [];
  try { pids = fs.readdirSync('/proc').filter(d => /^\d+$/.test(d)).map(Number); } catch { return []; }
  const portMap = readListenPorts();
  let uptime: number | null = null;
  try { uptime = parseFloat(fs.readFileSync('/proc/uptime', 'utf-8').split(/\s+/)[0]); } catch { /* */ }
  const out: ProxyProc[] = [];
  for (const pid of pids) {
    const cmd = readProcFile(pid, 'cmdline');
    if (!cmd) continue;
    if (!cmd.replace(/\0/g, ' ').includes(PROXY_SERVER_REL)) continue;
    const env = parseEnviron(readProcFile(pid, 'environ') ?? '');
    let ppid: number | null = null; let ageSec: number | null = null;
    const stat = readProcFile(pid, 'stat');
    if (stat) {
      const after = stat.slice(stat.lastIndexOf(')') + 2).split(/\s+/);
      ppid = parseInt(after[1]) || null;
      const st = parseInt(after[19]);
      if (uptime && Number.isFinite(st)) ageSec = Math.max(0, uptime - st / CLK_TCK);
    }
    out.push({
      pid, ppid,
      sid: env.CANNBOT_PROXY_SESSION_ID ?? null,
      upstream: env.CANNBOT_PROXY_ANTHROPIC_UPSTREAM ?? null,
      ageSec, port: portMap.get(pid) ?? null,
    });
  }
  return out.sort((a, b) => a.pid - b.pid);
}

// Dedupe the (npx-parent → node-child) pair: the npx wrapper has no port and
// is the parent of the actual server child. Keep the child (port bearer);
// drop port-less procs that are an ancestor of another matched proc.
function dedupePairs(procs: ProxyProc[]): ProxyProc[] {
  return procs.filter(p => !(p.port == null && procs.some(q => q.ppid === p.pid)));
}

async function probeUrl(url: string, ms: number): Promise<boolean> {
  try { const r = await fetch(url, { signal: AbortSignal.timeout(ms) }); return r.ok || r.status < 500; } catch { return false; }
}

function fmtAge(s: number | null): string {
  if (s == null || !Number.isFinite(s)) return '-';
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h${Math.floor((s % 3600) / 60)}m`;
}

function isAlive(pid: number): boolean {
  try { process.kill(pid, 0); return true; } catch { return false; }
}

// SIGTERM first, grace period, then SIGKILL stragglers.
async function killProcs(pids: number[]): Promise<void> {
  for (const pid of pids) { if (isAlive(pid)) { try { process.kill(pid, 'SIGTERM'); } catch { /* */ } } }
  await new Promise(r => setTimeout(r, 1500));
  for (const pid of pids) { if (isAlive(pid)) { try { process.kill(pid, 'SIGKILL'); } catch { /* */ } } }
}

function fmtSize(b: number): string {
  if (b < 1048576) return `${(b / 1024).toFixed(1)}KB`;
  return `${(b / 1048576).toFixed(1)}MB`;
}

async function runStatus(opts: { kill?: boolean; all?: boolean } = {}): Promise<void> {
  const proxies = dedupePairs(findProxyProcs());
  const insightUp = await probeUrl(`${INSIGHT_BASE}/api/observe/data?pageSize=1`, 1500);

  console.log(`cannbot-insight : ${insightUp ? `up   ${INSIGHT_BASE}` : 'down (will auto-start on next `cpx <agent>`)'}`);
  console.log(`proxy processes : ${proxies.length} running${proxies.length === 0 ? '  (none — proxy only lives during a cpx run)' : ''}`);
  for (const p of proxies) {
    const stale = p.sid == null ? '  [stale: no sid, likely orphaned]' : '';
    console.log(`  pid ${p.pid}  port ${p.port ?? '-'}  age ${fmtAge(p.ageSec)}${stale}`);
    console.log(`    sid ${p.sid ?? '-'}  upstream ${p.upstream ?? '-'}`);
  }

  if (opts.kill) {
    const targets = opts.all ? proxies : proxies.filter(p => p.sid == null);
    const spared = proxies.length - targets.length;
    if (!targets.length) {
      console.log(`\nkill: nothing to ${opts.all ? 'kill (no proxy procs)' : 'kill (no stale orphans)'}`);
    } else {
      const label = opts.all ? 'proxy process(es)' : 'stale orphan(s)';
      console.log(`\nkill: terminating ${targets.length} ${label} (spared ${spared} live session(s)) ...`);
      await killProcs(targets.map(t => t.pid));
      const remaining = dedupePairs(findProxyProcs());
      console.log(`  done. ${remaining.length} still running.`);
    }
  }

  console.log(`captures dir    : ${PROXY_DIR}`);
  console.log(`dedup injection : ${loadCpxConfig().dedupInjection ? 'on' : 'off'}  (cpx config dedup on|off)`);
  let caps: { f: string; size: number; mt: number }[] = [];
  try {
    caps = fs.readdirSync(PROXY_DIR)
      .filter(f => f.endsWith('.jsonl'))
      .map(f => { try { const st = fs.statSync(path.join(PROXY_DIR, f)); return { f, size: st.size, mt: st.mtimeMs }; } catch { return null; } })
      .filter((x): x is { f: string; size: number; mt: number } => !!x)
      .sort((a, b) => b.mt - a.mt)
      .slice(0, 8);
  } catch { console.log('  (dir missing)'); }
  if (caps.length) {
    const now = Date.now() / 1000;
    console.log('recent captures :');
    let savedTotal = 0;
    let dedupTotal = 0;
    for (const c of caps) {
      const { savedChars, dedupedCount } = dedupStats(path.join(PROXY_DIR, c.f));
      savedTotal += savedChars;
      dedupTotal += dedupedCount;
      const ratio = savedChars / (c.size + savedChars);
      const comp = dedupedCount > 0
        ? `  dedup ×${dedupedCount}  压缩率 ${(ratio * 100).toFixed(1)}%`
        : '';
      console.log(`  ${c.f}   ${fmtSize(c.size).padStart(9)}   ${fmtAge(now - c.mt / 1000)} ago${comp}`);
    }
    if (dedupTotal > 0) {
      const allSize = caps.reduce((s, c) => s + c.size, 0);
      console.log(`  注入去重合计    : ${dedupTotal} 份重复注入已压缩，节省约 ${fmtSize(savedTotal)}（压缩率 ${(savedTotal / (allSize + savedTotal) * 100).toFixed(1)}%）`);
    }
  }
}

// Sum dedup marker lines in a capture: {deduped:true, originalChars} lines
// record how much repeated injection content was replaced by a marker.
function dedupStats(file: string): { savedChars: number; dedupedCount: number } {
  let savedChars = 0;
  let dedupedCount = 0;
  try {
    const raw = fs.readFileSync(file, 'utf8').split('\n');
    for (const line of raw) {
      if (!line.includes('"deduped":true')) continue;
      try {
        const j = JSON.parse(line);
        if (j.deduped === true && typeof j.originalChars === 'number') {
          savedChars += j.originalChars;
          dedupedCount++;
        }
      } catch { /* skip malformed */ }
    }
  } catch { /* unreadable */ }
  return { savedChars, dedupedCount };
}

async function main(): Promise<void> {
  const sub = process.argv[2];
  if (sub === '-h' || sub === '--help' || sub === 'help') { console.log(USAGE); process.exit(0); }
  if (sub === 'status') {
    const flags = process.argv.slice(3);
    await runStatus({ kill: flags.includes('--kill'), all: flags.includes('--all') });
    return;
  }
  if (sub === 'config') {
    runConfig(process.argv.slice(3));
    return;
  }
  const { profile, agentCmd, agentArgs } = parseArgs(process.argv);
  if (!agentCmd) {
    console.error(USAGE);
    process.exit(2);
  }

  await ensureInsight();

  const claude = readClaudeSettings();
  // opencode's real upstream (OpenAI-protocol) — best-effort from env, since
  // opencode's built-in provider baseURLs live in opencode internals, not the
  // user's opencode.json. Override via CANNBOT_PROXY_OPENAI_UPSTREAM if wrong.
  const openaiUpstream = process.env.CANNBOT_PROXY_OPENAI_UPSTREAM ?? 'https://api.openai.com';

  // Validate opencode capture preconditions BEFORE starting the proxy, so a
  // missing provider config exits cleanly instead of starting an orphan proxy
  // that captures nothing (the original "0 capture file" bug). opencode ignores
  // OPENAI_BASE_URL, so without at least one provider to override via
  // OPENCODE_CONFIG_CONTENT, capture is impossible — fail loudly, don't silently
  // fall back to a no-op env override.
  if (profile === 'opencode' && resolveOpencodeProviderIds(agentArgs).length === 0) {
    log('opencode: no provider to override for capture.');
    log('  fix: pass `-m <provider>/<model>` (e.g. -m alibaba-cn/glm-5.2),');
    log('        or log into a provider first (`opencode providers login`) so cpx');
    log('        can read its id from auth.json and inject the baseURL override.');
    log('  (opencode ignores OPENAI_BASE_URL, so cpx cannot capture without an');
    log('   explicit OPENCODE_CONFIG_CONTENT provider override.)');
    process.exit(2);
  }

  // Discover each opencode provider's REAL upstream baseURL by scanning the
  // opencode binary (provider registry embeds api:"<url>"). Forwarded to the
  // proxy as a per-provider map so it routes /<providerId>/… to the right host —
  // fully transparent, no CANNBOT_PROXY_OPENAI_UPSTREAM needed for known providers.
  const opencodeProviderIds = profile === 'opencode' ? resolveOpencodeProviderIds(agentArgs) : [];
  const providerUpstreams = profile === 'opencode' ? discoverProviderUpstreams(opencodeProviderIds) : {};

  const sid = randomUUID();
  const sessionProxy = await startSessionProxy(sid, claude.upstream, openaiUpstream, providerUpstreams);
  const jsonlPath = sessionFilePath(sid);
  const { env, extraArgs } = buildLaunch(profile, sessionProxy.port, claude, agentArgs);

  const displayUpstream = profile === 'opencode' || profile === 'openai' || profile === 'generic'
    ? openaiUpstream
    : claude.upstream;

  // Ensure the per-session proxy dies with cpx-cli, even on signals. Kill the
  // whole process GROUP (npx → tsx → node server.ts + esbuild grandchild):
  // SIGTERM to the npx wrapper alone leaves the node grandchild alive — it
  // inherits the stdout pipe fd, keeping cpx's event loop from exiting (the
  // "no capture file produced — nothing to import" hang) and accumulating as
  // orphan proxy processes.
  // SIGINT path: SIGTERM the group first, wait ~1s for the proxy's
  // handleStream client-disconnect catch to flush the in-flight record to
  // disk (emit runs even on partial stream), THEN SIGKILL stragglers. Direct
  // SIGKILL on Ctrl+C would lose the in-flight record (stream not yet emitted).
  const killProxyGroup = (signal: 'SIGTERM' | 'SIGKILL' = 'SIGKILL') => {
    const pid = sessionProxy.child.pid;
    if (pid) { try { process.kill(-pid, signal); } catch { /* group gone */ } }
    if (signal === 'SIGKILL') {
      try { sessionProxy.child.stdout?.destroy(); } catch { /* */ }
      try { sessionProxy.child.stderr?.destroy(); } catch { /* */ }
    }
  };
  const cleanup = () => {
    killProxyGroup('SIGTERM');
    // grace period: let the proxy flush in-flight stream (client disconnect
    // → catch → emit partial record). ~800ms covers a disk write + stream end.
    try { killProxyGroupSync(800); } catch { /* */ }
  };
  function killProxyGroupSync(graceMs: number) {
    const pid = sessionProxy.child.pid;
    if (!pid) return;
    const deadline = Date.now() + graceMs;
    while (Date.now() < deadline) {
      // busy-wait is fine — sub-second, and we're exiting anyway
    }
    try { process.kill(-pid, 'SIGKILL'); } catch { /* */ }
    try { sessionProxy.child.stdout?.destroy(); } catch { /* */ }
    try { sessionProxy.child.stderr?.destroy(); } catch { /* */ }
  }
  process.on('SIGINT', () => { cleanup(); process.exit(130); });
  process.on('SIGTERM', () => { cleanup(); process.exit(143); });
  process.on('exit', () => { try { killProxyGroup('SIGKILL'); } catch { /* */ } });

  log(`session: ${sid}`);
  log(`profile: ${profile}  upstream: ${displayUpstream}`);
  if (profile === 'opencode') {
    const ids = opencodeProviderIds;
    const src = parseOpencodeProvider(agentArgs) ? '-m' : 'auth.json';
    log(`  opencode providers: ${ids.join(', ')}  (from ${src}; baseURL → proxy via /<providerId>/ prefix)`);
    for (const id of ids) {
      const up = providerUpstreams[id];
      if (up) log(`    ${id} → ${up}`);
      else log(`    ${id} → (upstream not found in binary; will use default ${openaiUpstream} — set CANNBOT_PROXY_OPENAI_UPSTREAM if wrong)`);
    }
  }
  log(`proxy: http://127.0.0.1:${sessionProxy.port}  (per-session, sid pinned via env)`);
  // claude 会话的捕获文件按 claude 自己的 session id（请求头）命名 —— 与
  // claude 原生 jsonl 1:1 对应；cpx 钉定的占位文件仅在无 header 的协议
  // （opencode/generic）下承载记录。live watch 跟踪整个目录。
  log(`capture: ${PROXY_DIR}/cpx-<claude-session-id>.jsonl  (本次占位: cpx-${sid.slice(0, 8)}…)`);
  log(`  watch live: tail -F "${PROXY_DIR}"/*.jsonl`);
  const maskSecrets = (args: string[]) => args.map(a => redactString(a));
  log(`launching: ${agentCmd} ${maskSecrets([...extraArgs, ...agentArgs]).join(' ')}`);

  const child = spawn(agentCmd, [...extraArgs, ...agentArgs], {
    stdio: 'inherit',
    env: { ...process.env, ...env },
  });

  await new Promise<void>((resolve) => {
    child.on('exit', () => resolve());
    child.on('error', (e) => { log(`agent error: ${e.message}`); resolve(); });
  });

  cleanup();

  // The proxy routes each record by claude's REAL session id (request header),
  // which differs from cpx's pinned sid — and /resume can switch ids mid-run,
  // producing several capture files. The <pinned>.sids manifest (appended on
  // first emit per sid) is the authoritative list of files this run produced.
  const sidsFile = sidsFilePath(sid);
  // {sid, file}: sid is the REAL session id (manifest value, unprefixed) — it
  // becomes the insight taskId/URL; file is the cpx- prefixed capture path.
  const produced: Array<{ sid: string; file: string }> = [];
  try {
    const seen = new Set<string>();
    for (const s of fs.readFileSync(sidsFile, 'utf-8').split('\n').map(x => x.trim()).filter(Boolean)) {
      if (seen.has(s)) continue;
      seen.add(s);
      const p = sessionFilePath(s);
      if (fs.existsSync(p) && fs.statSync(p).size > 0) produced.push({ sid: s, file: p });
    }
  } catch { /* no manifest (no records emitted) */ }
  // Legacy/no-header protocols (opencode/generic) emit to the pinned file only.
  if (produced.length === 0 && fs.existsSync(jsonlPath) && fs.statSync(jsonlPath).size > 0) {
    produced.push({ sid, file: jsonlPath });
  }

  // The pinned placeholder file is eagerly created at proxy startup so the
  // printed path / tail work before the first request; drop it when this run
  // routed everything elsewhere. The manifest is per-run scratch — always drop.
  try {
    if (!produced.some(x => x.file === jsonlPath) && fs.existsSync(jsonlPath) && fs.statSync(jsonlPath).size === 0) fs.unlinkSync(jsonlPath);
  } catch { /* best-effort */ }
  try { if (fs.existsSync(sidsFile)) fs.unlinkSync(sidsFile); } catch { /* best-effort */ }

  if (produced.length === 0) {
    log('no capture file produced — nothing to import');
    process.exit(0);
  }
  let lastUrl: string | null = null;
  for (const { sid: captureSid, file: p } of produced) {
    log(`captured ${(fs.statSync(p).size / 1024).toFixed(1)}KB [${captureSid}] → importing ...`);
    log(`  capture  : ${p}`);
    const agentLog = profile === 'claude' ? claudeNativeLogPath(captureSid)
      : profile === 'opencode' ? opencodeNativePaths()
      : null;
    if (agentLog) log(`  agent log: ${agentLog}`);
    const result = await triggerIngest(captureSid, p);
    if (result) lastUrl = `${INSIGHT_BASE}/session/${captureSid}`;
    else log('ingest failed; you can import manually via the cannbot-insight UI');
  }
  if (lastUrl) {
    log(`✅ imported → opening ${lastUrl}`);
    openBrowser(lastUrl);
  }
  process.exit(0);
}

main().catch((e) => { log(e instanceof Error ? e.message : String(e)); process.exit(1); });
