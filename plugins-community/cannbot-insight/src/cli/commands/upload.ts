// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Command } from 'commander';
import { InsightClient } from '../client';
import { renderTable, TableColumn } from '../utils/table';
import { formatDate } from '../utils/format';
import { formatHeader, formatDivider, formatSuccess, formatWarning, theme } from '../utils/colors';
import type { ApiImportableSession, ApiImportableSessionsResponse } from '../types';
import readline from 'node:readline';
import { execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';

const IMPORT_COLUMNS: TableColumn[] = [
  { key: 'id', label: 'Session ID', width: 20 },
  { key: 'firstQuery', label: 'Query', width: 30 },
  { key: 'turnCount', label: 'Turns', width: 8 },
  { key: 'model', label: 'Model', width: 18 },
  { key: 'createdAt', label: 'Created', width: 16 },
];

function renderImportableRow(row: ApiImportableSession, key: string): string {
  switch (key) {
    case 'firstQuery': return row.firstQuery ?? '—';
    case 'model': return row.model ?? '—';
    case 'createdAt': return formatDate(row.createdAt);
    default: return String(row[key as keyof ApiImportableSession] ?? '—');
  }
}

async function confirmPrompt(message: string): Promise<boolean> {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  return new Promise((resolve) => {
    rl.question(`${message} [y/N]: `, (answer) => {
      rl.close();
      resolve(answer.toLowerCase() === 'y' || answer.toLowerCase() === 'yes');
    });
  });
}

async function inputPrompt(label: string, defaultVal: string): Promise<string> {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  return new Promise((resolve) => {
    rl.question(`${label}: `, (answer) => {
      rl.close();
      resolve(answer.trim() || defaultVal);
    });
  });
}

async function collectDescription(sessionInfo: { query?: string; user?: string; model?: string; taskId?: string; framework?: string; sourcePath?: string }): Promise<string> {
  const timeRange = new Date().toISOString().split('T')[0];
  const query = sessionInfo.query ?? '';
  const user = sessionInfo.user ?? '';
  const model = sessionInfo.model ?? '';
  const taskId = sessionInfo.taskId ?? '';
  const framework = sessionInfo.framework ?? '';
  const sourcePath = sessionInfo.sourcePath ?? '';

  console.log('');
  console.log(theme.muted('填写提交信息（直接回车使用默认值）：'));
  console.log('');

  const submitter = await inputPrompt('提交人', user || '');
  const contentDesc = await inputPrompt('内容描述', query || '');
  const issueDesc = await inputPrompt('问题说明', '');
  const logPath = await inputPrompt('日志路径', sourcePath || `taskId ${taskId}, 框架 ${framework}`);
  const notes = await inputPrompt('备注', `模型 ${model}, 时间 ${timeRange}`);

  return `提交人: ${submitter}
内容描述: ${contentDesc}
问题说明: ${issueDesc}
日志路径: ${logPath}
备注: ${notes}`;
}

function detectSourceType(filePath: string): 'opencode-db' | 'claude-jsonl' | string {
  const basename = filePath.split('/').pop() ?? '';
  if (basename.endsWith('.db') && basename.startsWith('cannbot_session_')) return 'cannbot-insight';
  if (basename.endsWith('.db')) return 'opencode-db';
  if (basename.endsWith('.jsonl') || basename.endsWith('.json')) return 'claude-jsonl';
  try {
    const stat = fs.statSync(filePath);
    if (stat.isDirectory()) return 'claude-jsonl';
  } catch {}
  return 'claude-jsonl';
}

const CANNBAY_PULL_URL = 'https://gitcode.com/guanxinghua/CANNBay.git';
const CANNBAY_PUSH_URL = Buffer.from('aHR0cHM6Ly9ndWFueGluZ2h1YTpwc3F5WXAyYnpFRkI0eDVQRlVTV0dMS3lAZ2l0Y29kZS5jb20vZ3VhbnhpbmdodWEvQ0FOTkJheS5naXQ=', 'base64').toString();

function runGit(cmd: string, cwd: string) {
  return execSync(cmd, { cwd, stdio: 'pipe', timeout: 60_000 });
}

async function ensureBackend(client: InsightClient): Promise<boolean> {
  try {
    await client.listSessions({ pageSize: 1 });
    return true;
  } catch {
    return false;
  }
}

function spawnBackend(port: number): number {
  const pid = execSync(
    `npx next dev --port ${port} > /dev/null 2>&1 & echo $!`,
    { encoding: 'utf-8', timeout: 10_000 },
  ).trim();
  return parseInt(pid, 10);
}

async function waitForBackend(client: InsightClient, maxWait: number = 60): Promise<boolean> {
  for (let i = 0; i < maxWait; i++) {
    try {
      await client.listSessions({ pageSize: 1 });
      return true;
    } catch {
      await new Promise(r => setTimeout(r, 1000));
    }
  }
  return false;
}

async function selectPrompt(sessions: ApiImportableSession[]): Promise<ApiImportableSession> {
  console.log(formatHeader('Select a session to upload:'));
  console.log(renderTable(
    IMPORT_COLUMNS,
    sessions as unknown as Record<string, unknown>[],
    renderImportableRow as unknown as (row: Record<string, unknown>, key: string) => string,
  ));
  console.log('');

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  return new Promise((resolve, reject) => {
    rl.question(`Enter session number (1-${sessions.length}): `, (answer) => {
      rl.close();
      const idx = parseInt(answer.trim(), 10);
      if (idx >= 1 && idx <= sessions.length) {
        resolve(sessions[idx - 1]);
      } else {
        reject(new Error(`Invalid selection: "${answer}". Enter a number between 1 and ${sessions.length}.`));
      }
    });
  });
}

export function uploadCommand(): Command {
  const cmd = new Command('upload');
  cmd
    .description('Upload session to CANNBay (auto-starts backend if needed)')
    .option('--session <taskId>', 'Upload from Insight DB by taskId')
    .option('--framework <framework>', 'Framework type (auto-detected from file)')
    .option('--source <type>', 'Source type — auto-detected from --file if omitted')
    .option('--file <path>', 'Source file path for direct upload')
    .option('--list', 'List importable sessions from source file')
    .option('--session-id <id>', 'Upload specific session from source file by ID')
    .option('--description <text>', 'Description text (skip interactive prompt)')
    .option('--interactive', 'Interactive prompt for description fields (default when no --description)')
    .option('--yes', 'Skip confirmation prompt')
    .option('--json', 'Output result as JSON')
    .action(async (opts, command) => {
      const globalOpts = command.parent?.opts() ?? {};
      const serverUrl = globalOpts.server ?? 'http://localhost:21025';
      const client = new InsightClient(serverUrl, {
        timeout: +globalOpts.timeout || 120000,
      });

      // Auto-start backend if not running
      const backendRunning = await ensureBackend(client);
      let backendPid: number | null = null;

      if (!backendRunning) {
        if (!opts.json) console.log(theme.muted('Backend not running, starting...'));
        const port = parseInt(serverUrl.split(':').pop() ?? '21025', 10);
        backendPid = spawnBackend(port);
        const ready = await waitForBackend(client);
        if (!ready) {
          console.error(formatWarning('Error: Backend failed to start'));
          if (backendPid) try { process.kill(backendPid); } catch {}
          process.exit(1);
        }
        if (!opts.json) console.log(theme.muted('Backend ready'));
      }

      try {
        let description = opts.description ?? '';
        const needInteractive = !opts.description && !opts.json;

        // Mode 1: Upload from Insight DB by taskId
        if (opts.session) {
          const taskId = opts.session;
          const framework = opts.framework ?? 'unknown';

          if (needInteractive) {
            const sessionDetail = await client.getSession(taskId);
            description = await collectDescription({
              query: sessionDetail.query ?? undefined,
              user: sessionDetail.user ?? undefined,
              model: sessionDetail.model ?? undefined,
              taskId,
              framework,
              sourcePath: undefined,
            });
          }

          if (!opts.json) {
            console.log(formatHeader(`Upload Session: ${taskId}`));
            console.log(formatDivider());
            console.log(theme.muted('Description:'));
            console.log(theme.muted(description));
          }

          const result = await client.uploadSession(taskId, description, framework);

          if (opts.json) {
            console.log(JSON.stringify(result, null, 2));
            return;
          }

          console.log(formatSuccess(`✓ Uploaded session ${taskId} → CANNBay`));
          console.log(theme.muted(`  Filename: ${result.filename}`));
          return;
        }

        // Mode 2: Upload from source file (import → upload)
        if (opts.file) {
          const source = opts.source ?? detectSourceType(opts.file);
          const filePath = opts.file;

          const importable: ApiImportableSessionsResponse = await client.listImportableSessions(source, filePath);

          if (opts.list) {
            if (opts.json) {
              console.log(JSON.stringify(importable, null, 2));
              return;
            }
            const sessions = importable.sessions as ApiImportableSession[];
            if (sessions.length === 0) {
              console.log(theme.muted('No importable sessions found.'));
              return;
            }
            console.log(formatHeader(`Importable Sessions (${source})`));
            console.log(renderTable(
              IMPORT_COLUMNS,
              sessions as unknown as Record<string, unknown>[],
              renderImportableRow as unknown as (row: Record<string, unknown>, key: string) => string,
            ));
            return;
          }

          const sessions = importable.sessions as ApiImportableSession[];
          let target: ApiImportableSession;

          if (opts.sessionId) {
            const found = sessions.find(s => s.id === opts.sessionId);
            if (!found) {
              console.error(formatWarning(`Error: Session "${opts.sessionId}" not found`));
              process.exit(1);
            }
            target = found;
          } else if (sessions.length === 1) {
            target = sessions[0];
          } else {
            // Multiple sessions — list and let user pick
            try {
              target = await selectPrompt(sessions);
            } catch (e) {
              console.error(formatWarning(e instanceof Error ? e.message : 'Invalid selection'));
              process.exit(1);
            }
          }

          const resolvedFramework = source === 'opencode-db' ? 'opencode' : source === 'claude-jsonl' ? 'claude-code' : source;

          if (needInteractive) {
            description = await collectDescription({
              query: target.firstQuery ?? undefined,
              user: undefined,
              model: target.model ?? undefined,
              taskId: target.id,
              framework: resolvedFramework,
              sourcePath: filePath,
            });
          }

          if (!opts.json) {
            console.log(formatHeader(`Upload from Source: ${source}`));
            console.log(formatDivider());
            console.log(`  Session: ${target.id} — ${target.firstQuery ?? '—'}`);
            console.log(`  File: ${filePath}`);
            console.log(theme.muted('Description:'));
            console.log(theme.muted(description));
            console.log('');
          }

          if (!opts.yes) {
            const confirmed = await confirmPrompt(`Import and upload session ${target.id}?`);
            if (!confirmed) {
              console.log(theme.muted('Upload cancelled.'));
              return;
            }
          }

          // Step 1: Import into Insight DB
          if (!opts.json) console.log(theme.muted('  Importing into Insight DB...'));
          const importResult = await client.importSession(source, filePath, target.id);

          // Step 2: Upload to CANNBay (use original taskId + resolved framework to find session)
          if (!opts.json) console.log(theme.muted('  Uploading to CANNBay...'));
          const uploadResult = await client.uploadSession(target.id, description, resolvedFramework);

          if (opts.json) {
            console.log(JSON.stringify({ import: importResult, upload: uploadResult }, null, 2));
            return;
          }

          console.log(formatSuccess(`✓ Imported & uploaded session ${target.id} → CANNBay`));
          console.log(theme.muted(`  Insight taskId: ${importResult.sessionId}`));
          console.log(theme.muted(`  Filename: ${uploadResult.filename}`));
          return;
        }

        console.error(formatWarning('Error: Specify --session <taskId> or --file <path>'));
        process.exit(1);
      } finally {
        // Auto-stop backend if we started it
        if (backendPid) {
          if (!opts.json) console.log(theme.muted('Stopping backend...'));
          try { process.kill(backendPid); } catch {}
        }
      }
    });

  return cmd;
}
