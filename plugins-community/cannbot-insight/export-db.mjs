#!/usr/bin/env node

const _origEmitWarning = process.emitWarning.bind(process);
process.emitWarning = (warning, ...args) => {
  const msg = typeof warning === 'string' ? warning : String(warning?.message ?? warning);
  if (msg.includes('experimental')) return;
  return _origEmitWarning(warning, ...args);
};

const { DatabaseSync } = await import('node:sqlite');
const fs = await import('node:fs');
const path = await import('node:path');
const os = await import('node:os');
const readline = await import('node:readline');
const { fileURLToPath } = await import('node:url');

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_OPENCODE_DB = path.join(os.homedir(), '.local', 'share', 'opencode', 'opencode.db');
const DEFAULT_CLAUDE_DIR = path.join(os.homedir(), '.claude', 'projects');

function detectSourceType(filePath) {
  const basename = path.basename(filePath);
  if (basename.endsWith('.db')) return 'opencode-db';
  if (basename.endsWith('.jsonl')) return 'claude-jsonl';
  try {
    if (fs.statSync(filePath).isDirectory()) return 'claude-jsonl';
  } catch {}
  return 'claude-jsonl';
}

function getDefaultSource() {
  if (fs.existsSync(DEFAULT_OPENCODE_DB)) return { path: DEFAULT_OPENCODE_DB, type: 'opencode-db' };
  if (fs.existsSync(DEFAULT_CLAUDE_DIR)) return { path: DEFAULT_CLAUDE_DIR, type: 'claude-jsonl' };
  return { path: DEFAULT_OPENCODE_DB, type: 'opencode-db' };
}

function usage() {
  const def = getDefaultSource();
  console.log(`
Usage: node export-db.mjs [options]

Extract a single session (with subagents) into a standalone file.
Auto-detects framework from --file: .db → opencode, .jsonl/dir → Claude Code.
Zero dependencies — uses Node.js 22+ built-in node:sqlite (for opencode).

Options:
  -f, --file <path>        Source path (default: ${def.path})
                           opencode: .db file | Claude Code: .jsonl file or directory
  -s, --session-id <id>    Session ID to export (skip interactive selection)
  -o, --output <path>      Output file path
                           opencode default: <script-dir>/dbfile/session_<id>.db
                           claude default:   <script-dir>/jsonlfile/session_<id>.jsonl
  -l, --list               List available sessions and exit
  -j, --json               Output as JSON (for -l and export result)
  -h, --help               Show this help

Examples:
  node export-db.mjs --list                              # auto-detect source
  node export-db.mjs -s ses_0348d7dfaffeiNa9Bqbw6kvdLj   # opencode session
  node export-db.mjs -f ~/.claude/projects/ -s <uuid>     # Claude Code session
  node export-db.mjs                                       # interactive picker
  node export-db.mjs -s ses_xxx --json                    # JSON output

Requirements:
  Node.js >= 22 (built-in node:sqlite, no npm install needed)
`);
}

function parseArgs(argv) {
  const def = getDefaultSource();
  const opts = { file: def.path, sessionId: null, output: null, list: false, json: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    switch (a) {
      case '-f': case '--file':        opts.file = argv[++i]; break;
      case '-s': case '--session-id':  opts.sessionId = argv[++i]; break;
      case '-o': case '--output':      opts.output = argv[++i]; break;
      case '-l': case '--list':        opts.list = true; break;
      case '-j': case '--json':        opts.json = true; break;
      case '-h': case '--help':        usage(); process.exit(0);
      default:
        console.error(`Unknown option: ${a}`);
        usage();
        process.exit(1);
    }
  }
  return opts;
}

function truncate(str, max) {
  if (!str) return '';
  const s = String(str).replace(/\s+/g, ' ').trim();
  return s.length > max ? s.slice(0, max - 1) + '…' : s;
}

function pad(str, len) {
  const s = String(str ?? '');
  return s + ' '.repeat(Math.max(0, len - s.length));
}

function printSessionTable(sessions) {
  if (sessions.length === 0) {
    console.log('No sessions found.');
    return;
  }
  console.log('');
  console.log(
    pad('#', 4) +
    pad('Session ID', 38) +
    pad('Turns', 7) +
    pad('Model', 22) +
    'First Query'
  );
  console.log('-'.repeat(100));
  sessions.forEach((s, i) => {
    console.log(
      pad(String(i + 1), 4) +
      pad(s.id, 38) +
      pad(String(s.turnCount), 7) +
      pad(truncate(s.model, 21), 22) +
      truncate(s.firstQuery, 40)
    );
  });
  console.log('');
}

function prompt(question) {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  return new Promise(resolve => {
    rl.question(question, answer => { rl.close(); resolve(answer); });
  });
}

function formatSize(size) {
  return size > 1024 * 1024
    ? `${(size / 1024 / 1024).toFixed(2)} MB`
    : `${(size / 1024).toFixed(1)} KB`;
}

const TABLE_NAMES = ['session', 'message', 'part'];

function readSchema(db) {
  const tables = db.prepare(
    `SELECT sql FROM sqlite_master WHERE type='table' AND name IN ('session','message','part')`
  ).all();
  const indexes = db.prepare(
    `SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL AND tbl_name IN ('session','message','part')`
  ).all();
  return [...tables, ...indexes].map(r => r.sql);
}

function getAllSessionIds(db, rootId) {
  const ids = [rootId];
  const subagents = db.prepare('SELECT id FROM session WHERE parent_id = ?').all(rootId);
  for (const s of subagents) ids.push(s.id);
  return ids;
}

function readRows(db, table, whereClause, params) {
  return db.prepare(`SELECT * FROM ${table} WHERE ${whereClause}`).all(...params);
}

function insertRows(db, table, rows) {
  if (rows.length === 0) return 0;
  const cols = Object.keys(rows[0]);
  const colList = cols.map(c => `"${c}"`).join(', ');
  const placeholders = cols.map(() => '?').join(', ');
  const stmt = db.prepare(`INSERT INTO "${table}" (${colList}) VALUES (${placeholders})`);
  db.exec('BEGIN');
  for (const row of rows) stmt.run(...cols.map(c => row[c]));
  db.exec('COMMIT');
  return rows.length;
}

function listSessionsOpencode(db) {
  const sessions = db.prepare(
    `SELECT id, title, time_created FROM session
     WHERE parent_id IS NULL OR parent_id = ''
     ORDER BY time_created DESC`
  ).all();

  if (sessions.length === 0) return [];

  const ids = sessions.map(s => s.id);
  const placeholders = ids.map(() => '?').join(',');

  const firstUserMsgs = {};
  for (const r of db.prepare(
    `SELECT session_id, id, data FROM message
     WHERE session_id IN (${placeholders})
       AND json_extract(data, '$.role') = 'user'
     ORDER BY time_created`
  ).all(...ids)) {
    if (!firstUserMsgs[r.session_id]) firstUserMsgs[r.session_id] = r;
  }

  const firstMsgIds = Object.values(firstUserMsgs).map(m => m.id);
  const firstTextParts = {};
  if (firstMsgIds.length > 0) {
    const mpH = firstMsgIds.map(() => '?').join(',');
    for (const r of db.prepare(
      `SELECT message_id, data FROM part
       WHERE message_id IN (${mpH})
         AND json_extract(data, '$.type') = 'text'
       ORDER BY time_created`
    ).all(...firstMsgIds)) {
      if (!firstTextParts[r.message_id]) {
        try { firstTextParts[r.message_id] = JSON.parse(r.data)?.text ?? ''; }
        catch { firstTextParts[r.message_id] = ''; }
      }
    }
  }

  const counts = {};
  for (const r of db.prepare(
    `SELECT session_id, COUNT(*) as cnt FROM message
     WHERE session_id IN (${placeholders}) GROUP BY session_id`
  ).all(...ids)) counts[r.session_id] = r.cnt;

  const models = {};
  for (const r of db.prepare(
    `SELECT session_id, data FROM message
     WHERE session_id IN (${placeholders})
       AND json_extract(data, '$.role') = 'assistant'
     ORDER BY time_created`
  ).all(...ids)) {
    if (!models[r.session_id]) {
      try { models[r.session_id] = JSON.parse(r.data)?.modelID ?? ''; }
      catch {}
    }
  }

  const lastTs = {};
  for (const r of db.prepare(
    `SELECT session_id, MAX(time_created) as ts FROM message
     WHERE session_id IN (${placeholders}) GROUP BY session_id`
  ).all(...ids)) lastTs[r.session_id] = r.ts;

  return sessions.map(s => ({
    id: s.id,
    firstQuery: firstTextParts[firstUserMsgs[s.id]?.id] ?? s.title ?? '',
    turnCount: counts[s.id] ?? 0,
    model: models[s.id] ?? '',
    createdAt: s.time_created ? new Date(s.time_created).toISOString() : null,
    endedAt: lastTs[s.id] ? new Date(lastTs[s.id]).toISOString() : null,
  }));
}

async function exportOpencode(opts, srcPath, sessionId, outputPath) {
  const src = new DatabaseSync(srcPath);
  try {
    const exists = src.prepare('SELECT id FROM session WHERE id = ?').get(sessionId);
    if (!exists) {
      console.error(`Error: session ${sessionId} not found in ${srcPath}`);
      console.error('Hint: run with --list to see available sessions.');
      process.exit(1);
    }

    if (!opts.json) console.log(`Extracting session ${sessionId} ...`);

    const schemaSqls = readSchema(src);
    const allSessionIds = getAllSessionIds(src, sessionId);
    const ph = allSessionIds.map(() => '?').join(',');
    const sessionRows = readRows(src, 'session', `id IN (${ph})`, allSessionIds);
    const messageRows = readRows(src, 'message', `session_id IN (${ph})`, allSessionIds);
    const messageIds = messageRows.map(m => m.id);
    const partRows = messageIds.length > 0
      ? readRows(src, 'part', `message_id IN (${messageIds.map(() => '?').join(',')})`, messageIds)
      : [];
    src.close();

    if (fs.existsSync(outputPath)) fs.unlinkSync(outputPath);
    const out = new DatabaseSync(outputPath);
    try {
      out.exec('PRAGMA foreign_keys = OFF');
      for (const sql of schemaSqls) out.exec(sql);
      const nSession = insertRows(out, 'session', sessionRows);
      const nMessage = insertRows(out, 'message', messageRows);
      const nPart = insertRows(out, 'part', partRows);
      out.close();

      const size = fs.statSync(outputPath).size;
      if (opts.json) {
        console.log(JSON.stringify({
          framework: 'opencode', sessionId, outputPath, size,
          sessions: nSession, messages: nMessage, parts: nPart,
        }, null, 2));
      } else {
        console.log(`✓ Exported session ${sessionId}`);
        console.log(`  File:     ${outputPath}`);
        console.log(`  Size:     ${formatSize(size)}`);
        console.log(`  Sessions: ${nSession} (1 root + ${nSession - 1} subagents)`);
        console.log(`  Messages: ${nMessage}`);
        console.log(`  Parts:    ${nPart}`);
      }
    } catch (e) {
      try { out.close(); } catch {}
      try { if (fs.existsSync(outputPath)) fs.unlinkSync(outputPath); } catch {}
      throw e;
    }
  } finally {
    try { src.close(); } catch {}
  }
}

function collectJsonlFiles(dir) {
  const results = [];
  let entries = [];
  try { entries = fs.readdirSync(dir, { withFileTypes: true }); }
  catch { return results; }
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === 'subagents') continue;
      results.push(...collectJsonlFiles(fullPath));
    } else if (entry.isFile() && entry.name.endsWith('.jsonl')) {
      results.push(fullPath);
    }
  }
  return results;
}

function parseJsonlLines(filePath) {
  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = [];
  for (const line of content.split('\n')) {
    const t = line.trim();
    if (!t || t.startsWith('//')) continue;
    try { lines.push(JSON.parse(t)); } catch {}
  }
  return lines;
}

function extractTextContent(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    for (const c of content) {
      if (c?.type === 'text' && c.text) return c.text;
    }
    const tr = content.find(c => c?.type === 'tool_result' && typeof c.content === 'string');
    if (tr?.content) return tr.content;
  }
  return '';
}

function listSessionsClaude(srcPath) {
  let files = [];
  try {
    if (fs.statSync(srcPath).isFile()) files = [srcPath];
    else files = collectJsonlFiles(srcPath);
  } catch { return []; }

  const sessions = [];
  for (const file of files) {
    const id = path.basename(file, '.jsonl');
    const lines = parseJsonlLines(file);
    if (lines.length === 0) continue;

    let firstQuery = '';
    let model = '';
    let createdAt = null;
    let endedAt = null;

    for (const line of lines) {
      if (line.timestamp) {
        if (!createdAt) createdAt = line.timestamp;
        endedAt = line.timestamp;
      }
      if (!firstQuery && line.type === 'user' && line.message?.role === 'user') {
        firstQuery = extractTextContent(line.message.content);
      }
      if (!model && line.type === 'assistant' && line.message?.model) {
        model = line.message.model;
      }
    }

    if (!createdAt) {
      try { createdAt = fs.statSync(file).mtime.toISOString(); } catch {}
    }

    sessions.push({
      id, firstQuery: truncate(firstQuery, 200), turnCount: lines.length,
      model, createdAt, endedAt,
    });
  }

  sessions.sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)));
  return sessions;
}

function copyDir(src, dst) {
  if (!fs.existsSync(dst)) fs.mkdirSync(dst, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const s = path.join(src, entry.name);
    const d = path.join(dst, entry.name);
    if (entry.isDirectory()) copyDir(s, d);
    else if (entry.isFile()) fs.copyFileSync(s, d);
  }
}

async function exportClaude(opts, srcPath, sessionId, outputPath) {
  let srcFile = null;
  try {
    if (fs.statSync(srcPath).isFile()) {
      if (path.basename(srcPath, '.jsonl') === sessionId) srcFile = srcPath;
    } else {
      srcFile = collectJsonlFiles(srcPath).find(f => path.basename(f, '.jsonl') === sessionId);
    }
  } catch {}

  if (!srcFile) {
    console.error(`Error: session ${sessionId} not found in ${srcPath}`);
    console.error('Hint: run with --list to see available sessions.');
    process.exit(1);
  }

  if (!opts.json) console.log(`Extracting session ${sessionId} ...`);

  if (fs.existsSync(outputPath)) fs.unlinkSync(outputPath);
  fs.copyFileSync(srcFile, outputPath);

  const srcDir = path.dirname(srcFile);
  const subagentsDir = path.join(srcDir, sessionId, 'subagents');
  let subagentCount = 0;
  if (fs.existsSync(subagentsDir) && fs.statSync(subagentsDir).isDirectory()) {
    const outSubDir = path.join(path.dirname(outputPath), sessionId, 'subagents');
    copyDir(subagentsDir, outSubDir);
    subagentCount = fs.readdirSync(outSubDir).filter(f => f.endsWith('.jsonl')).length;
  }

  const size = fs.statSync(outputPath).size;
  const lineCount = parseJsonlLines(outputPath).length;

  if (opts.json) {
    console.log(JSON.stringify({
      framework: 'claude-code', sessionId, outputPath, size,
      lines: lineCount, subagents: subagentCount,
    }, null, 2));
  } else {
    console.log(`✓ Exported session ${sessionId}`);
    console.log(`  File:     ${outputPath}`);
    console.log(`  Size:     ${formatSize(size)}`);
    console.log(`  Lines:    ${lineCount}`);
    if (subagentCount > 0) console.log(`  Subagents: ${subagentCount}`);
  }
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));

  if (!fs.existsSync(opts.file)) {
    console.error(`Error: Source not found: ${opts.file}`);
    process.exit(1);
  }

  const sourceType = detectSourceType(opts.file);
  const isOpencode = sourceType === 'opencode-db';

  if (opts.list) {
    const sessions = isOpencode
      ? listSessionsOpencode(new DatabaseSync(opts.file))
      : listSessionsClaude(opts.file);
    if (opts.json) console.log(JSON.stringify({ framework: sourceType, sessions }, null, 2));
    else printSessionTable(sessions);
    return;
  }

  let sessionId = opts.sessionId;
  let sessions = null;

  if (!sessionId) {
    sessions = isOpencode
      ? listSessionsOpencode(new DatabaseSync(opts.file))
      : listSessionsClaude(opts.file);
    if (sessions.length === 0) {
      console.error('No sessions found.');
      process.exit(1);
    }
    printSessionTable(sessions);
    const answer = await prompt(`Enter session number (1-${sessions.length}): `);
    const idx = parseInt(answer.trim(), 10);
    if (isNaN(idx) || idx < 1 || idx > sessions.length) {
      console.error(`Invalid selection: "${answer}". Enter a number between 1 and ${sessions.length}.`);
      process.exit(1);
    }
    sessionId = sessions[idx - 1].id;
  }

  let outputPath = opts.output;
  if (!outputPath) {
    const dirName = isOpencode ? 'dbfile' : 'jsonlfile';
    const ext = isOpencode ? 'db' : 'jsonl';
    const fileStem = isOpencode ? `session_${sessionId}` : sessionId;
    const outDir = path.join(SCRIPT_DIR, dirName);
    if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
    outputPath = path.join(outDir, `${fileStem}.${ext}`);
  }

  const outDir = path.dirname(outputPath);
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

  if (isOpencode) {
    await exportOpencode(opts, opts.file, sessionId, outputPath);
  } else {
    await exportClaude(opts, opts.file, sessionId, outputPath);
  }
}

main().catch(e => {
  console.error(`Error: ${e.message}`);
  process.exit(1);
});
