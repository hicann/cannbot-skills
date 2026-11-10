#!/usr/bin/env node
// SPDX-License-Identifier: CANN-2.0
/**
 * docs-skills-benchmark 基准工具（bench.js）
 *
 * 用途：docs-common-review 家族的基准回归测试。固定输入样例 cann_sim工具指南.md，
 * 以 common_bench.md 为基线（golden），检测 skill 修改后检查结果是否仍与基线对齐；
 * 不对齐时输出差异明细（漏检 / 多检 / 关键字段变化 / 文案差异）反馈给用户。
 *
 * 子命令：
 *   convert   聚合结果 JSON（{summary, issues} 或 11 字段数组）→ common_bench.md 同构的"问题N" Markdown
 *   compare   基线与新结果对齐比对（新结果支持 bench md / 聚合 JSON / 原子结果数组 JSON）
 *             输出对齐报告；退出码 0=对齐，1=不对齐（存在阻断性差异），2=输入错误
 *   selftest  工具自检（golden 解析 / 引擎别名归一 / 匹配 / 差异判定 / convert 往返）
 *
 * 用法：
 *   node bench.js convert --input=<docs-common-review-result.json> --output=<runs/common_bench.md>
 *   node bench.js compare --golden=<common_bench.md> --new=<runs/common_bench.md|result.json> \
 *                         [--report=<runs/align_report.md>] [--strict] [--label=<运行标签>] [--engines=a,b]
 *   node bench.js selftest
 *
 * 对齐判定（compare）：
 *   阻断（不对齐，需反馈）：
 *     - 漏检：基线有、新结果无（疑似 skill 修改导致漏检）
 *     - 多检：新结果有、基线无（疑似新增检出 / 误检）
 *     - 严重程度变化 / 问题类型变化 / 行号漂移 / 文件变化（同条问题的关键字段改变）
 *     - --strict 时，上下文/描述/建议的文案差异也计入阻断
 *   非阻断（仅提示）：
 *     - 上下文/描述/建议的措辞差异（LLM 表述波动，附相似度）
 *     - 基线未覆盖引擎（如 markdownlint）的检出，不计入对齐判定
 *   网络注记：link-validity-check 的差异受网络环境影响，需人工确认后再定性
 *
 * 引擎命名归一：golden（common_bench.md）、聚合 JSON、原子结果 JSON 三方历史口径不一致
 * （如 AICheck/aicheck、link_validity-check/link-validity-check、compliance-check 的历史写法
 * StaticCheck_compliance 等），比较前统一归一到 canonical 名；convert 输出使用 golden 的写法，
 * 保证与基线文件可直接 diff。
 */
'use strict';

const fs = require('fs');
const path = require('path');

// ---------- 常量 ----------

// 各方历史写法 → canonical（键为小写化后的原文）
const ENGINE_ALIAS = {
  'tag-closed-check': 'tag-closed-check',
  'tag_closed-check': 'tag-closed-check',
  'tag_closed_check': 'tag-closed-check',
  'link-validity-check': 'link-validity-check',
  'link_validity-check': 'link-validity-check',
  'link_validity_check': 'link-validity-check',
  'link-check': 'link-validity-check',
  'resource-existence-check': 'resource-existence-check',
  'resource_existence_check': 'resource-existence-check',
  'codespell-check': 'codespell-check',
  'codespell_check': 'codespell-check',
  'filename-check': 'filename-check',
  'filename_check': 'filename-check',
  'staticcheck_filename_check': 'filename-check',
  'punctuation-check': 'punctuation-check',
  'punctuation_check': 'punctuation-check',
  'staticcheck_punctuation': 'punctuation-check',
  'staticcheck_punctuation_check': 'punctuation-check',
  'compliance-check': 'compliance-check',
  'compliance_check': 'compliance-check',
  'staticcheck_compliance': 'compliance-check',
  'staticcheck_compliance_check': 'compliance-check',
  'aicheck': 'aicheck',
  'markdownlint': 'markdownlint',
  'markdown-lint': 'markdownlint',
  'markdown_lint': 'markdownlint'
};

// canonical → golden（common_bench.md）使用的写法
const ENGINE_DISPLAY = {
  'tag-closed-check': 'tag-closed-check',
  'link-validity-check': 'link_validity-check',
  'resource-existence-check': 'resource-existence-check',
  'codespell-check': 'codespell-check',
  'filename-check': 'filename-check',
  'punctuation-check': 'punctuation-check',
  'compliance-check': 'compliance-check',
  'aicheck': 'AICheck',
  'markdownlint': 'markdownlint'
};

// bench md 字段（record 字段名 ↔ golden 标签），顺序即 convert 输出顺序
const FIELD_LABELS = [
  ['engine', '检查引擎'],
  ['type', '问题类型'],
  ['file', '文件路径'],
  ['line', '位置行号'],
  ['context', '问题上下文'],
  ['description', '问题描述'],
  ['suggestion', '修改建议'],
  ['severity', '严重程度']
];
const FIELD_KEYS = {};
for (const pair of FIELD_LABELS) FIELD_KEYS[pair[1]] = pair[0];

// 脚本类引擎（输出确定性高，措辞变化通常意味着工具/规则版本变化）
const SCRIPT_ENGINES = new Set([
  'tag-closed-check', 'link-validity-check', 'resource-existence-check',
  'codespell-check', 'filename-check', 'markdownlint'
]);
// 受网络环境影响的引擎
const NETWORK_ENGINES = new Set(['link-validity-check']);

const CROSS_LINE_THRESHOLD = 0.5; // 跨行兜底匹配的相似度阈值
const MIGRATION_THRESHOLD = 0.55; // 漏检↔多检 疑似同一问题（引擎/位置迁移）的相似度阈值

// ---------- 基础工具 ----------

function normText(s) {
  return String(s === undefined || s === null ? '' : s)
    .replace(/\uFEFF/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function canonEngine(name) {
  const k = normText(name).toLowerCase();
  return Object.prototype.hasOwnProperty.call(ENGINE_ALIAS, k) ? ENGINE_ALIAS[k] : k;
}

function displayEngine(canon) {
  return Object.prototype.hasOwnProperty.call(ENGINE_DISPLAY, canon) ? ENGINE_DISPLAY[canon] : canon;
}

function normLine(v) {
  const s = normText(v);
  if (!s) return 0; // 无行号（如 filename-check）与 golden 的 0 号位对齐
  const n = parseInt(s, 10);
  return Number.isFinite(n) ? n : 0;
}

function clip(s, n) {
  const t = normText(s);
  return t.length <= n ? t : t.slice(0, n) + '…';
}

// 统一 record 结构（golden md / 聚合 JSON / 原子 JSON 三方输入都归一到该形状）
function normalizeRecord(r) {
  return {
    engine: canonEngine(r.engine),
    type: normText(r.type),
    file: path.basename(normText(r.file)),
    line: normLine(r.line),
    context: normText(r.context),
    description: normText(r.description),
    suggestion: normText(r.suggestion),
    severity: normText(r.severity),
    rawIndex: r.rawIndex
  };
}

// 11 字段 issue（聚合 JSON / 原子结果 JSON）→ record
function recordFromIssue(i, idx) {
  return normalizeRecord({
    engine: (i.rule_item !== undefined && i.rule_item !== null && i.rule_item !== '')
      ? i.rule_item : (i.rule_item_name || ''),
    type: i.rule_type,
    file: (i.file_name !== undefined && i.file_name !== null && i.file_name !== '') ? i.file_name : i.location,
    line: i.line_number,
    context: i.snippet,
    description: i.description,
    suggestion: i.suggestion,
    severity: i.rule_level,
    rawIndex: idx
  });
}

// ---------- 相似度（字符二元组 Jaccard，中英文均适用） ----------

function bigrams(s) {
  const t = Array.from(s);
  const set = new Set();
  if (t.length === 1) set.add(t[0]);
  for (let i = 0; i < t.length - 1; i++) set.add(t[i] + t[i + 1]);
  return set;
}

function sim(a, b) {
  const A = normText(a);
  const B = normText(b);
  if (!A && !B) return 1;
  if (!A || !B) return 0;
  if (A === B) return 1;
  const ga = bigrams(A);
  const gb = bigrams(B);
  if (!ga.size || !gb.size) return 0;
  let inter = 0;
  for (const x of ga) if (gb.has(x)) inter++;
  return inter / (ga.size + gb.size - inter);
}

// 两条问题的整体相似度：上下文为主、描述兜底（0.85 折扣）
function recordSim(r1, r2) {
  return Math.max(sim(r1.context, r2.context), 0.85 * sim(r1.description, r2.description));
}

// ---------- bench md 解析 / 渲染 ----------

const BLOCK_HEAD = /^##\s*问题(\d+)\s*$/;
const FIELD_LINE = /^-\s*(检查引擎|问题类型|文件路径|位置行号|问题上下文|问题描述|修改建议|严重程度)\s*[：:]\s*(.*)$/;

// 解析 common_bench.md 同构的"问题N" Markdown → record 数组
function parseBenchMd(text) {
  const records = [];
  const skipped = [];
  let cur = null;
  for (const line of String(text).split(/\r?\n/)) {
    const h = BLOCK_HEAD.exec(line);
    if (h) {
      if (cur) records.push(cur.record);
      cur = { record: { rawIndex: parseInt(h[1], 10) } };
      continue;
    }
    if (!cur) continue;
    const f = FIELD_LINE.exec(line);
    if (f) cur.record[FIELD_KEYS[f[1]]] = f[2].trim();
  }
  if (cur) records.push(cur.record);
  const out = [];
  for (const r of records) {
    if (!normText(r.engine)) { skipped.push(r.rawIndex); continue; }
    out.push(normalizeRecord(r));
  }
  return { records: out, skippedBlocks: skipped };
}

// record 数组 → common_bench.md 同构 Markdown（引擎使用 golden 写法，便于与基线直接 diff）
function renderBenchMd(records, opts) {
  opts = opts || {};
  const lines = [];
  lines.push(opts.title || '# 通用规则测试集（基准重跑结果）');
  lines.push('');
  if (opts.note) {
    lines.push('> ' + opts.note);
    lines.push('');
  }
  records.forEach((r, idx) => {
    lines.push('## 问题' + (idx + 1));
    lines.push('');
    for (const pair of FIELD_LABELS) {
      const key = pair[0];
      let v;
      if (key === 'engine') v = displayEngine(r.engine);
      else if (key === 'line') v = String(r.line);
      else v = r[key];
      lines.push('- ' + pair[1] + '：' + (v === undefined || v === null ? '' : v));
    }
    lines.push('');
  });
  return lines.join('\n');
}

// ---------- 匹配 ----------

// golden × fresh 两两匹配：
//   第 1 阶段：同 (引擎, 行号) 分组，组内按相似度贪心配对（覆盖同名键的全部可配对项）
//   第 2 阶段：同引擎跨行兜底（相似度 >= CROSS_LINE_THRESHOLD，记为行号漂移）
//   剩余：golden 侧 → 漏检；fresh 侧 → 多检
function matchRecords(golden, fresh) {
  const usedG = new Array(golden.length).fill(false);
  const usedN = new Array(fresh.length).fill(false);
  const pairs = [];
  const keyOf = r => r.engine + '\u0000' + r.line;

  const gByKey = new Map();
  golden.forEach((r, i) => {
    const k = keyOf(r);
    if (!gByKey.has(k)) gByKey.set(k, []);
    gByKey.get(k).push(i);
  });
  const nByKey = new Map();
  fresh.forEach((r, i) => {
    const k = keyOf(r);
    if (!nByKey.has(k)) nByKey.set(k, []);
    nByKey.get(k).push(i);
  });

  // 第 1 阶段
  for (const entry of gByKey) {
    const nIdxs = nByKey.get(entry[0]) || [];
    const cand = [];
    for (const gi of entry[1]) {
      for (const ni of nIdxs) {
        cand.push({ gi: gi, ni: ni, s: recordSim(golden[gi], fresh[ni]) });
      }
    }
    cand.sort((a, b) => b.s - a.s);
    for (const c of cand) {
      if (!usedG[c.gi] && !usedN[c.ni]) {
        usedG[c.gi] = true;
        usedN[c.ni] = true;
        pairs.push({ g: c.gi, n: c.ni, score: c.s, crossLine: false });
      }
    }
  }

  // 第 2 阶段（同引擎、不同行）
  const cand2 = [];
  for (let gi = 0; gi < golden.length; gi++) {
    if (usedG[gi]) continue;
    for (let ni = 0; ni < fresh.length; ni++) {
      if (usedN[ni]) continue;
      if (fresh[ni].engine !== golden[gi].engine) continue;
      if (fresh[ni].line === golden[gi].line) continue;
      const s = recordSim(golden[gi], fresh[ni]);
      if (s >= CROSS_LINE_THRESHOLD) cand2.push({ gi: gi, ni: ni, s: s });
    }
  }
  cand2.sort((a, b) => b.s - a.s);
  for (const c of cand2) {
    if (!usedG[c.gi] && !usedN[c.ni]) {
      usedG[c.gi] = true;
      usedN[c.ni] = true;
      pairs.push({ g: c.gi, n: c.ni, score: c.s, crossLine: true });
    }
  }

  return {
    pairs: pairs,
    missing: golden.filter((r, i) => !usedG[i]),
    added: fresh.filter((r, i) => !usedN[i])
  };
}

// ---------- 对齐比较 ----------

const BLOCK_FIELDS = [
  ['severity', '严重程度'],
  ['type', '问题类型'],
  ['file', '文件']
];
const WORDING_FIELDS = [
  ['context', '上下文'],
  ['description', '描述'],
  ['suggestion', '建议']
];

/**
 * 对齐比较核心。
 * @param golden 基线 record 数组（parseBenchMd 产物）
 * @param fresh  新结果 record 数组（parseBenchMd 或 recordFromIssue 产物）
 * @param opts   { strict, engines }；engines 为空时默认取基线出现的引擎集合
 * @returns 比较结果对象（含 aligned、各项差异明细、统计）
 */
function compareRecords(golden, fresh, opts) {
  opts = opts || {};
  const goldenScope = new Set(golden.map(r => r.engine));
  const scope = (opts.engines && opts.engines.length) ? new Set(opts.engines.map(canonEngine)) : goldenScope;

  golden.forEach((r, i) => { r.label = '问题' + (r.rawIndex !== undefined ? r.rawIndex : i + 1); });
  fresh.forEach((r, i) => { r.label = '新#' + (r.rawIndex !== undefined ? r.rawIndex : i + 1); });

  const gIn = golden.filter(r => scope.has(r.engine));
  const gOut = golden.filter(r => !scope.has(r.engine));
  const nIn = fresh.filter(r => scope.has(r.engine));
  const nOut = fresh.filter(r => !scope.has(r.engine));

  const matched = matchRecords(gIn, nIn);

  const fieldChanges = [];
  const wordingChanges = [];
  for (const p of matched.pairs) {
    const g = gIn[p.g];
    const n = nIn[p.n];
    for (const pair of BLOCK_FIELDS) {
      if (g[pair[0]] !== n[pair[0]]) {
        fieldChanges.push({ g: g, n: n, field: pair[1], oldV: g[pair[0]], newV: n[pair[0]] });
      }
    }
    if (p.crossLine) {
      fieldChanges.push({ g: g, n: n, field: '行号（漂移）', oldV: 'L' + g.line, newV: 'L' + n.line });
    }
    for (const pair of WORDING_FIELDS) {
      if (g[pair[0]] !== n[pair[0]]) {
        wordingChanges.push({
          g: g, n: n, field: pair[1], sim: sim(g[pair[0]], n[pair[0]]),
          oldV: g[pair[0]], newV: n[pair[0]]
        });
      }
    }
  }

  // 漏检 ↔ 多检 疑似同一问题（引擎/规则迁移）提示
  const migrations = [];
  const usedAdded = new Set();
  for (const m of matched.missing) {
    let best = null;
    for (const a of matched.added) {
      if (usedAdded.has(a)) continue;
      const s = recordSim(m, a);
      if (s >= MIGRATION_THRESHOLD && (!best || s > best.s)) best = { a: a, s: s };
    }
    if (best) { usedAdded.add(best.a); migrations.push({ m: m, a: best.a, sim: best.s }); }
  }
  const migrationOf = rec => {
    for (const mig of migrations) {
      if (mig.m === rec || mig.a === rec) return mig;
    }
    return null;
  };

  // 统计（canonical 引擎名）
  const stats = new Map();
  const stat = engine => {
    if (!stats.has(engine)) {
      stats.set(engine, { golden: 0, fresh: 0, aligned: 0, missing: 0, added: 0, changed: 0, outOfScope: false });
    }
    return stats.get(engine);
  };
  for (const r of gIn) stat(r.engine).golden++;
  for (const r of nIn) stat(r.engine).fresh++;
  for (const r of matched.missing) stat(r.engine).missing++;
  for (const r of matched.added) stat(r.engine).added++;
  for (const c of fieldChanges) stat(c.g.engine).changed++;
  for (const p of matched.pairs) stat(gIn[p.g].engine).aligned++;
  for (const r of nOut) { const st = stat(r.engine); st.fresh++; st.outOfScope = true; }
  for (const r of gOut) { const st = stat(r.engine); st.golden++; st.outOfScope = true; }

  const enginesNoResult = [];
  for (const e of scope) {
    const hasFresh = nIn.some(r => r.engine === e);
    const hasGolden = gIn.some(r => r.engine === e);
    if (hasGolden && !hasFresh) enginesNoResult.push(e);
  }

  const blockingCount = matched.missing.length + matched.added.length + fieldChanges.length +
    (opts.strict ? wordingChanges.length : 0);
  const aligned = blockingCount === 0;

  return {
    goldenTotal: golden.length, freshTotal: fresh.length,
    scope: scope, gIn: gIn, gOut: gOut, nIn: nIn, nOut: nOut,
    pairs: matched.pairs, missing: matched.missing, added: matched.added,
    fieldChanges: fieldChanges, wordingChanges: wordingChanges,
    migrations: migrations, migrationOf: migrationOf,
    stats: stats, enginesNoResult: enginesNoResult,
    strict: !!opts.strict, aligned: aligned, blockingCount: blockingCount
  };
}

// ---------- 报告渲染 ----------

function recordBrief(r) {
  return '[' + r.label + '] ' + displayEngine(r.engine) + ' · L' + r.line + ' · ' + (r.severity || '-');
}

function networkNote(r) {
  return NETWORK_ENGINES.has(r.engine) ? ' 【网络相关，需人工确认】' : '';
}

function renderReport(res, meta) {
  const L = [];
  const disp = displayEngine;
  L.push('# 基准回归对齐报告');
  L.push('');
  L.push('- 基线（golden）：' + meta.goldenPath + ' — ' + res.goldenTotal + ' 条');
  L.push('- 新结果：' + meta.newPath + ' — ' + res.freshTotal + ' 条（比对范围内 ' + res.nIn.length +
    ' 条' + (res.nOut.length ? '，基线外 ' + res.nOut.length + ' 条' : '') + '）' +
    (meta.label && meta.label !== meta.newPath ? '（' + meta.label + '）' : ''));
  if (res.gOut.length) {
    L.push('- 注意：基线中 ' + res.gOut.length + ' 条不在本次比对范围（--engines 限定），不参与判定');
  }
  L.push('- 判定：' + (res.aligned ? '✅ 对齐' : '❌ 不对齐') + '（阻断性差异 ' + res.blockingCount + ' 处' +
    (res.strict ? '，含文案差异（--strict）' : '') + '）');
  L.push('');

  // 按引擎统计
  L.push('## 按引擎统计');
  L.push('');
  L.push('| 检查引擎 | 基线 | 新结果 | 对齐 | 漏检 | 多检 | 字段变化 |');
  L.push('| --- | --- | --- | --- | --- | --- | --- |');
  const engines = Array.from(res.stats.keys()).sort();
  for (const e of engines) {
    const s = res.stats.get(e);
    L.push('| ' + disp(e) + (s.outOfScope ? '（基线外）' : '') + ' | ' + (s.outOfScope && s.golden === 0 ? '-' : s.golden) +
      ' | ' + (s.outOfScope && s.fresh === 0 ? '-' : s.fresh) + ' | ' + (s.outOfScope ? '-' : s.aligned) +
      ' | ' + (s.outOfScope ? '-' : s.missing) + ' | ' + (s.outOfScope ? '-' : s.added) +
      ' | ' + (s.outOfScope ? '-' : s.changed) + ' |');
  }
  L.push('');

  // 漏检
  if (res.missing.length) {
    L.push('## ❌ 漏检（基线有、新结果无）— 疑似 skill 修改导致漏检');
    L.push('');
    res.missing.forEach((r, i) => {
      const mig = res.migrationOf(r);
      L.push((i + 1) + '. ' + recordBrief(r) + networkNote(r));
      L.push('   - 上下文：' + clip(r.context, 60));
      L.push('   - 描述：' + clip(r.description, 80));
      if (mig) {
        L.push('   - 疑似与 [' + mig.a.label + '] 为同一问题（相似度 ' + mig.sim.toFixed(2) +
          '），仅引擎/位置变化，请确认是否为规则迁移');
      }
    });
    L.push('');
  }

  // 多检
  if (res.added.length) {
    L.push('## ❌ 多检（新结果有、基线无）— 疑似新增检出或误检');
    L.push('');
    res.added.forEach((r, i) => {
      const mig = res.migrationOf(r);
      L.push((i + 1) + '. ' + recordBrief(r) + networkNote(r));
      L.push('   - 上下文：' + clip(r.context, 60));
      L.push('   - 描述：' + clip(r.description, 80));
      if (mig) {
        L.push('   - 疑似与基线 [' + mig.m.label + '] 为同一问题（相似度 ' + mig.sim.toFixed(2) +
          '），请确认是否为规则迁移');
      }
    });
    L.push('');
  }

  // 关键字段变化
  if (res.fieldChanges.length) {
    L.push('## ❌ 关键字段变化（同条问题，关键字段值改变）');
    L.push('');
    res.fieldChanges.forEach((c, i) => {
      L.push((i + 1) + '. [' + c.g.label + ' ↔ ' + c.n.label + '] ' + c.field + '：' + c.oldV + ' → ' + c.newV);
      L.push('   - ' + displayEngine(c.g.engine) + ' · L' + c.g.line + ' · 上下文：' + clip(c.g.context, 60));
    });
    L.push('');
  }

  // 文案差异
  if (res.wordingChanges.length) {
    L.push('## ⚠️ 文案差异（措辞变化，默认不阻断；--strict 可升级为阻断）');
    L.push('');
    const MAX_SHOW = 30;
    res.wordingChanges.slice(0, MAX_SHOW).forEach((c, i) => {
      L.push((i + 1) + '. [' + c.g.label + ' ↔ ' + c.n.label + '] ' + c.field + '（相似度 ' + c.sim.toFixed(2) + '）' +
        (SCRIPT_ENGINES.has(c.g.engine) ? ' 【脚本类引擎措辞变化，请确认工具/规则版本是否有变更】' : ''));
      L.push('   - 基线：' + clip(c.oldV, 90));
      L.push('   - 新结果：' + clip(c.newV, 90));
    });
    if (res.wordingChanges.length > MAX_SHOW) {
      L.push('… 其余 ' + (res.wordingChanges.length - MAX_SHOW) + ' 条文案差异略（详见基线与新结果文件 diff）');
    }
    L.push('');
  }

  // 说明
  L.push('## ℹ️ 说明');
  L.push('');
  if (NETWORK_ENGINES.size) {
    L.push('- link_validity-check 差异受网络环境影响（外链可访问性随环境变化），已标注【网络相关】，请人工确认后再定性');
  }
  if (res.nOut.length) {
    L.push('- 基线未覆盖引擎（' + Array.from(new Set(res.nOut.map(r => displayEngine(r.engine)))).join('、') +
      '）检出 ' + res.nOut.length + ' 条，不参与对齐判定；如需纳入，请更新 common_bench.md 基线');
  }
  if (res.enginesNoResult.length) {
    L.push('- 检查项无结果：' + res.enginesNoResult.map(disp).join('、') +
      '（新结果中该引擎 0 条，基线问题全部计为漏检，请确认该检查项是否执行/结果是否缺失）');
  }
  L.push('- 对齐口径：阻断=漏检/多检/严重程度/问题类型/行号/文件变化；上下文/描述/建议措辞差异默认仅提示（LLM 表述波动）');
  L.push('- 退出码：0=对齐；1=不对齐（存在阻断性差异）；2=输入错误');
  L.push('');
  L.push('对齐判定: ' + (res.aligned ? '对齐' : '不对齐') + '（exit ' + (res.aligned ? 0 : 1) + '）');
  return L.join('\n');
}

// ---------- CLI ----------

function usage() {
  console.error('用法:');
  console.error('  node bench.js convert --input=<result.json> --output=<bench.md>');
  console.error('  node bench.js compare --golden=<common_bench.md> --new=<bench.md|result.json> [--report=<out.md>] [--strict] [--label=<标签>] [--engines=a,b]');
  console.error('  node bench.js selftest');
  process.exit(2);
}

function parseKV(argv) {
  const args = {};
  for (const a of argv) {
    const m = /^--([a-z-]+)(?:=(.*))?$/.exec(a);
    if (!m) { console.error('未知参数: ' + a); usage(); }
    args[m[1]] = (m[2] === undefined || m[2] === '') ? true : m[2];
  }
  return args;
}

function readFileText(f) {
  try {
    return fs.readFileSync(f, 'utf8');
  } catch (e) {
    console.error('读取失败: ' + f + '（' + e.message + '）');
    process.exit(2);
  }
}

// 新结果输入：.json（聚合 {summary, issues} 或原子结果数组）或 .md（bench 同构）
function loadFresh(f) {
  const text = readFileText(f);
  if (path.extname(f).toLowerCase() === '.json') {
    let data;
    try {
      data = JSON.parse(text);
    } catch (e) {
      console.error('JSON 解析失败: ' + f + '（' + e.message + '）');
      process.exit(2);
    }
    const issues = Array.isArray(data) ? data : (data && Array.isArray(data.issues) ? data.issues : null);
    if (!issues) {
      console.error('输入既非 JSON 数组，也非含 issues 数组的聚合结果: ' + f);
      process.exit(2);
    }
    return issues.map((iss, i) => recordFromIssue(iss, i + 1));
  }
  const parsed = parseBenchMd(text);
  if (parsed.skippedBlocks.length) {
    console.error('警告: ' + f + ' 中 ' + parsed.skippedBlocks.length + ' 个问题块缺少字段，已跳过');
  }
  return parsed.records;
}

function cmdConvert(args) {
  if (!args.input || !args.output) usage();
  const text = readFileText(args.input);
  let data;
  try {
    data = JSON.parse(text);
  } catch (e) {
    console.error('JSON 解析失败: ' + args.input + '（' + e.message + '）');
    process.exit(2);
  }
  const issues = Array.isArray(data) ? data : (data && Array.isArray(data.issues) ? data.issues : null);
  if (!issues) {
    console.error('输入既非 JSON 数组，也非含 issues 数组的聚合结果: ' + args.input);
    process.exit(2);
  }
  const records = issues.map((iss, i) => recordFromIssue(iss, i + 1));
  const md = renderBenchMd(records, {
    note: '由 bench.js convert 生成：源 ' + path.basename(args.input) + '，共 ' + records.length +
      ' 条；仅用于与 common_bench.md 基线对齐比对。'
  });
  fs.mkdirSync(path.dirname(path.resolve(args.output)), { recursive: true });
  fs.writeFileSync(args.output, md + '\n', 'utf8');
  const byEngine = {};
  for (const r of records) byEngine[displayEngine(r.engine)] = (byEngine[displayEngine(r.engine)] || 0) + 1;
  console.log('转换完成: ' + args.output + '（共 ' + records.length + ' 条）');
  console.log('引擎分布: ' + Object.keys(byEngine).map(k => k + ' ' + byEngine[k]).join('、'));
}

function cmdCompare(args) {
  if (!args.golden || !args.new) usage();
  const goldenText = readFileText(args.golden);
  const goldenParsed = parseBenchMd(goldenText);
  if (goldenParsed.skippedBlocks.length) {
    console.error('警告: 基线中 ' + goldenParsed.skippedBlocks.length + ' 个问题块缺少字段，已跳过');
  }
  if (!goldenParsed.records.length) {
    console.error('基线解析失败（未解析到任何问题块）: ' + args.golden);
    process.exit(2);
  }
  const fresh = loadFresh(args.new);
  const engines = args.engines ? String(args.engines).split(',').map(s => s.trim()).filter(Boolean) : null;
  const res = compareRecords(goldenParsed.records, fresh, { strict: !!args.strict, engines: engines });
  const report = renderReport(res, {
    goldenPath: args.golden, newPath: args.new,
    label: args.label === true ? undefined : args.label
  });
  console.log(report);
  if (args.report && args.report !== true) {
    fs.mkdirSync(path.dirname(path.resolve(args.report)), { recursive: true });
    fs.writeFileSync(args.report, report + '\n', 'utf8');
    console.log('报告已写入: ' + args.report);
  }
  process.exit(res.aligned ? 0 : 1);
}

// ---------- 自检 ----------

// canonical → 管线各原子结果实际使用的 rule_item 写法（用于自检别名归一）
const PIPELINE_NAMES = {
  'punctuation-check': 'punctuation-check',
  'compliance-check': 'compliance-check',
  'filename-check': 'filename-check',
  'link-validity-check': 'link-validity-check',
  'aicheck': 'AICheck',
  'tag-closed-check': 'tag-closed-check',
  'resource-existence-check': 'resource-existence-check',
  'codespell-check': 'codespell-check'
};

function cloneRecord(r, extra) {
  const c = {
    engine: r.engine, type: r.type, file: r.file, line: r.line,
    context: r.context, description: r.description, suggestion: r.suggestion,
    severity: r.severity, rawIndex: r.rawIndex
  };
  return Object.assign(c, extra || {});
}

function cmdSelftest(args) {
  const goldenPath = args.golden || path.join(__dirname, '..', 'common_bench.md');
  const failures = [];
  let total = 0;
  const check = (name, ok, detail) => {
    total++;
    console.log((ok ? 'PASS' : 'FAIL') + '  ' + name + (detail ? '  — ' + detail : ''));
    if (!ok) failures.push(name);
  };

  // T1: 基线解析
  const goldenText = readFileText(goldenPath);
  const golden = parseBenchMd(goldenText).records;
  const engineCount = {};
  for (const r of golden) engineCount[r.engine] = (engineCount[r.engine] || 0) + 1;
  console.log('基线: ' + goldenPath + ' — ' + golden.length + ' 条');
  console.log('引擎分布: ' + Object.keys(engineCount).map(k => k + ' ' + engineCount[k]).join('、'));
  check('T1 基线解析（条数与字段完整性）',
    golden.length >= 50 &&
    golden.every(r => r.engine && r.severity && r.description !== '' && r.file),
    golden.length + ' 条');

  // T2: 自比对（基线 vs 基线副本）应完全对齐
  const selfRes = compareRecords(golden, cloneAll(golden), {});
  check('T2 自比对对齐',
    selfRes.aligned && selfRes.missing.length === 0 && selfRes.added.length === 0 &&
    selfRes.fieldChanges.length === 0 && selfRes.wordingChanges.length === 0,
    '配对 ' + selfRes.pairs.length);

  // T3: 变异检测（引擎改用管线写法 + 漏检/多检/严重程度/行号漂移/文案差异/基线外引擎）
  const fresh = cloneAll(golden).map(r => {
    r.engine = PIPELINE_NAMES[r.engine] || r.engine;
    return normalizeRecord(r);
  });
  // 变异前按基线 问题N（rawIndex）定位目标记录
  // 目标记录按基线实际存在的引擎动态选择（基线引擎分布会随重建变化，不硬编码）；
  // 引擎匹配做小写归一（基线用 AICheck 大写写法时也能命中 aicheck）；
  // 各变异目标互斥（dropLink/reword 回退时排除已被占用的目标），避免一条记录被两个变异覆盖
  const rawOf = eng => { const r = golden.find(x => x.engine.toLowerCase() === eng); return r ? r.rawIndex : null; };
  const dropPunct = rawOf('punctuation-check');    // 漏检 1（punctuation）
  const aiRaw = rawOf('aicheck');                  // 严重程度变化
  const compRaw = rawOf('compliance-check');       // 行号漂移
  const punctSecond = golden.find(x => x.engine === 'punctuation-check' && x.rawIndex !== dropPunct);
  // 文案差异目标：aicheck 第二条；不足时回退为任意未被其他变异占用的记录（排除重分类目标）
  const reservedRaws = new Set([dropPunct, aiRaw, compRaw, punctSecond ? punctSecond.rawIndex : null].filter(r => r !== null));
  let dropLink = rawOf('link-validity-check');     // 漏检 1（link，含网络标注）
  const hasLink = dropLink !== null;
  if (dropLink === null) {
    const other = golden.find(x => x.rawIndex !== null && !reservedRaws.has(x.rawIndex));
    dropLink = other ? other.rawIndex : null;
  }
  reservedRaws.add(dropLink);
  let rewordRec = golden.find(x => x.engine.toLowerCase() === 'aicheck' && x.rawIndex !== aiRaw && !reservedRaws.has(x.rawIndex));
  if (!rewordRec) rewordRec = golden.find(x => !reservedRaws.has(x.rawIndex));
  const rewordRaw = rewordRec ? rewordRec.rawIndex : null;
  for (const raw of [dropPunct, dropLink]) {
    const i = fresh.findIndex(r => r.rawIndex === raw);
    if (i >= 0) fresh.splice(i, 1);
  }
  const byRaw = raw => fresh.find(r => r.rawIndex === raw);
  const aiRec = byRaw(aiRaw);
  if (aiRec) aiRec.severity = aiRec.severity === '提示' ? '一般' : '提示';
  const compRec = byRaw(compRaw);
  if (compRec) compRec.line = compRec.line + 1;
  const rewordTarget = byRaw(rewordRaw);
  if (rewordTarget) rewordTarget.description = '（自检改写）' + rewordTarget.description;
  // 引擎重分类：一条 punctuation 记录改判 compliance-check → 形成 漏检+多检 对与"疑似迁移"提示
  const reclassTarget = punctSecond ? byRaw(punctSecond.rawIndex) : null;
  if (reclassTarget) reclassTarget.engine = 'compliance-check';
  fresh.forEach((r, i) => { r.rawIndex = i + 1; });
  fresh.push(normalizeRecord({
    engine: 'punctuation-check', type: '规范性', file: 'cann_sim工具指南.md', line: 999,
    context: '自检构造样例上下文', description: '自检构造：多检验证', suggestion: '无',
    severity: '提示', rawIndex: fresh.length + 1
  }));
  fresh.push(normalizeRecord({
    engine: 'markdownlint', type: '规范性', file: 'cann_sim工具指南.md', line: 999,
    context: '自检构造样例上下文', description: '自检构造：基线外引擎', suggestion: '无',
    severity: '提示', rawIndex: fresh.length + 1
  }));
  const mutRes = compareRecords(golden, fresh, {});
  const expectMissing = 3, expectAdded = 2, expectChanged = 2;
  check('T3 变异检测（漏检/多检/字段变化）',
    mutRes.missing.length === expectMissing && mutRes.added.length === expectAdded &&
    mutRes.fieldChanges.length === expectChanged && !mutRes.aligned,
    '漏检 ' + mutRes.missing.length + '（期望 ' + expectMissing + '）、多检 ' + mutRes.added.length +
    '（期望 ' + expectAdded + '）、字段变化 ' + mutRes.fieldChanges.length + '（期望 ' + expectChanged + '）');
  const hasSeverityChange = mutRes.fieldChanges.some(c => c.field === '严重程度');
  const hasLineDrift = mutRes.fieldChanges.some(c => c.field.indexOf('行号') === 0);
  const mutReportText = renderReport(mutRes, { goldenPath: goldenPath, newPath: '(变异)' });
  // 网络标注仅当基线含 link-validity-check 条目（被删除形成漏检）时才会出现；无 link 基线下跳过该子断言
  const hasNetworkFlag = mutReportText.indexOf('网络相关') >= 0;
  check('T3b 字段变化类别与网络标注',
    hasSeverityChange && hasLineDrift && (!hasLink || hasNetworkFlag),
    '严重程度变化=' + hasSeverityChange + '、行号漂移=' + hasLineDrift + '、网络标注=' + hasNetworkFlag +
    (hasLink ? '' : '（基线无 link 条目，跳过网络标注断言）'));
  check('T3e 疑似迁移提示',
    mutRes.migrations.length === 1 &&
    mutReportText.indexOf('疑似与基线 [') >= 0 && mutReportText.indexOf('疑似与 [') >= 0,
    '迁移对 ' + mutRes.migrations.length + '（期望 1），报告含迁移提示=' +
    (mutReportText.indexOf('疑似与基线 [') >= 0 && mutReportText.indexOf('疑似与 [') >= 0));
  const mutWording = mutRes.wordingChanges.length;
  check('T3c 文案差异不阻断',
    mutWording >= 1 && !mutRes.strict && mutRes.aligned === false &&
    mutRes.blockingCount === expectMissing + expectAdded + expectChanged,
    '文案差异 ' + mutWording + ' 条，阻断数 ' + mutRes.blockingCount);
  const strictRes = compareRecords(golden, fresh, { strict: true });
  check('T3d --strict 文案差异升级阻断',
    !strictRes.aligned && strictRes.blockingCount === expectMissing + expectAdded + expectChanged + mutWording,
    '阻断数 ' + strictRes.blockingCount);

  // T4: convert 往返（golden 记录 → 管线 11 字段 JSON → bench md → 解析 → 比对应完全对齐）
  const issues = golden.map(r => ({
    rule_group: '通用规则',
    rule_item: PIPELINE_NAMES[r.engine] || r.engine,
    rule_type: r.type,
    rule_name: '',
    snippet: r.context,
    description: r.description,
    rule_level: r.severity,
    file_name: r.file,
    line_number: String(r.line),
    location: 'E:/benchmark/' + r.file + ':' + (r.line || ''),
    suggestion: r.suggestion
  }));
  const convertedMd = renderBenchMd(issues.map((iss, i) => recordFromIssue(iss, i + 1)), {
    note: 'selftest round-trip'
  });
  const roundTrip = parseBenchMd(convertedMd).records;
  const rtRes = compareRecords(golden, roundTrip, {});
  check('T4 convert 往返对齐',
    rtRes.aligned && rtRes.wordingChanges.length === 0 && rtRes.pairs.length === golden.length,
    '配对 ' + rtRes.pairs.length + '/' + golden.length);

  console.log('');
  if (failures.length) {
    console.error('自检失败: ' + failures.join('、'));
    process.exit(1);
  }
  console.log('自检全部通过（' + total + ' 项）');
}

function cloneAll(records) {
  return records.map(r => cloneRecord(r));
}

function main() {
  const cmd = process.argv[2];
  const rest = process.argv.slice(3);
  if (cmd === 'convert') return cmdConvert(parseKV(rest));
  if (cmd === 'compare') return cmdCompare(parseKV(rest));
  if (cmd === 'selftest') return cmdSelftest(parseKV(rest));
  usage();
}

module.exports = {
  normText, canonEngine, displayEngine, normLine,
  normalizeRecord, recordFromIssue, parseBenchMd, renderBenchMd,
  sim, recordSim, matchRecords, compareRecords, renderReport,
  ENGINE_ALIAS, ENGINE_DISPLAY, FIELD_LABELS
};

if (require.main === module) main();
