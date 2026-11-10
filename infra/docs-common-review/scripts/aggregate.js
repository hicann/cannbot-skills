#!/usr/bin/env node
// SPDX-License-Identifier: CANN-2.0
/**
 * docs-common-review 聚合脚本
 *
 * 合并各检查结果文件，产出统一质检报告（默认 output/docs-common-review-result.json）：
 *   - 规范化：rule_group 统一"通用规则"；location 去 vscode:// 等协议前缀、反斜杠统一为 /
 *   - 固定排序（文件名 → 行号 → 检查项 → 描述），同输入产出逐字节一致（评测可复现）
 *   - summary：模式 / 总数 / 级别分布 / 检查项分布 / 检查项耗时(check_timing_ms) / 输入使用情况；
 *     可选 failed_check（仅当 --failed-file 提供且非空时存在）：未成功执行的检查项清单 [{check, reason}]
 *   - 门禁模式额外含 gate 判定（failed_check 非空时 passed 同样不可信）
 *   - 可选 csv / md 转换（--format=csv,md，与 JSON 同目录同名异扩展名）
 *
 * 用法：
 *   node aggregate.js --inputs=<f1,f2,...> --mode=<ci-gated|full|custom> --output=<file> [--timing=<file|dir>] [--failed-file=<file>] [--format=csv,md] [--stamp]
 *
 * 退出码：0 成功；1 用法错误；2 无任何可用输入。
 */

'use strict';

const fs = require('fs');
const path = require('path');

const FIELDS = [
  'rule_group', 'rule_item', 'rule_type', 'rule_name', 'snippet', 'description',
  'rule_level', 'file_name', 'line_number', 'location', 'suggestion'
];

// 社区 docs-ci 真实拦截项（LANJIE_TOOLS）：link / resource / tag_closed / compliance(标签+编码)
// compliance 含 canonical（compliance-check）与历史写法（StaticCheck_compliance*），两者均计拦截
const BLOCKING_ITEMS = [
  'link-validity-check', 'resource-existence-check', 'tag-closed-check',
  'compliance-check', 'StaticCheck_compliance', 'StaticCheck_compliance_check'
];

// CI 中仅告警、不拦截的项
const WARNING_ITEMS = ['markdownlint', 'codespell-check'];

function usage() {
  console.error('用法: node aggregate.js --inputs=<f1,f2,...> --mode=<ci-gated|full|custom> --output=<file> [--timing=<file|dir>] [--failed-file=<file>] [--format=csv,md] [--stamp]');
  process.exit(1);
}

function parseArgs(argv) {
  const args = { format: [], stamp: false };
  for (const a of argv) {
    if (a === '--stamp') { args.stamp = true; continue; }
    const m = /^--([a-z-]+)=(.*)$/.exec(a);
    if (!m) { console.error('未知参数: ' + a); usage(); }
    const key = m[1];
    const val = m[2];
    if (key === 'inputs') {
      args.inputs = val.split(',').map(s => s.trim()).filter(Boolean);
    } else if (key === 'mode') {
      args.mode = val;
    } else if (key === 'output') {
      args.output = val;
    } else if (key === 'timing') {
      args.timing = val;
    } else if (key === 'failed-file') {
      args.failedFile = val;
    } else if (key === 'format') {
      args.format = val.split(',').map(s => s.trim().toLowerCase()).filter(Boolean);
    } else {
      console.error('未知参数: ' + a);
      usage();
    }
  }
  if (!args.inputs || args.inputs.length === 0) usage();
  if (!args.mode || !args.output) usage();
  if (['ci-gated', 'full', 'custom'].indexOf(args.mode) < 0) {
    console.error('非法 mode: ' + args.mode + '（合法值: ci-gated / full / custom）');
    usage();
  }
  for (const f of args.format) {
    if (['csv', 'md'].indexOf(f) < 0) {
      console.error('非法 format: ' + f + '（合法值: csv / md）');
      usage();
    }
  }
  return args;
}

// 规范化单条问题：统一 11 字段、rule_group 固定、location 去协议前缀并统一正斜杠
function normalizeIssue(raw) {
  const src = (raw && typeof raw === 'object') ? raw : {};
  const issue = {};
  for (const f of FIELDS) {
    issue[f] = (src[f] === undefined || src[f] === null) ? '' : String(src[f]);
  }
  issue.rule_group = '通用规则';
  issue.location = normalizeLocation(issue.location);
  return issue;
}

function normalizeLocation(loc) {
  if (!loc) return '';
  let s = String(loc).trim();
  if (s.indexOf('vscode://file/') === 0) s = s.slice('vscode://file/'.length);
  else if (s.indexOf('vscode://') === 0) s = s.slice('vscode://'.length);
  return s.split('\\').join('/');
}

// 固定排序：文件名 → 行号（数值，无行号排最后）→ 检查项 → 描述
function lineKey(n) {
  const x = parseInt(n, 10);
  return Number.isFinite(x) ? x : Number.POSITIVE_INFINITY;
}

function cmpStr(a, b) {
  return a < b ? -1 : (a > b ? 1 : 0);
}

function compareIssues(a, b) {
  let c = cmpStr(a.file_name, b.file_name);
  if (c) return c;
  const la = lineKey(a.line_number);
  const lb = lineKey(b.line_number);
  if (la !== lb) return la - lb;
  c = cmpStr(a.rule_item, b.rule_item);
  if (c) return c;
  return cmpStr(a.description, b.description);
}

// 读取输入文件：JSON 数组 → 逐条规范化；不存在 / 解析失败 / 非数组 → 记入 skipped（判定保持可信）
function loadInputs(files) {
  const issues = [];
  const used = [];
  const skipped = [];
  for (const f of files) {
    let data;
    try {
      const text = fs.readFileSync(f, 'utf8');
      data = JSON.parse(text);
    } catch (e) {
      skipped.push({ file: f, reason: '不存在或解析失败' });
      continue;
    }
    if (!Array.isArray(data)) {
      skipped.push({ file: f, reason: '非 JSON 数组' });
      continue;
    }
    for (const raw of data) issues.push(normalizeIssue(raw));
    used.push(f);
  }
  return { issues: issues, used: used, skipped: skipped };
}

// 读取耗时输入：--timing 支持单文件、目录（读取其中全部 .txt，按文件名排序保证确定性）或逗号分隔多个
// 每个文件每行 "<检查项名> <毫秒数>"；空行 / # 注释 / 格式错误行跳过；同名条目取最后一条（重派覆盖）
function loadTiming(val) {
  const out = {};
  if (!val) return out;
  const files = [];
  for (const p of String(val).split(',')) {
    const t = p.trim();
    if (!t) continue;
    let st = null;
    try {
      st = fs.statSync(t);
    } catch (e) {
      continue;
    }
    if (st.isDirectory()) {
      for (const n of fs.readdirSync(t).filter(n => /\.txt$/.test(n)).sort()) {
        files.push(path.join(t, n));
      }
    } else {
      files.push(t);
    }
  }
  for (const f of files) {
    let text;
    try {
      text = fs.readFileSync(f, 'utf8');
    } catch (e) {
      continue;
    }
    for (const line of text.split(/\r?\n/)) {
      const ln = line.trim();
      if (!ln || ln.indexOf('#') === 0) continue;
      const m = /^(\S+)\s+(\d+)$/.exec(ln);
      if (m) out[m[1]] = parseInt(m[2], 10);
    }
  }
  return out;
}

// 读取失败检查项清单：--failed-file 指向 JSON 数组（元素为字符串或 {check, reason}）
// 文件不存在 / 解析失败 / 非数组 → stderr 告警并按无失败处理（不阻断报告产出）
function loadFailed(p) {
  if (!p) return [];
  let data;
  try {
    data = JSON.parse(fs.readFileSync(p, 'utf8'));
  } catch (e) {
    console.error('警告: 失败清单读取失败(' + p + ')，按无失败处理: ' + e.message);
    return [];
  }
  if (!Array.isArray(data)) {
    console.error('警告: 失败清单不是 JSON 数组(' + p + ')，按无失败处理');
    return [];
  }
  const out = [];
  for (const raw of data) {
    if (typeof raw === 'string') {
      const check = raw.trim();
      if (check) out.push({ check: CANONICAL_ITEMS[check] || check, reason: '' });
      continue;
    }
    if (raw && typeof raw === 'object') {
      const check = String(raw.check || '').trim();
      if (!check) continue;
      const reason = String(raw.reason || '').trim();
      out.push({ check: CANONICAL_ITEMS[check] || check, reason: reason });
    }
  }
  return out;
}

// 各检查项历史/原子口径 → canonical 名（仅用于 summary.by_check 统计键，不改写 issues 原文）
// aicheck 原子规则库（docs-basic-aicheck/references/rules.md）使用大写 AICheck；聚合统计统一为小写 aicheck
// filename 历史写法（StaticCheck_filename_check）已在原子脚本源头改为 filename-check，此处保留兼容旧中间产物
const CANONICAL_ITEMS = {
  'AICheck': 'aicheck',
  'StaticCheck_filename_check': 'filename-check'
};

function buildSummary(mode, issues, used, skipped, stamp, timing, failed) {
  const byLevel = {};
  const byCheck = {};
  for (const i of issues) {
    const lv = i.rule_level || '未知';
    byLevel[lv] = (byLevel[lv] || 0) + 1;
    const ck = CANONICAL_ITEMS[i.rule_item] || i.rule_item || '未知';
    byCheck[ck] = (byCheck[ck] || 0) + 1;
  }
  const summary = {
    mode: mode,
    total: issues.length,
    by_level: byLevel,
    by_check: byCheck,
    inputs: { used: used.length, skipped: skipped }
  };
  if (failed && failed.length) summary.failed_check = failed;
  if (timing && Object.keys(timing).length) summary.check_timing_ms = timing;
  if (mode === 'ci-gated') {
    const blocking = issues.filter(i => BLOCKING_ITEMS.indexOf(i.rule_item) >= 0);
    const warnings = issues.filter(i => WARNING_ITEMS.indexOf(i.rule_item) >= 0);
    const failedCount = (failed || []).length;
    summary.gate = {
      passed: blocking.length === 0 && skipped.length === 0 && failedCount === 0,
      blocking_issues: blocking.length,
      warning_issues: warnings.length,
      note: '拦截项=link/resource/tag-closed/compliance(标签+编码)；markdownlint/codespell 为告警级；skipped 或 failed_check 非空时 passed 不可信。'
    };
  }
  if (stamp) summary.generated_at = new Date().toISOString();
  return summary;
}

function csvEscape(v) {
  const s = (v === undefined || v === null) ? '' : String(v);
  if (/[",\r\n]/.test(s)) return '"' + s.split('"').join('""') + '"';
  return s;
}

function toCsv(issues) {
  const lines = [FIELDS.map(csvEscape).join(',')];
  for (const i of issues) {
    lines.push(FIELDS.map(f => csvEscape(i[f])).join(','));
  }
  return '﻿' + lines.join('\r\n') + '\r\n';
}

function mdEscape(v) {
  let s = String(v === undefined || v === null ? '' : v);
  s = s.split('\n').join(' ').split('\r').join(' ');
  s = s.split('|').join('\\|');
  return s.replace(/\s+$/, '');
}

function toMd(summary, issues) {
  const lvl = k => (summary.by_level[k] || 0);
  const lines = [];
  lines.push('# 通用文档质检报告');
  lines.push('');
  lines.push('- 模式：' + summary.mode);
  lines.push('- 问题总数：' + summary.total + '（严重 ' + lvl('严重') + ' / 一般 ' + lvl('一般') + ' / 提示 ' + lvl('提示') + '）');
  if (summary.gate) {
    lines.push('- 门禁判定：' + (summary.gate.passed ? '通过' : '未通过') +
      '（拦截项 ' + summary.gate.blocking_issues + '，告警项 ' + summary.gate.warning_issues + '）');
  }
  if (summary.failed_check && summary.failed_check.length) {
    lines.push('- ⚠️ 检查项执行失败（结果不完整）：' + summary.failed_check
      .map(f => f.check + (f.reason ? '（' + f.reason + '）' : '')).join('、'));
  }
  const checkKeys = Object.keys(summary.by_check);
  if (checkKeys.length) {
    lines.push('- 检查项分布：' + checkKeys.map(k => k + ' ' + summary.by_check[k]).join('、'));
  }
  if (summary.check_timing_ms) {
    const tk = Object.keys(summary.check_timing_ms);
    lines.push('- 检查耗时：' + tk.map(k => k + ' ' + (summary.check_timing_ms[k] / 1000).toFixed(1) + 's').join('、'));
  }
  if (summary.inputs.skipped.length) {
    lines.push('- 注意（结果不完整）：' + summary.inputs.skipped
      .map(s => path.basename(s.file) + ' ' + s.reason).join('；'));
  }
  lines.push('');
  lines.push('| 级别 | 检查项 | 文件 | 行号 | 问题描述 | 修改建议 |');
  lines.push('| --- | --- | --- | --- | --- | --- |');
  for (const i of issues) {
    const row = [
      i.rule_level, i.rule_item, mdEscape(i.file_name),
      i.line_number === '' ? '-' : i.line_number,
      mdEscape(i.description), mdEscape(i.suggestion)
    ];
    lines.push('| ' + row.join(' | ') + ' |');
  }
  return lines.join('\n') + '\n';
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const loaded = loadInputs(args.inputs);
  if (loaded.used.length === 0) {
    console.error('聚合失败：没有任何可用的输入结果文件。');
    for (const s of loaded.skipped) console.error('  - ' + s.file + '：' + s.reason);
    process.exit(2);
  }
  loaded.issues.sort(compareIssues);
  const failed = loadFailed(args.failedFile);
  const summary = buildSummary(args.mode, loaded.issues, loaded.used, loaded.skipped, args.stamp, loadTiming(args.timing), failed);
  const report = { summary: summary, issues: loaded.issues };

  fs.mkdirSync(path.dirname(args.output), { recursive: true });
  fs.writeFileSync(args.output, JSON.stringify(report, null, 2) + '\n', 'utf8');
  console.log('聚合完成: ' + args.output +
    '（问题 ' + loaded.issues.length + ' 条' +
    (loaded.skipped.length ? '，跳过输入 ' + loaded.skipped.length + ' 个' : '') + '）');

  const base = args.output.replace(/\.json$/i, '');
  if (args.format.indexOf('csv') >= 0) {
    const p = base + '.csv';
    fs.writeFileSync(p, toCsv(loaded.issues), 'utf8');
    console.log('CSV 已生成: ' + p);
  }
  if (args.format.indexOf('md') >= 0) {
    const p = base + '.md';
    fs.writeFileSync(p, toMd(summary, loaded.issues), 'utf8');
    console.log('Markdown 已生成: ' + p);
  }
}

main();

