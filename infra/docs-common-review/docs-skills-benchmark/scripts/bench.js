#!/usr/bin/env node
// SPDX-License-Identifier: CANN-2.0
/**
 * docs-skills-benchmark 基准工具（bench.js）
 *
 * 用途：docs-common-review 家族的基准回归测试。固定输入样例 bench测试数据集.md，
 * 以 common_bench.md 为基线（golden），检测 skill 修改后检查结果是否仍与基线对齐；
 * 不对齐时输出差异明细（漏检 / 多检 / 关键字段变化 / 文案差异）反馈给用户。
 *
 * 子命令：
 *   convert   聚合结果 JSON（{summary, issues} 或 11 字段数组）→ common_bench.md 同构的"问题N" Markdown
 *   compare   基线与新结果对齐比对（新结果支持 bench md / 聚合 JSON / 原子结果数组 JSON）
 *             输出对齐报告；退出码 0=对齐，1=不对齐（存在阻断性差异），2=输入错误
 *             --new 传聚合 JSON 时附带执行完备性硬校验（summary.failed_check 非空 → 强制不对齐）
 *   perf      耗时维度（提示级，不阻断对齐判定）：
 *             perf convert  聚合 JSON 的 summary.check_timing_ms → 耗时快照 md（runs/perf_bench_test.md）
 *             perf compare  耗时基线（perf_bench.md）与本次快照/聚合 JSON 对比 → 耗时对比报告
 *             退出码 0=已生成报告（提示级，不代表对齐结论），2=输入错误
 *   selftest  工具自检（golden 解析 / 引擎别名归一 / 匹配 / 差异判定 / convert 往返 /
 *             perf 提取与快照往返 / 耗时对比判定 / 执行完备性硬校验 /
 *             测试集保护判定 / 格式规范校验）
 *   protect   测试集保护校验：更新候选相对前一版基线，只允许“修改”与“新增”；
 *             “删除”必须逐条（检查引擎+位置行号）提供删除原因与必要证据（--reasons 文件）
 *             退出码 0=通过（无删除或全部删除已证据化），1=保护违规（存在未证据化删除），
 *             2=输入错误；exit 1 时禁止以候选覆盖基线
 *   format-check 问题集格式规范校验：编号连续、8 字段齐全且有序、引擎白名单、
 *             行号非负整数、问题类型/严重程度枚举、字段非空；--strict-ctx 另校验
 *             问题上下文反引号包裹（byself 本地约定）
 *             退出码 0=合规，1=存在违规，2=输入错误
 *
 * 用法：
 *   node bench.js convert --input=<docs-common-review-result.json> --output=<runs/common_bench_test.md>
 *   node bench.js compare --golden=<common_bench.md> --new=<runs/common_bench_test.md|result.json> \
 *                         [--report=<runs/align_report.md>] [--strict] [--label=<运行标签>] [--engines=a,b]
 *   node bench.js perf convert --input=<result.json> --output=<runs/perf_bench_test.md> [--label=<运行标签>]
 *   node bench.js perf compare --golden=<perf_bench.md> --new=<runs/perf_bench_test.md|result.json> \
 *                               [--report=<runs/perf_report.md>] [--label=<运行标签>]
 *             （内置 perf_bench.md 仅为量级参考，作为 --golden 时拒绝执行）
 *   node bench.js protect --old=<common_bench.md> --new=<runs/common_bench_test.md|result.json> \
 *                          [--reasons=<删除理由.json>] [--report=<runs/protect_report.md>] [--label=<运行标签>]
 *   node bench.js format-check --input=<common_bench.md|common_bench_byself.md> [--strict-ctx]
 *   node bench.js selftest
 *
 * 对齐判定（compare）：
 *   匹配：6 维度加权评分（引擎前置门槛；类型 1 + 严重程度 1 硬门 + 行号 0~2 + 上下文 0~2 +
 *         描述 0~2 + 建议 0~1，满分 9，中文 2 字 n-gram）；P1 主匹配 ≥6 一对一贪心；
 *         P2 严重程度变化诊断 ≥5.5（满分 8）；P2b 低置信度兜底 ≥5.5（行号分 ≥1、描述分 ≥0.5）；
 *         P3 重复检出吸收 ≥5.5（并入已配对基线，不阻断、不计多检）；漏检↔多检 疑似迁移提示
 *   阻断（不对齐，需反馈）：
 *     - 漏检：基线有、新结果无（疑似 skill 修改导致漏检）
 *     - 多检：新结果有、基线无（疑似新增检出 / 误检）
 *     - 严重程度变化 / 问题类型变化 / 行号漂移 / 文件变化（同条问题的关键字段改变）
 *     - --strict 时，上下文/描述/建议的文案差异也计入阻断
 *     - 执行完备性（仅 --new 为聚合 JSON 时可校验）：summary.failed_check 非空 → 本次结果不完整，
 *       强制不对齐，失败检查项逐项列入报告
 *   非阻断（仅提示）：
 *     - 上下文/描述/建议的措辞差异（LLM 表述波动，附相似度）
 *     - 基线未覆盖引擎（如 markdownlint）的检出，不计入对齐判定
 *     - 重复检出吸收（同一缺陷的重复报告并入已配对基线，报告单列供人工复核）
 *   网络注记：link-validity-check 的差异受网络环境影响，需人工确认后再定性
 *
 * 耗时判定（perf compare，提示级，不改变 compare 的对齐结论）：
 *   - 数据源：聚合 JSON 的 summary.check_timing_ms（脚本项=命令计时，模型项=subagent 墙钟）；
 *     多目标计时键（filename-check-1/2）提取时折叠求和归一
 *   - 显著回归（相对与绝对须同时超过）：脚本项 >+50% 且 >+10s；模型项 >+100% 且 >+60s
 *     （模型项为 subagent 墙钟：LLM 出 token + API 延迟，天然波动大故放宽）
 *   - 对称标记显著改善；本次无计时（结合执行完备性确认）；基线未收录（新增计时项）
 *   - 硬约束：仅同环境可比（换环境先重建耗时基线）；首跑一次性成本须预热后取第二跑
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

// 模型类引擎（subagent 执行，墙钟波动大 → 耗时对比使用宽阈值）
const MODEL_ENGINES = new Set(['punctuation-check', 'compliance-check', 'aicheck']);
// 耗时对比阈值（提示级判定）：相对与绝对须同时超过才标记显著
const PERF_THRESHOLD = {
  script: { relPct: 50, absMs: 10000 },  // 脚本项：>+50% 且 >+10s
  model: { relPct: 100, absMs: 60000 }    // 模型项：>+100% 且 >+60s
};

const MIGRATION_THRESHOLD = 0.55; // 漏检↔多检 疑似同一问题（引擎/位置迁移）的相似度阈值

// 6 维度加权匹配（满分 9）：问题类型 1 + 严重程度 1 + 行号 0~2 + 上下文 0~2 + 描述 0~2 + 建议 0~1
const MATCH_THRESHOLD = 6;       // 主匹配命中线
const SOFT_THRESHOLD = 5.5;     // 低置信度兜底 / 重复吸收的放宽命中线
const SEV_DIAG_THRESHOLD = 5.5; // 严重程度变化诊断线（不含严重程度 1 分，满分 8）

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

// ---------- 匹配（6 维度加权评分，多阶段配对） ----------
// 引擎恒为前置门槛；同引擎两两评分，按总分多阶段一对一/吸收配对：
//   P1 主匹配：问题类型+严重程度双门 + 总分 ≥ MATCH_THRESHOLD，按总分降序一对一贪心
//   P2 严重程度变化诊断：仅类型门 + 严重程度不同 + 其余分值（满分 8）≥ SEV_DIAG_THRESHOLD
//       → 配对成功但记为"严重程度"字段变化（阻断），保留旧口径的关键字段变化能力
//   P2b 低置信度兜底：双门 + 总分 ≥ SOFT_THRESHOLD + 行号分 ≥1 + 描述分 ≥0.5
//       → 配对成功（对齐），报告中标注低置信度供人工复核
//   P3 重复检出吸收：仍未配对的新结果 → 并入已配对基线（同一缺陷的重复报告，不阻断、不计多检）
//       条件：双门 + 总分 ≥ SOFT_THRESHOLD + 行号分 ≥0.5 + 描述分 ≥1.5
//   剩余：golden 侧 → 漏检；fresh 侧 → 多检

// 中文 2 字 n-gram（bigram）+ 英文/数字 token（≥2 字符，小写归一）
function textUnits(s) {
  const t = String(s === undefined || s === null ? '' : s);
  const units = new Set();
  let cjk = '';
  let aln = '';
  const flushCjk = () => {
    const arr = Array.from(cjk);
    if (arr.length === 1) units.add(arr[0]);
    for (let i = 0; i < arr.length - 1; i++) units.add(arr[i] + arr[i + 1]);
    cjk = '';
  };
  const flushAln = () => { if (aln.length >= 2) units.add(aln.toLowerCase()); aln = ''; };
  for (const ch of t) {
    if (/[\u4e00-\u9fff]/.test(ch)) { flushAln(); cjk += ch; }
    else if (/[0-9A-Za-z]/.test(ch)) { flushCjk(); aln += ch; }
    else { flushCjk(); flushAln(); }
  }
  flushCjk();
  flushAln();
  return units;
}

function unitOverlap(a, b) {
  const A = textUnits(a);
  const B = textUnits(b);
  let n = 0;
  for (const x of A) if (B.has(x)) n++;
  return n;
}

// 行号分（record 行号已由 normLine 归一为数字；区间行号取起始行）
function scoreLine(gLine, nLine) {
  const a = Number(gLine);
  const b = Number(nLine);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return 0;
  const d = Math.abs(a - b);
  if (d === 0) return 2;
  if (d <= 2) return 1;
  if (d <= 5) return 0.5;
  return 0;
}

// 上下文分：精确 2；包含 1.5；bigram 部分匹配（≥3 得 1，≥1 得 0.5）
function scoreContext(g, n) {
  const A = normText(g).replace(/[`\s]/g, '');
  const B = normText(n).replace(/[`\s]/g, '');
  if (!A || !B) return 0;
  if (A === B) return 2;
  if (A.includes(B) || B.includes(A)) return 1.5;
  const o = unitOverlap(g, n);
  if (o >= 3) return 1;
  if (o >= 1) return 0.5;
  return 0;
}

// 描述分：中文二元组高度重合（≥5）得 2；较重合（≥3）得 1.5；部分（≥1）得 0.5
function scoreDesc(g, n) {
  const o = unitOverlap(g, n);
  if (o >= 5) return 2;
  if (o >= 3) return 1.5;
  if (o >= 1) return 0.5;
  return 0;
}

// 建议分：中文二元组较重合（≥3）得 1；部分（≥1）得 0.5
function scoreSuggestion(g, n) {
  const o = unitOverlap(g, n);
  if (o >= 3) return 1;
  if (o >= 1) return 0.5;
  return 0;
}

// 综合评分：问题类型/严重程度为硬门（相等才各得 1 分）
function scorePair(g, n) {
  const typeOk = !!g.type && normText(g.type) === normText(n.type);
  const sevOk = !!g.severity && normText(g.severity) === normText(n.severity);
  const sLine = scoreLine(g.line, n.line);
  const sCtx = scoreContext(g.context, n.context);
  const sDesc = scoreDesc(g.description, n.description);
  const sSug = scoreSuggestion(g.suggestion, n.suggestion);
  const total = (typeOk ? 1 : 0) + (sevOk ? 1 : 0) + sLine + sCtx + sDesc + sSug;
  return { total: total, typeOk: typeOk, sevOk: sevOk, sLine: sLine, sCtx: sCtx, sDesc: sDesc, sSug: sSug };
}

function matchRecords(golden, fresh) {
  // 同引擎候选对全集
  const cands = [];
  const candByN = new Map(); // ni → [{gi, sc}]
  for (let gi = 0; gi < golden.length; gi++) {
    for (let ni = 0; ni < fresh.length; ni++) {
      if (golden[gi].engine !== fresh[ni].engine) continue;
      const sc = scorePair(golden[gi], fresh[ni]);
      const c = { gi: gi, ni: ni, sc: sc };
      cands.push(c);
      if (!candByN.has(ni)) candByN.set(ni, []);
      candByN.get(ni).push(c);
    }
  }
  const usedG = new Array(golden.length).fill(false);
  const usedN = new Array(fresh.length).fill(false);
  const pairs = [];
  const sevPairs = [];
  const byScoreDesc = (a, b) => (b.sc.total - a.sc.total) || (a.gi - b.gi) || (a.ni - b.ni);

  // P1 主匹配
  cands.filter(c => c.sc.typeOk && c.sc.sevOk && c.sc.total >= MATCH_THRESHOLD)
    .sort(byScoreDesc)
    .forEach(c => {
      if (!usedG[c.gi] && !usedN[c.ni]) {
        usedG[c.gi] = true;
        usedN[c.ni] = true;
        pairs.push({ g: c.gi, n: c.ni, score: c.sc.total, crossLine: c.sc.sLine < 2,
          parts: c.sc, lowConfidence: false, sevChanged: false });
      }
    });

  // P2 严重程度变化诊断（不含严重程度 1 分，满分 8）
  cands.filter(c => !usedG[c.gi] && !usedN[c.ni] && c.sc.typeOk && !c.sc.sevOk &&
    c.sc.total >= SEV_DIAG_THRESHOLD)
    .sort(byScoreDesc)
    .forEach(c => {
      if (!usedG[c.gi] && !usedN[c.ni]) {
        usedG[c.gi] = true;
        usedN[c.ni] = true;
        sevPairs.push({ g: c.gi, n: c.ni, score: c.sc.total, crossLine: c.sc.sLine < 2,
          parts: c.sc, lowConfidence: false, sevChanged: true });
      }
    });

  // P2b 低置信度兜底
  cands.filter(c => !usedG[c.gi] && !usedN[c.ni] && c.sc.typeOk && c.sc.sevOk &&
    c.sc.total >= SOFT_THRESHOLD && c.sc.sLine >= 1 && c.sc.sDesc >= 0.5)
    .sort(byScoreDesc)
    .forEach(c => {
      if (!usedG[c.gi] && !usedN[c.ni]) {
        usedG[c.gi] = true;
        usedN[c.ni] = true;
        pairs.push({ g: c.gi, n: c.ni, score: c.sc.total, crossLine: c.sc.sLine < 2,
          parts: c.sc, lowConfidence: true, sevChanged: false });
      }
    });

  // P3 重复检出吸收：仍未配对的 fresh → 并入已配对 golden（同一缺陷的重复报告）
  const absorbed = [];
  for (let ni = 0; ni < fresh.length; ni++) {
    if (usedN[ni]) continue;
    let best = null;
    for (const c of (candByN.get(ni) || [])) {
      if (!usedG[c.gi]) continue; // 只并入已配对基线
      if (!c.sc.typeOk || !c.sc.sevOk) continue;
      if (c.sc.total < SOFT_THRESHOLD || c.sc.sLine < 0.5 || c.sc.sDesc < 1.5) continue;
      if (!best || c.sc.total > best.sc.total) best = c;
    }
    if (best) {
      usedN[ni] = true;
      absorbed.push({ n: ni, g: best.gi, score: best.sc.total, parts: best.sc });
    }
  }

  return {
    pairs: pairs,
    sevPairs: sevPairs,
    absorbed: absorbed,
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
  // P1/P2b 主匹配对 + P2 严重程度变化诊断对统一走字段比对
  // （严重程度诊断对 g.severity ≠ n.severity，由 BLOCK_FIELDS 自然产出"严重程度"字段变化）
  for (const p of matched.pairs.concat(matched.sevPairs)) {
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
      stats.set(engine, { golden: 0, fresh: 0, aligned: 0, missing: 0, added: 0, absorbed: 0, changed: 0, outOfScope: false });
    }
    return stats.get(engine);
  };
  for (const r of gIn) stat(r.engine).golden++;
  for (const r of nIn) stat(r.engine).fresh++;
  for (const r of matched.missing) stat(r.engine).missing++;
  for (const r of matched.added) stat(r.engine).added++;
  for (const a of matched.absorbed) stat(nIn[a.n].engine).absorbed++;
  for (const c of fieldChanges) stat(c.g.engine).changed++;
  for (const p of matched.pairs) stat(gIn[p.g].engine).aligned++;
  for (const p of matched.sevPairs) stat(gIn[p.g].engine).aligned++;
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
    pairs: matched.pairs, sevPairs: matched.sevPairs, absorbed: matched.absorbed,
    lowConfidenceCount: matched.pairs.filter(p => p.lowConfidence).length,
    missing: matched.missing, added: matched.added,
    fieldChanges: fieldChanges, wordingChanges: wordingChanges,
    migrations: migrations, migrationOf: migrationOf,
    stats: stats, enginesNoResult: enginesNoResult,
    strict: !!opts.strict, aligned: aligned, blockingCount: blockingCount
  };
}

/**
 * 执行完备性硬校验：--new 为聚合 JSON 且 summary.failed_check 非空时由 cmdCompare 调用。
 * 存在未成功执行的检查项 → 本次结果不完整（漏检数与问题总数不代表全量结论），
 * 强制判不对齐并把失败项计入阻断数；失败明细渲染进对齐报告（见 renderReport）。
 */
function applyCompletenessGate(res, failedCheck) {
  const failed = normalizeFailedCheck(failedCheck);
  if (!failed.length) return res;
  res.completenessIssues = failed;
  res.blockingCount += failed.length;
  res.aligned = false;
  return res;
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
    (res.completenessIssues ? '，含执行完备性 ' + res.completenessIssues.length + ' 项' : '') +
    (res.strict ? '，含文案差异（--strict）' : '') + '）');
  L.push('');

  // 执行完备性（硬校验：存在未成功执行的检查项）
  if (res.completenessIssues && res.completenessIssues.length) {
    L.push('## ❌ 执行完备性（存在未成功执行的检查项，本次结果不完整）');
    L.push('');
    L.push('本次基准重跑有 ' + res.completenessIssues.length + ' 个检查项未成功执行，其检出结果缺失，' +
      '漏检数与问题总数不代表全量结论；该情况计为阻断（exit 1），须修复检查项执行后重跑基准，再进行对齐判定。');
    L.push('');
    res.completenessIssues.forEach((f, i) => {
      L.push((i + 1) + '. ' + displayEngine(f.check) + '：' + (f.reason || '原因未记录'));
    });
    L.push('');
  }

  // 按引擎统计（命中率 = 对齐/基线，即基线条目被本次新结果命中的比例）
  L.push('## 按引擎统计');
  L.push('');
  L.push('| 检查引擎 | 基线 | 新结果 | 对齐 | 漏检 | 多检 | 吸收 | 字段变化 | 命中率 |');
  L.push('| --- | --- | --- | --- | --- | --- | --- | --- | --- |');
  const engines = Array.from(res.stats.keys()).sort();
  const hitRate = (aligned, golden) => golden > 0 ? (aligned / golden * 100).toFixed(1) + '%' : '-';
  const tot = { golden: 0, fresh: 0, aligned: 0, missing: 0, added: 0, absorbed: 0, changed: 0 };
  for (const e of engines) {
    const s = res.stats.get(e);
    L.push('| ' + disp(e) + (s.outOfScope ? '（基线外）' : '') + ' | ' + (s.outOfScope && s.golden === 0 ? '-' : s.golden) +
      ' | ' + (s.outOfScope && s.fresh === 0 ? '-' : s.fresh) + ' | ' + (s.outOfScope ? '-' : s.aligned) +
      ' | ' + (s.outOfScope ? '-' : s.missing) + ' | ' + (s.outOfScope ? '-' : s.added) +
      ' | ' + (s.outOfScope ? '-' : s.absorbed) +
      ' | ' + (s.outOfScope ? '-' : s.changed) +
      ' | ' + (s.outOfScope ? '-' : hitRate(s.aligned, s.golden)) + ' |');
    if (!s.outOfScope) {
      tot.golden += s.golden; tot.fresh += s.fresh; tot.aligned += s.aligned;
      tot.missing += s.missing; tot.added += s.added; tot.absorbed += s.absorbed; tot.changed += s.changed;
    }
  }
  L.push('| **合计（比对范围内）** | ' + tot.golden + ' | ' + tot.fresh + ' | ' + tot.aligned +
    ' | ' + tot.missing + ' | ' + tot.added +
    ' | ' + tot.absorbed + ' | ' + tot.changed +
    ' | ' + hitRate(tot.aligned, tot.golden) + ' |');
  L.push('');
  L.push('- 命中率 = 对齐 / 基线 × 100%：基线条目被本次新结果命中（配对成功，含字段变化对与低置信度兜底对）的比例；' +
    '基线为 0 或基线外引擎不参与，显示 `-`；重复检出吸收并入已配对条目，不影响对齐计数');
  L.push('');

  // 匹配评分统计（6 维度加权，命中线 ≥6；吸收/低置信度兜底 ≥5.5）
  L.push('## 📊 匹配评分统计');
  L.push('');
  const bucket = v => (v >= 7 ? 'hi' : 'lo');
  const allPairs = res.pairs.concat(res.sevPairs || []);
  const pairHi = allPairs.filter(p => bucket(p.score) === 'hi').length;
  const pairLo = allPairs.length - pairHi;
  const absHi = (res.absorbed || []).filter(a => bucket(a.score) === 'hi').length;
  const absLo = (res.absorbed || []).length - absHi;
  L.push('| 分数区间 | 主匹配（含严重程度诊断） | 重复检出吸收 |');
  L.push('| --- | --- | --- |');
  L.push('| ≥7.0（高置信度） | ' + pairHi + ' | ' + absHi + ' |');
  L.push('| 5.5~6.9（中低置信度） | ' + pairLo + ' | ' + absLo + ' |');
  L.push('');
  const lowN = res.lowConfidenceCount || 0;
  L.push('- 匹配算法：6 维度加权评分（类型 1 + 严重程度 1 + 行号 0~2 + 上下文 0~2 + 描述 0~2 + 建议 0~1，满分 9），' +
    '命中线 ≥6，类型/严重程度为硬门；引擎为前置门槛；中文匹配采用 2 字 n-gram（bigram）');
  if (lowN) L.push('- ⚠️ 低置信度兜底配对 ' + lowN + ' 条（总分 5.5~5.9 或行号漂移），请人工复核');
  if ((res.absorbed || []).length) {
    L.push('- 重复检出吸收 ' + res.absorbed.length + ' 条：同一缺陷的重复报告并入已配对基线，不阻断、不计多检（同一缺陷一票制）');
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

  // 重复检出吸收（不阻断）
  if (res.absorbed && res.absorbed.length) {
    L.push('## 🔁 重复检出吸收（同一缺陷的重复报告，不阻断、不计多检）');
    L.push('');
    res.absorbed.forEach((a, i) => {
      const n = res.nIn[a.n];
      const g = res.gIn[a.g];
      L.push((i + 1) + '. ' + recordBrief(n) + ' → 并入基线 [' + g.label + ']（评分 ' + a.score.toFixed(1) + '）');
      L.push('   - 上下文：' + clip(n.context, 60));
      L.push('   - 描述：' + clip(n.description, 80));
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
  L.push('- 对齐口径：阻断=漏检/多检/严重程度/问题类型/行号/文件变化；上下文/描述/建议措辞差异默认仅提示（LLM 表述波动）；' +
    '同一缺陷的重复检出吸收不阻断');
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
  console.error('  node bench.js perf convert --input=<result.json> --output=<runs/perf_bench_test.md> [--label=<标签>]');
  console.error('  node bench.js perf compare --golden=<perf_bench.md> --new=<runs/perf_bench_test.md|result.json> [--report=<runs/perf_report.md>] [--label=<标签>]');
  console.error('  node bench.js protect --old=<common_bench.md> --new=<runs/common_bench_test.md|result.json> [--reasons=<删除理由.json>] [--report=<runs/protect_report.md>] [--label=<标签>]');
  console.error('  node bench.js format-check --input=<bench.md> [--strict-ctx]');
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

function readJson(f) {
  const text = readFileText(f);
  try {
    return JSON.parse(text);
  } catch (e) {
    console.error('JSON 解析失败: ' + f + '（' + e.message + '）');
    process.exit(2);
  }
}

// 新结果输入：.json（聚合 {summary, issues} 或原子结果数组）或 .md（bench 同构）
// 返回 { records, failedCheck }；failedCheck 仅在 --new 为聚合 JSON 且含 summary.failed_check 时非空
// （传 convert 产物 md 时无完备性信息，执行完备性硬校验不生效）
function loadFresh(f) {
  const text = readFileText(f);
  if (path.extname(f).toLowerCase() === '.json') {
    const data = readJson(f);
    const issues = Array.isArray(data) ? data : (data && Array.isArray(data.issues) ? data.issues : null);
    if (!issues) {
      console.error('输入既非 JSON 数组，也非含 issues 数组的聚合结果: ' + f);
      process.exit(2);
    }
    const failedCheck = (data && !Array.isArray(data) && data.summary && Array.isArray(data.summary.failed_check))
      ? data.summary.failed_check : [];
    return { records: issues.map((iss, i) => recordFromIssue(iss, i + 1)), failedCheck: failedCheck };
  }
  const parsed = parseBenchMd(text);
  if (parsed.skippedBlocks.length) {
    console.error('警告: ' + f + ' 中 ' + parsed.skippedBlocks.length + ' 个问题块缺少字段，已跳过');
  }
  return { records: parsed.records, failedCheck: [] };
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
  const res = compareRecords(goldenParsed.records, fresh.records, { strict: !!args.strict, engines: engines });
  if (fresh.failedCheck.length) applyCompletenessGate(res, fresh.failedCheck);
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

// ---------- 耗时维度（perf，提示级） ----------

// check_timing_ms（engine → 毫秒）→ 归一化 Map：
//   引擎名归一（canonEngine）+ 多目标计时键折叠（filename-check-1/2 → filename-check，求和）
function extractTiming(checkTimingMs) {
  const map = new Map();
  if (!checkTimingMs || typeof checkTimingMs !== 'object') return map;
  for (const rawKey of Object.keys(checkTimingMs)) {
    const ms = Number(checkTimingMs[rawKey]);
    if (!Number.isFinite(ms) || ms < 0) continue;
    let key = normText(rawKey);
    const m = /^(.*)-(\d+)$/.exec(key);
    if (m) key = m[1];
    key = canonEngine(key);
    map.set(key, (map.get(key) || 0) + ms);
  }
  return map;
}

// summary.failed_check → [{check, reason}]（check 归一为 canonical）
function normalizeFailedCheck(arr) {
  if (!Array.isArray(arr)) return [];
  const out = [];
  for (const f of arr) {
    if (!f || typeof f !== 'object') continue;
    const check = canonEngine(f.check);
    if (check) out.push({ check: check, reason: normText(f.reason) });
  }
  return out;
}

// 耗时快照 md（头部元信息 + "## 逐项耗时" 列表；数值单位毫秒，行格式兼容无单位的历史快照）
const PERF_TIMING_HEAD = /^##\s*逐项耗时(?:\s*[（(][^）)]*[）)])?\s*$/;
const PERF_TIMING_LINE = /^-\s*(.+?)\s*[：:]\s*(\d+(?:\.\d+)?)\s*(?:ms|毫秒)?\s*(?:[（(][^）)]*[）)])?\s*$/;

// 毫秒值人类可读化：显式 ms 单位；≥1s 追加秒/分钟注释（仅展示用，注释不参与解析）
function fmtMsHuman(v) {
  const ms = Math.round(v);
  let ann = '';
  if (ms >= 60000) {
    ann = '（约 ' + (ms / 60000).toFixed(1) + ' 分钟）';
  } else if (ms >= 1000) {
    ann = '（约 ' + (ms / 1000).toFixed(1) + ' 秒）';
  }
  return ms + 'ms' + ann;
}

function renderPerfMd(timingMap, meta) {
  meta = meta || {};
  const L = [];
  L.push('# 耗时基线快照');
  L.push('');
  L.push('- 生成时间：' + (meta.stamp || new Date().toISOString()));
  L.push('- 运行标签：' + (meta.label || '-'));
  L.push('- 数据来源：' + (meta.source || '-'));
  const failed = meta.failed || [];
  L.push('- 执行完备性：' + (failed.length
    ? '失败 ' + failed.length + ' 项：' + failed.map(f => f.check).join('、')
    : '无失败检查项'));
  if (failed.length) {
    L.push('- 失败原因：' + failed.map(f => f.check + '：' + clip(f.reason || '原因未记录', 80)).join('；'));
  }
  L.push('- 环境注记：耗时基线仅在同环境可比（机器、CPU 负载、网络、缓存状态）；重建流程见 README"耗时基线建档"');
  L.push('- 计时单位：毫秒（ms），取自聚合报告 summary.check_timing_ms');
  L.push('');
  L.push('## 逐项耗时（毫秒）');
  L.push('');
  for (const k of Array.from(timingMap.keys()).sort()) {
    L.push('- ' + displayEngine(k) + '：' + fmtMsHuman(timingMap.get(k)));
  }
  L.push('');
  return L.join('\n');
}

function parsePerfMd(text) {
  const timing = new Map();
  const failedEngines = [];
  let inTiming = false;
  for (const line of String(text).split(/\r?\n/)) {
    if (PERF_TIMING_HEAD.test(line)) { inTiming = true; continue; }
    if (/^##\s/.test(line)) { inTiming = false; continue; }
    if (inTiming) {
      const m = PERF_TIMING_LINE.exec(line);
      if (m) timing.set(canonEngine(m[1]), Number(m[2]));
      continue;
    }
    const fm = /^-\s*执行完备性\s*[：:]\s*(.*)$/.exec(line);
    if (fm) {
      const t = normText(fm[1]);
      if (t && t.indexOf('无') !== 0) {
        const body = t.replace(/^失败\s*\d+\s*项[：:]\s*/, '');
        for (const part of body.split(/[、,，]/)) {
          const c = canonEngine(part.trim());
          if (c) failedEngines.push(c);
        }
      }
    }
  }
  return { timing: timing, failedEngines: failedEngines };
}

// 耗时输入：.json（聚合 {summary}）或 .md（perf 快照）
function loadPerf(f) {
  if (path.extname(f).toLowerCase() === '.json') {
    const data = readJson(f);
    const summary = (data && !Array.isArray(data) && data.summary) ? data.summary : {};
    return {
      timing: extractTiming(summary.check_timing_ms),
      failed: normalizeFailedCheck(summary.failed_check),
      source: f
    };
  }
  const p = parsePerfMd(readFileText(f));
  return { timing: p.timing, failed: p.failedEngines.map(e => ({ check: e, reason: '' })), source: f };
}

// 耗时对比核心：基线 Map × 本次 Map → 行结果 + 分类清单（判定口径见文件头"耗时判定"）
function comparePerf(goldenMap, newMap) {
  const rows = [];
  const flagged = [], improved = [], missing = [], added = [];
  const keys = Array.from(new Set(Array.from(goldenMap.keys()).concat(Array.from(newMap.keys())))).sort();
  for (const k of keys) {
    const g = goldenMap.has(k) ? goldenMap.get(k) : null;
    const n = newMap.has(k) ? newMap.get(k) : null;
    const isModel = MODEL_ENGINES.has(k);
    const thr = isModel ? PERF_THRESHOLD.model : PERF_THRESHOLD.script;
    const row = {
      engine: k, category: isModel ? '模型' : '脚本',
      golden: g, current: n, deltaMs: null, deltaPct: null, mark: '-'
    };
    if (g === null) {
      row.mark = 'ℹ️ 基线未收录';
      added.push(k);
    } else if (n === null) {
      row.mark = '⚠️ 本次无计时';
      missing.push(k);
    } else {
      row.deltaMs = n - g;
      row.deltaPct = g > 0 ? (row.deltaMs / g) * 100 : null;
      if (row.deltaMs > thr.absMs && row.deltaPct !== null && row.deltaPct > thr.relPct) {
        row.mark = '❌ 显著回归';
        flagged.push(k);
      } else if (row.deltaMs < -thr.absMs && row.deltaPct !== null && row.deltaPct < -thr.relPct) {
        row.mark = '✅ 显著改善';
        improved.push(k);
      }
    }
    rows.push(row);
  }
  return { rows: rows, flagged: flagged, improved: improved, missing: missing, added: added };
}

function fmtMs(v) {
  if (v === null || v === undefined) return '-';
  return String(Math.round(v));
}

function fmtDeltaMs(v) {
  if (v === null || v === undefined) return '-';
  const r = Math.round(v);
  return (r > 0 ? '+' : '') + r;
}

function fmtDeltaPct(v) {
  if (v === null || v === undefined) return '-';
  return (v > 0 ? '+' : '') + v.toFixed(1) + '%';
}

function renderPerfCompare(res, meta) {
  meta = meta || {};
  const L = [];
  L.push('# 耗时对比报告（提示级，不阻断基线对齐判定）');
  L.push('');
  L.push('- 耗时基线：' + meta.goldenPath + ' — ' + meta.goldenCount + ' 项');
  L.push('- 本次结果：' + meta.newPath + ' — ' + meta.newCount + ' 项' +
    (meta.label && meta.label !== meta.newPath ? '（' + meta.label + '）' : ''));
  L.push('- 生成时间：' + (meta.stamp || new Date().toISOString()));
  L.push('- 判定：显著回归 ' + res.flagged.length + ' 项、显著改善 ' + res.improved.length +
    ' 项、本次无计时 ' + res.missing.length + ' 项、基线未收录 ' + res.added.length +
    ' 项 —— 均为提示级，不改变基准回归（compare）的对齐结论');
  L.push('');
  L.push('## 逐项耗时对比');
  L.push('');
  L.push('| 检查项 | 类别 | 基线(ms) | 本次(ms) | Δ(ms) | Δ% | 标记 |');
  L.push('| --- | --- | --- | --- | --- | --- | --- |');
  for (const r of res.rows) {
    L.push('| ' + displayEngine(r.engine) + ' | ' + r.category + ' | ' + fmtMs(r.golden) +
      ' | ' + fmtMs(r.current) + ' | ' + fmtDeltaMs(r.deltaMs) + ' | ' + fmtDeltaPct(r.deltaPct) +
      ' | ' + r.mark + ' |');
  }
  L.push('');
  L.push('## 执行完备性（本次结果）');
  L.push('');
  const failed = meta.failed || [];
  if (failed.length) {
    L.push('本次有 ' + failed.length + ' 个检查项未成功执行（其计时与检出缺失，耗时对比中相应项不可信）：');
    L.push('');
    failed.forEach((f, i) => {
      L.push((i + 1) + '. ' + displayEngine(f.check) + '：' + (f.reason || '原因未随输入记录，见对应聚合报告'));
    });
  } else {
    L.push('无失败检查项。');
  }
  L.push('');
  L.push('## ℹ️ 说明');
  L.push('');
  L.push('- 提示级口径：耗时差异不产生阻断、不改变 compare 的对齐结论；仅当同环境连续多次出现"显著回归"且差异可复现时，才怀疑 skill 性能劣化');
  L.push('- 显著回归阈值（相对与绝对须同时超过）：脚本项 >+' + PERF_THRESHOLD.script.relPct + '% 且 >+' +
    (PERF_THRESHOLD.script.absMs / 1000) + 's；模型项 >+' + PERF_THRESHOLD.model.relPct + '% 且 >+' +
    (PERF_THRESHOLD.model.absMs / 1000) + 's（模型项为 subagent 墙钟，波动大故放宽）；对称标记"显著改善"');
  L.push('- 环境约束：耗时基线仅在同环境可比（机器、CPU 负载、网络、缓存状态）；换环境后先按 README"耗时基线建档"重建');
  L.push('- 预热：首跑含脚本缓存下载、cspell 词典安装、术语源拉取等一次性成本，预热一次后取第二跑对比');
  L.push('- "本次无计时"的项结合执行完备性确认是否检查失败/跳过；"基线未收录"的项如需纳入，走耗时基线重建');
  L.push('- 退出码：0=已生成报告（提示级）；2=输入错误');
  return L.join('\n');
}

function cmdPerfConvert(args) {
  if (!args.input || !args.output) usage();
  const data = readJson(args.input);
  const summary = (data && !Array.isArray(data) && data.summary) ? data.summary : {};
  const timing = extractTiming(summary.check_timing_ms);
  if (!timing.size) {
    console.error('输入缺少可用的计时数据（summary.check_timing_ms）: ' + args.input);
    process.exit(2);
  }
  const failed = normalizeFailedCheck(summary.failed_check);
  const md = renderPerfMd(timing, {
    stamp: new Date().toISOString(),
    label: args.label === true ? undefined : args.label,
    source: path.basename(args.input),
    failed: failed
  });
  fs.mkdirSync(path.dirname(path.resolve(args.output)), { recursive: true });
  fs.writeFileSync(args.output, md + '\n', 'utf8');
  console.log('耗时快照已写入: ' + args.output + '（' + timing.size + ' 项计时）');
  console.log('执行完备性: ' + (failed.length
    ? '失败 ' + failed.length + ' 项（' + failed.map(f => f.check).join('、') + '）'
    : '无失败检查项'));
}

function cmdPerfCompare(args) {
  if (!args.golden || !args.new) usage();
  if (!fs.existsSync(args.golden)) {
    console.error('耗时基线不存在: ' + args.golden);
    console.error('请先按 README"耗时基线建档"流程建立 perf_bench.md，再执行耗时对比');
    process.exit(2);
  }
  // 参考耗时护栏（原始定位不变：仅量级预判，禁止作为对比基线；改名占用基线名后，
  // 检测方式由文件名跟随为头部"参考性质"标志行，建档覆盖后该行随文件消失即恢复基线语义）
  const goldenHead = fs.readFileSync(args.golden, 'utf8');
  if (goldenHead.includes('- 参考性质：')) {
    console.error('拒绝以内置参考耗时 ' + args.golden + ' 作为对比基线：');
    console.error('该文件仅用于量级预判，耗时跨环境不可比；如需正式对比，请按 README"耗时基线建档"流程建立本环境基线');
    process.exit(2);
  }
  const g = loadPerf(args.golden);
  const n = loadPerf(args.new);
  if (!g.timing.size) {
    console.error('耗时基线无可解析的逐项耗时: ' + args.golden);
    process.exit(2);
  }
  const res = comparePerf(g.timing, n.timing);
  const report = renderPerfCompare(res, {
    goldenPath: args.golden, newPath: args.new,
    label: args.label === true ? undefined : args.label,
    goldenCount: g.timing.size, newCount: n.timing.size,
    failed: n.failed
  });
  console.log(report);
  if (args.report && args.report !== true) {
    fs.mkdirSync(path.dirname(path.resolve(args.report)), { recursive: true });
    fs.writeFileSync(args.report, report + '\n', 'utf8');
    console.log('耗时对比报告已写入: ' + args.report);
  }
  process.exit(0);
}

// ---------- 测试集保护（protect） ----------

// 删除理由文件（JSON 数组）加载与校验：每项必须含 engine/line/reason/evidence 四要素，
// 缺任一项视为无效理由（告警并忽略）——对应“明确问题定位 + 删除原因 + 必要证据”要求
function loadReasons(f) {
  const data = readJson(f);
  if (!Array.isArray(data)) {
    console.error('理由文件须为 JSON 数组: ' + f);
    process.exit(2);
  }
  const out = [];
  data.forEach((r, i) => {
    if (!r || typeof r !== 'object') {
      console.error('警告: 理由第 ' + (i + 1) + ' 项非对象，已忽略');
      return;
    }
    const engine = canonEngine(r.engine || '');
    const lineMissing = r.line === undefined || r.line === null || String(r.line).trim() === '';
    const reason = normText(r.reason);
    const evidence = normText(r.evidence);
    if (!engine || lineMissing || !reason || !evidence) {
      console.error('警告: 理由第 ' + (i + 1) + ' 项缺少 engine/line/reason/evidence 必填要素，已忽略');
      return;
    }
    out.push({ engine: engine, line: normLine(r.line), reason: reason, evidence: evidence });
  });
  return out;
}

// 删除记录 × 理由匹配：按（canonical 引擎 + 行号）精确定位；同键多条时逐条消耗理由
function matchReasons(deleted, reasons) {
  const used = new Array(reasons.length).fill(false);
  return deleted.map(d => {
    let hit = -1;
    for (let i = 0; i < reasons.length; i++) {
      if (used[i]) continue;
      if (reasons[i].engine === d.engine && reasons[i].line === d.line) { hit = i; break; }
    }
    if (hit >= 0) {
      used[hit] = true;
      return { del: d, reason: reasons[hit] };
    }
    return { del: d, reason: null };
  });
}

function renderProtectReport(res, assign, meta) {
  const L = [];
  const unjustified = assign.filter(a => !a.reason);
  const justified = assign.filter(a => a.reason);
  const changedCount = res.fieldChanges.length + res.wordingChanges.length;
  L.push('# 测试集保护校验报告');
  L.push('');
  L.push('- 前一版基线：' + meta.oldPath + ' — ' + res.goldenTotal + ' 条');
  L.push('- 更新候选：' + meta.newPath + ' — ' + res.freshTotal + ' 条' +
    (meta.label && meta.label !== meta.newPath ? '（' + meta.label + '）' : ''));
  let verdict;
  if (assign.length === 0) {
    verdict = '✅ 通过（无删除）';
  } else if (unjustified.length) {
    verdict = '❌ 保护违规（删除 ' + assign.length + ' 条，其中未证据化 ' + unjustified.length + ' 条）';
  } else {
    verdict = '✅ 通过（删除 ' + assign.length + ' 条，全部已证据化）';
  }
  L.push('- 判定：' + verdict);
  L.push('');
  if (assign.length) {
    L.push('## 删除（前版有、候选无）— 须逐条证据化');
    L.push('');
    assign.forEach((a, i) => {
      L.push((i + 1) + '. ' + recordBrief(a.del));
      L.push('   - 上下文：' + clip(a.del.context, 60));
      L.push('   - 描述：' + clip(a.del.description, 80));
      if (a.reason) {
        L.push('   - ✅ 已证据化 — 原因：' + a.reason.reason + '；证据：' + clip(a.reason.evidence, 100));
      } else {
        L.push('   - ❌ 未证据化（须在 --reasons 文件中按 检查引擎+位置行号 定位，并给出删除原因与必要证据）');
      }
    });
    L.push('');
  }
  L.push('## 修改与新增（允许，无须经由）');
  L.push('');
  L.push('- 修改（匹配对中字段/文案变化）：' + changedCount + ' 处；新增：' + res.added.length + ' 条');
  L.push('');
  L.push('## ℹ️ 口径');
  L.push('');
  L.push('- 保护规则：测试集更新只允许“修改”与“新增”；不允许删除前一版本基线的条目，除非逐条提供问题定位（检查引擎+位置行号）、删除原因与必要证据');
  L.push('- 证据要求：误报须给出源码/规则依据；重复须给出重复对象条目；数据集变更须给出变更 diff');
  L.push('- 未证据化删除存在时（exit 1）禁止以候选覆盖基线');
  L.push('- 退出码：0=通过；1=保护违规；2=输入错误');
  return L.join('\n');
}

function cmdProtect(args) {
  if (!args.old || !args.new) usage();
  const oldParsed = parseBenchMd(readFileText(args.old));
  if (!oldParsed.records.length) {
    console.error('前一版基线解析失败（未解析到任何问题块）: ' + args.old);
    process.exit(2);
  }
  const fresh = loadFresh(args.new);
  const res = compareRecords(oldParsed.records, fresh.records, {});
  const reasons = (args.reasons && args.reasons !== true) ? loadReasons(args.reasons) : [];
  const assign = matchReasons(res.missing, reasons);
  const unjustified = assign.filter(a => !a.reason);
  const report = renderProtectReport(res, assign, {
    oldPath: args.old, newPath: args.new,
    label: args.label === true ? undefined : args.label
  });
  console.log(report);
  if (args.report && args.report !== true) {
    fs.mkdirSync(path.dirname(path.resolve(args.report)), { recursive: true });
    fs.writeFileSync(args.report, report + '\n', 'utf8');
    console.log('保护校验报告已写入: ' + args.report);
  }
  if (unjustified.length) {
    console.error('保护违规：存在 ' + unjustified.length + ' 条未证据化删除，禁止以候选覆盖基线');
    process.exit(1);
  }
  process.exit(0);
}

// ---------- 问题集格式规范校验（format-check） ----------

const VALID_SEVERITIES = new Set(['严重', '一般', '提示']);
const VALID_TYPES = new Set(['规范性', '正确性', '可读性', '完备性']);
const CANONICAL_ENGINES = new Set(Object.keys(ENGINE_DISPLAY));
const GENERIC_FIELD_LINE = /^-\s*([^：:]+?)\s*[：:]\s*(.*)$/;

/**
 * 格式规范：问题编号从 1 起连续；每块 8 字段齐全、有序且非空；检查引擎在白名单
 * （canonical 或历史别名）；位置行号为非负整数；问题类型/严重程度在枚举内。
 * opts.requireCtxWrap 时另要求问题上下文用反引号正确包裹（byself 本地约定）。
 */
function checkBenchFormat(text, opts) {
  opts = opts || {};
  const issues = [];
  const required = FIELD_LABELS.map(p => p[1]);
  const requiredSet = new Set(required);
  const blocks = [];
  let cur = null;
  for (const line of String(text).split(/\r?\n/)) {
    const h = BLOCK_HEAD.exec(line);
    if (h) {
      if (cur) blocks.push(cur);
      cur = { num: parseInt(h[1], 10), fields: {}, order: [], unknown: [] };
      continue;
    }
    if (!cur) continue;
    const f = GENERIC_FIELD_LINE.exec(line);
    if (f) {
      if (requiredSet.has(f[1])) {
        cur.fields[f[1]] = f[2];
        cur.order.push(f[1]);
      } else {
        cur.unknown.push(f[1]);
      }
    }
  }
  if (cur) blocks.push(cur);

  blocks.forEach((b, idx) => {
    const id = '问题' + b.num;
    if (b.num !== idx + 1) {
      issues.push(id + '：编号应为 问题' + (idx + 1) + '（编号须从 1 起连续递增）');
    }
    for (const label of required) {
      if (!(label in b.fields)) {
        issues.push(id + '：缺少字段「' + label + '」');
      } else if (b.fields[label].trim() === '') {
        issues.push(id + '：字段「' + label + '」为空');
      }
    }
    for (const u of b.unknown) {
      issues.push(id + '：存在未知字段「' + u + '」');
    }
    const dupCounts = {};
    b.order.forEach(label => { dupCounts[label] = (dupCounts[label] || 0) + 1; });
    const dupLabels = Object.keys(dupCounts).filter(label => dupCounts[label] > 1);
    for (const label of dupLabels) {
      issues.push(id + '：字段「' + label + '」重复出现 ' + dupCounts[label] + ' 次');
    }
    if (!dupLabels.length && b.order.length === required.length &&
      b.order.join('\u0000') !== required.join('\u0000')) {
      issues.push(id + '：字段顺序与规范不一致（应为 ' + required.join('、') + '）');
    }
    if ('检查引擎' in b.fields && b.fields['检查引擎'].trim() !== '') {
      const eng = canonEngine(b.fields['检查引擎']);
      if (!CANONICAL_ENGINES.has(eng)) {
        issues.push(id + '：检查引擎「' + b.fields['检查引擎'] + '」不在合法白名单');
      }
    }
    if ('位置行号' in b.fields && b.fields['位置行号'].trim() !== '') {
      const lv = b.fields['位置行号'].trim();
      const rm = /^(\d+)(~(\d+))?$/.exec(lv);
      const rangeOk = rm && (rm[3] === undefined || parseInt(rm[1], 10) <= parseInt(rm[3], 10));
      if (!rangeOk) {
        issues.push(id + '：位置行号「' + lv + '」应为非负整数或合法行号区间（区间形如 287~289，起始 ≤ 结束）');
      }
    }
    if (b.fields['问题类型'] && b.fields['问题类型'].trim() !== '' && !VALID_TYPES.has(b.fields['问题类型'].trim())) {
      issues.push(id + '：问题类型「' + b.fields['问题类型'] + '」不在枚举（' + Array.from(VALID_TYPES).join('/') + '）');
    }
    if (b.fields['严重程度'] && b.fields['严重程度'].trim() !== '' && !VALID_SEVERITIES.has(b.fields['严重程度'].trim())) {
      issues.push(id + '：严重程度「' + b.fields['严重程度'] + '」不在枚举（' + Array.from(VALID_SEVERITIES).join('/') + '）');
    }
    if (opts.requireCtxWrap && '问题上下文' in b.fields && b.fields['问题上下文'].trim() !== '') {
      const v = b.fields['问题上下文'];
      const lead = (/^`+/.exec(v) || [''])[0].length;
      const tail = (/`+\s*$/.exec(v) || [''])[0].length;
      const inner = v.slice(lead, v.length - tail).replace(/^\s/, '').replace(/\s$/, '').match(/`+/g) || [];
      const maxInner = inner.reduce((m, s) => Math.max(m, s.length), 0);
      if (lead === 0 || tail !== lead || maxInner >= lead) {
        issues.push(id + '：问题上下文未正确用反引号包裹');
      }
    }
  });
  return issues;
}

function cmdFormatCheck(args) {
  if (!args.input) usage();
  const text = readFileText(args.input);
  const issues = checkBenchFormat(text, { requireCtxWrap: !!args['strict-ctx'] });
  const head = '问题集格式检查: ' + args.input;
  console.log(head);
  if (!issues.length) {
    const count = (String(text).match(/^##\s*问题\d+\s*$/gm) || []).length;
    console.log('✅ 通过（' + count + ' 个问题块，格式全部合规' + (args['strict-ctx'] ? '，含上下文反引号包裹' : '') + '）');
    process.exit(0);
  }
  console.log('❌ 发现 ' + issues.length + ' 处违规：');
  issues.forEach((s, i) => console.log((i + 1) + '. ' + s));
  process.exit(1);
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
    engine: 'punctuation-check', type: '规范性', file: 'bench测试数据集.md', line: 999,
    context: '自检构造样例上下文', description: '自检构造：多检验证', suggestion: '无',
    severity: '提示', rawIndex: fresh.length + 1
  }));
  fresh.push(normalizeRecord({
    engine: 'markdownlint', type: '规范性', file: 'bench测试数据集.md', line: 999,
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

  // T3f: 重复检出吸收——fresh 含某条目的重复报告（同引擎/行号/描述，仅措辞微调）时，
  //      应并入已配对基线（吸收 1 条），不计多检、不阻断对齐
  const absBase = cloneAll(golden);
  const absFresh = cloneAll(golden);
  const dupRec = JSON.parse(JSON.stringify(absFresh[0]));
  dupRec.rawIndex = absFresh.length + 1;
  dupRec.suggestion = dupRec.suggestion + '（重复报告）';
  absFresh.push(dupRec);
  const absRes = compareRecords(absBase, absFresh, {});
  const absReportText = renderReport(absRes, { goldenPath: goldenPath, newPath: '(吸收自检)' });
  check('T3f 重复检出吸收（不阻断、不计多检）',
    absRes.aligned && absRes.absorbed.length === 1 && absRes.added.length === 0 &&
    absRes.missing.length === 0 && absReportText.indexOf('重复检出吸收') >= 0,
    '吸收 ' + absRes.absorbed.length + '（期望 1）、多检 ' + absRes.added.length + '（期望 0）、对齐=' + absRes.aligned);

  // T3g: 低置信度兜底——行号/类型/严重程度一致但上下文无关、描述部分重合（总分 5.5）时，
  //      仍可配对（对齐），并标注低置信度供人工复核
  const lcBase = cloneAll(golden);
  const lcFresh = cloneAll(golden);
  // 目标记录按引擎动态选择而非按位置硬编码（基线按引擎重排后顺序会变化）：
  // filename-check 条目的描述与合成描述「文件名规范提示」有确定 bigram 重合，保证描述分有效；
  // 基线无该引擎时回退取首条，并以原描述前缀构造部分重合
  const lcTargetIdx = lcFresh.findIndex(r => r.engine.toLowerCase() === 'filename-check');
  const lcIdx = lcTargetIdx >= 0 ? lcTargetIdx : 0;
  const lcSrc = lcFresh[lcIdx];
  const lcDesc = lcTargetIdx >= 0
    ? '自检构造：文件名规范提示'
    : '自检构造：' + String(lcSrc.description || '').slice(0, 4);
  lcFresh[lcIdx] = normalizeRecord({
    engine: lcSrc.engine, type: lcSrc.type, file: lcSrc.file, line: lcSrc.line,
    context: '自检构造完全无关的上下文内容样例', description: lcDesc,
    suggestion: '', severity: lcSrc.severity
  });
  const lcRes = compareRecords(lcBase, lcFresh, {});
  check('T3g 低置信度兜底配对（≥5.5 且行号/描述有效）',
    lcRes.aligned && lcRes.lowConfidenceCount === 1 && lcRes.missing.length === 0 && lcRes.added.length === 0,
    '低置信度 ' + lcRes.lowConfidenceCount + '（期望 1）、对齐=' + lcRes.aligned);

  // T3h: 严重程度变化诊断阈值——仅严重程度不同、其余维度较弱（总分 5.5~6.4）时仍应配对为
  //      "严重程度"字段变化，不得重复扣 1 分退化为漏检+多检（阈值口径回归）
  //      样例评分：类型 1 + 行号 2 + 上下文 1.5（包含）+ 描述 1.5（bigram=3）+ 建议 0 = 6.0
  const sdBase = cloneAll(golden);
  const sdFresh = cloneAll(golden);
  sdBase.push(normalizeRecord({
    engine: 'punctuation-check', type: '规范性', file: 'bench测试数据集.md', line: 777,
    context: '甲乙丙丁', description: '一二三四五六七八', suggestion: '建议样例',
    severity: '提示', rawIndex: sdBase.length + 1
  }));
  sdFresh.push(normalizeRecord({
    engine: 'punctuation-check', type: '规范性', file: 'bench测试数据集.md', line: 777,
    context: '前缀甲乙丙丁后缀', description: '一二三四', suggestion: '',
    severity: '一般', rawIndex: sdFresh.length + 1
  }));
  const sdRes = compareRecords(sdBase, sdFresh, {});
  const sdPairCount = (sdRes.sevPairs || []).length;
  check('T3h 严重程度变化诊断阈值（总分 5.5~6.4 不重复扣分）',
    sdPairCount === 1 && sdRes.missing.length === 0 && sdRes.added.length === 0 &&
    sdRes.fieldChanges.length === 1 && sdRes.fieldChanges[0].field === '严重程度' &&
    sdPairCount === 1 && sdRes.sevPairs[0].score === 6 && !sdRes.aligned && sdRes.blockingCount === 1,
    '严重程度变化配对 ' + sdPairCount + '（期望 1，评分 ' +
    (sdPairCount ? sdRes.sevPairs[0].score : '-') + '；修复前该场景退化为漏检+多检）');

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

  // T5: 耗时提取（别名归一 + 多目标计时键折叠求和）
  const perfTiming = extractTiming({
    markdownlint: 12000,
    'link_validity-check': 5000,
    'filename-check-1': 800,
    'filename-check-2': 600,
    punctuation_check: 65000,
    AICheck: 120000
  });
  check('T5 耗时提取（别名归一 + 多目标键折叠求和）',
    perfTiming.get('markdownlint') === 12000 &&
    perfTiming.get('link-validity-check') === 5000 &&
    perfTiming.get('filename-check') === 1400 &&
    perfTiming.get('punctuation-check') === 65000 &&
    perfTiming.get('aicheck') === 120000,
    'filename-check=' + perfTiming.get('filename-check') + '，共 ' + perfTiming.size + ' 项');

  // T6: 耗时快照渲染/解析往返（含执行完备性行）
  const perfMd = renderPerfMd(perfTiming, {
    stamp: '2026-01-01T00:00:00.000Z', label: 'selftest', source: 'selftest.json',
    failed: [{ check: 'codespell-check', reason: '自检构造：词典安装失败' }]
  });
  const perfParsed = parsePerfMd(perfMd);
  const perfRoundTripOk = perfParsed.timing.size === perfTiming.size &&
    Array.from(perfParsed.timing.keys()).every(k => perfParsed.timing.get(k) === perfTiming.get(k)) &&
    perfParsed.failedEngines.length === 1 && perfParsed.failedEngines[0] === 'codespell-check';
  check('T6 耗时快照渲染/解析往返（含执行完备性）', perfRoundTripOk,
    '解析 ' + perfParsed.timing.size + '/' + perfTiming.size + ' 项，失败项 ' + perfParsed.failedEngines.join('、'));

  // T7: 耗时对比判定（脚本项紧阈值 / 模型项宽阈值 / 缺失 / 未收录 / 改善）
  const gPerf = new Map([['markdownlint', 10000], ['aicheck', 60000], ['codespell-check', 3000], ['tag-closed-check', 20000]]);
  const nPerf = new Map([['markdownlint', 26000], ['aicheck', 65000], ['punctuation-check', 70000], ['tag-closed-check', 5000]]);
  const perfRes = comparePerf(gPerf, nPerf);
  const perfReportText = renderPerfCompare(perfRes, {
    goldenPath: '(自检基线)', newPath: '(自检本次)', goldenCount: gPerf.size, newCount: nPerf.size,
    failed: [{ check: 'codespell-check', reason: '自检构造：词典安装失败' }]
  });
  check('T7 耗时对比判定（紧/宽阈值、缺失、未收录、改善）',
    perfRes.flagged.length === 1 && perfRes.flagged[0] === 'markdownlint' &&
    perfRes.improved.length === 1 && perfRes.improved[0] === 'tag-closed-check' &&
    perfRes.missing.length === 1 && perfRes.missing[0] === 'codespell-check' &&
    perfRes.added.length === 1 && perfRes.added[0] === 'punctuation-check' &&
    perfRes.flagged.indexOf('aicheck') < 0 &&
    perfReportText.indexOf('显著回归') >= 0 && perfReportText.indexOf('本次无计时') >= 0 &&
    perfReportText.indexOf('基线未收录') >= 0 && perfReportText.indexOf('词典安装失败') >= 0,
    '回归 ' + perfRes.flagged.join('、') + '；改善 ' + perfRes.improved.join('、') +
    '；无计时 ' + perfRes.missing.join('、') + '；未收录 ' + perfRes.added.join('、'));

  // T8: 执行完备性硬校验（failed_check 非空 → 强制不对齐并渲染失败项）
  const compRes = compareRecords(golden, cloneAll(golden), {});
  applyCompletenessGate(compRes, [{ check: 'codespell-check', reason: '自检构造：词典安装失败' }]);
  const compReportText = renderReport(compRes, { goldenPath: goldenPath, newPath: '(自检)' });
  check('T8 执行完备性硬校验（强制不对齐 + 失败项渲染）',
    !compRes.aligned && compRes.blockingCount === 1 &&
    compReportText.indexOf('执行完备性') >= 0 && compReportText.indexOf('词典安装失败') >= 0,
    'aligned=' + compRes.aligned + '，阻断 ' + compRes.blockingCount + ' 处');

  // T9: 测试集保护判定（删除须逐条证据化；修改/新增放行）
  const oldP = [
    normalizeRecord({ engine: 'aicheck', type: '正确性', file: 'a.md', line: 10, context: 'c1', description: 'd1', suggestion: 's1', severity: '严重' }),
    normalizeRecord({ engine: 'punctuation-check', type: '规范性', file: 'a.md', line: 20, context: 'c2', description: 'd2', suggestion: 's2', severity: '提示' }),
    normalizeRecord({ engine: 'compliance-check', type: '规范性', file: 'a.md', line: 30, context: 'c3', description: 'd3', suggestion: 's3', severity: '严重' })
  ];
  const newP = [
    normalizeRecord({ engine: 'aicheck', type: '正确性', file: 'a.md', line: 10, context: 'c1', description: 'd1（修改）', suggestion: 's1', severity: '严重' }),
    normalizeRecord({ engine: 'punctuation-check', type: '规范性', file: 'a.md', line: 20, context: 'c2', description: 'd2', suggestion: 's2', severity: '提示' }),
    normalizeRecord({ engine: 'aicheck', type: '正确性', file: 'a.md', line: 99, context: 'c9', description: 'd9', suggestion: 's9', severity: '提示' })
  ];
  const protRes = compareRecords(oldP, newP, {});
  const protNoReason = matchReasons(protRes.missing, []);
  const protWithReason = matchReasons(protRes.missing, [
    { engine: 'compliance-check', line: 30, reason: '确认误报', evidence: 'tag_checker.js 无此规则' },
    { engine: 'compliance-check', line: 31, reason: '多余', evidence: '多余' }
  ]);
  const protReportText = renderProtectReport(protRes, protNoReason, { oldPath: '(前版)', newPath: '(候选)' });
  check('T9 测试集保护判定（删除须证据化，修改/新增放行）',
    protRes.missing.length === 1 && protRes.added.length === 1 &&
    protRes.wordingChanges.length === 1 &&
    protNoReason.every(a => !a.reason) &&
    protWithReason.every(a => a.reason) &&
    protReportText.indexOf('保护违规') >= 0 && protReportText.indexOf('未证据化') >= 0,
    '删除 ' + protRes.missing.length + '、新增 ' + protRes.added.length + '、修改 ' + protRes.wordingChanges.length +
    '；无理由=' + protNoReason.filter(a => !a.reason).length + '/1、带理由=' + protWithReason.filter(a => a.reason).length + '/1');

  // T10: 格式规范校验（合规样例通过，含行号区间；违规样例逐项检出）
  const fmtRangeBlock = ['## 问题4', '',
    '- 检查引擎：compliance-check', '- 问题类型：规范性', '- 文件路径：a.md',
    '- 位置行号：287~289', '- 问题上下文：`ctx`', '- 问题描述：d', '- 修改建议：s', '- 严重程度：严重', ''].join('\n');
  const fmtGood = renderBenchMd(cloneAll(golden).slice(0, 3), { note: 'selftest format' }) + '\n' + fmtRangeBlock;
  const fmtGoodIssues = checkBenchFormat(fmtGood, {});
  const fmtBadLines = ['## 问题2', '',
    '- 检查引擎：punctuation-check', '- 问题类型：正确性', '- 文件路径：a.md',
    '- 位置行号：x3', '- 问题上下文：`ctx`', '- 问题描述：d', '- 修改建议：s', '- 严重程度：超高',
    '- 备注：未知字段', '- 修改建议：重复字段', '',
    '## 问题4', '',
    '- 检查引擎：不存在的引擎', '- 问题类型：正确性', '- 文件路径：a.md',
    '- 位置行号：9~3', '- 问题上下文：ctx', '- 问题描述：', '- 修改建议：s', '- 严重程度：提示', ''];
  const fmtBadIssues = checkBenchFormat(fmtBadLines.join('\n'), {});
  check('T10 格式规范校验（合规/违规双路径，含行号区间与重复字段）',
    fmtGoodIssues.length === 0 &&
    fmtBadIssues.length === 9 &&
    fmtBadIssues.some(s => s.indexOf('编号应为') >= 0) &&
    fmtBadIssues.some(s => s.indexOf('未知字段') >= 0) &&
    fmtBadIssues.some(s => s.indexOf('重复出现') >= 0) &&
    fmtBadIssues.some(s => s.indexOf('位置行号') >= 0 && s.indexOf('x3') >= 0) &&
    fmtBadIssues.some(s => s.indexOf('位置行号') >= 0 && s.indexOf('行号区间') >= 0) &&
    fmtBadIssues.some(s => s.indexOf('严重程度') >= 0 && s.indexOf('不在枚举') >= 0) &&
    fmtBadIssues.some(s => s.indexOf('检查引擎') >= 0 && s.indexOf('白名单') >= 0) &&
    fmtBadIssues.some(s => s.indexOf('问题描述') >= 0 && s.indexOf('为空') >= 0),
    '合规 4/4 通过（含 287~289 区间）；违规检出 ' + fmtBadIssues.length + ' 处（期望 9，含重复字段）');

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
  if (cmd === 'perf') {
    const sub = process.argv[3];
    const perfRest = process.argv.slice(4);
    if (sub === 'convert') return cmdPerfConvert(parseKV(perfRest));
    if (sub === 'compare') return cmdPerfCompare(parseKV(perfRest));
    usage();
  }
  const rest = process.argv.slice(3);
  if (cmd === 'convert') return cmdConvert(parseKV(rest));
  if (cmd === 'compare') return cmdCompare(parseKV(rest));
  if (cmd === 'protect') return cmdProtect(parseKV(rest));
  if (cmd === 'format-check') return cmdFormatCheck(parseKV(rest));
  if (cmd === 'selftest') return cmdSelftest(parseKV(rest));
  usage();
}

module.exports = {
  normText, canonEngine, displayEngine, normLine,
  normalizeRecord, recordFromIssue, parseBenchMd, renderBenchMd,
  sim, recordSim, matchRecords, compareRecords, renderReport,
  applyCompletenessGate, extractTiming, normalizeFailedCheck,
  renderPerfMd, parsePerfMd, comparePerf, renderPerfCompare,
  loadReasons, matchReasons, renderProtectReport, checkBenchFormat,
  ENGINE_ALIAS, ENGINE_DISPLAY, FIELD_LABELS, MODEL_ENGINES, PERF_THRESHOLD
};

if (require.main === module) main();
