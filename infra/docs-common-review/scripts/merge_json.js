#!/usr/bin/env node
// SPDX-License-Identifier: CANN-2.0
/**
 * JSON 数组结果合并脚本（docs-common-review 家族）
 *
 * 将多个 JSON 数组结果文件按输入顺序拼接为一个数组，写入单一输出文件。
 * 主要用途：filename-check 多目标各自独立运行（脚本每次会删除重建结果文件，
 * 必须分目录）后，合并为单一结果文件供 aggregate.js 聚合。
 *
 * 用法：
 *   node merge_json.js --inputs=<f1,f2,...> --output=<file>
 *
 * 语义：
 *   - 逐个读取输入文件，JSON 数组则按序拼接；不存在 / 解析失败 / 非数组则跳过并告警
 *   - 无任何可用输入时退出码 2（不写输出文件）
 *
 * 退出码：0 成功；1 用法错误；2 无任何可用输入。
 */

'use strict';

const fs = require('fs');
const path = require('path');

function usage() {
  console.error('用法: node merge_json.js --inputs=<f1,f2,...> --output=<file>');
  process.exit(1);
}

function parseArgs(argv) {
  const args = {};
  for (const a of argv) {
    const m = /^--([a-z-]+)=(.*)$/.exec(a);
    if (!m) { console.error('未知参数: ' + a); usage(); }
    if (m[1] === 'inputs') {
      args.inputs = m[2].split(',').map(s => s.trim()).filter(Boolean);
    } else if (m[1] === 'output') {
      args.output = m[2];
    } else {
      console.error('未知参数: ' + a);
      usage();
    }
  }
  if (!args.inputs || args.inputs.length === 0) usage();
  if (!args.output) usage();
  return args;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const merged = [];
  let used = 0;
  for (const f of args.inputs) {
    let data;
    try {
      data = JSON.parse(fs.readFileSync(f, 'utf8'));
    } catch (e) {
      console.error('跳过（不存在或解析失败）: ' + f);
      continue;
    }
    if (!Array.isArray(data)) {
      console.error('跳过（非 JSON 数组）: ' + f);
      continue;
    }
    for (const item of data) merged.push(item);
    used++;
  }
  if (used === 0) {
    console.error('合并失败：没有任何可用的输入结果文件。');
    process.exit(2);
  }
  fs.mkdirSync(path.dirname(args.output), { recursive: true });
  fs.writeFileSync(args.output, JSON.stringify(merged, null, 2) + '\n', 'utf8');
  console.log('合并完成: ' + args.output + '（共 ' + merged.length + ' 条，来自 ' + used + ' 个输入）');
}

main();
