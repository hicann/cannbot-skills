// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// proxy 捕获分类的单一权威源。所有"这个 jsonl 是不是 proxy 捕获 / 属于哪个
// agent 框架"的判定都必须走这里，不得在各调用点自行实现。
//
// 产物代际（演进实测，信号只增不减 —— 新代际向后兼容旧代际）：
//   Gen1  仅顶层 system/tools wire 字段（无任何标记）
//   Gen2  + 行级 source:"claude-proxy"/"opencode-proxy"
//   Gen3  + cpx- 文件名前缀 / 按真实 sid 分流（文件名不参与判定，改名不失效）
//   Gen4  + x_cannbay 声明式（cc-wire-round 等）+ <sid>.meta.json（cc-session-meta）
//   Gen5  + codex（Responses API）：source:"codex-proxy"、meta.framework:"codex"
//
// 信号优先级（cannbay-schema-spec §6 双轨：声明式权威 > legacy 行级）：
//   meta.json cc-session-meta（文件级） > 行级 source 标记 > wire 指纹兜底
import fs from 'node:fs';
import { parseJsonlLines, extractSystemText, type ContentBlock } from './adapters/claude-jsonl';

export type ProxyAgentFramework = 'opencode' | 'claude-code' | 'codex' | null;

export interface ProxyClassification {
  /** 是否 proxy（cpx）捕获。insight 导出件不算（它复用部分 wire 字段但非代理捕获）。 */
  isProxy: boolean;
  /** proxy 来源标记原值：claude-proxy / opencode-proxy / codex-proxy（早期无标记件由指纹推断合成）。 */
  marker: string | null;
  /** agent 归属（opencode / claude-code / codex）；native 文件为 null。 */
  framework: ProxyAgentFramework;
  /** 命中的判定信号。 */
  via: 'meta' | 'source-line' | 'wire-fingerprint' | null;
}

/** meta.json cc-session-meta 声明的 framework → 归一化 agent 框架。 */
const META_FRAMEWORK_MAP: Record<string, Exclude<ProxyAgentFramework, null>> = {
  'opencode': 'opencode',
  'claude-code': 'claude-code',
  'claude': 'claude-code',
  'codex': 'codex',
};

/** 行级 source 标记 → agent 框架。未列举的 *-proxy 一律归 claude-code（默认族）。 */
const MARKER_FRAMEWORK_MAP: Record<string, Exclude<ProxyAgentFramework, null>> = {
  'opencode-proxy': 'opencode',
  'codex-proxy': 'codex',
};

/** meta.json cc-session-meta 声明的内容（producer=cpx 时 framework/ccVersion 有效）。
 *  全部 meta.json 消费方（分类器 / 导入 ccVersion / cannbay2 重导入 patch）共用，
 *  不再各自内联 JSON 解析。 */
export interface CcSessionMeta {
  producer: string | null;
  framework: string | null;
  ccVersion: string | null;
}

export function readCcSessionMeta(jsonlPath: string): CcSessionMeta | null {
  const metaPath = jsonlPath.replace(/\.jsonl$/, '.meta.json');
  try {
    if (!fs.existsSync(metaPath)) return null;
    const raw = JSON.parse(fs.readFileSync(metaPath, 'utf8')) as {
      x_cannbay?: { schema?: string; version?: number; data?: { producer?: string; framework?: string; ccVersion?: string } };
    };
    const xb = raw.x_cannbay;
    if (!xb || xb.schema !== 'cc-session-meta' || xb.version !== 1) return null;
    return {
      producer: xb.data?.producer ?? null,
      framework: xb.data?.framework ?? null,
      ccVersion: xb.data?.ccVersion ?? null,
    };
  } catch {
    return null;
  }
}

function readMetaFramework(jsonlPath: string): { framework: ProxyAgentFramework; marker: string | null } | null {
  const meta = readCcSessionMeta(jsonlPath);
  if (!meta || (meta.producer && meta.producer !== 'cpx')) return null;
  const fw = meta.framework ? META_FRAMEWORK_MAP[meta.framework] : undefined;
  if (!fw) return null;
  return { framework: fw, marker: `${meta.framework}-proxy`.replace('claude-code-proxy', 'claude-proxy') };
}

export function classifyProxyCapture(jsonlPath: string): ProxyClassification {
  // 1. 文件级权威源：cc-session-meta（producer=cpx）
  const meta = readMetaFramework(jsonlPath);
  if (meta) {
    return { isProxy: true, marker: meta.marker, framework: meta.framework, via: 'meta' };
  }

  const lines = parseJsonlLines(jsonlPath);
  let lineMarker: string | null = null;
  let hasWireFingerprint = false;
  let hasXCannbay = false;
  let systemText: string | null = null;

  for (const l of lines) {
    const s = (l as { source?: unknown }).source;
    if (lineMarker == null && typeof s === 'string' && s.endsWith('-proxy')) lineMarker = s;
    const xb = (l as { x_cannbay?: { data?: { system?: unknown } } }).x_cannbay;
    if (xb != null) hasXCannbay = true;
    const sys = xb?.data?.system ?? (l as { system?: unknown }).system;
    if (sys != null && (typeof sys === 'string' ? sys.length > 0 : Array.isArray(sys))) {
      hasWireFingerprint = true;
      if (systemText == null) systemText = extractSystemText(sys as string | ContentBlock[]);
    }
    if ((l as { tools?: unknown }).tools != null) hasWireFingerprint = true;
  }

  // 2. 行级 source 标记 —— 需 wire 证据交叉校验：真实 cpx 捕获（任意代际）
  // 至少带 system/tools wire 字段或 x_cannbay 声明。历史上 DB 导出件曾错误
  // 标 source:'claude-proxy'（a25ebdfd 修复前上传的污染件仍留在 cannbay 仓），
  // 只有裸标记无任何 wire 数据 → 判为导出件，不算 proxy 捕获。
  if (lineMarker && (hasWireFingerprint || hasXCannbay)) {
    return {
      isProxy: true,
      marker: lineMarker,
      framework: MARKER_FRAMEWORK_MAP[lineMarker] ?? 'claude-code',
      via: 'source-line',
    };
  }
  if (lineMarker && !hasWireFingerprint && !hasXCannbay) {
    // 污染导出件降级为 insight-export 语义（不标 proxy 徽标，不触发覆盖度）
    return { isProxy: false, marker: null, framework: null, via: null };
  }

  // 3. wire 指纹兜底（Gen1 早期无标记件；native claude jsonl 结构性无 system/tools 顶层字段）
  if (hasWireFingerprint) {
    const framework: Exclude<ProxyAgentFramework, null> =
      systemText && /You are opencode/i.test(systemText.slice(0, 400)) ? 'opencode' : 'claude-code';
    return { isProxy: true, marker: `${framework === 'opencode' ? 'opencode' : 'claude'}-proxy`, framework, via: 'wire-fingerprint' };
  }

  return { isProxy: false, marker: null, framework: null, via: null };
}
