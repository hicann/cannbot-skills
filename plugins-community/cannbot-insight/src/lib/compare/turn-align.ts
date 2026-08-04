// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

export interface TurnData {
  turnId: string
  turnIndex: number
  role: string
  content: string | null
  contentSummary: string | null
  totalTokens: number
  inputTokens: number
  outputTokens: number
  reasoningTokens: number
  latencyMs: number
  model: string | null
  toolCalls: Array<{ toolCallId: string; toolName: string; state: string; durationMs: number }>
  skillEvents: Array<{ skillName: string; eventType: string; success: boolean }>
}

export type AlignmentType = "match" | "aOnly" | "bOnly"

export interface AlignedPair {
  indexA: number | null
  indexB: number | null
  a: TurnData | null
  b: TurnData | null
  similarity: number
  type: AlignmentType
  isManual?: boolean
}

// Light pair: alignment metadata only, no TurnData. The worker returns these
// to avoid structured-cloning 8000 × full TurnData objects on the way out
// (Phase 8). The main thread rebuilds full pairs from turnsA/B (already held
// in main-thread memory) using indexA/indexB — no re-transfer needed.
export interface LightAlignedPair {
  indexA: number | null
  indexB: number | null
  similarity: number
  type: AlignmentType
  isManual?: boolean
}

export interface ManualAlignment {
  indexA: number
  indexB: number
}

const SIMILARITY_THRESHOLD = 0.35
const GAP_PENALTY = 0.35
const ROLE_MISMATCH_PENALTY = 0.4

const CJK_STOP = new Set([
  "请", "问", "那", "好", "的", "了", "是", "在", "有", "和", "与", "或",
  "不", "没", "这", "下", "上", "中", "个", "到", "会", "能", "要", "就",
  "都", "也", "还", "又", "很", "更", "最", "把", "被", "让", "给", "对",
  "从", "为", "以", "于", "但", "而", "所", "之", "其", "如", "何", "我",
  "来", "去", "做", "看", "说", "用", "帮", "修", "查", "找", "前", "后",
  "里", "外", "们", "吧", "啊", "呢", "嗯", "哦", "哈", "呀", "啦", "嘛",
  "过", "起", "些", "多", "少", "大", "小", "先", "后", "已", "可", "并",
  "及", "等", "因", "此", "所", "每", "该", "本", "种", "样", "两", "无",
])

const EN_STOP = new Set([
  "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
  "her", "was", "one", "our", "out", "has", "his", "how", "its", "let",
  "may", "new", "now", "old", "see", "way", "who", "did", "get", "got",
  "him", "hit", "low", "man", "run", "say", "she", "too", "use", "that",
  "this", "with", "have", "will", "what", "when", "then", "than", "they",
  "from", "been", "call", "come", "each", "make", "like", "long", "look",
  "more", "much", "must", "name", "need", "next", "only", "over", "part",
  "same", "some", "tell", "them", "very", "want", "well", "went", "were",
  "into", "just", "also", "back", "could", "still", "should", "would",
  "about", "after", "again", "being", "first", "great", "right", "since",
  "under", "while", "where", "which", "there", "these", "those", "thing",
  "think", "doing", "going", "using", "other", "because", "already",
  "let", "ll", "ve", "re", "won", "ain",
])

function isCJK(ch: string): boolean {
  const cp = ch.charCodeAt(0)
  return (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf) || (cp >= 0xf900 && cp <= 0xfaff)
}

function extractCJKChars(text: string): string[] {
  const chars: string[] = []
  for (const ch of text) {
    if (isCJK(ch) && !CJK_STOP.has(ch)) chars.push(ch)
  }
  return chars
}

function extractWordFreq(text: string, includeThinking: boolean): Map<string, number> {
  const freq = new Map<string, number>()

  let body = text.toLowerCase()
  if (!includeThinking) {
    body = body.replace(/<thinking>[\s\S]*?<\/thinking>/g, "")
  }
  body = body.replace(/<[^>]+>/g, " ").replace(/[^\w\u4e00-\u9fff]/g, " ")

  for (const seg of body.split(/\s+/)) {
    if (!seg) continue

    const cjkChars = extractCJKChars(seg)
    for (const ch of cjkChars) {
      freq.set(ch, (freq.get(ch) ?? 0) + 1)
    }

    const enPart = seg.replace(/[\u4e00-\u9fff\u3400-\u4dbf]/g, "").toLowerCase()
    if (enPart.length > 2 && !EN_STOP.has(enPart)) {
      freq.set(enPart, (freq.get(enPart) ?? 0) + 1)
    }
  }

  return freq
}

function extractThinkingFreq(text: string): Map<string, number> {
  const match = text.match(/<thinking>([\s\S]*?)<\/thinking>/)
  if (!match) return new Map()
  return extractWordFreq(match[1], true)
}

function cosineSimilarity(freqA: Map<string, number>, normA: number, freqB: Map<string, number>, normB: number): number {
  if (normA === 0 || normB === 0) return 0
  // Iterate the smaller map for fewer hash lookups in the larger one.
  const small = freqA.size <= freqB.size ? freqA : freqB
  const large = small === freqA ? freqB : freqA
  let dot = 0
  for (const [word, count] of small) {
    const other = large.get(word)
    if (other !== undefined) dot += count * other
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB))
}

function jaccard(setA: Set<string>, setB: Set<string>): number {
  if (setA.size === 0 && setB.size === 0) return 0
  if (setA.size === 0 || setB.size === 0) return 0
  const intersection = new Set([...setA].filter(x => setB.has(x)))
  const union = new Set([...setA, ...setB])
  return intersection.size / union.size
}

interface PreparedTurn {
  turn: TurnData
  bodyFreq: Map<string, number>
  thinkFreq: Map<string, number>
  // Pre-computed sum of squared counts; cosineSimilarity uses these instead
  // of recomputing norm across the full map each call (huge on large sessions).
  bodyNorm: number
  thinkNorm: number
  toolNames: Set<string>
  skillNames: Set<string>
  isSkeleton: boolean
}

function sumSquaredCounts(freq: Map<string, number>): number {
  let n = 0
  for (const [, c] of freq) n += c * c
  return n
}

function prepareTurn(turn: TurnData): PreparedTurn {
  const content = turn.content ?? turn.contentSummary ?? ""
  const bodyFreq = extractWordFreq(content, false)
  const thinkFreq = extractThinkingFreq(content)
  return {
    turn,
    bodyFreq,
    thinkFreq,
    bodyNorm: sumSquaredCounts(bodyFreq),
    thinkNorm: sumSquaredCounts(thinkFreq),
    toolNames: new Set(turn.toolCalls.map(tc => tc.toolName)),
    skillNames: new Set(turn.skillEvents.map(se => se.skillName)),
    isSkeleton: bodyFreq.size === 0 && thinkFreq.size === 0,
  }
}

function computeSimilarity(pa: PreparedTurn, pb: PreparedTurn, posA: number, posB: number, totalA: number, totalB: number): number {
  const a = pa.turn
  const b = pb.turn
  const bothSkeleton = pa.isSkeleton && pb.isSkeleton

  if (bothSkeleton) {
    const roleBonus = a.role === b.role ? 0.05 : -0.15
    const toolJ = jaccard(pa.toolNames, pb.toolNames)
    const skillJ = jaccard(pa.skillNames, pb.skillNames)
    const relPosA = totalA > 1 ? posA / (totalA - 1) : 0
    const relPosB = totalB > 1 ? posB / (totalB - 1) : 0
    const posBonus = 1 - Math.abs(relPosA - relPosB)
    const score = roleBonus + toolJ * 0.10 + skillJ * 0.05 + posBonus * 0.05
    return Math.max(0, Math.min(1, score))
  }

  let score = 0

  if (a.role === b.role) {
    score += 0.15
  } else {
    score -= ROLE_MISMATCH_PENALTY
  }

  const hasContent = pa.bodyFreq.size > 0 || pb.bodyFreq.size > 0
  const hasThinking = pa.thinkFreq.size > 0 || pb.thinkFreq.size > 0

  const bodyCos = cosineSimilarity(pa.bodyFreq, pa.bodyNorm, pb.bodyFreq, pb.bodyNorm)
  const thinkCos = cosineSimilarity(pa.thinkFreq, pa.thinkNorm, pb.thinkFreq, pb.thinkNorm)

  const minBodyWords = Math.min(pa.bodyFreq.size, pb.bodyFreq.size)
  const richnessFactor = minBodyWords >= 6 ? 1.0
    : minBodyWords >= 3 ? 0.75
    : minBodyWords >= 2 ? 0.5
    : 0.25

  if (hasContent) {
    score += bodyCos * 0.40 * richnessFactor
  }
  if (hasThinking) {
    score += thinkCos * 0.10
  }

  const toolJ = jaccard(pa.toolNames, pb.toolNames)
  const oneSkeleton = pa.isSkeleton || pb.isSkeleton
  score += toolJ * (oneSkeleton ? 0.05 : (hasContent ? 0.15 : 0.20))

  score += jaccard(pa.skillNames, pb.skillNames) * 0.10

  const relPosA = totalA > 1 ? posA / (totalA - 1) : 0
  const relPosB = totalB > 1 ? posB / (totalB - 1) : 0
  const posDist = Math.abs(relPosA - relPosB)
  const posBonus = 1 - posDist
  score += posBonus * 0.10

  return Math.max(0, Math.min(1, score))
}

export function alignTurnsWithManual(
  turnsA: TurnData[],
  turnsB: TurnData[],
  manualAlignments: ManualAlignment[],
  onProgress?: (progress: number) => void,
): AlignedPair[] {
  if (manualAlignments.length === 0) return alignTurns(turnsA, turnsB, onProgress)

  const preparedA = turnsA.map(prepareTurn)
  const preparedB = turnsB.map(prepareTurn)

  const anchors = [...manualAlignments].sort((a, b) => a.indexA - b.indexA || a.indexB - b.indexB)

  const usedA = new Set(anchors.map(a => a.indexA))
  const usedB = new Set(anchors.map(a => a.indexB))

  // Each segment tracks the source indices (mapA/mapB) so we can slice the
  // matching preparedA/preparedB entries — without re-running prepareTurn
  // for the same turns on every segment (saves 30%+ when there are anchors).
  const segments: Array<{
    startA: number; endA: number; startB: number; endB: number;
    subA: TurnData[]; subB: TurnData[];
    subPreparedA: PreparedTurn[]; subPreparedB: PreparedTurn[];
    mapA: number[]; mapB: number[];
  }> = []

  let prevAnchorA = -1
  let prevAnchorB = -1

  for (const anchor of anchors) {
    const rangeStartA = prevAnchorA + 1
    const rangeStartB = prevAnchorB + 1
    const rangeEndA = anchor.indexA - 1
    const rangeEndB = anchor.indexB - 1

    const subA: TurnData[] = []
    const subB: TurnData[] = []
    const subPreparedA: PreparedTurn[] = []
    const subPreparedB: PreparedTurn[] = []
    const mapA: number[] = []
    const mapB: number[] = []

    for (let i = rangeStartA; i <= rangeEndA; i++) {
      if (!usedA.has(i)) { subA.push(turnsA[i]); subPreparedA.push(preparedA[i]); mapA.push(i) }
    }
    for (let j = rangeStartB; j <= rangeEndB; j++) {
      if (!usedB.has(j)) { subB.push(turnsB[j]); subPreparedB.push(preparedB[j]); mapB.push(j) }
    }

    if (subA.length > 0 || subB.length > 0) {
      segments.push({ startA: rangeStartA, endA: rangeEndA, startB: rangeStartB, endB: rangeEndB, subA, subB, subPreparedA, subPreparedB, mapA, mapB })
    }

    prevAnchorA = anchor.indexA
    prevAnchorB = anchor.indexB
  }

  const tailStartA = prevAnchorA + 1
  const tailStartB = prevAnchorB + 1
  const subA: TurnData[] = []
  const subB: TurnData[] = []
  const subPreparedA: PreparedTurn[] = []
  const subPreparedB: PreparedTurn[] = []
  const mapA: number[] = []
  const mapB: number[] = []

  for (let i = tailStartA; i < turnsA.length; i++) {
    if (!usedA.has(i)) { subA.push(turnsA[i]); subPreparedA.push(preparedA[i]); mapA.push(i) }
  }
  for (let j = tailStartB; j < turnsB.length; j++) {
    if (!usedB.has(j)) { subB.push(turnsB[j]); subPreparedB.push(preparedB[j]); mapB.push(j) }
  }

  if (subA.length > 0 || subB.length > 0) {
    segments.push({ startA: tailStartA, endA: turnsA.length - 1, startB: tailStartB, endB: turnsB.length - 1, subA, subB, subPreparedA, subPreparedB, mapA, mapB })
  }

  const result: AlignedPair[] = []

  // Per-segment progress scaling: each segment's internal [0,1] progress is
  // mapped onto a slice of the global [0,1] range proportional to how many
  // A-side turns that segment covers. Without this, the manual-alignment
  // path silently keeps progress at 0% until everything finishes.
  const totalA = turnsA.length
  let processedA = 0

  let anchorIdx = 0
  for (const seg of segments) {
    const segLenA = seg.subA.length
    const segBase = totalA > 0 ? processedA / totalA : 0
    const segRatio = totalA > 0 ? segLenA / totalA : 0
    const segOnProgress = onProgress
      ? (p: number) => onProgress(segBase + p * segRatio)
      : undefined

    const segPairs = alignTurnsInternal(seg.subA, seg.subB, seg.subPreparedA, seg.subPreparedB, segOnProgress)
    for (const p of segPairs) {
      result.push({
        ...p,
        indexA: p.indexA !== null ? seg.mapA[p.indexA] : null,
        indexB: p.indexB !== null ? seg.mapB[p.indexB] : null,
        a: p.a,
        b: p.b,
        isManual: false,
      })
    }

    processedA += segLenA

    while (anchorIdx < anchors.length && anchors[anchorIdx].indexA <= seg.endA) {
      const anchor = anchors[anchorIdx]
      const sim = computeSimilarity(
        preparedA[anchor.indexA], preparedB[anchor.indexB],
        anchor.indexA, anchor.indexB, turnsA.length, turnsB.length,
      )
      result.push({
        indexA: anchor.indexA,
        indexB: anchor.indexB,
        a: turnsA[anchor.indexA],
        b: turnsB[anchor.indexB],
        similarity: sim,
        type: "match",
        isManual: true,
      })
      anchorIdx++
    }
  }

  while (anchorIdx < anchors.length) {
    const anchor = anchors[anchorIdx]
    const sim = computeSimilarity(
      preparedA[anchor.indexA], preparedB[anchor.indexB],
      anchor.indexA, anchor.indexB, turnsA.length, turnsB.length,
    )
    result.push({
      indexA: anchor.indexA,
      indexB: anchor.indexB,
      a: turnsA[anchor.indexA],
      b: turnsB[anchor.indexB],
      similarity: sim,
      type: "match",
      isManual: true,
    })
    anchorIdx++
  }

  const resultUsedA = new Set(result.filter(p => p.indexA !== null).map(p => p.indexA!))
  const resultUsedB = new Set(result.filter(p => p.indexB !== null).map(p => p.indexB!))

  for (let i = 0; i < turnsA.length; i++) {
    if (!usedA.has(i) && !resultUsedA.has(i)) {
      result.push({ indexA: i, indexB: null, a: turnsA[i], b: null, similarity: 0, type: "aOnly" })
    }
  }
  for (let j = 0; j < turnsB.length; j++) {
    if (!usedB.has(j) && !resultUsedB.has(j)) {
      result.push({ indexA: null, indexB: j, a: null, b: turnsB[j], similarity: 0, type: "bOnly" })
    }
  }

  result.sort((a, b) => {
    const aKey = a.indexA ?? a.indexB ?? 0
    const bKey = b.indexA ?? b.indexB ?? 0
    return aKey - bKey
  })

  // Manual path: segments covered [0, processedA/totalA); the trailing anchors
  // don't carry progress, so mark 100% once everything's stitched together.
  if (onProgress) onProgress(1)

  return result
}

// Default half-width of the DP band. Two sessions being compared usually have
// matching user↔assistant structure with bounded drift, so a band of 200 lets
// the alignment shift by up to ~10% across a 2000-turn session while skipping
// 80%+ of the O(N*M) matrix on large sessions.
const DEFAULT_BAND_WIDTH = 200
const NEG_INF = Number.NEGATIVE_INFINITY
// trace codes (Uint8Array cells): 0=unset/edge, 1=diag, 2=up, 3=left.
const TRACE_DIAG = 1
const TRACE_UP = 2
const TRACE_LEFT = 3

// Internal: takes pre-computed PreparedTurn arrays so callers can reuse the
// cache across segment calls (see alignTurnsWithManual).
function alignTurnsInternal(
  turnsA: TurnData[],
  turnsB: TurnData[],
  preparedA: PreparedTurn[],
  preparedB: PreparedTurn[],
  onProgress?: (progress: number) => void,
): AlignedPair[] {
  const lenA = turnsA.length
  const lenB = turnsB.length

  if (lenA === 0) return turnsB.map((b, i) => ({ indexA: null, indexB: i, a: null, b, similarity: 0, type: "bOnly" as AlignmentType }))
  if (lenB === 0) return turnsA.map((a, i) => ({ indexA: i, indexB: null, a, b: null, similarity: 0, type: "aOnly" as AlignmentType }))

  // Adaptive band: small sessions run the full O(N*M) matrix; large sessions
  // get banded DP. The band widens to cover the length difference plus a
  // 50-turn buffer, so sessions with very different turn counts (e.g. a
  // 4000-turn session vs a 4300-turn session) still take the banded path
  // instead of falling back to the full matrix.
  //
  //   4000x4300: |diff|=300 → band=350 (was 200 → full matrix → 8s)
  //              banded cells = 4000 * 700 = 2.8M → ~1.3s (6× speedup)
  //   4000x4000: |diff|=0   → band=200, banded cells 1.6M → ~760ms
  //   500x100:   |diff|=400  → band=450, but min(900,100)=100 so cells=50K
  //              (degenerates to full matrix — no penalty)
  //   50x50:     lenA+lenB=100 < 400 → full matrix (no point banding)
  const diff = Math.abs(lenA - lenB)
  const adaptiveBand = Math.max(DEFAULT_BAND_WIDTH, diff + 50)
  const useBanded = (lenA + lenB) >= 2 * DEFAULT_BAND_WIDTH
  const band = useBanded ? adaptiveBand : Math.max(lenA, lenB)

  // trace is the only full-matrix structure we keep — needed for backtrack.
  // Uint8Array (1 byte/cell) instead of string[][] cuts memory ~16× on 4k×4k.
  const trace: Uint8Array[] = new Array(lenA + 1)
  for (let i = 0; i <= lenA; i++) trace[i] = new Uint8Array(lenB + 1)

  // dp: rolling 2-row Float64Array. Memory stays O(lenB), not O(lenA*lenB).
  let dpPrev = new Float64Array(lenB + 1)  // dp[i-1]
  let dpCur = new Float64Array(lenB + 1)   // dp[i]

  // Edge initialization. dp[0][0] = 0 (default). dp[i][0] / dp[0][j] chain
  // along the axes with gap penalty, but only in-band cells are reachable.
  for (let i = 1; i <= Math.min(lenA, band); i++) trace[i][0] = TRACE_UP
  for (let j = 1; j <= Math.min(lenB, band); j++) {
    dpPrev[j] = -j * GAP_PENALTY
    trace[0][j] = TRACE_LEFT
  }

  for (let i = 1; i <= lenA; i++) {
    dpCur.fill(0)
    const jStart = Math.max(1, i - band)
    const jEnd = Math.min(lenB, i + band)

    // Edge column j=0: only in-band rows chain the gap.
    if (i <= band) {
      dpCur[0] = dpPrev[0] - GAP_PENALTY
      // trace[i][0] already set to TRACE_UP above.
    }

    const pa = preparedA[i - 1]

    for (let j = jStart; j <= jEnd; j++) {
      // Band-reachability for each of the three predecessors. Inline rather
      // than via a helper to avoid 3× function-call overhead per cell (this
      // is the innermost loop on 4k×4k sessions).
      const d = j - i
      const diagOK = d >= -band && d <= band            // |i-1 - (j-1)| = |i-j|
      const upOK = d >= -band - 1 && d <= band - 1     // |i-1 - j|     = |i-j-1|
      const leftOK = d >= -band + 1 && d <= band + 1   // |i - (j-1)|   = |i-j+1|

      // sim computed on demand — not stored, since backtracking only revisits
      // O(N+M) cells and the full simMatrix would dominate memory on 4k×4k.
      const sim = computeSimilarity(pa, preparedB[j - 1], i - 1, j - 1, lenA, lenB)

      const diagPrev = diagOK ? dpPrev[j - 1] : NEG_INF
      const upPrev = upOK ? dpPrev[j] : NEG_INF
      const leftPrev = leftOK ? dpCur[j - 1] : NEG_INF

      const matchScore = sim >= SIMILARITY_THRESHOLD
        ? diagPrev + sim
        : diagPrev - GAP_PENALTY * 2 - 0.05
      const gapUp = upPrev - GAP_PENALTY
      const gapLeft = leftPrev - GAP_PENALTY

      if (matchScore >= gapUp && matchScore >= gapLeft) {
        dpCur[j] = matchScore
        trace[i][j] = TRACE_DIAG
      } else if (gapUp >= gapLeft) {
        dpCur[j] = gapUp
        trace[i][j] = TRACE_UP
      } else {
        dpCur[j] = gapLeft
        trace[i][j] = TRACE_LEFT
      }
    }

    // Swap rows for next i. dpCur's old buffer becomes dpPrev (reused to avoid GC).
    const tmp = dpPrev
    dpPrev = dpCur
    dpCur = tmp

    if (onProgress && (i % 50 === 0 || i === lenA)) onProgress(i / lenA)
  }
  if (onProgress) onProgress(1)

  const pairs: AlignedPair[] = []
  let i = lenA
  let j = lenB

  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && trace[i][j] === TRACE_DIAG) {
      const sim = computeSimilarity(preparedA[i - 1], preparedB[j - 1], i - 1, j - 1, lenA, lenB)
      pairs.push({
        indexA: i - 1,
        indexB: j - 1,
        a: turnsA[i - 1],
        b: turnsB[j - 1],
        similarity: sim,
        type: "match",
      })
      i--
      j--
    } else if (i > 0 && (j === 0 || trace[i][j] === TRACE_UP)) {
      pairs.push({
        indexA: i - 1,
        indexB: null,
        a: turnsA[i - 1],
        b: null,
        similarity: 0,
        type: "aOnly",
      })
      i--
    } else {
      pairs.push({
        indexA: null,
        indexB: j - 1,
        a: null,
        b: turnsB[j - 1],
        similarity: 0,
        type: "bOnly",
      })
      j--
    }
  }

  pairs.reverse()
  return pairs
}

export function alignTurns(
  turnsA: TurnData[],
  turnsB: TurnData[],
  onProgress?: (progress: number) => void,
): AlignedPair[] {
  const preparedA = turnsA.map(prepareTurn)
  const preparedB = turnsB.map(prepareTurn)
  return alignTurnsInternal(turnsA, turnsB, preparedA, preparedB, onProgress)
}

// Estimate wall-clock time for the align-and-render pipeline in milliseconds.
// The UI shows this while the worker runs so users see a realistic ETA.
//
// Covers three phases that the previous formula missed:
//   B. Worker structured-clone of turnsA+turnsB on postMessage (in)
//   C. alignTurns CPU (banded or full-matrix DP)
//   D. Worker structured-clone of the resulting pairs array (out)
//   E. React render of the pairs list + computeContentDiff per pair
//
// Phase A (turns API fetch) is NOT included — compare/page.tsx shows that
// under a separate "正在获取 turn 数据..." loading state.
//
// Calibrated against Phase 2/2.5/6 bench data + user-reported wall time
// (~10s observed for 4000x4000 after all phases landed):
//   - prepareTurn: ~3 µs/turn
//   - DP cell:     ~0.46 µs  (similarity + DP transition)
//   - transfer:    ~0.3 ms/turn each way  (structured clone + worker startup)
//   - render:      base ~1.2s (React effect chain + initial layout + scroll
//                  listener setup) + ~0.7 ms per visible pair
//
// Phase 6 virtualization fundamentally changed render scaling: previously
// render linear in total pair count (12000 * 0.7 = 8.4s on 4000x4000),
// now only ~30 PairCards are mounted at once (viewport + buffer), so render
// is roughly constant ~1.2s regardless of total pair count.
//
// Validation against the user's 4000x4000 scenario (banded active, band=200):
//   transfer = 8000 * 0.3            = 2.4s
//   align    = 24ms + 1.6M * 0.46µs  = 0.76s
//   render   = 1200 + min(12000, 30) * 0.7 = 1.22s
//   subtotal (B+C+D+E)               = 4.4s
//   + fetch A (~5-6s on user machine, shown separately) ≈ 10s — matches.
//
// Pre-Phase-2 the formula was O(N*M)*0.0026 + (N+M)*4.4, which gave 77s for
// 4000x4000 — wildly inaccurate now that the actual cost is sub-second in
// the banded path.
export interface AlignTimeBreakdown {
  transferMs: number
  alignMs: number
  renderMs: number
  totalMs: number
}

export function estimateAlignTimeBreakdown(lenA: number, lenB: number): AlignTimeBreakdown {
  if (lenA === 0 || lenB === 0) {
    return { transferMs: 0, alignMs: 0, renderMs: 0, totalMs: 0 }
  }
  // Phase B+D: worker structured clone (in: turnsA+B, out: pairs).
  const transferMs = (lenA + lenB) * 0.3

  // Phase C: alignTurns CPU (banded or full-matrix DP).
  // The band is adaptive: max(DEFAULT_BAND_WIDTH, |lenA-lenB| + 50), so
  // length-mismatched sessions (e.g. 4000x4300) still take the banded path
  // and don't fall back to the full N*M matrix.
  const preparePerTurnMs = 0.003
  const perCellMs = 0.00046
  const diff = Math.abs(lenA - lenB)
  const adaptiveBand = Math.max(DEFAULT_BAND_WIDTH, diff + 50)
  const useBanded = (lenA + lenB) >= 2 * DEFAULT_BAND_WIDTH
  const dpCells = useBanded
    ? lenA * Math.min(2 * adaptiveBand, lenB)
    : lenA * lenB
  const alignMs = (lenA + lenB) * preparePerTurnMs + dpCells * perCellMs

  // Phase E: React render + computeContentDiff.
  // Phase 6 virtualization means only ~30 PairCards are mounted at any time
  // (viewport ~2-4 + buffer 20-30), so render cost is dominated by the base
  // (React effect chain + initial layout + scroll listener setup) plus a
  // small per-visible-pair component. It no longer scales with total pair
  // count — the 12000 unmounted pairs are pure placeholder divs.
  const VIRTUAL_WINDOW = 30
  const pairsEst = lenA + lenB + Math.min(lenA, lenB)
  const renderMs = 1200 + Math.min(pairsEst, VIRTUAL_WINDOW) * 0.7

  return {
    transferMs,
    alignMs,
    renderMs,
    totalMs: transferMs + alignMs + renderMs,
  }
}

export function estimateAlignTimeMs(lenA: number, lenB: number): number {
  return estimateAlignTimeBreakdown(lenA, lenB).totalMs
}

export function computeAlignStats(pairs: AlignedPair[]): {
  matched: number
  aOnly: number
  bOnly: number
  avgSimilarity: number
  highSimilarity: number
  mediumSimilarity: number
  lowSimilarity: number
} {
  const matched = pairs.filter(p => p.type === "match")
  const aOnly = pairs.filter(p => p.type === "aOnly").length
  const bOnly = pairs.filter(p => p.type === "bOnly").length
  const similarities = matched.map(p => p.similarity)
  const avgSim = similarities.length > 0 ? similarities.reduce((a, b) => a + b, 0) / similarities.length : 0

  return {
    matched: matched.length,
    aOnly,
    bOnly,
    avgSimilarity: avgSim,
    highSimilarity: similarities.filter(s => s >= 0.7).length,
    mediumSimilarity: similarities.filter(s => s >= 0.4 && s < 0.7).length,
    lowSimilarity: similarities.filter(s => s < 0.4).length,
  }
}

export function computeContentDiff(contentA: string | null, contentB: string | null): DiffRange[] {
  if (!contentA && !contentB) return []
  if (!contentA) return [{ type: "added", text: contentB ?? "" }]
  if (!contentB) return [{ type: "removed", text: contentA ?? "" }]
  if (contentA === contentB) return [{ type: "equal", text: contentA }]

  const linesA = contentA.split("\n")
  const linesB = contentB.split("\n")
  const ranges: DiffRange[] = []

  const lcs = computeLCS(linesA, linesB)
  let posA = 0
  let posB = 0

  for (const { aIdx, bIdx } of lcs) {
    const removedLines = linesA.slice(posA, aIdx)
    const addedLines = linesB.slice(posB, bIdx)

    if (removedLines.length > 0) {
      ranges.push({ type: "removed", text: removedLines.join("\n") })
    }
    if (addedLines.length > 0) {
      ranges.push({ type: "added", text: addedLines.join("\n") })
    }

    const common = linesA[aIdx]
    ranges.push({ type: "equal", text: common })
    posA = aIdx + 1
    posB = bIdx + 1
  }

  if (posA < linesA.length) {
    ranges.push({ type: "removed", text: linesA.slice(posA).join("\n") })
  }
  if (posB < linesB.length) {
    ranges.push({ type: "added", text: linesB.slice(posB).join("\n") })
  }

  return ranges
}

export type DiffType = "equal" | "added" | "removed" | "modified"

export interface DiffRange {
  type: DiffType
  text: string
}

function computeLCS(a: string[], b: string[]): Array<{ aIdx: number; bIdx: number }> {
  const m = a.length
  const n = b.length

  if (m > 200 || n > 200) {
    return fastAlign(a, b)
  }

  const dp: number[][] = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0))
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i - 1].trim() === b[j - 1].trim()) {
        dp[i][j] = dp[i - 1][j - 1] + 1
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1])
      }
    }
  }

  const result: Array<{ aIdx: number; bIdx: number }> = []
  let i = m
  let j = n
  while (i > 0 && j > 0) {
    if (a[i - 1].trim() === b[j - 1].trim()) {
      result.push({ aIdx: i - 1, bIdx: j - 1 })
      i--
      j--
    } else if (dp[i - 1][j] >= dp[i][j - 1]) {
      i--
    } else {
      j--
    }
  }
  result.reverse()
  return result
}

function fastAlign(a: string[], b: string[]): Array<{ aIdx: number; bIdx: number }> {
  const result: Array<{ aIdx: number; bIdx: number }> = []
  const setB = new Set(b.map(l => l.trim()))
  let bOffset = 0

  for (let i = 0; i < a.length; i++) {
    const line = a[i].trim()
    const j = b.findIndex((l, k) => k >= bOffset && l.trim() === line)
    if (j >= 0 && setB.has(line)) {
      result.push({ aIdx: i, bIdx: j })
      bOffset = j + 1
    }
  }

  return result
}

export function similarityLabel(sim: number): { label: string; color: string } {
  if (sim >= 0.4) return { label: "相似", color: "text-red-600 dark:text-red-400" }
  return { label: "不相似", color: "text-muted-foreground" }
}
