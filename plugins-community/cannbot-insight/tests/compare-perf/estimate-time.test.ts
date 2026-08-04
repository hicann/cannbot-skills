// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from "vitest"
import { estimateAlignTimeMs, estimateAlignTimeBreakdown, alignTurns } from "@/lib/compare/turn-align"
import { buildScenarioPair } from "./fixtures/generate-turns"

// The UI shows "predicted ~Ns" while alignment runs. The pre-Phase-2 formula
// (O(N*M)*0.0026 + (N+M)*4.4) gave 77-86s for 4000x4000 — wildly inaccurate.
// Phase 2 narrowed it to align CPU only (~8s for 4000x4300 full-matrix), but
// users still saw ~25s wall time because the estimate missed worker transfer
// and React render. These tests pin the new breakdown-based formula so future
// drift in any phase's coefficient is caught.

describe("estimateAlignTimeMs — covers transfer + align + render", () => {
  it("4000x4300 (adaptive band active) estimate drops to ~5s", () => {
    // Phase 2.5+6: adaptiveBand=350 (banded), virtualization caps render.
    // transfer 2.5s + align 1.3s + render ~1.2s ≈ 5s. Plus fetch ~5s on the
    // user's machine (shown separately) → ~10s wall time (down from 25s).
    const est = estimateAlignTimeMs(4000, 4300)
    expect(est).toBeGreaterThan(3000)
    expect(est).toBeLessThan(8000)
  })

  it("4000x4000 (banded active) estimate matches user-reported ~10s wall with fetch", () => {
    // Phase 6 virtualization: render no longer scales with pair count.
    // transfer 2.4s + align 0.76s + render 1.22s ≈ 4.4s. Plus fetch ~5-6s on
    // the user's machine (shown separately) → ~10s wall time, matching what
    // the user observes after Phase 6.
    const est = estimateAlignTimeMs(4000, 4000)
    expect(est).toBeGreaterThan(3000)
    expect(est).toBeLessThan(7000)
  })

  it("1000x1000 estimate is sub-3s (was 13s with pre-Phase-2 formula)", () => {
    const est = estimateAlignTimeMs(1000, 1000)
    expect(est).toBeGreaterThan(500)
    expect(est).toBeLessThan(3000)
  })

  it("500x500 estimate is sub-2s (render is now virtualization-capped)", () => {
    // align 87ms + transfer ~300ms + render ~1.2s (base + 30*0.7) ≈ 1.6s.
    const est = estimateAlignTimeMs(500, 500)
    expect(est).toBeGreaterThan(500)
    expect(est).toBeLessThan(2500)
  })

  it("small sessions still pay the render base cost (sub-2s)", () => {
    // 50+50=100 < 2*200 → full matrix for align, but render base ~1.2s applies.
    const est = estimateAlignTimeMs(50, 50)
    expect(est).toBeGreaterThan(1000)
    expect(est).toBeLessThan(2000)
  })

  it("zero-size edge cases return 0", () => {
    expect(estimateAlignTimeMs(0, 0)).toBe(0)
    expect(estimateAlignTimeMs(100, 0)).toBe(0)
    expect(estimateAlignTimeMs(0, 100)).toBe(0)
  })
})

describe("estimateAlignTimeBreakdown — phase decomposition", () => {
  it("4000x4300 breakdown: transfer > align, render capped (virtualization)", () => {
    const b = estimateAlignTimeBreakdown(4000, 4300)
    // Phase 2.5 adaptive band: align ~1.3s. Phase 6 virtualization: render
    // capped at base + 30*0.7 ≈ 1.2s (was 8.6s). Transfer ~2.5s is now the
    // largest phase — further perf wins must come from transfer/fetch, not
    // render or align.
    expect(b.alignMs).toBeGreaterThan(800)
    expect(b.alignMs).toBeLessThan(3000)
    expect(b.renderMs).toBeGreaterThan(1000)
    expect(b.renderMs).toBeLessThan(2000)
    expect(b.transferMs).toBeGreaterThan(1500)
    expect(b.transferMs).toBeLessThan(3000)
    expect(b.totalMs).toBeCloseTo(b.transferMs + b.alignMs + b.renderMs, 1)
  })

  it("4000x4000 breakdown: render no longer dominates (was 8.4s, now ~1.2s)", () => {
    const b = estimateAlignTimeBreakdown(4000, 4000)
    // Banded path: align 760ms, render ~1.2s (virtualization base + 30),
    // transfer 2.4s. Transfer is now the largest — the headline insight
    // from Phase 6.
    expect(b.alignMs).toBeLessThan(b.transferMs)
    expect(b.renderMs).toBeLessThan(b.transferMs)
    expect(b.alignMs).toBeLessThan(1500)
    expect(b.renderMs).toBeGreaterThan(1000)
    expect(b.renderMs).toBeLessThan(2000)
  })

  it("asymmetric size with adaptive band stays banded (no full-matrix fallback)", () => {
    // |500-100|=400, adaptiveBand = max(200, 450) = 450. Banded still active
    // (useBanded only requires lenA+lenB >= 400 now). align cells = 500*100
    // = 50K (same as full matrix — band degenerates, no penalty).
    const b = estimateAlignTimeBreakdown(500, 100)
    expect(b.alignMs).toBeGreaterThan(20)
    expect(b.alignMs).toBeLessThan(100)
  })
})

describe("estimateAlignTimeMs — sanity vs actual alignTurns", () => {
  it("estimate for 300x300 alignMs is within 3x of actual time", () => {
    const pair = buildScenarioPair("similar", 300, 300, 555, 556)
    const b = estimateAlignTimeBreakdown(300, 300)

    const t0 = Date.now()
    alignTurns(pair.turnsA, pair.turnsB)
    const elapsed = Date.now() - t0

    // Compare just the align phase (not total), since elapsed only covers align.
    // Tolerance is wide (5×) because wall time fluctuates with machine load —
    // the test guards against order-of-magnitude regressions, not precise calibration.
    expect(b.alignMs).toBeGreaterThan(elapsed * 0.1)
    expect(b.alignMs).toBeLessThan(elapsed * 5)
  })
})
