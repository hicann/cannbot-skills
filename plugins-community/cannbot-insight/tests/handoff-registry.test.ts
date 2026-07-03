// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, beforeEach, afterEach } from "vitest"
import fs from "node:fs"
import path from "node:path"
import {
  HandoffSessionRecord,
  computeSessionIdShort,
  computeHandoffNum,
  computeHandoffDocName,
  computeHandoffSessionTitle,
  computeContinuationTitle,
  computeDoneFlagPath,
  extractHandoffNumFromBirthDoc,
  extractSourceShortFromDocName,
  findBirthHandoffDoc,
  findOriginalSession,
  findSessionRecord,
  addOrUpdateSessionRecord,
  addChildToSession,
  readHandoffRegistry,
  writeHandoffRegistry,
  extractOperatorNameFromPath,
  buildHandoffLinks,
} from "../src/lib/breather/handoff-registry"

const TMP_DIR = path.join("/var/folders/ft/1jjnlymd2vj5hdzhp3_pccx40000gn/T/opencode", "handoff-reg-test")

beforeEach(() => { fs.mkdirSync(TMP_DIR, { recursive: true }) })
afterEach(() => { fs.rmSync(TMP_DIR, { recursive: true, force: true }) })

function makeRecord(
  sessionId: string,
  from: string | null,
  to: string[],
  birthHandoffDoc: string | null = null,
  handoffDoc: string | null = null
): HandoffSessionRecord {
  return { sessionId, from, to, birthHandoffDoc, handoffDoc }
}

describe("computeSessionIdShort", () => {
  it("returns first 12 chars of session ID", () => {
    expect(computeSessionIdShort("ses_0fb7f9916ffe")).toBe("ses_0fb7f991")
  })

  it("returns full ID if shorter than 12 chars", () => {
    expect(computeSessionIdShort("h1")).toBe("h1")
  })

  it("different session IDs produce different shorts (collision test)", () => {
    const a = computeSessionIdShort("ses_0fb7f9916ffe")
    const b = computeSessionIdShort("ses_0fb7a22cdffe")
    expect(a).toBe("ses_0fb7f991")
    expect(b).toBe("ses_0fb7a22c")
    expect(a).not.toBe(b)
  })
})

describe("computeHandoffNum", () => {
  it("returns 1 when session has no record in registry", () => {
    expect(computeHandoffNum([], "ses_0fb7f9916ffe")).toBe(1)
  })

  it("returns 1 when session record has empty to list", () => {
    const registry = [makeRecord("ses_0fb7f9916ffe", null, [])]
    expect(computeHandoffNum(registry, "ses_0fb7f9916ffe")).toBe(1)
  })

  it("returns 2 after one handoff completed", () => {
    const registry = [makeRecord("ses_0fb7f9916ffe", null, ["h1"])]
    expect(computeHandoffNum(registry, "ses_0fb7f9916ffe")).toBe(2)
  })

  it("returns 3 after two handoffs completed", () => {
    const registry = [makeRecord("ses_0fb7f9916ffe", null, ["h1", "h1_2"])]
    expect(computeHandoffNum(registry, "ses_0fb7f9916ffe")).toBe(3)
  })

  it("does not confuse handoffs from different sessions", () => {
    const registry = [
      makeRecord("ses_0fb7f9916ffe", null, ["h1"]),
      makeRecord("ses_0fb7a22cdffe", null, []),
    ]
    expect(computeHandoffNum(registry, "ses_0fb7f9916ffe")).toBe(2)
    expect(computeHandoffNum(registry, "ses_0fb7a22cdffe")).toBe(1)
  })

  it("counts correctly when continuation session does second handoff", () => {
    const c1Id = "ses_0fb7a22cdffe"
    const registry = [
      makeRecord("ses_0fb7f9916ffe", null, ["h1"]),
      makeRecord("h1", "ses_0fb7f9916ffe", [c1Id]),
      makeRecord(c1Id, "h1", ["h2"], "SESSION-HANDOFF-ses_0fb7f991-1.md"),
    ]
    expect(computeHandoffNum(registry, c1Id)).toBe(2)
  })
})

describe("computeHandoffDocName", () => {
  it("formats doc name with 12-char short and handoff num", () => {
    expect(computeHandoffDocName("ses_0fb7f991", 1)).toBe("SESSION-HANDOFF-ses_0fb7f991-1.md")
  })
})

describe("computeHandoffSessionTitle", () => {
  it("formats title with 12-char short and handoff num", () => {
    expect(computeHandoffSessionTitle("ses_0fb7f991", 1)).toBe("ses_0fb7f991-handoff-1")
  })
})

describe("computeContinuationTitle", () => {
  it("formats title with 12-char short and handoff num", () => {
    expect(computeContinuationTitle("ses_0fb7f991", 1)).toBe("ses_0fb7f991-continue-1")
  })
})

describe("computeDoneFlagPath", () => {
  it("returns path under operators docs dir when operatorName provided", () => {
    const result = computeDoneFlagPath("/project", "xlog1py", "ses_0fb7f991", 1)
    expect(result).toBe("/project/operators/xlog1py/docs/SESSION-HANDOFF-ses_0fb7f991-1.done")
  })

  it("returns path under project root when no operatorName", () => {
    const result = computeDoneFlagPath("/project", null, "ses_0fb7f991", 1)
    expect(result).toBe("/project/SESSION-HANDOFF-ses_0fb7f991-1.done")
  })
})

describe("extractHandoffNumFromBirthDoc", () => {
  it("extracts handoff num from birthHandoffDoc filename", () => {
    expect(extractHandoffNumFromBirthDoc("SESSION-HANDOFF-ses_0fb7f991-1.md")).toBe(1)
    expect(extractHandoffNumFromBirthDoc("SESSION-HANDOFF-ses_0fb7f991-3.md")).toBe(3)
  })

  it("returns 1 if no number found in filename", () => {
    expect(extractHandoffNumFromBirthDoc("SESSION-HANDOFF-ses_0fb7f991.md")).toBe(1)
  })
})

describe("extractSourceShortFromDocName", () => {
  it("extracts source short from handoff doc name", () => {
    expect(extractSourceShortFromDocName("SESSION-HANDOFF-ses_0fb7f991-1.md")).toBe("ses_0fb7f991")
  })

  it("extracts source short from continuation birth doc", () => {
    expect(extractSourceShortFromDocName("SESSION-HANDOFF-ses_0fb7a22c-2.md")).toBe("ses_0fb7a22c")
  })

  it("returns empty string for invalid format", () => {
    expect(extractSourceShortFromDocName("random-file.md")).toBe("")
  })
})

describe("findSessionRecord", () => {
  it("returns record when found", () => {
    const registry = [makeRecord("h1", "orig", [])]
    expect(findSessionRecord(registry, "h1")).toEqual(makeRecord("h1", "orig", []))
  })

  it("returns null when not found", () => {
    expect(findSessionRecord([], "h1")).toBeNull()
  })
})

describe("addOrUpdateSessionRecord", () => {
  it("adds new record when not exists", () => {
    const registry: HandoffSessionRecord[] = []
    const result = addOrUpdateSessionRecord(registry, makeRecord("h1", "orig", []))
    expect(result).toHaveLength(1)
    expect(result[0]).toEqual(makeRecord("h1", "orig", []))
  })

  it("updates existing record", () => {
    const registry = [makeRecord("h1", "orig", [])]
    const result = addOrUpdateSessionRecord(registry, makeRecord("h1", "orig", ["c1"]))
    expect(result).toHaveLength(1)
    expect(result[0]).toEqual(makeRecord("h1", "orig", ["c1"]))
  })

  it("does not mutate original registry", () => {
    const registry = [makeRecord("h1", "orig", [])]
    const result = addOrUpdateSessionRecord(registry, makeRecord("h2", "orig", []))
    expect(registry).toHaveLength(1)
    expect(result).toHaveLength(2)
  })
})

describe("addChildToSession", () => {
  it("adds child id to parent to list", () => {
    const registry = [makeRecord("orig", null, ["h1"])]
    const result = addChildToSession(registry, "orig", "h1_2")
    expect(result[0].to).toEqual(["h1", "h1_2"])
  })

  it("returns unchanged registry if parent not found", () => {
    const registry = [makeRecord("orig", null, [])]
    const result = addChildToSession(registry, "nonexistent", "h1")
    expect(result).toEqual(registry)
  })

  it("does not mutate original registry", () => {
    const registry = [makeRecord("orig", null, [])]
    const result = addChildToSession(registry, "orig", "h1")
    expect(registry[0].to).toEqual([])
    expect(result[0].to).toEqual(["h1"])
  })
})

describe("findBirthHandoffDoc", () => {
  it("returns null for original session (not born from handoff)", () => {
    const registry = [makeRecord("ses_0fb7f9916ffe", null, [], null, null)]
    expect(findBirthHandoffDoc(registry, "ses_0fb7f9916ffe")).toBeNull()
  })

  it("returns birthHandoffDoc for continuation session", () => {
    const registry = [makeRecord("c1", "h1", [], "SESSION-HANDOFF-ses_0fb7f991-1.md", null)]
    expect(findBirthHandoffDoc(registry, "c1")).toBe("SESSION-HANDOFF-ses_0fb7f991-1.md")
  })

  it("returns null for handoff session (not born from handoff)", () => {
    const registry = [makeRecord("h1", "ses_0fb7f9916ffe", [], null, "SESSION-HANDOFF-ses_0fb7f991-1.md")]
    expect(findBirthHandoffDoc(registry, "h1")).toBeNull()
  })

  it("returns null for session not in registry at all", () => {
    expect(findBirthHandoffDoc([], "c1")).toBeNull()
  })
})

describe("findOriginalSession", () => {
  it("returns null for session not in registry", () => {
    expect(findOriginalSession([], "h1")).toBeNull()
  })

  it("returns null for session that has no parent (from=null)", () => {
    const registry = [makeRecord("orig", null, [])]
    expect(findOriginalSession(registry, "orig")).toBeNull()
  })

  it("traces back to original session through handoff chain", () => {
    const origId = "ses_0fb7f9916ffe"
    const registry = [
      makeRecord(origId, null, ["h1"]),
      makeRecord("h1", origId, ["c1"]),
      makeRecord("c1", "h1", [], "SESSION-HANDOFF-ses_0fb7f991-1.md"),
    ]
    expect(findOriginalSession(registry, "c1")).toBe(origId)
    expect(findOriginalSession(registry, "h1")).toBe(origId)
  })
})

describe("readHandoffRegistry / writeHandoffRegistry", () => {
  it("returns empty array when registry file does not exist", () => {
    expect(readHandoffRegistry(TMP_DIR)).toEqual([])
  })

  it("returns empty array when registry file is invalid JSON", () => {
    fs.writeFileSync(path.join(TMP_DIR, ".opencode-handoff-registry.json"), "bad json", "utf-8")
    expect(readHandoffRegistry(TMP_DIR)).toEqual([])
  })

  it("reads and writes registry correctly", () => {
    const registry = [
      makeRecord("orig", null, ["h1"], null, null),
      makeRecord("h1", "orig", ["c1"], null, "SESSION-HANDOFF-ses_0fb7f991-1.md"),
    ]
    writeHandoffRegistry(TMP_DIR, registry)
    const read = readHandoffRegistry(TMP_DIR)
    expect(read).toEqual(registry)
  })
})

describe("extractOperatorNameFromPath", () => {
  it("returns null when operators dir does not exist", () => {
    expect(extractOperatorNameFromPath(TMP_DIR)).toBeNull()
  })

  it("returns null when no operator has docs dir", () => {
    fs.mkdirSync(path.join(TMP_DIR, "operators", "myop"), { recursive: true })
    expect(extractOperatorNameFromPath(TMP_DIR)).toBeNull()
  })

  it("returns first operator with docs dir alphabetically", () => {
    fs.mkdirSync(path.join(TMP_DIR, "operators", "xlog1py", "docs"), { recursive: true })
    fs.mkdirSync(path.join(TMP_DIR, "operators", "ascendc", "docs"), { recursive: true })
    expect(extractOperatorNameFromPath(TMP_DIR)).toBe("ascendc")
  })
})

describe("naming across multi-level handoff chain", () => {
  it("original session first handoff naming", () => {
    const origId = "ses_0fb7f9916ffe"
    const origShort = computeSessionIdShort(origId)
    const registry = [makeRecord(origId, null, [])]
    const handoffNum = computeHandoffNum(registry, origId)
    expect(handoffNum).toBe(1)
    expect(computeHandoffDocName(origShort, handoffNum)).toBe("SESSION-HANDOFF-ses_0fb7f991-1.md")
    expect(computeHandoffSessionTitle(origShort, handoffNum)).toBe("ses_0fb7f991-handoff-1")
  })

  it("continuation session first handoff naming uses continuation session ID, not original", () => {
    const origId = "ses_0fb7f9916ffe"
    const c1Id = "ses_0fb7a22cdffe"
    const c1Short = computeSessionIdShort(c1Id)
    const registry = [
      makeRecord(origId, null, ["h1"]),
      makeRecord("h1", origId, [c1Id]),
      makeRecord(c1Id, "h1", [], "SESSION-HANDOFF-ses_0fb7f991-1.md"),
    ]
    const handoffNum = computeHandoffNum(registry, c1Id)
    expect(handoffNum).toBe(1)
    expect(computeHandoffDocName(c1Short, handoffNum)).toBe("SESSION-HANDOFF-ses_0fb7a22c-1.md")
    expect(computeHandoffSessionTitle(c1Short, handoffNum)).toBe("ses_0fb7a22c-handoff-1")
  })

  it("original session second handoff naming", () => {
    const origId = "ses_0fb7f9916ffe"
    const origShort = computeSessionIdShort(origId)
    const registry = [makeRecord(origId, null, ["h1"])]
    const handoffNum = computeHandoffNum(registry, origId)
    expect(handoffNum).toBe(2)
    expect(computeHandoffDocName(origShort, handoffNum)).toBe("SESSION-HANDOFF-ses_0fb7f991-2.md")
    expect(computeHandoffSessionTitle(origShort, handoffNum)).toBe("ses_0fb7f991-handoff-2")
  })

  it("no collision between different sessions at 12-char prefix", () => {
    const a = computeSessionIdShort("ses_0fb7f9916ffe")
    const b = computeSessionIdShort("ses_0fb7a22cdffe")
    expect(a).not.toBe(b)
    expect(computeHandoffDocName(a, 1)).not.toBe(computeHandoffDocName(b, 1))
  })
})

describe("buildHandoffLinks", () => {
  it("returns no parent and no children for session not in any handoff", () => {
    const links = buildHandoffLinks([], "orig", new Map())
    expect(links.parent).toBeNull()
    expect(links.children).toEqual([])
  })

  it("returns no parent and no children for original session with empty to", () => {
    const registry = [makeRecord("orig", null, [])]
    const links = buildHandoffLinks(registry, "orig", new Map())
    expect(links.parent).toBeNull()
    expect(links.children).toEqual([])
  })

  it("original session has child handoff session", () => {
    const origId = "ses_1234abcd1234"
    const registry = [
      makeRecord(origId, null, ["h1"]),
      makeRecord("h1", origId, [], null, "SESSION-HANDOFF-ses_1234abcd-1.md"),
    ]
    const links = buildHandoffLinks(registry, origId, new Map())
    expect(links.parent).toBeNull()
    expect(links.children).toEqual([{ id: "h1", title: "ses_1234abcd-handoff-1" }])
  })

  it("handoff session: parent=original (via titleMap), child=continuation", () => {
    const origId = "ses_1234abcd1234"
    const registry = [
      makeRecord(origId, null, ["h1"]),
      makeRecord("h1", origId, ["c1"], null, "SESSION-HANDOFF-ses_1234abcd-1.md"),
      makeRecord("c1", "h1", [], "SESSION-HANDOFF-ses_1234abcd-1.md"),
    ]
    const titleMap = new Map([[origId, "xlog1py算子生成"]])
    const links = buildHandoffLinks(registry, "h1", titleMap)
    expect(links.parent).toEqual({ id: origId, title: "xlog1py算子生成" })
    expect(links.children).toEqual([{ id: "c1", title: "ses_1234abcd-continue-1" }])
  })

  it("handoff session: parent=original (no titleMap, shows null)", () => {
    const origId = "ses_1234abcd1234"
    const registry = [
      makeRecord(origId, null, ["h1"]),
      makeRecord("h1", origId, ["c1"], null, "SESSION-HANDOFF-ses_1234abcd-1.md"),
      makeRecord("c1", "h1", [], "SESSION-HANDOFF-ses_1234abcd-1.md"),
    ]
    const links = buildHandoffLinks(registry, "h1", new Map())
    expect(links.parent).toEqual({ id: origId, title: null })
  })

  it("continuation session: parent=handoff (computed title), no children", () => {
    const origId = "ses_0fb7f9916ffe"
    const registry = [
      makeRecord(origId, null, ["h1"]),
      makeRecord("h1", origId, ["c1"], null, "SESSION-HANDOFF-ses_0fb7f991-1.md"),
      makeRecord("c1", "h1", [], "SESSION-HANDOFF-ses_0fb7f991-1.md"),
    ]
    const links = buildHandoffLinks(registry, "c1", new Map())
    expect(links.parent).toEqual({ id: "h1", title: "ses_0fb7f991-handoff-1" })
    expect(links.children).toEqual([])
  })

  it("continuation session does second handoff: parent=handoff, children=[handoff-2]", () => {
    const origId = "ses_0fb7f9916ffe"
    const c1Id = "ses_0fb7a22cdffe"
    const c1Short = computeSessionIdShort(c1Id)
    const registry = [
      makeRecord(origId, null, ["h1"]),
      makeRecord("h1", origId, [c1Id], null, "SESSION-HANDOFF-ses_0fb7f991-1.md"),
      makeRecord(c1Id, "h1", ["h2"], "SESSION-HANDOFF-ses_0fb7f991-1.md"),
      makeRecord("h2", c1Id, [], null, `SESSION-HANDOFF-${c1Short}-1.md`),
    ]
    const linksC1 = buildHandoffLinks(registry, c1Id, new Map())
    expect(linksC1.parent).toEqual({ id: "h1", title: "ses_0fb7f991-handoff-1" })
    expect(linksC1.children).toEqual([{ id: "h2", title: `${c1Short}-handoff-1` }])
  })

  it("second handoff session: parent=continuation (computed title), children=[continuation-2]", () => {
    const origId = "ses_0fb7f9916ffe"
    const c1Id = "ses_0fb7a22cdffe"
    const c1Short = computeSessionIdShort(c1Id)
    const registry = [
      makeRecord(origId, null, ["h1"]),
      makeRecord("h1", origId, [c1Id], null, "SESSION-HANDOFF-ses_0fb7f991-1.md"),
      makeRecord(c1Id, "h1", ["h2"], "SESSION-HANDOFF-ses_0fb7f991-1.md"),
      makeRecord("h2", c1Id, ["c2"], null, `SESSION-HANDOFF-${c1Short}-1.md`),
      makeRecord("c2", "h2", [], `SESSION-HANDOFF-${c1Short}-1.md`),
    ]
    const linksH2 = buildHandoffLinks(registry, "h2", new Map())
    expect(linksH2.parent).toEqual({ id: c1Id, title: "ses_0fb7f991-continue-1" })
    expect(linksH2.children).toEqual([{ id: "c2", title: `${c1Short}-continue-1` }])
  })

  it("second continuation session: parent=handoff-2 (computed title), no children", () => {
    const origId = "ses_0fb7f9916ffe"
    const c1Id = "ses_0fb7a22cdffe"
    const c1Short = computeSessionIdShort(c1Id)
    const registry = [
      makeRecord(origId, null, ["h1"]),
      makeRecord("h1", origId, [c1Id], null, "SESSION-HANDOFF-ses_0fb7f991-1.md"),
      makeRecord(c1Id, "h1", ["h2"], "SESSION-HANDOFF-ses_0fb7f991-1.md"),
      makeRecord("h2", c1Id, ["c2"], null, `SESSION-HANDOFF-${c1Short}-1.md`),
      makeRecord("c2", "h2", [], `SESSION-HANDOFF-${c1Short}-1.md`),
    ]
    const linksC2 = buildHandoffLinks(registry, "c2", new Map())
    expect(linksC2.parent).toEqual({ id: "h2", title: `${c1Short}-handoff-1` })
    expect(linksC2.children).toEqual([])
  })
})

describe("buildHandoffLinks - one session handoffs multiple times", () => {
  it("original session handoffs twice: two handoff children with different nums", () => {
    const origId = "ses_0fb7f9916ffe"
    const origShort = computeSessionIdShort(origId)
    const h1Doc = `SESSION-HANDOFF-${origShort}-1.md`
    const h2Doc = `SESSION-HANDOFF-${origShort}-2.md`
    const registry = [
      makeRecord(origId, null, ["h1", "h1_2"]),
      makeRecord("h1", origId, ["c1"], null, h1Doc),
      makeRecord("c1", "h1", [], h1Doc),
      makeRecord("h1_2", origId, ["c1_2"], null, h2Doc),
      makeRecord("c1_2", "h1_2", [], h2Doc),
    ]
    const titleMap = new Map([[origId, "算子开发"]])
    const links = buildHandoffLinks(registry, origId, titleMap)
    expect(links.parent).toBeNull()
    expect(links.children).toEqual([
      { id: "h1", title: `${origShort}-handoff-1` },
      { id: "h1_2", title: `${origShort}-handoff-2` },
    ])
  })

  it("first handoff session: parent=original, child=continuation-1", () => {
    const origId = "ses_0fb7f9916ffe"
    const origShort = computeSessionIdShort(origId)
    const h1Doc = `SESSION-HANDOFF-${origShort}-1.md`
    const registry = [
      makeRecord(origId, null, ["h1", "h1_2"]),
      makeRecord("h1", origId, ["c1"], null, h1Doc),
      makeRecord("c1", "h1", [], h1Doc),
      makeRecord("h1_2", origId, ["c1_2"], null, `SESSION-HANDOFF-${origShort}-2.md`),
      makeRecord("c1_2", "h1_2", [], `SESSION-HANDOFF-${origShort}-2.md`),
    ]
    const linksH1 = buildHandoffLinks(registry, "h1", new Map())
    expect(linksH1.parent).toEqual({ id: origId, title: null })
    expect(linksH1.children).toEqual([{ id: "c1", title: `${origShort}-continue-1` }])
  })

  it("second handoff session: parent=original, child=continuation-2", () => {
    const origId = "ses_0fb7f9916ffe"
    const origShort = computeSessionIdShort(origId)
    const h2Doc = `SESSION-HANDOFF-${origShort}-2.md`
    const registry = [
      makeRecord(origId, null, ["h1", "h1_2"]),
      makeRecord("h1", origId, ["c1"], null, `SESSION-HANDOFF-${origShort}-1.md`),
      makeRecord("c1", "h1", [], `SESSION-HANDOFF-${origShort}-1.md`),
      makeRecord("h1_2", origId, ["c1_2"], null, h2Doc),
      makeRecord("c1_2", "h1_2", [], h2Doc),
    ]
    const linksH1_2 = buildHandoffLinks(registry, "h1_2", new Map())
    expect(linksH1_2.parent).toEqual({ id: origId, title: null })
    expect(linksH1_2.children).toEqual([{ id: "c1_2", title: `${origShort}-continue-2` }])
  })

  it("first continuation: parent=handoff-1, no children", () => {
    const origId = "ses_0fb7f9916ffe"
    const origShort = computeSessionIdShort(origId)
    const h1Doc = `SESSION-HANDOFF-${origShort}-1.md`
    const registry = [
      makeRecord(origId, null, ["h1", "h1_2"]),
      makeRecord("h1", origId, ["c1"], null, h1Doc),
      makeRecord("c1", "h1", [], h1Doc),
      makeRecord("h1_2", origId, ["c1_2"], null, `SESSION-HANDOFF-${origShort}-2.md`),
      makeRecord("c1_2", "h1_2", [], `SESSION-HANDOFF-${origShort}-2.md`),
    ]
    const linksC1 = buildHandoffLinks(registry, "c1", new Map())
    expect(linksC1.parent).toEqual({ id: "h1", title: `${origShort}-handoff-1` })
    expect(linksC1.children).toEqual([])
  })

  it("second continuation: parent=handoff-2, no children", () => {
    const origId = "ses_0fb7f9916ffe"
    const origShort = computeSessionIdShort(origId)
    const h2Doc = `SESSION-HANDOFF-${origShort}-2.md`
    const registry = [
      makeRecord(origId, null, ["h1", "h1_2"]),
      makeRecord("h1", origId, ["c1"], null, `SESSION-HANDOFF-${origShort}-1.md`),
      makeRecord("c1", "h1", [], `SESSION-HANDOFF-${origShort}-1.md`),
      makeRecord("h1_2", origId, ["c1_2"], null, h2Doc),
      makeRecord("c1_2", "h1_2", [], h2Doc),
    ]
    const linksC1_2 = buildHandoffLinks(registry, "c1_2", new Map())
    expect(linksC1_2.parent).toEqual({ id: "h1_2", title: `${origShort}-handoff-2` })
    expect(linksC1_2.children).toEqual([])
  })

  it("computeHandoffNum after two handoffs = 3", () => {
    const origId = "ses_0fb7f9916ffe"
    const registry = [makeRecord(origId, null, ["h1", "h1_2"])]
    expect(computeHandoffNum(registry, origId)).toBe(3)
  })
})

describe("buildHandoffLinks - chain: A handoff → B → C handoff → D", () => {
  it("full chain: original → h1 → c1 → h2 → c2, each session's parent and children", () => {
    const origId = "ses_0fb7f9916ffe"
    const origShort = computeSessionIdShort(origId)
    const c1Id = "ses_0fb7a22cdffe"
    const c1Short = computeSessionIdShort(c1Id)
    const h1Doc = `SESSION-HANDOFF-${origShort}-1.md`
    const h2Doc = `SESSION-HANDOFF-${c1Short}-1.md`

    const registry = [
      makeRecord(origId, null, ["h1"]),
      makeRecord("h1", origId, [c1Id], null, h1Doc),
      makeRecord(c1Id, "h1", ["h2"], h1Doc),
      makeRecord("h2", c1Id, ["c2"], null, h2Doc),
      makeRecord("c2", "h2", [], h2Doc),
    ]

    const linksOrig = buildHandoffLinks(registry, origId, new Map())
    expect(linksOrig.parent).toBeNull()
    expect(linksOrig.children).toEqual([{ id: "h1", title: `${origShort}-handoff-1` }])

    const linksH1 = buildHandoffLinks(registry, "h1", new Map())
    expect(linksH1.parent).toEqual({ id: origId, title: null })
    expect(linksH1.children).toEqual([{ id: c1Id, title: `${origShort}-continue-1` }])

    const linksC1 = buildHandoffLinks(registry, c1Id, new Map())
    expect(linksC1.parent).toEqual({ id: "h1", title: `${origShort}-handoff-1` })
    expect(linksC1.children).toEqual([{ id: "h2", title: `${c1Short}-handoff-1` }])

    const linksH2 = buildHandoffLinks(registry, "h2", new Map())
    expect(linksH2.parent).toEqual({ id: c1Id, title: `${origShort}-continue-1` })
    expect(linksH2.children).toEqual([{ id: "c2", title: `${c1Short}-continue-1` }])

    const linksC2 = buildHandoffLinks(registry, "c2", new Map())
    expect(linksC2.parent).toEqual({ id: "h2", title: `${c1Short}-handoff-1` })
    expect(linksC2.children).toEqual([])
  })

  it("findOriginalSession traces to original through chain", () => {
    const origId = "ses_0fb7f9916ffe"
    const c1Id = "ses_0fb7a22cdffe"
    const registry = [
      makeRecord(origId, null, ["h1"]),
      makeRecord("h1", origId, [c1Id]),
      makeRecord(c1Id, "h1", ["h2"]),
      makeRecord("h2", c1Id, ["c2"]),
      makeRecord("c2", "h2", []),
    ]
    expect(findOriginalSession(registry, "c2")).toBe(origId)
    expect(findOriginalSession(registry, "h2")).toBe(origId)
    expect(findOriginalSession(registry, c1Id)).toBe(origId)
    expect(findOriginalSession(registry, "h1")).toBe(origId)
  })

  it("chain with titleMap: original name shown in parent chain", () => {
    const origId = "ses_0fb7f9916ffe"
    const origShort = computeSessionIdShort(origId)
    const registry = [
      makeRecord(origId, null, ["h1"]),
      makeRecord("h1", origId, ["c1"], null, `SESSION-HANDOFF-${origShort}-1.md`),
      makeRecord("c1", "h1", [], `SESSION-HANDOFF-${origShort}-1.md`),
    ]
    const titleMap = new Map([[origId, "算子生成任务"]])
    const linksH1 = buildHandoffLinks(registry, "h1", titleMap)
    expect(linksH1.parent).toEqual({ id: origId, title: "算子生成任务" })
  })
})

describe("buildHandoffLinks - uses actual registry data", () => {
  it("matches expected parent/child titles with real IDs", () => {
    const registry: HandoffSessionRecord[] = [
      makeRecord("ses_0fb42c2b", null, ["ses_0fb42908"]),
      makeRecord("ses_0fb42908", "ses_0fb42c2b", ["ses_0fb41f88"], null, "SESSION-HANDOFF-ses_0fb42c2b-1.md"),
      makeRecord("ses_0fb41f88", "ses_0fb42908", ["ses_0fb41a0a"], "SESSION-HANDOFF-ses_0fb42c2b-1.md"),
      makeRecord("ses_0fb41a0a", "ses_0fb41f88", ["ses_0fb410d5"], null, "SESSION-HANDOFF-ses_0fb41f88-1.md"),
      makeRecord("ses_0fb410d5", "ses_0fb41a0a", [], "SESSION-HANDOFF-ses_0fb41f88-1.md"),
    ]

    const linksContinuation = buildHandoffLinks(registry, "ses_0fb41f88", new Map())
    expect(linksContinuation.parent).toEqual({ id: "ses_0fb42908", title: "ses_0fb42c2b-handoff-1" })
    expect(linksContinuation.children).toEqual([{ id: "ses_0fb41a0a", title: "ses_0fb41f88-handoff-1" }])

    const linksHandoff = buildHandoffLinks(registry, "ses_0fb41a0a", new Map())
    expect(linksHandoff.parent).toEqual({ id: "ses_0fb41f88", title: "ses_0fb42c2b-continue-1" })
    expect(linksHandoff.children).toEqual([{ id: "ses_0fb410d5", title: "ses_0fb41f88-continue-1" }])
  })
})
