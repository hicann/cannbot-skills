#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""KB tiering 24h MVP — end-to-end demo across REAL, PERSISTED tiers through the shared Arbiter.

Owner set a 24h MVP deadline (2026-07-02). The demo runs the two-real-tier
`Arbiter([autoport-c, a5ops-b])` fallback and accepts a cannbot-c provider as
an optional third tier.

What it proves (all against REAL adapters, nothing mocked):
  1. both adapters satisfy the §14.2 `KBProvider` contract (`isinstance` runtime-checkable)
  2. b-tier resolves out of the REAL committed `src/skills/references/KB_INDEX.md`
  3. write to c = admit → SHARED gate → put, and it PERSISTS to `INDEX.jsonl`
  4. c > b PRECEDENCE via the ordered provider list
  5. conflict-surfacing: a c positive_pattern contradicting a b official-canonical is SURFACED, not silent
  6. promote c→b (owner-gated, trust=verified) → tombstone in c → entry now served from b
  7. persistence + resurrection guard: reload c from disk → tombstone survives → re-learning the
     promoted lesson is REJECTED by the gate (points to the b serial) — the §7 resurrection loop closed
  8. gate evidence-by-kind: an unbacked anti_pattern is rejected

Run: python3 src/scripts/kb_tiering/demo/kb_tiering_mvp_demo.py
Exit 0 + all [OK] = MVP closed on 2 real tiers.  (3rd tier: pass a cannbot-c provider → same script.)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_KB = Path(__file__).resolve().parents[1]          # src/scripts/kb_tiering
sys.path.insert(0, str(_KB))                        # flat-import convention (interface / adapter_a5ops)
sys.path.insert(0, str(_KB / "adapters"))

from interface import Entry, Arbiter, KBProvider, gate, hard_key, full_sig  # noqa: E402
from adapter_a5ops import A5opsProvider                                     # noqa: E402
from autoport_c import make_autoport_c                                       # noqa: E402

_KBINDEX = _KB.parents[3] / "kb" / "KB_INDEX.md"   # 2026-07-05: <plugin_root>/kb/ (_KB.parents[2]==engine)

_FAILS = []


def check(label: str, ok: bool, detail: str = ""):
    tag = "[OK]" if ok else "[FAIL]"
    if not ok:
        _FAILS.append(label)
    print(f"  {tag} {label}" + (f" — {detail}" if detail else ""))


def build_arbiter(tmp: Path, cannbot_c: KBProvider | None = None):
    """Compose the ordered provider list. c-tiers precede b (precedence = list order).
    cannbot_c is the optional third tier; None selects the two-real-tier fallback.
    """
    autoport_c = make_autoport_c(tmp / "user_kb")                          # scan, persisted JSONL
    a5ops_b = A5opsProvider(_KBINDEX, inbox=tmp / "b_promoted_inbox.jsonl")  # main, real KB_INDEX
    providers = [autoport_c]
    if cannbot_c is not None:
        providers.append(cannbot_c)                                        # optional deployment tier
    providers.append(a5ops_b)
    return Arbiter(providers), autoport_c, a5ops_b


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="kb_mvp_"))
    arb, c, b = build_arbiter(tmp)
    tiers = "+".join(p.tier for p in arb.providers)
    print(f"KB tiering MVP demo — Arbiter([{tiers}]) — real KB_INDEX b + persisted autoport-c\n")

    # 1. contract conformance (runtime-checkable Protocol)
    print("§ contract")
    check("autoport-c isinstance KBProvider", isinstance(c, KBProvider))
    check("a5ops-b   isinstance KBProvider", isinstance(b, KBProvider))
    check("a5ops-b parsed real KB_INDEX (>=500 rows)", len(b.index_rows()) >= 500,
          f"{b.parse_ok}/{b.parse_total} rows")

    # 2. b resolves out of the real committed KB
    print("§ resolve from real b-tier KB_INDEX")
    b_hit = None
    for e in b.index_rows():                                               # pick a real b entry, query its own sig
        if len(full_sig(e.claim).split()) >= 6:
            b_hit = e
            break
    r = arb.resolve(full_sig(b_hit.claim))
    check("resolve() returns a real b entry", r.status == "RESOLVED" and r.tier == "b",
          f"{r.status} -> {r.entry.id if r.entry else None} @ {r.tier}")

    # 3. write to c (admit -> shared gate -> put) + persistence
    print("§ write c (admit -> shared gate) + persist")
    # distinctive fictional lesson (fictional code 991737 + fictional symbols) so it does NOT hard-key
    # dedup-collide with any real committed KB entry — keeps the promote/persist asserts deterministic.
    lesson = "demo_kernel on chip_demo must reserve demo_dcache_slab for demo_dispatch or hit InitBuffer 991737"
    c_entry = Entry(
        id="", tier="c", role="user-local", kind="anti_pattern",
        scope={"chip_scope": "chip_demo", "applies_to": "demo_kernel"},
        key=hard_key(lesson), claim=lesson,
        evidence={"reproducible_failure": "507035 on InitBuffer, repro 2x"},
        trust="unverified", provenance={"source": "user run", "agent": "demo"},
    )
    status, ident = arb.write(c_entry, "c")
    check("write c admitted (evidence-backed anti_pattern)", status == "WRITTEN", f"{status} {ident}")
    reloaded = make_autoport_c(tmp / "user_kb")                            # fresh instance = disk reload
    check("c entry PERSISTED (survives reload)", reloaded.lookup(c_entry.key) is not None)

    # 4. c > b precedence via ordered list — use an `experience` entry: §5 kind-aware rule makes
    #    experience a SILENT local override (no conflict-surface), isolating precedence from step 5.
    print("§ c > b precedence (experience = silent local override)")
    exp_claim = "on chip_demo the demo_widget_probe warms up faster if demo_prewarm_flag is set first"
    exp = Entry(id="", tier="c", role="user-local", kind="experience", scope={},
                key=hard_key(exp_claim), claim=exp_claim, evidence={}, trust="unverified",
                provenance={"source": "user run"})
    arb.write(exp, "c")
    r2 = arb.resolve(full_sig(exp_claim))
    check("resolve() prefers c over b (clean RESOLVED, silent)", r2.status == "RESOLVED" and r2.tier == "c",
          f"{r2.status} -> tier {r2.entry.tier if r2.entry else None}")

    # 5. conflict-surfacing: c positive contradicts a seeded b official-canonical (deterministic)
    print("§ conflict-surfacing (kind-aware, not silent)")
    canon_claim = "on A5 the selective_scan backward kernel must fuse the gz chain into a single MicroAPI VF"
    b_canon = Entry(id="OL-DEMO", tier="b", role="official-canonical", kind="positive_pattern",
                    scope={}, key=hard_key(canon_claim), claim=canon_claim,
                    evidence={"canonical": True}, trust="verified",
                    provenance={"source": "a5_ops KB_INDEX"})
    b.put(b_canon)                                                         # seed a known b canonical
    c_contra = Entry(id="", tier="c", role="user-local", kind="positive_pattern",
                     scope={}, key=hard_key(canon_claim),
                     claim="on A5 the selective_scan backward kernel must NOT fuse the gz chain MicroAPI VF split it",
                     evidence={"works_proof": "local A/B faster unsplit"}, trust="unverified",
                     provenance={"source": "user run"})
    arb.write(c_contra, "c")
    rc = arb.resolve(full_sig(canon_claim))
    check(
        "c-vs-b canonical conflict SURFACED",
        rc.status == "CONFLICT_SURFACED",
        (
            f"{rc.status} (c={rc.entry.tier if rc.entry else None} "
            f"vs b={rc.conflict_with.tier if rc.conflict_with else None})"
        ),
    )

    # 6. promote c->b (owner-gated; needs trust=verified) -> tombstone c
    print("§ promote c -> b (owner-gated, evidence-verified)")
    verified = Entry(**{**c_entry.__dict__, "trust": "verified",
                        "evidence": {"reproducible_failure": "507035", "ab": "same-condition A/B"}})
    pstatus, new_serial = arb.promote(verified, "c", "b")
    check("promote c->b succeeds with verified evidence", pstatus == "PROMOTED", f"{pstatus} -> {new_serial}")
    check("promoted lesson now served from b", any(e.claim == lesson for e in b.index_rows()))
    check("source c entry tombstoned", verified.content_hash in c.tombstones())

    # 7. persistence + resurrection guard
    print("§ resurrection guard (reload c, re-learn same lesson -> rejected)")
    c_reload = make_autoport_c(tmp / "user_kb")
    check("tombstone PERSISTED across reload", verified.content_hash in c_reload.tombstones())
    arb2, c2, b2 = build_arbiter(tmp)                                      # rebuild against same disk
    # rebuild reads the same user_kb (c2 carries the tombstone) AND the same b_promoted_inbox.jsonl
    # (b2 auto-loads the promoted entry from the inbox — promotion persists across rebuild, no manual seed).
    check("promoted entry reloaded into b from inbox", any(e.claim == lesson for e in b2.index_rows()))
    relearn = Entry(**{**c_entry.__dict__, "id": ""})
    rstatus, rwhy = arb2.write(relearn, "c")
    check("re-learning promoted lesson REJECTED (tombstone hit)", rstatus == "REJECTED", rwhy)

    # 8. gate evidence-by-kind
    print("§ gate evidence-by-kind")
    unbacked = Entry(id="", tier="c", role="user-local", kind="anti_pattern", scope={},
                     key="x", claim="never do the thing", evidence={}, trust="unverified", provenance={})
    us, uwhy = arb.write(unbacked, "c")
    check("unbacked anti_pattern rejected", us == "REJECTED", uwhy)

    print()
    if _FAILS:
        print(f"MVP DEMO: {len(_FAILS)} FAIL — {_FAILS}")
        return 1
    print(f"MVP DEMO: ALL PASS on Arbiter([{tiers}]) — 2 real persisted tiers, e2e closed.")
    print("  (Optional 3rd tier: build_arbiter(tmp, cannbot_c=<adapter>) → Arbiter([c,cannbot-c,b]).)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
