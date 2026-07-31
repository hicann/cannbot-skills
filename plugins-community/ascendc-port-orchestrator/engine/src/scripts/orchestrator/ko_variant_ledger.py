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
"""ko_variant_ledger.py — deterministic kernel-identity gate for ko/fo variant re-run.

Problem (back issue #2, Gap 1): aog-kernel-optimizer / aog-fused-optimizer are
re-spawned from scratch under short quota windows (ko/fo ~29 min per run). Each
iteration is already persisted (`optimization_log.md`, `## Opt{N}` entries), but
nothing reads it back on re-spawn -> the optimizer re-runs variants it already
tried -> livelock when the usable quota window < one iteration.

Naive fix "skip already-run variants" is a CORRECTNESS HAZARD: if the kernel was
rewritten between two spawns (a Kind-2 directive from the worker), the prior
variant conclusions were drawn against a DIFFERENT kernel and are now stale.
Skipping them applies old-kernel conclusions to a new kernel — worse than
re-running (re-running is only slow; skip-on-stale is WRONG).

So the gate keys on KERNEL IDENTITY (md5), not variant name. This module is the
structural enforcer (deterministic scanner)
principle "structurally enforced, not relying on LLM compliance". The ko/fo brief
does NOT make the md5 judgment; it calls this and follows the verdict.

LAST-RECORD SEMANTICS (the load-bearing design decision)
--------------------------------------------------------
The write side records `kernel_md5` AFTER each keep/revert DECISION — i.e. after
the kernel has been left in its post-decision resting state (kept edit stays;
reverted edit is undone). Therefore **the last Opt entry's recorded md5 is, by
construction, the identity of the kernel currently on disk** — *unless* something
outside the log (the worker) rewrote the kernel since the optimizer last ran.

The verdict compares ONLY the last recorded md5 to the current kernel md5:

  - last_recorded == current  -> the log's endpoint IS the current kernel, so the
    whole variant history is a faithful record of what was already explored on
    this kernel lineage. Every logged variant is SKIPPABLE (already tried); the
    optimizer should pick something new. baseline_changed = False.
  - last_recorded != current  -> the kernel was changed OUTSIDE the log (worker
    rewrite), so the entire history was measured against a now-superseded kernel
    and is stale. Every logged variant is MUST_RERUN. baseline_changed = True.
  - no entries / no recorded md5 (old-format log) / unresolvable kernel -> cannot
    establish identity -> nothing skippable (safe re-run), baseline_changed False.

Why last-record and not per-entry: per-entry comparison is WRONG here. Across a
real N>=2 session the earlier entries legitimately hold *different* md5s (Opt0
reverted -> baseline md5; Opt1 kept -> new md5), so a per-entry "skippable iff
this entry's md5 == current" makes every entry but the last look stale, flips
baseline_changed on every re-spawn, and (via the brief's "baseline_changed ->
rerun everything" rule) nukes the entire skip set — the livelock is left intact
and a false "baseline changed" line is written to the provenance log each spawn.
The endpoint is the only record that must equal the current kernel; compare that.

Kernel identity = md5 over the sorted concatenation of `kernel/*.h` + `kernel/*.cpp`
contents (the sources the optimizer edits). Missing log or kernel -> safe default:
nothing skippable (re-run), never a false skip.

CLI:
  python3 ko_variant_ledger.py --workspace workspace/<op>                     # -> JSON verdict
  python3 ko_variant_ledger.py --workspace workspace/<op> --print-kernel-md5  # -> the one md5 to record in an Opt entry

IMPORTANT — write side must use `--print-kernel-md5`, NOT a hand-rolled hash.
The recorded `kernel_md5` is only meaningful if it is produced by the *same*
function the read side compares against (`compute_kernel_md5`). A shell
`md5sum kernel/*.h kernel/*.cpp` emits one hash per file (not a single value),
and `cat ... | md5sum` yields yet a third value — neither equals what the ledger
computes. Record exactly what `--print-kernel-md5` prints, AFTER the keep/revert
decision (so the last entry == the current kernel).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Optional

# Files that constitute "the kernel" for identity purposes — the sources the
# optimizer edits. Deterministic order (sorted by name) so the hash is stable.
_KERNEL_GLOBS = ("*.h", "*.cpp")

# An Opt entry starts with a markdown H2 whose text begins "Opt<N>". Lenient on
# the separator/description so it survives brief-format drift.
_OPT_HEADER_RE = re.compile(r"^##\s+(Opt\d+)\b(.*)$", re.MULTILINE)

# The kernel_md5 field the write-side brief records inside each Opt block. Lenient
# on label spelling/case and bullet/prefix so a hand-written log still parses.
_MD5_FIELD_RE = re.compile(
    r"(?im)^[ \t>*\-]*kernel[_\- ]?md5\s*[:=]\s*`?([0-9a-f]{32})`?\s*$"
)


def compute_kernel_md5(workspace: Path) -> Optional[str]:
    """md5 over the sorted concat of kernel/*.h + kernel/*.cpp byte contents.

    Returns None if there is no kernel dir or no kernel source files — the caller
    treats that as "cannot establish identity" -> nothing skippable (safe re-run).
    """
    kernel_dir = workspace / "kernel"
    if not kernel_dir.is_dir():
        return None
    files: list[Path] = []
    for pat in _KERNEL_GLOBS:
        files.extend(kernel_dir.glob(pat))
    if not files:
        return None
    h = hashlib.md5()
    # Sort by name so the digest is independent of filesystem enumeration order.
    for f in sorted(files, key=lambda p: p.name):
        # Include the name so a rename that swaps two files' contents still
        # changes the identity.
        h.update(f.name.encode("utf-8"))
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def parse_opt_entries(log_text: str) -> list[dict]:
    """Parse `## Opt{N}` blocks. Each -> {name, desc, kernel_md5|None}.

    kernel_md5 is the first kernel_md5 field found within that block (None if the
    entry predates the write-side change or omitted it).
    """
    entries: list[dict] = []
    matches = list(_OPT_HEADER_RE.finditer(log_text))
    for i, m in enumerate(matches):
        name = m.group(1)
        desc = m.group(2).lstrip(" —-\t").strip()
        block_start = m.end()
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(log_text)
        block = log_text[block_start:block_end]
        md5_m = _MD5_FIELD_RE.search(block)
        entries.append({
            "name": name,
            "desc": desc,
            "kernel_md5": md5_m.group(1) if md5_m else None,
        })
    return entries


def _last_recorded_md5(entries: list[dict]) -> Optional[str]:
    """The md5 of the LAST Opt entry that recorded one (walk from the end).

    This is the log's endpoint identity. Old-format entries without a recorded
    md5 are skipped so a trailing legacy entry doesn't blind the check; if NO
    entry recorded an md5, returns None (identity cannot be established).
    """
    for e in reversed(entries):
        if e["kernel_md5"] is not None:
            return e["kernel_md5"]
    return None


def build_verdict(workspace: Path) -> dict:
    """Deterministic verdict the ko/fo brief follows verbatim (last-record model)."""
    workspace = Path(workspace)
    current_md5 = compute_kernel_md5(workspace)
    log_path = workspace / "optimization_log.md"
    log_present = log_path.is_file()
    entries = parse_opt_entries(log_path.read_text()) if log_present else []
    names = [e["name"] for e in entries]
    last_md5 = _last_recorded_md5(entries)

    # Decide the single verdict for the WHOLE log by comparing its endpoint to
    # the current kernel. See module docstring "LAST-RECORD SEMANTICS".
    if not entries:
        # Fresh op / empty log -> nothing to skip, nothing stale.
        skippable, must_rerun, baseline_changed = [], [], False
    elif current_md5 is None:
        # Kernel identity unresolvable (no kernel sources) -> never a false skip.
        skippable, must_rerun, baseline_changed = [], list(names), False
    elif last_md5 is None:
        # Log has entries but none recorded an md5 (predates write-side change).
        # Can't prove the endpoint matches -> re-run all, but this is NOT a
        # baseline change (missing != different).
        skippable, must_rerun, baseline_changed = [], list(names), False
    elif last_md5 == current_md5:
        # Endpoint identity == current kernel -> whole history is a faithful
        # record of what was already explored on this kernel -> all skippable.
        skippable, must_rerun, baseline_changed = list(names), [], False
    else:
        # Current kernel differs from where the log left off -> the kernel was
        # rewritten outside the log -> the whole history is stale.
        skippable, must_rerun, baseline_changed = [], list(names), True

    return {
        "current_kernel_md5": current_md5,
        "kernel_resolved": current_md5 is not None,
        "log_present": log_present,
        "n_entries": len(entries),
        "last_recorded_md5": last_md5,
        "skippable": skippable,       # brief: do NOT re-run these
        "must_rerun": must_rerun,     # brief: re-run these (stale/unverifiable)
        "baseline_changed": baseline_changed,  # brief: note "baseline changed" in log
    }


# ---------------------------------------------------------------------------
# Byte-identical-optimizer CONVERGENCE ledger (2026-07-24, iter_cap
# await_optimizer graceful-finalize fix).
#
# Problem: when the optimizer LANDS a kernel whose perf sits in the band
# between the FINALIZE floor (e.g. 0.6× shippable) and the PARITY optimization
# target (1.0×, owner default 2026-07-21), the FSM `await_optimizer` finalize
# transition (`verification_perf_below_threshold: false`) does NOT fire (perf is
# below the parity target) and control falls through to "keep iterating within
# budget". The optimizer re-spawns and re-lands the SAME byte-identical kernel
# (0 net edits — nothing left to improve) until iter_cap trips → exit-2, with no
# terminal finalize. The FSM needs to recognize "optimizer converged
# (byte-identical) + already shippable → finalize" instead of looping.
#
# This ledger provides that byte-identical signal DETERMINISTICALLY — the
# orchestrator records the current kernel md5 (via compute_kernel_md5, the same
# identity function the variant gate uses) AFTER each await_optimizer spawn, so
# the check does NOT depend on the ko brief self-reporting `kernel_md5` in
# optimization_log.md (which it does not, as of 2026-07-24). The FSM DSL
# primitive `optimizer_kernel_converged` reads this ledger.
# ---------------------------------------------------------------------------
_OPTIMIZER_SIG_LEDGER = ".optimizer_kernel_sig.jsonl"


def record_optimizer_kernel_signature(
    workspace: Path, *, spawn_index: Optional[int] = None
) -> Optional[str]:
    """Append the CURRENT kernel md5 to the optimizer-signature ledger after an
    aog-kernel-optimizer spawn. Returns the recorded md5 (or None when the kernel
    is unresolvable). Fail-open: never raises — a signature-recording failure must
    not break the FSM (worst case: convergence is simply not detected and the
    legacy iter_cap behavior applies).
    """
    workspace = Path(workspace)
    try:
        md5 = compute_kernel_md5(workspace)
        entry = {
            "ts": _dt.datetime.now(_dt.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "spawn_index": spawn_index,
            "kernel_md5": md5,
        }
        path = workspace / _OPTIMIZER_SIG_LEDGER
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return md5
    except Exception:
        return None


def optimizer_kernel_converged(workspace: Path, *, min_repeat: int = 2) -> bool:
    """True iff the optimizer produced a BYTE-IDENTICAL kernel across the last
    `min_repeat` (default 2) consecutive spawns — i.e. it re-ran and changed
    nothing, so optimization has converged (0 net edits, nothing left to improve).

    Reads the deterministic signature ledger written by
    `record_optimizer_kernel_signature`. Fail-CLOSED: any error, fewer than
    `min_repeat` recorded signatures, a None md5 in the trailing window, or a
    differing md5 in the window → False. Never reports a false convergence, so an
    optimizer that is still making edits (kernel md5 changes each spawn) is never
    finalized early.
    """
    workspace = Path(workspace)
    path = workspace / _OPTIMIZER_SIG_LEDGER
    if not path.is_file():
        return False
    try:
        sigs = [
            json.loads(ln).get("kernel_md5")
            for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    except (json.JSONDecodeError, OSError):
        return False
    if len(sigs) < min_repeat:
        return False
    window = sigs[-min_repeat:]
    first = window[0]
    if first is None:
        return False
    return all(s == first for s in window)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workspace", required=True, type=Path,
                    help="op workspace dir (contains optimization_log.md + kernel/)")
    ap.add_argument("--print-kernel-md5", action="store_true",
                    help="print ONLY the current kernel md5 (the value to record "
                         "in an Opt entry's kernel_md5 field, AFTER keep/revert), "
                         "nothing else")
    args = ap.parse_args(argv)
    if args.print_kernel_md5:
        md5 = compute_kernel_md5(args.workspace)
        if md5 is None:
            sys.stderr.write("ERROR: no kernel sources under workspace/kernel/\n")
            return 2
        sys.stdout.write(md5 + "\n")
        return 0
    verdict = build_verdict(args.workspace)
    json.dump(verdict, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
