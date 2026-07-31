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
"""run_kw_graybox.py — hermetic standalone runner for the FA-A5 KW graybox.

The FA path uses repository template assembly by the standard worker, so the
graybox question is
now: *given the codified KB (templates K1-K4 + op_kernel/op_host .h assets) +
the arch22 (V220) algorithm source + the spec, with external undeclared target
trees unavailable, does the kw reproduce a correct A5 kernel?*

ISOLATED INPUT-PROVENANCE CONTRACT (load-bearing):
  - The kw spawns inside a bwrap mount-namespace (graybox_sandbox=true) that binds
    ONLY: the KB (src/skills/references/), the copied-in arch22 spec, the workspace,
    and the agent toolchain. cann/ + output/ are NEVER bound => the arch35 answer and
    the whole-port kernel are ABSENT from the sandbox.
  - The sandbox is no-network (graybox_sandbox.build_bwrap_cmd default share_net=False)
    => the kw cannot git-fetch / web-fetch the answer either.
  - Because no-net blocks the A5 SSH build, the kw AUTHORS only; build+verify on A5 is
    a separate outside-sandbox step run by an independent evaluator.
  - The copier prunes external arch35 trees before traversal. Production may
    separately stage provenance-tracked advisory artifacts; this standalone
    probe deliberately does not, and neither path may use them as truth.

USAGE
  python3 src/scripts/orchestrator/run_kw_graybox.py \
      --workspace workspace/fa_kw_graybox \
      [--op flash_attention_score] [--lane 1] [--target a5] [--timeout 7200]

After it runs: build the produced kernel on A5 and verify it with an independent
reference outside the authoring sandbox.
"""
import argparse
import fnmatch
import json
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Default arch22 source (the V220 algorithm = the legal port_a3 input).
_FA_CANN = Path("~/workspace/cann/ops-transformer/attention/flash_attention_score").expanduser()
# These undeclared generated artifacts are outside this isolated probe's inputs.
_FORBIDDEN_NAMES = ["arch35", "kernel", "generated_ascendc_kernel", "model_new_ascendc.py"]
_FORBIDDEN_GLOBS = ["*_wp.cpp", "*flash_attention*_wp.*", "*.cce"]


def _copy_arch22_tree(src_root: Path, dst_root: Path) -> list[str]:
    """Copy allowed source files while pruning target dirs before traversal."""
    copied: list[str] = []
    pending = [(src_root, dst_root)]
    while pending:
        source, target = pending.pop()
        target.mkdir(parents=True, exist_ok=True)
        for item in sorted(source.iterdir()):
            if item.name.lower() == "arch35":
                continue
            if item.is_symlink():
                raise SystemExit(f"INPUT-PROVENANCE VIOLATION — source symlink is not allowed: {item}")
            if any(fnmatch.fnmatch(item.name, pattern) for pattern in _FORBIDDEN_GLOBS):
                continue
            destination = target / item.name
            if item.is_dir():
                pending.append((item, destination))
            elif item.is_file():
                shutil.copy2(item, destination)
                copied.append(str(destination.relative_to(dst_root)))
    return sorted(copied)


def _find_target_dirs(root: Path, relative_to: Path) -> list[str]:
    """Find forbidden target directory entries without traversing them."""
    found: list[str] = []
    pending = [root]
    while pending:
        current = pending.pop()
        for item in sorted(current.iterdir()):
            if item.name.lower() == "arch35":
                found.append(str(item.relative_to(relative_to)))
                continue
            if item.is_symlink():
                found.append(str(item.relative_to(relative_to)))
            elif item.is_dir():
                pending.append(item)
    return found


def _assert_hermetic(ws: Path, arch22_src: Path) -> None:
    """Refuse to spawn if the workspace holds an answer artifact, or if the arch22
    copy still contains an arch35 tree (the answer)."""
    # no arch35 anywhere under the arch22 copy
    leaked = _find_target_dirs(arch22_src, ws)
    # no whole-port / answer artifacts at workspace top
    for name in _FORBIDDEN_NAMES:
        if name == "arch35":
            continue
        if (ws / name).exists():
            leaked.append(name)
    for pat in _FORBIDDEN_GLOBS:
        leaked += [str(p.relative_to(ws)) for p in ws.rglob(pat)]
    if leaked:
        raise SystemExit(
            f"INPUT-PROVENANCE VIOLATION — arch22 staging contains undeclared target artifacts: "
            f"{sorted(set(leaked))}. The kw must reproduce from KB + arch22 ALONE."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="hermetic standalone FA-A5 kw-graybox runner")
    ap.add_argument("--workspace", required=True, help="FRESH workspace dir (refuses if non-empty)")
    ap.add_argument("--op", default="flash_attention_score")
    ap.add_argument(
        "--arch22",
        default=str(_FA_CANN),
        help="arch22 (V220) source dir; target directories are excluded before traversal",
    )
    ap.add_argument("--spec", required=True, help="user-provided model.py specification")
    ap.add_argument("--lane", type=int, default=1)
    ap.add_argument("--target", default="a5")
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()

    src = Path(args.arch22).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"--arch22 not found: {src}")
    spec = Path(args.spec).expanduser().resolve()
    if not spec.is_file():
        raise SystemExit(f"--spec not found: {spec}")

    ws = Path(args.workspace).resolve()
    if ws.exists() and any(ws.iterdir()):
        raise SystemExit(f"--workspace {ws} non-empty; a hermetic run needs a FRESH dir.")
    ws.mkdir(parents=True, exist_ok=True)

    # 1) the spec (immutable reference input)
    shutil.copy2(spec, ws / "model.py")

    # 2) copy only allowed arch22 source; target dirs are pruned before traversal
    arch22_dst = ws / "arch22_src"
    arch22_dst.mkdir()
    for sub in ("op_host", "op_kernel"):
        s = src / sub
        if s.is_dir():
            _copy_arch22_tree(s, arch22_dst / sub)
    print(f"[input-provenance] copied arch22-only inputs -> {arch22_dst.relative_to(ws)}", flush=True)

    # 3) init durable state (port_a3 mode)
    import phase_o05
    rep = phase_o05.init_durable_state(
        ws, args.op, lane=args.lane, target=args.target, opgen_mode="port_a3_to_a5"
    )
    print(f"[init] {rep.summary}", flush=True)

    # 3b) op_classification.json — the UPSTREAM classify-phase output (a legal
    # pipeline-produced input, NOT the answer). Without it the kw_brief FA-class
    # template-assembly recipe block bails (_fa_class_template_assembly_block
    # requires op_classification.json) → the kw falls back to the generic brief
    # and escalates structural_rewrite_needed instead of assembling from templates.
    # FA's classification (FA-class / L4 / fused-softmax / attention) is the known,
    # deterministic output of the classify step. Discovered 2026-06-08 graybox run-1.
    cls = {
        "op": args.op,
        "op_class_tags": ["a3_to_a5_port", "FUSED_SOFTMAX", "ATTENTION", "fa_class", "CUBE_MIX"],
        "op_complexity": "L4",
        "kb_recommendations": [],
        "source": "kw_graybox_runner",
        "schema_version": 1,
    }
    (ws / "op_classification.json").write_text(json.dumps(cls, indent=2))
    print(f"[classify] op_classification.json written (tags={cls['op_class_tags']}) — recipe gate", flush=True)

    # 4) enable the graybox sandbox + point at the arch22 copy (NOT the cann tree)
    state_file = ws / ".opgen_state.json"
    state = json.loads(state_file.read_text())
    state["graybox_sandbox"] = True
    state["graybox_arch22_dir"] = str(arch22_dst)
    state["port_a3_source"] = str(arch22_dst)
    state_file.write_text(json.dumps(state, indent=2))
    print(f"[graybox] sandbox=ON arch22_dir={arch22_dst.relative_to(ws)} (external source/output UNBOUND)", flush=True)

    # 5) hermetic pre-assert (no answer leaks)
    _assert_hermetic(ws, arch22_dst)
    print(
        f"[hermetic] OK — workspace top: {sorted(p.name for p in ws.iterdir())}. Spawning kw in sandbox...", flush=True)

    # 6) spawn JUST the kw worker (with IL disabled, port_a3 FA routes here)
    import agent_dispatch
    result = agent_dispatch.spawn_for_state(
        args.op, ws, "await_worker", lane=args.lane, spawn_index=1, timeout_sec=args.timeout,
    )
    print(f"[kw] spawn returned (verdict={getattr(result, 'verdict', '?')}).", flush=True)

    # 7) what did the kw author?
    kdir = ws / "kernel"
    produced = sorted(str(p.relative_to(ws)) for p in kdir.rglob("*")) if kdir.exists() else []
    print(f"[kw-graybox] kernel/ files produced: {produced}", flush=True)
    if not produced:
        raise SystemExit("KW PRODUCED NO KERNEL — inspect the spawn output above.")
    print(
        "[kw-graybox] kernel authored from isolated declared inputs. NEXT (graybox VERDICT, outside sandbox, "
        "by independent verifier): build on A5 + verify vs the declared reference — correct from KB + arch22 ALONE?",
        flush=True,
    )


if __name__ == "__main__":
    main()
