# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Phase O2.5 A3-CANN reference — derivation + input-gen helpers (decomposed 2026-07-06).

Cohesive LEAF: pure-ish derivation of the port_a3 op's aclnn entry point,
input-gen source, OpDef signature parsing, edge-input generation from that
signature, cross-op dependency discovery, and A3 perf-log parsing. Imports only
stdlib (+ a call-time torch/case_gen inside generate_edge_inputs_from_signature).
NEVER imports from the phase_o25_a3_ref facade (unidirectional edge, no cycle).
"""
from __future__ import annotations
import logging

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from a3_ref_common import log
from source_op import resolve_logical_op_name


def derive_aclnn_entry(op_dir: Path) -> Optional[Path]:
    """Locate an aclnn entry-point .cpp for the op.

    Search order (first match wins):
    1. `examples/test_aclnn_<op>.cpp` — canonical (top_k_top_p_sample_v2,
       ctc_loss_v3, group_norm_silu_quant all use this).
    2. `pytorch/<op>.asc` — alternative format used by some ops (top_k +
       group_norm_silu_quant ship both in PR4778).
    3. `tests/ut/op_host/test_aclnn_<op>.cpp` — some ops put aclnn UT
       under tests/ instead of examples/.
    4. `examples/test_aclnn_<op>_*.cpp` (variant suffix) — multi-variant ops
       (grouped_matmul_swiglu_quant_v2 ships
       `test_aclnn_..._{a4w4,a8w4_weight_nd,a8w4_weight_nz,a8w8,a8w8_multi}.cpp`;
       grid_sample ships `test_aclnn_grid_sample{2_d,3_d}.cpp`). Pick the
       lexicographically-first variant — kw can later override via
       analysis.md note if a specific variant is needed for the test cases.
    5. `examples/test_aclnn_<op_without_v_suffix>*.cpp` — `_v<N>` versions
       sometimes share the base aclnn naming (e.g.
       `masked_select_v3` → `test_aclnn_masked_select.cpp`).

    Returns None if no entry found.
    """
    op_name = resolve_logical_op_name(op_dir)
    candidates = [
        op_dir / "examples" / f"test_aclnn_{op_name}.cpp",
        op_dir / "pytorch" / f"{op_name}.asc",
        op_dir / "tests" / "ut" / "op_host" / f"test_aclnn_{op_name}.cpp",
    ]
    for c in candidates:
        if c.is_file():
            return c

    # 2026-05-21 owner-direct fix: multi-variant ops + _vN-suffix ops were
    # silently missing aclnn entry detection, blocking 3 user-priority ports
    # (grouped_matmul_swiglu_quant_v2 has 5 variants; masked_select_v3's aclnn
    # is named without the _v3 suffix). Glob-fallback to find ANY matching
    # variant.
    examples_dir = op_dir / "examples"
    if examples_dir.is_dir():
        # (4) variant-suffix match
        variants = sorted(examples_dir.glob(f"test_aclnn_{op_name}_*.cpp"))
        if variants:
            return variants[0]
        # (5) try without _vN suffix (e.g. "masked_select_v3" → "masked_select")
        import re as _re
        base = _re.sub(r"_v\d+$", "", op_name)
        if base != op_name:
            without_v = examples_dir / f"test_aclnn_{base}.cpp"
            if without_v.is_file():
                return without_v
            # Combined: base + variant suffix
            base_variants = sorted(examples_dir.glob(f"test_aclnn_{base}_*.cpp"))
            if base_variants:
                return base_variants[0]

    # (6) task#24 content-fallback: some cube ops name their example after the
    # aclnn API, not the op dir (mat_mul_v3 ships test_aclnn_mm.cpp /
    # test_aclnn_matmul.cpp / test_aclnn_addmm*.cpp — NONE name-matches the dir).
    # All 5 name patterns above miss → false MISSING_ENTRY. Disambiguate by
    # matching the normalized op name against each candidate's PRIMARY aclnn API
    # symbol (aclnn<Name>GetWorkspaceSize). Auto-pick ONLY on a UNIQUE exact match
    # — 0 or >1 stays None so the caller surfaces the ambiguous candidates rather
    # than capturing the wrong op's reference (e.g. mat_mul_v3 → test_aclnn_matmul
    # via aclnnMatmul; aclnnMm/aclnnAddmm/aclnnMatmulWeightNz correctly excluded).
    if examples_dir.is_dir():
        import re as _re3

        def _norm(s: str) -> str:
            return _re3.sub(r"[^a-z0-9]", "", _re3.sub(r"_v\d+$", "", s.lower()))

        target = _norm(op_name)
        api_matches = []
        for cpp in sorted(examples_dir.glob("test_aclnn_*.cpp")):
            try:
                txt = cpp.read_text(errors="ignore")
            except OSError:
                continue
            apis = set(_re3.findall(r"aclnn([A-Z][A-Za-z0-9]*)GetWorkspaceSize", txt))
            if any(_norm(a) == target for a in apis):
                api_matches.append(cpp)
        if len(api_matches) == 1:
            return api_matches[0]

    return None


def derive_input_gen_source(op_dir: Path) -> Optional[Path]:
    """Locate the CANN UT input-generator script for the op.

    Canonical location: `tests/ut/op_kernel/<op>_data/gen_data.py`
    (gather_elements_v2, top_k_top_p_sample_v2, group_norm_silu_quant all
    ship this in PR4778; ctc_loss_v3 does NOT — its input_gen must be
    hand-authored or SCHEMA-derived).

    Returns None if not present.
    """
    op_name = resolve_logical_op_name(op_dir)
    candidate = op_dir / "tests" / "ut" / "op_kernel" / f"{op_name}_data" / "gen_data.py"
    if candidate.is_file():
        return candidate
    return None


# GE OpDef dtype token → torch dtype string.
_GE_DTYPE_TO_TORCH = {
    "DT_FLOAT": "float32",
    "DT_FLOAT16": "float16",
    "DT_BF16": "bfloat16",
    "DT_DOUBLE": "float64",
    "DT_INT8": "int8",
    "DT_UINT8": "uint8",
    "DT_INT16": "int16",
    "DT_INT32": "int32",
    "DT_INT64": "int64",
    "DT_BOOL": "bool",
}


def parse_op_def_signature(op_dir: Path) -> Optional[dict]:
    """Parse `op_host/<op>_def.cpp` GE OpDef into an input/attr signature.

    Every CANN op (aclnn-callable or pure-GE-IR like celu) ships an
    `op_host/<op>_def.cpp` with `this->Input("name").DataType({...})` and
    `this->Attr("name").AttrType(...).<Type>(<default>)` lines. This is the
    cleanest signature source that exists for the WHOLE port_a3 class — it does
    NOT depend on an aclnn entry being present.

    Returns a dict:
        {
          "op_name": <logical op name>,
          "inputs":  [{"name": str, "dtypes": [torch-dtype-name, ...],
                        "required": bool}, ...],
          "attrs":   [{"name": str, "dtype": "float|int|bool|str|list",
                        "default": <python value>}, ...],
          "outputs": [{"name": str, "dtypes": [...]}, ...],
        }
    Returns None if the def file is absent or no `Input(...)` is found.
    """
    op_name = resolve_logical_op_name(op_dir)
    def_file = op_dir / "op_host" / f"{op_name}_def.cpp"
    if not def_file.is_file():
        # Some ops register under a slightly different file stem; fall back to
        # the single *_def.cpp in op_host/ if exactly one exists.
        op_host = op_dir / "op_host"
        if op_host.is_dir():
            cands = sorted(op_host.glob("*_def.cpp"))
            if len(cands) == 1:
                def_file = cands[0]
            else:
                return None
        else:
            return None
    try:
        text = def_file.read_text()
    except Exception:
        return None

    # Strip line comments to avoid matching commented-out registrations.
    text = re.sub(r"//[^\n]*", "", text)

    sig: dict = {"op_name": op_name, "inputs": [], "attrs": [], "outputs": []}

    # The registration is a chained builder: this->Input("x").ParamType(...)
    # .DataType({DT_A, DT_B}).Format(...). We capture, per Input/Output, the
    # name and the FIRST DataType({...}) list that follows it (before the next
    # this->Input / this->Output / this->Attr).
    # Tokenize on `this->Input("..")` / `this->Output("..")` / `this->Attr("..")`
    # boundaries so each chunk holds exactly one builder chain.
    boundary_re = re.compile(
        r'this->(Input|Output|Attr)\s*\(\s*"([^"]+)"\s*\)')
    matches = list(boundary_re.finditer(text))
    for i, m in enumerate(matches):
        kind, name = m.group(1), m.group(2)
        chunk = text[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        if kind in ("Input", "Output"):
            dt_m = re.search(r"DataType\s*\(\s*\{([^}]*)\}", chunk)
            dtypes: list[str] = []
            if dt_m:
                for tok in re.findall(r"DT_[A-Z0-9_]+", dt_m.group(1)):
                    if tok in _GE_DTYPE_TO_TORCH:
                        dtypes.append(_GE_DTYPE_TO_TORCH[tok])
            if not dtypes:
                dtypes = ["float32"]
            required = "REQUIRED" in chunk or "OPTIONAL" not in chunk
            entry = {"name": name, "dtypes": dtypes, "required": required}
            if kind == "Input":
                sig["inputs"].append(entry)
            else:
                sig["outputs"].append(entry)
        else:  # Attr
            # Attr type + default: .Float(1.0) / .Int(0) / .Bool(false) / .String("x")
            # .ListInt({..}) / .ListFloat({..}). Capture first such call in chunk.
            dtype = None
            default: object = None
            fm = re.search(r"\.Float\s*\(\s*([-0-9.eEf]+)?\s*\)", chunk)
            im = re.search(r"\.Int\s*\(\s*([-0-9]+)?\s*\)", chunk)
            bm = re.search(r"\.Bool\s*\(\s*(true|false)?\s*\)", chunk)
            sm = re.search(r'\.String\s*\(\s*"([^"]*)"?\s*\)', chunk)
            if fm:
                dtype = "float"
                default = float(fm.group(1).rstrip("fF")) if fm.group(1) else 0.0
            elif im:
                dtype = "int"
                default = int(im.group(1)) if im.group(1) else 0
            elif bm:
                dtype = "bool"
                default = (bm.group(1) == "true")
            elif sm:
                dtype = "str"
                default = sm.group(1) if sm.group(1) is not None else ""
            else:
                # Unknown attr type — record name with no usable default; the
                # generator will skip attrs it can't materialize.
                dtype = "unknown"
                default = None
            sig["attrs"].append({"name": name, "dtype": dtype, "default": default})

    if not sig["inputs"]:
        return None
    return sig


def generate_edge_inputs_from_signature(
    op_dir: Path,
    workspace: Path,
    *,
    coverage_tier: str = "pilot",
) -> tuple[bool, str]:
    """Generate `workspace/edge_inputs.pt` for a port_a3 op from its OpDef
    signature, reusing the case_gen engine for shape/distribution coverage.

    The produced `edge_inputs.pt` is a LIST of kwargs-dicts (one per case),
    where each dict maps `Input` names → tensors and `Attr` names → scalar
    defaults. This is the shape the reference validators and
    `kw_brief`'s `edge_runner.py` consume via `Model(**case)`.

    Multi-dtype inputs (e.g. elu accepts bf16/fp16/fp32) collapse to the
    FIRST listed dtype for input generation (the highest-precision the op
    declares is usually listed first; we pick index 0 which is fp32 for
    celu/elu). Per-dtype coverage is a downstream verification concern, not
    an O2.5 reference-dataset concern.

    Returns (ok, message). ok=False if signature can't be parsed or case_gen
    fails — caller falls through (synth will then report edge_inputs missing,
    same as before this fix; no regression).
    """
    sig = parse_op_def_signature(op_dir)
    if sig is None:
        return False, f"could not parse op_host/{resolve_logical_op_name(op_dir)}_def.cpp signature"
    if not sig["inputs"]:
        return False, "op signature has no tensor inputs"

    try:
        import torch  # type: ignore
    except Exception as e:
        return False, f"torch import failed: {e!r}"

    # Make case_gen importable (reference_provider/ is not on path by default).
    import importlib
    _rp_dir = (Path(__file__).resolve().parents[2]
               / "scripts" / "reference_provider")
    added = False
    if str(_rp_dir) not in sys.path:
        sys.path.insert(0, str(_rp_dir))
        added = True
    try:
        case_gen = importlib.import_module("case_gen")
    except Exception as e:
        return False, f"case_gen import failed: {e!r}"
    finally:
        if added:
            try:
                sys.path.remove(str(_rp_dir))
            except ValueError:
                pass

    # Build a case_gen SCHEMA from the parsed signature. One tensor per Input;
    # scalar Attrs become Model.forward kwargs (passed as-is from defaults).
    first_dtype_name = sig["inputs"][0]["dtypes"][0]
    try:
        gen_dtype = getattr(torch, first_dtype_name)
    except AttributeError:
        gen_dtype = torch.float32

    schema = {
        "op_name": sig["op_name"],
        "formula": f"{sig['op_name']}(<inputs>) [auto-derived from op_def]",
        "tensor_inputs": [{"name": inp["name"], "role": "operand"}
                          for inp in sig["inputs"]],
        "scalar_inputs": [],   # attrs handled as fixed kwargs below, not swept
        "tensor_output": (sig["outputs"][0]["name"] if sig["outputs"] else "y"),
        "rank": 1,
    }

    try:
        raw_cases = case_gen.generate_cases(
            schema, coverage_tier=coverage_tier, dtype=gen_dtype)
    except Exception as e:
        return False, f"case_gen.generate_cases failed: {e!r}"
    if not raw_cases:
        return False, "case_gen produced 0 cases"

    # Fixed attr kwargs (defaults from the OpDef). Skip attrs with no usable
    # default (dtype 'unknown' / default None).
    attr_kwargs: dict = {}
    for a in sig["attrs"]:
        if a["dtype"] == "unknown" or a["default"] is None:
            continue
        attr_kwargs[a["name"]] = a["default"]

    # Convert case_gen cases ({idx,name,shape,inputs:{n:tensor}}) into the
    # kwargs-dict shape Model(**case) expects: merge input tensors + attrs.
    edge_cases: list[dict] = []
    for c in raw_cases:
        case_kwargs: dict = {}
        for inp_name, tensor in c.get("inputs", {}).items():
            case_kwargs[inp_name] = tensor
        case_kwargs.update(attr_kwargs)
        edge_cases.append(case_kwargs)

    edge_path = workspace / "edge_inputs.pt"
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(edge_cases, edge_path)
    except Exception as e:
        return False, f"torch.save(edge_inputs.pt) failed: {e!r}"

    # Drop a provenance marker so downstream agents know these inputs were
    # auto-derived from the OpDef signature (not from a UT gen_data.py).
    try:
        (workspace / ".edge_inputs_provenance.json").write_text(json.dumps({
            "source": "op_def_signature_via_case_gen",
            "op_name": sig["op_name"],
            "coverage_tier": coverage_tier,
            "n_cases": len(edge_cases),
            "inputs": [{"name": i["name"], "dtype_used": first_dtype_name}
                       for i in sig["inputs"]],
            "attr_kwargs": {k: v for k, v in attr_kwargs.items()},
            "generated_ts": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
    except Exception as error:
        logging.getLogger(__name__).debug(
            "Recoverable operation failed.", exc_info=error
        )

    return True, (f"generated {len(edge_cases)} edge cases for "
                  f"{sig['op_name']} from op_def signature "
                  f"(inputs={[i['name'] for i in sig['inputs']]}, "
                  f"attrs={list(attr_kwargs.keys())}, dtype={first_dtype_name})")


def ensure_edge_inputs(op_dir: Path, workspace: Path) -> tuple[bool, str]:
    """Ensure `workspace/edge_inputs.pt` exists for a port_a3 op.

    Order of preference (cleanest existing mechanism first):
      1. Already present (a prior run / hand-authored fixture) → no-op.
      2. The op ships `input_gen.py` in the workspace → it owns generation
         (the aog-input-gen-builder / a3-author path); leave it alone.
      3. Else synthesize from the OpDef signature via case_gen.

    Returns (present, message). present=True iff edge_inputs.pt exists after
    the call. Never raises — failure to synthesize returns (False, reason) and
    the caller proceeds exactly as before this fix (no regression).
    """
    edge_path = workspace / "edge_inputs.pt"
    if edge_path.is_file():
        return True, "edge_inputs.pt already present"
    # If the workspace has an input_gen.py authored already, the a3-author /
    # input-gen-builder mechanism owns generation; don't pre-empt it.
    if (workspace / "input_gen.py").is_file():
        return False, "input_gen.py present — leaving edge_inputs.pt to that path"
    ok, msg = generate_edge_inputs_from_signature(op_dir, workspace)
    if ok:
        log.info(f"[phase_o25_a3_ref] ensure_edge_inputs: {msg}")
    else:
        log.info(f"[phase_o25_a3_ref] ensure_edge_inputs: synthesis skipped — {msg}")
    return ok, msg


def derive_op_dependencies(op_dir: Path) -> list[str]:
    """Parse `op_host/CMakeLists.txt` for `DEPENDENCIES <peer_op>` lines.

    Critical for cross-op router modification (gap #1 from REPORT — the
    v3-via-v2-aclnn pattern in ctc_loss_v3). If this returns a non-empty
    list, kw_brief (W5) MUST surface those peer ops' op_api/<peer>.cpp
    paths as candidates for router patch.

    Returns sorted list of peer op names; empty if no DEPENDENCIES line
    found. Comment lines and dependencies that aren't a single bare name
    are ignored.
    """
    cmake = op_dir / "op_host" / "CMakeLists.txt"
    if not cmake.is_file():
        return []
    text = cmake.read_text()
    deps: set[str] = set()
    # Match `DEPENDENCIES <name>` (with optional indent / trailing args).
    # CANN convention: DEPENDENCIES is followed by space-separated peer-op
    # names; we capture all of them on the same line.
    for line in text.splitlines():
        # Strip comments
        if line.lstrip().startswith("#"):
            continue
        m = re.search(r"\bDEPENDENCIES\b\s+(.+?)\s*(?:\)|$)", line)
        if not m:
            continue
        for tok in m.group(1).split():
            if re.fullmatch(r"[a-z_][a-z0-9_]*", tok):
                deps.add(tok)
    return sorted(deps)


def parse_a3_perf_log(perf_log_text: str) -> dict[str, float]:
    """Pure parser: extract per-case timing (ms) from an aclnn UT runner log.

    The aclnn UT binaries print timing in a few different formats depending
    on the op. We support two common shapes:
        "case_<id> elapsed_ms=<float>"
        "[case <id>] device_time=<float>ms"

    Returns dict mapping case_id (str) → time_ms (float). Empty dict if
    no matches.
    """
    times: dict[str, float] = {}
    for line in perf_log_text.splitlines():
        m = re.match(r"\s*case_(\S+)\s+elapsed_ms=([\d.]+)", line)
        if m:
            times[m.group(1)] = float(m.group(2))
            continue
        m = re.match(r"\s*\[\s*case\s+(\S+?)\s*\]\s+device_time=([\d.]+)\s*ms", line)
        if m:
            times[m.group(1)] = float(m.group(2))
    return times
