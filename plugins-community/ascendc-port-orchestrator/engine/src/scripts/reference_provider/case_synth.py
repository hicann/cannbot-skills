#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""case_synth — tensor + case synthesis (builders, generators, expanders), extracted from
case_gen.py (behavior-neutral, 2026-07-05). Depends on case_schema for shape/dtype helpers."""
from __future__ import annotations
import math
import struct
from typing import Any, NamedTuple

import torch

from case_schema import _base_shape_for_rank, _tensor_dtype_for, _tensor_shape_for

_DEFAULT_MAX_CASE_TENSOR_BYTES = 100 * 1024 * 1024


class _ResolvedTensorSpec(NamedTuple):
    """One immutable shape/dtype plan shared by checking and allocation."""

    spec: dict
    name: str
    dtype: torch.dtype
    shapes: tuple[tuple[int, ...], ...]


def _list_item_shapes(
    tinput: dict, base_shape: list[int], scalars: dict[str, Any]
) -> list[list[int]]:
    """Resolve list-of-tensors item shapes without allocating tensor storage."""
    import inspect

    name = tinput["name"]
    list_length_plan = tinput.get("list_length_plan")
    if not list_length_plan:
        raise ValueError(
            f"tensor_input {name!r} declared kind=list_of_tensors but "
            "missing list_length_plan (need non-empty list[int])"
        )
    length = int(scalars.get(f"__list_length__{name}", list_length_plan[0]))
    if length <= 0:
        raise ValueError(
            f"tensor_input {name!r}: per-case list_length must be > 0, got {length}"
        )
    derive = tinput.get("per_item_shape_derive")
    shapes: list[list[int]] = []
    for item_idx in range(length):
        if derive is None:
            shape = list(base_shape)
        else:
            try:
                n_params = len(inspect.signature(derive).parameters)
            except (TypeError, ValueError):
                n_params = 3
            shape = (
                derive(base_shape, item_idx, length, scalars)
                if n_params >= 4
                else derive(base_shape, item_idx, length)
            )
            if not isinstance(shape, (list, tuple)) or not all(
                isinstance(dim, int) for dim in shape
            ):
                raise ValueError(
                    f"per_item_shape_derive for {name!r} item {item_idx}/{length} "
                    f"returned {shape!r}; expected list[int]"
                )
            shape = list(shape)
        shapes.append(shape)
    return shapes


def _resolve_case_tensor_plan(
    tensor_inputs: list[dict],
    base_shape: list[int],
    global_dtype: torch.dtype,
    scalars: dict[str, Any] | None = None,
) -> tuple[_ResolvedTensorSpec, ...]:
    """Resolve every tensor shape and dtype exactly once for one case."""
    resolved_scalars = {} if scalars is None else scalars
    plan: list[_ResolvedTensorSpec] = []
    for tinput in tensor_inputs:
        name = tinput["name"]
        dtype = _tensor_dtype_for(tinput, global_dtype)
        shapes = (
            _list_item_shapes(tinput, base_shape, resolved_scalars)
            if tinput.get("kind") == "list_of_tensors"
            else [_tensor_shape_for(tinput, base_shape, resolved_scalars)]
        )
        frozen_shapes: list[tuple[int, ...]] = []
        for shape in shapes:
            if not isinstance(shape, (list, tuple)) or any(
                not isinstance(dim, int) or dim < 0 for dim in shape
            ):
                raise ValueError(
                    f"tensor_input {name!r} resolved invalid shape {shape!r}; "
                    "dimensions must be non-negative ints"
                )
            frozen_shapes.append(tuple(shape))
        plan.append(
            _ResolvedTensorSpec(
                spec=tinput,
                name=name,
                dtype=dtype,
                shapes=tuple(frozen_shapes),
            )
        )
    return tuple(plan)


def _estimate_resolved_plan_bytes(
    resolved_plan: tuple[_ResolvedTensorSpec, ...],
) -> tuple[int, list[str]]:
    """Estimate payload bytes from a previously resolved immutable plan."""
    total = 0
    details: list[str] = []
    for item in resolved_plan:
        element_size = torch.empty((), dtype=item.dtype).element_size()
        input_bytes = sum(int(math.prod(shape)) * element_size for shape in item.shapes)
        total += input_bytes
        details.append(
            f"{item.name}:shapes={list(item.shapes)},dtype={item.dtype},bytes={input_bytes}"
        )
    return total, details


def _estimate_case_tensor_bytes(
    tensor_inputs: list[dict],
    base_shape: list[int],
    global_dtype: torch.dtype,
    scalars: dict[str, Any] | None = None,
) -> tuple[int, list[str]]:
    """Estimate all tensor storage for one case before any large allocation."""
    return _estimate_resolved_plan_bytes(
        _resolve_case_tensor_plan(tensor_inputs, base_shape, global_dtype, scalars)
    )


def _guard_case_tensor_budget(
    tensor_inputs: list[dict],
    base_shape: list[int],
    global_dtype: torch.dtype,
    scalars: dict[str, Any] | None,
    *,
    max_case_tensor_bytes: int | None,
    case_label: str,
    resolved_plan: tuple[_ResolvedTensorSpec, ...] | None = None,
) -> tuple[_ResolvedTensorSpec, ...]:
    """Fail before allocation when one case exceeds its configured budget."""
    if max_case_tensor_bytes is not None and max_case_tensor_bytes <= 0:
        raise ValueError("max_case_tensor_bytes must be a positive integer")
    plan = (
        resolved_plan
        if resolved_plan is not None
        else _resolve_case_tensor_plan(tensor_inputs, base_shape, global_dtype, scalars)
    )
    if max_case_tensor_bytes is None:
        return plan
    estimated, details = _estimate_resolved_plan_bytes(plan)
    if estimated > max_case_tensor_bytes:
        raise ValueError(
            f"case tensor allocation exceeds budget before allocation: case={case_label!r}, "
            f"base_shape={base_shape!r}, estimated_bytes={estimated}, "
            f"limit_bytes={max_case_tensor_bytes}, inputs=[{'; '.join(details)}]. "
            "Use a source-supported representative rank and smaller representative "
            "shapes, or explicitly justify a larger schema max_case_tensor_bytes."
        )
    return plan


def _mk_tensor(gen_fn, shape: list[int], dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Generate a 1D tensor of correct product-of-shape size, then reshape."""
    n = int(math.prod(shape))
    flat = gen_fn(n, dtype, seed)
    return flat.reshape(*shape)


def _build_inputs(tensor_inputs: list[dict], base_shape: list[int],
                   default_gen_fn, global_dtype: torch.dtype, seed_base: int,
                   force_gen_fn_per_tensor: dict | None = None,
                   scalars: dict[str, Any] | None = None,
                   max_case_tensor_bytes: int | None = _DEFAULT_MAX_CASE_TENSOR_BYTES,
                   case_label: str = "unknown",
                   resolved_plan: tuple[_ResolvedTensorSpec, ...] | None = None,
                   ) -> dict[str, torch.Tensor]:
    """Construct all tensor inputs for one case, honoring per-tensor shape_derive + dtype.

    force_gen_fn_per_tensor: optional dict {tensor_name: callable(n, dtype, seed) -> tensor}
        Used by special bands (NaN sprinkle, cancellation) that want a specific tensor built
        differently from the band's default. HOWEVER: tensors declaring `invariant` in their
        schema entry (permutation / positive / non_negative / index_range:<N>) are exempt
        from forced override — those invariants must hold across all bands to keep cases
        semantically valid (fixes handover T12 "correlated_cancel breaks permutation").

    scalars: current per-case scalar values (for 2-arg shape_derive / invariant="index_range:<scalar>").
    """
    plan = (
        resolved_plan
        if resolved_plan is not None
        else _resolve_case_tensor_plan(tensor_inputs, base_shape, global_dtype, scalars)
    )
    _guard_case_tensor_budget(
        tensor_inputs,
        base_shape,
        global_dtype,
        scalars,
        max_case_tensor_bytes=max_case_tensor_bytes,
        case_label=case_label,
        resolved_plan=plan,
    )
    inputs: dict[str, torch.Tensor] = {}
    for i, (tinput, item_plan) in enumerate(zip(tensor_inputs, plan)):
        name = tinput["name"]
        # V1.7 (2026-05-21): list_of_tensors primitive. Build a Python list
        # of N tensors per scalars["__list_length__<name>"] (set by the
        # post-expansion pass `_expand_list_of_tensors_lengths`). Each item
        # may use per_item_shape_derive(base_shape, item_idx, N, scalars).
        if tinput.get("kind") == "list_of_tensors":
            inputs[name] = _build_list_of_tensors_item(
                tinput, base_shape, default_gen_fn, global_dtype,
                seed_base + i * 17, scalars or {},
                resolved_shapes=item_plan.shapes,
                resolved_dtype=item_plan.dtype,
            )
            continue
        shape = list(item_plan.shapes[0])
        dtype = item_plan.dtype
        seed = seed_base + i * 17
        invariant = tinput.get("invariant")

        # Resolve generator: forced (unless an invariant prohibits it), then
        # per-tensor value_gen, invariant auto-gen, and finally the default.
        forced = force_gen_fn_per_tensor and name in force_gen_fn_per_tensor
        if forced and not invariant:
            gen = force_gen_fn_per_tensor[name]
        elif "value_gen" in tinput and callable(tinput["value_gen"]):
            gen = tinput["value_gen"]
        elif invariant:
            gen = _mk_invariant_gen(invariant, tinput, scalars or {}, base_shape)
        else:
            gen = default_gen_fn

        # Non-float dtypes may not accept randn/rand from torch_generator; use int variant
        # (skip when invariant gen already handles dtype)
        if dtype in (torch.int8, torch.int32, torch.int64) and gen is default_gen_fn:
            gen = _mk_int_uniform_gen(tinput.get("int_range", (0, 1000)))
        if dtype == torch.bool and gen is default_gen_fn:
            gen = _mk_bool_uniform_gen()

        inputs[name] = _mk_tensor(gen, shape, dtype, seed)
    return inputs


def _build_list_of_tensors_item(
    tinput: dict, base_shape: list[int], default_gen_fn,
    global_dtype: torch.dtype, seed_base: int, scalars: dict[str, Any],
    *,
    resolved_shapes: tuple[tuple[int, ...], ...] | None = None,
    resolved_dtype: torch.dtype | None = None,
) -> list[torch.Tensor]:
    """V1.7 (2026-05-21): build the per-case list of tensors for a
    `kind="list_of_tensors"` input. Used by torch.cat / torch.stack / similar
    variadic-list ops.

    The current list length is read from scalars[`__list_length__<name>`],
    set by `_expand_list_of_tensors_lengths` during the post-expansion pass.
    If absent (single-length case or non-expanded usage), falls back to the
    first entry of `tinput["list_length_plan"]`.

    Each list item uses:
      - `per_item_shape_derive(base_shape, item_idx, list_length[, scalars])`
        if present, else `base_shape` (uniform list).
      - The tensor's global dtype + the default generator (same as a normal
        single-tensor input). Per-item dtype/invariant overrides are NOT
        supported in V1.7 (future work; YAGNI until a real op needs it).

    Returns: list[torch.Tensor] of length N.
    """
    name = tinput["name"]
    list_length_plan = tinput.get("list_length_plan")
    if not list_length_plan:
        raise ValueError(
            f"tensor_input {name!r} declared kind=list_of_tensors but "
            f"missing list_length_plan (need non-empty list[int])"
        )
    # Per-case length (post-expansion sets the per-case value; otherwise default
    # to first plan value so a single-length-test sweep still works).
    n = int(scalars.get(f"__list_length__{name}", list_length_plan[0]))
    if n <= 0:
        raise ValueError(
            f"tensor_input {name!r}: per-case list_length must be > 0, got {n}"
        )

    shapes = resolved_shapes
    if shapes is None:
        shapes = tuple(tuple(shape) for shape in _list_item_shapes(tinput, base_shape, scalars))
    if len(shapes) != n:
        raise ValueError(
            f"resolved list shape plan for {name!r} has {len(shapes)} items; expected {n}"
        )
    dtype = resolved_dtype or _tensor_dtype_for(tinput, global_dtype)

    items: list[torch.Tensor] = []
    for item_idx, frozen_shape in enumerate(shapes):
        shape = list(frozen_shape)
        gen = default_gen_fn
        if dtype in (torch.int8, torch.int32, torch.int64) and gen is default_gen_fn:
            gen = _mk_int_uniform_gen(tinput.get("int_range", (0, 1000)))
        if dtype == torch.bool and gen is default_gen_fn:
            gen = _mk_bool_uniform_gen()
        # Per-item seed: derive deterministically from seed_base + item_idx.
        items.append(_mk_tensor(gen, shape, dtype, seed_base + item_idx * 23))
    return items


def _mk_invariant_gen(invariant: str, tinput: dict, scalars: dict[str, Any],
                       base_shape: list[int] | None = None):
    """Generator factory for tensors with declared invariants.

    invariant ∈ {
        "permutation", "positive", "non_negative",
        "index_range:<spec>",
        "index_range_dim:<scalar>",      # NEW V3.7.5 (2026-05-02): scatter/gather/index ops
    }
    where <spec> is a scalar input name (resolved via scalars dict) or integer literal.

    `index_range_dim:<scalar>` resolves the upper bound dynamically per-case as
    `base_shape[normalize_dim(scalars[<scalar>], rank)]`. This is the canonical
    invariant for ops where index values must be valid offsets along a runtime-
    selected axis (torch.scatter, torch.gather, torch.index_select, torch.scatter_add,
    torch.gather along configurable dim). Without this, scalar-shaped ops like
    21_Scatter (where dim attr varies per-case and index.shape[dim] != x.shape[dim])
    cannot express "indices must be in [0, x.shape[dim])" cleanly. base_shape is
    the per-case base_shape passed by _build_inputs.
    """
    if invariant == "permutation":
        def _fn(n, dt, seed):
            g = torch.Generator().manual_seed(seed + 31337)
            return torch.randperm(n, generator=g).to(dt)
        return _fn
    if invariant == "non_negative":
        lo, hi = tinput.get("int_range", (0, 1000))

        def _fn(n, dt, seed):
            g = torch.Generator().manual_seed(seed)
            return torch.randint(max(0, lo), max(1, hi), (n,), generator=g, dtype=dt)
        return _fn
    if invariant == "positive":
        lo, hi = tinput.get("int_range", (1, 1000))
        lo = max(1, lo)

        def _fn(n, dt, seed):
            g = torch.Generator().manual_seed(seed)
            return torch.randint(lo, max(lo + 1, hi), (n,), generator=g, dtype=dt)
        return _fn
    if invariant.startswith("index_range_dim:"):
        # V3.7.5 (2026-05-02): scalar-shaped ops where index.shape[dim] != x.shape[dim]
        # but index values must lie in [0, x.shape[dim]). Resolves upper bound from
        # per-case base_shape + scalar value.
        scalar_name = invariant.split(":", 1)[1]
        if base_shape is None or len(base_shape) == 0:
            raise ValueError(
                f"invariant={invariant!r} requires base_shape but caller passed None/empty. "
                f"_build_inputs must pass base_shape through to _mk_invariant_gen."
            )
        rank = len(base_shape)
        dim_raw = scalars.get(scalar_name)
        if dim_raw is None:
            raise ValueError(
                f"invariant={invariant!r} refers to scalar {scalar_name!r} which is "
                f"absent from per-case scalars {sorted(scalars.keys())!r}. "
                f"Make sure scalar {scalar_name!r} is declared in SCHEMA's scalar_inputs."
            )
        # Normalize negative dim to [0, rank-1]
        dim_norm = int(dim_raw) if int(dim_raw) >= 0 else rank + int(dim_raw)
        if dim_norm < 0 or dim_norm >= rank:
            raise ValueError(
                f"invariant={invariant!r}: scalar {scalar_name!r}={dim_raw!r} "
                f"normalizes to {dim_norm} which is out of bounds for rank={rank}"
            )
        upper = int(base_shape[dim_norm])
        if upper <= 0:
            raise ValueError(
                f"invariant={invariant!r}: base_shape[{dim_norm}]={upper} must be positive"
            )
        allow_dup = bool(tinput.get("allow_dup_indices", True))  # scatter typically allows dup

        def _fn(n, dt, seed):
            g = torch.Generator().manual_seed(seed)
            if not allow_dup and n <= upper:
                return _unique_indices_without_large_permutation(n, upper, dt, seed, g)
            return torch.randint(0, upper, (n,), generator=g, dtype=dt)
        return _fn
    if invariant.startswith("index_range:"):
        # V3.2 (2026-04-24): unique-by-default when feasible (n <= upper).
        # Avoids generating duplicate indices that would race the reference
        # CANN scatter (op#12 cold-start lesson). Opt-in to duplicates via
        # `tensor_inputs[i]["allow_dup_indices"] = True` for ops that
        # specifically test scatter race semantics.
        spec = invariant.split(":", 1)[1]
        # Resolve: integer literal or scalar input name
        try:
            upper = int(spec)
        except ValueError as exc:
            upper = int(scalars.get(spec, 0))
            if upper <= 0:
                raise ValueError(
                    f"invariant={invariant!r} refers to scalar {spec!r} which is "
                    f"{scalars.get(spec)!r} — must be positive integer. "
                    f"Make sure scalar {spec!r} has a `derive` or `default` that resolves "
                    f"before this tensor is built."
                ) from exc
        allow_dup = bool(tinput.get("allow_dup_indices", False))

        def _fn(n, dt, seed):
            g = torch.Generator().manual_seed(seed)
            if not allow_dup and n <= upper:
                # Unique sample: randperm + slice
                return _unique_indices_without_large_permutation(n, upper, dt, seed, g)
            # n > upper or duplicates explicitly allowed: fall back to randint
            return torch.randint(0, upper, (n,), generator=g, dtype=dt)
        return _fn
    raise ValueError(f"unknown tensor invariant {invariant!r}")


def _unique_indices_without_large_permutation(
    n: int,
    upper: int,
    dtype: torch.dtype,
    seed: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return unique indices while bounding temporary storage by O(n)."""
    if upper <= max(1_000_000, 4 * n):
        return torch.randperm(upper, generator=generator)[:n].to(dtype)
    if upper > torch.iinfo(torch.int64).max:
        raise ValueError(f"index upper bound {upper} exceeds int64 capacity")
    if n == 0:
        return torch.empty((0,), dtype=dtype)
    start = seed % upper
    step = 2 * (seed % 1024) + 1
    while math.gcd(step, upper) != 1:
        step = (step + 1) % upper or 1
    values = (torch.arange(n, dtype=torch.int64) * step + start) % upper
    return values.to(dtype)


def _mk_int_uniform_gen(int_range: tuple[int, int]):
    lo, hi = int_range

    def _f(n, dt, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randint(lo, hi, (n,), generator=g, dtype=dt)
    return _f


def _mk_bool_uniform_gen():
    def _f(n, _dt, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randint(0, 2, (n,), generator=g, dtype=torch.int8).to(torch.bool)
    return _f


def _scalar_only_cases(schema: dict[str, Any], coverage_tier: str) -> list[dict]:
    """Scalar-only schema case generation (tensor_inputs empty).

    Coverage = baseline (all defaults) + per-scalar probe variants + at
    sign_off/production tier, also emit 1-2 "cross-probe" combinations
    (two scalars varied simultaneously) for interaction coverage.

    Each case's `inputs` dict is purely scalars. `shape` is omitted
    (set to [] — consumers should treat as "shape derived from scalars
    per op semantics"; edge_runner computes it at Model.forward time).
    """
    scalar_inputs = schema.get("scalar_inputs", [])

    def _coerce_scalar(val, dtype_str):
        if dtype_str in ("int", "int32", "int64"):
            return int(val)
        if dtype_str == "bool":
            return bool(val)
        if dtype_str in ("str", "string"):
            # Pass-through for enum/string scalars (e.g. rotary_mode: "half"|"interleave").
            # Added 2026-04-22 for op#1 RotaryMul where mode is a string enum.
            return str(val)
        return float(val)

    defaults = {s["name"]: _coerce_scalar(s.get("default", 0), s.get("dtype", "float"))
                for s in scalar_inputs}

    cases = []
    idx = 0

    # Baseline: all defaults.
    cases.append({
        "idx": idx, "name": "scalar_only_baseline",
        "shape": [], "inputs": dict(defaults),
        "meta": {"band": "S_baseline", "note": "all defaults"},
    })
    idx += 1

    # Per-scalar probe variants (like Band C in tensor mode).
    # 2026-05-18 Gap #3 unblock: probe_values may be a CALLABLE
    # (lambda rank: [...] or lambda base_shape: [...]) so the probe
    # set can be rank/shape-dependent — needed for ops like 9_TopK
    # whose valid `dim` range is {-rank..rank-1}.
    rank_for_probe = schema.get("rank")

    def _resolve_probe_values(s_spec):
        pv = s_spec.get("probe_values", [])
        if not callable(pv):
            return list(pv)
        import inspect as _inspect
        try:
            sig = _inspect.signature(pv)
            param_names = [p.name for p in sig.parameters.values()]
        except (TypeError, ValueError):
            param_names = []
        # Pass rank by default; if caller wants base_shape, use that
        # parameter name explicitly.
        if param_names and param_names[0] in ("base_shape", "shape"):
            return list(pv(_base_shape_for_rank(rank_for_probe or 1)))
        return list(pv(rank_for_probe if rank_for_probe is not None else 1))

    for s_spec in scalar_inputs:
        name = s_spec["name"]
        dtype_str = s_spec.get("dtype", "float")
        for probe_v in _resolve_probe_values(s_spec):
            if probe_v == s_spec.get("default"):
                continue
            inputs = dict(defaults)
            inputs[name] = _coerce_scalar(probe_v, dtype_str)
            cases.append({
                "idx": idx, "name": f"scalar_{name}_{probe_v}",
                "shape": [], "inputs": inputs,
                "meta": {"band": "S_probe", "scalar": name, "value": probe_v},
            })
            idx += 1

    # Sign_off + production: add cross-probe pairs (two scalars varied simultaneously).
    # Targets interaction bugs (e.g. past_kv_length > 0 AND sliding_window < seq_length).
    if coverage_tier in ("sign_off", "production"):
        probed = [s for s in scalar_inputs if s.get("probe_values")]
        if len(probed) >= 2:
            # Take first probe_value from each of the first two probed scalars.
            s1, s2 = probed[0], probed[1]
            s1_values = _resolve_probe_values(s1)
            s2_values = _resolve_probe_values(s2)
            v1 = next((v for v in s1_values if v != s1.get("default")), None)
            v2 = next((v for v in s2_values if v != s2.get("default")), None)
            if v1 is not None and v2 is not None:
                inputs = dict(defaults)
                inputs[s1["name"]] = _coerce_scalar(v1, s1.get("dtype", "float"))
                inputs[s2["name"]] = _coerce_scalar(v2, s2.get("dtype", "float"))
                cases.append({
                    "idx": idx,
                    "name": f"scalar_cross_{s1['name']}_{v1}_{s2['name']}_{v2}",
                    "shape": [], "inputs": inputs,
                    "meta": {"band": "S_cross", "scalars": [s1["name"], s2["name"]],
                             "values": [v1, v2], "note": "interaction probe"},
                })
                idx += 1

    return cases


def _expand_list_of_tensors_lengths(
    cases: list[dict], list_specs: list[dict], tensor_inputs: list[dict],
    schema: dict[str, Any], *, coverage_tier: str, global_dtype: torch.dtype,
    max_case_tensor_bytes: int | None = _DEFAULT_MAX_CASE_TENSOR_BYTES,
) -> list[dict]:
    """V1.7 (2026-05-21): fork each case over the cartesian product of
    `list_length_plan` values across all `kind=list_of_tensors` inputs.

    For each fork (per-case x per-list-length-combo), re-materialize the
    list-of-tensors inputs at the new N (using the per-case base_shape from
    `case["shape"]` + scalars + the spec's `per_item_shape_derive`). Non-list
    inputs and scalars are preserved verbatim from the original case.

    Case count multiplier = Π_i len(spec_i.list_length_plan). For a single
    list with plan=[2,3,4] over 50 base cases → 150 cases.

    Mutates: new case `name` gets `_N<value>` suffix per list spec; `meta`
    records the per-list length value at `list_length_<name>` key.
    """
    import itertools
    if not list_specs:
        return cases

    # Cartesian product of all plans (handles multi-list ops if any exist;
    # 13_Cat has only one but future ops may have several).
    plans = [spec["list_length_plan"] for spec in list_specs]
    spec_names = [spec["name"] for spec in list_specs]
    combos = list(itertools.product(*plans))

    expanded: list[dict] = []
    next_idx = 0
    for c in cases:
        base_shape = c["shape"]
        # Scalars present in the case (everything in inputs that's not a Tensor
        # and not a list-of-Tensors and not None).
        scalars: dict[str, Any] = {
            k: v for k, v in c["inputs"].items()
            if not isinstance(v, torch.Tensor) and not isinstance(v, list) and v is not None
        }
        for combo in combos:
            # Inject per-fork list lengths into scalars dict so the per-item
            # shape derive callable + builder helper can read them.
            fork_scalars = dict(scalars)
            for name, n in zip(spec_names, combo):
                fork_scalars[f"__list_length__{name}"] = int(n)
            resolved_plan = _resolve_case_tensor_plan(
                tensor_inputs, base_shape, global_dtype, fork_scalars
            )
            _guard_case_tensor_budget(
                tensor_inputs,
                base_shape,
                global_dtype,
                fork_scalars,
                max_case_tensor_bytes=max_case_tensor_bytes,
                case_label=f"{c['name']}:list_lengths={combo}",
                resolved_plan=resolved_plan,
            )
            new_inputs = dict(c["inputs"])
            for spec, n in zip(list_specs, combo):
                # Re-materialize THIS list spec at the new length.
                # Seed deterministically from case idx + spec name + N so
                # the same (case, spec, N) triple always produces the same
                # tensors across runs.
                seed_base = (c["idx"] + 9001) * 31 + hash(spec["name"]) % 997 + int(n) * 7
                # Default generator: standard randn — matches single-tensor path.

                def _randn(num, dt, seed):
                    g = torch.Generator().manual_seed(seed)
                    return torch.randn(num, generator=g, dtype=dt)
                item_plan = next(
                    plan_item for plan_item in resolved_plan
                    if plan_item.name == spec["name"]
                )
                new_inputs[spec["name"]] = _build_list_of_tensors_item(
                    spec, base_shape, _randn, global_dtype, seed_base,
                    fork_scalars,
                    resolved_shapes=item_plan.shapes,
                    resolved_dtype=item_plan.dtype,
                )
            new_meta = dict(c.get("meta", {}))
            for name, n in zip(spec_names, combo):
                new_meta[f"list_length_{name}"] = int(n)
            length_tags = "_".join(
                f"{name}N{n}" for name, n in zip(spec_names, combo)
            )
            expanded.append({
                "idx": next_idx,
                "name": c["name"] + "_" + length_tags,
                "shape": c["shape"],
                "inputs": new_inputs,
                "meta": new_meta,
            })
            next_idx += 1
    return expanded


def _expand_optional_tensor_presence(
    cases: list[dict], optional_names: list[str]
) -> list[dict]:
    """Fork each case into 2^k variants for k optional tensors.

    Each fork sets `inputs[name]` to either the original tensor (present=True)
    or `None` (present=False). The case `name` and `meta` get a `_<name>{T|F}`
    suffix per optional tensor so downstream filters can locate the absent-path
    cases.

    Args:
        cases: list of case dicts as produced by the main band loops
        optional_names: tensor names declared `optional=True` in SCHEMA

    Returns:
        new list of cases, length = len(cases) * (2 ** len(optional_names))
    """
    if not optional_names:
        return cases
    expanded: list[dict] = []
    next_idx = 0
    # All 2^k presence patterns (k=len(optional_names))
    n_patterns = 1 << len(optional_names)
    for c in cases:
        for pattern in range(n_patterns):
            new_inputs = dict(c["inputs"])
            new_meta = dict(c.get("meta", {}))
            presence_tags = []
            for j, name in enumerate(optional_names):
                present = bool(pattern & (1 << j))
                if not present:
                    new_inputs[name] = None
                presence_tags.append(f"{name}{'T' if present else 'F'}")
                new_meta[f"optional_{name}_present"] = present
            new_case = {
                "idx": next_idx,
                "name": c["name"] + "_" + "_".join(presence_tags),
                "shape": c["shape"],
                "inputs": new_inputs,
                "meta": new_meta,
            }
            expanded.append(new_case)
            next_idx += 1
    return expanded
