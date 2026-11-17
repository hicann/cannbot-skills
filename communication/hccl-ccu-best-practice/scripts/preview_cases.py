#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Read-only, source-verified hccl_test input models for six collectives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

PROFILE = (
    Path(__file__).resolve().parents[1]
    / "references/hccl-test-collectives-profile.json"
)
DTYPE_BYTES = {
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "fp16": 2,
    "bfp16": 2,
    "int32": 4,
    "uint32": 4,
    "fp32": 4,
    "int64": 8,
    "uint64": 8,
    "fp64": 8,
}
OPS = ("all_gather", "all_reduce", "reduce_scatter", "broadcast", "reduce", "scatter")
SPLIT_OPS = {"all_gather", "reduce_scatter", "scatter"}
ROOTED_OPS = {"broadcast", "reduce", "scatter"}
REDUCE_OPS = {"all_reduce", "reduce_scatter", "reduce"}
HVM_OPS = {"all_gather": "allgather", "all_reduce": "allreduce"}
U64_MAX = (1 << 64) - 1


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def positive(value, name):
    if type(value) is not int or not 0 < value <= U64_MAX:
        raise ValueError(name + " must be a positive uint64 integer")


def selected_profile(profile, op):
    # The old public AllGather entry accepts its existing source profile.
    if (
        profile.get("model") == "hccl-test-allgather-align-ceil-v1"
        and op == "all_gather"
    ):
        return profile["model"], profile["sources"], "all_gather_test"
    if profile.get("model") != "hccl-test-collectives-v1":
        raise ValueError("unsupported reviewed model profile")
    model = profile["operators"][op]
    sources = {**profile["common_sources"], **model["sources"]}
    return "hccl-test-" + op + "-v1", sources, model["binary"]


def preview(
    root,
    rank_count,
    dtype,
    sizes,
    *,
    op,
    root_rank=None,
    reduce_op=None,
    max_total_bytes=None,
    max_aggregate_bytes=None,
    max_count_per_loop=None,
    loop_basis=None,
    test_binary=None,
    profile_path=PROFILE,
):
    if op not in OPS:
        raise ValueError("unsupported operator directory name: " + str(op))
    positive(rank_count, "rank_count (not a communication-domain ID)")
    if rank_count > (1 << 31) - 1:
        raise ValueError("rank_count exceeds the reviewed MPI int domain")
    if dtype not in DTYPE_BYTES:
        raise ValueError("unsupported dtype in this reviewed model: " + str(dtype))
    if not sizes or any(
        type(s) is not int or not 0 <= s <= (1 << 63) - 1 for s in sizes
    ):
        raise ValueError(
            "sizes must be nonempty nonnegative signed-64 integers (hccl_test long bytes)"
        )
    if op in ROOTED_OPS:
        root_rank = 0 if root_rank is None else root_rank
        if type(root_rank) is not int or not 0 <= root_rank < rank_count:
            raise ValueError("root_rank must be an integer in [0, rank_count)")
    elif root_rank is not None:
        raise ValueError("root_rank is not applicable to " + op)
    if op in REDUCE_OPS:
        reduce_op = "sum" if reduce_op is None else reduce_op
        if reduce_op not in ("sum", "min", "max", "prod"):
            raise ValueError("unsupported hccl_test reduce operation")
    elif reduce_op is not None:
        raise ValueError("reduce_op is not applicable to " + op)
    for name, value in (
        ("max_total_bytes", max_total_bytes),
        ("max_aggregate_bytes", max_aggregate_bytes),
        ("max_count_per_loop", max_count_per_loop),
    ):
        if value is not None:
            positive(value, name)
    if max_count_per_loop is not None and (
        not isinstance(loop_basis, str) or not loop_basis.strip()
    ):
        raise ValueError(
            "a source-grounded loop_basis is required with max_count_per_loop"
        )
    root = Path(root).resolve()
    profile = json.loads(Path(profile_path).read_text())
    model, source_map, binary_name = selected_profile(profile, op)
    if not source_map:
        raise ValueError("empty reviewed model sources")
    sources = []
    for relative, expected in source_map.items():
        p = (root / relative).resolve()
        if not p.is_relative_to(root):
            raise ValueError("profile path escapes hccl_test root")
        actual = digest(p) if p.is_file() else None
        sources.append(
            dict(
                path=relative,
                expected_sha256=expected,
                sha256=actual,
                status="MATCH"
                if actual == expected
                else "CHANGED"
                if actual
                else "MISSING",
            )
        )
    out = dict(
        schema_version=1,
        op=op,
        model=model,
        model_verified=False,
        profile_sha256=digest(profile_path),
        sources=sources,
        cases=[],
        scope="input prediction only; not algorithm support, branch coverage or a test verdict",
    )
    if any(s["status"] != "MATCH" for s in sources):
        out.update(
            ok=False,
            error="SOURCE_MODEL_UNVERIFIED: review only the changed model functions; no prediction emitted",
        )
        return out
    binary = Path(test_binary).resolve() if test_binary else root / "bin" / binary_name
    flags = ["-r", str(root_rank)] if root_rank is not None else []
    flags += ["-o", reduce_op] if reduce_op is not None else []
    out.update(
        model_verified=True,
        rank_count=rank_count,
        dtype=dtype,
        dtype_bytes=DTYPE_BYTES[dtype],
        root_rank=root_rank,
        reduce_op=reduce_op,
        test_binary=dict(
            path=str(binary),
            sha256=digest(binary) if binary.is_file() else None,
            source_binding="UNVERIFIED: confirm with a trusted build manifest before relying on predictions",
        ),
        execution_requirements=dict(
            hvm_operator=HVM_OPS.get(op, op),
            hccl_test_flags=flags,
            hvm_defaults_suffice=(
                root_rank in (None, 0) and reduce_op in (None, "sum")
            ),
            runner_basis=(
                "Reviewed hvm run.sh defaults: root 0; reduce classes use sum. "
                "Recheck installed runner; nondefault flags require a verified "
                "runner or direct command."
            ),
        ),
        size_scope=(
            "all ranks concatenated data span" if op in SPLIT_OPS else "one rank buffer"
        ),
        max_total_bytes=max_total_bytes,
        max_aggregate_bytes=max_aggregate_bytes,
        aggregate_budget_scope=(
            "sum of basic device send/recv allocations across MPI ranks; excludes "
            "host copies, Checker/VM amplification, scratch, alignment and runtime overhead"
        ),
        max_count_per_loop=max_count_per_loop,
        loop_basis=loop_basis,
        duplicate_effective_inputs=[],
    )
    width = DTYPE_BYTES[dtype]
    seen = {}
    for requested in sizes:
        row = dict(requested_total_bytes=requested, tool_accepts=requested > 0)
        if not requested:
            row["reason"] = (
                "hccl_test rejects nonpositive min/max bytes; does not exercise the CCU zero-data path"
            )
            out["cases"].append(row)
            continue
        aligned = requested
        if op in SPLIT_OPS:
            aligned = max(requested, rank_count)
            alignment = rank_count * (512 if aligned > 16 * 1024 * 1024 else 1)
            aligned = aligned // alignment * alignment
        total_elements = (aligned + width - 1) // width
        count = (
            (total_elements + rank_count - 1) // rank_count
            if op in SPLIT_OPS
            else total_elements
        )
        per_rank = count * width
        effective = per_rank * rank_count if op in SPLIT_OPS else per_rank
        send = effective if op in ("reduce_scatter", "scatter") else per_rank
        recv = effective if op == "all_gather" else 0 if op == "broadcast" else per_rank
        if (
            max(
                aligned + width - 1,
                total_elements + rank_count - 1,
                send,
                recv,
                effective,
            )
            > U64_MAX
        ):
            raise ValueError("input exceeds reviewed uint64 arithmetic bounds")
        aggregate = rank_count * (send + recv)
        # Semantic roles differ from the test tool's allocation on every rank.
        semantic = dict(all_ranks=dict(input_bytes=send, output_bytes=recv))
        if op == "broadcast":
            semantic = dict(
                root=dict(input_bytes=per_rank, output_bytes=per_rank),
                non_root=dict(input_bytes=0, output_bytes=per_rank),
            )
        elif op == "reduce":
            semantic = dict(
                root=dict(input_bytes=per_rank, output_bytes=per_rank),
                non_root=dict(input_bytes=per_rank, output_bytes=0),
            )
        elif op == "scatter":
            semantic = dict(
                root=dict(input_bytes=effective, output_bytes=per_rank),
                non_root=dict(input_bytes=0, output_bytes=per_rank),
            )
        row.update(
            normalized_tool_bytes=aligned,
            count_per_rank=count,
            input_bytes_per_rank=send,
            output_bytes_per_rank=per_rank if op == "broadcast" else recv,
            allocated_send_bytes_per_rank=send,
            allocated_recv_bytes_per_rank=recv,
            semantic_bytes=semantic,
            broadcast_buffer_alias=(op == "broadcast"),
            effective_total_bytes=effective,
            aggregate_input_output_bytes=aggregate,
            within_budget=(
                None
                if max_total_bytes is None
                else max(requested, effective) <= max_total_bytes
            ),
            within_aggregate_budget=(
                None
                if max_aggregate_bytes is None
                else aggregate <= max_aggregate_bytes
            ),
            non_dtype_aligned_api_span_expressible=False,
            kernel_slice_alignment_proven=False,
            note="API span bytes = integer count * dtype width; internal template partitioning needs its own proof",
            fastlaunch_buffer_change_proven=False,
        )
        if op == "all_gather":
            row["non_dtype_aligned_slice_expressible"] = False
        if count == 0:
            row.update(
                model_supports_case=False,
                reason="rank alignment collapsed positive CLI bytes to zero; not a normal positive-count test",
            )
        if max_count_per_loop is not None:
            row["predicted_loop_times"] = (
                count + max_count_per_loop - 1
            ) // max_count_per_loop
            row["loop_coverage"] = "requires actual executor-path and loop log evidence"
        if effective in seen:
            out["duplicate_effective_inputs"].append(
                dict(
                    requested_total_bytes=requested,
                    same_effective_input_as=seen[effective],
                    effective_total_bytes=effective,
                )
            )
        else:
            seen[effective] = requested
        out["cases"].append(row)
    out["ok"] = all(
        r["tool_accepts"]
        and r.get("model_supports_case") is not False
        and r.get("within_budget") is not False
        and r.get("within_aggregate_budget") is not False
        for r in out["cases"]
    )
    return out


def main(default_op=None, profile_path=PROFILE):
    p = argparse.ArgumentParser(description=__doc__)
    if default_op is None:
        p.add_argument("--op", choices=OPS, required=True)
    p.add_argument("--hccl-test-root", required=True)
    p.add_argument(
        "--rank-count",
        type=int,
        required=True,
        help="actual MPI rank count, never the -t domain code",
    )
    p.add_argument("--dtype", required=True, choices=sorted(DTYPE_BYTES))
    p.add_argument("--sizes", type=int, nargs="+", required=True)
    p.add_argument("--root-rank", type=int)
    p.add_argument("--reduce-op", choices=("sum", "min", "max", "prod"))
    p.add_argument("--max-total-bytes", type=int)
    p.add_argument("--max-aggregate-bytes", type=int)
    p.add_argument("--max-count-per-loop", type=int)
    p.add_argument("--loop-basis")
    p.add_argument("--test-binary")
    a = vars(p.parse_args())
    op = a.pop("op", default_op)
    root = a.pop("hccl_test_root")
    try:
        result = preview(root, op=op, profile_path=profile_path, **a)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        result = dict(ok=False, error=str(exc), cases=[])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
