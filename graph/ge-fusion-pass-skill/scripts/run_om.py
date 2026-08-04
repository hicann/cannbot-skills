#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run an OM model with the ACL runner and emit validation-evidence manifests."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


DTYPES = {
    "float32": "float32", "float16": "float16", "float64": "float64",
    "int8": "int8", "uint8": "uint8", "int16": "int16", "uint16": "uint16",
    "int32": "int32", "uint32": "uint32", "int64": "int64", "uint64": "uint64",
    "bool": "bool",
}
REQUIRED_CONTEXT_KEYS = (
    "source_model_sha256", "input_sha256", "seed", "preprocess", "soc_version",
    "compile_parameters", "run_parameters", "environment",
)


@dataclass(frozen=True)
class RunInputs:
    """Validated paths and provenance used by one ACL runner invocation."""

    runner: Path
    model: Path
    output_dir: Path
    context_path: Path
    context: dict
    source_model: Path
    source_model_sha256: str
    input_files: list
    input_sha256: str
    input_hash_scheme: str


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _resolve_context_path(value, context_path):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (context_path.parent / path).resolve()


def input_fingerprint(input_paths):
    """Return per-file hashes and a deterministic multi-input fingerprint.

    A single raw input keeps the historical ``input_sha256 == file sha256``
    spelling.  For multiple inputs, ``input_sha256`` is the SHA256 of the
    ordered ``index:sha256`` lines.  The manifest records both forms so a
    later comparison never has to trust a caller-provided context alone.
    """
    files = []
    for index, value in enumerate(input_paths):
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"input file does not exist: {path}")
        files.append({
            "index": index,
            "path": str(path),
            "sha256": sha256(path),
            "size": path.stat().st_size,
        })
    canonical_lines = (f"{item.get('index')}:{item.get('sha256')}" for item in files)
    canonical = "\n".join(canonical_lines).encode("utf-8")
    aggregate = sha256_bytes(canonical)
    if len(files) == 1:
        selected = files[0]["sha256"]
        scheme = "single-file-sha256-v1"
    else:
        selected = aggregate
        scheme = "ordered-index-sha256-v1"
    return files, selected, {aggregate, selected}, scheme


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_output_stem(name, index):
    """Convert an ACL/GE tensor name into a portable artifact filename stem."""
    raw_name = name if isinstance(name, str) and name else f"output_{index}"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("._")
    return stem or f"output_{index}"


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, help="ACL OM runner binary explicitly supplied by the case")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--source-model",
        help="原始 ONNX/AIR/PB 模型；用于核验 context.source_model_sha256",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input", action="append", required=True, dest="inputs")
    parser.add_argument("--context-json", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    return parser


def _load_context(path):
    try:
        context = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read context-json: {exc}") from exc
    if not isinstance(context, dict):
        raise ValueError("context-json must contain a JSON object")
    missing = [key for key in REQUIRED_CONTEXT_KEYS if key not in context]
    if missing:
        raise ValueError(f"context-json missing required keys: {missing}")
    return context


def _validate_context_inputs(args, context, context_path):
    source_model_value = args.source_model or context.get("source_model_file")
    if not source_model_value:
        raise ValueError(
            "source model provenance is unbound: pass --source-model or set "
            "source_model_file in context-json"
        )
    source_model = _resolve_context_path(source_model_value, context_path)
    if not source_model.is_file():
        raise FileNotFoundError(f"source model does not exist: {source_model}")
    source_hash = sha256(source_model)
    if context.get("source_model_sha256") != source_hash:
        raise ValueError(
            "source_model_sha256 mismatch: "
            f"context={context.get('source_model_sha256')} actual={source_hash}"
        )
    fingerprint = input_fingerprint(args.inputs)
    input_files, input_sha256, accepted_hashes, input_hash_scheme = fingerprint
    if context.get("input_sha256") not in accepted_hashes:
        raise ValueError(
            "input_sha256 mismatch: "
            f"context={context.get('input_sha256')} actual={input_sha256}"
        )
    return source_model, source_hash, input_files, input_sha256, input_hash_scheme


def _validate_expected_inputs(context, input_files):
    expected_files = context.get("input_files")
    if expected_files is None:
        return
    if not isinstance(expected_files, list):
        raise ValueError("context input_files must be a JSON array")
    if not all(isinstance(item, dict) for item in expected_files):
        raise ValueError("context input_files must be a JSON array")
    expected_hashes = [item.get("sha256") for item in expected_files]
    actual_hashes = [item.get("sha256") for item in input_files]
    if expected_hashes != actual_hashes:
        raise ValueError(
            f"input_files mismatch: context={expected_hashes} actual={actual_hashes}"
        )


def _load_run_inputs(args):
    runner = Path(args.runner).resolve()
    model = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    context_path = Path(args.context_json).resolve()
    if not runner.is_file() or not model.is_file() or not context_path.is_file():
        raise FileNotFoundError("runner, model, and context-json must exist")
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("warmup must be >= 0 and runs must be >= 1")
    context = _load_context(context_path)
    source_values = _validate_context_inputs(args, context, context_path)
    source_model, source_hash, input_files, input_sha256, hash_scheme = source_values
    _validate_expected_inputs(context, input_files)
    return RunInputs(
        runner=runner,
        model=model,
        output_dir=output_dir,
        context_path=context_path,
        context=context,
        source_model=source_model,
        source_model_sha256=source_hash,
        input_files=input_files,
        input_sha256=input_sha256,
        input_hash_scheme=hash_scheme,
    )


def _runner_command(args, run_inputs):
    command = [
        str(run_inputs.runner),
        "--model",
        str(run_inputs.model),
        "--output-dir",
        str(run_inputs.output_dir),
        "--device",
        str(args.device),
        "--warmup",
        str(args.warmup),
        "--runs",
        str(args.runs),
    ]
    for input_path in args.inputs:
        command.extend(["--input", str(Path(input_path).resolve())])
    return command


def _run_acl_runner(command, output_dir):
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    (output_dir / "runner.stdout.log").write_text(completed.stdout, encoding="utf-8")
    stderr_path = output_dir / "runner.stderr.log"
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"ACL runner failed with exit {completed.returncode}; see {stderr_path}")


def _load_raw_outputs(output_dir):
    path = output_dir / "raw-outputs.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read ACL runner outputs: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("raw-outputs.json must contain a JSON object")
    return value


def _convert_output(np, output_dir, item, index):
    dtype_name = item.get("dtype")
    if dtype_name not in DTYPES:
        raise ValueError(f"unsupported ACL output dtype: {dtype_name!r}")
    shape = item.get("shape")
    if not isinstance(shape, list):
        raise ValueError(f"invalid output shape for {item.get('name')!r}: {shape!r}")
    if any(not isinstance(dim, int) or dim < 0 for dim in shape):
        raise ValueError(f"invalid output shape for {item.get('name')!r}: {shape!r}")
    raw_path = output_dir / item.get("path", "")
    array = np.fromfile(raw_path, dtype=np.dtype(DTYPES.get(dtype_name)))
    expected = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if array.size != expected:
        raise ValueError(f"{raw_path}: {array.size} values do not match shape {shape}")
    array = array.reshape(shape)
    name = item.get("name") or f"output_{index}"
    npy_name = f"{index:02d}_{safe_output_stem(name, index)}.npy"
    np.save(output_dir / npy_name, array, allow_pickle=False)
    return {"name": name, "path": npy_name, "raw_sha256": sha256(raw_path)}


def _convert_outputs(np, output_dir, raw):
    outputs = raw.get("outputs", [])
    if not isinstance(outputs, list):
        raise ValueError("raw-outputs.json outputs must be an array")
    converted = []
    for index, item in enumerate(outputs):
        if not isinstance(item, dict):
            raise ValueError(f"raw output at index {index} must be an object")
        converted.append(_convert_output(np, output_dir, item, index))
    if not converted:
        raise ValueError("ACL runner produced no outputs")
    return converted


def _execution_record(args, run_inputs):
    return {
        "runner": str(run_inputs.runner),
        "runner_sha256": sha256(run_inputs.runner),
        "model": str(run_inputs.model),
        "model_sha256": sha256(run_inputs.model),
        "device": args.device,
        "source_model": str(run_inputs.source_model),
        "source_model_sha256": run_inputs.source_model_sha256,
        "input_sha256": run_inputs.input_sha256,
        "input_hash_scheme": run_inputs.input_hash_scheme,
        "input_files": run_inputs.input_files,
    }


def _write_manifests(args, run_inputs, raw, converted):
    performance_context = dict(run_inputs.context)
    performance_context.update({"warmup": args.warmup, "runs": args.runs})
    execution = _execution_record(args, run_inputs)
    write_json(
        run_inputs.output_dir / "outputs.json",
        {"context": dict(run_inputs.context), "execution": execution, "outputs": converted},
    )
    write_json(
        run_inputs.output_dir / "performance.json",
        {
            "context": performance_context,
            "execution": execution,
            "latencies_ms": raw.get("latencies_ms", []),
        },
    )
    shutil.copy2(run_inputs.context_path, run_inputs.output_dir / "input-context.json")


def main():
    parser = _build_parser()
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit(f"numpy is required to write .npy outputs: {exc}") from exc
    try:
        run_inputs = _load_run_inputs(args)
        run_inputs.output_dir.mkdir(parents=True, exist_ok=True)
        command = _runner_command(args, run_inputs)
        _run_acl_runner(command, run_inputs.output_dir)
        raw = _load_raw_outputs(run_inputs.output_dir)
        converted = _convert_outputs(np, run_inputs.output_dir, raw)
        _write_manifests(args, run_inputs, raw, converted)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
