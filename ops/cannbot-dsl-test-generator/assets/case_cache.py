#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Reusable CPU input and reference data; target execution is never cached."""

import hashlib
import importlib.util
import json
import os
import pickle
import tempfile
from pathlib import Path

import torch

CACHE_VERSION = 1


def load_golden(path):
    path = Path(path)
    source = path.read_bytes()
    spec = importlib.util.spec_from_file_location("cannbotdsl_golden", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"missing golden: {path}")
    module = importlib.util.module_from_spec(spec)
    # The code and cache identity must use the same source, even for edits with
    # identical size/mtime that Python's timestamp bytecode cache can miss.
    exec(compile(source, str(path), "exec"), module.__dict__)
    module.__cannbotdsl_source_hash__ = hashlib.sha256(source).hexdigest()
    return module


def _check_cpu_data(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("case cache requires CPU tensors")
    elif isinstance(value, dict):
        if not all(type(key) is str for key in value):
            raise ValueError("case cache mapping keys must be strings")
        for item in value.values():
            _check_cpu_data(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _check_cpu_data(item)
    elif value is not None and type(value) not in (bool, int, float, str):
        raise ValueError(f"unsupported case cache value: {type(value).__name__}")


class CaseCache:
    def __init__(self, test_dir, golden):
        cache_dir = os.environ.get("CANNBOTDSL_CASE_CACHE_DIR")
        self.root = Path(cache_dir).expanduser() if cache_dir else Path(test_dir) / ".case_cache"
        self.golden = golden
        self.identity = {
            "version": CACHE_VERSION,
            "golden": golden.__cannbotdsl_source_hash__,
            "torch": str(torch.__version__),
        }
        self.hits = 0
        self.built = 0

    def get(self, row, call, index):
        recipe = {
            **self.identity,
            "sheet": row["sheet"],
            "case_id": row["case_id"],
            "call": index,
            "inputs": call["inputs"],
            "seed": call["seed"],
            "expected": row.get("expected"),
            "assertions": call["assertions"],
            "outputs": json.loads(row.get("output_json") or "{}"),
        }
        key = hashlib.sha256(
            json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        path = self.root / f"{key}.pt"
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
            self._validate(data, key)
        except (
            OSError,
            RuntimeError,
            EOFError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            pickle.UnpicklingError,
        ):
            data = self._build(recipe, key)
            self._write(path, data)
            self.built += 1
        else:
            self.hits += 1
        return data

    @staticmethod
    def _validate(data, key):
        if (
            not isinstance(data, dict)
            or set(data) != {"version", "key", "inputs", "expected", "outputs"}
            or data["version"] != CACHE_VERSION
            or data["key"] != key
            or not isinstance(data["inputs"], dict)
            or not isinstance(data["outputs"], dict)
        ):
            raise ValueError("invalid case cache entry")
        _check_cpu_data(data)

    def _build(self, recipe, key):
        inputs = self.golden.make_inputs(recipe["inputs"], seed=recipe["seed"])
        expected = None
        if recipe["expected"] in ("match_golden", "match_golden_nan"):
            _, expected = self.golden.simulate(recipe["inputs"], seed=recipe["seed"])
        outputs = {}
        if recipe["outputs"]:
            if not isinstance(recipe["outputs"], dict) or set(recipe["outputs"]) != set(
                self.golden.OUTPUT_NAMES
            ):
                raise ValueError("output_json must contain exactly the spec outputs")
            outputs = self.golden.make_inputs(recipe["outputs"], seed=0)
            for tensor in outputs.values():
                if isinstance(tensor, torch.Tensor):
                    tensor.zero_()
        data = {
            "version": CACHE_VERSION,
            "key": key,
            "inputs": inputs,
            "expected": expected,
            "outputs": outputs,
        }
        self._validate(data, key)
        return data

    def _write(self, path, data):
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.root, prefix=".case-", suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
                torch.save(data, handle)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def prepare(self, rows, expand_calls):
        for row in rows:
            if row["status"] != "active":
                continue
            inputs = json.loads(row["inputs_json"])
            seed = int(row.get("seed") or 42)
            assertions = json.loads(row.get("assertions_json") or "[]")
            if (row.get("expected") or "").startswith("raises_error:"):
                calls = [{"inputs": inputs, "seed": seed, "assertions": assertions}]
            else:
                _, _, calls = expand_calls(inputs, seed, assertions)
            for index, call in enumerate(calls):
                self.get(row, call, index)
        return {"built": self.built, "hits": self.hits}
