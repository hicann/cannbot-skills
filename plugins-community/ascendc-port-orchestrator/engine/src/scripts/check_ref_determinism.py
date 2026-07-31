# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Phase O2.5 reference-determinism pre-flight (orchestrator-side).

Why this exists: torch_npu CANN ops are not always run-to-run deterministic on
Ascend950PR. Examples confirmed 2026-04-24:
  - npu_kv_rmsnorm_rope_cache PA_BLK_BNSD: 3-run max_diff ~10
  - npu_kv_rmsnorm_rope_cache + duplicate indices: 3-run max_diff ~8
  - npu_kv_rmsnorm_rope_cache + is_output_kv=False: garbage in [2]/[3] outputs

If the worker writes a deterministic kernel against a non-deterministic reference,
verifier bit-exact compare will FAIL no matter how correct the kernel is. This
script fires reference 3x on identical inputs from edge_inputs.pt, computes
per-output run-to-run max_diff, and emits ref_determinism.json the worker
reads in Phase A.

Usage (from orchestrator host):
    python3 src/scripts/check_ref_determinism.py \\
        --op {op_slug} \\
        --workspace workspace/{op}/ \\
        --edge-inputs workspace/{op}/edge_inputs.pt \\
        --model-py /path/to/{N}_{Name}.py \\
        --runner-args '{"cache_mode": "Norm", "is_output_kv": false}' \\
        --num-runs 3

The script ssh's into A5, runs the model 3x on the first N cases of edge_inputs.pt,
diffs each output, writes ref_determinism.json:

    {
      "op": "kvrmsnormropecache",
      "n_cases_probed": 5,
      "n_runs": 3,
      "results": [
        {"case_idx": 0, "case_name": "...", "outputs": {
            "k_cache_out":   {"max_diff_run01": 0.0, "max_diff_run02": 0.0, "verdict": "DETERMINISTIC"},
            "k_embed_ret":   {"max_diff_run01": 6.78, "max_diff_run02": 0.0, "verdict": "NON_DETERMINISTIC"},
            ...
        }},
        ...
      ],
      "summary": {
        "all_outputs_deterministic": false,
        "non_deterministic_outputs": ["k_embed_ret", "y_ret"],
        "scoping_recommendation": "Worker should scope kernel only to outputs marked DETERMINISTIC. "
        "Non-deterministic outputs cannot be bit-matched and should be either (a) dropped from "
        "comparison or (b) zero-filled to match common ref cold-buffer behavior."
      }
    }

Exit codes: 0 if probe ran (regardless of det/non-det verdict), 1 if probe failed.
"""

from __future__ import annotations
import argparse
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import textwrap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--op", required=True, help="op slug (workspace dir name)")
    p.add_argument("--workspace", required=True, help="workspace/{op}/ path")
    p.add_argument("--edge-inputs", required=True, help="path to edge_inputs.pt")
    p.add_argument("--model-py", required=True, help="path to reference Model.py")
    p.add_argument("--runner-args", default="{}", help="JSON dict of extra kwargs (e.g. cache_mode, is_output_kv)")
    p.add_argument("--num-runs", type=int, default=3, help="# repeat runs (default 3)")
    p.add_argument("--n-cases", type=int, default=5, help="# probe cases from edge_inputs.pt (default 5)")
    p.add_argument("--a5-host", default=os.environ.get("A5_HOST", "198.51.100.35"))
    p.add_argument("--a5-user", default=os.environ.get("A5_USER", "root"))
    # no hardcoded credential; from env / --a5-password
    p.add_argument("--a5-password", default=os.environ.get("A5_PASSWORD"))
    p.add_argument("--a5-container", default=os.environ.get("A5_CONTAINER", "npu_dev3"))
    p.add_argument("--cann-env", default="/usr/local/Ascend/cann-9.0.0/set_env.sh")
    p.add_argument("--npu-id", type=int, default=1, help="ASCEND_RT_VISIBLE_DEVICES (default 1)")
    return p.parse_args()


def _executable_path(name: str) -> str:
    """Return the resolved command path, preserving subprocess' missing-file error."""
    return str(pathlib.Path(shutil.which(name) or name).resolve())


PROBE_TEMPLATE = r"""
import importlib.util, json, pathlib, sys
import torch
import torch_npu  # noqa: F401

INPUTS_PT = pathlib.Path("/data/ascendc_op_gen_stage/{op_slug}/edge_inputs.pt")
MODEL_PY  = pathlib.Path("/data/ascendc_op_gen_stage/{op_slug}/{model_basename}")

def load_model():
    spec = importlib.util.spec_from_file_location("bench_model", MODEL_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Model

def to_cpu(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu()
    return x

def deep_diff(a, b, path=""):
    # Return max abs diff between two same-structured outputs (tensor / tuple / None).
    if a is None and b is None: return 0.0
    if a is None or b is None:  return float("inf")
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape: return float("inf")
        return (a.float() - b.float()).abs().max().item()
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b): return float("inf")
        return max(deep_diff(x, y) for x, y in zip(a, b))
    return 0.0

def main():
    import inspect as _inspect
    d = torch.load(INPUTS_PT, weights_only=False)
    cases = d["cases"][:{n_cases}]
    Model = load_model()
    model = Model().npu()

    runner_args = {runner_args!r}
    runner_kwargs = json.loads(runner_args) if runner_args else {{}}

    output_names = {output_names!r}
    n_runs = {n_runs}

    # Get the actual forward() signature param names (drop 'self')
    fwd_params = list(_inspect.signature(model.forward).parameters.keys())
    print(f"[refdet] Model.forward params: {{fwd_params}}")

    results = []
    for ci, c in enumerate(cases):
        inp_cpu = c["inputs"]
        # Restrict to params Model.forward actually accepts
        inp_npu = {{}}
        for k, v in inp_cpu.items():
            if k not in fwd_params: continue  # SCHEMA-only scalar (e.g. batch_size)
            inp_npu[k] = (v.npu().contiguous() if isinstance(v, torch.Tensor) else v)

        # Each run: clone the .npu() inputs (cache buffers may be mutated)
        run_outputs = []
        for r in range(n_runs):
            args_for_run = {{}}
            for k, v in inp_npu.items():
                args_for_run[k] = v.clone() if isinstance(v, torch.Tensor) else v
            # Add runner_kwargs (only those Model.forward accepts)
            for k, v in runner_kwargs.items():
                if k in fwd_params:
                    args_for_run[k] = v
            with torch.no_grad():
                out = model(**args_for_run)
            if isinstance(out, torch.Tensor): out = (out,)
            run_outputs.append(tuple(to_cpu(x) for x in out))

        # Per-output diff vs run 0
        case_result = {{
            "case_idx": ci,
            "case_name": c.get("name", f"case_{{ci}}"),
            "outputs": {{}}
        }}
        for oi in range(len(run_outputs[0])):
            name = output_names[oi] if oi < len(output_names) else f"output_{{oi}}"
            diffs = [deep_diff(run_outputs[0][oi], run_outputs[r][oi]) for r in range(1, n_runs)]
            verdict = "DETERMINISTIC" if max(diffs) <= 1e-6 else "NON_DETERMINISTIC"
            case_result["outputs"][name] = {{
                "max_diff_per_run": [float(d) for d in diffs],
                "verdict": verdict,
                "is_none_run0": run_outputs[0][oi] is None,
            }}
        results.append(case_result)

    # Aggregate verdict per output name
    by_output = {{}}
    for r in results:
        for n, info in r["outputs"].items():
            by_output.setdefault(n, []).append(info["verdict"])
    summary = {{}}
    non_det = []
    for n, verdicts in by_output.items():
        non_det_count = sum(1 for v in verdicts if v == "NON_DETERMINISTIC")
        summary[n] = {{
            "n_cases": len(verdicts),
            "n_non_det": non_det_count,
            "verdict": "NON_DETERMINISTIC" if non_det_count > 0 else "DETERMINISTIC",
        }}
        if non_det_count > 0:
            non_det.append(n)

    out = {{
        "op": "{op_slug}",
        "n_cases_probed": len(cases),
        "n_runs": n_runs,
        "runner_kwargs": runner_kwargs,
        "results": results,
        "summary_per_output": summary,
        "non_deterministic_outputs": non_det,
        "all_outputs_deterministic": len(non_det) == 0,
        "scoping_recommendation": (
            "All outputs are RUN-TO-RUN DETERMINISTIC; worker may bit-exact compare against reference."
            if not non_det else
            "Worker MUST scope kernel comparison to deterministic outputs only. "
            "Non-deterministic outputs ({{nd}}) cannot be bit-matched and require either (a) dropping "
            "from comparison or (b) zero-filling to match common ref cold-buffer behavior. "
            "Document the scope decision in analysis.md.".format(nd=", ".join(non_det))
        )
    }}
    print("===REF_DET_JSON_BEGIN===")
    print(json.dumps(out, indent=2))
    print("===REF_DET_JSON_END===")

main()
"""


def main():
    args = parse_args()
    workspace = pathlib.Path(args.workspace).resolve()
    if not workspace.is_dir():
        print(f"ERROR: workspace {workspace} not a dir", file=sys.stderr)
        return 1

    edge_inputs = pathlib.Path(args.edge_inputs).resolve()
    model_py = pathlib.Path(args.model_py).resolve()
    if not edge_inputs.is_file():
        print(f"ERROR: --edge-inputs {edge_inputs} not found", file=sys.stderr)
        return 1
    if not model_py.is_file():
        print(f"ERROR: --model-py {model_py} not found", file=sys.stderr)
        return 1
    if not args.a5_password:
        print("ERROR: --a5-password or A5_PASSWORD is required", file=sys.stderr)
        return 1

    # Try to infer output_names from a known map, else number them generically
    op_to_outputs = {
        "kvrmsnormropecache": ["k_cache_out", "ckv_cache_out", "k_embed_ret", "y_ret"],
        "fusedropewithqknormandkvcacheupdate": [
            "query_rotated", "key_rotated", "key_cache_out", "value_cache_out"],
        "kvcacheupdatewithropebackward": ["q_grad", "k_grad", "v_grad", "kc_grad"],
    }
    output_names = op_to_outputs.get(args.op, [f"output_{i}" for i in range(8)])

    # Stage edge_inputs + model.py to A5 if not already there
    op_stage = f"/data/ascendc_op_gen_stage/{args.op}"
    host = f"{args.a5_user}@{args.a5_host}"
    host_stage = f"/tmp/{args.op}_refdet_stage"  # nosec B108 - remote staging contract
    print(f"==> Staging probe inputs to A5 {op_stage}/")
    ssh_prefix = ["sshpass", "-p", args.a5_password]
    stage_commands = [
        (
            "create remote staging directory",
            [*ssh_prefix, "ssh", "-o", "StrictHostKeyChecking=no", host,
             shlex.join(["mkdir", "-p", host_stage])],
        ),
        (
            "copy probe inputs",
            [*ssh_prefix, "scp", "-o", "StrictHostKeyChecking=no",
             str(edge_inputs), str(model_py), f"{host}:{host_stage}/"],
        ),
    ]
    container_commands = [
        ["docker", "exec", args.a5_container, "mkdir", "-p", op_stage],
        ["docker", "cp", f"{host_stage}/{edge_inputs.name}",
         f"{args.a5_container}:{op_stage}/edge_inputs.pt"],
        ["docker", "cp", f"{host_stage}/{model_py.name}",
         f"{args.a5_container}:{op_stage}/{model_py.name}"],
    ]
    stage_commands.append((
        "copy inputs into container",
        [*ssh_prefix, "ssh", "-o", "StrictHostKeyChecking=no", host,
         " && ".join(shlex.join(command) for command in container_commands)],
    ))
    for description, command in stage_commands:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"STAGE FAILED ({description}):\n{result.stderr}", file=sys.stderr)
            return 1

    # Build the probe script
    probe = PROBE_TEMPLATE.format(
        op_slug=args.op,
        model_basename=model_py.name,
        n_cases=args.n_cases,
        runner_args=args.runner_args,
        output_names=output_names,
        n_runs=args.num_runs,
    )
    probe_path = workspace / "_refdet_probe.py"
    probe_path.write_text(probe)
    print(f"==> Wrote probe script to {probe_path}")

    # scp the probe script to A5
    sshpass = _executable_path("sshpass")
    scp = _executable_path("scp")
    ssh = _executable_path("ssh")
    r = subprocess.run([
        sshpass, "-p", args.a5_password, scp, "-o", "StrictHostKeyChecking=no",
        str(probe_path),
        f"{args.a5_user}@{args.a5_host}:/tmp/{args.op}_refdet_stage/_refdet_probe.py",
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"SCP probe FAILED: {r.stderr}", file=sys.stderr)
        return 1
    r = subprocess.run([
        sshpass, "-p", args.a5_password, ssh, "-o", "StrictHostKeyChecking=no",
        f"{args.a5_user}@{args.a5_host}",
        f"docker cp /tmp/{args.op}_refdet_stage/_refdet_probe.py {args.a5_container}:{op_stage}/_refdet_probe.py",
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"docker cp probe FAILED: {r.stderr}", file=sys.stderr)
        return 1

    # Run probe inside container
    print(f"==> Running probe on A5 NPU {args.npu_id} ...")
    r = subprocess.run([
        sshpass, "-p", args.a5_password, ssh, "-o", "StrictHostKeyChecking=no",
        f"{args.a5_user}@{args.a5_host}",
        f"docker exec -e ASCEND_RT_VISIBLE_DEVICES={args.npu_id} {args.a5_container} "
        f"bash -c 'source {args.cann_env} ; cd {op_stage} ; python3 _refdet_probe.py 2>&1'",
    ], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"PROBE FAILED:\n{r.stdout}\n---\n{r.stderr}", file=sys.stderr)
        return 1

    out = r.stdout
    # Extract JSON between markers
    begin = out.find("===REF_DET_JSON_BEGIN===")
    end = out.find("===REF_DET_JSON_END===")
    if begin < 0 or end < 0:
        print(f"PROBE OUTPUT MISSING JSON MARKERS:\n{out}", file=sys.stderr)
        return 1

    json_text = out[begin + len("===REF_DET_JSON_BEGIN==="):end].strip()
    try:
        result = json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"INVALID JSON: {e}\nText:\n{json_text}", file=sys.stderr)
        return 1

    # Save artifact
    out_path = workspace / "ref_determinism.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"==> Wrote {out_path}")

    # Print summary
    print()
    print("=" * 60)
    print(f"Reference determinism: {args.op}")
    print("=" * 60)
    print(f"  n_cases probed: {result['n_cases_probed']}, n_runs: {result['n_runs']}")
    print(f"  All outputs deterministic? {result['all_outputs_deterministic']}")
    print(f"  Non-deterministic outputs: {result['non_deterministic_outputs']}")
    print("  Per-output verdict:")
    for name, info in result["summary_per_output"].items():
        print(f"    {name}: {info['verdict']} ({info['n_non_det']}/{info['n_cases']} cases non-det)")
    print()
    print("Recommendation:")
    print(textwrap.fill(result["scoping_recommendation"], width=80, initial_indent="  ", subsequent_indent="  "))
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
