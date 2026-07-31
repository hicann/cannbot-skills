# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""B3.2: backward-mode Phase O2.5 self-contained reference + orchestrator dispatch.

Two layers:
  A. Unit — phase_o25_backward.provision_backward_reference (pure CPU): forward-spec
     contract validation, autograd-vs-analytic golden (§8 layer 4), §5.4 degenerate
     skip, model.py generation + equivalence to compute_backward_reference, NOT_DIFF.
  B. Call-level — run_single_op routes a backward workspace through the O2.5
     `elif opgen_mode == "backward"` dispatch → provision → returns 98 (B3.2
     boundary); benchmark/port_a3 modes unaffected. Mirrors the hermetic fixture
     pattern of test_run_single_op_port_a3_dispatch.py.

No hardware; pure CPU + autograd. The worker kernel-gen + on-NPU verify is B3.3.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock

import pytest

_HERE = Path(__file__).resolve()
_ORCH_DIR = _HERE.parent.parent  # src/scripts/orchestrator/
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))
_RP_DIR = _ORCH_DIR.parent / "reference_provider"
if str(_RP_DIR) not in sys.path:
    sys.path.insert(0, str(_RP_DIR))

import phase_o25_backward as p25b  # noqa: E402

torch = pytest.importorskip("torch")


# ── forward-spec fixtures (mirror the real --backward <.py> contract) ─────────

_MUL_SPEC = '''\
import torch

def forward(x, w):
    return x * w

BACKWARD_SPEC = {
    "wrt": ["x", "w"],
    "inputs": {
        "x": {"shape": ["N"], "dtype": "float32"},
        "w": {"shape": ["N"], "dtype": "float32"},
    },
    "cases": [{"N": 8}, {"N": 16}],
    "dtypes": ["float32"],
    "grad_output": "explicit",
    "seed": 1234,
}
'''

_RMS_SPEC = '''\
import torch

def forward(x, w):
    var = (x * x).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + 1e-6) * w

BACKWARD_SPEC = {
    "wrt": ["x", "w"],
    "inputs": {
        "x": {"shape": ["R", "H"]},
        "w": {"shape": ["H"]},
    },
    "cases": [{"R": 4, "H": 32}],
    "dtypes": ["float32", "float16"],
    "grad_output": "explicit",
}
'''


# Spec that OMITS `cases` entirely (symbolic shape) — provision must auto-derive
# a default sweep (owner 2026-06-16: user brings forward op, not test cases).
_NOCASE_SPEC = '''\
import torch

def forward(x, w):
    return x * w

BACKWARD_SPEC = {
    "wrt": ["x", "w"],
    "inputs": {
        "x": {"shape": ["N"], "dtype": "float32"},
        "w": {"shape": ["N"], "dtype": "float32"},
    },
    "dtypes": ["float32"],
    "grad_output": "explicit",
}
'''


def _write_spec(tmp_path: Path, body: str, name: str = "mul") -> Path:
    p = tmp_path / f"{name}.py"
    p.write_text(body)
    return p


# ════════════════════════════ A. provision unit ══════════════════════════════

def test_provision_ready_emits_all_artifacts(tmp_path):
    spec = _write_spec(tmp_path, _MUL_SPEC)
    ws = tmp_path / "workspace" / "mul_grad"
    rep = p25b.provision_backward_reference(ws, spec, op="mul_grad")
    assert rep.verdict == "READY", rep.summary
    from backward_input_gen import DEFAULT_PROFILES
    # input_gen (2026-06-11): each (case × dtype) expands over value-profiles.
    assert rep.n_ok == 2 * len(DEFAULT_PROFILES) and rep.n_skipped == 0
    for art in ("forward_spec.py", "model.py", "backward_ref.json",
                "backward_cpu_truth.pt"):
        assert (ws / art).is_file(), f"missing artifact {art}"
    ref = json.loads((ws / "backward_ref.json").read_text())
    assert ref["verdict"] == "RUNNABLE"
    assert ref["wrt"] == ["x", "w"]
    assert ref["reference_method"] == "self_contained_autograd_fp64"
    assert "seeding_recipe" in ref
    assert ref["input_profiles"] == DEFAULT_PROFILES
    # stub-form (#406 rework): per-case enumeration is OMITTED (regenerable from
    # forward_spec.cases x dtypes x input_profiles via seeding_recipe); metadata + the
    # value-profiles list + a cases_note are retained, and the golden lives in the .pt.
    assert "cases" not in ref and "cases_note" in ref
    assert ref["n_ok"] == 2 * len(DEFAULT_PROFILES)
    assert set(ref["dtypes"]) and ref["seed"] is not None


def test_auto_derive_cases_unit():
    """_auto_derive_cases: symbolic dims → small/medium/large sweep; concrete → [{}]."""
    assert getattr(p25b, "_auto_derive_cases")({"x": {"shape": ["N"]}, "w": {"shape": ["N"]}}) == [
        {"N": 64}, {"N": 128}, {"N": 256}]
    assert getattr(p25b, "_auto_derive_cases")(
        {"a": {"shape": ["M", "K"]}, "b": {"shape": ["K", "N"]}}) == [
        {"M": 64, "K": 64, "N": 64}, {"M": 128, "K": 128, "N": 128},
        {"M": 256, "K": 256, "N": 256}]
    assert getattr(p25b, "_auto_derive_cases")({"x": {"shape": [8]}}) == [{}]


def test_provision_auto_derives_cases_when_omitted(tmp_path):
    """Owner 2026-06-16: a forward spec with NO `cases` must still provision —
    cases auto-derive from the input signature (3-size sweep), not error out.
    """
    spec = _write_spec(tmp_path, _NOCASE_SPEC, name="nocase")
    ws = tmp_path / "workspace" / "nocase_grad"
    rep = p25b.provision_backward_reference(ws, spec, op="nocase_grad")
    assert rep.verdict == "READY", rep.summary
    ref = json.loads((ws / "backward_ref.json").read_text())
    from backward_input_gen import DEFAULT_PROFILES
    # 3 auto-derived sizes × 1 dtype × profiles
    assert ref["n_ok"] == 3 * len(DEFAULT_PROFILES)


def test_provision_golden_matches_analytic_mul(tmp_path):
    """§8 layer 4: autograd reference == hand-derived analytic gradient.
    For y = x*w with upstream gy: dx = gy*w, dw = gy*x.
    """
    spec = _write_spec(tmp_path, _MUL_SPEC)
    ws = tmp_path / "workspace" / "mul_grad"
    rep = p25b.provision_backward_reference(ws, spec, op="mul_grad")
    assert rep.verdict == "READY"
    truth = torch.load(ws / "backward_cpu_truth.pt", weights_only=False)
    recs = [r for r in truth["records"] if r["status"] == "ok"]
    assert recs
    for r in recs:
        # input_gen (2026-06-11): inputs are STORED in the record (per value-
        # profile) — use them directly rather than re-seeding.
        x = r["inputs"]["x"].double()
        w = r["inputs"]["w"].double()
        gy = r["grad_outputs"]
        gy = (gy[0] if isinstance(gy, list) else gy).double()
        dx_an = gy * w
        dw_an = gy * x
        assert torch.allclose(r["grads"]["x"].double(), dx_an, atol=1e-5), r
        assert torch.allclose(r["grads"]["w"].double(), dw_an, atol=1e-5), r


def test_generated_model_py_matches_oracle(tmp_path, monkeypatch):
    """The generated (deployable, inlined) model.py grads must equal
    compute_backward_reference (the authoritative oracle) bit-for-bit.
    """
    from autograd_backward_reference import compute_backward_reference
    spec = _write_spec(tmp_path, _MUL_SPEC)
    ws = tmp_path / "workspace" / "mul_grad"
    rep = p25b.provision_backward_reference(ws, spec, op="mul_grad")
    assert rep.verdict == "READY"

    # import the generated model.py (it imports forward from sibling forward_spec.py)
    monkeypatch.syspath_prepend(str(ws))
    mspec = importlib.util.spec_from_file_location("_gen_model_mul", ws / "model.py")
    mod = importlib.util.module_from_spec(mspec)
    mspec.loader.exec_module(mod)

    torch.manual_seed(7)
    x = torch.randn([8])
    w = torch.randn([8])
    gy = torch.randn([8])
    out = mod.Model()(x, w, gy)  # explicit → (x, w, gy)
    import forward_spec as fs
    oracle = compute_backward_reference(fs.forward, {"x": x, "w": w}, ["x", "w"], gy)
    assert torch.allclose(out[0].double(), oracle["x"])
    assert torch.allclose(out[1].double(), oracle["w"])


def test_provision_rms_multi_dtype(tmp_path):
    spec = _write_spec(tmp_path, _RMS_SPEC, name="rms_norm")
    ws = tmp_path / "workspace" / "rms_norm_grad"
    rep = p25b.provision_backward_reference(ws, spec, op="rms_norm_grad")
    assert rep.verdict == "READY", rep.summary
    from backward_input_gen import DEFAULT_PROFILES
    assert rep.n_ok == 2 * len(DEFAULT_PROFILES)  # 1 case × 2 dtypes × profiles


def test_provision_ones_grad_output(tmp_path):
    body = _MUL_SPEC.replace('"grad_output": "explicit"', '"grad_output": "ones"')
    spec = _write_spec(tmp_path, body, name="mul_ones")
    ws = tmp_path / "workspace" / "mul_ones_grad"
    rep = p25b.provision_backward_reference(ws, spec, op="mul_ones_grad")
    assert rep.verdict == "READY"
    # ones grad_output: dx = 1*w = w, dw = 1*x = x (use STORED inputs per profile)
    truth = torch.load(ws / "backward_cpu_truth.pt", weights_only=False)
    r = [r for r in truth["records"] if r["status"] == "ok"][0]
    x = r["inputs"]["x"].double()
    w = r["inputs"]["w"].double()
    assert torch.allclose(r["grads"]["x"].double(), w, atol=1e-5)
    assert torch.allclose(r["grads"]["w"].double(), x, atol=1e-5)


# ── §5.4 degenerate-reference guard ──────────────────────────────────────────

def test_provision_all_degenerate(tmp_path):
    """A forward whose fp64 output is non-finite → §5.4 skip → ALL_DEGENERATE."""
    body = dedent('''\
        import torch
        def forward(x):
            return x * float("inf")
        BACKWARD_SPEC = {
            "wrt": ["x"],
            "inputs": {"x": {"shape": [4]}},
            "cases": [{}],
            "dtypes": ["float32"],
            "grad_output": "ones",
        }
    ''')
    spec = _write_spec(tmp_path, body, name="ovf")
    ws = tmp_path / "workspace" / "ovf_grad"
    rep = p25b.provision_backward_reference(ws, spec, op="ovf_grad")
    assert rep.verdict == "ALL_DEGENERATE", rep.summary
    assert rep.n_ok == 0 and rep.n_skipped >= 1
    assert not (ws / "model.py").exists()  # no artifacts on block


# ── validation / block verdicts ──────────────────────────────────────────────

def test_forward_spec_missing(tmp_path):
    ws = tmp_path / "workspace" / "x_grad"
    rep = p25b.provision_backward_reference(ws, tmp_path / "nope.py", op="x_grad")
    assert rep.verdict == "FORWARD_SPEC_MISSING"


def test_spec_missing_backward_spec(tmp_path):
    spec = _write_spec(tmp_path, "import torch\ndef forward(x):\n    return x*2\n", "nb")
    ws = tmp_path / "workspace" / "nb_grad"
    rep = p25b.provision_backward_reference(ws, spec, op="nb_grad")
    assert rep.verdict == "SPEC_INVALID"
    assert any("BACKWARD_SPEC" in e for e in rep.errors)


def test_spec_wrt_not_a_forward_param(tmp_path):
    body = dedent('''\
        import torch
        def forward(x):
            return x * 2
        BACKWARD_SPEC = {
            "wrt": ["y"],
            "inputs": {"x": {"shape": [4]}},
            "cases": [{}],
        }
    ''')
    spec = _write_spec(tmp_path, body, "badwrt")
    ws = tmp_path / "workspace" / "badwrt_grad"
    rep = p25b.provision_backward_reference(ws, spec, op="badwrt_grad")
    assert rep.verdict == "SPEC_INVALID"
    assert any("wrt" in e for e in rep.errors)


def test_spec_missing_required_forward_input_is_typed_invalid(tmp_path):
    body = dedent('''\
        import torch
        def forward(x, w):
            return x * w
        BACKWARD_SPEC = {
            "wrt": ["x"],
            "inputs": {"x": {"shape": [4]}},
            "cases": [{}],
        }
    ''')
    spec = _write_spec(tmp_path, body, "missing_input")
    ws = tmp_path / "workspace" / "missing_input_grad"
    rep = p25b.provision_backward_reference(
        ws, spec, op="missing_input_grad"
    )
    assert rep.verdict == "SPEC_INVALID"
    assert any("missing required forward params" in e and "w" in e for e in rep.errors)


def test_explicit_multi_output_forward_is_typed_invalid(tmp_path):
    body = dedent('''\
        import torch
        def forward(x):
            return x * 2.0, x.square()
        BACKWARD_SPEC = {
            "wrt": ["x"],
            "inputs": {"x": {"shape": [4]}},
            "cases": [{}],
            "grad_output": "explicit",
        }
    ''')
    spec = _write_spec(tmp_path, body, "multi_output")
    ws = tmp_path / "workspace" / "multi_output_grad"
    rep = p25b.provision_backward_reference(
        ws, spec, op="multi_output_grad"
    )
    assert rep.verdict == "SPEC_INVALID"
    assert "single-output forward" in " ".join(rep.errors)
    assert not (ws / "model.py").exists()


def test_spec_not_differentiable(tmp_path):
    """wrt input that does not affect the output → NOT_DIFFERENTIABLE."""
    body = dedent('''\
        import torch
        def forward(x, w):
            return x * 2.0   # w unused → grad w.r.t. w is None
        BACKWARD_SPEC = {
            "wrt": ["w"],
            "inputs": {
                "x": {"shape": [4]},
                "w": {"shape": [4]},
            },
            "cases": [{}],
            "grad_output": "ones",
        }
    ''')
    spec = _write_spec(tmp_path, body, "nondiff")
    ws = tmp_path / "workspace" / "nondiff_grad"
    rep = p25b.provision_backward_reference(ws, spec, op="nondiff_grad")
    assert rep.verdict == "NOT_DIFFERENTIABLE", rep.summary


# ════════════════════════ B. orchestrator O2.5 dispatch ═══════════════════════

import orchestrator as _orchestrator_import  # noqa: E402

orch = (
    _orchestrator_import
    if hasattr(_orchestrator_import, "run_single_op")
    else importlib.import_module("orchestrator.orchestrator")
)


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".ascendc_env"
    env_file.write_text(dedent("""\
        TARGET=a3
        A3_HOST=198.51.100.70
        A3_USER=root
        A3_CONTAINER=npu-a3-back
        A3_CANN_PATH=/home/npu_user/cann/cann-9.0.0
        A3_SOC_VERSION=Ascend910_9382
    """))
    from briefs import _common
    monkeypatch.setattr(_common, "DEFAULT_ASCENDC_ENV", env_file)
    return env_file


@pytest.fixture
def stub_phases(monkeypatch):
    monkeypatch.setattr(orch.phase_o0, "check_hook_integrity",
                        lambda ws: MagicMock(verdict="GREEN", summary="stub",
                                             missing_files=[]))
    monkeypatch.setattr(orch.phase_o05, "init_durable_state",
                        lambda workspace, op, **kw: MagicMock(summary="stub o05"))
    monkeypatch.setattr("phase_o17_classify.classify",
                        lambda ws: MagicMock(error=None, op_class_tags=["GRADIENT"],
                                             kb_recommendations=[]), raising=False)
    monkeypatch.setattr(orch.phase_o15, "classify_det_policy",
                        lambda *a, **kw: MagicMock(policy="n_a", summary="stub",
                                                   det_floor=None),
                        raising=False)
    monkeypatch.setattr(orch.phase_o15, "store_in_durable_state",
                        lambda ws, policy, det_floor=None: None, raising=False)
    monkeypatch.setattr(orch, "_sync_benchmark_jsonl_to_workspace",
                        lambda ws, op, env: None, raising=False)
    monkeypatch.setattr(orch.phase_o3, "init_progress_md",
                        lambda ws, op, **kw: MagicMock(summary="stub o3"))
    monkeypatch.setattr(orch.events, "emit", lambda *a, **kw: None)
    # Force the spawn loop to terminal immediately so flow-to-worker paths
    # (B3.3b E2E) return without dispatching a real agent (mirrors the port_a3
    # dispatch test). The 98-boundary tests return before the loop and don't
    # depend on this, but the E2E test flows into it.
    monkeypatch.setattr(orch.state_executor, "snapshot",
                        lambda ws: MagicMock(current_state="done", is_terminal=True))


def _backward_ws(tmp_path, spec_path: Path, op="mul_grad"):
    ws = tmp_path / "workspace" / op
    ws.mkdir(parents=True)
    (ws / ".opgen_state.json").write_text(json.dumps({
        "schema_version": 1, "op": op, "opgen_mode": "backward",
        "backward_forward_source": str(spec_path),
    }))
    (ws / "op_classification.json").write_text(json.dumps({
        "op": op, "op_class_tags": ["backward", "GRADIENT"],
    }))
    return ws


def test_dispatch_routes_backward_returns_98(tmp_path, fake_env, stub_phases, monkeypatch):
    """opgen_mode=backward → run_single_op O2.5 dispatch calls phase_o25_backward,
    produces the reference. With the BACKWARD_E2E=0 opt-out it stops at the B3.2
    reference boundary (98). Real provision (CPU).
    """
    monkeypatch.setenv("BACKWARD_E2E", "0")  # opt-out → stop at reference boundary
    spec = _write_spec(tmp_path, _MUL_SPEC)
    ws = _backward_ws(tmp_path, spec)
    rc = orch.run_single_op("mul_grad", workspace=ws, lane=0)
    assert rc == 98, f"expected B3.2 boundary code 98, got {rc}"
    assert (ws / "model.py").is_file()
    assert (ws / "backward_ref.json").is_file()


def test_backward_skips_o17_classify_when_preseeded(tmp_path, fake_env, stub_phases, monkeypatch):
    """B3.3b live-e2e fix (2026-05-31): backward mode with op_classification.json
    pre-seeded by _cmd_backward MUST NOT invoke the /aog-op-classify subprocess
    (phase_o17_classify.classify). The --backward CLI flag IS the classification;
    re-running the skill is redundant + flaky (returns a clarifying question for a
    bare forward-spec → pauses the live orch --backward run).
    """
    monkeypatch.setenv("BACKWARD_E2E", "0")  # stop at reference boundary → deterministic rc=98
    classify_mock = MagicMock(
        side_effect=AssertionError("phase_o17_classify.classify must NOT be called "
                                   "for a pre-seeded backward workspace"))
    monkeypatch.setattr("phase_o17_classify.classify", classify_mock, raising=False)
    spec = _write_spec(tmp_path, _MUL_SPEC)
    ws = _backward_ws(tmp_path, spec)  # seeds .opgen_state(backward) + op_classification(tags)
    rc = orch.run_single_op("mul_grad", workspace=ws, lane=0)
    assert rc == 98, f"expected B3.2 boundary code 98, got {rc}"
    classify_mock.assert_not_called()  # O1.7 skip is BEFORE the O2.5 BACKWARD_E2E branch


def test_backward_runs_o17_classify_when_not_preseeded(tmp_path, fake_env, stub_phases, monkeypatch):
    """Guard the skip condition: backward workspace WITHOUT a pre-seeded
    op_classification.json (op_class_tags) must still fall through to the classify
    skill (so the skip is gated on the pre-seed, not on mode alone).
    """
    called = {"n": 0}

    def _track(ws):
        called["n"] += 1
        return MagicMock(error=None, op_class_tags=["GRADIENT"], kb_recommendations=[])
    monkeypatch.setenv("BACKWARD_E2E", "0")  # stop at reference boundary → deterministic rc=98
    monkeypatch.setattr("phase_o17_classify.classify", _track, raising=False)
    spec = _write_spec(tmp_path, _MUL_SPEC)
    ws = _backward_ws(tmp_path, spec)
    (ws / "op_classification.json").unlink()  # remove the pre-seed → must re-run classify
    rc = orch.run_single_op("mul_grad", workspace=ws, lane=0)
    assert rc == 98
    assert called["n"] == 1, "classify should run when op_classification.json is absent"


def test_dispatch_backward_missing_source_returns_7(tmp_path, fake_env, stub_phases):
    ws = tmp_path / "workspace" / "x_grad"
    ws.mkdir(parents=True)
    (ws / ".opgen_state.json").write_text(json.dumps({
        "op": "x_grad", "opgen_mode": "backward",  # NO backward_forward_source
    }))
    rc = orch.run_single_op("x_grad", workspace=ws, lane=0)
    assert rc == 7


def test_dispatch_backward_spec_invalid_returns_7(tmp_path, fake_env, stub_phases):
    spec = _write_spec(tmp_path, "import torch\ndef forward(x):\n    return x\n", "bad")
    ws = _backward_ws(tmp_path, spec, op="bad_grad")
    rc = orch.run_single_op("bad_grad", workspace=ws, lane=0)
    assert rc == 7  # SPEC_INVALID → block
    assert not (ws / "model.py").exists()


# ════════════════════ B3.3b: end-to-end is the DEFAULT (flag flipped 2026-05-31) ════════════════════

def test_dispatch_backward_default_flows_to_worker(tmp_path, fake_env, stub_phases, monkeypatch):
    """B3.3b DEFAULT (flipped 2026-05-31 after increment-2 #313 + full cold-start e2e
    to archive): with BACKWARD_E2E UNSET, the O2.5 backward branch produces the reference
    then FLOWS into the worker loop end-to-end (does NOT return 98). stub_phases forces
    terminal → run_single_op returns 0, i.e. it passed the O2.5 boundary into the worker.
    """
    monkeypatch.delenv("BACKWARD_E2E", raising=False)  # default = end-to-end
    spec = _write_spec(tmp_path, _MUL_SPEC)
    ws = _backward_ws(tmp_path, spec)
    rc = orch.run_single_op("mul_grad", workspace=ws, lane=0)
    assert rc == 0, f"expected flow-to-worker (0 via stub terminal), got {rc}"
    assert (ws / "model.py").is_file()  # reference still produced en route


def test_dispatch_backward_e2e_1_still_flows(tmp_path, fake_env, stub_phases, monkeypatch):
    """BACKWARD_E2E=1 (explicit, legacy) still flows end-to-end (only =0 opts out)."""
    monkeypatch.setenv("BACKWARD_E2E", "1")
    spec = _write_spec(tmp_path, _MUL_SPEC)
    ws = _backward_ws(tmp_path, spec, op="mul1_grad")
    rc = orch.run_single_op("mul1_grad", workspace=ws, lane=0)
    assert rc == 0, f"expected flow-to-worker (0 via stub terminal), got {rc}"


def test_dispatch_backward_optout_0_returns_98(tmp_path, fake_env, stub_phases, monkeypatch):
    """BACKWARD_E2E=0 opt-out → stop at the B3.2 reference boundary (98), no worker."""
    monkeypatch.setenv("BACKWARD_E2E", "0")
    spec = _write_spec(tmp_path, _MUL_SPEC)
    ws = _backward_ws(tmp_path, spec, op="mul2_grad")
    rc = orch.run_single_op("mul2_grad", workspace=ws, lane=0)
    assert rc == 98


def test_expected_truth_source_backward(tmp_path):
    """B3.3b: phase_o5 recognizes the backward self-contained truth source."""
    import phase_o5
    ws = tmp_path / "x_grad"
    ws.mkdir()
    (ws / ".opgen_state.json").write_text(json.dumps({"opgen_mode": "backward"}))
    assert phase_o5.expected_truth_source(ws) == "backward_autograd"
    # Migration has its own fresh source-NPU truth.
    (ws / ".opgen_state.json").write_text(json.dumps({"opgen_mode": "port_a3_to_a5"}))
    assert phase_o5.expected_truth_source(ws) == "a3_cann"
    (ws / ".opgen_state.json").write_text(json.dumps({"opgen_mode": "unsupported"}))
    with pytest.raises(RuntimeError, match="cannot resolve supported truth source"):
        phase_o5.expected_truth_source(ws)
