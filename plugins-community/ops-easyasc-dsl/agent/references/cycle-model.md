# Simulator Cycle Model

The checked-in model is a repository conclusion. This page describes its owners
and use; it intentionally does not document calibration history or procedure.

## Owners

- A2 parameters: `easyasc/simulator/timing/a2_cycle_model.json`
- A5 parameters: `easyasc/simulator/timing/a5_cycle_model.json`
- loading and estimation: `easyasc/simulator/timing/cycle_model.py`
- regression coverage: `agent/example/testcases/simulator/timing/test_cycle_model_datamove.py`

The JSON profiles own numerical parameters. Do not duplicate parameter tables in
agent documentation.

## Meaning and scope

With `SimulatorConfig(cycle_model_enabled=True)`, the simulator estimates task
cycles and pipe scheduling for the selected device profile. Trace output can be
used to inspect modeled start/end cycles, makespan, and pipe overlap. These are
analytical estimates for comparison and regression; they are not proof of
real-device latency, compilation, or bit-exact behavior.

The model covers the operations represented by its profile and estimator.
Unknown or incompletely modeled instructions may use a fallback estimate. Cache,
contention, synchronization visibility, and cross-pipe overlap are represented
only to the extent implemented in `cycle_model.py` and the selected JSON. Treat
large model/device disagreement as a model limitation to investigate, not as a
kernel correctness result.

## Use and regression

Cycle modeling is enabled by default in `SimulatorConfig`. Disable it for a
functional-only run with:

```python
SimulatorConfig(cycle_model_enabled=False)
```

Verify the checked-in behavior with:

```bash
pytest -q agent/example/testcases/simulator/timing/test_cycle_model_datamove.py
```

When parameters or estimator logic change, update the owning JSON or Python
source and add a focused regression. Hardware validation may be required for a
new conclusion, but temporary probes and machine-specific artifacts are not
tracked documentation dependencies.
