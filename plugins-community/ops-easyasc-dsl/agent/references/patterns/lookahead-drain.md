# Lookahead, Steady-State, and Drain Scheduling

## Applies when

Use this pattern when several pipeline stages operate on different logical work
items in the same outer iteration, so a producer runs ahead of one or more
delayed consumers.

## Logical dataflow

For an `N`-stage pipeline, stage `d` consumes work item `pipe_work - d`.
Execution has three visible phases:

```text
warmup -> steady state with overlapped stages -> drain
```

## Physical invariants

- write each stage delay explicitly before choosing a buffer family;
- size DBuff/TBuff/QBuff from the number of overlapping beats and simultaneous
  live roles, not from source-code proximity;
- keep producer and delayed-consumer counters separate;
- a final stage reading producer-loaded storage extends that storage lifetime
  through the final stage, even if an intermediate stage uses the same work id;
- mutex depth and physical slot rotation must describe the same in-flight
  topology.

## Minimal skeleton

```python
for pipe_work in range(work_begin, work_end + (N - 1)):
    if pipe_work < work_end:
        produce_stage0(pipe_work)
    if (pipe_work > work_begin) and (pipe_work < work_end + 1):
        consume_stage1(pipe_work - 1)
    if pipe_work > work_begin + 1:
        finalize_stage2(pipe_work - 2)
```

For a one-tile lookahead, the compact form is:

```python
for step in range(0, steps + 1):
    if step < steps:
        produce(step)
    if step > 0:
        consume(step - 1)
```

## Failure signatures

- final output misses the last tile: the drain iteration is absent;
- a reused buffer is overwritten only with multiple work items per core: slot
  depth was chosen from conceptual tensors rather than live beats;
- mutex events remain balanced but values are stale: semaphore depth exceeds
  actual physical slot rotation;
- changing source order or adding a broad barrier appears to help: the true
  delayed lifetime is still not represented in counters and slots.

## Runnable references

- `agent/references/patterns/a2-mixed-pipeline.md`: GM-bridge two/three/four
  stage schedules and delayed consumers.
- `agent/references/patterns/a5-mixed-pipeline.md`: direct on-chip lookahead
  schedules.
- `agent/example/kernels/a2/attention/flash_attn_score_pv.py`: one-tile A2 delayed consumer.
- `agent/example/kernels/a5/attention/mha_ifa.py`: A5 producer/consumer lookahead and drain.
- `agent/references/constraints/sync.md`: mutex depth, events, barriers, and
  buffer-lifetime legality.

## Do not use when

- stages consume the same work item synchronously and no lifetime crosses an
  outer iteration;
- the only dependency is inside one VF body; use `vf_barrier` constraints;
- the task is merely to increase mutex depth without a proven overlapping
  schedule.

## Source escape

When event ordering or generated control flow disagrees with the schedule,
follow `agent/references/evidence-escalation.md`. Inspect split instructions and
generated C++ for the first missing or extra producer/consumer edge before
changing buffer depth.
