# Buffer Slot Lifetime and Role Lift

## Applies when

Use this pattern when a local tensor is reused while an earlier pipe still
reads it, or when several delayed beats and several local roles overlap. This is
the composition layer above mutex/event constraints.

## Logical dataflow

```text
producer beat i writes role A
consumer pipe still reads role A
later beat or stage wants the same-shaped storage for role B
-> assign distinct physical slots until both lifetimes retire
```

## Physical invariants

- required slots are at least
  `(overlapped beats) * (simultaneously live roles)`;
- `Tensor`, `DBuff`, `TBuff`, and `QBuff` provide one, two, three, and four
  physical slots respectively;
- mutex depth describes in-flight handoff credits, not local role separation;
- raise mutex depth only when the explicit handoff also rotates through the
  additional slots;
- source order, one `auto_sync()` region, or `bar_all()` is not proof that a
  specific MTE3/FIX/V source has retired;
- L0C reuse from a FIX consumer back to M may require an explicit
  `SEvent(Pipe.FIX, Pipe.M)` even when M -> FIX ordering already exists.

## Minimal skeleton

Two overlapping roles in one beat:

```python
storage = DBuff(DT.float, [L, D], Position.UB)
output_src = storage[0]
later_scratch = storage[1]

out[...] <<= output_src[0:rows, 0:D]
l0c_to_ub(later_scratch, l0c_value, ...)
```

Two beats with two live roles require four slots:

```python
storage = QBuff(DT.float, [M, N], Position.L0C)
base_slot = work_idx * 2
tmp_slot = base_slot + 1
base = storage[base_slot]
tmp = storage[tmp_slot]
```

Do not replace the QBuff with DBuff in the second form: modulo wrapping maps the
next beat back onto the previous beat's still-live roles.

## Failure signatures

- one active core passes one tile but fails when it reuses the same slot family;
- mutex events are balanced yet values are stale or overwritten;
- adding a broad barrier changes the symptom but not the ownership model;
- output writeback is corrupted by a later scratch fill of the same base tensor;
- raising mutex depth makes correctness worse because local storage depth did
  not rise with handoff credits.

## Runnable references

- `agent/example/kernels/a2/attention/flash_attn_full.py`: QBuff score/P/expdiff families for
  grouped delayed lifetimes.
- `agent/example/kernels/a5/attention/mha_ifa.py`: TBuff K/V families for lookahead beats.
- `agent/references/patterns/lookahead-drain.md`: explicit delayed-stage timing.
- `agent/references/constraints/sync.md`: event, mutex depth, and pipe legality.
- `doc/api/tensor_buffer.md`: physical slot-family definitions.

## Do not use when

- two names have disjoint proven lifetimes and one physical slot is sufficient;
- the hazard is inside a single VF register recurrence rather than local-memory
  pipe overlap;
- the buffer is a GM workspace whose ownership requires a separate cross-core
  or cross-lane contract;
- capacity cannot afford the lifted family; restructure the schedule instead of
  silently reusing live storage.

## Source escape

When the last consumer is unclear, follow
`agent/references/evidence-escalation.md`. Inspect split instructions, generated
event ordering, and a multi-reuse probe before choosing slot depth or mutex
depth.
