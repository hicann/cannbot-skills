# Long-Run State and Context

This is an optional reference for a workflow that must pause, compact context,
or coordinate scouts. Do not read it at task startup. The owning playbook's one
task-local state file remains authoritative.

## Keep one resume capsule

The capsule should answer only:

- route, contract, and active phase;
- current candidate path/hash/count;
- passed gates;
- first failing boundary, fix count, command, and evidence path;
- ruled-out alternatives;
- active background task/scout ids;
- one next action.

Keep source, candidate revisions, generated code, evaluator output, and full
logs in separate files. Never paste them into the capsule or checkpoint.

Update the capsule and runtime checkpoint before a long background task,
candidate architecture change, pause, resume, or context compaction. Continue
from the capsule rather than replaying raw investigation history.

## Close resolved phases

At contract freeze, reference/probe completion, candidate replacement, resolved
failure, and public validation:

1. preserve source/logs and hashes;
2. update the capsule and ruled-out alternatives;
3. end or summarize resolved exploration;
4. compact only complete turns/tool exchanges;
5. continue with the recorded next action.

Do not compact an unresolved blocker, current design decision, recent candidate
change, or half-finished tool exchange.

## Keep agent topology small

One parent owns the contract, architecture, candidate writes, validation order,
and final claim. Use no scout by default. Add at most one read-only scout for a
separate reference/API or log question; use a second only when the two evidence
domains are independent.

Scouts return a conclusion, evidence paths/line references, searches or
commands, ruled-out alternatives, and the unresolved boundary. They do not
return raw transcripts or edit the candidate.

Use tools, not agents, for deterministic tests, hashing, JSON parsing, case
generation, and log collection.

## Background tasks and sessions

Run long compile/simulator/hardware commands with completion notification. Store
task id, command, status/output paths, and expected completion in the capsule;
continue independent work or park instead of polling.

Prefer one parent session for one contract. Start a fresh process/session only
when isolation/private exposure changes, the contract/task changes, the model
configuration changes, or the capsule can no longer reconstruct the work
without stale assumptions.
