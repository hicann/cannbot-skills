---
name: pypto-op-perf-panko-manual
description: Stage 7 PANKO world-model + evaluator contract. Run by the search executor. Defines evaluation, refinement and the CODE/STOP handoff over the persisted search tree. No API key. Use when the user asks for PANKO search-based auto-tuning in Stage 7.
---

# pypto-op-perf-panko-manual — Stage 7 PANKO (world model + evaluator)

## Search cycle

You are the **World Model** (propose δ, reflect, rescore V, Insert/Update/Prune) **and the
Evaluator** (run the frozen `E(x)→(s,p)`). The deterministic core — selection (argmax
`pypto_action_priority`, short symbol `V_pypto` = our composite `w_stage·prior_gain·novelty·V`;
V is PANKO's per-node value),
stagnation, `J = s·(P_ref/p)·100`, tree-edit application, stop conditions, state I/O — lives in
`scripts/panko_harness.py`. You **call** it; you do not re-implement its decisions.

The world model is the **persisted tree** in `custom/<op>/optimization/search_state.json`, not
this session. Each cycle may resume the same task-scoped optimizer conversation: **read the state → apply one transition →
persist → return a directive.** Therefore every V update / prune MUST carry its reason
(`--edits ... "reasons"`); a later dispatch re-hydrates the belief from your log.

## Optimization principles — how to value actions (raise `V`)

Beyond the catalog `prior_gain`, raise an action's **`V`** (the belief factor of
`pypto_action_priority = w_stage · prior_gain · novelty · V`) by how strongly its intent serves the
two principles that make a pypto kernel faster and lift utilization — and **propose (insert) new
intents** along these axes:

1. **Minimize data copy_in / copy_out** (Global-Memory traffic). Keep intermediates on-chip
   (L1 / UB) and never round-trip them through GM: fuse producer→consumer ops, reuse already-loaded
   data (L1Reuse), drop redundant loads/stores.
2. **Reduce computation complexity.** Fewer / cheaper ops: compute-once-and-reuse, skip tail
   zero-pad compute (`valid_shape`), simplify algebra, shrink effective work.

**Utilization ratio — overlap:** pipeline **MTE data-movement ∥ Cube compute ∥ Vec compute** so they
run **concurrently** (double-buffer / nbuffer / CubeNBuffer / `submit_before_loop` / stitch), filling
the idle **bubble** and lowering latency.

Each seed's **`prior_gain`** already encodes how **directly** it serves these principles — strong lever
(fuse / keep-on-chip, cut ops, MTE∥Cube∥Vec overlap) ≈ 0.9, moderate ≈ 0.6, weak / general tuning
≈ 0.3 — so principle-aligned actions are prioritized by the harness. Weight `V` the same way, and give
every inserted intent a `prior_gain` by this same principle-directness scale.

## References (load as needed)
- `references/evaluator.md` — the frozen `E(x)→(s,p)` you must run (fixed commands/thresholds).
- `references/search_acceleration.md` — evaluator modes, cache validity and worker lifecycle.
- `references/state_schema.md` — `search_state.json` + `<op>_optimization.md`.
- `references/action_catalog.json` — stage-tagged seed intents (used at INIT).
- `references/tunable_constants.md` — design note: extending INIT normalisation
  from tile calls to the constants that drive loop and view extents. Not
  implemented yet; read it before touching `bayesian_optimization.normalize` or `bayesian_optimization.apply`.

**Two-directional actions.** A `"directive": true` entry names a compiler or
runtime option that is either on the program or absent from it, and every one of
them has a counterpart carrying `"counterpart_of"` that removes it. The removal
is not decorative: on one kernel removing the directive is the larger win and
on another setting it is a heavy regression, so neither direction is safe to
assume. Treat the pair as one axis and let the device pick the direction.

**Held actions.** At INIT the harness splits the catalog into `open` and `held`.
An action is held when the kernel does not currently satisfy its `requires`
(no matmul, no broadcast, no chunked scan, single module). Held actions are
never deleted: after every improvement the harness recomputes the kernel's
structure and returns any that now qualify, so a transformation that introduces
a matmul re-admits the whole cube family. Do not try to work around the split;
if you believe an action should be reachable, propose the transformation that
creates its precondition (F-20 does this for reductions).

**Measurement scope.** `p` measures the AICore End-to-End Time of the JIT kernel, including
work it delegates to device-only helpers. Keep host code frozen so candidate measurements
remain comparable. The [evaluator contract](references/evaluator.md) defines which edits
the controller accepts.

**The chip envelope is derived, not assumed.** `init` reads the buffer capacities
(UB / L0A / L0B / L0C / L1) and the cube/vector core counts out of the platform ini
the installed CANN ships — `<cann_home>/<arch>/data/platform_config/<SoC>.ini`, the
SoC resolved from the device — and every static rule in the run validates against
that one envelope. It **refuses and asks you** when it cannot derive them; it does
not fall back to another chip's numbers, because those reject legal tiles before any
device sees them. `--chip-envelope '<json>'` supplies them deliberately and is
recorded as such. The ini path used is recorded in the state and echoed by `init`.

**And re-checked against the device before the first evaluation.** The SoC is resolved
*for* `--device`, not for the box, and the name the envelope was built from is asked
again before any block measures anything. A different chip in the slot **halts and asks
you** — re-initialise here, or point PANKO at the chip it was initialised for. `init`
itself never refuses on this (it must stay runnable with no device attached), but it
reports the mismatch the moment it sees one. Where no runtime can be asked, the run
proceeds and the state records `confirmed: unknown`, so a report can say whether the
numbers were checked against silicon or only against a name.

Checkpointed searches use `evaluation_session.py --mode subprocess` for the baseline and
`block --evaluation-mode subprocess` for BO. The measurement process enables PyPTO
profiling before loading the kernel. Keep this mode for the entire search.
Use the operator's representative inputs in `eval/panko_inputs.py` or the
existing `eval/test_inputs.py`. Expose `build_inputs(device)` returning positional
wrapper arguments or `INPUTS`. The device argument is the selected NPU index,
so torch inputs use `torch.device("npu", int(device))`. Reuse the case generator,
scalar arguments and workload specified by the user. See `references/evaluator.md`.

Use `optuna >= 2.7` for `Study.ask(fixed_distributions=...)` and Python 3.10-compatible NumPy dependencies.
`init` runs a capability preflight and REFUSES when
`bayesian_optimization_tile_search` is on and optuna is not importable: tile values
are the one lever the model cannot propose, so running without it is a different
method rather than a degraded one.

⛔ **That refusal carries `ask_user`, and it is a question for the USER.** Relay
`ask_user.question` with its `options` and stop. Do not install anything, do not
re-run `init` with the tile search switched off, and do not fall back to the
stepwise path on your own — which of those three the run should be is a decision
about what is being measured, and it is not yours. Resume only on the user's
answer.

Every `init` reports `bayesian_optimization: {available, optuna_version, reason}`
and persists it, so a report names the algorithm that ran rather than the one that
was configured.

Set `SKILL_ROOT` to the absolute directory containing this `SKILL.md`. Resolve `scripts/` and `references/` from that directory.

Let `HARNESS = python <SKILL_ROOT>/scripts/panko_harness.py`.

Run the baseline with `python <SKILL_ROOT>/scripts/evaluation_session.py --mode subprocess --op-dir <op_dir> --op <op> --op-file <op_file> --device <DEVICE> --test-command '<test_command>' --lint-root <lint_root> --eval-timeout-s <eval_timeout_s>`. It runs the tests, lint and frozen measurement. STEP describes candidate evaluation. For searches without checkpoints, a caller using the resident worker passes `--mode resident` and enables profiling on its JIT kernel. A terminal `HARNESS stop` closes the worker. On abort, run `python <SKILL_ROOT>/scripts/evaluation_session.py --close --op-dir <op_dir>`.

For searches without checkpoints, select `--mode cached` (complete results and exact-source static failures) or `--mode prepared` (also guarded deterministic inputs and pure golden references). Before a new INIT's first measurement, call `evaluation_session.py --reset --op-dir <op_dir>` once; never reset a resumed STEP. It starts a new cache generation and removes stale terminal markers while preserving prior evidence. Use the same mode for BO's `--evaluation-mode`. Tests, references, SPEC, lint, configuration, runtime versions, host and environment invalidate the cache. Generated tuning notes do not. Set `PANKO_RUNTIME_SIGNATURE` when replacing runtime binaries without changing their version metadata or changing test dependencies outside the operator directory. Prepared mode requires deterministic tests and a pure `<op>_golden` in `<op>_golden.py`; unsupported fixtures keep fresh preparation. Tensor contents and versions are checked before reuse, including reference outputs.

For checkpointed searches, or to retain harness imports and API metadata, set `HARNESS = python <SKILL_ROOT>/scripts/harness_session.py --op-dir <op_dir> --`. Its arguments and stdout remain the original harness contract. It checks for a natural stop after `init`, `record`, and any `block` that has no winner to record. A natural stop restores the best source, runs the report, and writes `optimization/controller-terminal.json`. Before returning STOP, verify that the terminal's `run_start`, `reason` and `delivered_code_hash` agree with the persisted run and the restored source. On abort, call `harness_session.py --op-dir <op_dir> --close` to release both owned workers. Its `--timeout` bounds a complete harness command (default 3600 seconds). Set it to the command's BO time budget.

If the client exposes a resumable optimizer task/session ID, reuse it only within the same operator directory, role and search configuration; keep the coder's role separate. Always reload `search_state.json` and the current implementation before a resumed cycle. Start a new conversation when that scope changes. Context reuse does not replace the persisted state or bypass any transition.

The action catalogue resolves itself: **omit `--catalog`** and the harness reads
`<SKILL_ROOT>/references/action_catalog.json`, wherever that is. Pass `--catalog`
only to point at a different catalogue. `init` echoes `skill_root` and `catalog`
so what it resolved is on the record.

## Inputs (from the dispatch prompt)
`op`, `op_dir=custom/<op>`, `DEVICE=TILE_FWK_DEVICE_ID`, `op_file`, `test_command`, and either:
- **INIT cycle** (first dispatch — no `candidate_file` given): also `P_ref`, `config`
  (`stagnation_K` / `staged_prior` / limits).
- **STEP cycle** (`candidate_file` + `action_id` given): the candidate the coder just produced.

For a fresh search, start at INIT. For a checkpointed search with an existing
`optimization/search_state.json`, check for a natural STOP or a measured BO winner awaiting
`record` first (see STEP). If `progress.pending_code` exists, `select` returns that handoff.
An acknowledged hash enters STEP directly. If the manual edit is outstanding, inspect the
current bytes against `base_hash` and acknowledge the selected edit before measuring it.
If `progress.refine` still has an action and no CODE is pending, continue that refinement.
Use the last controller response when available. If it was lost, the `close` gate checks the
persisted counters. If it refuses an early close, use `handoff` to continue the same action.
For a caller passing individual cycles without checkpoints, `candidate_file` selects STEP.

## Code snapshots & greedy accumulation (harness-owned — do NOT restore/snapshot by hand)
Node code lives under `custom/<op>/optimization/nodes/<code_hash>.py`. **The harness enforces greedy
accumulation deterministically**, so every action stacks on the running global best without you copying
any files:
- `init` (pass `--op-file <op_file>`) snapshots the preopt `op_file` → `nodes/<preopt_hash>.py`.
  The hash is **computed from the file**; `--preopt-hash` is optional and, if given, is verified
  against those bytes. `init` **refuses** if `op_file` cannot be read, if a supplied hash disagrees
  with it, or if the snapshot cannot be written — a run whose baseline is missing has nothing for
  later cycles to stack on, and would silently accumulate onto whatever is in the working file.
- Selecting a **new** action restores the running global best into `op_file` (`restore_code_hash` /
  `restored`). Resuming an existing `pending_code` leaves its source in place.
  If that restore fails and the file is not already the best, `select` **refuses** rather than hand out
  an action: the response names the program the δ is to be applied to, and it has to be that program.
- `record` snapshots a **kept** candidate → `nodes/<code_hash>.py`; on a **reverted** candidate it rolls
  `op_file` back to the action's current best (or the global best) automatically.
- `close` leaves `op_file` at the action's best; `stop` restores the global-best `op_file` for delivery
  and **refuses to record a stop it cannot deliver** — `stop_reason` is sticky and the Stage 7 gate
  reads it, so writing it over some other program would complete Stage 7 on a kernel the search
  never chose.

So `best_latency_us` trends down monotonically and the best code can never be lost. A candidate slower
than the running best simply regressed **on top of** the best — the harness reverts it. Your only file
action is to have the coder **apply the δ** to `op_file`. Insert new intents with parent defaulting to
`best_node`.

**This is greedy hill climbing, and that is deliberate — say so rather than implying otherwise.**
The tree in `search_state.json` is a tree of *intents*. The reachable program states are not a tree:
they are a single non-decreasing chain. A candidate is kept only if it beats the bar (the global best,
or the action's own best once the action has produced one, which was itself ≥ the global best when it
was set), so **no program that is worse than the running best is ever built on**. The consequence
follows directly: a route that must get *worse* before it gets better is out of reach, and PANKO does
not claim to search one.

Two things do reach combinations, and they are the only two:

- **Within one δ.** An action may apply several changes at once and be judged as one step. `S-8`
  (L1Reuse + CubeNBuffer) is exactly this, and it is the pattern to use whenever two changes are only
  jointly profitable — `F-8` names the `F-7` chunk in its own δ for the same reason. A pair split
  across two catalogue entries is a pair the ratchet can lose: the first half regresses, is reverted,
  and the second half never sees it.
- **Within one tile block.** The tile search proposes all Cube and Vector tile parameters *together*
  (TPE over the joint space), not one axis at a time, so combinations inside that space are reached
  without the ratchet being involved.

Everything else is one change at a time, kept or reverted on its own measurement. If a campaign needs
a genuinely non-monotone search, that is a different optimizer, not a setting here.

## Cycle — INIT
> **Maximize mode:** when no target latency is given, the orchestrator passes a small floor `P_ref`
> (a normalizer, not a target). The `J≥100` success stop then never fires by construction, so the run
> ends on the safe stops (`wall_clock_limit_s` + `global_stagnation_limit`). Report progress by
> `best_speedup_vs_preopt` / `best_latency_us`, not `J`.

0. **Normalize the kernel, before anything is measured.** `bayesian_optimization.apply` tunes a tile call only when
   every argument is an integer literal. A kernel written `pypto.set_vec_tile_shapes(TILE_B, TILE_D)`
   is classified `vec_dyn`, `tunable_sites()` returns nothing, and the tile lever produces **zero**
   trials for the whole run — whatever else is configured. Exposing that parameter is what moves the
   tile from your judgment to the core's arithmetic, and it is the one step that cannot be done later:
   submitted as a search candidate a performance-neutral refactor measures the same as the incumbent,
   is reverted as unimproved, and its bytes are burned into the eval cache so it can never be
   re-issued.

   - `cp <op_file> <op_file>.preopt` first. You need the original to check against and to restore.
   - Rewrite so that **every** `set_vec_tile_shapes` / `set_cube_tile_shapes` argument is an integer
     literal. Where one name drives both a tile and a view extent, a loop bound or the host wrapper's
     padding, give the view its own name and put the literal at the tile call. Every binding site
     moves together or none does.
   - `HARNESS normalize --before <op_file>.preopt --after <op_file> --op-dir custom/<op> --op <op>`.
     On `ok: false` the reasons name what failed; fix and re-run, or restore `.preopt` and continue
     without normalizing. **Never** proceed on a failed verdict.
   - Then run `E(op_file)` and confirm it is correct and that its latency matches `E(<op_file>.preopt)`
     within the noise floor. A normalization that made the kernel *faster* changed a value; restore
     `.preopt` and try again.

1. Run the frozen eval `E(preopt)` → `(s0, p0)` on the **normalized** kernel (evaluator.md; wrap the
   run in `timeout <eval_timeout_s>`). If `s0 == 0`: return `{"directive":"STOP","reason":"preopt_incorrect"}`.
2. `HARNESS init --op-dir custom/<op> --op <op> --p-ref <P_ref> --device <DEVICE> --config '<json>' --preopt-s <s0> --preopt-p <p0> --op-file <op_file>`
   (no `--catalog`: it defaults to this skill's own. `--preopt-hash` is optional
   and verified against `op_file` if given.)
   - If `E(preopt)` reported core-util % / bubble %, pass them as `--preopt-util <u0> --preopt-bubble <b0>` — the harness then biases the frontier toward the actions that fix the current bottleneck (deterministic symptom → V boost). The harness re-applies this at every `close` from the measured metrics, so the frontier always points at the *current* bottleneck. **Required whenever `symptom_boost > 1`**: without them the harness cannot infer a symptom and profile anchoring silently does nothing. `init` returns a `warning` field when they are missing.
3. Check for a natural STOP before selecting a candidate. The baseline spends one
   evaluation and may already meet the target or exhaust the budget. With
   `harness_session.py`, read the terminal record after INIT. When running
   `panko_harness.py` directly, issue `stop` and run `report` if it returns `stop:true`.
4. Select the next action with `HARNESS select --op-dir <op_dir>`. A new selection starts
   from the running best. A CODE directive carries the implementation path, action ID and intent.
   - To save handoff checkpoints, set `checkpoint_handoffs=true` at INIT or run `select --resume`
     on the first selection. For a checkpointed manual edit, read `progress.pending_code`, check
     `base_hash` and apply the recorded intent. Acknowledge the edit before STEP.

     ```sh
     HARNESS ack-code --op-dir "$OP_DIR" --parent "$ACTION_ID" --base-hash "$BASE_SHA256" \
       --intent "$STORED_INTENT" --summary "$CHANGE_SUMMARY"
     ```

     For a controller-written candidate (`coder: false`), enter STEP with `pending_code.candidate_hash`.
   - Without checkpoints, dispatch the manual edit with `selected.id` and `selected.delta`.
     If `bayesian_optimization.applied` is true, enter STEP directly with `bayesian_optimization.code_hash`.

## Cycle — STEP
Check the persisted state for a terminal record before starting or resuming a
candidate. If a controller command returns `error: stop_required`, call
`HARNESS stop --op-dir custom/<op>` and relay its decision before evaluating
again. When `block` returns a winner, record its `code_hash` and `best_latency`
without another evaluation. If it returns no winner, check for STOP before
continuing.
If a resumed command returns `error: block_winner_pending`, call `record` with
its `pending_record` values. The controller restores that measured candidate
from its snapshot if needed. Check for STOP after recording it.

0. **Check static feasibility first**: `HARNESS feasible --op-dir custom/<op> --parent <action_id> --op-file <candidate_file>`.
   On `feasible: false` the candidate never reaches the device: it costs no `eval_budget`, the harness has already
   rolled the working file back, and `stop_refine` tells you whether the action is now exhausted. Skip `probe` and
   `record` entirely and propose the next concrete δ (or close the action). Do not argue with the rejection: the
   reason names the rule and the numbers.

   **`retune: true` in the same response.** The δ you just applied moved the tile space: the view's
   extents, the dtype or the residency changed, so the tile this program inherited was chosen for a
   space that no longer exists. Run `block --parent <action_id>` **before** `record`, so the
   candidate the ratchet judges is the (structure, tile) **pair**.

   This is not a nicety. A structural change submitted at its stale tile measures worse, is reverted
   as unimproved, and its bytes are burned into the eval cache — so the combination that would have
   won becomes unreachable for the rest of the run. A hand-optimised kernel is typically a structural
   rewrite **and** the tile that suits it; neither half wins alone, and a search that judges one
   variable at a time cannot reach it at all.

   A rename, a comment or a new tile value leaves the space alone and `retune` stays false. If you
   believe the optimum moved anyway — a rewrite that changed the access pattern without changing any
   extent — you may ask for a block yourself. That block's outcome decides the action's fate:
   the core owns the mechanical detection, you own the judgement call and you pay for it.
   - **A rejection does NOT advance `n`.** `n` counts measurements that failed to improve, and the gate never ran
     the program, so nothing was learned about the action. Rejections are counted in `n_static` against
     `infeasible_K` (16), a separate and much looser bound that exists only so an action whose whole neighbourhood
     is infeasible still terminates. When it fires, the response carries `stop_cause: "infeasible_K"`.
   - **So do not treat a rejection as a signal to give up on the action.** It is a signal about one number. The
     reason string names the binding constraint — read it and move the candidate to the other side of it. A
     `UB estimate 264KB (view 256KB + 2x tile 4KB) > budget 192KB` says the *view* is what does not fit, so
     shrinking the tile cannot help and enlarging it makes the message worse.
1. For an ordinary candidate that passes feasibility, use the path selected at INIT.
   - With `config.checkpoint_handoffs=true`, pass every acknowledged candidate to this command,
     including hashes already in the cache:

     ```sh
     python3 "$SKILL_ROOT/scripts/evaluate_candidate.py" --op-dir "$OP_DIR" \
       --parent "$ACTION_ID" --code-hash "$CANDIDATE_SHA256" --device "$DEVICE" \
       --test-command "$TEST_COMMAND" --lint-root "$LINT_ROOT"
     ```

     A cache miss runs the frozen evaluation with `config.eval_timeout_s` and saves the result
     before `record`. A completed `s=0`, including a candidate timeout, spends one evaluation.
     An environment fault leaves the candidate pending. A retry reuses a receipt only when the
     run, source hash and evaluation settings match. A cache hit for the current handoff goes
     through the controller's duplicate handling, which restores the incumbent at zero cost.
   - Without checkpoints, use `HARNESS probe --op-dir <op_dir> --code-hash <sha256>` first.
     On a hit, reuse the cached result. On a miss, run the frozen `E(candidate_file)` and pass
     `--parent <action_id> --code-hash <sha256> --s <s>` to `HARNESS record --op-dir <op_dir>`.
     For a successful evaluation also pass `--p <latency>` and available utilization and bubble metrics.
2. Follow the controller's `record` decision. A retry whose action and hash no longer match
   the current handoff returns the cached result or STOP.
   - Check for STOP after `record` before applying `stop_refine`. Read the terminal
     record with `harness_session.py` or call `stop` with the bare CLI. If the run
     stopped, return its reason and delivered best. Run `report` only with the bare CLI.
   - A response carrying `duplicate: true` means those bytes were already measured; nothing was
     recorded and `n` is unchanged. Propose a distinct variation for the open action or close it when eligible.
   - The harness snapshots a kept candidate and rolls a reverted one back to the running best **for you** — no manual file ops. `improved` here means it **beat the running global best** (a candidate that only beats this action's own local best but is still slower than the global best is a regression → reverted, `n++`).
   - If `stop_refine` is false and the controller has already written the next tile candidate,
     continue with its `coder: false` directive. Otherwise propose a concrete variation of the
     current action. With checkpointed handoffs, use
     `HARNESS handoff --op-dir <op_dir> --parent <action_id> --intent "<next variation>"`
     to persist it before handing it to the editor.
     Use `evolve` for a different optimization idea and wait for its selection.
     Duplicate values cost no budget. Close early only if the action could produce no candidate.
3. If `stop_refine`:
   1. `HARNESS close --op-dir custom/<op> --action <action_id>` (attach CLOSED, update global best).
   2. **Reflect** over the trajectory, then emit tree edits:
      `HARNESS evolve --op-dir custom/<op> --edits '{"insert":[{"parent":..,"stage":..,"delta":..,"V":..,"prior_gain":..}],"update":[{"id":..,"V":..}],"prune":[..],"prune_permanent":[..],"reasons":[".."]}'`
      - If this evolve would leave the frontier empty, insert at least one new intent in the same
        call. The response carries `expansion_required: true` when it did not.`
      Insert = new promising child intents; Update = rescore frontier V on new evidence;
      Prune = drop infeasible/redundant. **`reasons` is mandatory** (write the learned insight).
      **Calibrate `prior_gain`/`V` of an inserted (non-catalog) action to the catalog's star
      scale** (`references/action_catalog.json`: strong ≈ 0.9 / medium ≈ 0.6 / weak or speculative
      ≈ 0.3), so novel actions mix fairly with seeds — do not assign > 0.9 without hard evidence.
   3. `HARNESS stop --op-dir custom/<op>` → `{stop, reason}`.
      - If `stop`, use the restored global-best `op_file` for delivery and return
        `{"directive":"STOP","reason":"<reason>","best":{...}}`.
        Run `HARNESS report --op-dir custom/<op>` if using the bare CLI.
      - Otherwise select the next action as in INIT.
        Return its CODE directive. A controller-written candidate carries `"coder": false`.


## Tile tuning: ask for a block, get one configuration back

```
HARNESS block --op-dir custom/<op> --parent <action_id> --test-command '<E(x) command>'
              [--reason '<why now>'] [--live-tiles <n>] [--max-trials <n>] [--lint-root <path>]
```

`--test-command` runs **without a shell**. The supported form is
`cd custom/<op> && python3 test_<op>.py`, with optional `NAME=value` prefixes and
further `&&` steps, which stop at the first failure. A pipe, a redirect, a
subshell, a glob or a background `&` is refused by name (exit 126) rather than
silently mis-run — put such a command in a script and name the script.

The core runs its own optimization loop — propose, write, evaluate, learn, repeat
against a fixed base — and hands back **one** configuration, already written to
the op_file. You then `record` it and decide what to do next.

**When to ask.** Two moments, one mechanism.

1. A tile-shape action was selected. The number is the core's; ask for a block.
2. A structural action has landed and the tile it inherited describes a program
   that no longer exists. A flatten, a change of view extents, a fusion that
   changes how many tensors are resident. The core fires a block automatically
   when its derived domain changes; ask explicitly when you believe the optimum
   moved even though the domain did not.

**What comes back.**

```json
{"ran": true, "best_config": {...}, "best_latency": 4.02, "code_hash": "…",
 "n_trials": 21, "n_feasible": 19, "stopped_early": true,
 "device_trials": 21, "charged": 20, "granted": 42}
```

Then: `record --parent <action_id> --code-hash <code_hash> --s 1 --p <best_latency>`.

- **Do not dispatch the coder.** The winner is already on disk, and every tile
  binding site moved together. A coder that changes the kernel's tile but not the
  wrapper's padding arithmetic ships a kernel that fails verification; the action
  is then scored unproductive and closed. That mistake has cost real wins.
- **Do not re-measure it.** `best_latency` was measured on the device inside the
  block. Pass it through.
- **Do not propose a tile value of your own,** before or after. Inside tile
  tuning the concrete number is the core's; your search is which action to take
  and what the result implies for the *others*.

**`aborted` naming environment faults ends the block.** Record any winner the
block returned, then call `stop` and relay its decision. The evaluator already
waited for the device before reporting the fault. The controller checks whether
the device is usable before ending the run:

- `device_unavailable` — the probe confirmed it. Return the STOP.
- `harness_fault` — the command never started; the fault is a bug here, not a
  busy device. Return the STOP and quote the marker.
- no reason at all — the device recovered. **Continue.** Select the next action
  normally; the aborted block's observations are kept and a later block on the
  same structure resumes from them.

Deciding this in the optimizer is what ended one recorded campaign a fraction
of the way into its budget, on a handful of faults that had cost no budget at
all. Never assert a device is unavailable from a block that finished some time
ago.

`ran: false` or `best_config: null` carries a `reason`. The one that matters:
*"no tunable tile site; the tile call is not written with integer literals"* means
normalization at INIT did not happen or did not succeed, and **no amount of
searching will make the core own this kernel's tile.** Say so in your reflection.

**Budget.** `charged` counts the block's immediately charged device evaluations.
`record` adds any remaining charge for a measured winner. A winner replayed
without new device evaluations uses its previous measurement. `granted` is what the core allowed
after clamping your `--max-trials` against the remaining budget and the run-level
share. A block stops on seven consecutive non-improving trials, so a converged
block hands the rest back.

**`--live-tiles` is a fact, not a bound.** How many tile-shaped tensors are
simultaneously resident is a dataflow property the extractor cannot read after
you have fused or flattened anything, so you may supply it. You may not supply a
domain, a range, or a maximum: those the core derives, and a domain narrowed by
judgment is the one error that leaves no trace anywhere in the run.

**Omitting it is not neutral.** Without it the footprint bound is OFF and the
block checks alignment only, so every tile too large for the buffer costs a
device evaluation instead of being rejected for free. It has been forgotten in
every recorded block to date; `block` now says so in the trajectory and in its
own output, and the count it is worth is two evaluations per block.

**A block is worth spending only when the tile space actually moved.** A
structural delta that changes a view extent triggers `retune: true`, but that
flag says the space CHANGED, not that it changed usefully. One recorded run
raised the batch extent twice on shapes where the loop count was already 1, so
both deltas produced the same single-task program and the second block spent
nine evaluations re-searching the space the first had just converged on. Before
accepting a retune, ask what about the tile space is different; if the answer is
"nothing the core can see", the delta is a tile proposal wearing a structural
name, and the core has already searched that axis.

## Tile actions: the core has already written the file

`select` returns an extra `bayesian_optimization` block when the chosen action is one of the four
tile-shape levers (`F-9`, `F-10`, `S-11`, `S-12`) and `bayesian_optimization_tile_search` is on:

```json
"bayesian_optimization": {"applied": true, "config": {"vec#0": {"M": 32, "N": 256}},
       "label": "floor", "scope": "vec", "trial": 2, "sites": 1, "code_hash": "..."}
```

**Each action moves one site kind, and only that one.** `F-9` and `S-11` are Cube
actions and reach Cube sites; `F-10` and `S-12` are Vector actions and reach Vector
sites — `scope` names which, and the config handed back contains nothing outside it, so
a gain is attributable to the action that earned it. A Cube action on a kernel with no
Cube work declines with `no cube tile site` rather than quietly tuning the Vector tiles;
retire it. Searching Cube **and** Vector together is still available and still wanted —
that is `block`, which you ask for deliberately and which is recorded as a block. The
two actions over one kind share one study (per structure), so the second continues the
first's surrogate instead of re-exploring the same space on fresh device evaluations.

`record` returns the same `bayesian_optimization` block whenever the action continues
(`stop_refine: false`) and it is one of the four tile levers. Treat it exactly
as you treat the one from `select`: **the value is already on disk**, so emit
`{"directive":"CODE", ..., "coder": false}` and do not propose a tile value of
your own. Inside a tile action the concrete variation is the lever's to pick,
not yours — that is the whole reason the lever exists. Your search continues to
be the choice of action, and the reflection at `close`.

**`applied: true` means the candidate is already on disk.** The deterministic
core chose the number from its own Bayesian study and rewrote the tile call's
argument list itself. Two consequences, and both are contractual:

1. Emit `{"directive":"CODE", ..., "coder": false}` and go directly to STEP with
   the hash of the source the controller wrote. That measurement belongs to its tile study.
2. **Do not choose a tile number.** For these four actions the number is not
   yours. Your job is whether to select the action at all, and what the result
   implies for the *other* actions.

This is the one place where "picking each concrete variation is your search"
does not apply. It applies everywhere else, and this is exactly why: a tile
value is a number the core can compute from the program and the hardware, and
the other forty-four actions are not.

`applied: false` with a `reason` means the lever had nothing to propose (no
tunable tile site, or the space is exhausted). Treat the action as an ordinary
one and pick the δ yourself.

## Cycle directives
- `{"directive":"CODE","op_file":"custom/<op>/<op>_impl.py","action_id":"u13","intent":"<δ>"}`
- `{"directive":"CODE","op_file":"custom/<op>/<op>_impl.py","action_id":"u10","intent":"<δ>","coder":false}`
  — the core already wrote it; the orchestrator must not dispatch the coder.
- `{"directive":"STOP","reason":"success|converged|wall_clock|frontier_exhausted|global_stagnation|eval_budget|preopt_incorrect|device_unavailable|harness_fault","best":{"latency_us":..,"J":..,"speedup_vs_preopt":..}}`
  (`device_unavailable` and `harness_fault` come from `stop` after a block aborted on environment
  faults. Report whichever it returns; never choose between them yourself.)
  (`converged` = a full frontend→swimlane→incore cycle produced no ≥ `stage_min_gain` improvement.)

## Hard rules
- Read `search_state.json` every dispatch; never rely on session memory. Persist only via the harness.
- Never evaluate a candidate without probing first. Duplicate submissions are rejected by the
  harness and counted in `progress.duplicate_rejected`; a high count means actions are being
  selected that the kernel cannot express, not that the coder is failing.
- `prune` retires an action **against the structure now in the file**, not forever. Your evidence
  is a set of measurements, and measurements are taken on a program: once a later action
  restructures the kernel, the disproof is stale and the action returns to the frontier as an
  ordinary candidate ranked by `pypto_action_priority` like any other. This is not a suggestion to
  revisit it — the harness does it, and the response names what it retired under
  `pruned_on_structure` and the structure hash under `structure`. **Geometry is conditional on
  structure**: a tile sweep or a loop-granularity sweep disproves those values *for the shape the
  program currently has*, and a restructuring that changes a cube extent or a loop trip count
  invalidates it. A tile VALUE is not a restructuring, so a BO trial does not revive anything.
- `prune_permanent` retires an action outright, through any restructuring. Use it only when the
  intent is **inexpressible on this operator** — it would change the public contract, or the
  option does not exist in this build — and say which, from the API or the schema rather than from
  a measurement. A mechanism that merely lost is a `prune`, not a `prune_permanent`.
- Pruning an action that is currently *held* is ignored under either key: unmet preconditions are
  the harness's business, and a later transformation may make that action reachable.
- **Extending the frontier is your job; ending the search is not.** `insert` is the world model:
  when the trajectory teaches you something the catalogue did not contain, add it. An `evolve`
  that leaves the frontier empty without inserting anything is a stop request, and the harness
  will not honour it. `select` then returns `reason: expansion_required` and you must reflect and
  insert. If there is genuinely nothing left, say so on the record with
  `evolve --edits '{"exhausted":"<why>"}'`; only a declared exhaustion permits
  `frontier_exhausted`. Pruning is for retiring intents you have disproved, never for clearing
  the board.
- Keep the evaluation method and thresholds fixed throughout the search.
- Preserve the operator's public contract (name / args / shape / dtype / layout); only the internal
  implementation is tuned. DEVICE is a per-run user parameter — never hard-code it.
