---
name: aog-prior-art-verify
description: Use when an arch22→arch35 migration can reuse an existing arch35 candidate. Scan, provenance-stage, build, measure, and learn from it without replacing fresh arch22 truth capture or customer-facing verification.
---

# Prior-art candidate verification

Use existing target, sibling, or archive implementations to accelerate authoring without treating them as truth. This skill supplies provenance-bound seeds and counterexamples to the standard AscendC worker; it does not decide that an operator is complete.

## Required pipeline

1. Run `scripts/scan_prior_art.py` to discover candidate implementations.
2. Run `scripts/stage_candidate.py` to copy only the files explicitly authorized by the scan into `.prior_art_candidate/`. It aborts if any scan-time digest changed and writes `manifest.json` with source, build-overlay path, file digest, and a whole-candidate digest.
3. Capture fresh arch22 source-NPU outputs for the current run. This step is mandatory even when an identical candidate was verified previously.
4. Run `scripts/build_candidate.py` against the staged candidate. It builds a SHA-checked overlay in a unique container-side checkout; it never builds or pulls artifacts from the pristine operator checkout.
5. Provide `workspace/<op>/verify_prior_art_candidate.py`, an adapter that accepts the explicit candidate/truth arguments documented by `scripts/verify_candidate.py` and echoes the supplied binding in its JSON result. Then run `scripts/verify_candidate.py` against the current run's fresh source truth. A missing adapter or binding mismatch fails closed. Keep the result in `.prior_art_candidate/verification_prior_art.json`.
6. Run `scripts/classify.py`. Its `CANDIDATE_*` verdict describes only the candidate:

   | Verdict | Use by the standard workflow |
   |---|---|
   | `CANDIDATE_PASS` | Retain as a provenance-bound implementation seed and comparison point. |
   | `CANDIDATE_PRECISION_GAP` | Add the failing cases to the worker brief. |
   | `CANDIDATE_PERF_GAP` | Record an optimization target. |
   | `CANDIDATE_DET_GAP` | Record the determinism gap. |
   | `CANDIDATE_BUILD_GAP` | Record the build failure as a counterexample. |

7. Run `scripts/learn_extract.py` to produce reviewable knowledge candidates.
8. Continue the standard worker, build, and O4/O5 customer verification path.

## Hard invariants

- A fresh arch22 source-NPU capture is always required. A cached capture or candidate output cannot replace it.
- A target, sibling, or archive candidate is prior art, never an independent truth source.
- Target-tree files are eligible for staging only when the scan explicitly consulted that source; staging never rediscovers an unlisted file from disk.
- Candidate evidence remains in `verification_prior_art.json`; never copy it to the customer-facing `verification.json`.
- This skill never writes final `output/`, completion markers, skip markers, or customer success verdicts.
- Every staged file records its origin and digest. Any derived principle remains an unverified knowledge candidate until reviewed.
- Build and verification reports must agree on the candidate digest, manifest digest, built-library digest, fresh truth digests, and capture id. Missing, stale, or contradictory bindings are invalid rather than candidate success.
- Only the standard O4/O5 path may establish customer success.

## Orchestrator contract

The orchestrator may cache discovery and staging when the recorded source digest still matches. It must rebuild, remeasure, and reclassify the candidate against the fresh capture in every workflow run.

```text
candidate = scan_and_stage_with_provenance(op)
source_truth = provision_fresh_arch22_reference(op)
prior_art_advisory = build_measure_and_classify(candidate, source_truth)
worker_result = run_standard_worker(op, source_truth, prior_art_advisory)
return run_standard_customer_verification(worker_result, source_truth)
```

Legacy schema keys or script arguments may retain compatibility names, but user-facing migration scope remains arch22→arch35.

## Outputs

All outputs are workspace-local and advisory:

- `.prior_art_candidate/manifest.json`
- `.prior_art_candidate/build/`
- `.prior_art_candidate/verification_prior_art.json`
- `prior_art_verdict.json`
- `prior_art_learn.md`

Do not automatically commit copied source or learned material.
