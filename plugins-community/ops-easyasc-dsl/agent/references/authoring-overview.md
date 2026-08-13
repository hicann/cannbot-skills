# Kernel Authoring Overview

This is a conceptual map, not a workflow entry. Start from `agent/ROUTER.md` and
read this only when a high-level view of a single kernel is useful.

An EasyASC kernel has four layers:

1. a public `GMTensor` ABI and runtime shape contract;
2. a tile/dataflow plan constrained by one device;
3. local-buffer transfers, compute, and explicit ownership edges;
4. simulator and golden validation, followed by real-device validation when
   requested.

Choose topology from the formula. Pure vector work stays vec-only; matmul or
convolution uses Cube; mixed formulas cross pipes only where the data dependency
requires it. `splitk` and `splitn` are single-core matmul tiling modes, not
automatic cross-core distribution or merging.

For normal matmul, candidate split widths at or above 32 are often useful search
points, but 32 is an empirical heuristic, not a correctness restriction. Hard
limits come from dtype/layout requirements, local-buffer capacity, shortcut
validation, and the selected device.

Use these focused references as needed:

- safety gate: `authoring-preflight.md`
- capacity and target behavior: `facts-device-runtime.md`
- detailed buffer and scheduling facts: `facts-authoring.md`
- reusable dataflow: `pattern-index.md`
- examples: `agent/scripts/select_kernel_example.py`
- implementation evidence: `code-paths.md`
