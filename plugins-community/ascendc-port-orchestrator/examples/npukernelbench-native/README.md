# Native KernelBench-style task example

`level1/example_add.py` and `level1/example_add.json` are a first-party, tiny KernelBench-style task/sidecar pair
(task `.py` + same-stem `.json`/`.jsonl` sidecar).  They are provided to smoke-test the plugin's native `npubench`
input path and to show the required file layout:

```text
<root>/level1/example_add.py
<root>/level1/example_add.json
```

The `.json` file intentionally contains JSONL, which KernelBench-style tasks accept despite the `.json`
suffix.  The task reads that sidecar itself, exports `Model` and `get_input_groups()`, and includes two small Add
broadcast cases.

This is not an upstream benchmark corpus case and is not a general acceptance golden.  Use it only with an ABI-
compatible Add source to verify native loading or as a layout template; replace it with the original task and sidecar
that represent the operator before acceptance testing.
