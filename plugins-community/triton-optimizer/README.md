# Triton Optimizer Plugin

`triton-optimizer` packages the Claude optimize workflow and Triton convert workflow for Triton Agent in CANNBot.

It provides:

- `triton-npu-optimizer`: optimize an existing Triton Ascend NPU operator with baseline/round state management.
- `triton-npu-converter`: convert a PyTorch operator into a PyTorch-facing Triton Ascend NPU implementation and validate it.
- Workflow hooks for optimize state bootstrap, guardrails, and session cleanup.

The reusable skills are installed from the repository-level `ops/` directory through the `triton-optimizer-skills` dependency. The plugin directory intentionally keeps only agents, hooks, and installation entrypoints.
