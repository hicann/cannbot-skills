# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Python orchestrator for AscendC op-gen pipeline.

Replaces LLM-as-orchestrator (Claude Code main session driving phases) with
deterministic Python script that reads workflows/opgen_state_machine.yaml
and dispatches sub-agents via `claude --print --agent` CLI.

LLM continues to run INSIDE worker/probe/optimizer/etc. agents where
kernel-write/probe-analysis/KB-quality genuinely need LLM judgment. LLM
never drives the top level.

See docs/design/PYTHON_ORCHESTRATOR_DESIGN.md for full architecture.
See docs/design/CONTRACT_AND_MATURITY_NOTES.md#spike-cc-agent-transport for transport-layer validation.

Status: V1 implementation in progress (DEBT-077, 2026-05-04).
"""
__version__ = "0.1.0-dev"
