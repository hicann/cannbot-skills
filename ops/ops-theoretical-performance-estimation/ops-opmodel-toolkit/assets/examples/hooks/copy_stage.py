# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""User Stage declaration: copy one fp32 tile through a UB buffer."""

from accel_model.kernel.stages_base import BuildContext, ProgramBuilder, Stage


class CopyStage(Stage):
    def build(self, context: BuildContext) -> dict:
        shape = [context.params["rows"], context.params["cols"]]
        program = ProgramBuilder(
            pools={"scratch": {"scope": "stage"}},
            dtypes={"src": "fp32", "local": "fp32", "dst": "fp32"},
        )
        program.move(
            id="load",
            src={"tensor": "src", "mem": "GM"},
            dst={"tensor": "local", "mem": "UB"},
            size=shape,
            pool="scratch",
            aiv_presplit=True,
        )
        store = program.move(
            id="store",
            src={"tensor": "local", "mem": "UB"},
            dst={"tensor": "dst", "mem": "GM"},
            size=shape,
            use_pools=["scratch"],
            aiv_presplit=True,
        )
        return program.finish(signals={"copy_done": store})
