# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plain console output for CLI and machine-protocol entry points.

Operational diagnostics belong in ``logging``. This module is only for data
that is intentionally part of a command's stdout/stderr contract.
"""

import sys
from typing import TextIO


def emit(
    *values: object,
    sep: str = " ",
    end: str = "\n",
    file: TextIO | None = None,
    flush: bool = False,
) -> None:
    """Write one protocol/console record with print-compatible semantics."""
    stream = file if file is not None else sys.stdout
    stream.write(sep.join(str(value) for value in values) + end)
    if flush:
        stream.flush()
