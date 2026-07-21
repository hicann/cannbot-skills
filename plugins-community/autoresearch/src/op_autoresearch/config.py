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

"""Small runtime configuration helpers shared by verifier and workers."""

import os
from typing import Optional


def get_env_var(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read a namespaced setting, with the unprefixed name as fallback."""
    return os.environ.get(f"OP_AUTORESEARCH_{name}", os.environ.get(name, default))

