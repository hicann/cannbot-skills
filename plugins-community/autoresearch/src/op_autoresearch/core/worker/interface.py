# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Optional

PackageData = bytes | str
TaskName = str
ProfileSettings = dict[str, Any]
ProfileResult = dict[str, Any]
VerificationResult = tuple[bool, str, dict[str, Any]]
ReferenceResult = tuple[bool, str, bytes]
Lease = tuple[int, int]


def empty_profile_result(error: Optional[str] = None) -> dict[str, Any]:
    """Canonical profile failure shape shared by local and remote workers."""
    keys = (
        "gen_time", "base_time", "speedup", "per_shape_gen_us",
        "per_shape_base_us", "gen_method", "base_method", "roofline_time",
        "roofline_speedup", "roofline", "artifacts",
    )
    defaults = (None, None, 0.0, [], [], None, None, None, 0.0, None, {})
    result = dict(zip(keys, defaults))
    if error is not None:
        result["error"] = error
    return result


class WorkerInterface(ABC):
    """Execution boundary implemented by local and HTTP-backed workers."""

    @abstractmethod
    async def verify(
        self,
        package_data: PackageData,
        task_id: TaskName,
        op_name: str,
        timeout: int | None = None,
    ) -> VerificationResult:
        """Run a prepared verification package and return its artifacts."""
        raise NotImplementedError

    @abstractmethod
    async def profile(
        self,
        package_data: bytes,
        task_id: TaskName,
        op_name: str,
        profile_settings: ProfileSettings,
    ) -> ProfileResult:
        """Measure the candidate and baseline scripts in a package."""
        raise NotImplementedError

    @abstractmethod
    async def generate_reference(
        self,
        package_data: bytes,
        task_id: TaskName,
        op_name: str,
        timeout: int | None = None,
    ) -> ReferenceResult:
        """Execute the reference producer and return serialized output."""
        raise NotImplementedError

    @abstractmethod
    async def profile_single_task(
        self,
        package_data: bytes,
        task_id: TaskName,
        op_name: str,
        profile_settings: ProfileSettings,
    ) -> ProfileResult:
        """Measure one script without a baseline comparison."""
        raise NotImplementedError

    @abstractmethod
    async def get_doc(self, doc_name: str) -> str:
        """Read documentation available in the worker environment."""
        raise NotImplementedError

    @abstractmethod
    async def acquire_device(
        self, task_id: TaskName = "unknown", timeout: float | None = None
    ) -> Lease:
        """Reserve a device and return its id together with a lease token."""
        raise NotImplementedError

    @abstractmethod
    async def release_device(
        self, device_id: int, lease_id: int, task_id: TaskName = "unknown"
    ) -> None:
        """Return a device only when the supplied lease still owns it."""
        raise NotImplementedError

    @asynccontextmanager
    async def device_lease(
        self, task_id: TaskName = "unknown", *, timeout: float | None = None
    ):
        """Expose a reserved device id for the duration of one async block."""
        reservation = await self.acquire_device(task_id, timeout=timeout)
        try:
            yield reservation[0]
        finally:
            await self.release_device(*reservation, task_id)
