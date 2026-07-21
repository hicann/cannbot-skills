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

"""Worker registration and load-aware selection."""

import asyncio
import logging
from dataclasses import field, make_dataclass
from functools import cache
from typing import Callable, Optional

from op_autoresearch.cli.service.worker_config import worker_timing
from op_autoresearch.config import get_env_var

from .interface import WorkerInterface

logger = logging.getLogger(__name__)


class _FactoryDefault:
    def __init__(self, constructor):
        self.constructor = constructor


def _record(name: str, fields: list[tuple], *, frozen: bool = False):
    return make_dataclass(
        name,
        [
            (*spec[:2], field(default_factory=spec[2].constructor))
            if len(spec) == 3 and isinstance(spec[2], _FactoryDefault)
            else spec
            for spec in fields
        ],
        frozen=frozen,
        namespace={"__module__": __name__},
    )


WorkerInfo = _record(
    "WorkerInfo",
    [
        ("worker", WorkerInterface),
        ("backend", str),
        ("arch", str),
        ("tags", set[str], _FactoryDefault(set)),
        ("capacity", int, 1),
        ("load", int, 0),
    ],
)

RemoteWorkerRegistration = _record(
    "RemoteWorkerRegistration",
    [
        ("backend", str),
        ("arch", str),
        ("worker_url", Optional[str], None),
        ("capacity", Optional[int], None),
        ("tags", Optional[set[str]], None),
        ("expected_device_ids", Optional[list[int]], None),
        ("on_transient_failure", Optional[Callable[[], None]], None),
    ],
    frozen=True,
)

WorkerRegistration = _record(
    "WorkerRegistration",
    [
        ("backend", str),
        ("arch", str),
        ("device_ids", Optional[list[int]], None),
        ("worker_url", Optional[str], None),
        ("tags", Optional[set[str]], None),
        ("on_transient_failure", Optional[Callable[[], None]], None),
    ],
    frozen=True,
)


def _matches(
    info: WorkerInfo,
    backend: str,
    arch: Optional[str],
    tags: Optional[set[str]],
) -> bool:
    identity_matches = info.backend == backend and (
        not arch or info.arch == arch
    )
    return identity_matches and (not tags or tags <= info.tags)


class WorkerManager:
    """Registry that routes work by identity and current load."""

    def __init__(self):
        self._workers: list[WorkerInfo] = []
        self._lock = asyncio.Lock()

    async def register(self, info: WorkerInfo):
        async with self._lock:
            info.capacity = max(info.capacity, 1)
            self._workers.append(info)
        logger.info(
            "Registered worker: backend=%s, arch=%s, capacity=%s",
            info.backend,
            info.arch,
            info.capacity,
        )

    async def select(
        self,
        backend: str,
        arch: Optional[str] = None,
        tags: Optional[set[str]] = None,
    ) -> Optional[WorkerInterface]:
        async with self._lock:
            matches = self._matching_infos(backend, arch, tags)
            if not matches:
                return None
            selected = min(matches, key=lambda info: info.load / info.capacity)
            selected.load += 1
            logger.debug(
                "Selected worker %s (load=%s/%s)",
                id(selected.worker),
                selected.load,
                selected.capacity,
            )
            return selected.worker

    async def reserve(self, worker: WorkerInterface) -> bool:
        async with self._lock:
            info = next(
                (item for item in self._workers if item.worker is worker),
                None,
            )
            if info is None:
                return False
            info.load += 1
            return True

    async def has_worker(
        self,
        backend: str,
        arch: Optional[str] = None,
        tags: Optional[set[str]] = None,
    ) -> bool:
        async with self._lock:
            return bool(self._matching_infos(backend, arch, tags))

    async def list_matching(
        self,
        backend: str,
        arch: Optional[str] = None,
        tags: Optional[set[str]] = None,
    ) -> list[WorkerInterface]:
        async with self._lock:
            return [
                info.worker
                for info in self._matching_infos(backend, arch, tags)
            ]

    async def release(self, worker: WorkerInterface) -> bool:
        async with self._lock:
            info = next(
                (item for item in self._workers if item.worker is worker),
                None,
            )
            if info is None:
                logger.error("Release called for unknown worker id=%s", id(worker))
                return False
            info.load = max(info.load - 1, 0)
            logger.debug(
                "Released worker %s (load=%s/%s)",
                id(info.worker),
                info.load,
                info.capacity,
            )
            return True

    async def get_status(self) -> list[dict]:
        async with self._lock:
            return [
                {
                    "backend": info.backend,
                    "arch": info.arch,
                    "load": info.load,
                    "capacity": info.capacity,
                    "tags": list(info.tags),
                }
                for info in self._workers
            ]

    def _matching_infos(
        self,
        backend: str,
        arch: Optional[str],
        tags: Optional[set[str]],
    ) -> list[WorkerInfo]:
        return [
            info
            for info in self._workers
            if _matches(info, backend, arch, tags)
        ]


@cache
def get_worker_manager() -> WorkerManager:
    """Return the lazily created process-wide worker registry."""
    return WorkerManager()


async def register_local_worker(
    device_ids: list[int],
    backend: str,
    arch: str,
    tags: Optional[set[str]] = None,
) -> WorkerInterface:
    """Create one in-process worker and add it to the global registry."""
    from ..async_pool.device_pool import DevicePool
    from .local_worker import LocalWorker

    local_worker = LocalWorker(DevicePool(device_ids), backend=backend)
    info = WorkerInfo(
        local_worker,
        backend,
        arch,
        tags or set(),
        len(device_ids),
    )
    await get_worker_manager().register(info)
    logger.info(
        "Registered LocalWorker: backend=%s, arch=%s, devices=%s",
        backend,
        arch,
        device_ids,
    )
    return local_worker


def _registration_url(request: RemoteWorkerRegistration) -> str:
    resolved = request.worker_url or get_env_var("WORKER_URL")
    if resolved:
        return resolved
    raise ValueError(
        "worker_url was not provided and WORKER_URL is unset; pass a "
        "worker URL or export OP_AUTORESEARCH_WORKER_URL"
    )


async def _query_remote_devices(worker_url: str) -> list[int]:
    import httpx

    endpoint = worker_url.rstrip("/") + "/api/v1/status"
    timeout = worker_timing().status_timeout
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(endpoint)
        response.raise_for_status()
        reported = response.json().get("devices", [])
    if not isinstance(reported, list):
        return []
    return list(map(int, reported))


def _validate_remote_devices(
    worker_url: str,
    remote_devices: list[int],
    expected_devices: Optional[list[int]],
) -> None:
    if not expected_devices:
        return
    if not remote_devices:
        raise RuntimeError(
            f"worker {worker_url} returned no devices; cannot validate "
            f"requested devices={sorted(expected_devices)}"
        )
    if set(remote_devices).isdisjoint(expected_devices):
        raise RuntimeError(
            f"worker {worker_url} device mismatch: daemon reports "
            f"{remote_devices}, task requests {sorted(expected_devices)}"
        )


async def _remote_capacity(
    request: RemoteWorkerRegistration, worker_url: str
) -> int:
    if request.capacity is not None and not request.expected_device_ids:
        return max(request.capacity, 1)
    try:
        devices = await _query_remote_devices(worker_url)
    except Exception as exc:  # network and response-decoding boundary
        if request.expected_device_ids:
            raise RuntimeError(
                f"could not validate worker {worker_url}: /status failed "
                f"({exc}); expected devices={request.expected_device_ids}"
            ) from exc
        logger.warning(
            "Remote worker status query failed: %s; using configured capacity",
            exc,
        )
        return max(request.capacity or 1, 1)

    _validate_remote_devices(worker_url, devices, request.expected_device_ids)
    if request.capacity is None:
        capacity = len(devices) or 1
        logger.info("Remote worker capacity=%s from devices=%s", capacity, devices)
    else:
        capacity = request.capacity
    return max(capacity, 1)


async def register_remote_worker(
    request: RemoteWorkerRegistration,
) -> WorkerInterface:
    """Connect to a daemon and add its capacity to the global registry."""
    from .remote_worker import RemoteWorker

    worker_url = _registration_url(request)
    capacity = await _remote_capacity(request, worker_url)
    remote_worker = RemoteWorker(
        worker_url,
        on_transient_failure=request.on_transient_failure,
    )
    await get_worker_manager().register(
        WorkerInfo(
            remote_worker,
            request.backend,
            request.arch,
            request.tags or set(),
            capacity,
        )
    )
    logger.info(
        "Registered RemoteWorker: backend=%s, arch=%s, url=%s, capacity=%s",
        request.backend,
        request.arch,
        worker_url,
        capacity,
    )
    return remote_worker


async def register_worker(request: WorkerRegistration) -> WorkerInterface:
    """Register the configured remote worker, or a local worker as fallback."""
    worker_url = request.worker_url or get_env_var("WORKER_URL")
    if worker_url:
        remote = RemoteWorkerRegistration(
            backend=request.backend,
            arch=request.arch,
            worker_url=worker_url,
            tags=request.tags,
            expected_device_ids=request.device_ids,
            on_transient_failure=request.on_transient_failure,
        )
        return await register_remote_worker(remote)
    if request.device_ids:
        return await register_local_worker(
            request.device_ids,
            request.backend,
            request.arch,
            request.tags,
        )
    raise RuntimeError(
        "未找到可用的 Worker。请先注册 Worker 再运行 evolve：\n"
        "  方式一：设置远程 Worker URL\n"
        "    export OP_AUTORESEARCH_WORKER_URL=http://<worker-host>:<port>\n"
        "  方式二：指定本地设备列表调用 register_worker(..., device_ids=[0])"
    )
