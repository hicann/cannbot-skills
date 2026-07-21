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

"""HTTP implementation of the worker execution boundary."""

import base64
import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Optional

import httpx

from op_autoresearch.cli.service.worker_config import worker_timing
from op_autoresearch.config import get_env_var

from .eval_config import resolve_eval_timeout, resolve_reference_timeout
from .interface import WorkerInterface, empty_profile_result

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RemoteRequest:
    method: str
    url: str
    read_timeout: float
    task_id: str
    files: Any = None
    data: Any = None


def _environment_seconds(name: str, fallback: str) -> float:
    return float(get_env_var(name, fallback))


def _connect_timeout_s() -> float:
    return _environment_seconds("WORKER_CONNECT_TIMEOUT_S", "5.0")


def _write_timeout_s() -> float:
    return _environment_seconds("WORKER_WRITE_TIMEOUT_S", "30.0")


def _pool_timeout_s() -> float:
    return _environment_seconds("WORKER_POOL_TIMEOUT_S", "5.0")


def _transient_retry_attempts() -> int:
    configured = int(get_env_var("WORKER_TRANSIENT_ATTEMPTS", "2"))
    return max(configured, 1)


def _http_timeout(read_seconds: float) -> httpx.Timeout:
    budgets = {
        "connect": _connect_timeout_s(),
        "read": read_seconds,
        "write": _write_timeout_s(),
        "pool": _pool_timeout_s(),
    }
    return httpx.Timeout(**budgets)


def _single_profile_failure(message: str) -> dict[str, Any]:
    return {"time_us": None, "success": False, "log": message}


def _response_artifacts(task_id: str, reply: dict) -> dict:
    artifacts = reply.get("artifacts", {})
    if artifacts:
        logger.info(
            "[%s] Received %s artifact files from remote worker",
            task_id,
            len(artifacts),
        )
    return artifacts


class RemoteWorker(WorkerInterface):
    """Delegate worker operations to a daemon while tracking an exact lease."""

    def __init__(
        self,
        worker_url: str,
        on_transient_failure: Optional[Callable[[], None]] = None,
    ):
        self.worker_url = worker_url.rstrip("/")
        self.on_transient_failure = on_transient_failure
        self._active_lease = ContextVar(
            f"remote_worker_lease_{id(self)}",
            default=None,
        )

    async def acquire_device(
        self,
        task_id: str = "unknown",
        timeout: Optional[float] = None,
    ) -> tuple[int, int]:
        timing = worker_timing()
        wait_budget = timing.acquire_timeout if timeout is None else float(timeout)
        request = _RemoteRequest(
            method="POST",
            url=self._endpoint("acquire_device"),
            read_timeout=wait_budget + timing.http_read_margin,
            task_id=task_id,
            data={"task_id": task_id, "timeout": str(wait_budget)},
        )
        try:
            reply = await self._post_with_reconnect(request)
            lease = (reply.get("device_id"), reply.get("lease_id"))
            if not all(isinstance(value, int) for value in lease):
                raise RuntimeError(
                    "worker returned invalid lease token: "
                    f"device_id={lease[0]!r}, lease_id={lease[1]!r}"
                )
            self._active_lease.set((task_id, *lease))
            logger.info(
                "[%s] Acquired remote device %s (lease %s)",
                task_id,
                *lease,
            )
            return lease
        except Exception as exc:
            logger.error("[%s] Failed to acquire remote device: %s", task_id, exc)
            raise RuntimeError(f"Failed to acquire remote device: {exc}") from exc

    async def release_device(
        self,
        device_id: int,
        lease_id: int,
        task_id: str = "unknown",
    ) -> None:
        lease = (task_id, device_id, lease_id)
        request = _RemoteRequest(
            method="POST",
            url=self._endpoint("release_device"),
            read_timeout=worker_timing().release_timeout,
            task_id=task_id,
            data={
                "task_id": task_id,
                "device_id": device_id,
                "lease_id": lease_id,
            },
        )
        try:
            await self._post_with_reconnect(request)
            logger.info("[%s] Released remote device %s", task_id, device_id)
        except Exception as exc:
            logger.error("[%s] Failed to release remote device: %s", task_id, exc)
        finally:
            if self._active_lease.get() == lease:
                self._active_lease.set(None)

    async def get_doc(self, doc_name: str) -> str:
        request_url = self._endpoint(f"docs/{doc_name}")
        try:
            reply = await self._get_with_reconnect(
                request_url,
                read_timeout=worker_timing().doc_timeout,
                task_id=f"doc:{doc_name}",
            )
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Remote worker returned %s for doc %r: %s",
                exc.response.status_code,
                doc_name,
                exc.response.text,
            )
            return ""
        except Exception as exc:
            logger.warning(
                "Failed to fetch remote doc %r from %s: %s",
                doc_name,
                self.worker_url,
                exc,
            )
            return ""
        return reply.get("content", "") if isinstance(reply, dict) else ""

    async def verify(
        self,
        package_data: bytes,
        task_id: str,
        op_name: str,
        timeout: Optional[int] = None,
    ) -> tuple[bool, str, dict[str, Any]]:
        budget = resolve_eval_timeout(timeout)
        try:
            request = self._package_request(
                "verify",
                package_data,
                budget + worker_timing().http_read_margin,
                {
                    "task_id": task_id,
                    "op_name": op_name,
                    "timeout": str(budget),
                },
            )
            reply = await self._send_package(request)
            artifacts = _response_artifacts(task_id, reply)
            return reply.get("success", False), reply.get("log", ""), artifacts
        except Exception as exc:
            message = self._operation_error("verification", exc)
            logger.error("[%s] %s", task_id, message)
            return False, message, {}

    async def profile(
        self,
        package_data: bytes,
        task_id: str,
        op_name: str,
        profile_settings: dict[str, Any],
    ) -> dict[str, Any]:
        per_script = resolve_eval_timeout(profile_settings.get("timeout"))
        try:
            request = self._package_request(
                "profile",
                package_data,
                2 * per_script + worker_timing().http_read_margin,
                {
                    "task_id": task_id,
                    "op_name": op_name,
                    "profile_settings": json.dumps(profile_settings),
                },
            )
            reply = await self._send_package(request)
            _response_artifacts(task_id, reply)
            return reply
        except Exception as exc:
            logger.error("[%s] Remote profiling failed: %s", task_id, exc)
            return empty_profile_result(error=str(exc))

    async def profile_single_task(
        self,
        package_data: bytes,
        task_id: str,
        op_name: str,
        profile_settings: dict[str, Any],
    ) -> dict[str, Any]:
        budget = resolve_eval_timeout(profile_settings.get("timeout"))
        try:
            request = self._package_request(
                "profile_single_task",
                package_data,
                budget + worker_timing().http_read_margin,
                {
                    "task_id": task_id,
                    "op_name": op_name,
                    "profile_settings": json.dumps(profile_settings),
                },
            )
            return await self._send_package(request)
        except Exception as exc:
            message = self._operation_error("profile_single_task", exc)
            logger.error("[%s] %s", task_id, message)
            return _single_profile_failure(message)

    async def generate_reference(
        self,
        package_data: bytes,
        task_id: str,
        op_name: str,
        timeout: Optional[int] = None,
    ) -> tuple[bool, str, bytes]:
        budget = resolve_reference_timeout(timeout)
        try:
            request = self._package_request(
                "generate_reference",
                package_data,
                budget + worker_timing().http_read_margin,
                {
                    "task_id": task_id,
                    "op_name": op_name,
                    "timeout": str(budget),
                },
            )
            reply = await self._send_package(request)
            success = reply.get("success", False)
            log = reply.get("log", "")
            encoded = reply.get("reference_data", "")
            if not success:
                return False, log, b""
            if not encoded:
                return False, f"No reference data in response:\n{log}", b""
            reference = base64.b64decode(encoded)
            logger.info(
                "[%s] Received reference data: %s bytes",
                task_id,
                len(reference),
            )
            return True, log, reference
        except Exception as exc:
            message = self._operation_error("generate_reference", exc)
            logger.error("[%s] %s", task_id, message)
            return False, message, b""

    def _endpoint(self, operation: str) -> str:
        return f"{self.worker_url}/api/v1/{operation}"

    def _package_request(
        self,
        operation: str,
        package_data: bytes,
        read_timeout: float,
        form: dict[str, Any],
    ) -> _RemoteRequest:
        task_id = form["task_id"]
        return _RemoteRequest(
            method="POST",
            url=self._endpoint(operation),
            read_timeout=read_timeout,
            task_id=task_id,
            files={
                "package": (
                    "package.tar",
                    package_data,
                    "application/x-tar",
                )
            },
            data=self._attach_active_lease(form, task_id),
        )

    async def _send_package(self, request: _RemoteRequest) -> dict:
        operation = request.url.rsplit("/", maxsplit=1)[-1]
        logger.info(
            "[%s] Sending %s request to %s",
            request.task_id,
            operation,
            request.url,
        )
        return await self._post_with_reconnect(request)

    def _operation_error(self, operation: str, error: Exception) -> str:
        if isinstance(error, httpx.RequestError):
            message = (
                f"Network error communicating with worker at "
                f"{self.worker_url}: {error}"
            )
            if operation == "verification":
                message += (
                    ". Please check if the worker service is running and "
                    "accessible."
                )
            return message
        if isinstance(error, httpx.HTTPStatusError):
            response = error.response
            return (
                f"Worker returned error status: {response.status_code} - "
                f"{response.text}"
            )
        label = {
            "profile_single_task": "profile_single_task",
            "generate_reference": "generate_reference",
        }.get(operation, operation.replace("_", " "))
        return f"Remote {label} failed: {error}"

    def _attach_active_lease(
        self, data: dict[str, Any], task_id: str
    ) -> dict[str, Any]:
        lease = self._active_lease.get()
        if lease is None or lease[0] != task_id:
            return data
        return {
            **data,
            "device_id": str(lease[1]),
            "lease_id": str(lease[2]),
        }

    async def _post_with_reconnect(self, request: _RemoteRequest):
        return await self._request_with_reconnect(request)

    async def _get_with_reconnect(
        self,
        url: str,
        *,
        read_timeout: float,
        task_id: str,
    ):
        request = _RemoteRequest("GET", url, read_timeout, task_id)
        return await self._request_with_reconnect(request)

    async def _request_with_reconnect(self, request: _RemoteRequest):
        attempts = 1
        if self.on_transient_failure is not None:
            attempts = _transient_retry_attempts()

        for attempt in range(1, attempts + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=_http_timeout(request.read_timeout)
                ) as client:
                    response = await client.request(
                        request.method,
                        request.url,
                        files=request.files,
                        data=request.data,
                    )
                response.raise_for_status()
                return response.json()
            except httpx.ConnectError:
                if attempt == attempts:
                    raise
                self._recover_transient_connection(request, attempt, attempts)
        raise RuntimeError("remote request retry loop ended unexpectedly")

    def _recover_transient_connection(
        self,
        request: _RemoteRequest,
        attempt: int,
        attempts: int,
    ) -> None:
        logger.warning(
            "[%s] %s %s connection failed (%s/%s); reconnecting before retry",
            request.task_id,
            request.method,
            request.url,
            attempt,
            attempts,
        )
        try:
            self.on_transient_failure()
        except Exception as exc:
            logger.error(
                "[%s] transient-failure callback failed: %s",
                request.task_id,
                exc,
            )
