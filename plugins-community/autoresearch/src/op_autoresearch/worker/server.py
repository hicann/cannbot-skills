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

import asyncio
import base64
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from op_autoresearch.cli.service.worker_config import worker_timing
from op_autoresearch.core.async_pool.device_pool import DevicePool
from op_autoresearch.core.worker.eval_config import (
    resolve_eval_timeout,
    resolve_reference_timeout,
)
from op_autoresearch.core.worker.local_worker import LocalWorker
from op_autoresearch.op.utils.config_utils import check_backend_arch
from op_autoresearch.op.utils.json_safe import sanitize_floats
from op_autoresearch.utils.process_utils import reap_orphaned_process_groups

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FORMAT,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# Global worker instance
worker: Optional[LocalWorker] = None


def get_worker_config():
    """Get worker configuration from environment variables."""
    names = ("WORKER_BACKEND", "WORKER_ARCH", "WORKER_DEVICES")
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        raise RuntimeError(
            f"worker configuration missing: {', '.join(missing)}; "
            "start the daemon through `op-autoresearch worker --start`")

    backend, arch, devices_str = (os.environ[name].strip() for name in names)
    check_backend_arch(backend, arch)
    try:
        devices = [int(value.strip()) for value in devices_str.split(",")]
        if not devices or any(device < 0 for device in devices):
            raise ValueError
    except ValueError as exc:
        raise ValueError(
            f"WORKER_DEVICES must be comma-separated non-negative integers, "
            f"got {devices_str!r}") from exc

    return backend, arch, devices


@asynccontextmanager
async def lifespan(_application: FastAPI):
    """Initialize worker resources on startup."""
    global worker
    backend, arch, devices = get_worker_config()

    reaped = reap_orphaned_process_groups()
    if reaped:
        logger.warning("Reaped orphan eval process groups from predecessor: %s",
                       reaped)

    logger.info(
        "Initializing Worker Service: Backend=%s, Arch=%s, Devices=%s",
        backend,
        arch,
        devices,
    )

    timing = worker_timing()
    device_pool = DevicePool(devices, lease_ttl_s=timing.lease_ttl)
    device_pool.start_reaper(timing.lease_reap_interval)
    worker = LocalWorker(device_pool, backend=backend)

    yield

    await device_pool.stop_reaper()
    logger.info("Shutting down Worker Service")


app = FastAPI(title="OP_AUTORESEARCH Worker Service", lifespan=lifespan)


def _require_worker() -> None:
    """503 if the worker isn't initialized yet (startup race)."""
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")


def _current_worker() -> LocalWorker:
    _require_worker()
    return worker


@asynccontextmanager
async def _guarded(task_id: str, *, device_id: Optional[int] = None,
                   lease_id: Optional[int] = None):
    """Shared eval-endpoint guard: renew the device lease for the whole
    request, and map any unhandled error to a 500 (HTTPExceptions — e.g. the
    400 from a bad profile_settings parse done before entering — pass
    through). Single owner of the keepalive + error-mapping boilerplate.
    """
    try:
        async with worker.device_pool.keepalive(
                task_id, device_id=device_id, lease_id=lease_id):
            yield
    except LookupError as e:
        # The script contains a device id acquired under this exact token.
        # If the token is stale, that device may already have a new owner.
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error('[%s] request failed: %s', task_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


class _TimedEvalForm:
    def __init__(
        self,
        task_id: str = Form(...),
        op_name: str = Form(...),
        timeout: Optional[int] = Form(None),
        device_id: Optional[int] = Form(None),
        lease_id: Optional[int] = Form(None),
    ) -> None:
        self.task_id = task_id
        self.op_name = op_name
        self.timeout = timeout
        self.device_id = device_id
        self.lease_id = lease_id


class _ProfileEvalForm:
    def __init__(
        self,
        task_id: str = Form(...),
        op_name: str = Form(...),
        profile_settings: str = Form("{}"),
        device_id: Optional[int] = Form(None),
        lease_id: Optional[int] = Form(None),
    ) -> None:
        self.task_id = task_id
        self.op_name = op_name
        self.profile_settings = profile_settings
        self.device_id = device_id
        self.lease_id = lease_id

    def parse_settings(self) -> dict:
        try:
            return json.loads(self.profile_settings)
        except json.JSONDecodeError as error:
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON for profile_settings",
            ) from error


@app.post("/api/v1/verify")
async def verify(
    package: UploadFile = File(...),
    form: _TimedEvalForm = Depends(),
):
    """
    Execute verification task.
    
    Returns:
        - success: 验证是否成功
        - log: 执行日志
        - artifacts: 执行过程中生成的 JSON 文件内容
    """
    _require_worker()
    logger.info(
        "[%s] received verification request for %s",
        form.task_id,
        form.op_name,
    )
    async with _guarded(
        form.task_id,
        device_id=form.device_id,
        lease_id=form.lease_id,
    ):
        package_data = await package.read()
        success, log, artifacts = await worker.verify(
            package_data,
            form.task_id,
            form.op_name,
            resolve_eval_timeout(form.timeout),
        )
        return sanitize_floats({
            "success": success,
            "log": log,
            "artifacts": artifacts,
        })


@app.post("/api/v1/profile")
async def profile(
    package: UploadFile = File(...),
    form: _ProfileEvalForm = Depends(),
):
    """
    Execute profiling task.
    """
    _require_worker()
    settings = form.parse_settings()
    async with _guarded(
        form.task_id,
        device_id=form.device_id,
        lease_id=form.lease_id,
    ):
        package_data = await package.read()
        result = await worker.profile(
            package_data, form.task_id, form.op_name, settings
        )
        return sanitize_floats(result)


@app.post("/api/v1/generate_reference")
async def generate_reference(
    package: UploadFile = File(...),
    form: _TimedEvalForm = Depends(),
):
    """Generate and return the serialized reference payload."""
    active_worker = _current_worker()
    logger.info(
        "[%s] received reference request for %s",
        form.task_id,
        form.op_name,
    )
    async with _guarded(
        form.task_id,
        device_id=form.device_id,
        lease_id=form.lease_id,
    ):
        package_data = await package.read()
        success, log, ref_bytes = await active_worker.generate_reference(
            package_data,
            form.task_id,
            form.op_name,
            resolve_reference_timeout(form.timeout),
        )
        encoded = base64.b64encode(ref_bytes).decode("ascii") if success else ""
        return {"success": success, "log": log, "reference_data": encoded}


@app.post("/api/v1/profile_single_task")
async def profile_single_task(
    package: UploadFile = File(...),
    form: _ProfileEvalForm = Depends(),
):
    """Profile one uploaded script without running a baseline."""
    active_worker = _current_worker()
    settings = form.parse_settings()
    logger.info(
        "[%s] received single-task profile request for %s",
        form.task_id,
        form.op_name,
    )
    async with _guarded(
        form.task_id,
        device_id=form.device_id,
        lease_id=form.lease_id,
    ):
        package_data = await package.read()
        result = await active_worker.profile_single_task(
            package_data, form.task_id, form.op_name, settings
        )
        return sanitize_floats(result)


@app.get("/api/v1/docs/{doc_name}")
async def get_doc(
    doc_name: str,
):
    """Return documentation discovered in the daemon environment."""
    active_worker = _current_worker()
    try:
        content = await active_worker.get_doc(doc_name)
        return {"doc_name": doc_name, "content": content}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Get doc request failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/acquire_device")
async def acquire_device(
    task_id: str = Form(...),
    timeout: Optional[float] = Form(None),
):
    """Reserve an exact renewable device lease."""
    active_worker = _current_worker()
    wait_timeout = (
        worker_timing().acquire_timeout if timeout is None else float(timeout)
    )
    try:
        device_id, lease_id = await active_worker.device_pool.acquire_device(
            owner=task_id, timeout=wait_timeout, renewable=True
        )
        logger.info(
            "[%s] Acquired device %s (lease %s)",
            task_id,
            device_id,
            lease_id,
        )
        return {"device_id": device_id, "lease_id": lease_id}
    except TimeoutError as exc:
        logger.warning(
            "[%s] No device free within %ss: %s", task_id, wait_timeout, exc
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("[%s] Failed to acquire device: %s", task_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/release_device")
async def release_device(
    task_id: str = Form(...),
    device_id: int = Form(...),
    lease_id: int = Form(...)
):
    """Release a lease only when its token still owns the device."""
    active_worker = _current_worker()
    try:
        await active_worker.device_pool.release_device(device_id, lease_id)
        logger.info(
            "[%s] Released device %s (lease %s)",
            task_id,
            device_id,
            lease_id,
        )
        return {"status": "ok"}
    except Exception as exc:
        logger.error("[%s] Failed to release device: %s", task_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/v1/status")
async def status():
    """Return daemon readiness, identity, devices, and its active log path."""
    log_file = os.environ.get("OP_AUTORESEARCH_WORKER_LOG_FILE") or ""
    if worker is None:
        return {"status": "initializing", "log_file": log_file}

    backend, arch, devices = get_worker_config()
    return {
        "status": "ready",
        "backend": backend,
        "arch": arch,
        "devices": devices,
        "log_file": log_file,
    }


@app.get("/api/v1/health")
async def health():
    """非阻塞健康探活 —— 验"daemon 接 verify 时的请求路径还活着"，但
    不抢占设备：

      - 用 ``asyncio.Queue.get_nowait()`` 试取一次 device，能取就立刻
        放回；空队列（满载）当作 healthy（"忙不是坏"），不报 degraded
      - 整个 handler 按 worker.health_timeout 超时；超时仅当事件循环本身卡了

    /status 只验证 HTTP server 在线；/health 走一遍真实的 queue 操作
    路径，能抓出"event loop 卡住"或"queue 锁竞争"那类故障。**不会**
    阻塞等设备，所以满载 worker 不会被误判 degraded。
    """
    if worker is None:
        return {"status": "initializing", "healthy": False, "free": 0}

    timing = worker_timing()
    backend, arch, devices = get_worker_config()
    device_pool = worker.device_pool
    pool = device_pool.available_devices
    base = {
        "status": "ready",
        "backend": backend,
        "arch": arch,
        "devices": devices,
        "free": pool.qsize(),
        "healthy": False,
    }

    async def _probe():
        # Exercise the real Queue path (get + immediate put) to catch a
        # wedged event loop / queue. The pool's free set is a plain
        # asyncio.Queue, so put_nowait wakes any pending getter on its own —
        # no Condition to coordinate. A real acquirer racing this only waits
        # the instant between get and put.
        try:
            device_id = pool.get_nowait()
        except asyncio.QueueEmpty:
            # All devices busy — daemon is fine, just at capacity.
            return None
        pool.put_nowait(device_id)
        return device_id

    try:
        device_id = await asyncio.wait_for(_probe(), timeout=timing.health_timeout)
        base["healthy"] = True
        if device_id is not None:
            base["probed_device"] = device_id
        else:
            base["note"] = "all devices busy (healthy, just at capacity)"
        return base
    except asyncio.TimeoutError:
        base["error"] = (
            f"event loop unresponsive (>{timing.health_timeout}s) "
            "—— 事件循环可能阻塞"
        )
        logger.warning(
            "健康探活超时：event loop %s 秒内未响应",
            timing.health_timeout,
        )
        return base
    except Exception as e:
        base["error"] = f"健康探活异常：{type(e).__name__}: {e}"
        logger.warning('健康探活异常：%s', e)
        return base


def start_server(host: Optional[str] = None, port: Optional[int] = None):
    """
    启动 OP_AUTORESEARCH Worker Service。
    
    Args:
        host: 监听地址。可从环境变量 WORKER_HOST 设置。
              - IPv4: "0.0.0.0" (所有接口), "127.0.0.1" (本地)
              - IPv6: "::" (所有接口，双栈), "::1" (本地)
              默认: "0.0.0.0"
        port: 监听端口。可从环境变量 WORKER_PORT 设置。
              默认: 9001
    """
    host = host or os.environ.get("WORKER_HOST", "0.0.0.0")
    port = (
        port
        if port is not None
        else int(os.environ.get("WORKER_PORT", "9001"))
    )

    logger.info('Starting Worker Service on %s:%s', host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
