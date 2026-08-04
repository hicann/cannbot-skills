# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""smart-agent HTTP 服务（stdlib，免装 fastapi）。

POST /analyze  body: {trajectoryText, provider:{baseUrl,apiKey,model}, promptMd?, outputDir?, outputBasename?}
  → NDJSON stream: progress events → {stage:"result", outputPath, rounds, analysis}
POST /compress  body: 同 /analyze 但只压缩 → NDJSON stream → {stage:"result", outputPath}
POST /compress-and-analyze  body: 同 /analyze → 先压缩再分析，单 NDJSON 流
GET /health → {status: ok}
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from trajectory_analyzer import run_analysis_pipeline, AIProviderConfig, AnalysisError, SchemaError, RunContext
from trajectory_compressor import compress_trajectory, deterministic_compress
from trajectory_parser import inject_flow_metrics
from jsonl_logger import JsonlLogger

PORT = int(os.environ.get("CANNBOT_AGENT_PORT", "21026"))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "prompts")


class Handler(BaseHTTPRequestHandler):
    @staticmethod
    def _ca_v4(write_event, provider, prompt_md, ctx, agent_io):
        if not isinstance(agent_io, dict):
            write_event({"stage": "error", "msg": "v4 缺少 agentIo 字段"})
            return
        agent_count = len(agent_io.get("agents", []))
        write_event({"stage": "extract-start",
                     "msg": f"agent-IO 提取完成（{agent_count} 个 agent，确定性，由路由算好）"})
        write_event({"stage": "extract-done",
                     "msg": f"agent-IO 就绪：{agent_count} 个 agent",
                     "outputPath": os.path.join(ctx.output_dir, f"{ctx.output_basename}-agentio.json")})
        analyze_log_path = os.path.join(
            ctx.output_dir, f"smart-agent-{ctx.output_basename}.jsonl")
        analyze_logger = JsonlLogger(log_path=analyze_log_path, cwd=os.getcwd())
        v4ctx = RunContext(output_dir=ctx.output_dir, output_basename=ctx.output_basename,
                           on_progress=ctx.on_progress, logger=analyze_logger,
                           mode="v4", agent_io=agent_io)
        try:
            result = run_analysis_pipeline("", prompt_md, provider, v4ctx)
            write_event({
                "stage": "result",
                "outputPath": result.output_path,
                "steps": result.steps,
                "analysis": result.analysis,
            })
        finally:
            analyze_logger.close()

    @staticmethod
    def _ca_compress(write_event, provider, trajectory_text,
                     compress_prompt_path, ctx):
        if ctx.mode == "claude":
            write_event({"stage": "compress-start", "msg": "确定性压缩（秒级，不依赖 LLM）…"})
            compressed_text = deterministic_compress(trajectory_text)
            compressed_path = os.path.join(ctx.output_dir, f"{ctx.output_basename}-compressed.md")
            Path(compressed_path).write_text(compressed_text, encoding="utf-8")
        else:
            compress_log_path = os.path.join(
                ctx.output_dir, f"smart-agent-{ctx.output_basename}-compress.jsonl")
            compress_logger = JsonlLogger(log_path=compress_log_path, cwd=os.getcwd())
            cctx = RunContext(output_dir=ctx.output_dir, output_basename=ctx.output_basename,
                              on_progress=ctx.on_progress, logger=compress_logger, mode=ctx.mode)
            try:
                compressed_path = compress_trajectory(
                    trajectory_text, provider, compress_prompt_path, cctx)
                compressed_text = Path(compressed_path).read_text(encoding="utf-8")
            finally:
                compress_logger.close()
        orig_size = len(trajectory_text.encode("utf-8"))
        comp_size = len(compressed_text.encode("utf-8"))
        ratio = (comp_size / orig_size * 100) if orig_size else 0
        write_event({
            "stage": "compress-done",
            "msg": f"压缩完成：{comp_size / 1024:.0f}KB ({ratio:.1f}%) → {compressed_path}",
            "outputPath": compressed_path,
        })
        return compressed_text

    def do_POST(self):
        if self.path == "/analyze":
            return self._handle_analyze()
        if self.path == "/compress":
            return self._handle_compress()
        if self.path == "/compress-and-analyze":
            return self._handle_compress_and_analyze()
        return self._json(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        # 简化日志，避免 stderr 噪音
        pass

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"status": "ok"})
        return self._json(404, {"error": "not found"})

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _handle_analyze(self):
        req = self._read_json_body()
        if req is None:
            return
        write_event = self._make_writer()
        try:
            trajectory_text = req["trajectoryText"]
            pcfg = req["provider"]
            prompt_md = req.get("promptMd", "")
            output_dir = req.get("outputDir", os.path.join(os.getcwd(), "logs"))
            output_basename = req.get("outputBasename", "session")
            provider = AIProviderConfig(
                base_url=pcfg["baseUrl"],
                api_key=pcfg["apiKey"],
                model=pcfg["model"],
            )
            # Claude Code JSONL 日志：记录完整中间过程
            log_path = os.path.join(output_dir, f"smart-agent-{output_basename}.jsonl")
            logger = JsonlLogger(log_path=log_path, cwd=os.getcwd())
            ctx = RunContext(output_dir=output_dir, output_basename=output_basename,
                             on_progress=write_event, logger=logger)
            result = run_analysis_pipeline(trajectory_text, prompt_md, provider, ctx)
            write_event({
                "stage": "result",
                "outputPath": result.output_path,
                "steps": result.steps,
                "analysis": result.analysis,
            })
        except (AnalysisError, SchemaError) as e:
            write_event({"stage": "error", "msg": str(e)})
        except KeyError as e:
            write_event({"stage": "error", "msg": f"missing field: {e}"})
        except Exception as e:
            write_event({"stage": "error", "msg": f"{type(e).__name__}: {e}"})

    def _handle_compress(self):
        req = self._read_json_body()
        if req is None:
            return
        write_event = self._make_writer()
        try:
            trajectory_text = req["trajectoryText"]
            pcfg = req["provider"]
            output_dir = req.get("outputDir", os.path.join(os.getcwd(), "logs"))
            output_basename = req.get("outputBasename", "session")
            prompt_path = req.get("promptPath",
                os.path.join(PROMPTS_DIR, "trajectory-compress.md"))
            provider = AIProviderConfig(
                base_url=pcfg["baseUrl"],
                api_key=pcfg["apiKey"],
                model=pcfg["model"],
            )
            log_path = os.path.join(output_dir, f"smart-agent-{output_basename}-compress.jsonl")
            logger = JsonlLogger(log_path=log_path, cwd=os.getcwd())
            ctx = RunContext(output_dir=output_dir, output_basename=output_basename,
                             on_progress=write_event, logger=logger)
            output_path = compress_trajectory(trajectory_text, provider, prompt_path, ctx)
            logger.close()
        except KeyError as e:
            write_event({"stage": "error", "msg": f"missing field: {e}"})
        except Exception as e:
            write_event({"stage": "error", "msg": f"{type(e).__name__}: {e}"})

    def _handle_compress_and_analyze(self):
        req = self._read_json_body()
        if req is None:
            return
        write_event = self._make_writer()
        try:
            trajectory_text = req["trajectoryText"]
            pcfg = req["provider"]
            prompt_md = req.get("promptMd", "")
            output_dir = req.get("outputDir", os.path.join(os.getcwd(), "tmp"))
            output_basename = req.get("outputBasename", "session")
            compress_prompt_path = req.get("compressPromptPath",
                os.path.join(PROMPTS_DIR, "trajectory-compress.md"))
            provider = AIProviderConfig(
                base_url=pcfg["baseUrl"],
                api_key=pcfg["apiKey"],
                model=pcfg["model"],
            )
            mode = req.get("mode", "agent")
            ctx = RunContext(output_dir=output_dir, output_basename=output_basename,
                             on_progress=write_event, mode=mode)

            # v4：agent 中心三维度审计。不走压缩；agent-IO 已由 Next.js 路由确定性算好传入。
            if mode == "v4":
                self._ca_v4(write_event, provider, prompt_md, ctx, req.get("agentIo"))
                return

            # 阶段 1：压缩。claude 模式用确定性压缩（秒级，不依赖 LLM，
            # claude code 自带分析能力只需文件变小）；agent 模式用 LLM 压缩
            compressed_text = self._ca_compress(
                write_event, provider, trajectory_text, compress_prompt_path, ctx)

            # 阶段 2：分析（用压缩后文本）
            analyze_log_path = os.path.join(
                output_dir, f"smart-agent-{output_basename}.jsonl")
            analyze_logger = JsonlLogger(log_path=analyze_log_path, cwd=os.getcwd())
            actx = RunContext(output_dir=output_dir, output_basename=output_basename,
                              on_progress=write_event, logger=analyze_logger, mode=mode)
            try:
                result = run_analysis_pipeline(compressed_text, prompt_md, provider, actx)
                # 注入确定性提取的耗时/token 数据（不依赖 LLM，从原始轨迹提取）
                if result.analysis and isinstance(result.analysis, dict):
                    result.analysis = inject_flow_metrics(result.analysis, trajectory_text)
                write_event({
                    "stage": "result",
                    "outputPath": result.output_path,
                    "steps": result.steps,
                    "analysis": result.analysis,
                })
            finally:
                analyze_logger.close()
        except (AnalysisError, SchemaError) as e:
            write_event({"stage": "error", "msg": str(e)})
        except KeyError as e:
            write_event({"stage": "error", "msg": f"missing field: {e}"})
        except Exception as e:
            write_event({"stage": "error", "msg": f"{type(e).__name__}: {e}"})

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8") if length else ""
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON body"})
            return None

    def _make_writer(self):
        # 流式响应头（NDJSON，逐行 flush）
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self._cors()
        self.end_headers()

        write_lock = threading.Lock()

        def write_event(obj: dict):
            line = json.dumps(obj, ensure_ascii=False) + "\n"
            with write_lock:
                try:
                    self.wfile.write(line.encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
        return write_event

    def _json(self, status: int, obj: dict):
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self._cors()
        self.end_headers()
        self.wfile.write(payload)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    logging.info(f"smart-agent server on http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("\nshutting down")
        server.shutdown()
