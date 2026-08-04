# smart-agent

trajectory-analyzer 的 Python 版（功能与 `src/lib/ai/trajectory-analyzer.ts` 对齐）。

## 文件

| 文件 | 职责 |
|------|------|
| `trajectory_parser.py` | 正则提取器：骨架 / skill_content / stats / 门控 / 异常 / readSection |
| `trajectory_analyzer.py` | 有界 LLM 循环 + schema 校验 + 写 JSON |
| `server.py` | stdlib HTTP 服务（免装 fastapi），POST /analyze |
| `analyze.py` | CLI 入口（离线分析单文件） |
| `tests/test_trajectory.py` | 17 用例（真实 log1.md + mock LLM 循环） |

## 运行

```bash
pip install -r requirements.txt   # 仅 requests
pip install pytest                # 跑测试用

# 测试
python -m pytest tests/ -v

# 起服务（默认 21026 端口）
python server.py
# 或指定端口
CANNBOT_AGENT_PORT=21027 python server.py

# 离线分析单文件
CANNBOT_LLM_API_KEY=sk-xxx python analyze.py /path/to/log1.md --model qwen3.7-max
```

## 与前端联动

Next.js 代理路由 `/api/ai/audit-session-py` 会把当前 session 的 MD 文本 POST 到 `http://localhost:21026/analyze`。Audit tab 的 v1/v2 开关切换走 TS 版（v1）或 Python 版（v2）。

环境变量 `CANNBOT_AGENT_URL` 可覆盖代理目标（默认 `http://localhost:21026`）。

## API

```
POST /analyze
body: {trajectoryText, provider:{baseUrl,apiKey,model}, promptMd?, outputDir?, outputBasename?}
→ 200 {outputPath, rounds, analysis}
→ 400/500 {error}

GET /health → {status: ok}
```

## 与 TS 版的差异

- **仅 MD 文本入口**：Python 不接 Prisma，由 Next.js 生成 MD 文本后转发。
- **HTTP 库**：`requests` 替代 `fetch`，2 次重试 + 2s 间隔逻辑一致。
- **正则**：`re.search` 对齐 JS `String.match` / `RegExp.test`，输出数值与 TS 版逐字节一致。
