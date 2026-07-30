# ascendc-port-orchestrator 快速入门

## 概述
跨代际 AscendC 算子移植插件，两个入口：
- `ascendc-cross-gen-port`：跨代际移植（当前 arch22→arch35）。
- `ascendc-backward-gen`：正向→反向生成。

## 前置条件
- CANN Toolkit（建议 ≥ 9.0.0）、已配置 NPU 设备。

> ### ⚠ 前端支持：当前仅支持 Claude Code
>
> **本插件目前只能在 Claude Code 前端运行**，暂不支持 OpenCode / Cursor / TRAE / Copilot 等其他底座。
> 原因是本插件与其他 cannbot 插件（纯 skill、由前端 LLM 直接读取执行）不同：它**打包了一个确定性编排器引擎**，
> 该引擎在流水线中**直接以 `claude --agent` 命令拉起子 agent**（worker / optimizer / probe 等），并依赖
> Claude Code 的 hook 机制做安全网。因此运行时强依赖 Claude Code CLI。
>
> `init.sh` 可以把 skill/agent 安装到其他前端的配置目录，但**编排器引擎本身无法在没有 `claude --agent`
> 的前端上驱动子 agent**——安装成功 ≠ 可运行。多前端适配是后续单独工作，见 README「底座依赖与适配路线」
> 与 `docs/ARCHITECTURE.md` §8。

## 安装

```bash
claude plugin marketplace add /path/to/cannbot-skills
claude plugin install ascendc-port-orchestrator@cannbot
PLUGIN_INSTALL_PATH="$(claude plugin list --json | jq -r \
  '.[] | select(.id == "ascendc-port-orchestrator@cannbot") | .installPath')"
bash "$PLUGIN_INSTALL_PATH/init.sh" global claude
```

`plugin install` 只复制并注册插件，不会自动执行 `init.sh`。隔离安装时，三条命令必须使用
同一个 `CLAUDE_CONFIG_DIR`。

## 用法
```
# 跨代际移植（自然语言指定目标，来源自动识别）
> 把这个算子移植到 arch35：<算子源/名称>
# 正向→反向
> 为这个正向算子生成反向（目标 a5）：<正向算子>
```
产出：目标算子 + 精度/性能验证报告 + 复现指引。详见 README / docs/ARCHITECTURE.md。
