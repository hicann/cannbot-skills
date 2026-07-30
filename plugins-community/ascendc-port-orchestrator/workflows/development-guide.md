# ascendc-port-orchestrator — 开发/使用流程指南

本插件是 **bundle-orch**：确定性编排引擎打包在 `engine/`，两个入口 skill 作薄 NL 前端。
与其它官方插件不同——流程逻辑不写在 `workflows/` 的提示词里，而在 `engine/` 的可执行 FSM 中。
本文档说明该流程；权威规格见 `workflows/opgen_state_machine.yaml (本目录) + docs/ARCHITECTURE.md` + `docs/ARCHITECTURE.md §8`（bundle-orch 设计与耦合边界）。

## 两项能力（两个入口 skill）

| skill | 用途 | 底层调用 |
|---|---|---|
| `ascendc-cross-gen-port` | 跨代际算子移植（arch22→arch35） | `PYTHONPATH=engine python3 -m orchestrator --port-a3 …` |
| `ascendc-backward-gen` | 正向→反向（梯度）算子生成 | `PYTHONPATH=engine python3 -m orchestrator --backward …` |

两个 skill 都不含流水线逻辑；它们把用户意图翻成引擎参数，引擎跑确定性 FSM。

## 编排流程（引擎 FSM，Phase O0–O6）

1. **O0** hook 完整性门 → **O1** 解析+配置 → **O1.5** DET_POLICY 分类
2. **O2.5** 按模式建立独立真值 → **O3** PROGRESS 初始化
3. **O3.5** phase_prep 自动修复 → **O4** agent 循环（调度 `agents/aog-*`）
4. **O5** 后验 + self-critic → **O6** KB 合并 + commit

## 子 agent（`agents/aog-*.md`，plugin.json 注册）

`aog-kernel-worker`（首发：分析+生成+构建+验证）/ `aog-precision-probe` / `aog-kernel-optimizer` /
`aog-fused-optimizer` / `aog-researcher` / `aog-determinism-analyzer` /
`aog-cann-learner` / `aog-hardware-probe` / `aog-report-gen`。FA/L4 算子也走同一条标准 AscendC
worker → probe → optimizer/research 流，不存在旁路 DSL 或目标实现学习链。

## 双层 KB（c > b）+ 社区 skills 知识源（a）

- **a** = 社区 skills（CBA 路由） · **b** = 插件自带 `references/`（只读，官方/canonical）
- **c** = 用户本地 `~/.ascendc-port/user_kb`（升级安全，`ASCENDC_PORT_USER_KB` 可覆盖）
- 读路径 c>b>a + 冲突显式化；写路径由 `kb_write_root()` 决定（分布式默认写 c）。设计见 `docs/ARCHITECTURE.md §5`。

## 安装

`claude plugin install` 先安装主插件与 marketplace 依赖；随后从
`claude plugin list --json` 取得缓存路径并执行 `init.sh global claude`。`init.sh`
会链接客户 Skills/Agents、注册 Claude 安全网 hooks、创建 c 层 KB 根，并
scaffold `.ascendc_env`。完整命令见 `quickstart.md` 和 `docs/USAGE.md`。
