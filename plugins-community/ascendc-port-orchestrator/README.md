# ascendc-port-orchestrator

**跨代际 AscendC 算子移植插件。** 两项能力，两个入口 skill，共享一条确定性流水线 + 安全网 + 双层 KB 反馈环：

1. **`ascendc-cross-gen-port` — 跨代际算子移植**：把 AscendC 算子从来源架构移植到目标架构/产品。**当前支持 arch22 → arch35**（如 Ascend910C/V220 → Ascend950PR/V300）；规划支持更多目标架构/产品与**反向跨代移植**（如 910C→910A）。
2. **`ascendc-backward-gen` — 正向→反向生成**：由正向算子自动生成反向（梯度）算子，真值由 CPU/fp64 autograd 生成。

**入口**：用户用自然语言指定目标架构/产品（`arch35` / `950PR` / `A5` / SoC 编号 / 代际皆可）；**来源架构由代码分析自动识别**，无需指定。

## 架构

确定性流水线（状态机）+ 安全网（防作弊/防退化）+ 双层 KB 反馈环（用户本地 KB > 插件自带 KB > 社区 skills）。详见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## ⚠ 底座依赖与适配路线

CANNBot 与底座 agent harness（OpenCode / Claude Code 等）**不直接耦合**。但本插件由 a5_ops 移植而来，**当前与 Claude Code 耦合**，体现在三点：

1. **skill 格式**：沿用 Claude Code 的 skill 约定。
2. **hook 机制**：流水线的安全网（如完整性门）依赖 Claude Code 的 hook。
3. **sub-agent 调度**：流水线**直接调用 `claude` 命令**拉起子 agent（worker / optimizer / probe 等）。

**因此本插件当前需要 Claude Code 运行时。**

**适配路线（后续工作）**：
- **skill 格式归一**：对齐 CANNBot 的 skill 约定。
- **hook 抽象**：把安全网的 hook 依赖抽象为底座中立接口。
- **sub-agent 调度中立化**：把直接 `claude` 调用改为底座中立的子 agent 调度接口，使插件可在 OpenCode / Claude Code 等底座上运行。

## 知识库

- **用户本地 KB**：用户可写层（格式与维护见 `docs/ARCHITECTURE.md` §5.1）。
- **插件自带 KB**：随 #611 以 OKF 格式交付的 arch/平台经验。

## 安装

见 [`quickstart.md`](./quickstart.md) 与 `init.sh`。本插件处于社区维护阶段，源码位于
`plugins-community/ascendc-port-orchestrator/`。
