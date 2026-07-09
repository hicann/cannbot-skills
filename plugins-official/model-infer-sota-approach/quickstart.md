# NPU 推理优化编排（baseline 之上）快速入门

## 概述

`model-infer-sota-approach` 是 NPU 模型推理「baseline 之上探索式优化」编排 plugin。它在一个**已可运行、可复现精度的 baseline** 之上，通过 `workflows/sota-approach-workflow.md` 由 profiling 数据驱动并行发现优化候选，再用 Plan / round 自循环（实施 → 复核 → 派生 → 淘汰）逐步逼近最优。具体优化下沉调用 `model-infer-*` 单点 skill，本 plugin 只编排不替工。

它与 `model-infer-optimize` 以 baseline 为分界、前后衔接：没有 baseline 先用 `model-infer-optimize` 建立，有了 baseline 再用本 plugin 继续挖潜。

## 一、环境搭建

### Claude Code

**首选：Plugin Marketplace（一键安装）**

```text
# 注册 marketplace（首次，GitCode 仓库需完整 URL）
/plugin marketplace add https://gitcode.com/cann/cannbot-skills.git

# 安装插件
/plugin install model-infer-sota-approach@cannbot
/reload-plugins
```

安装后新开会话，或在当前会话执行 `/clear` 触发插件上下文加载。`model-infer-sota-approach` 是主对话入口，会把 AGENTS.md 注入上下文，并按强制规则读取 `workflows/sota-approach-workflow.md`。

**备选：init.sh 脚本**

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/model-infer-sota-approach
bash init.sh project claude     # 项目级
bash init.sh global claude      # 全局级
```

### OpenCode / Trae / Cursor / Copilot

```bash
git clone https://gitcode.com/cann/cannbot-skills.git
cd cannbot-skills/plugins-official/model-infer-sota-approach
bash init.sh project opencode   # 或 trae / cursor / copilot；global 同理
```

`init.sh` 会安装 skills/workflows、写 manifest，并把 `cann-recipes-infer` 参考仓 clone 到插件目录，使工作流里 `cann-recipes-infer/...` 前缀的路径可解析。

### 验证安装

```bash
# Claude Code
claude plugin list
# 应看到 model-infer-sota-approach@cannbot ✔ enabled
```

## 二、快速上手

在目标 `cann-recipes-infer` 或模型仓中，确认已有可运行 baseline 后提出需求：

```text
帮我在 deepseek-r1 现有 baseline 上继续做 NPU 推理的多方向探索式优化
```

primary agent 会按 AGENTS.md 中的强制规则自动读取 `workflows/sota-approach-workflow.md`，先确认场景与 baseline，采集并分析 profiling，发现候选并初始化 Plan Dashboard，再进入 Plan 实施 / review / 派生循环，最后做同口径最终验收。

## 三、核心工作流（8 步）

```text
1. 确认推理场景与（可选）性能目标
    ↓
2. 构造推理输入并跑通精度基线（scenario）
    ↓
3. 采集 baseline profiling / round0（profiling-instrumenter，非交互）
    ↓
4. 分析 baseline profiling（主 agent 敲定拆解 spec → profile-analyzer）
    ↓
5. 候选发现（每个来源并行拉 candidate）
    ↓
6. 初始化 Plan Dashboard（主 agent 唯一写者，裁定互斥/叠加）
    ↓
7. Plan 实施 / review / 派生循环（每 Plan 一个全局 roundN，按需重采）
    ↓
8. 最终验收（与 baseline 同口径对照，以 profile-analyzer 报告为准）
```

一切性能判断以 `model-infer-perf-breakdown` 的分析报告为准，不以裸计时下结论。

## 四、安装内容

| 内容 | 说明 |
| --- | --- |
| 原子 skills（14 个） | 来自 `model/model-infer-*`，覆盖采集、性能拆解与各专项优化能力 |
| workflow 文档 | `plugins-official/model-infer-sota-approach/workflows/sota-approach-workflow.md` |
| references / templates | 场景细则、裁决规则、subagent 模板、Plan Dashboard / plan / progress 模板 |
| 配置入口 | `AGENTS.md` / `CLAUDE.md`，强制读取 `workflows/sota-approach-workflow.md` |

## 五、与 model-infer-optimize 的分工

| 场景 | 推荐方式 |
|------|---------|
| 模型尚未适配 / 还没有 baseline | 先用 `model-infer-optimize`（从零到 baseline 的基础工作流） |
| 已有可运行 baseline，要继续多方向探索式优化 | 用本 plugin（`帮我在现有 baseline 上继续优化 …`） |
| 只需某一种单点优化（KVCache / 融合 / 多流 …） | 直接调用对应 `model-infer-*` skill，不进入编排流程 |

## 六、常见问题

### Q: 如何查看帮助信息？

```bash
bash init.sh --help
```

### Q: 如何更新？

```bash
# Claude Code
/plugin update model-infer-sota-approach@cannbot

# OpenCode (init.sh 方式)
cd cannbot-skills/plugins-official/model-infer-sota-approach && bash init.sh
```

---

## 总结

1. 本 plugin 在 baseline 之上做 profiling 驱动的探索式优化，通过 `workflows/sota-approach-workflow.md` 编排 8 步流程
2. Claude Code 用户用 `/plugin install` 一键安装，OpenCode/Trae/Cursor/Copilot 用户用 `init.sh`
3. 没有 baseline 先用 `model-infer-optimize`，两者前后衔接、互补不替代
4. 一切性能判断以 perf-breakdown 报告为准，Plan 状态机驱动收敛，支持跨上下文接力
