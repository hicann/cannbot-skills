# SHMEM Ops

面向 Ascend SHMEM 通信算子与通算融合算子开发的 AI Agent 插件。它通过 10 个 skills 组成 Phase 0–7 状态机，覆盖需求确认、方案设计、用例与代码生成、编译验证、代码走读、性能评测和性能优化。

> 本插件用于辅助生成 SHMEM 算子代码与文档，不提供性能、泛化性或生产可用性保证。生成内容必须经过人工审查、编译验证和充分测试后才能用于生产环境。

## 工作流

```text
Phase 0  需求确认
  → Phase 1  设计
  → Phase 2  用例生成
  → Phase 3  代码生成
  → Phase 4  编译 + 正确性验证
  → Phase 5  代码走读
  → Phase 5.5  Torch 接入（条件性）
  → Phase 6  性能基线评估（条件性）
  → Phase 6.5  性能优化（条件性）
  → Phase 7  最终交付
```

端到端任务由 `shmem-ops-dev` 编排。每一阶段的检查点全部通过后才能进入下一阶段；正确性未通过时不能进行性能评测或优化。

条件阶段由 Phase 0 的 intake 控制：

- `meta.torch_required: true`：执行 Phase 5.5 Torch 接入。
- `meta.performance_required: true`：执行 Phase 6 性能评测。
- `meta.performance_auto_optim: true` 且 Phase 6 未达标：进入 Phase 6.5；进入后按配置跑满优化轮次。

更完整的 Phase、输入输出和产物说明见 [skills/README.md](skills/README.md)。

## 安装

项目级安装必须明确给出目标项目目录。推荐使用插件脚本与目标项目的绝对路径：

```bash
# 安装到目标项目的 OpenCode 配置
bash /absolute/path/to/cannbot-skills/plugins-community/shmem-ops-generator/init.sh \
  project opencode \
  /absolute/path/to/target-project

# 安装到目标项目的 Claude Code 配置
bash /absolute/path/to/cannbot-skills/plugins-community/shmem-ops-generator/init.sh \
  project claude \
  /absolute/path/to/target-project
```

其中第三个参数是目标项目目录。若省略，脚本会把当前工作目录作为目标项目；因此仅应在已进入目标项目目录时省略它：

```bash
cd /absolute/path/to/target-project
bash /absolute/path/to/cannbot-skills/plugins-community/shmem-ops-generator/init.sh \
  project opencode
```

全局安装不需要目标项目目录，例如：

```bash
bash /absolute/path/to/cannbot-skills/plugins-community/shmem-ops-generator/init.sh \
  global cursor
```

支持 `opencode`、`claude`、`trae`、`cursor` 和 `copilot`。

## 快速开始

安装后，在目标工作区向 Agent 描述算子和约束。例如：

```text
基于 SHMEM 在 custom-ops 下实现 allreduce，dtype 为 bf16，使用 4 个 PE；
先完成设计、实现、编译与正确性验证，再采集性能并与 HCCL baseline 对比。
```

编排器首先读取 Phase 0 intake 规则，探测消息、环境和仓库中已经明确的信息，只询问缺失项；确认 `phase0_intake` 后才进入设计和实现。

单独处理某个阶段时，也可以直接指定对应 skill，例如：

```text
使用 shmem-ops-performance-eval 对当前 allgather 做性能评测，
并与相同 PE 和数据配置下的 HCCL baseline 对比。
```

## Skills

| Skill | Phase | 作用 |
| --- | --- | --- |
| `shmem-ops-dev` | 0–7 | 端到端编排、阶段门禁和最终交付 |
| `shmem-ops-design` | 1 | 通信算法、PE、内存、同步和核资源设计 |
| `shmem-ops-testcase-gen` | 2 | case matrix、golden、checker 和测试脚本生成 |
| `shmem-ops-code-gen` | 3 | SHMEM 算子代码和工程文件生成 |
| `shmem-ops-compile-debug` | 4 | 构建、运行、失败分类和调试 |
| `shmem-ops-correctness-eval` | 4 | case matrix 执行和正确性评估 |
| `shmem-ops-code-review` | 5、7 | 设计—代码一致性走读和最终审查 |
| `shmem-ops-torch-bind` | 5.5 | 条件性 PyTorch CustomClass 接入 |
| `shmem-ops-performance-eval` | 6 | HCCL/aclnn baseline、时延、带宽和瓶颈评测 |
| `shmem-ops-performance-optim` | 6.5 | 正确性门禁下的多轮性能优化 |

## 边界

| 项目 | 约束 |
| --- | --- |
| SHMEM 核心库 | 不修改 `src/`、`include/` 等核心库目录；交付内容限定在用户指定的算子工程或示例目录 |
| 正确性 | 不承诺一次生成即可通过，必须执行完整 case matrix 并保留失败证据 |
| 性能 | 不承诺达到特定带宽或时延目标，结论必须来自实际测量 |
| Torch | 仅在用户启用 `torch_required` 时执行，绑定与测试仍须独立验证 |
| 生产使用 | 必须经过人工审查、环境匹配和完整测试 |

详细规则、环境契约、测试口径和代码模板位于各 skill 的 `SKILL.md`、`references/` 和 `templates/` 中。
