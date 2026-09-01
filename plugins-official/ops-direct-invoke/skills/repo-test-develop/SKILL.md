---
name: repo-test-develop
description: 仓库测试开发指导，介绍本仓测试框架的使用与测试代码的开发方法。触发：实现 golden、编写分级功能用例、补全白盒测试、搭建性能采集框架、执行精度测试时加载。
---

# 仓库测试开发指导

本 skill 介绍本仓测试工程的使用与测试代码开发方法。

本仓采用 direct launch 工程架构。测试开发按需求文档的**评测来源模式**分两条路径：

- **模式 A（有评测集）**：评测集自带 golden / cases / 算子原型定义，提交方不重写——测试开发职责为对齐评测契约（schema/用例覆盖核对）、补全白盒用例、以评测集评测脚本作为 harness。
- **模式 B（无评测集）**：由 developer-test 自建 golden + 用例 + 测试框架——参考主流评测集（如 cann-bench）的 golden/cases/评测脚本范式搭建，cann-bench 仅作参考实现。

## 按场景路由

| 场景 | 加载 |
|------|------|
| 测试工程组成（模式 A 对齐评测集 / 模式 B 自建 golden+cases+run.sh） | [references/test-framework.md](references/test-framework.md) |
| 黑盒用例设计（模式 B 从需求设计用例 / 模式 A 核对评测集 cases 覆盖并补充） | [references/blackbox-design.md](references/blackbox-design.md) |
| 白盒用例补全（源码分支枚举、尾块/非对齐/tilingkey 覆盖，两模式通用） | [references/whitebox-design.md](references/whitebox-design.md) |
| 精度测试标准与性能采集（模式 A 评测集容差+HAP / 模式 B 自建容差+msprof） | [references/precision-and-perf.md](references/precision-and-perf.md) |

## 依赖的共享 skill

| Skill | 用途 |
|--------|------|
| `ascendc-st-design` | 黑盒用例设计引擎：因子提取→约束→求解→分级 + 覆盖报告；模式 B 用于从需求设计用例，模式 A 用于核对评测集 cases 覆盖与补充设计（见 blackbox-design.md） |
| `ascendc-whitebox-design` | 白盒设计引擎：源码分析→路径枚举→tilingkey 覆盖，供复杂/tilingkey 算子复用（见 whitebox-design.md） |
| `ops-precision-standard` | 各 dtype 的 atol/rtol 精度标准；用于模式 B 及权威源未声明处的取值，模式 A 下以评测框架实际执行判定的口径为最终裁定 |
| `ops-profiling` | msprof op 性能采集、CSV 指标解读；模式 A 性能评测由评测集评测脚本内置 profiler 完成，模式 B 用本 skill 自采 |
| `ascendc-precision-debug` | 精度不达标时诊断根因 |
