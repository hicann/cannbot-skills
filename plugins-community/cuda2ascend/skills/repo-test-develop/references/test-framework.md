# 测试框架与测试开发方法

> 测试工程的**搭建与执行**：golden、gen_data、run、用例表的组织。测试配置（dtype/shape/容差/oracle）以上游测试方案与需求文档为真值源。
> **用例怎么设计**（覆盖维度、等价类/边界/特殊值、分级派生、空 tensor）见 [blackbox-design.md](blackbox-design.md)；**白盒怎么补全**（源码分支枚举、尾块/非对齐/tilingkey 覆盖）见 [whitebox-design.md](whitebox-design.md)；精度标准与性能采集见 [precision-and-perf.md](precision-and-perf.md)。

## 测试工程组成

| 文件 | 作用 |
|------|------|
| `gen_data.py` | 测试数据生成（按 dtype/shape 生成输入与 CPU golden 期望输出） |
| `run.sh` | 运行脚本（编译 + 执行 + 结果比对的入口） |
| golden 实现 | CPU 侧参考实现，产出期望输出，作为精度比对基准 |
| 用例表 | 按 L0/L1/L2 分级组织的 shape × dtype 组合 |
| 性能采集框架 | 跑出各 shape/dtype 的耗时、带宽、利用率 |

## 分级功能用例

三级的**意图与规模**（本仓直调工程执行口径）：

| 级别 | 名称 | 规模/覆盖意图 | 用途 |
|------|------|------|------|
| L0 | 门槛用例 | 常规 shape 与 dtype，用例小、执行快（8-16 元素基础功能） | 开发时简单功能验证 |
| L1 | 功能用例 | 典型 shape、竞品 shape（1K 元素典型场景） | 验证常用功能覆盖完全 |
| L2 | 异常用例 | 超大 shape、空指针、极值/零值等异常输入 | 边界与异常输入验证 |

> **每级用什么覆盖策略选值**（L0 单因子覆盖 / L1 pairwise / L2 每异常一条 + 空 tensor 派生）见 [blackbox-design.md](blackbox-design.md)。分级用例的 shape/dtype 组织可复用 `ascendc-st-design` 引擎产出的因子值表**物化**为本仓用例表，与需求文档保持一致。

## 白盒测试补全

- 以算子代码与已有黑盒测试为输入，基于源码枚举执行分支，补充黑盒未覆盖的分支（尾核/尾块、非对齐 DataCopyPad、多核边界、tilingkey 等），并产出分支覆盖说明。
- 具体方法（源码证据、分支反解 shape、观测 vs 期望覆盖、clean/stress 分离、引擎复用）见 [whitebox-design.md](whitebox-design.md)；分支覆盖达标阈值由验收环节的绑定实现给定。

## golden 实现要点

- golden 为 CPU 侧独立实现，不复用被测 Kernel 逻辑，保证比对独立性。
- golden 输出作为精度比对基准，容差按 dtype 取 `ops-precision-standard` 标准。
- 发现算子疑似缺陷时，以测试暴露问题并回退给算子开发角色，不自行改算子实现。

> 测试代码可执行、可复现是交付底线：用例须能实际运行并给出通过/失败结论，非对齐场景须纳入覆盖。
