# 测试工程组成与开发方法

> 测试工程按需求文档的**评测来源模式**分两条路径：
> - **模式 B（无评测集，默认）**：developer-test 自建 golden + 用例 + 测试脚本，参考主流评测集（如 cann-bench）的 golden/cases/评测脚本范式组织。
> - **模式 A（有评测集）**：评测集自带 golden / cases / 算子原型定义，提交方不重写——测试开发职责为对齐评测契约、补全白盒用例、以评测集评测脚本作为 harness。
>
> **用例怎么设计**（模式 B 从需求设计 / 模式 A 核对覆盖）见 [blackbox-design.md](blackbox-design.md)；**白盒怎么补全**见 [whitebox-design.md](whitebox-design.md)；精度与性能见 [precision-and-perf.md](precision-and-perf.md)。

## 模式 B：自建测试工程（无评测集）

### 测试工程组成

| 文件 | 作用 |
|------|------|
| `gen_data.py` | 测试数据生成（按 dtype/shape 生成输入与 CPU golden 期望输出） |
| `run.sh` | 运行脚本（编译 + 执行 + 结果比对的入口） |
| golden 实现 | CPU 侧参考实现，产出期望输出，作为精度比对基准 |
| 用例表 | 按 L0/L1/L2 分级组织的 shape × dtype 组合 |
| 性能采集框架 | 跑出各 shape/dtype 的耗时、带宽、利用率 |
| `whitebox_cases/` | 白盒补充用例（源码分支覆盖，见 [whitebox-design.md](whitebox-design.md)） |

> 自建测试框架的组织范式参考主流评测集（如 cann-bench 的 golden.py / cases.yaml / run_evaluation.sh），但 golden 与用例由本仓 developer-test 实现，不依赖外部评测集。

### 分级功能用例

三级的**意图与规模**（本仓执行口径）：

| 级别 | 名称 | 规模/覆盖意图 | 用途 |
|------|------|------|------|
| L0 | 门槛用例 | 常规 shape 与 dtype，用例小、执行快（8-16 元素基础功能） | 开发时简单功能验证 |
| L1 | 功能用例 | 典型 shape、竞品 shape（1K 元素典型场景） | 验证常用功能覆盖完全 |
| L2 | 异常用例 | 超大 shape、空指针、极值/零值等异常输入 | 边界与异常输入验证 |

> 每级用什么覆盖策略选值见 [blackbox-design.md](blackbox-design.md)；分级用例可复用 `ascendc-st-design` 引擎产出的因子值表物化为本仓用例表。

### golden 实现要点

- golden 为 CPU 侧独立实现，不复用被测 Kernel 逻辑，保证比对独立性。
- golden 输出作为精度比对基准，容差按 dtype 取 `ops-precision-standard` 标准。
- 发现算子疑似缺陷时，以测试暴露问题并回退给算子开发角色，不自行改算子实现。

## 模式 A：对齐评测集（有评测集）

评测集自带评测输入（如 cann-bench 的 `proto.yaml` / `golden.py` / `cases.yaml` / `metadata/<hw>.json`），提交方只读、不重写：

| 评测集文件 | 提交方角色 |
|-----------|-----------|
| 算子原型定义（如 `proto.yaml`） | **契约真值源**——schema / dtype / attrs，提交注册须逐字对齐 |
| golden（如 `golden.py`） | 评测集自带，不重写——精度比对基准 |
| cases（如 `cases.yaml` / `cases.csv`） | 评测集自带，不重写——评测用例集 |
| 性能基线（如 `metadata/<hardware>.json`） | 性能评分锚点（baseline_perf_us / t_hw_us） |

> 提交方不应在提交工程里另立 golden 或覆盖 cases——评测集评测时以自带的为准。

### 模式 A 自测脚本

提交工程的 `tests/` 目录为可选自测（评测集评测不依赖此目录，用于开发期本地验证）：

| 文件 | 作用 |
|------|------|
| `verify_schema.py` | 验证 `torch.ops.<pkg>.<op>` schema 与算子原型定义一致、可调用 |
| `run_local_eval.sh` | 封装评测集评测脚本（如 cann-bench `run_evaluation.sh --source-dir .. --operator <Op> --no-perf`）做本地精度快验 |
| `whitebox_cases/` | 白盒补充用例（见 [whitebox-design.md](whitebox-design.md)） |

### 模式 A 分级用例核对

评测集 cases（如 cann-bench）每个算子约 20 条开放用例。测试设计的职责是**核对覆盖**而非从零设计：

| 级别 | 评测集 cases 意图 | 提交方核对/补充 |
|------|----------------------|----------------|
| L0 门槛 | 常规 shape/dtype 基础功能 | 核对 cases 是否覆盖算子原型声明的所有 dtype |
| L1 功能 | 典型 shape、常用场景 | 核对 shape 覆盖典型/竞品 shape |
| L2 异常 | 超大 shape、极值/零值 | 核对边界/异常场景是否齐备 |

> 覆盖核对方法见 [blackbox-design.md](blackbox-design.md)。

## 白盒测试补全（两模式通用）

- 以算子代码（`op_kernel/*.cpp`）与已有黑盒用例（模式 B 自建 / 模式 A 评测集 cases）为输入，基于源码枚举执行分支，补充未覆盖的分支（尾核/尾块、非对齐 DataCopyPad、多核边界、tilingkey 等）。
- 白盒补充用例放 `tests/whitebox_cases/`，产出分支覆盖说明。
- 具体方法见 [whitebox-design.md](whitebox-design.md)；分支覆盖达标阈值由 CP3 验收给定。

## golden 同源纪律（两模式通用）

- **模式 B**：golden 为 CPU 侧独立实现，不复用被测 Kernel；本仓仅一份 golden，由数据生成与 torch 通路校验共同引用，保持唯一。
- **模式 A**：golden 由评测集提供，提交方不重写、不复用被测 kernel；自测时引用评测集 golden，不另立第二份。
- **同源截断**：设备侧对输入所做的量化/截断，golden 必须对同一份数据做同样处理——整型造数据须先 round 再 clamp；fp16/bf16 须逐操作数先舍入到该 dtype 再计算。
- 发现算子疑似缺陷时，以测试/评测结果暴露问题并回退给算子开发角色，不自行改算子实现。

> 测试代码可执行、可复现是交付底线：模式 B 须能跑通 `run.sh`；模式 A 须能跑通评测集评测脚本或至少 `verify_schema.py`（schema 对齐）。
