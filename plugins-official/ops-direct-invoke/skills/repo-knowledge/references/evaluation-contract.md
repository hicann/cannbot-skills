# direct launch 算子评测契约

> 本仓采用 direct launch 工程架构。direct launch 算子的**评测契约**由评测集约定——本文是该通用契约的领域知识：算子原型 / golden / 用例由评测集提供、提交工程须对齐 schema、三阶段评测（编译/精度/性能）、HAP 性能评分。
>
> 这些契约是 direct launch 算子开发的通用知识，主流评测集（如 cann-bench）均遵循此范式。本文以通用契约为主，**cann-bench 作为参考实现**在文中标注其文件路径供 Agent 按需查阅原文——但不依赖某一特定评测集的转述，因其规则会演进。

> **适用范围（评测来源模式）**：本契约描述**评测集**的评测约定——**模式 A（有评测集）**下，评测集按此契约评测提交工程；**模式 B（无评测集，默认）**下，测试框架由 developer-test 自建，本契约描述其组织范式（cann-bench 作为参考实现 / 参考范式，不是真值源）。模式 B 的自建测试工程组成见 `repo-test-develop` 的 test-framework.md。

## 1. 评测集结构

评测集按算子复杂度分级，每级一个目录。每个算子目录包含**评测输入文件**，由评测集提供，提交方**不修改、不重写**：

| 文件 | 说明 | 提交方角色 |
|------|------|-----------|
| 算子原型定义（如 `proto.yaml`） | schema、输入输出张量、属性参数、dtype | **契约真值源**——提交注册的 schema 必须与之逐字对齐 |
| golden 参考实现（如 `golden.py`） | PyTorch 参考实现，用于精度比对 | 提交方不实现，评测集自带 |
| 测试用例（如 `cases.yaml` / `cases.csv`） | shape × dtype × attrs × value_range | 提交方不实现，评测集自带 |
| 算子说明（如 `desc.md`） | 公式、约束、实现要点 | 需求分析的输入 |
| 性能基线（如 `metadata/<hardware>.json`） | `baseline_perf_us`（baseline 实测）、`t_hw_us`（硬件理论上界） | 性能评分的锚点 |

> **关键**：算子原型定义的 `schema:` 字段是提交工程的注册契约。提交方在 plugin 层 `m.def(...)` 必须与之**逐字一致**（参数名、类型、默认值、返回值），否则评测注册不上算子。
>
> 参考实现：cann-bench 评测集按 4 级难度组织（`tasks/level1~4/<op>/`），Level 1 基础算子、Level 2 中级、Level 3 高级、Level 4 复杂。

## 2. 提交工程契约

提交方产出一个 **direct launch 算子工程目录**，经 `build.sh` 编译出 wheel 包。工程结构以 direct launch 标准模板为准（详见 `repo-op-templates`）：

```
generated_project/
├── build.sh            # 编译入口（--install 时装出 wheel）
├── setup.py            # wheel 打包（ABI3、cmake_build）
├── CMakeLists.txt      # 顶层：bisheng 编译 kernel + g++ 编译 plugin → _C.abi3.so
├── cmake/              # func.cmake(注册宏) + ascend/python/torch/torch_npu.cmake
├── <pkg>/__init__.py   # Python 包：导出 torch.ops.<pkg>.<op>
├── csrc/
│   ├── extension.cpp   # Python 扩展入口（PyInit__C）
│   └── ops/
│       ├── CMakeLists.txt  # 自动发现算子子目录
│       └── <op>/
│           ├── CMakeLists.txt      # 调用注册宏注册
│           ├── op_kernel/
│           │   ├── <op>_kernel.cpp # bisheng 编译：Kernel 实现 + Tiling + Launch extern "C"
│           │   └── <op>_launch.h   # Launch 声明（g++ 可见）
│           └── op_plugin/
│               └── <op>_plugin.cpp # g++ 编译：torch.library 注册 + Meta + NPU 实现
└── tests/              # 提交方自测脚本（评测不依赖）
```

> 包名（如 `cann_bench`）与算子命名空间（如 `torch.ops.cann_bench.<op>`）由评测集约定，提交方须与评测集一致。

**双编译器分工**：

| 文件 | 编译器 | 职责 |
|------|--------|------|
| `op_kernel/*.cpp` | bisheng（`--npu-arch=<arch> -xasc`） | Kernel 实现 + Tiling 计算 + `extern "C"` Launch 函数 |
| `op_kernel/*.h` | bisheng / g++ | Launch 函数声明（供 plugin include） |
| `op_plugin/*.cpp` | g++ | `TORCH_LIBRARY_FRAGMENT(<pkg>, m)` 注册 schema + Meta + NPU 实现 |

## 3. 三阶段评测流程

评测分三阶段（评测集的评测脚本入口，如 cann-bench 的 `scripts/run_evaluation.sh`）：

```
编译正确性 ──▶ 功能精度 ──▶ 性能优化(HAP)
 (w_c=0.2)    (w_f=0.3)    (w_p=0.5)
```

### 3.1 编译正确性

- 提交工程经 `build.sh` 编译出 wheel 并安装。
- **整批编译失败 = 相关算子全部计 0 分**（不隔离、不补救）。一份提交多算子一起编译时，任一编译失败，本次提交涉及算子全部 0 分。
- 编译错误诊断在 `_compile.log`（收至 `reports/build/`）。

> ⚠ 提交前务必本地 `bash build.sh` 验证编译通过——这是最容易因疏忽丢分的环节。

### 3.2 功能精度

- 按算子目录的用例集全量执行，候选输出与 golden 比对。
- 单用例通过/失败按**评测框架的精度判定口径**裁定——该口径是精度的唯一权威，开发期自建容差不能替代它。
- 某用例精度不过只扣该用例得分，不影响其他用例。

#### 权威精度口径：溯源与实例化

评测框架的判定口径必须在**开发期**溯源到具体出处、并落成可执行断言，不能停留在「参照某标准」的引用：

1. **溯源**：按下列优先级定位该算子**每个输出张量**的判定口径（判定指标 + 阈值），并记录出处路径——
   1. 评测框架实际执行判定的实现（比对/评分函数源码）；
   2. 评测集的精度标准规范文档（如 cann-bench `docs/spec/benchmark_spec.md`）；
   3. 算子目录的算子原型定义（如 `proto.yaml` 的精度阈值字段）与算子说明（如 `desc.md`）。

   三者冲突时**以 1 实际执行的口径为准**——规范与说明可能滞后于实现。
2. **逐输出张量成表**：同一算子的不同输出可能走不同口径（如整型输出按逐元素误差判定、浮点输出按相对误差统计量判定），须逐输出张量列「dtype → 判定指标 → 阈值 → 出处」，不合并、不省略。
3. **实例化为断言**：把权威口径实现为开发期可执行的硬断言，逐用例产出**指标实测值与阈值的对照**，而非等评测时才揭晓。

> ⚠ **口径分叉是本契约下最隐蔽的失分点**：自建的宽松容差（如相对误差 < 1e-3）与框架的严格口径（如平均相对误差 < 2^-13 ≈ 1.22e-04）会对同一份误差给出相反结论——本地全绿、评测失分，且失分点直到评测才暴露。凡本地口径与权威口径不一致（指标不同 / 阈值更宽），一律以权威口径为准；宽松容差只能标注为交叉核对项，不作通过依据。

### 3.3 性能优化（HAP）

**HAP（Hardware-Anchored Performance，硬件锚定性能）** 是单用例性能得分，以硬件理论上界 $T_{HW}$ 为锚点：

$$
\text{HAP}_i = \frac{T_{\text{baseline},i} - T_{\text{HW},i}}{(T_{\text{cand},i} - T_{\text{HW},i}) + (T_{\text{baseline},i} - T_{\text{HW},i})}
$$

- $T_{HW}$ = 性能基线的 `t_hw_us`（硬件理论性能上界）
- $T_{baseline}$ = `baseline_perf_us`（baseline 实测）
- $T_{cand}$ = 候选 kernel 实测耗时

**HAP 是饱和型指标，不是加速比**：衡量「候选 kernel 逼近硬件上界的程度」。

| HAP 取值 | 含义 |
|----------|------|
| HAP < 0.5 | 性能低于 baseline（$T_{cand} > T_{baseline}$） |
| HAP = 0.5 | 性能等于 baseline |
| HAP > 0.5 | 性能优于 baseline |
| HAP ≥ 1 | 性能达到或超过硬件理论上界（$T_{cand} \le T_{HW}$），允许超过 100 分 |

**异常取值**：锚点非法（$T_{cand} \le 0$ 或 $T_{HW} \le 0$）或分母 $\le 0$ 时，该用例 HAP 返回 `None`（不计入性能项，不输出 $\pm\infty$ 或负分）。

### 3.4 综合评分

$$
\text{EachOperatorScore} = \left[ w_c \cdot \delta_{\text{pass}} + \sum_{i \in \text{cases}} \frac{\delta_{\text{accuracy},i} (w_f + w_p \cdot \text{HAP}_i)}{|\text{cases}|} \right] \cdot 100
$$

- 编译失败（$\delta_{pass}=0$）⇒ 整算子 0 分。
- 某用例精度不过（$\delta_{accuracy,i}=0$）⇒ 只扣该用例。
- 常规物理有效区间内满分约 100；快于硬件上界允许超 100（不截断）。

## 4. 提交反作弊红线（总览）

评测的是「提交者实现真实 NPU kernel」的能力。以下行为判为无效提交（详见 `repo-coding-rules` 的红线条款，权威原文在评测集的提交规则文档，如 cann-bench `docs/guide/submission_rules.md`）：

1. 调用 PyTorch / torch_npu 内置计算 API 代算（matmul/conv/softmax 等）
2. 用 PyTorch / torch_npu 处理输入输出 tensor（transpose/permute/cast/gather 等实质性搬运）
3. 路由到 CANN 内置同名算子（aclnnXxx / ADD_TO_LAUNCHER_LIST_AICORE）
4. CPU fallback（搬回 CPU 计算）
5. 缓存输出 / 固定输出 / 按输入地址命中
6. 篡改 profiler 或 timing API
7. 返回 FakeTensor / 懒求值包装器 / 伪 Tensor

> 性能测量阶段框架会**轮换输入地址**（每个 repeat 喂独立 clone，`data_ptr()` 不同）——按输入地址命中的缓存会 cache miss 并在精度复检中暴露。

## 5. 评测运行方式

```bash
# 本地（需本机已装 CANN toolkit + torch_npu）
<评测集>/scripts/run_evaluation.sh --source-dir <提交目录> --operator <Op> --no-perf   # 仅精度
<评测集>/scripts/run_evaluation.sh --source-dir <提交目录> --operator <Op>             # 精度+性能

# Docker（评测镜像自带 CANN 环境，harness + tasks 冻结在镜像）
<评测集>/docker/eval/run.sh <提交目录> --operator <Op> --no-perf
```

评测报告输出到 `reports/`，分阶段产出 correctness / performance / 终版报告（json / md / html）。

## 参考实现索引

主流评测集遵循上述通用契约。以 cann-bench 为参考实现，引导 Agent 按需查阅其原文（路径相对于 cann-bench 仓库根）：

| 主题 | 原文路径 |
|------|---------|
| 评测总览 | `README.md` |
| 评测基准规范 | `docs/spec/benchmark_spec.md` |
| 提交规则与反作弊 | `docs/guide/submission_rules.md` |
| 快速入门 | `docs/guide/quick_start.md` |
| direct launch 工程样例 | `examples/direct_launch_example/` |
| 评测任务样例（fixture） | `examples/tasks/` |
| 评测脚本入口 | `scripts/run_evaluation.sh` |
| Docker 评测镜像 | `docker/eval/README.md` |
