# Catlass 代码审查参考手册

> 本文件由 Reviewer 在 Step 3（代码质量评估）时读取，逐项对照检查。包含通用 Ascend C 检视项 + catlass 专属检视项 C1–C11。

---

## catlass 专属检视项（C1–C11）

| # | 检查项 | 检查方法 | 严重级别 |
|---|--------|---------|---------|
| C1 | `{operator_name}` 含 `catlass` 子串（snake_case），CamelCase 一致映射 | 文件名 / namespace / 类名 grep | 阻塞 |
| C2 | catlass 源码位于工作区根 `./catlass/`，**未**克隆到 `operators/{operator_name}/` 内 | `find operators -name catlass -type d` 应不命中 | 阻塞 |
| C3 | CMakeLists.txt 注入 `-I<catlass>/include` + `-DCATLASS_ARCH=<arch>` | `verify_cmake_config.py` | 阻塞 |
| C4 | op_kernel **禁用** `DeviceGemm` 适配器；必须直接 `Kernel` + `Kernel::Params` + `Kernel{}(params)` | `grep -n DeviceGemm op_kernel/*.asc` 应不命中；`grep -n 'Kernel{}'` 应命中 | 阻塞 |
| C5 | op_kernel **禁止**自实现矩阵乘 / 逐元素 / 拷贝循环 | 目视 + grep 标量循环模式 | 阻塞 |
| C6 | 必须 `AscendC::GetUserWorkspace(workspace)`；**禁用** `SetSysWorkspaceForce` | `grep -n 'GetUserWorkspace\|SetSysWorkspaceForce' op_kernel/*.asc` | 阻塞 |
| C7 | op_kernel **禁止** `#include` 算子自身的 tiling 实现文件（仅可 include 共享 POD `*_tiling.h`） | `grep -n '#include.*tiling' op_kernel/*.asc` | 阻塞 |
| C8 | TilingKey 分支实例化与 DESIGN.md §2.1 列出的合法组合一致 | 对照 DESIGN.md | 高 |
| C9 | 测试 shape 满足 catlass 运行期约束（避免过小 M/N，选 L1 分块整数倍） | 阅读 `scripts/gen_data.py` / Level 0–2 用例 | 高 |
| C10 | catlass 拼装类 `using` 与 DESIGN.md §1.2 选型表一致 | 对照 op_kernel 顶部 namespace | 高 |
| C11 | 调优阶段已加载 `/catlass-op-perf-tune`，PRE/POST 报告归档至 `docs/perf/round_NNN/` | 检查 `perf/` 目录 | 中 |

### Grep 速查命令

```bash
# C1：命名校验
grep -rn "catlass" operators/{operator_name}/op_*/*.asc | head
grep -n "namespace NsCatlass" operators/{operator_name}/op_kernel/*.asc

# C2：源码位置
find operators -maxdepth 3 -type d -name catlass
ls -d ./catlass/include ./catlass/examples

# C4：禁用 DeviceGemm，必须 Kernel{}(params)
grep -n "DeviceGemm" operators/{operator_name}/op_kernel/*.asc
grep -n "Kernel{}(params)\|Kernel{}\s*(" operators/{operator_name}/op_kernel/*.asc

# C5：禁用自实现循环
grep -n "for\s*(" operators/{operator_name}/op_kernel/*.asc | head -20

# C6：Workspace
grep -n "GetUserWorkspace" operators/{operator_name}/op_kernel/*.asc
grep -n "SetSysWorkspaceForce" operators/{operator_name}/

# C7：include 边界
grep -n "#include" operators/{operator_name}/op_kernel/*.asc | grep -i tiling
```

---

## 通用架构合规性检查

| 检查项 | 标准 | 严重级别 |
|--------|------|----------|
| 入口属性 | `__global__ __aicore__` | 高 |
| 函数定义顺序 | Kernel 函数定义在调用之前，无前向声明 | 高 |
| 代码结构 | catlass 拼装类 → kernel 入口 → host 函数 → main | 中 |

## 通用编码规范检查

| 检查项 | 标准 | 严重级别 |
|--------|------|----------|
| catlass 拼装一致性 | 与 DESIGN.md §1.2 选型表逐字一致（C10） | 高 |
| 数据对齐 | host Tiling 计算的 strides / leading dims 满足 catlass layout 要求 | 高 |
| 硬件参数 | 动态获取核数；切分维度由 BlockScheduler 决定，禁止写死 blockDim/blockIdx 数值 | 高 |
| 命名规范 | snake_case ↔ CamelCase 一致映射（C1） | 中 |

## 性能分析检查

### 流水线（DispatchPolicy）

| DispatchPolicy | 适用场景 |
|----------------|---------|
| `MmadAtlasA2Pingpong<true>` | A/B 双 Pingpong，最大化流水重叠（典型默认） |
| `MmadAtlasA2Preload<true>` | 提前预取 B（K 较大时） |
| `MmadAtlasA2FullLoadA*` / `FullLoadB*` | 一侧矩阵全量驻留 L1（小矩阵） |

| 检查项 | 说明 | 问题级别 |
|--------|------|---------|
| DispatchPolicy 与算子形态匹配 | M/N/K 形态、A/B 大小判断 | 高 |
| Swizzle 选择 | `GemmIdentityBlockSwizzle<3, 0>` 是否合理 | 中 |

### 同步策略（自定义 Tile 内）

> catlass 内置 BlockMmad / BlockEpilogue 已由模板保证内部同步；只有 op_kernel 内自定义 Tile / 桥接代码需要分析。

对每个自定义 Tile 内的 `PipeBarrier`，按 ops-direct-invoke 同款规则执行逐项依赖分析（前操作 / 前 Pipe / 后操作 / 后 Pipe / 依赖类型 / 判定）。

### 上板性能验证

**独立采集**：调用 `/ops-profiling`，独立执行 msprof op 采集。

**审查要点**：
1. Task Duration 与 catlass 同形态 example（如 `catlass/examples/00_basic_matmul/`）的差距，差距 <30% 视为达标
2. PipeUtilization 分布：
   - GEMM 算子 cube 利用率 > 50%
   - Matmul + Epilogue 算子 vector 利用率不应过低
3. 核间负载均衡（各核耗时差异 <10%）
4. 调优场景：PRE/POST 单变量变更证据是否齐全（C11）

**与 Developer 性能数据对比**：读取 `operators/{operator_name}/docs/perf/` 目录下 Developer 数据，与 Reviewer 独立采集结果对比，差异过大需在 REVIEW.md 中说明。

---

## API 选择审查

### catlass 拼装类型选择

| 组件 | 应在哪里选型 | 检查点 |
|------|------------|--------|
| ArchTag | DESIGN.md §1.2 | 与 SoC 一致（A2/A3/A5） |
| DispatchPolicy | DESIGN.md §1.2 + §2.1 | 与算子形态、TilingKey 分支一致（C10） |
| BlockMmad / BlockEpilogue / BlockScheduler / Kernel | DESIGN.md §1.2 | 类型组合在 catlass header 中存在 |

### 数据搬运 / 计算 API（自定义 Tile 内）

适用 ops-direct-invoke 同款规则：
- GM↔UB 用 `DataCopy`（对齐）/ `DataCopyPad`（非对齐）
- UB→UB 用 `Copy`，禁止 `Copy` 操作 GM
- 矢量计算用基础矢量 API（Add/Mul/Sub/Div/Exp/Log/ReduceSum/ReduceMax/Cast）
- **禁用** Softmax / LayerNorm 等高阶封装
- 禁止 GetValue/SetValue 逐元素操作

---

## 调试代码检查（最终轮专用）

| 检查项 | Grep 命令 | 问题级别 |
|--------|----------|---------|
| printf 残留 | `grep -n "printf\|cout" operators/{operator_name}/op_*/*.asc` | 低（最终轮要求清理） |
| TODO/FIXME | `grep -n "TODO\|FIXME" operators/{operator_name}/op_*/*.asc` | 低 |
| 调试用硬编码 | 目视 | 低 |
