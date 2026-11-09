# ArgReduce 范式 CANNBotDSL 能力映射

> **层级**：CANNBotDSL 语言适配层。对 [../../generic/capability-contract.md](../../generic/capability-contract.md) 中全部能力谓词给出 DSL 真值与 API 实体。
>
> **怎么读**：左列是通用层用的"空位名"（见 [../../generic/capability-contract.md](../../generic/capability-contract.md) §2 的白话解释）；中间是 DSL 这边的答案（填什么 API/什么数值）；右列是证据来源。
>
> **真值状态声明**：本表经实战验证（CANNBotDSL argmin 在 Ascend950PR 真机 55/55 用例 bitwise 通过），初版 12 项待验证已全部收敛。

## 1 谓词真值表

| 谓词 | DSL 真值 | 依据 / 备注 |
|------|------|------|
| `INDEX_TRACKING_REDUCE` | **已收敛**：`vreduce_min(src: RawVReg, *, mask: Mask) -> RawVReg`——dst[0]=最小值, dst[1]=最小值索引（原始位模式，按 src 位宽无符号解释）；**文档承诺"多个最小值保留最小索引"（first-occurrence 内置）**；支持 int16/uint16/fp16/int32/uint32/fp32；AIV 生效；Ascend 950PR/950DT 支持（A2/A3 不支持）。**NaN 行为未声明**（文档无 NaN 条款）——NaN 路径一律走显式 `vne` 掩码保守方案，不依赖原生 NaN（AscendC 教训 + DSL 实战验证保守路径正确） | vreduce-min.md（产品支持+功能+约束说明）；workflow7 argmin B1 实现真机 55/55 验证 |
| `VECTOR_COMPARE_SELECT` | **已收敛**：`vne/vlt/vle/vgt/vge/veq`（向量比较族）+ `vnes/vles/...`（标量比较族）+ `vselect(cond_mask, src_true, src_false)`——**掩码逐 lane 一位**，位 1 取 src_true、位 0 取 src_false；`mask_or/mask_and/mask_not/mask_xor` 掩码布尔族。**int16 掩码粒度实测确认**：b16 掩码可同时选择 fp16 值与 int16 位置（同 lane 生效，无错位） | reg_compare/ 全族 + reg_logic/ 掩码族；workflow7 B0 行迭代选择段真机验证 |
| `INDEX_ARITH_OP` | **已收敛**：**首选 `vdups` 常量广播**（支持 int16——广播常量到 b16 向量，实战确认 dtype 保持 int16）+ 标量基址加法（段号/块偏移在 Host 侧算好后传入）；**等价备选 `varange`**（等差数列生成，reg_index/varange.md 支持 int16）——选择依据：段内常量广播 + 标量基址更高效（免逐 lane 增量计算），varange 适合非常量位置模式 | reg_broadcast/vdups + reg_index/varange；workflow7 设计裁剪 varange→vdups 并真机验证 |
| `INDEX_STORAGE_TYPE` | **已收敛**：**等宽原理在 DSL 成立**——fp16 值通道（2B/128 lane@256B 寄存器）配 int16 位置通道（2B/128 lane，同掩码粒度共享）；vselect b16 掩码双通道同 lane 无错位（实战真机确认）。**转换到输出**：B16_TO_B32 展开 + `vcast` int16→int32（含 RoundingMode）+ `vstore` | workflow7 B0 位置通道 vdups int16 → vselect 双通道 → B16_TO_B32 展开 → int32 输出全链路验证 |
| `VECTOR_LANE_WIDTH` | **已收敛**：VL = 256B / 元素宽度——fp16/int16 为 **128 lane**，int32/fp32 为 **64 lane**；可通过 `get_block_num()` 获取核数间接推算 | 950PR 架构 + workflow7 ARA_LANE=128 常量真机验证 |
| `MULTICORE_BALANCE` | **已收敛**：**Host 侧自制**（余数摊派）——`blkFactor = aTotal / usedCoreNum; blkTailFactor = aTotal % usedCoreNum`，前 N 核各 +1；DSL 无内建均衡原语（确认）；核数 `get_block_num()` 运行时获取 | workflow7 Host 切分真机 64 核验证 |
| `ONCHIP_CAPACITY` | **已收敛**：`get_mem_size("ub")` 返回 **253952 B**（Ascend950PR 实测）；单 buffer 上限同源查询；**运行时获取纪律**沿用——严禁硬编码 | host/platform-profiling/get-mem-size.md + workflow7 真机实测 |
| `DATA_ALIGNMENT` | **已收敛**：32 字节（BLOCK_SIZE）搬运/存储对齐；Buffer 构造按 shape×dtype 计占 | workflow7 尾块 padding 验证 |
| `CROSS_CORE_SYNC` | **已收敛（本范式不消费）**：`ops.sync` 含跨核同步类目（`global_sync_all` 等）；ArgReduce 的 A 切分路径不使用跨核同步；GroupReduce 若在 DSL 实现，用 `ops.sync` 栅栏而非调度模式（DSL 无该控制点） | resource-management/global-sync-all；workflow7 不使用 |
| `WORKSPACE_MODEL` | **已收敛（本范式不消费）**：跨核中间存储用 `Buffer(MemLoc.GM, ...)` 表达；A 切分路径无 workspace（核间独立） | — |

## 2 通用术语 → DSL 实体对照

| 通用层术语 | DSL 实体 |
|------|------|
| 值流/位置流双通道 | `ops.reg` 寄存器比较/掩码/广播（fp16 值 + int16 位置等宽双通道，vselect 同掩码选择） |
| 片上 buffer（阶段内临时量） | `Buffer(MemLoc.UB, shape, dtype)` |
| 片上 buffer（生产/消费交接） | `Channel`（acquire/commit/wait/release）——本范式 A 切分路径不使用 |
| 设备侧计算（选择段） | `@kernel` 类 + `@jit` 计算片段 |
| Host 侧轴折叠与切分计算 | 公开入口纯元数据操作（`_check_inputs` / `_fold_axes` / 场景判定） |
| 搬入/搬出 | `ops.reg` `vmask_load` / `vstore`（寄存器级） |
| 位置流生成 | `vdups`（常量广播 + 标量基址）；备选 `varange` |
| 位置→输出转换 | **B16_TO_B32 展开** + `vcast` int16→int32 + `vstore` |
| 多核/架构参数 | `get_block_num()` / `get_block_idx()`（核数/核索引） |
| 编译与启动 | `build_program(scenario, ...)` → `cb.compile` → `ProviderCallable`；`@host` 方法启动 `@kernel` |
| 导出 | `@export("op_name")` + `export_operator()`（变体枚举 + Native 收集发布） |
| 跨核同步 | `ops.sync`（本范式 A 切分不使用） |

## 3 收敛规则（已全部收敛，保留为后续修订指引）

1. **产出声明**：设计/代码产出引用任一谓词支撑的结论时，标注真值来源（文档/实战/推导）
2. **禁止虚假确定**：不得把"职责类目存在"写成"能力已验证"；不得编造 DSL API 签名
3. **NaN 行为强制穿刺**：凡引入 DSL 规约/比较类 API 支撑 NaN 语义的路径，上线前以 NaN 五态做端到端 bitwise 验证（保守路径为默认）
4. **真值修订**：新平台/新版本 cannbot-dsl 发布后，重新核对全部真值并更新来源标注

## 4 关键语言级附加约束

1. **设备计算归属**：选择段（比较/掩码/位置算术）、尾块掩码、输出合成必须落在 DSL kernel 内；禁止在公开入口或 Host launcher 用 `torch` 小算子补全。Host 仅限：shape/dtype 元数据读取、输出分配、零拷贝视图、轴折叠（A1/R/A0 计算）、切分参数计算、launch
2. **场景判定在 Host**：轴折叠与 EMPTY/COPY/AR/ARA 判定是纯 shape 运算，在 Host 入口完成（元数据操作），判定结果作 kernel 分支/参数传入
3. **Channel/Buffer 选择**：值流 tile 等阶段内临时量用 Buffer；本范式 A 切分路径无 Channel 交接
4. **平台参数运行时获取**：核数 `get_block_num()` / UB 容量 `get_mem_size("ub")` 运行时获取，严禁写死
5. **R 超 UB 单块容量时声明义务**：设计必须显式声明处理策略（R 切块循环 / 显式上界限制 + 报错 / 降级路径），不得隐式默认（语言中立原则，见 [../../generic/splitting.md](../../generic/splitting.md) §2）
