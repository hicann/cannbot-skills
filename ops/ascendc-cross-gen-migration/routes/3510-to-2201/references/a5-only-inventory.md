# A5 专有能力清单与 A2 处置

> 阶段 1 的主表：**逐项识别 A5 专有依赖，给出 A2 处置与验证方式**。`scripts/scan_a5_only.py` 覆盖了本表大部分识别信号；扫描器没覆盖到的（注释、宏拼接、动态字符串）靠人工补。

## 0. 严重级定义

| 级别 | 含义 | 处理 |
|---|---|---|
| `blocking` | 不处理则 A2 编不过或必错 | 阶段 2 必须消除，GATE-2 要求 `RESIDUAL=0` |
| `decision` | 有多个可行方案，但语义/精度/功能会变，必须用户确认 | 阶段 1 出方案与影响分析，用户签字后执行 |
| `verify` | 是否 A2 专有取决于 CANN 版本/头文件 | 阶段 1 现场查证；查不到按 `blocking` 处理 |
| `advisory` | 不阻塞，但影响性能或可维护性 | 阶段 5 复核，报告记录结论 |

## 1. 编程模型

| 能力 | 识别信号 | A5 语义 | A2 处置 | 级别 |
|---|---|---|---|---|
| **RegBase / MicroAPI** | `__VEC_SCOPE__`、`MicroAPI::`、`RegTensor<`、`MaskReg`、`CreateMask<`、`UpdateMask<`、`LocalMemBar<`、`LoadDist::`、`StoreDist::`、`CAST_B16_TO_B32` 等 CastTrait | 数据进向量寄存器计算，段内由 MicroAPI 管理同步 | **重写为 MemBase**：`LocalTensor` + `DataCopy/DataCopyPad` + count/mask 语义 + `SetFlag/WaitFlag`/`PipeBarrier`。见 `references/api-migration-recipes.md` §2 | `blocking`（D3） |
| **SIMT** | `__simt_vf__`、`Simt::VF_CALL`、`AscendC::Simt::`、`asc_syncthreads`、`asc_threadfence`、`LAUNCH_BOUND(`、`GetThreadIdx`、`GetThreadNum`、`__local_mem__` | 标量线程 + Warp 调度，适合 scatter/gather、索引密集路径 | **必须重写**（二选一）：① SIMD 重写（Gather/Scatter/排序归并/分块搬运）；② AICPU 实现（索引极不规则、计算量小）。若两路都不可行，按 `BLOCKED` 记录证据并取得用户确认，**不得仅以“声明不支持”结案**。见 §3 决策树 | `blocking`（D3） |

## 2. 数据类型

| 能力 | 识别信号 | A2 现状 | 处置 | 级别 |
|---|---|---|---|---|
| **FP8（e4m3 / e5m2）** | `fp8_e4m3`、`fp8_e5m2`、`float8_e4m3`、`float8_e5m2`、`FLOAT8_E4M3`、`DT_FLOAT8_*` | A2 无 FP8 计算通路（FP8 是 DAV_3510 新增数据格式，见 `npu-arch`）；个别 CANN 版本可能提供 fp8→fp16 的 Cast，但仍需转换后计算 | 三选一：① 反量化到 bf16/fp16 计算（精度↑、带宽↑）；② 走 int8 量化路径（需 scale/zero-point 语义）；③ 不支持该 dtype。必须记录误差与性能影响 | `decision`（D2） |
| **HiFloat8** | `hifloat8`、`HiFloat8`、`DT_HIFLOAT8` | 同上，A2 无 HiF8 计算通路 | 同 FP8；注意 HiF8 动态范围与 fp8 不同，不能按 fp8 处理 | `decision`（D2） |
| **MXFP8 / MXFP4** | `mxfp8`、`mxfp4`、`MXFP8`、`MXFP4`、`mxfloat8` | A2 无块缩放（block-scaled）格式 | 语义上不等价于 per-tensor 量化：必须重新设计量化粒度（per-channel/per-token），并向用户确认是否接受语义变化 | `decision`（D2） |
| **int4 / 4:2 稀疏** | `int4b_t`、`Sparse`、`4:2` | A2 **支持**（A5 不支持） | 属于"A2 恢复项"：若 A5 代码为规避不支持而 cast 成 int8 或删掉稀疏，可评估回退（阶段 5） | `advisory` |

> dtype 处置的共同约束：**禁止 reinterpret**（把 fp8 指针当 bf16 读）。必须先转换再计算，且转换路径要进精度用例。

## 3. SIMT 替代决策树

```
SIMT 逻辑是否本质是"每个元素独立、无跨线程通信"的 map/reduce？
├─ 是 → 能否用 DataCopy + 向量 API 表达（含 GatherMask / 排序 / 分段归约）？
│        ├─ 是 → SIMD 重写（首选，性能与可维护性最好）
│        └─ 否 → 索引是否可用 Brcb/ArithProgression/GatherB 生成？
│                 ├─ 是 → SIMD + 索引生成重写
│                 └─ 否 → AICPU 实现
└─ 否（依赖 warp shuffle / 线程间共享 / 原子累加顺序）→ AICPU 实现，或与用户确认改为确定性算法
```

选择 AICPU 时 MUST 说明：额外 kernel 启动开销、与 AIC/AIV 的流水关系、性能预期，并在阶段 5 实测。

## 4. 数据通路

| 能力 | 识别信号 | A5 语义 | A2 替代路径 | 级别 |
|---|---|---|---|---|
| **L0C→UB 直通** | `isToUB`、`FixpipeConfig`、`dualDstCtrl`、`subBlockId`、`TPosition::VECIN` 作为 C 位置 | Cube 结果直接落 UB，省 GM 往返 | **L0C→GM（workspace）→UB**：`Fixpipe` 写 workspace，再 `DataCopy` 回 UB；或降低 base 块+分段回写。同步点相应增加一次 | `blocking`（D2） |
| **UB→L1 直通** | `DataCopy` 的 UB→L1 路径、相关 `TPosition` 组合 | 免 GM 中转装 L1 | **UB→GM→L1** 两段搬运；或改数据组织让 L1 直接从 GM 装载（`DataCopy` GM→L1） | `blocking`（D2） |
| **L1→GM / GM→L0A/L0B** | 反向信号：A5 代码里"没有"这些调用 | A5 删除，A2 **存在** | 迁移可**恢复**这些通路简化数据流（阶段 5 优化项） | `advisory` |
| **SSBuffer 跨核参数** | `SSBuffer`、`SSbuf`、`CrossCoreSetFlag<4` + 标量读取 | 零拷贝小参数通道 | 改 **GM workspace 传参** + `CrossCoreSetFlag/WaitFlag`（模式 2）或普通 `SetFlag/WaitFlag`；注意对齐与可见性（`DataCacheCleanAndInvalid` 等按平台要求） | `blocking`（D2） |
| **NDDMA（高维 DMA）** | `NDDMA`、多维 stride 描述符 | 一条指令完成高维搬运 | 拆成 **循环 + `DataCopy`/`DataCopyPad`**，或预处理成连续布局；循环层数多时评估是否先在 GM 做一次重排 | `blocking`（D2） |
| **CCU 通算融合** | `CCU`、KFC 调度改造 | 通信与计算融合 | A2 无对应能力：改为 GM/workspace 交换 + 显式同步，或判定该算子不适合 A2 并上报 | `blocking`（D2） |

## 5. 同步与控制

| 能力 | 识别信号 | A5 语义 | A2 替代 | 级别 |
|---|---|---|---|---|
| **BufferID 同步** | 代码中不出现 `SetFlag/WaitFlag` 配对，改由 buffer 依赖推导 | 由硬件按 BufferID 保序 | 显式 `SetFlag<HardEvent::X_Y>` / `WaitFlag<HardEvent::X_Y>` 配对；`PipeBarrier` 用于同 pipe 保序。**逐点核对事件类型与 flagId 不冲突** | `blocking`（D2） |
| **同步模式 4** | `CrossCoreSetFlag<4, PIPE_S>`、`CrossCoreWaitFlag<4, PIPE_S>` | AIC↔单个 AIV 细粒度同步（仅 950） | 改 **模式 2**（标准跨核同步）或 `SyncAll`；重新检查参与集合与死锁风险 | `blocking`（D2） |
| **Mutex** | 锁/互斥相关调用（以 A5 头文件为准） | 核内异步流水间互斥 | 用 `SetFlag/WaitFlag` 事件依赖表达先后，避免真互斥 | `blocking`（D2） |
| **溢出模式控制** | `GetCtrlSpr`、`SetCtrlSpr`、`GLOBAL_OVERFLOW_MODE_CTRL` | 关溢出检测提吞吐，需保存/恢复 | **A2 无此 SPR 能力**：删除调用（及其宏），不得保留空实现；若依赖它规避溢出，改为算法/数据类型层面处理 | `blocking`（D2） |

## 6. Cube 侧

| 能力 | 识别信号 | A5 语义 | A2 处置 | 级别 |
|---|---|---|---|---|
| **Fixpipe 参数体系** | `FixpipeParamsArch3510`、`FixpipeParamsC310` | 新字段（nSize 16 倍数、srcStride 单位 C0_SIZE、dstStride 单位 element） | 改 **`FixpipeParamsV220` + `Fixpipe`**（或 `DataCopyCO12DstParams`，按目标 CANN 版本）；字段单位不同，**必须重写参数，不能只改类型名** | `blocking`（D2） |
| **L0A 分形** | L0A 装载/布局相关参数、`LoadData2DParams` | NZ | **ZZ**：重算分形与 `srcStride/dstStride` | `blocking`（D2） |
| **LoadData 扩展** | `LoadData2DParamsV2`、`LoopMode`、MX 模式参数 | mStep/kStep 字段体系 | 改 `LoadData2DParams`（repeatTimes/srcStride）并重算；或改 `LoadDataWithTranspose` 等 A2 可用形态。**是否 A2 存在 `V2` 必须查头文件** | `verify`（D2） |
| **双主模式** | `enableMixDualMaster`、`EnableMixDualMaster` | AIC/AIV 独立推进 | 删除该配置，回到标准 MIX 消息驱动；重新审查原依赖"双主"的自定义同步 | `blocking`（D2） |
| **低阶 AIC 直跑** | 直接 `Mmad`/`LoadData`/`Fixpipe` 手工流水 | A5 上替代高阶 Matmul | A2 也可低阶直跑（能力存在），但 A2 的同步/分形/参数体系不同：**保留形态、重写参数与同步**，或回退高阶 Matmul API | `verify`（D2） |

## 7. 工程接线

| 项 | A5 形态 | A2 形态 | 级别 |
|---|---|---|---|
| kernel 目录 | `op_kernel/arch35/` | `op_kernel/arch22/`（内容必须真的 arch22 可用） | `blocking`（D1） |
| 架构宏 | `__NPU_ARCH__ == 3510`、`__CCE_AICORE__ == 310` | `__NPU_ARCH__ == 2201`、`__CCE_AICORE__ == 220` | `blocking`（D1） |
| soc 字符串 | `ascend950` | `ascend910b`（910_93 用 `ascend910_93`，以工程先例为准） | `blocking`（D1） |
| 原型配置 | `AddConfig("ascend950", cfg)` + `opFile.value` | `AddConfig("ascend910b")`（910_93 另配）；删除 A5 专属 `opFile` 切换 | `blocking`（D1） |
| config 目录 | `op_host/config/ascend950/` | `op_host/config/ascend910b/`（`*_binary.json`、`*_simplified_key.ini`） | `blocking`（D1） |
| CMake | `SUPPORT_COMPUTE_UNIT "ascend950"`、`SUPPORT_TILING_DIR "arch35"` | `"ascend910b"` / `"arch22"` | `blocking`（D1） |
| tiling 源码 | `*_tiling_arch35.cpp` | `*_tiling_arch22.cpp`（或公共实现，按仓约定） | `blocking`（D1） |
| 构建命令 | `--soc=ascend950` | `--soc=ascend910b` | `blocking`（D1） |
| A5 专属判断 | `IsRegbaseSocVersion`、`SIMT_UB_SIZE_BYTE` 预留 40KB | 删除 RegBase 分支与 40KB 预留；UB 预算按 `GetCoreMemSize(UB)` 重算（典型 192KB） | `blocking`（D1/D2） |

## 8. 精度相关（A5 侧规避手段，迁移时反向处理）

| 形态 | 识别信号 | A5 为什么这么做 | A2 处置 | 级别 |
|---|---|---|---|---|
| FTZ 软件模拟 | `PRECISION_1ULP_FTZ_FALSE`、`PRECISION_0ULP_FTZ_FALSE`、`XxxSpecificMode`、`XxxConfig = { ...Algo... }` | A5 硬件 FTZ，需软件补 subnormal | **删除模板参数/Config，回到默认 API**（A2 原生支持 subnormal），并用 subnormal 用例验证；若用户要求跨平台行为一致，保留 eps 保护但记录性能代价。Config 家族（`LnConfig` 等）先按 `verify` 查证目标 CANN 是否存在同名重载 | `blocking`（D2） |
| eps 规避 | `+ 1e-38`、`+ epsilon`、分母保护常量 | 避免输入落 subnormal | 评估是否仍需：保留会引入偏差；删除可提精度与性能。**必须做 dtype 可表示性检查**（fp16 的 1e-8 会下溢为 0） | `decision` |
| high_precision 分支 | 按 tiling/属性切换精度模式 | A2 上该分支无意义 | 删除分支与对应 tilingKey，简化路径（除非用户要求保留跨平台一致的精度模式） | `advisory` |
| 精度断言 | 代码/测试里对 A5 残差写死阈值 | A5 行为 | 阈值统一以 `ops-precision-standard` 重设，禁止沿用 A5 专门放宽的阈值 | `blocking` |

## 9. A2 恢复项（迁移带来的"新增能力"，阶段 5 评估）

A5 上被删除或降速、A2 上可用的能力，迁移后可能**比 A5 实现更简单或更快**。不要在阶段 2 盲目复刻 A5 的规避写法，阶段 5 逐项评估是否恢复：

| 恢复项 | A2 能力 | 典型收益 |
|---|---|---|
| int4 Mmad / 4:2 稀疏 | A2 支持 | 量化 matmul 省 cast、省带宽 |
| L1→GM、GM→L0A/L0B | A2 支持 | 少一次 GM 中转，简化流水 |
| `SetLoadDataBoundary` | A2 支持 | 边界处理更直接 |
| ZZ 分形 + `FixpipeParamsV220` | A2 原生 | 参数直观、样例多 |
| 高吞吐 `vnchwconv`（`TransDataTo5HD`） | A2 吞吐高 | 若 A5 代码为规避降速改成了手写转置/Copy 重组，A2 上可回退到 `TransDataTo5HD` |
| VSLDB / 归约路径 | A2 上 VSLDB 吞吐约为 A5 的两倍 | 若 A5 代码为规避 R=16/24 退化做了特殊 tiling，A2 上可回退常规 tiling |
| 原生 `BilinearInterpolation` | A2 为融合实现 | 若 A5 代码改成了手写 Gather+乘加循环，A2 上可回退 API |
| L2 更大（典型 192MB vs 950PR 112/128MB） | A2 缓存更宽 | 可考虑把 A5 上"为省 L2 而拆得很碎"的 tiling 合并 |

> 恢复项必须**逐项 A/B 实测**，禁止"理论上更好"直接改。判定与回退规则见 `references/performance-tuning-a2.md` §5。

## 10. 清单使用方式

1. 用 `scripts/scan_a5_only.py --root <工程> --format md --out scan.md` 生成初判。
2. 人工补三类扫描器盲区：宏拼接的 API 名、`#if` 死分支里的 A5 代码、构建脚本/JSON 里的 `ascend950` 字符串。
3. 每条命中落到 `assets/migration-analysis.md.template` 的清单表，填"处置/级别/依据/负责人"。
4. `blocking` 与 `decision` 项全部有结论后，才能出 GATE-1。
