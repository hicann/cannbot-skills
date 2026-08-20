# apace 算子开发指南

> 本文档定义 apace 算子开发的验收标准与改造食谱：从官网样例起手，按 [REUSE]/[MODIFY] 标记定点改造。样例代码经 scripts/fetch_apace.sh 从官网现取。

## 目录

1. [开发起点与 [REUSE]/[MODIFY] 地图](#1-开发起点与-reusemodify-地图)
2. [开发步骤](#2-开发步骤)
3. [改造场景食谱](#3-改造场景食谱)
4. [工程构建](#4-工程构建)
5. [验收清单](#5-验收清单)

---

## 1. 开发起点与 [REUSE]/[MODIFY] 地图

### 标记规则

| 标记 | 含义 | 验收条件 |
|:---|:---|:---|
| `[REUSE]` | 稳定共享层 | 禁止修改，与官网仓原始文件完全一致 |
| `[MODIFY]` | 算子层 | 复制后重命名 + 按场景改造 |

**[REUSE] 文件清单**：`block/`（`aiv_comm/` 通信 API + `blaze_ext/` Blaze 扩展）、`tiling/`（`comm_tiling_data.h` 通信切分 + `quant_matmul_tiling_*.h` 计算切分）、`basic/`（`fragment_tensor/`）、`utils/`（`comm_channel_builder.h` 等）

**[MODIFY] 文件清单**：`kernel/{op}/` 下 3 个头文件（udma_impl.h + quant_matmul kernel + tiling_data.h，见下）、`tests/st/{op}/` 下完整 ST 工程（CMakeLists.txt、run.sh、cases.csv、src/、scripts/）

### 官网 PUT 3 文件模式（以 `kernel/all_to_all_quant_matmul/` 为基准）

| 文件 | 关键符号 | 职责 |
|:---|:---|:---|
| `all_to_all_mx_quant_matmul_udma_impl.h` | `AllToAllMxQuantMatmulUdmaImpl` | Impl：`Init` / `Run` / `RunAllToAll`（AIV）/ `RunLocalMatmul` / `RunMatmul`（AIC）/ `SetupParams` |
| `quant_matmul_mx_kernel.h` | `Kernel::QuantMatmulMxKernel` | Kernel：`Init` / `Run` / `ProcessSingleBatch`，Blaze `BlockMmad` + `BlockScheduler` 编排 |
| `all_to_all_matmul_tiling_data.h` | `allToAllMatmulTilingData`、`CommContext` | tiling 结构 + 通信上下文（`CommUdmaContext udmaCtx` + `CommUbmemContext ubmemCtx`） |

> 同目录另有 `all_to_all_mx_quant_matmul_hcomm_impl.h`，为 HCCL windows 变体（非直调场景用），Kernel 直调工作流不使用。
>
> `kernel/all_gather_quant_matmul/` 同构：`all_gather_mx_matmul_udma_impl.h` + `qmm_mx_kernel_ag_udma.h`（`QmmMxKernelAgUdma`）+ `all_gather_mx_matmul_udma_tiling_data.h`（`AllGatherMxMatmulUdmaTilingData`）。

### 目录拓扑约束

kernel 头文件中的 `#include "../../block/..."`（见 `all_to_all_matmul_tiling_data.h`）要求 `kernel/<op>/` 与 `block/` 保持两级相对关系。

**include 双风格**：kernel 头文件内部为 `../../block/...`、`../../tiling/...` 相对风格；`tests/st/` 源码用 `apace/...` 前缀 include（如 `tests/st/all_to_all_quant_matmul/src/kernel_launcher.h` 的 `#include "apace/kernel/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_udma_impl.h"`），依赖用例 CMakeLists 提供 `${OP_KERNEL_ROOT}` include 根。

**验收条件**：新算子工程必须保持 `kernel/`、`block/`、`tiling/` 的相对位置，否则编译失败。不能只复制 kernel 头文件。

### 样例选择

官网 `kernel/` 当前共 2 个算子，均为 PUT（通信→计算）方向：

| 通信模式 | 样例目录 | 切分轴 |
|:---|:---|:---|
| AllToAll PUT | `kernel/all_to_all_quant_matmul/` | K 轴 |
| AllGather PUT | `kernel/all_gather_quant_matmul/` | M 轴 |

> 官网暂无 GET、ReduceScatter 样例。GET 钩子（`AllToAllCommGetImpl`）存在于共享层 `block/aiv_comm/`，GET 语义见 [`communication.md`](communication.md)；ReduceScatter 需求参考 [`fusion.md`](fusion.md) §6 的 AllToAll PUT + AtomicAdd 替代实现。

### 获取方式

skill 不持有代码快照。开发前运行 `scripts/fetch_apace.sh`（支持 `APACE_REPO` 环境变量指向已有克隆，或 sparse clone 到 `~/.cache/apace-reference/ops-transformer`）获取官网最新代码，然后参考 `kernel/all_to_all_quant_matmul/` 起手。

### 推荐布局

```
operators/{op_name}/
├── kernel/{op_name}/       ← [MODIFY] 3 个头文件（按官网 PUT 3 文件模式）
├── block/                  ← [REUSE] 从官网仓复制
├── tiling/                 ← [REUSE]
├── basic/                  ← [REUSE]
├── utils/                  ← [REUSE]
├── heavy_kernels.h         ← skill 侧模板（`references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/heavy_kernels.h`，从 SHMEM 基底工程复制），ops-transformer 仓未引入（L2 flush 可内联实现）
└── st/{op_name}/           ← [MODIFY] 完整 ST 工程（参照官网 tests/st/{op}/）
    ├── tests/st/utils/     ← root_info_exchanger.h、utils.h；comm_channel_builder.h 属 apace/utils/
    └── st/CMakeLists.txt   ← [REUSE] 顶层构建（参照官网 tests/st/CMakeLists.txt）
```

---

## 2. 开发步骤

1. **获取官网代码**：运行 `scripts/fetch_apace.sh` 获取官网最新代码（skill 不持有代码快照）
2. **复制起手**：从 `kernel/all_to_all_quant_matmul/` 或 `kernel/all_gather_quant_matmul/` 复制 [MODIFY] 文件并重命名；`block/` `tiling/` `basic/` `utils/` [REUSE] 层原样复制。禁止从零写文件
3. **定点改造**：按 §3 改造场景食谱逐项修改，每处改造对照验收条件自查
4. **编译 + 冒烟**：编译无错误/警告后，单 rank 运行冒烟，输出非全 0。禁止跳过冒烟编译直接全量测试
5. **精度验证**：多 rank + 多 shape `verify_result.py` PASS（精度标准见 §4.5）
6. **性能调优**：按 §5 tileCnt 两阶段策略扫描（先串行基线，再扫 {1,2,4,8,16,32}）
7. **文档同步**：PLAN.md 更新改造内容和测试结果；DESIGN.md 与代码一致

---

## 3. 改造场景食谱

### 3.1 改造场景速查

锚点对照官网 PUT kernel（`kernel/all_to_all_quant_matmul/quant_matmul_mx_kernel.h`，类 `QuantMatmulMxKernel`）与 impl（`all_to_all_mx_quant_matmul_udma_impl.h`，类 `AllToAllMxQuantMatmulUdmaImpl`）：

| 场景 | udma_impl.h | quant_matmul_mx_kernel.h | tiling_data.h | main.cpp | scripts | 共享层 |
|:---|:---|:---|:---|:---|:---|:---|
| 换 dtype | 改模板参数 | 改模板参数 | 无 | 改 kernel 名 + tilingEngine 模板 | 改 dtype | 不动 |
| 换 shape | 无（`InitBaseParams` 为 impl 方法，随 tiling 自动推导） | 无 | 无 | 改参数解析 | 改 gen_data | 不动 |
| 改 localMatmul (PUT) | 无（`Run()` 已含 `RunLocalMatmul` 分支）；用 localMatmul=1 时加 PipeBarrier 补丁（推荐修复） | 无（LocalParams 已含 localMatmul） | 已有 localMatmul 字段 | 设 localMatmul=0/1/2 | 不动 | 不动 |
| 加 bias | SetupParams 补 mmadParams.biasGmAddr + qbmmParams.isBias | 无（bias 通道已现成） | 加 bias 字段 | 加 bias GM 分配 + 传参 | 改 gen_data | 不动 |
| 去 Scale (非 MX) | 改 DispatchPolicy | 去 scale 字段与 Tensor | 去 scale 字段 | 去 scale 参数 | 改 gen_data | 不动 |
| 改流水深度 | 无 | 无 | 调 headMSize/tileCnt（CommTilingData 字段） | 调 headMSize/tileCnt（经 ST main.cpp 参数） | 不动 | 不动 |
| 调通算并行 | 无 | 无 | 改 CommTilingData | 改 headMSize 参数（tileCnt 派生） | 不动 | 不动 |
| 换卡数 | 无 | 无 | 无 | 改 rankNum 参数 | 改 gen_data | 不动 |

### 3.2 udma_impl.h 改造

#### 场景：换 dtype

**验收条件**：Impl 类模板参数（`AllToAllMxQuantMatmulUdmaImpl<AType, BType, CType, TransA, TransB>`）和 4 个 `__global__` 入口（官网 `tests/st/all_to_all_quant_matmul/src/kernel_launcher.h`：E4M3E4M3 / E5M2E5M2 / E4M3E5M2 / E5M2E4M3 四变体）的模板参数均改为目标 dtype。

#### PUT 编排验收标准（以官网 `AllToAllMxQuantMatmulUdmaImpl` 为基准）

> 完整 PUT 编排详见 [`communication.md`](communication.md)（四段式契约）和 [`operator-anatomy.md`](operator-anatomy.md)（Impl 骨架）。

**验收条件**（全部可在 `RunAllToAll()` / `Init()` 中验证）：

| # | 验收项 | 官网锚点 |
|:---|:---|:---|
| 1 | 通信对象数 = 2（A data + A scale），共享同一 channel | `Init()` 中 `allToAllA_` + `allToAllScaleA_` 两个通信对象 |
| 2 | scale 对象 winOffset = `rankSize × rankDataBytes` | `Init()` 中 `allToAllScaleA_.Init(...)` 末参 |
| 3 | Commit 顺序：scale 先、data 后 | `RunAllToAll()` 中 `allToAllScaleA_.Commit()` → `allToAllA_.Commit()` |
| 4 | Wait 方式为 `Wait<BARRIER_DEVICE>()`，同一 channel 只 wait 一次 | `RunAllToAll()` 中 `allToAllA_.Wait<BARRIER_DEVICE>()` |
| 5 | 每轮必须 `SyncAll<true>()` 确保 WriteNbi 对端可见 | `RunAllToAll()` 循环体内 |
| 6 | UB 需求：双 commBuf（各 `COMM_WORKSPACE_SIZE`=512B，`block/aiv_comm/collective_comm_context.h`）+ barrierBuf | `Init()` 中 ubOffset 累加 |
| 7 | AIV 先 → AIC 后 | `Run()` 中 `ASCEND_IS_AIV` 分支在前 |

#### 场景：改 localMatmul（PUT 模式）

> 完整的 localMatmul 模式选择决策和 AtomicAdd 时序分析见 [`fusion.md`](fusion.md)。

**验收条件**：

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | Run() 分支 | `localMatmul==1` 时推荐含 `RunLocalMatmul()` → `PipeBarrier<PIPE_ALL>()` → `RunMatmul()`（官网 `Run()` 当前未实现 PipeBarrier，为推荐修复） |
| 2 | PipeBarrier 必要性 | 不加 PipeBarrier 可能触发 MTE 异常（aclError:507015），详见 [`fusion.md`](fusion.md) |
| 3 | kernel LocalParams | 含 `localMatmul` + `matmulMode` 字段（`quant_matmul_mx_kernel.h` `LocalParams`）；`localMatmul==1 && REMOTE` 时 `isAtomicAdd_` 置位（`Init()`） |
| 4 | tiling_data.h | 含 `uint32_t localMatmul` 字段 |
| 5 | main.cpp | 显式设置 `localMatmul` 值（官网 `tests/st/all_to_all_quant_matmul/src/main.cpp` 显式 `tilingData.localMatmul = 0`，不依赖默认值） |
| 6 | splitKNum 规则 | `localMatmul==1` 时 REMOTE 只有 `rankSize-1` 个远程卡参与，否则为 `rankSize`（官网 `SetupParams()`） |
| 7 | CMakeLists | 参考：`kernel/all_to_all_quant_matmul` 用例含 `-DASC_DEVKIT_MAJOR=9`，`all_gather` 用例当前不含 |

### 3.3 quant_matmul_mx_kernel.h 改造

#### 场景：去 Scale（非 MX 量化）

官网基线为 MX 量化：`DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<NONE_FULL_LOAD_MODE, false>`（`all_to_all_mx_quant_matmul_udma_impl.h`）。去 scale 需替换为非 MX 策略（替代类型以 ops-tensor `blaze/gemm/policy/dispatch_policy.h` 为准）并联动清理 scale 链路。

**验收条件**（5 处修改）：

| # | 修改位置 | 验收条件 |
|:---|:---|:---|
| 1 | DispatchPolicy 定义（udma_impl.h） | 类型由 `MatmulWithScaleMx` 改为非 MX scale 策略 |
| 2 | mmadParams 赋值（udma_impl.h `SetupParams()`） | 不含 `scaleBGmAddr` 赋值 |
| 3 | kernel `ResetGmAddr()` | 不含 `scaleAGmAddr_` / `scaleBGmAddr_` 赋值 |
| 4 | kernel `ProcessSingleBatch()` | 不含 ScaleA/ScaleB 的 Layout/Tensor 创建和 Slice；`mmadOp_()` 调用不含 scale 参数 |
| 5 | tiling_data.h + main.cpp | 去 scaleCommTilingData 相关字段与 scale GM 分配 |

#### 场景：加 bias

官网 kernel 的 bias 通道已现成：`BiasType`（`BlockMmad::BiasType`）、`biasGmAddr_`（`ResetGmAddr()` 中由 `isBias_` 门控赋值）、`layoutBias`/`gmBias`（`ProcessSingleBatch()`）、`QBMMTiling::isBias` + `BIAS_ENABLED`、`mmadOp_.Init(..., isBias_, ...)`。**kernel 文件无需修改**。

**验收条件**（3 处修改，均在 impl / host 侧）：

| # | 修改位置 | 验收条件 |
|:---|:---|:---|
| 1 | udma_impl.h `SetupParams()` | 补 `mmadParams.biasGmAddr = ...`（官网 UDMA impl 当前未设置；HCCL windows 变体 `all_to_all_mx_quant_matmul_hcomm_impl.h` 已设 `mmadParams.biasGmAddr`，可对照）；`QBMMTiling` 构造末参 `isBias` 由 `false` 改为按输入置位 |
| 2 | tiling_data.h | 含 bias 相关字段（如需） |
| 3 | main.cpp `LaunchKernel` | 含 bias GM 分配 + biasGM 参数透传 |

#### 场景：改 B 切分维度（N→K）

> 参考官网 `all_to_all_quant_matmul` PUT 实现：B 按 K 段拼接通过张量布局表达，无显式 B 指针累加。

**验收条件**：

| # | 修改位置 | 验收条件 |
|:---|:---|:---|
| 1 | kernel `LocalParams` | 含 `splitKNum` 字段（`quant_matmul_mx_kernel.h`），按参与 rank 数切分 K |
| 2 | B 张量布局 | 用 `MakeLayoutB{}(rankSize * K, N)` 张量布局拼接各 rank K 段（`ProcessSingleBatch()`），无显式 B 指针累加 |
| 3 | ProblemShape | N = 完整 `axisN`（非 `axisNPerRank`） |
| 4 | CommTilingData | data 通道 `nonSplitAxisSize` = ka（per-rank K）；scale 通道 = ka/32（`tiling/comm_tiling_data.h` `CommTilingData::nonSplitAxisSize`，与 [`operator-anatomy.md`](operator-anatomy.md) 一致） |

> K 轴切分通常用于 PUT 模式，GET 模式一般用 N 轴切分。

### 3.4 tiling_data.h 改造

#### 场景：加字段

**验收条件**：新字段在 tiling 结构体中正确声明，main.cpp 中正确赋值。

#### 场景：改 CommContext

**验收条件**：
- UDMA 模式：tiling_data.h 含 `CommContext` 结构体（`CommUdmaContext udmaCtx` + `CommUbmemContext ubmemCtx`，官网 `all_to_all_matmul_tiling_data.h`）
- HCCL windows 模式（非直调）：tiling_data.h 不含 `CommContext`，入口去掉 `__gm__ CommContext*` 参数

> 注意：CommContext 判据针对**具体使用的 tiling 结构体**，而非整个文件——官网 `all_to_all_matmul_tiling_data.h` 中 `CommContext`（UDMA 用）与 `ccuAllToAllMatmulTilingData`（CCU 用）共存。

---

## 4. 工程构建

### 4.1 顶层 CMakeLists [REUSE]（官网 `tests/st/CMakeLists.txt`）

**验收条件**（不得修改，必须满足以下功能）：
- 正确定位 bisheng 编译器和 CANN 工具链（`ASCEND_HOME_PATH`）
- 经主仓 `cmake/third_party/ops-tensor.cmake` 拉取 ops-tensor 依赖（blaze/tensor_api 头文件）
- 创建 `apace_blaze_api` INTERFACE 库（提供 blaze/tensor_api include 路径）
- `add_subdirectory` 接入各用例

**ops-tensor 获取机制**（主仓 `cmake/third_party/ops-tensor.cmake`）：

| 机制 | 说明 |
|:---|:---|
| FetchContent 自动拉取 | commit 锁定（`OPTENSOR_TAG_ID`），落地路径 `${OPS_TRANSFORMER_ROOT}/third_party/ops-tensor` |
| `CANN_3RD_LIB_PATH` 覆盖 | 指定后从 `${CANN_3RD_LIB_PATH}/ops-tensor` 取本地副本 |
| `OPS_TENSOR_ROOT` | 非用户入口，由 `OPTENSOR_SOURCE_PATH` 反设（`tests/st/CMakeLists.txt`） |

> ⚠️ 主仓无 `-DOPS_TENSOR_ROOT` 用户入口。

### 4.2 用例 CMakeLists [MODIFY]（官网 `tests/st/{op}/CMakeLists.txt`）

**验收条件**：

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | target 名 | 改为新算子名 |
| 2 | KERNEL_OPTS | 含 `--npu-arch=dav-3510`（两个用例都有）；`-DASC_DEVKIT_MAJOR=9` 仅 all_to_all 用例有，all_gather 当前不含 |
| 3 | include 路径 | 含 `${APACE_ROOT}/kernel/{op_name}` |
| 4 | 链接库 | `COMMON_LIBS`（`dl platform tiling_api ascendcl runtime hccl stdc++`）+ `apace_blaze_api` |
| 5 | hccl_fwk | 用 `-Wl,--no-as-needed` 包裹，放在 target_link_libraries 最后 |

**include 路径验收**：

| 路径 | 作用 | 新算子需改？ |
|:---|:---|:---|
| `${OP_KERNEL_ROOT}` | `apace/...` 根相对 include（tests/st 用）需要此 include 根 | 否（自动指向） |
| `${APACE_ROOT}/kernel/{op_name}` | 使本算子头文件生效 | **是**（改 `{op_name}`） |
| `${CMAKE_CURRENT_SOURCE_DIR}/../utils` | tests/st/utils/ 中的工具头文件 | 否 |

> 相对 include（`#include "../../block/..."`）由文件相对路径解析，不依赖 `-I`；`apace/...` 根相对风格（tests/st 用）才需要 include 根。

### 常见构建失败对照

| 现象 | 根因 |
|:---|:---|
| Ascend toolkit not found | ASCEND_HOME_PATH 未设置 |
| bisheng not found | CANN 安装不完整或架构不匹配 |
| blaze not found | ops-tensor 未拉取或路径错误 |
| tensor_api/tensor.h not found | ops-tensor 的 submodule 未初始化 |
| HcclChannelAcquire undefined | hccl_fwk 链接顺序错误（需 `--no-as-needed`） |

### 4.3 run.sh / cases.csv [MODIFY]

**验收条件**：cases.csv 中的 shape 和参数与新算子匹配。官网 cases.csv 列为 `m,k,n,rank_num,head_m_size`；run.sh 支持按行号/`all`/`--cli`/`--perf` 运行（`--perf` 模式产出经 `scripts/parse_prof.py` 解析）。

### 4.4 main.cpp [MODIFY]（官网 `tests/st/{op}/src/main.cpp`）

**验收条件**：

| 改造场景 | 验收条件 |
|:---|:---|
| 换算子名 | 全局无旧算子名残留（含 ctxTag） |
| 换 kernel 名 | `LaunchKernel` 调用使用新 kernel 类名 |
| 换 dtype | `tilingEngine` 模板参数为目标 dtype（官网为 `QuantMatmulTilingSwat<DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN>`） |
| 换 tiling 结构 | `tilingData` 字段与 tiling_data.h 一致 |
| 加 bias | 含 bias GM 分配 + `LaunchKernel` 传 biasGM |
| localMatmul | 显式赋值（官网 `tilingData.localMatmul = 0`） |

> main.cpp 完整结构详见 [`host-and-testing.md`](host-and-testing.md)

### 4.5 scripts [MODIFY]（官网 `tests/st/{op}/scripts/`）

**验收条件**：
- `gen_data.py`：dtype、shape、golden 计算逻辑与新算子一致
- `verify_result.py`：dtype、容差、输出形状与新算子一致
- 精度标准：all_to_all 用 `rtol=atol=1e-2`（`ERROR_TOL`），all_gather 用 bit-exact 或 ≤1 ULP（bf16 raw uint16 比较）——各算子容差不同，以对应 `verify_result.py` 为准

---

## 5. 验收清单

### 工程验收条件

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | 工程结构 | 保持 `kernel/↔block/` 两级目录关系 |
| 2 | 共享层完整性 | `block/` `tiling/` `basic/` `utils/` 与官网仓对应原始文件完全一致 |
| 3 | 命名一致性 | 全局无旧算子名残留（类名、文件名、namespace、ctxTag） |
| 4 | 编译通过 | 无编译错误/警告 |
| 5 | 冒烟测试 | 单 rank 运行输出非全 0 |
| 6 | 精度验证 | 多 rank + 多 shape `verify_result.py` PASS |
| 7 | 文档同步 | PLAN.md 已更新改造内容和测试结果；DESIGN.md 与代码一致 |

### 禁止行为

- 禁止从零写文件（必须经 `scripts/fetch_apace.sh` 获取官网代码后，从 `kernel/all_to_all_quant_matmul/` 或 `kernel/all_gather_quant_matmul/` 复制起手）
- 禁止修改 `[REUSE]` 文件（`block/`、`tiling/`、`basic/`、`utils/`）
- 禁止跳过冒烟编译直接全量测试
- 禁止添加 `__schedmode__(1)` 或 `[[bisheng::core_ratio(1,1)]]`
- 禁止修改 `localMatmul` 等关键参数后不同步更新 DESIGN.md

### tileCnt 策略

| 阶段 | tileCnt | 目的 |
|:---|:---|:---|
| 精度调试阶段 | 1（串行基线） | 消除流水干扰，flag 出错直接 deadlock 便于排查 |
| 性能调优阶段 | 扫描 {1,2,4,8,16,32} | 找 Task Duration 最小值 |

**验收条件**：切换 tileCnt 必须重新调 `QuantMatmulTilingSwat::GetTilingData`，不能复用旧 tiling。

> 详见 [`pipeline_tuning.md`](../../shared/pipeline_tuning.md)（apace 用 `splitAxisTileCnt` 命名）

---

## 后续阅读

- [`architecture.md`](architecture.md) — 心智模型入口（三层架构、GET/PUT、四大约束）
- [`operator-anatomy.md`](operator-anatomy.md) — 算子骨架共性模式
- [`host-and-testing.md`](host-and-testing.md) — host/ST
- [`fusion.md`](fusion.md) / [`compute.md`](compute.md) / [`communication.md`](communication.md) — 接口与组合模式
