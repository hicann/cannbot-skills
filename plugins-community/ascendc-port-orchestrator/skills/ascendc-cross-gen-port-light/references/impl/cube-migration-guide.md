# Cube 类算子迁移实施详细指南（L1 适配 + 硬件差异）

> 定位：cube（矩阵计算）类算子从 a2/a3(910b/910_93) 迁移到 A5(950) 的整体迁移指引——数据通路（L1/L0A/L0B/L0C）、分形变化（ZZ→NZ）、跨核同步等与 Vector 正交的硬件差异；层级按性能定级（非性能关键按 L1 适配，性能关键按 L2 评估，判定见 stage_1 Step 1.3）。
> 配套：`cube-debug-lessons.md` = 测试与 debug 排查。流程：按本文件迁移 → 测试阶段查 debug。
> 依据 asc-devkit 全量仓官方文档与兼容性样例验证（见文末「相关参考文档路径」）。

## 触发条件（满足任一）

| 信号 | 说明 |
|------|------|
| kernel 中使用 `Mmad` / `LoadData` / `LoadDataWithTranspose` / `Fixpipe` / `DataCopyCO12DstParams` 等 cube API | Cube 数据通路 |
| kernel 中操作 L1 / L0A / L0B / L0C 位置 Tensor | Cube 存储层级 |
| AIC（C 核）与 AIV（V 核）流水协作，需要跨核握手 | 同步机制 |
| kernel 中存在 `SetLoadDataPaddingValue` / `SetNdParaImpl` / `SetFixpipeNz2ndFlag` / `SetLoadDataBoundary` 等已废弃全局配置 | 平台差异清理 |

## 配套经验引用表（skill 既有经验，按环节加载）

cube 迁移**不是只读本文件**——以下环节必须配套使用 skill 既有经验（它们与 cube-guide 互补，适用边界见各自头部声明）：

| 迁移环节 | 配套经验文件 | 加载时机 |
|---------|-------------|---------|
| AIV 侧 Vector 路径评估/重写（L2 语义 ③、VF 归属判断） | `l2-guide.md`、`api-mapping.md` | L2 改造前 |
| AIV 侧 RegBase 最佳实践（白名单/四层模型/陷阱/参考算子） | `cannbot-skills/ops/ascendc-regbase-best-practice/references/regbase_development_guide.md`（外部 Sub-Skill，见 SKILL.md） | AIV 侧写 RegBase 代码前 |
| Host 侧 tiling/配置（L1 适配范围） | `l1-guide.md`（host 侧步骤通用） | L1/L2 改造前 |
| 测试与 debug 排查 | `cube-debug-lessons.md` | 阶段 4（迁移后测试） |

### 性能关键 cube 算子的 L2 语义（定级 L2 后必读）

性能关键 cube 算子定级 L2 后，**默认路径是评估低阶直跑（改动 2/3/4）而非沿用高阶 Matmul API**。触发信号：3510 上 matmul 高阶内部同步占用 flagId `[0, 2N−1] ∪ [16, 16+2N−1]`（N = 实例化 Matmul 对象数，见改动 6 flagId 冲突检查）；自定义同步点的避让区随 N 收缩，冲突不可解时 → 弃高阶走低阶直跑。L2 整体 = ① 低阶直跑评估（改动 2/3/4）+ ② 同步协议重写（改动 6）+ ③ AIV 侧 RegBase 评估（`l2-guide.md`；**先查 VF 封装实现归属**：接口头 → `base_impl.h` 的 `__NPU_ARCH__` 分流，3510 有 regbase/专用实现则保留接口即可、无需重写，见 stage_1 Step 1.3），三部分缺一不可。

### 保留高阶 Matmul API 的必查项：双主模式约束（评估时静态核对）

950 上保留高阶 Matmul API 且启用双主模式（`enableMixDualMaster`）时，MUST 逐条核对 devkit 文档约束（`$DEVKIT_PATH/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/MatmulConfig.md` 的 enableMixDualMaster 参数说明、`GetNormalConfig.md`）：

| # | 约束（文档原文依据） | 核对点 |
|---|---|---|
| 1 | 双主模式语义：区别于 MIX 模式"通过消息机制驱动 AIC 运行"，双主模式为"AIC 和 AIV 独立运行代码，不依赖消息驱动" | 保留高阶 + 双主时，算子原同步协议/数据通路设计是否仍适用（消息驱动机制不可再依赖） |
| 2 | 仅当核函数类型为 MIX 且 AIC:AIV = 1:1，或 1:2 且 **A、B 矩阵同时开启 IBSHARE** 时，双主模式才可用 | 核对 matmul 模板参数 IBSHARE 是否开启（A2 代码直接沿用高阶时最容易漏） |
| 3 | 同一算子中所有 Matmul 对象的 enableMixDualMaster 取值必须一致 | 多 matmul 实例逐一核对 |
| 4 | A/B/Bias 矩阵只支持从 GM 搬入 | 核对原代码是否有 UB/L1 中间装载路径 |
| 5 | 获取矩阵计算结果只支持调用 `IterateAll` 输出到 GM/LocalTensor，不能调用 `GetTensorC` 等 | 核对结果获取方式 |

任一项无法满足 → 双主模式不可用：改为普通 MIX（消息驱动）或弃高阶走低阶直跑（改动 2/3/4）。判断依据全部为文档约束，静态可查，禁止"先保留运行时验证"式风险转移。

### 分离模式核能力模型（950 迁移前置知识）

950（Ascend 950PR/950DT）为**分离模式**（`$DEVKIT_PATH/docs/zh/guide/programming_guide/advanced_programming/hardware_implementation/basic_architecture.md`「AI Core 的工作模式」）：

- 矩阵计算单元与矢量计算单元**各自对应独立的 Scalar 调度单元，分离部署在 Cube Core 和 Vector Core 上**；Cube Core "不包括矢量计算单元"、Vector Core "不包括矩阵计算单元"；
- Cube Core 与 Vector Core 按 1:N 组合视为一个 AI Core，**AI Core 的核数以 Cube Core 为准**；
- A2/A3/950 均为分离模式；Atlas 推理/训练系列为耦合模式（矩阵与矢量单元同核、同一 Scalar 调度）。

由此得出的结构推论（非文档原文，标注推导）：kernel 中矩阵路径与矢量路径的代码由不同核执行（AIC 执行矩阵段、AIV 执行矢量段），评估迁移后 kernel 时需说明每个核执行哪段代码（核类型分支 `ASCEND_IS_AIC`/`ASCEND_IS_AIV` 或等效机制）；host 侧 blockDim 以 Cube Core（AIC）为基准折算——本工程工具 `common/include/tiling_base/tiling_base.h` 的 `CalcTschBlockDim(sliceNum, aicNum, aivNum)` 即按 aiv/aic 比例把 slice 数折算为 AIC 任务数，迁移时 slice 数语义须与折算后任务空间一致（A2 的"AIV 数切分"语义在 950 分离模式下与折算规则不同，需核对）。

## 平台宏约定

| 宏 | 平台 | 说明 |
|----|------|------|
| `__CCE_AICORE__ == 310` | A5 950 | 迁移目标平台（等价官方 `__NPU_ARCH__ == 3510`） |
| `__CCE_AICORE__ == 200`、`defined __DAV_310R6__` | a2/a3 | 老平台 |
| `__NPU_ARCH__ == 2201` / `== 3510` | 官方文档口径 | 官方样例和文档统一使用 |

## 改动 1：工程组织（arch 目录拆分 + 入口宏分发）

迁移后 kernel 代码按架构拆分为独立目录，入口通过平台宏选择头文件：

```cpp
#if (__CCE_AICORE__ == 310)
    #include "arch35/xxx_kernel.h"     // A5 实现
#else
    #include "arch22/xxx_kernel.h"     // a2/a3 实现
#endif
```

| 内容 | 建议 |
|---|---|
| 平台差异代码 | 放 `op_kernel/arch22/`、`op_kernel/arch35/`，互不干扰 |
| 公共代码 | `common.h`、`template_tiling_key.h`、算子入口 `.cpp` 留在 `op_kernel/` 根目录 |
| 入口模板分发 | 保持 `ASCENDC_TPL_*` 分发结构不变，必要时扩展模板参数 |

### 模式 A：就地宏隔离

单个 API 级差异（如 L0A 装载）用 `#if (__CCE_AICORE__ == 310) ... #else ... #endif` 就地隔离。

### 模式 B：arch 目录拆分

整文件级差异用 `arch22/` / `arch35/` 目录隔离，公共头文件下沉。

> 迁移前常见写法是入口处 `#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__CCE_AICORE__ == 200)` 整段跳过 310 实现（老工程里 310 常是空分支），迁移时必须补全。
> 官方样例采用单文件内 `#if __NPU_ARCH__ == 2201 / == 3510` 条件编译的双路径写法，两种组织方式均可。

## 改动 2：L1→L0A/L0B 装载（LoadData2DParamsV2）

### 装载调用改写判定（agent 执行规则）

在 kernel 中发现装载调用时，按"发现的调用 → 改写动作"执行，不要仅凭函数名判断新旧：

| 发现的调用（信号） | 改写动作 |
|------|---------|
| `LoadData(dst, src, LoadData2DParams)`（字段 repeatTimes/srcStride） | 改写为 `LoadData(dst, src, LoadData2DParamsV2)`（字段 mStep/kStep），参数换算按下表 |
| `LoadData(dst, src, LoadData3DParamsV2<...>)` / `LoadData3DParamsV1` | 3D 卷积式装载；确认目标 API 在 3510 的支持情况后改写，参数体系不兼容需重写 |
| `LoadData(dst, src, LoadData2DParamsV2)` | 已是 A5 目标写法，仅核对参数单位与约束（下表），无需改写 |
| `LoadDataWithTranspose(dst, src, LoadData2dTransposeParams)` | 已是独立转置 API，仅需确认通路：**950 仅支持 L1→L0B**，L0A 通路需预转置（见改动 5） |
| `LoadDataWithSparse(...)` / `MmadWithSparse(...)` | 950 不支持 4:2 稀疏，改写为稠密计算（见"官方数据通路变更总表"） |

**同名不同义排查**：`LoadData2DParams`（repeatTimes/srcStride）与 `LoadData2DParamsV2`（mStep/kStep）字段体系完全不同，**必须重写参数，不能只改类型名**；`LoadData3DParamsV1/V2/V2Pro` 是 3D 装载参数，与 2D V2 无关；`FixpipeParamsV220` 与 `FixpipeParamsArch3510` 同理（见改动 3）。

### LoadData2DParamsV2 参数单位（易错点）

参数单位是本 API 的最大易错点，速记：**mStep=16 个元素、kStep=32 字节、srcStride/dstStride=512 字节（一个数据分形）**、`mStep/kStep=0` 为 NOP。注意 stride 的 512 字节是**分形大小**——b4/b8/b16/b32 的一个分形都是 512 字节（分别为 16×64 / 16×32 / 16×16 / 16×8），故该单位与 dtype 无关。`ifTranspose` 仅 L1→L0A/L0B 通路可开，且开启时需满足：**b4 需 mStep 为 4 的倍数、b8 为 2 的倍数、b16 无额外约束、b32 需 kStep 为 2 的倍数**；float 数据 L0B 装载 `dstStride` 需减半。

**完整参数表与通路约束**见 API 最佳实践 `cannbot-skills/ops/ascendc-api-best-practices/references/api-loaddata.md`（LoadData2DParamsV2 字段单位节）。


### 标准用法（M×K 左矩阵，参考官方样例 data_copy_l1togm）

```cpp
// A5：LoadData2DParamsV2，一次描述整块 2D 搬移
LoadData2DParamsV2 loadDataParams;
loadDataParams.mStartPosition = 0;
loadDataParams.kStartPosition = 0;
loadDataParams.mStep = DivCeil(M, 16);               // M 方向块数
loadDataParams.kStep = DivCeil(K * sizeof(T), 32);   // K 方向 32B 块数
loadDataParams.srcStride = DivCeil(M, 16);
loadDataParams.dstStride = DivCeil(M, 16);
loadDataParams.sid = 0;
loadDataParams.ifTranspose = false;
LoadData(a2Local, a1Local, loadDataParams);
```

**旧平台写法（对照）**：

```cpp
// ===== a2/a3：LoadData2DParams，手动循环按 16×16 块逐块装载 =====
LoadData2DParams loadDataParams;
loadDataParams.repeatTimes = kBlocks;
loadDataParams.srcStride = mBlocks;
loadDataParams.ifTranspose = false;
LoadData(a2[dstOffset], a1[srcOffset], loadDataParams);   // 循环 + 手动算偏移

// 部分老算子使用卷积式 3D 装载：
LoadData3DParamsV2<Q_T> loadData3dParams;   // l1H/l1W/channelSize/padList/mExtension/kExtension/...
LoadData<Q_T, LOAD3D_CONFIG>(a2, a1, loadData3dParams);
```

### 要点与易错点

- 两套结构**字段不兼容，必须重写**：V2 以"行列块数 + 起点"描述，旧版以"repeatTimes + srcStride 重复块"描述。
- 官方兼容性样例 `pattern_transformation` 展示了等价写法：3510 分支仍可用 `LoadData2DParams`（repeatTimes 版），但循环方向从"按 M 块循环"改为"**按 K 块循环、M 方向用 repeatTimes**"（对应 NZ 分形排布，见改动 5）。**新写代码优先用 V2**。
- A5 上不再需要：`SetLoadDataPaddingValue()`、`SetNdParaImpl()`、`SetFixpipeNz2ndFlag()` 等全局配置，310 分支直接省略。
- L0B（右矩阵）装载时 mStep/kStep 语义互换（B 矩阵按 [K×N] 装载），`ifTranspose = true`；**float 数据时 `dstStride` 需减半**（`dstStride = nAlign / 2`，见官方样例）。
- M 轴跨块切分（splitM）时，通过循环改 `mStartPosition`、累加 `dstOffset` 实现连续拼接装载。

## 改动 3：L0C 回写（FixpipeParamsArch3510 + Fixpipe）

| 平台 | 支持的 Fixpipe 参数结构 |
|---|---|
| A5 950 | `FixpipeParamsV220` + **`FixpipeParamsArch3510`**（新增，推荐） |
| A2/A3 | 仅 `FixpipeParamsV220` |
| Atlas 200I/500 A2 | 仅 `FixpipeParamsM300` |

### Fixpipe 输出格式配置（950 增强）

`FixpipeConfig{format, isToUB}`：`format`=NZ（保持）/ ROW_MAJOR（NZ2ND）/ COLUMN_MAJOR（950 新增 NZ2DN）；`isToUB=true` 走 L0C→UB 新通路（配 `dualDstCtl`/`subBlockId`）。

**FixpipeParamsArch3510 完整字段表（nSize 16 倍数、srcStride 单位 C0_SIZE、dstStride 单位 element ≠ V220 的 datablock 等差异点）**见 API 最佳实践 `cannbot-skills/ops/ascendc-api-best-practices/references/api-cross-gen-fixpipe.md`。


### 标准用法（L0C→GM 普通 NZ2NZ 搬运）

```cpp
FixpipeParamsV220 fixpipeParams;
fixpipeParams.nSize = N;
fixpipeParams.mSize = M;
fixpipeParams.srcStride = M;
fixpipeParams.dstStride = N;
fixpipeParams.ndNum = 1;
fixpipeParams.srcNdStride = 2;
fixpipeParams.dstNdStride = M * N;
fixpipeParams.quantPre = QuantMode_t::NoQuant;
Fixpipe<S, S, CFG_ROW_MAJOR>(cGlobal, co1Local, fixpipeParams);
```

### 易错点

- 老算子从 L0C 直接回写 GM 的 `DataCopyCO12DstParams` 用法在 A5 上建议整体替换为 Fixpipe；`SetFixpipeNz2ndFlag(1,1,1)` 已废弃。
- 代码中出现的 `FixpipeParamsC310` 与官方 `FixpipeParamsArch3510` 是同一套结构在不同 CANN 版本中的命名，迁移时以头文件 `kernel_struct_fixpipe.h` 为准。
- **L0C→UB 通路是 A5 新增**：`FixpipeConfig.isToUB = true` 时配合 `dualDstCtrl`/`subBlockId` 使用（A2/A3 不支持）。

## 改动 4：Mmad 与分块重排

```cpp
AscendC::MmadParams mmadParams;
mmadParams.m = M;
mmadParams.n = N;
mmadParams.k = K;
mmadParams.isBias = false;
mmadParams.cmatrixInitVal = true;   // L0C 初始化为 0，乒乓复用 L0C 的标配
AscendC::Mmad(co1Local, a2Local, b2Local, mmadParams);
```

| 要点 | 说明 |
|---|---|
| `cmatrixInitVal = true` | 保证每次累加前 L0C 清零（官方样例默认开启） |
| M=1 奇异分块 | 补齐为 2（实践经验，避免硬件异常） |
| splitM | L0C 的 M 向容量小于老平台（M_BASE_SIZE 512→256），大 M 分块需引入 M 轴切分循环，每个分块独立做"装载→Mmad→回写"，配套独立 L0A/L0C 缓冲序号 |

### 片上缓冲容量对照

| 资源 | a2/a3 | A5 | 影响 |
|---|---|---|---|
| L0A/L0B/L0C 容量 | 较大 | 缩小（M_BASE_SIZE 512→256、S2 基本块 256→128） | 分块 size、乒乓缓冲数量要重排 |
| L1 容量 | 较大 | 缩小 | Key 缓存块切分策略要调整 |
| UB 容量 | 128KB（910b）/ 192KB（910_93） | **256KB**（8 bank group × 2 bank × 16KB，对比 2201 的 16×3×4KB） | 单次 vector 处理规模可提高，但 bank 布局变了 |

- 老平台"一份 L1 buf 放两个 256 行分块、按 offset 判断前/后半块"的紧凑排布技巧在 A5 上往往不再成立，简化为"一个分块一份 buffer + 独立序号"更稳妥。
- 常量（`M_BASE_SIZE`、`S2_BASIC_BLOCK`、`BASIC_BLOCK_LENGTH`、`BLOCK_CUBE` 等）必须按 A5 容量重新取值并确认注释与 magic number 一致。

## 改动 5：L0A 分形变化（ZZ → NZ）

| 平台 | 矩阵乘 A×B=C 排布 | 矩阵 A 分形（L0A） | 矩阵 B 分形（L0B） | 矩阵 C（L0C） |
|---|---|---|---|---|
| 2201 | ZZ × ZN → NZ | 16 × (32B/sizeof(A))，分形内行主序、分形间行主序 | (32B/sizeof(B)) × 16，分形内列主序、分形间行主序 | 16×16 |
| 3510 | **NZ** × ZN → NZ | 16 × (32B/sizeof(A))，分形内行主序、**分形间列主序** | 同 2201 | 同 2201 |

**关键洞察**：
- L1 Buffer 的数据分形本来就是 **NZ**，2201 下 L1→L0A 需要做 NZ→ZZ 的额外分形转换；**3510 下 L1→L0A 不需要转换分形**，装载更简单。
- **非 L0A 切分场景**（整块装载）官方声明兼容 2201 的写法（如 V2 整块装载 `mStep=RoundUp(M,16)/16, kStep=RoundUp(K,16)/16`）。
- **L0A 切分场景**必须按 NZ 分形重新计算 L0A 地址：官方样例 `pattern_transformation` 中 3510 分支按 **kBlocks 循环、M 方向用 repeatTimes**（每块 [M × 16]），切分时上半/下半 NZ 分形分开搬。

```cpp
// 3510 非切分（官方 pattern_transformation）：
#if __NPU_ARCH__ == 3510
    for (i = 0; i < kBlocks; ++i) {
        loadDataParams.repeatTimes = mBlocks;   // M 方向
        loadDataParams.srcStride = 1;
        loadDataParams.ifTranspose = false;
        LoadData(a2[dstOffset], a1[srcOffset], loadDataParams);
        srcOffset += 16 * 16 * mBlocks;   // 按 NZ 分形步进
        dstOffset += 16 * 16 * mBlocks;
    }
#endif
```

### LoadDataWithTranspose（转置装载独立 API）

- **A5 专属**（950PR/950DT 支持，其它不支持）；950 上**仅支持 L1→L0B 通路，不支持 L1→L0A**。
- 参数类型 `LoadData2dTransposeParamsV2`（startIndex、dstFracGap 等，512 字节分形为单位）。
- 左矩阵（A）转置场景：装载前用 vector 预先转置，或注意 `LoadData2DParamsV2.ifTranspose` 的通路限制。

## 改动 6：跨核同步与事件管理

### CrossCoreSetFlag / CrossCoreWaitFlag：模式决定参与集合（核心知识）

```cpp
// 模板：<modeId, pipe>
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(...);
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreWaitFlag(...);
```

**该 API 在 2201/3510 均存在，但模式集合与参与集合语义随架构不同——迁移时以目标平台文档为准，不得假设与旧平台一致（模式号相同 ≠ 语义相同）。**

**模式参与集合语义**（官方定义；跨核同步 API 的完整规则见 `cannbot-skills/ops/ascendc-api-best-practices/references/api-crosscore-sync.md`）：

| 模式 | 参与集合 |
|---|---|
| 0 | 全核（全部 AIC 或全部 AIV） |
| 1 | 单个 AI Core 内全部 AIV |
| 2 | AIC ↔ 单个 AI Core 内**全部** AIV |
| 4 | AIC ↔ **单个** AIV（3510 新增，仅 950PR/DT） |

**型号支持范围**：950PR/950DT 支持模式 0/1/2/4；其他型号仅支持 0/1/2。**模式 4（AIC↔单个 AIV）是 3510 新增**——2201 上 AIC 无法与单个 AIV 显式同步。

**flagId 计数器驱动条件**：flagId 与计数器绑定，每个 AIC 和每个 AIV 各自有 16 个 flagId；模式 2 下 AIC 的 flagId=X 计数器需**组内全部 AIV 都用 flagId=X 完成 SetFlag** 才 +1。**SetFlag/WaitFlag 必须按参与集合配对使用**（模式 2：AIC 1 次 SetFlag ↔ M 个 AIV 各 1 次 WaitFlag；M 个 AIV 各 1 次 SetFlag ↔ AIC 1 次 WaitFlag），缺任何一侧即挂死。

| 约束 | 说明 |
|---|---|
| pipe | 模式 0/1/2：PIPE_V / PIPE_M / PIPE_MTE1 / PIPE_MTE2 / PIPE_MTE3 / PIPE_FIX，**不支持 PIPE_S/PIPE_ALL**；**模式 4（950PR/DT）额外支持 PIPE_S**（SSbuf 消息通道场景），仍不支持 PIPE_ALL |
| 双 AIV | 用事件偏移区分 AIV0/AIV1（如 `CROSS_CV_EVENT + loop % 2 + AIV0_AIV1_OFFSET`，事件常量统一在 `common.h` 定义） |
| 模式 0 全核同步 | 建议开启 batchmode 使算子独占核资源，否则可能死锁 |

### 模式 4 方向映射与双发双收（1 AIC + 2 AIV 场景）

**模式 4 flagId 方向映射**（`CrossCoreSetFlag_ISASI.md` 官方定义；flagId 为**核内局部编号，不跨核**）：

| 发起方 Set | 使用的 flagId | 接收方 Wait | 接收方实际等待的 flagId |
|---|---|---|---|
| AIV0 | 0-15 | AIC | 0-15 |
| AIV1 | 0-15 | AIC | **16-31** |
| AIC | 0-15 | AIV0 | 0-15 |
| AIC | **16-31** | AIV1 | 0-15 |

**实例化规则**（把同步边翻译成代码的规范，全部可由文档推出）：

1. **AIV 侧代码一律写基础编号（0-15）**——AIV0/AIV1 的 Set/Wait 写同一个数；两个 AIV 用同一编号不冲突（各自计数器独立）。若 AIV 侧代码出现 16-31 区间编号，即已违反映射。
2. **AIC 侧用 +16 表达方向**：发给 AIV0 / 收 AIV0 的信号 → `N`；发给 AIV1 / 收 AIV1 的信号 → **`N+16`**。±16 是文档规定的方向寻址，不是自定义约定。
3. **「广播」不存在**：模式 2 下「AIC 发一次、组内所有 AIV 都等到」的同步点，模式 4 必须按**该边覆盖的 AIV 数量**展开——覆盖 2 个 AIV 的事件 → 双发 `Set(N)` + `Set(N+16)`；需 2 个 AIV 都完成的事件 → 双收 `Wait(N)` + `Wait(N+16)`；单对通信 → 单发单收。**每一条 Wait 都必须能找到同核、同方向窗口的 Set 喂它**（按上表逐条核对）。
4. **编号是「边」的标签，不是核/流的标签**：flagId 核内局部 → 所有核同码运行，编号不需要（也不应该）包含 taskId / blockIdx / subBlockIdx 身份。旧代码中「flagId 含核身份索引」（如 `FLAG[(taskId+1)&1][blockIdx&1]`）在旧模式广播语义下可能只是无害的命名习惯，**移植后必须逐一删除并重新推导**——同一表达式在新模式下语义不同。

**计数平衡验证（纸面推演，静态必做）**：对每条边，统计一侧所有 Set 点与另一侧所有 Wait 点的执行次数，**必须覆盖：首尾迭代（软件流水偏移）、条件分支（needExec/needPair 截断）、空范围（分片边界）、变长截断**。Wait 数 > Set 数 → 必然死锁；Set 数 > Wait 数 → 计数残留（视计数器语义可能污染后续轮次）。逐 (flagId, 方向窗口) 验证并输出验证表。同步协议含双流（如 task/pair）且编号按流划分时，每条流的边各自独立验证。

### PIPE 参数与数据通路匹配（3510 生效，静态必做）

**2201 上 CrossCoreSetFlag/WaitFlag 的 PIPE 参数被忽略（阻塞全部流水）→ 旧代码任何 PIPE 值都能运行，形成"随便写都行"的假象；3510 上 PIPE 参数生效（只阻塞指定流水）→ 沿用旧值 = 同步约束未作用于目标流水。**

**PIPE 值必须等于"被等待的操作所在流水"**（逐同步点重选，不从旧代码沿用）。跨核场景：等待对端核完成某操作后其数据才可见/可覆写 → PIPE 填该操作所在流水：

| 等待的对端操作 | 所在流水 | PIPE 值 |
|---|---|---|
| L1 数据被装载消费（L1→L0A/L0B、L1→UB） | MTE1 | `PIPE_MTE1` |
| GM→L1 装载完成 | MTE2 | `PIPE_MTE2` |
| L0C→UB/GM 回写完成（Fixpipe） | FIX | `PIPE_FIX` |
| GM→UB 装载 / UB→GM 回写完成 | MTE3 | `PIPE_MTE3` |
| 矢量计算完成（UB 结果可用） | V | `PIPE_V` |

数据通路→流水对应关系来自架构规格（搬运单元与流水定义，`2201_to_3510_arch_changes.md` 与 `CrossCoreWaitFlag_ISASI.md` 的 pipe 支持表），不依赖具体算子实现。**PIPE 选错的表现是数据竞态而非死锁**：约束未作用于目标流水 → 行为时序相关、非确定性（偶发错值、偶发超时、特定 shape 组合偶发），与死锁诊断（确定性卡死）区分——排查时先按上表重画 PIPE 值，再查配对与计数。

### flagId 选号冲突检查（静态必做，迁移方案阶段）

**选择自定义跨核 flagId 前，先排除目标平台文档声明被占用的范围**（`CrossCoreSetFlag_ISASI.md`「flagId 冲突说明」；来源随文档版本核对）：

1. **Matmul 高阶 API 内部占用 `[0, 2×N−1]`**（N = 当前 kernel 中定义的 Matmul 对象数；最多 4 个对象 → 占用 `[0,7]`）。自定义跨核 flagId **不得落在这个区间**——Matmul 内部同步与自定义同步共用同一计数器，同号即互相消费/阻塞（表现为特定 shape 组合下的卡死或数据错乱，与参与集合错配现象相似）。
2. **KFC 同步通道占用 flagId 15/31**（源自 impl 源码，docs 无记录）：3510 上 Matmul 高阶 API 即 KFC（Kernel Fusion Compute）实现（`include/adv_api/matmul/matmul_intf.h`：`__NPU_ARCH__ == 2201 || 3510` 时走 `impl/adv_api/detail/kfc/kernel_kfc.h`，其余架构走普通 matmul），KFC 的跨核同步通道用 `KFC_SYNC_ID=15`（`impl/adv_api/detail/kfc/kfc_register_obj.h` ClearWorkspace：`CrossCoreSetFlag<PIPE_S>(KFC_SYNC_ID)` + `(KFC_SYNC_ID+16)`，SSBUF 与 GM 双模式）。**自定义 flagId 禁止使用 15/31；11-14 无文档占用记录但建议避开（框架预留区）——自定义 flagId 建议落 `[0,10]`**。docs 对该占用零记录（feature guide 标注"资料开发中"），只有挖 impl 源码才能确认——选号检查时必须执行本步，**不得仅凭 docs 无冲突记录就判定"无冲突"**。
3. **SyncAll 硬件同步接口占用**见 `SyncAll.md` 的「flagId 占用情况」章节，同样必须避开。
4. **选号后复核边界**：模式 4 下 AIC 侧发给 AIV1 需 +16，双发/双收把基号推向 `N+16`（AIC 侧合法区间 0-31，AIV 侧恒为 0-15）——基号必须落在 `[2N, 15]` 自由区间内；**若协议需要的边数 > 自由基号数，说明编号方案需重设计**（缩减边数、合并同参与集合的边），而不是把编号溢出进占用区。
5. **撞车处置后必须重跑计数平衡验证**：编号平移/重排后，每条边的 (flagId, 方向窗口) 验证表要重建——不同边不许合号，除非参与集合与触发语义完全一致。

### 跨核参数传递通道（3510 新增 SSBuffer）

**3510 新增 SSBuffer 核间存储**（`docs/zh/guide/cross_gen_migration_guide/instructions_for_new_features/3510_new_features.md` 特性 8；docs 标注"资料开发中"，以 `impl/` 源码为准）：AIC/AIV 通过标量直接访问对方写入的 SSbuf，零拷贝、无 GM 往返延迟。

跨核共享参数（tiling 派生值、每迭代更新的控制信息）传递时**优先评估 SSbuf 通道**：写入方写 SSbuf 后 `CrossCoreSetFlag<4, PIPE_S>(flagId)` 通知（SSbuf 消息通道即模式 4 支持 PIPE_S 的用途；KFC 的 SSBUF 通道同构，`impl/adv_api/detail/kfc/kfc_register_obj.h`：ClearSSbufImpl + `CrossCoreSetFlag<KFC_INTRA_MODE, PIPE_S>(KFC_SYNC_ID)`），接收方 `CrossCoreWaitFlag<4, PIPE_S>(flagId)` 后直读。GM/workspace 往返有搬移延迟 + 同步成本，参数量小且每迭代传递时差异显著。

**注意：SSbuf 通道占用一个 flagId（KFC 用 15），选号时计入占用区**（见「flagId 选号冲突检查」第 2 条）。

### 跨核同步封装模式（同步点稠密时的结构性手段）

同步点稠密（>3 对 Set/Wait）且分布在一个以上的函数/循环时，评估**封装模式**：为每对跨核 buffer 绑定生命周期——构造时自动分配 flagId（全局计数器自动避开占用区）、数据生产完成 Set、消费完成 Wait（Set/Wait 成对出现）、PIPE 随 buffer 数据通路分支。封装消除三类结构性缺陷：**漏配**（Set/Wait 必然成对出现）、**撞号**（编号自动分配，冲突检查收敛到一处）、**PIPE 错选**（PIPE 与 buffer 类型绑定，不存在逐点重选）。裸原语散布在算法循环中时，每个同步点都要单独论证——出错概率随同步点数线性增长，改版时漏配/撞号风险随之增长。

判定：迁移方案中给出"裸原语逐点核对"与"封装"的取舍及理由；同步点数 >3 时默认倾向封装。

### 本地事件消费语义（3510）

**SetFlag/WaitFlag（核内硬事件）在 3510 为消费语义**：WaitFlag 将标志位置 0，置 0 时阻塞；**每个 WaitFlag 必须对应一个未被消费的 SetFlag**。「SetFlag → 其它操作 → WaitFlag」三明治模式（如 `SetFlag<MTE3_MTE2>` → `CrossCoreSetFlag<..., PIPE_MTE3>` → `WaitFlag<MTE3_MTE2>`）在消费语义下必须论证：事件源（流水交接点）单次触发、与 WaitFlag 一一配对、不被跨迭代/跨分支错配消费——否则 WaitFlag 永久阻塞。核内事件计数与跨核 flag 计数都要做纸面推演（同上一节方法）。

### 硬事件链（官方样例 data_copy_l1togm 标准流水）

```cpp
CopyGmToL1A(a1Local);  CopyGmToL1B(b1Local);
SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
Load2DL1AToL0A(a1Local, a2Local);
Load2DL1BToL0B(b1Local, b2Local);
SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
Compute(co1Local, a2Local, b2Local);       // Mmad
SetFlag<HardEvent::M_FIX>(EVENT_ID2);      // Mmad 完成 → 释放 L0C
WaitFlag<HardEvent::M_FIX>(EVENT_ID2);
CopyL0CToGm(co1Local);                     // Fixpipe
```

**A5 常用硬事件类型**：

| 事件 | 用途 |
|---|---|
| `MTE2_MTE1` / `MTE1_MTE2` | GM→L1 装载与 L1→L0A/L0B 装载之间 |
| `MTE1_M` / `M_MTE1` | L0A/L0B 装载完成与 Mmad 之间 |
| `M_FIX` / `FIX_M` | **Fixpipe 与 Mmad 之间传递 L0C 所有权**（Mmad 完成可回写 / Fixpipe 结束可覆写） |
| `MTE2_MTE3` / `MTE3_MTE2` | UB 装载与 GM 回写之间 |
| `V_MTE3` / `MTE3_V` | Vector 计算（Duplicate 等）与 MTE3 回写之间 |

**通用要点**：
- 一个核计算、另一个核消费的流水结构必须成对添加 CrossCoreSetFlag/CrossCoreWaitFlag，**缺任何一侧的 Set/Wait 都会挂死**。
- 同步必须按**参与集合配对**（见上文模式语义表）：模式 2 下 AIC 1 次 SetFlag 对应 M 个 AIV 各 1 次 WaitFlag，反之 M 次 SetFlag 对应 1 次 WaitFlag——一侧按"单个对端"计数、另一侧按"全部对端"计数，协议即死锁。
- 多核写同一输出用两段式：每核写各自 workspace 分区 → `SyncAll()` 屏障 → 再从各分区读回累加（`SetAtomicAdd`/`SetAtomicNone` 原子累加可替代手工归并）。**轮空核（分不到数据的核）也要参与 `SyncAll()`，否则死锁**。
- L0C 缓冲数不足时在 Mmad/Fixpipe 之间加 `PipeBarrier<PIPE_M>()` 防 L0C 读写冲突。

### Mutex（950 新增）

核内异步流水指令之间的锁式同步，可替代部分 SetFlag/WaitFlag 组合。

### 配对协议检查（死锁高发区）

**flagId 含核身份索引（`taskId`/`blockIdx`/subblock 号参与索引，如 `FLAG[(taskId + 1) & 1][blockIdx & 1]`）即隐式假设"flagId 级配对粒度"**——用不同 flagId 区分不同对端（如 AIC 与 AIV0 用一组 flagId、与 AIV1 用另一组）。

**这类协议在 2201 上可能成立，在 3510 上必然失效的典型场景**：协议假设单对粒度（AIC↔指定 AIV），而目标平台模式 2 的参与集合是"全部 AIV"——等待方永远凑不齐全部对端 → **死锁**。

核对步骤（对应 stage_1 Step 1.5）：
1. 每个同步点推断"谁等谁"粒度：单对 / 组内全集合 / 全核
2. 与目标平台该模式的参与集合语义对照（上表）
3. 假设粒度 < 文档粒度 → 禁用该路径走标准流程；需单对粒度时改用模式 4（**核对型号支持，仅 950PR/DT**）

### 同步协议沿用决策规则（迁移方案阶段完成，全部静态可查）

**沿用 = 通过全部静态论证；任一项论证不了 → 禁用该路径走标准流程。禁止"沿用 + 运行时验证"式风险转移。**

1. **集合粒度 ≥ 目标平台文档粒度**（含等号）——对照上表模式参与集合。
2. **模式受支持、型号范围**——模式 4 仅 950PR/DT；其他型号仅 0/1/2。
3. **参数写法按目标平台文档生效**（静态：读文档）——**无参调用必须显式核对目标平台默认参数语义**：旧平台上被忽略的模板参数（modeId/pipe）在目标平台可能生效（如 910b 无参 `CrossCoreWaitFlag` 照常工作、950 按默认模式 0 执行，与配对 SetFlag 的模式号错位）——**不写参数，语义也在变**；模式号一致 ≠ 语义一致。
4. **参与集合恒定性**（静态：读 kernel）——同一 flagId 的 Set/Wait 必须在**所有参与核的无条件路径**上执行；`needExec`/`needPair` 条件分支、循环 0 次（分片边界）、空闲核 → 部分参与核缺 Set → 全集合语义下等待方永远等不到 → 死锁。
5. **与分片解耦**（静态：读 tiling 源码推导）——同步区执行次数与核数/分片无关；出方案时给出**参与集合推导表**（参与核 × 角色 × 执行条件 × flagId 映射），推导不闭合（存在某参与核可能缺 Set）即不满足。
6. **flagId 不与文档声明占用区冲突**（静态：读目标平台 API 文档「flagId 冲突说明」）——Matmul 高阶 API 占用 `[0, 2×N−1]`（N=Matmul 对象数）、SyncAll 占用见 `SyncAll.md`；自定义编号落在占用区 → 不满足。选号、边界复核（+16 双发）、撞车后的平移重排见「flagId 选号冲突检查」小节。
7. **3510 双主模式（AIC 与 AIV 均执行算法代码）下，含自定义跨核 flag 协议的路径禁止沿用 Matmul 高阶 API（REGIST_MATMUL_OBJ）**（静态：读 `matmul_intf.h` + `impl/adv_api/detail/kfc/`）——3510 上高阶 API 即 KFC 实现，REGIST 后 AIC 进入 **KFC server 循环**（`kernel_kfc.h` KfcServer：循环消费 AIV 委托请求直至 Quit），循环期间不执行算法代码中的任何自定义 CrossCoreSetFlag；且 KFC 同步通道占用 flagId 15/31，与自定义协议共享同一 flagId 空间。满足以下任一条件即违反本规则（Step 1.5 同步点清单直接喂入，静态可判）：
   - (a) 自定义跨核同步点位于主循环内、每迭代触发——AIC 被 server 循环占用，永远执行不到自定义 Set → 对端 Wait 永不解除 → 死锁；
   - (b) 跨核流水含 L1 通路（如 AIC 产 L1 → AIV 消费）——3510 的 UB→L1 通路由低阶 API 直接支撑（`docs/zh/guide/cross_gen_migration_guide/instructions_for_new_features/3510_new_features.md` 特性 5），高阶 API 的 L1 管理模式受限，需要精细控制装载/回写时机时不可控；
   - (c) 自定义 flagId 与 KFC 通道同号或相邻（15/31）——互相消费，行为与 shape 组合相关、非确定性。
   符合条件时**禁止"保留高阶 API + 仅避开编号"**（那是把正确性押在 KFC 通道的未文档化行为上），必须评估低阶直跑路径（CopyInL1 / Mmad / Fixpipe，见改动 2/3/4）；弃用高阶后第 6 条的 `[0, 2×N−1]` 占用区自动消失，自定义 flagId 可用区间恢复。

> 验证方法：参数生效性判定 = 读目标平台 API 文档语义 → 与参考实现写法逐字对照 → 用判别力强的测试验证（判别方法见 `cube-debug-lessons.md` Part 2）。

### 死锁诊断（aicore timeout，如 507014）——排查方法见 `cube-debug-lessons.md` Part 1

| 现象 | 原因 |
|------|------|
| 精度测试卡死，aicore timeout（507014），特定 shape/dtype 组合全部触发（如 fp16/bf16 命中某跨核路径，fp32 不命中） | 同步点集合粒度错配：flagId 配对的等待方永远等不到"全部对端" |

死锁**已发生**时的判别特征、定位方法（无同步路径对照法、PIPE 选错 vs 死锁的区分）与取证约束（死锁类问题 PRINTF/DumpTensor 无效，不得依赖插桩定位）见 `cube-debug-lessons.md` Part 1。

## 改动 7：Vector 伴随差异（cube 算子迁移必然触碰）

| 差异 | 说明 |
|---|---|
| GM 直接初始化 | `AscendC::InitGlobalMemory(gmTensor, size, initValue)` 替代"UB Duplicate → DataCopyPad 回写"两段式 |
| 非对齐/跨行输出 | `DataCopyPad` + `DataCopyExtParams`（blockCount/blockLen/dstStride）处理非 32B 对齐或跨行写 GM |
| UB bank 结构 | A5 UB 为 256KB（8 bank group × 2 bank × 16KB），与老平台（16×3×4KB）不同，reduce/gather 类临时 buffer 排布偏移按新布局重算 |
| 负载切分 | cube/vector 核的 S/M 分块从"全局一次性切分"改为**按 batch 逐轮切分**（每轮独立流水：pre1/pre2/pre3 → MID → end3/end2/end1），每轮之间用跨核事件衔接 |

## 改动 8：Host 侧（tiling / def）

| 差异 | 说明 |
|---|---|
| 平台注册 | `def.cpp` 必须新增 `this->AICore().AddConfig("ascend950", aicore_config)`；可为新平台扩展输入 datatype 组合与 `DynamicCompileStaticFlag` 等能力开关 |
| tiling 数据类型 | workspace 偏移、尺寸字段从 `uint32_t` 扩为 `int64_t`（A5 多核 workspace 规划更大，防溢出）；`get_*/set_*` 同步改 |
| workspace 规划 | 新增 per-core 分段（如 `dkCoreWorkspaceOffset`、`dkCoreSize`），每个 AIC 一份部分结果区，偏移对齐 `GM_ALIGN` |
| 确定性开关 | 新增 `deterministic` 字段，取 `attrs` 与 `context_->GetDeterministic() == 1`（全局确定性配置）的或值 |
| 参数校验 | 迁移时顺手补齐 `OP_CHECK_IF` 形状/取值校验，错误信息带上具体数值便于定位 |
| 调度基准 | `SetBlockDim`/`CalcTschBlockDim` 的取数口径（AIC 数 vs AIV 数）直接决定核利用率 |

## 改动 9：迁移 diff 审查清单（所有算子通用，审查"看起来无害"的删除/遗漏）

迁移完成后 **MUST 用 git diff 逐文件对照参考实现审查**。下面两类"顺手删除"与一类"迁移遗漏"是已实测的破坏性改动——**无编译错误、非触发路径不暴露，只有完整测试套件才暴露**。三类改动逐字核对：

**1. 编译期守卫（`if constexpr`）是"类型级"分支——禁止删除/扁平化**

`if constexpr (cond)` 在 cond=false 的实例化中整个块编译期消除，相关 buffer **不初始化**；改成裸 `{}` 后无该路径的用例仍执行块内代码 → 读未初始化 UB → 输出被随机污染。观测特征：无 X 路径错、同 shape 全 0 的 X 路径完美（同 shape 走 X 路径反而对）——**路径级对比是定位这种问题的最短路径**。审查 diff 时模板参数的每个使用点都要与参考实现逐字核对。

**2. 初始化调用删除也可能是正确适配——先查目标平台 API 实现，不能一刀切"恢复"**

参考实现的初始化调用在目标平台可能**不存在**（直接编译错误：undeclared identifier）——目标平台 API 实现可能已移除预初始化需求（预初始化表被忽略、改为直接逐元素计算）。此时删除 = 正确适配，恢复反而编译失败。**删除≠错误，恢复≠正确**：以编译结果 + 目标平台 API 实现（`asc/impl/adv_api/detail/...`）为准。

**3. host 框架级调用也可能被迁移遗漏——逐行核对 `SetBlockDim`/`SetScheduleMode`/`GetWorkspaceSizes` 顺序与条件分支**

这类行无编译错误、不影响非触发路径，只有触发路径（非对齐/特殊 shape 的分支）才暴露。案例特征：某 host 条件分支漏调度配置（如 BATCH 调度模式）→ 该分支触发时 aicore timeout 死锁（"预处理 kernel + 非全核 blockDim + 全核同步"组合必须配 BATCH 调度，否则核间同步等待不收敛），分支未触发的用例全部正常。**用 git diff 逐行核对 host 文件是唯一可靠方法**。

> 完整验证必须覆盖"所有功能路径"（验证通过必须定义到路径级别）的判别与拆解方法见 `cube-debug-lessons.md` Part 3。

## 官方数据通路变更总表（L2 指南"架构差异全景"的 Cube 侧展开）

下表来自官方 `2201_to_3510_arch_changes.md` 搬运单元/计算单元/存储单元/同步变更四张表，是 cube 算子迁移的硬件级依据：

| 通路/能力 | 2201（a2/a3） | 3510（A5） | 迁移对策 | 受影响 API |
|---|---|---|---|---|
| L1→GM 通路 | ✓ 支持 | ✗ **删除** | 纯 Cube 场景：GM 多分配单位矩阵，用 Mmad 算到 L0C 再 Fixpipe 到 GM；CV 融合场景：L1→UB→GM | DataCopy/DumpTensor |
| GM→L0A/L0B 通路 | ✓ 支持 | ✗ **删除** | 拆两步：GM→L1（Nd2NzParams）+ L1→L0A/L0B（LoadData2DParamsV2） | LoadData |
| UB→L1 通路 | ✗ | ✓ **新增** | 可直接 UB→L1，无需经 GM 中转 | DataCopy（UBToL1） |
| L0C→UB 通路 | ✗ | ✓ **新增**（单向） | Fixpipe 直接 L0C→UB（`FixpipeConfig.isToUB = true`），配合 dualDstCtrl/subBlockId | Fixpipe（L0CToUB） |
| L0C→GM 随路转换 | NZ2ND（V220） | NZ2ND + **NZ2DN（950 新增，CFG_COLUMN_MAJOR）** | 输出 DN 格式用 950 专属 COLUMN_MAJOR | Fixpipe |
| LoadData 能力 | — | 扩展 MX（MicroScaling）、ND-DMA、loop 模式 | 按需使用 | LoadData_2D_MX / DataCopy NDDMA / SetLoopModePara |
| L0A 分形 | ZZ | **NZ** | 非切分兼容；**切分场景按 NZ 重新计算 L0A 地址**（改动 5） | LoadData / LoadDataWithTranspose |
| L1→L0A 转置装载 | LoadData 支持 | **LoadData 不再支持，独立 API** | 转置场景用 `LoadDataWithTranspose`（**950 仅支持 L1→L0B 通路**） | LoadDataWithTranspose |
| L0A/L0B 初始化指令 | ✓ | ✗ **删除** | `Fill` 只支持非 L0A/L0B 位置 → 先 Fill L1 再 LoadData 到 L0A/L0B | Fill |
| L1 边界值设定 | ✓ | ✗ **删除** | `SetLoadDataBoundary` 失效；需要 L1 循环读取时手动拆分多条指令、手动绕回（样例 set_loaddata_boundary） | SetLoadDataBoundary |
| int4b（s4）Cube 计算 | ✓ | ✗ **不支持** | 先 Cast：int4b→half→int8，再做 int8 Mmad（样例 matmul_s4）；图层面可在算子前插 Cast 节点 | LoadData / Mmad |
| 4:2 结构化稀疏 | ✓ | ✗ **不支持** | 不用 LoadDataWithSparse/MmadWithSparse，改用稠密矩阵计算，或利用 Vector 核做稠密转稀疏 | LoadDataWithSparse / MmadWithSparse |
| 核间同步 | 模式 0/1/2 | 模式 0/1/2/**4**（**模式 4 = AIC↔单个 AIV，3510 新增**，仅 950PR/DT）+ Mutex | C/V 配对流水必须显式跨核握手；**同步点参与集合粒度须与目标平台模式语义核对** | CrossCoreSetFlag / CrossCoreWaitFlag / Mutex |
| UB 结构 | 16 bank group × 3 bank × 4KB | 8 bank group × 2 bank × **16KB** | 避免 bank 冲突的排布偏移按新结构重算 | — |
| SSBuffer | 无 | ✓ 新增 | AIC/AIV 核可通过 Scalar 访问的核内存储单元 | — |

> **转置装载执行规则**：kernel 中出现转置装载需求时，**不要**试图在 `LoadData` 内完成转置——950 上转置装载走独立 API `LoadDataWithTranspose`，且仅支持 L1→L0B 通路；左矩阵（A）转置需在装载前用 vector 预转置，或检查 `LoadData2DParamsV2.ifTranspose` 是否满足通路要求后再使用。

## Cube 迁移检查清单

- [ ] **1. 搭骨架**：建 `arch22/`、`arch35/` 目录；入口 `.cpp` 加 `#if (__CCE_AICORE__ == 310)` include 分发；`def.cpp` 注册 `ascend950` 平台；确认 310 宏在编译链路中生效（可用空实现先打通编译）。
- [ ] **2. 换装载**：所有 `LoadData2DParams` / `LoadData3DParamsV2` 调用改 `LoadData2DParamsV2`（注意 mStep/kStep/srcStride/dstStride 的单位）；删除 `SetLoadDataPaddingValue`/`SetNdParaImpl`/`SetFixpipeNz2ndFlag` 等全局设置。
- [ ] **3. 换回写**：`DataCopyCO12DstParams` 改 `FixpipeParamsArch3510`（或 V220）+ `Fixpipe`；确认 nSize 16 倍数、dstStride 单位（element vs datablock）、`isToUB` 与 UB 双目标布局匹配。
- [ ] **4. 适配分形**：检查是否 L0A 切分场景——是则按 NZ 分形重排装载循环；非切分场景 V2 整块装载即可。
- [ ] **5. 重排分块**：按 A5 的 L0/L1/UB 容量重设 `M/S2/K` 基本块大小与乒乓缓冲数；大 M 引入 splitM 循环。
- [ ] **6. 补同步**：为每一对 C→V 数据依赖补 `CrossCoreSetFlag/WaitFlag`（模式 4）；L0C 生命周期加 `M_FIX/FIX_M` 事件；多核共享输出用 per-core workspace + `SyncAll` + 原子累加。
- [ ] **7. 更新 tiling**：类型扩 `int64_t`、新增 per-core workspace 偏移与确定性字段；`common.h` 的 `ConstInfo`/`RunInfo` 与 tiling 结构保持同步。
- [ ] **8. 回归验证**：双平台编译（310 与老宏）互不影响；精度对齐（含 TND padding、确定性开关）；性能对比分块参数是否吃满 L0C。

## 高频踩坑速查表

| 现象 | 原因 |
|---|---|
| 310 编译后 kernel 是空的 | 入口宏仍保留"310 跳过"的旧分支，迁移后必须移除 |
| `LoadData` 参数对不上（字段不存在） | 新旧结构同名不同义：`LoadData2DParams`（repeatTimes/srcStride）≠ `LoadData2DParamsV2`（mStep/kStep） |
| 装载结果错位 | mStep 单位是 16 元素、kStep 是 32 字节、stride 是 512 字节（分形），混用必错 |
| 转置装载失败/结果错误 | 950 上 `LoadData2DParamsV2.ifTranspose` 只覆盖装载级转置；需 L1→L0A 转置时 950 不支持 LoadDataWithTranspose 的 L0A 通路，要预先转置或换方案 |
| 死锁/卡死（aicore timeout 507014） | `CrossCoreSetFlag`/`WaitFlag` 没有按参与集合配对；或同步点集合粒度错配（flagId 配对假设单对、目标平台模式是全部 AIV）；或轮空核未参与 `SyncAll`；或用了不支持的 PIPE_S/PIPE_ALL |
| 偶发数据错误 | L0C 读写冲突：Fixpipe 未等 `M_FIX` 事件就覆写 L0C；需 `PipeBarrier<PIPE_M>()` |
| 回写结果错位 | FixpipeParamsArch3510 的 dstStride 单位是 element（V220 是 datablock），照搬旧参数必错；nSize 未按 16 对齐 |
| 大 shape 出界/异常 | workspace 偏移用 `uint32_t` 溢出，需 `int64_t`；per-core 分区未按 `GM_ALIGN` 对齐 |
| 精度不齐 | M=1 分块未补齐为 2；`cmatrixInitVal` 未置 true 导致 L0C 残留；L0A 切分场景仍用 ZZ 分形地址 |
| int4b 数据计算错误/不支持 | 950 Cube 不支持 int4b，需先 Cast 到 int8（int4b→half→int8 两步） |
| 涉及 L0A/L0B 初始化 | 950 删除 L0A/L0B 初始化指令，Fill 只作用于 L1 等位置，需 Fill L1 后再 LoadData |

> 表中条目的**判别与定位方法**（现象如何确认、如何缩小范围）见 `cube-debug-lessons.md`：死锁/卡死（含连锁污染、PIPE 选错区分）→ Part 1；装载错位/偶发数据错误/回写错位/精度不齐等结果错误（含 mask 与消费粒度）→ Part 2；单路径 PASS 但整体失败（路径级验证）→ Part 3。

## 标准实现样例索引（asc-devkit 全量仓）

迁移前先读对应样例，可大幅降低踩坑率：

| 迁移主题 | 样例路径（`$DEVKIT_PATH`） |
|---|---|
| L1→GM 通路删除的绕行方案 + LoadData2DParamsV2 + Fixpipe + M_FIX 事件链的标准全流程 | `examples/01_simd_cpp_api/06_compatibility_guide/data_copy_l1togm/` |
| L0A 分形 ZZ→NZ、L0A 切分场景适配 | `examples/01_simd_cpp_api/06_compatibility_guide/pattern_transformation/` |
| int4b→int8 的 Cast 兼容计算 | `examples/01_simd_cpp_api/06_compatibility_guide/matmul_s4/` |
| L0A/L0B 初始化指令删除后的 Fill 替代方案 | `examples/01_simd_cpp_api/06_compatibility_guide/fill/` |
| SetLoadDataBoundary 删除后的手动绕回 | `examples/01_simd_cpp_api/06_compatibility_guide/set_loaddata_boundary/` |
| Matmul 高阶 API 系列（L0Cache/L2Cache/MX/双缓冲等） | `examples/01_simd_cpp_api/04_advanced_api/00_matmul/` |

## 低阶直跑改造常见编译错误速查（FA-score arch22→arch35 实测沉淀）

> 来源：flash_attention_score 迁移构建失败复盘（ascend950 全量模板实例化构建）。四类错误覆盖 181 处报错、拖垮 30+ 模板轴组合，全部为**签名/约束级**错误（API 存在但调用不兼容），与改动 1-6 的"选哪个 API"问题正交。构建失败时 MUST 先查本表再去 devkit 文档。

| 错误特征 | 根因 | 修复方法 |
|---------|------|---------|
| `no matching function for call to 'DataCopyGmNDToL1'` + `candidate not viable: 1st argument ... would lose const qualifier`（大面积爆发，实测 96 处） | 低阶直跑封装要求**非 const 左值引用**（`DataCopyGmNDToL1(LocalTensor<L1Type> &l1Tensor, ...)`——内部要写 ND→NZ 布局参数），而调用链上游以 `const LocalTensor<T>&` 传播目标 tensor，const 限定符一路传播到调用点导致候选全部不可行 | 沿调用链**去 const**：上游拷贝函数（如 `CopyKeySlice`/`CopyValueSlice`）的 dst 参数从 `const LocalTensor<T>&` 改为 `LocalTensor<T>&`；禁止改封装 API 签名迁就 const |
| `no matching constructor for initialization of '...Kernel<...>'`（跨全部 dtype/layout/pse/mask/drop/rope 模板组合成片出现，实测 61 处） | 入口 `GET_TILING_DATA_PTR_WITH_STRUCT(T, tilingData, tiling)` 宏展开产出的 tilingData 形态（指针/引用/`__gm__` 限定）与 Kernel 构造函数形参（如 `FA35Kernel(const __gm__ optiling::XxxTilingData *td)`）不匹配——取 tiling 宏家族（`WITH_STRUCT` vs `PTR_WITH_STRUCT`）产出类型不同 | 先 `grep -rn "define GET_TILING_DATA" $CANN_INCLUDE` 确认宏展开类型，再修正一侧：构造形参改匹配宏产物，或换用产出指针形态的宏；禁止盲目加/减解引用试错 |
| `no matching function for call to '<辅助函数>'`（点状出现，实测 ComputeInnerPseOffset 8 处） | **函数只有调用没有定义**——arch22 侧辅助函数在移植时定义遗漏（调用点照抄了但实现没搬），并非签名真的不匹配 | `grep -rn <函数名> op_kernel/ common/` 确认定义存在性；缺失则从 arch22 源补齐定义或改为等价已有函数。此错误是"调用先于定义"的移植遗漏信号，MUST 回对照 arch22 侧文件清单 |
| `static assertion failed due to requirement 'sizeof(UbLayout) <= 248 * 1024': UB buffer too large`（实测 16 处） | 自研 `struct UbLayout` 把多块 UB buffer 打包成结构体并 `static_assert` 守门，总量超 3510 可用 UB 编译期上限 **248KB** | 缩减 UB 占用：优先 `GetWithOffset` 多阶段复用压峰值，其次缩 tile 尺寸；static_assert 是守门约束，**禁止删断言/放大常量绕过** |

**通用判别规则**：3510 全量模板实例化构建会把单点签名错误**放大到所有模板组合**（一次构建报 60-180 处）。禁止按报错数量评估问题规模——按错误类型去重后通常只有 2-4 类根因，修一类消一片；统计根因类别用 `grep "error:" build.log | sed 's/.*error: //' | sort | uniq -c | sort -rn`。

## 相关参考文档路径

| 文档类别 | 路径 | 说明 |
|---------|------|------|
| 架构变更（权威） | `$DEVKIT_PATH/docs/zh/guide/cross_gen_migration_guide/3510_arch_migration/2201_to_3510_arch_changes.md` | 数据通路/计算/存储/同步变更详述（本指南"变更总表"出处） |
| 逐 API 迁移 | `$DEVKIT_PATH/docs/zh/guide/cross_gen_migration_guide/3510_arch_migration/2201_to_3510_guide/` | 数据搬运/矩阵计算章节 |
| LoadData2DParamsV2 参数 | `$DEVKIT_PATH/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/LoadData_2D_V2.md` | 参数单位、约束、地址公式 |
| LoadDataWithTranspose | `$DEVKIT_PATH/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/LoadDataWithTranspose.md` | A5 专属、通路限制 |
| Fixpipe（L0C→GM / L0C→UB） | `$DEVKIT_PATH/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_store/` | FixpipeParamsArch3510 全字段、FixpipeConfig |
| 跨核同步 | `$DEVKIT_PATH/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreSetFlag_ISASI.md` | 模式 0/1/2/4、型号支持范围、pipe 约束 |
| 同步模式语义（权威） | `$DEVKIT_PATH/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/key_features.md` | 各模式参与集合定义、flagId 计数器驱动条件 |
| 头文件原型 | `$DEVKIT_PATH/include/basic_api/kernel_operator_mm_intf.h` | LoadData 重载家族、LoadDataWithTranspose 声明 |

> 注意：devkit 仓库实际路径含 `zh` 层（`docs/zh/guide/`、`docs/zh/api/`）；skill 索引文件（knowledge-index.md / search-rules.md）中写的 `docs/zh/guide/`、`docs/zh/api/` 缺少 `zh` 层，按本表实际路径访问。
