# L1 迁移实施详细指南

> **适用对象**：vector 与 cube 类算子通用——本指南的 host 侧配置步骤（_def.cpp / config 目录 / 入口分发）不区分算子类别；cube 类算子 kernel 侧的 A5 差异（装载/回写/分形/同步）见 `cube-migration-guide.md`。

## 改动 1：_def.cpp 独立配置

950 使用独立 `OpAICoreConfig`，通过 `opFile` 切换到 RegBase kernel：

```cpp
OpAICoreConfig regbaseCfg;
regbaseCfg.DynamicCompileStaticFlag(true)
    .DynamicRankSupportFlag(true)
    .DynamicShapeSupportFlag(true)
    .ExtendCfgInfo("opFile.value", "算子名_apt");  // 指向 RegBase kernel
this->AICore().AddConfig("ascend950", regbaseCfg);
```

如果 950 需要扩展数据类型（如新增 FP8/HiFloat8），在独立配置中添加：

```cpp
regbaseCfg.Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8,
               ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN})
    .Format({...});
```

**配置层共性特征**：

| 共性特征 | 说明 |
|---------|------|
| 独立 OpAICoreConfig | 950 使用独立的 `OpAICoreConfig` |
| opFile.value 设置 | 设置 `opFile.value = "算子名_apt"` |
| DynamicCompileStaticFlag | 950 配置普遍开启动态编译静态化 |
| DynamicRankSupportFlag | 950 配置普遍开启动态 Rank 支持 |
| DynamicShapeSupportFlag | 950 配置普遍开启动态 Shape 支持 |

### 配置层迁移模式

#### 模式 A：独立配置 + opFile 切换

950 使用独立 `OpAICoreConfig`，通过 `opFile` 切换到 RegBase kernel 入口文件。

```cpp
auto &config910b = this->AICore().AddConfig("ascend910b");
config910b.SetOpFile("moe_init_routing");

auto &config950 = this->AICore().AddConfig("ascend950");
config950.SetOpFile("moe_init_routing_apt");
```

**适用算子**：GroupNormSwish、SwigluQuant、MoeInitRouting 等

#### 模式 B：独立配置 + 扩展数据类型

950 使用独立配置，同时扩展 FP8/HiFloat8/INT8 等量化数据类型。

```cpp
auto &config950 = this->AICore().AddConfig("ascend950");
config950.SetOpFile("kv_rms_norm_rope_cache_apt");
config950.Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8,
               ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN})
    .Format({...});
```

**适用算子**：DequantSwigluQuant（6→76 组数据类型）、KvRmsnormRopeCache（8→26 组数据类型）

## 改动 2：创建 _apt.cpp

```cpp
// 算子名_apt.cpp — RegBase kernel 入口
#include "arch35/算子名_impl.hpp"    // ← 仅 include 路径变更
#include "arch35/算子名_bf16.hpp"
#include "arch35/算子名_single.hpp"

extern "C" __global__ __aicore__ void 算子名(GM_ADDR input_gm, GM_ADDR output_gm,
                                              GM_ADDR workspace, GM_ADDR tiling) {
    // 与 910b 版本相同的逻辑
}
```

**关键差异**：
- include 路径从根目录改为 `arch35/`
- Tiling 传参可能从 `tiling` 改为 `tempTilingGm`（解析后的结构体）

## 改动 3：创建 arch35/ 目录

将根目录下的实现头文件复制到 `arch35/`，做以下微调：

| 调整项 | 910b 版本 | 950 arch35/ 版本 |
|--------|-----------|-----------------|
| BF16 条件编译 | `#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220` | 直接支持，移除条件编译 |
| BF16 架构保护 | `#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 \|\| __NPU_ARCH__ == 3113))` | 移除保护 |
| Tiling 传参 | `op.Init(..., tiling)` — 传原始 tiling 指针 | `op.Init(..., tempTilingGm)` — 传解析后的结构体 |

**BF16 条件编译移除示例**：

```cpp
// 910b 版本：
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
  BF16 路径
#endif

// 950 arch35/ 版本：
BF16 路径（直接支持，无需条件编译）
```

**BF16 架构保护移除示例**：

```cpp
// 910b 版本：
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    MoeFinalizeRouting::MoeFinalizeRoutingBf16CutK<bfloat16_t> op;
    op.Init(...);
    op.Process();
#endif

// 950 arch35/ 版本：
    MoeFinalizeRouting::MoeFinalizeRoutingBf16CutK<bfloat16_t> op;
    op.Init(...);
    op.Process();
```

## 改动 4：CMakeLists.txt

```cmake
set(SUPPORT_COMPUTE_UNIT "ascend950")
set(SUPPORT_TILING_DIR "arch35")
add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR}
    OPTYPE 算子名 ACLNNTYPE aclnn_exclude COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT}
    TILING_DIR ${SUPPORT_TILING_DIR} DISABLE_IN_OPP TRUE)
```

**注意**：如果 `op_host/CMakeLists.txt` 中已有 ascend950 的 `add_ops_compile_options` 分支，直接复用，不要重复添加。

## 改动 5：config/ascend950/

从 `config/ascend910b/` 复制两个文件：

```bash
mkdir -p op_host/config/ascend950
cp op_host/config/ascend910b/<算子名>_binary.json op_host/config/ascend950/
cp op_host/config/ascend910b/<算子名>_simplified_key.ini op_host/config/ascend950/
```

**binary.json**：定义算子的二进制 kernel 编译信息（输入输出数据类型、格式等）。
**simplified_key.ini**：定义 opc 工具编译时的 `--simplified_key_mode` 选项值。

---

## 编译环境注意事项

### bisheng 脚本 shebang 问题

950 编译时，构建系统会在 `build/gen_bisheng_dir/` 下自动生成 `bisheng` 脚本：

```bash
ccache_args="/usr/bin/ccache <cann_path>/bin/bisheng"
args=$@
eval "${ccache_args} $args"
```

**问题**：该脚本缺少 `#!/bin/bash` shebang 行，导致 `Exec format error`。

**修复**：在首行添加 `#!/bin/bash`：

```bash
#!/bin/bash
ccache_args="/usr/bin/ccache <cann_path>/bin/bisheng"
args=$@
eval "${ccache_args} $args"
```

**注意**：每次 `rm -rf ./build` 后需重新修复，因为该文件是自动生成的。

### 编译命令格式

```bash
source <cann_path>/set_env.sh
cd ops-transformer
rm -rf ./build
bash build.sh --pkg --ops=<算子名> --soc=ascend950
```

**注意**：`--soc` 是双横线，不是 `-soc`。

### 日志落盘与错误定位

```bash
bash build.sh --pkg --ops=<算子名> --soc=ascend950 2>&1 | tee build.log
# 搜索错误
grep -n "[Ee]rror" build.log
```

重点关注：
- `[ERROR] TBE` — TBE 编译器错误（通常是 kernel 编译问题）
- `OSError: [Errno 8]` — bisheng 脚本 shebang 问题
- `gmake: *** [Makefile:156: all] Error 2` — 顶层编译失败，需查看具体子错误
- `Error 137` — 进程被 kill（通常是 OOM）

---

## L4 升级信号指引

> 注：本节"L4"为旧分级口径（Tiling 适配 + 数据类型扩展）；现行层级模型只有 L1/L2/L3（见 SKILL.md 决策树），下列信号命中时按现行模型评估为"超出 L1，需升级评估"。

当 L1 迁移不足以覆盖 950 适配需求时，需升级评估（Tiling 适配 + 数据类型扩展）。以下为判断信号：

| 升级信号 | 判断依据 | 说明 |
|---------|---------|------|
| Tiling 需要 `IsRegbaseSocVersion` 判断 | Tiling 中需根据芯片架构动态调整 UB 可用空间 | 950 SIMT DCache 复用 UB，必须预留 40KB |
| UB 预留不足 | 当前预留值 < `SIMT_UB_SIZE_BYTE`(40960) | 使用 `UB_REVERSE`/`UB_RESERVED_BYTE` 等自定义常量（如 1024）均不合规 |
| 950 需要扩展数据类型 | 需新增 FP8/HiFloat8/INT8 量化支持 | 量化类算子（如 DequantSwigluQuant、KvRmsnormRopeCache）数据类型组合大幅扩展 |
| Tiling 结构需要重构 | 950 使用全新 Tiling Key 体系 | 如 MoeGatingTopKSoftmax 使用 `TILING_KEY_IS(10000)`/`TILING_KEY_IS(20000)` 等新 Key |
| 涉及 Subnormal 处理 | 算子使用 Exp/Ln/Sqrt/Rsqrt/Div/Reciprocal 等 API | 351x 默认不支持 Subnormal，需通过 Config 配置软仿 |

**决策流程**：

```
算子是否使用 Exp/Ln/Sqrt/Div/Reciprocal/Rsqrt？
  └─ 是 → L4（需 Subnormal Config 配置）
  └─ 否 → UB 使用量是否需要预留 SIMT 空间？
        └─ 是 → L4（需 IsRegbaseSocVersion + SIMT_UB_SIZE_BYTE）
        └─ 否 → 950 是否需要扩展数据类型？
              └─ 是 → L4（需独立配置 + 数据类型扩展）
              └─ 否 → Tiling 结构是否需要重构？
                    └─ 是 → L4（需全新 Tiling Key 体系）
                    └─ 否 → 维持 L1
```

---

## UB 预留规范

### 950 平台 UB 预留要求

950 平台 SIMT DCache 复用 UB 空间，**必须**预留 `SIMT_UB_SIZE_BYTE = 40960`（40KB）。未预留或预留不足将导致 UB 越界风险。

### 各算子 UB 预留合规状态

| 算子 | Tiling 文件 | IsRegbaseSocVersion 判断 | SIMT_UB_SIZE_BYTE 预留 | 可用 UB（扣预留后） | 改造状态 | 合规性 |
|------|------------|------------------------|----------------------|---------------|---------|--------|
| MoeInitRouting | `moe_init_routing_tiling.cpp` | ✓ 有 | ✓ `ubSizePlatForm - SIMT_UB_SIZE_BYTE` | 216KB | 已完成 | ✓ 合规 |
| MoeGatingTopKSoftmax | `moe_gating_top_k_softmax_tiling_arch35.cpp` | ✓ 有 | ✗ 未预留 | 256KB（未扣） | 未改造 | ✗ 不合规 |
| DequantSwigluQuant | `dequant_swiglu_quant_tiling_arch35.cpp` | ✓ 有 | ✗ 使用 `UB_REVERSE`(1KB) | 部分改造 | △ 需确认 |
| KvRmsnormRopeCache | `kv_rms_norm_rope_cache_base_tiling.cpp` | ✓ 有 | ✗ 使用 `UB_RESERVED_BYTE`(1KB) | 部分改造 | △ 需确认 |
| GroupNormSwish | `group_norm_swish_tiling.cpp` | ✗ 无 | ✗ 未预留 | 未改造 | ✗ 不合规 |
| MoeComputeExpertTokens | `moe_compute_expert_tokens_tiling.cpp` | ✗ 无 | ✗ 未预留 | 未改造 | ✗ 不合规 |
| SwigluQuant | `swi_glu_quant_tiling.cpp` | ✗ 无 | ✗ 未预留 | 未改造 | ✗ 不合规 |
| MoeFinalizeRouting | `moe_finalize_routing_tiling.cpp` | ✗ 无 | ✗ 未预留 | 未改造 | ✗ 不合规 |
| InterleaveRope | `interleave_rope_tiling.cpp` | ✗ 无 | ✗ 未预留 | 未改造 | ✗ 不合规 |

### UB 空间使用档案

各算子的 UB 容量 / SIMT 预留 / 可用 UB / 用途 → 见 `ub-budget-guide.md`「四、10 个典型算子 UB 用量参考卡」（UB 用量的唯一权威表，含数据来源）。

### 统一 UB 预留模板代码

```cpp
namespace Ops { namespace Common {
    constexpr int64_t SIMT_UB_SIZE_BYTE = 40960;

    inline uint64_t GetAvailableUbSize(platform_ascendc::PlatformAscendC& platform,
                                        bool isRegbase) {
        uint64_t ubSizePlatForm;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        if (isRegbase) {
            ubSizePlatForm -= SIMT_UB_SIZE_BYTE;
        }
        return ubSizePlatForm;
    }
}}
```

**Tiling 中标准用法**：

```cpp
const static int64_t SIMT_UB_SIZE_BYTE = 40960;

uint64_t ubSizePlatForm;
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
aicoreParams_.ubSize = ubSizePlatForm;
if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
    aicoreParams_.ubSize = ubSizePlatForm - SIMT_UB_SIZE_BYTE;
}
```

---

## Kernel 入口层迁移模式

### 模式 A：直接复用根目录实现

arch35/ 代码与根目录代码逻辑完全一致，仅 include 路径不同。

```cpp
#include "arch35/group_norm_swish_base.h"
```

**适用算子**：GroupNormSwish

### 模式 B：编译时宏切换（SIMT 优化）

950 使用 SIMT 替代传统 Scatter/Gather，通过 `__NPU_ARCH__` 宏在同一 `_apt.cpp` 中切换实现路径。

```cpp
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "arch35/moe_src_to_dst_simt_op.h"
#endif
#include "arch35/moe_src_to_dst_op.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    MoeSrcToDstSimtOp srcToDstSimtOp;
    srcToDstSimtOp.Init(expandedRowIdx, userWS, t);
    srcToDstSimtOp.Process();
#else
    TPipe srcToDstPipe;
    MoeSrcToDstOp srcToDstOp;
    srcToDstOp.Init(expandedRowIdx, userWS, t, &srcToDstPipe);
    srcToDstOp.Process();
#endif
```

**适用算子**：MoeInitRouting

### 模式 C：溢出模式控制 + Register-based 重写

950 使用 MicroAPI 重写核心计算路径，并通过 SPR 寄存器控制溢出模式。

```cpp
#define GLOBAL_OVERFLOW_MODE_CTRL 60

#if (__NPU_ARCH__ == 3510)
    int64_t globalOriOverflowMode = AscendC::GetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>();
#endif

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>(0);
#endif

    // ... 核心计算（使用 MicroAPI Register-based 实现）...

#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>(globalOriOverflowMode);
#endif
```

**适用算子**：KvRmsnormRopeCache

### 模式切换机制汇总

| 机制 | 使用算子 | 实现方式 | 适用场景 |
|------|---------|---------|---------|
| 编译时宏切换 | MoeInitRouting, KvRmsnormRopeCache | `#if (__NPU_ARCH__ == 3510)` | 同一 _apt.cpp 中选择不同实现路径 |
| opFile 切换 | 所有深度迁移算子 | _def.cpp 中 `opFile.value = "xxx_apt"` | 编译时选择不同的 kernel 入口文件 |
| Tiling 运行时切换 | MoeInitRouting, DequantSwigluQuant, KvRmsnormRopeCache | `IsRegbaseSocVersion()` | 运行时根据芯片平台调整 Tiling 参数 |

---

## Tiling 层迁移模式

### 模式 A：UB 容量 + SIMT 预留

950 Tiling 中获取 UB 容量后，扣除 SIMT DCache 预留（`SIMT_UB_SIZE_BYTE` = 40KB）。模板与标准用法见上文「UB 预留规范 → 统一 UB 预留模板代码」，不重复。

### 模式 B：架构判断 + Tiling Key 分流

950 使用全新 Tiling Key 体系，通过 `IsRegbaseSocVersion` 判断是否进入 950 Tiling 路径。

```cpp
if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
    return false;
}
// 950 使用 TILING_KEY_IS(10000) / TILING_KEY_IS(20000) 等新 Key
```

### 模式 C：自定义预留常量（需确认合规性）

使用自定义常量预留 UB 空间，但预留值可能不足。

```cpp
// DequantSwigluQuant — 使用 UB_REVERSE = 1024（仅 1KB，远小于 40KB）
constexpr int64_t UB_REVERSE = 1024;
int64_t ubAvailable = ubSize_ - UB_REVERSE - 1 * BLOCK_SIZE;

// KvRmsnormRopeCache — 使用 UB_RESERVED_BYTE = 1024（仅 1KB，远小于 40KB）
static constexpr int64_t UB_RESERVED_BYTE = 1024;
int64_t ubFlexible_ = ubSize_ - UB_RESERVED_BYTE - ...;
```

**⚠️ 风险提示**：`UB_REVERSE` 和 `UB_RESERVED_BYTE` 均为 1024（1KB），远小于 `SIMT_UB_SIZE_BYTE = 40960`（40KB）。如果 950 平台上 SIMT DCache 确实复用 UB 空间，这两个算子存在 UB 越界风险。**必须替换为标准 `SIMT_UB_SIZE_BYTE = 40960`**。

### Tiling 层共性特征

| 共性特征 | 说明 |
|---------|------|
| IsRegbaseSocVersion() | 深度迁移算子的 Tiling 中普遍使用此函数判断架构 |
| UB 空间预留 | 950 上需预留 SIMT_UB_SIZE_BYTE（40KB） |
| 排序粒度调整 | 涉及排序的算子（MoE 系列），mrgSortListMaxElement 从 1024→2048 |

---

## Subnormal 处理（L1 视角：升级信号）

**L1 阶段不改 kernel 代码，因此本阶段只需识别、不需适配**：算子若使用 `Exp / Ln / Reciprocal / Sqrt / Rsqrt / Div`，在 950 上存在 subnormal 精度差异——这是一个**升级信号**（见上文「L4 升级信号指引」），说明该算子不能停留在纯 L1，需按 stage_1 定级升级到 L2 处理。

- **API 知识**（algo 参数取值含义、Config 结构体、2200x/3510 代码对比）：`cannbot-skills/ops/ascendc-api-best-practices/references/api-cross-gen-migration.md`
- **L2 改造动作**（换 algo 模板参数 / Reg 路径用 `SpecificMode`）：`l2-guide.md`「补充 3：Subnormal 适配」
- **扫描与适配策略**（策略 0 结构性排除、eps 可表示性检查、何时必须处理）：`api-diff-guide.md` §1

## 相关参考文档路径

> 下表中带「KB 快照」标注的目录位于插件自带 KB 快照 `plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/`（本方法论 2026-05 旧版存档），按此实际路径访问。

| 文档类别 | 路径 | 说明 |
|---------|------|------|
| 迁移相关官方文档 | KB 快照 `migration/` 子目录 | 220x→351x 架构迁移指导、基础/高阶 API 迁移指导、算子编译迁移指导、兼容性说明 |
| Memory-based Vector 操作 | KB 快照 `memory-base-vector/` 子目录 | 传统 AscendC Vector 编程模式参考（TPipe/TQue/DataCopy 等） |
| API 选型与概述 | KB 快照 `api-overview/` 子目录 | API 兼容性分层、高阶/基础/MicroAPI/SIMT 接口概述 |
| Cube 类算子迁移 | `references/impl/cube-migration-guide.md` | Cube 数据通路/分形/跨核同步/Fixpipe 等 cube 专用迁移（含官方样例索引与踩坑速查） |
