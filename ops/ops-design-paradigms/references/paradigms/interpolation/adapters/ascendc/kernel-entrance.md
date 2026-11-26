<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Interpolation 范式 Kernel 入口与 Tiling 组织

> AscendC SIMT 的入口模式、TilingData/TilingKey 组织与 host 侧要点。以 three_interpolate / crop_and_resize 实测结构为基准。

## 1 入口结构（三段式）

```cpp
template <uint32_t schMode>
__global__ __aicore__ void three_interpolate(
    GM_ADDR features, GM_ADDR idx, GM_ADDR weight,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ThreeInterpolateTilingData);
    GET_TILING_DATA_WITH_STRUCT(ThreeInterpolateTilingData, tilingData, tiling);  // 1. 解 tiling
    // 2. 32/64 位索引路径分派（见 §3）
    if (tilingData.totalElements <= static_cast<uint64_t>(INT32_MAX)) {
        NsThreeInterpolate::Process<DTYPE_FEATURES, DTYPE_IDX, uint32_t>(...);
    } else {
        NsThreeInterpolate::Process<DTYPE_FEATURES, DTYPE_IDX, uint64_t>(...);
    }
}
```

- `__simt_vf__` 内核函数不能是类的成员函数；参数只支持裸指针与基本类型（`CAP_VF_PARAM_MAX` = 28）
- 纯 SIMT 正向算子单段 `asc_vf_call`；grad 两段（零初始化 + 散射）段间同步见 [sampling-ops.md](sampling-ops.md) §6

## 2 TilingData 设计规则

| 规则 | 说明 |
|------|------|
| 字段 = host 可算的标量 | three_interpolate：`bs/ns/cs/ms` + `totalElements` + `needCoreNum` |
| **不含线程数** | 线程数是编译期常量，与 `__launch_bounds__` 共用同一常量，禁止入 TilingData |
| **不含 kernel 可自算的量** | 快除 magic/shift 在 Process 段算好经 UB 传递，不占 TilingData |
| **字段自证消费点** | 字段表逐字段标注 kernel 侧消费位置；无真实消费点的维度字段（可由 index/stride 推导的量，如 n/c）不得入 TilingData——删字段优于留占位（实测评审建议项） |
| scale/extent 类标量可入 tiling 或 VF 入参 | 形态选择按参数个数预算（≤ 28 直传） |
| 维度上限校验在 tiling 侧 | `OP_CHECK_IF(bs > UINT32_MAX ...)` 前置报错，不放 kernel |

host tiling 侧骨架（three_interpolate 实测）：

```cpp
constexpr int64_t PER_CORE_MIN = 1024;
uint64_t perCore = (totalElements + maxCoreNum - 1) / maxCoreNum;   // Step 1 均分
if (perCore < PER_CORE_MIN) { perCore = PER_CORE_MIN; }             // Step 1' 下限保护
perCore = ((perCore + ALIGN - 1) / ALIGN) * ALIGN;                  // 对齐（编组粒度）
int32_t needCoreNum = (totalElements + perCore - 1) / perCore;      // Step 2 反推
if (needCoreNum < 1) { needCoreNum = 1; }                            // 空输出下限 1 核
context->SetBlockDim(needCoreNum);
// 正向无输出初始化需求；grad 才需要 SetScheduleMode(1)（SyncAll 前提）
```

## 3 32/64 位索引双路径

`INDEX_SIZE_T` 作为模板参数双实例化：`uint32_t`（totalElements ≤ INT32_MAX，循环变量/商/余数 32 位）与 `uint64_t`（超限路径）。入口一次比较分派，kernel 模板复用。**批次基址偏移（`bIdx × ms × cs`）即使 32 位路径也以 `uint64_t` 承载**——32 位下标不保证 32 位基址（实测修复）。

## 4 TilingKey 编码规则

- **只编码场景模式，不编码 dtype**：dtype 由 `asc_vf_call` 模板参数 + `DTYPE_` 宏编译期多实例化（three_interpolate：`DTYPE_FEATURES (float/half) × DTYPE_IDX (int32_t/int64_t)` 四组合；schMode 单模式 DEFAULT）
- 场景维度举例：对齐模式（align_corners T/F）、method 分档（nearest/bilinear/bilinear_v2）、grad 正反向
- **实例数预算教训**（crop_and_resize 实测）：4 维度 TilingKey（24 组合）× 12 dtype = 288 kernel 实例过大——精简为 method-only（3 × 11 = 33 实例）后编译时间与产物可控，主优化维度保留模板特化
- 例外：线程数随场景分档时（512 / 2048）该维度可显式进 TilingKey

## 5 infershape / op 定义侧

- 输出 shape 逐轴推导（RegularResize：由 size/scale_factor 属性或值依赖张量定形；RoiSample：`crop_size` 值依赖）
- 值依赖输入三处声明缺一不可：op 定义 `ValueDepend` / infershape 注册 `InputsDataDependency` / tiling 注册 `TilingInputsDataDependency`（详见 [../../../common/value-depend-process.md](../../../common/value-depend-process.md)）
- 越界报错类校验（crop_size ≤ 0、维度越界）放 tiling 侧 `OP_CHECK_IF`；kernel 内分支只做忽略/外插
- **负向行为单一契约**：凡规格全局约束判非法的输入（如 `size[i] ≤ 0`、维度越界），处理方式必须**全链同向**——Interface「不支持场景」、InferShapeDtype、BranchRoute 不路由场景、HostTiling tiling 校验、spec 公式五处一致；范式方向＝tiling 侧 `OP_CHECK_IF` 报错（非静默回退）。同一设计内「约束 > 0」与「`max(·, 1)` 抬升」并存即接口契约自相矛盾（实测评审严重项）；确有竞品语义要求抬升/截断的，须以竞品逐字核对裁定后全链统一并补对拍说明
- **属性并存与值域同属负向单一契约**：互斥属性（如 size 类 `output_size` 与显式因子类 `scales`）同时生效、因子值域非法（如 ≤0）等**属性级非法**，与输入级非法同等对待——五处（Interface attr 约束/「不支持场景」、InferShapeDtype、BranchRoute 不路由场景、HostTiling tiling 校验、spec `attributes.machine_constraint`）同向落字，范式方向＝tiling 侧 `OP_CHECK_IF` 报错；spec 侧须**机读编码**（互斥约束、列表长度、值域 bounds），只写 semantics 不编码即缺陷。任两处「报错」与「某属性优先/静默取一/回退计算」并存即接口契约自相矛盾（实测插值 r2 严重项：InferShapeDtype「output_size 优先」vs HostTiling/BranchRoute「互斥报错」）。**默认值不构成「显式给出」**：规格须写明默认值与显式给定的区分口径（空列表/哨兵/None 等），实现无法区分时不得声明对应可选路径——否则在另一属性带非空默认值的情况下，该可选路径任何调用都不可达，分支/矩阵却声称覆盖它＝自相矛盾（实测插值 r2 严重项）
- **OpDef 片段是 Proto/def 生成的事实源**：Interface 的 OpDef 代码片段须与 DevView / HostTiling 的同名信息**字段级一致**——值依赖输入的 `.ValueDepend(...)` 链（三处声明的 op 定义侧）、`PrecisionReduceFlag` 等 AICore flag 集合、`ExtendCfgInfo("opFile.value")`（与 kernel 入口文件名 / DevView 文件树一致）；任一处缺失或不一致，下游生成即丢失值依赖声明或 kernel 文件名查找失配（实测评审严重项）
- 空输出（num_boxes=0 / N=0）不在 infershape 端短路：shape 自然带 0 维，tiling 侧下限 1 核、kernel 侧短路返回（`totalElements == 0 → return`）
