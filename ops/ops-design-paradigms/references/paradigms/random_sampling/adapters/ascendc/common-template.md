<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling 公共模板接入契约

> **层级**：AscendC 语言适配层。语言中立模型见 [../../generic/index.md](../../generic/index.md)。
>
> 本范式与纯方法论范式的最大差异：存在**活代码公共底座** `random_common`（开源算子仓 [cann/ops-math](https://gitcode.com/cann/ops-math) 的 `random/random_common/`，host tiling 模板 + device kernel 模板；获取方式见 §1）。底座面向**路由 a 纯生成形态**（统一 TilingData 的 key/counter 字段族）；路由 b 复合型（双输出 / attr 透传 / 扁平 TilingData）套不进统一结构，走自管编排。实测代码量基准（路由 a）：kernel 73~100 行、tiling 89~152 行。

## 0 路线判定表（接入方式与场景路由对应）

| 场景路由（scene §3） | 接入方式 |
|------|------|
| (a) 纯随机生成 | **继承公共底座**：OpTilingConfig + 统一 TilingData + PhiloxSimtKernel / RandomKernelBaseOp |
| (b) 随机 + 逐元素复合 | **自管编排**：自管 TQue/TBuf、扁平 TilingData、attr 透传 + **复用原子件**（Skip / u 构造路径 / U32Conversion 常量池 / 落核决策） |
| (c) 多范式并集 | 主计算范式主导编排，随机段复用原子件 |

## 1 底座组件地图

> **底座获取（开源仓）**：`random_common` 位于 CANN 开源算子仓 **[cann/ops-math](https://gitcode.com/cann/ops-math)**（GitCode），路径 [`random/random_common/`](https://gitcode.com/cann/ops-math/tree/master/random/random_common)——下表"相对 `ops-math/random/random_common/`"即该仓内路径（本地 clone 目录名为 `ops-math` 时路径同构）。**无内部 math 仓检出时直接使用该开源仓**：`git clone https://gitcode.com/cann/ops-math.git` 或网页浏览。底座持续演进（统一 TilingData 字段会扩），接入前以目标分支实际代码为准核对，禁止凭本文件记忆的字段直接编码。内嵌样例包 [example/random_uniform_v2_package/](example/random_uniform_v2_package/) 构建同样依赖该底座（`DEPENDENCIES random_common`），阅读对照无需检出、独立构建需要。

| 组件 | 位置（相对 `ops-math/random/random_common/`） | 对应通用层概念 |
|------|------|------|
| tiling 基类 + 配置 | `op_host/arch35/random_tiling_arch35.{h,cpp}` | splitting（均分/块协议）、状态解析 |
| seed/shape 公共工具 | `op_host/arch35/random_tiling_base.h` | state-management（seed 语义、值依赖） |
| 统一 TilingData | `op_kernel/arch35/random_unified_tiling_data_arch35.h` | 切分与流字段 |
| device 公共基类/模板 | `op_kernel/arch35/random_kernel_base.h` | prng-model、distribution-transform |
| infershape 基类 | `op_host/random_infershape_base.{h,cpp}` | shape 值依赖推导 |
| graph 推导基类 | `op_graph/random_graph_infer_base.{h,cpp}` | 图模式 shape 推导 |

### 术语对照表（通用层规范名 ↔ 底座字段名 ↔ 自然用法）

| 通用层规范名（splitting） | 底座字段名（统一 TilingData） | 设计文档自然用法 |
|---|---|---|
| usedUnits | usedCoreNum | blockDim |
| normalPer | normalCoreProNum | blockFactor |
| tailPer | tailCoreProNum | —（尾核差量） |
| 单元缓冲量 | singleBufferSize | ubFactor（单次循环元素数，口径以设计文档公式节为准） |

对照仅助映射，**口径以各设计文档公式节为准**。

## 2 host 侧接入：OpTilingConfig 配置化

tiling 基类 `RandomTilingArch35` 固化 8 步流水（校验→平台→前置→FillTilingData(SIMD/SIMT 分叉)→TilingKey/Workspace→Unique→WriteBack→Dump）。算子只做两件事：

**(a) 填校验规则表**（dtype 集/shapeSize/维数/自定义检查）：

```cpp
config.inputCheckRules = {
    {0, {{ge::DT_INT32, ge::DT_INT64}, -1, {0, 1}, nullptr}},  // shape 张量
    {1, {{ge::DT_INT64}, 1, {}, nullptr}},                     // offset
};
config.outputCheckRules = {
    {0, {{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, -1, {0..8}, nullptr}}};
```

**(b) 实现 4 个字段回调**（缺省的用公共实现）：

| 回调 | 职责 | 公共实现 |
|------|------|---------|
| `getOutputSize` | 从 shape 张量读值推导 L（并校验输出规模一致） | `RandomUtils::GetAndCheckOutputSize<I,O,CHECK>` |
| `getKeyAndCounter` | seed/seed2 属性 → key[2]/counter[4]（**仅 C 族语义**：seed=0 且 seed2=0 走 `New64()` 时间熵；B/D 族 seed 为裸 key **禁用此公共实现**的时间熵分支） | `RandomUtils::GetKeyAndCounter<S1,S2>` |
| `getBufferNum` | 片上存活节点数（流缓冲 + 输出×缓冲份数） | 算子自填（含 dtype 字节数） |
| `getSeedAndOffset` | SIMT 路线的 seed/offset 张量读取 | 算子自填 |

**getBufferNum 的两型公式**：

| 场景 | 公式 |
|------|------|
| 纯生成（uniform_v2 型） | `bufNum = sizeof(uint32_t) + dtypeSize × BUFFER_NUM` |
| 随机+逐元素复合（rrelu_with_noise 型，R9） | RNG fp32 暂存（philoxBuf_，VECCALC TBuf）按 **K = sizeof(float)/dtypeSize 份折算**；UB 预算分母 = 数据面份数×BUFFER_NUM **+ K**；TBuf 不随 BUFFER_MODE 翻倍；32B 对齐（流生成接口 dstLocal 约束） |

漏算 K 份 → UB 溢出（编译期不报，运行时踩内存）。

**(c) 短路口径 TilingData 填充语义（R18，generic/splitting §4）**：计数字段按 N 清零/置 1；**平台派生量（ubFactor 等）无条件按公式填充**——设计文档与 UT 预言机同源，偏离须双侧同步声明。

**(d) 关键启动开关**：

| 字段 | 含义 | 默认 |
|------|------|------|
| `kernelMode` | `SIMD` / `SIMT` 二选一（决定 Fill 分叉与 TilingData 类型） | SIMD |
| `enableSplitBlocks` | 开启大张量块协议（TensorSliceState 递归二分 + 每块流预算） | false |
| `isNeedSyncAll` | 需要全核栅栏（回写时序）→ `SetScheduleMode` | false |
| `DcacheSize` | kernel 内含 SIMT VF 调用时扣减片上容量（`SetLocalMemorySize`） | 0（无 simtvf） |
| `coreAlignSize` / `unrollFactor` | 单元对齐粒度 a / 展开因子 U | 4 / 4 |

**(e) 注册**（含值依赖）：

```cpp
IMPL_OP_OPTILING(OpName)
    .Tiling(TilingOpName)
    .TilingParse<RandomOperatorCompileInfo>(TilingPrepare4OpName)
    .TilingInputsDataDependency({0});   // shape 张量值依赖，索引对应输入位
```

## 3 device 侧接入：TransformFunc/Launcher 契约（SIMT 路线）

`PhiloxSimtKernelContinuous/Discontinuous` 模板承担流的推进与线程映射，算子只提供**输出变换**：

```cpp
template <typename T>
struct MyTransform {
    float param_;
    __simt_callee__ __aicore__ inline void operator()(
        __gm__ volatile T* outputGm, uint64_t li, const uint32_t* results,
        uint32_t iStep, [[maybe_unused]] uint32_t randPerOutput = 1)
    {
        float u = results[iStep] * RAND_2POW32_INV + RAND_2POW32_INV_HALF;  // 乘法+半步口径 (0,1)——SIMT 模板族实存；须与流基准同路径（R7，位重解释为 [0,1) 的另一口径）
        outputGm[li] = static_cast<T>(变换(u));                             // distribution-transform §1
    }
};
```

| 契约项 | 语义 |
|--------|------|
| `li` | 当前写入的线性索引（布局由模板管理，变换不感知） |
| `results[4]` | 本轮 Philox 产出（k=4 个 uint32） |
| `iStep` | 本元素消费的第几个随机数（0..U-1） |
| `randPerOutput` | 每输出消耗数（U/4，c>1 时 >1） |
| **禁止** | 变换内做流推进/线程映射——那是模板职责 |

块协议下再包一层 Launcher（每块一次模板调用，块参数由 tiling 下发）：

```cpp
struct MyLauncher {
    __aicore__ inline void operator()(const ExecutionPolicyKernel& policy,
        int64_t gmOffset, int64_t kernelOffset, int64_t numel, int64_t grid, int64_t totalThreads)
    {   // gmPtr = base + gmOffset；kernelOffset = 本块流起点（offset_b）
        AscendC::Simt::VF_CALL<PhiloxSimtKernelDiscontinuous<T, MyTransform<T>>>(Dim3(512), ...);
    }
};
// kernel Process 内一行：ProcessWithSplitBlocks(tilingData, launcher);
```

## 4 SIMD 路线接入

继承 `RandomKernelBaseOp`（`VarsInit/Skip/GenRandomSIMD`），自管 TQue/TBuf 与搬出（形态见 [simd-path.md](simd-path.md)）。

## 5 新算子接入 checklist

1. [ ] 场景路由判定（scene §3）→ 接入方式（§0 路线判定表）；算子族归类 → 消耗数 c / 变长预留 R
2. [ ] 接口形态（stateless/带状态）→ 回写开关 `isNeedSyncAll`
3. [ ] kernelMode 选型（见 [index.md](index.md) 路线选型表）
4. [ ] OpTilingConfig：规则表 + 4 回调 + 启动开关
5. [ ] TransformFunc / 或 SIMD 继承类
6. [ ] 值依赖三处声明（def/infershape/tiling，见 [../../../common/value-depend-process.md](../../../common/value-depend-process.md)）
7. [ ] 边界用例对齐 [../../generic/verification.md](../../generic/verification.md) §3
