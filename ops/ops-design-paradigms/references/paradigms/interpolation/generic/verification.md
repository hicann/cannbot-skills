<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Interpolation 范式验证方法

> **层级**：通用知识层（generic/），语言中立。
>
> 覆盖 golden 口径、精度分档与边界用例；坐标映射逐位语义与累加口径的验证义务同步列出。

## 1 golden 基准选择

| 输出语义 | golden 基准 | 理由 |
|---------|-----------|------|
| 正向插值（确定性） | 竞品 CPU/参考实现（**中间量位宽按源码逐字核对**） | 语义确定可比对；golden 的 floor/累加位宽与 kernel fp32 域的**舍入路径差异是口径的一部分**，见 §2 |
| 正向插值（极值值域） | 同上 + 声明逐位对齐点 | ±FLT_MAX/异号相消点对舍入序极敏感（[weight-model.md](weight-model.md) §4），容差按"与 golden 同序模拟"验收而非放宽 |
| grad 散射累积（原子） | 竞品 **GPU** 实现（atomicAdd） | 双方同为非确定实现，容差内可比；CPU 顺序实现舍入路径不同，逐位比对误报 |
| grad 散射累积（确定性） | NumPy / CPU 参考实现 | 确定语义（分桶预归约路径），可逐位比对 |

golden 脚本的坐标映射公式、对齐模式、项序必须与竞品逐字核对（half_pixel 的 max 时机、floor 位宽、显式缩放因子的方向与 recompute 换算是高频偏差源，[coordinate-mapping.md](coordinate-mapping.md) §2）。

golden 累加位宽不是通识默认值：CPU 实现的累加位宽由其累加类型特化决定（float 输入多为 fp32 累加、half/bf16 输入升 fp32、仅 double 路径为 fp64），「常为 fp64」类通识不得替代源码核对。golden 口径（中间量位宽、项序）必须**全文档唯一**——Validation、Kernel、分支设计、spec 的表述逐字一致，两说并存即口径缺陷（实测评审一般项）。

## 2 精度标准（按 dtype 分档）

| dtype | 口径 |
|-------|------|
| int32 / int64（最近邻索引类输出） | 逐位一致 |
| float16 / bfloat16 | 最大相对误差 < 1e-3（升 fp32 累加后通常远优于此） |
| float32 | 最大相对误差 < 1e-6（常规值域）；极值值域用 mare/mere 双指标并声明残余带（[weight-model.md](weight-model.md) §4.6） |
| fp64 golden 模拟路径 | mere < 1e-6 / mare < 1e-3 量级（实测 df32 收敛值 4.4e-14 / 7.1e-8），按算子容差核定 |

**指标口径**：`mere`（最大相对误差，分母 `|g|`）与 `mare`（分母 `|g| + eps`）须同时报告——golden 近零元素（相消/外插）只看 mare。

**分档刚性**：本表分档是默认验收标准——放宽（如 bf16 用 1e-2）须附量化论证（误差源核算＋实测误差分布支撑），无论证的放宽即缺陷（实测评审一般项）；mere/mare 双报是硬性义务，不得以「逐像素线性、无跨元素规约」为由省略——混合符号输入的近零相消输出在本范式真实存在。验证方案的容差表须逐 dtype 落 mere/mare 双报列（或双报判定行）；仅 rtol/atol＋`np.allclose` 单口径即缺陷（实测插值 r2 复现项）。验证方案的容差表须逐 dtype 映射到本表分档，或给出偏离论证。

## 3 边界用例（必覆盖）

命中算子族的每一行＝端到端矩阵的**必测行**：须在验证方案的端到端矩阵中有对应用例，并在矩阵旁附**逐行存在性自检块**——每条必测行独占一行：命中 → 标注对应矩阵行号；不适用 → 显式标注原因与归属（族未命中 / 模式未启用）。自检块缺失、或任一必测行既无行号又无原因标注，即验证方案缺陷（实测插值 r2 严重项：in_extent=1、out_extent=1（对齐模式缺一）、权重恰零×inf 构造、±FLT_MAX、大 shape>INT32_MAX 核算、AutoContiguous 行整体缺失且无自检块兜底）；分支最小用例集不豁免矩阵行。

| 用例 | 验证点 | 对应算子族 |
|------|--------|-----------|
| **两种对齐模式全覆盖**（align_corners T/F 分别建用例） | scale 公式与 max 时机分派 | 全族 |
| **out_extent = 1**（OH/OW=1；两种对齐模式各 ≥1 行） | 退化分支 | RegularResize / RoiSample |
| **in_extent = 1**（H=1 与 W=1 至少各 1 行；邻居坍缩单点） | lerp 恒 0、i0=i1 clamp | RegularResize |
| **最近邻模式**（method=nearest） | 取整后直取、无权重段 | RoiSample |
| **源坐标越界**（外插值替代 / clamp 收边 / 贡献丢弃） | 越界策略与竞品一致 | RoiSample / RegularResize / grad |
| **idx 越界 / 负值**（契约保证类 + 防御 clamp 类） | OOB 防御或契约声明 | IndexedInterpolate |
| **权重不归一**（和 ≠ 1） | 不校验、直乘直加 | IndexedInterpolate |
| **空输出**（num_boxes=0 / N=0 / 空 grid） | 空载核短路、不崩溃 | 全族 |
| **值依赖尺寸**（crop_size 等读值定形） | 值依赖三处声明、输出 shape 正确 | RoiSample |
| **NaN / ±Inf 传播**（含权重恰零 × inf 样本分支——构造须保证权重精确为 0 的邻居恰为非有限样本） | IEEE 754 传播逐位核对 | 全族 |
| **±FLT_MAX 值域**（极值全谱） | 溢出防护与吸收截断生效 | RegularResize（与是否启用补偿路线无关，必测） |
| **大 shape**（totalElements > INT32_MAX，须附元素数核算；同时触发 64 位索引分派与偏移提升点） | 宽整型提升点、32/64 位双路径 | 全族 |
| **非连续输入 / 分片坐标系**（srcStart ≠ 0；AutoContiguous 承接路径须 ≥1 例 E2E，不因声明豁免） | 重基次序、承接契约 | RegularResize |

大 shape 行的判据是**元素总数真正越界**：仅 H/W 数千级而 totalElements ≤ INT32_MAX 不满足本行（实测 1×256×2160×3840 = 2,123,366,400 < 2^31−1 = 2,147,483,647，未触发 64 位路径；越界示例 (1,260,2160,3840) = 2,156,544,000）。

## 4 性能验证要点

- **输出规模主导**：正向耗时 ∝ totalOut（邻居 gather 次数 = totalOut × k）；性能用例按输出规模分档（小/中/大 output shape 各含至少一例）
- **离散度敏感**：RoiSample 的框分布（聚簇框重复采样同区域 → GM 命中率高）与均匀框分别测；IndexedInterpolate 的 idx 分布同理
- 对照点：坐标映射含快除与不含（magic/shift 生效性）、grad 零初始化两段 launch 与单核小任务的少核路径
- 常量开销：totalOut 极小时启动开销主导，验证 needCoreNum 收缩行为（PER_CORE_MIN 抬升）
- **关键 shape 的场景档覆盖**：性能关键 shape 须按 TilingKey 场景档分组覆盖——每个 tilingKey（如 align_corners T/F）≥1 条关键 shape，全部落在单一 key 即覆盖缺陷（实测评审建议项）

## 5 坐标映射逐位验证义务

- 浮点同序声明（[coordinate-mapping.md](coordinate-mapping.md) §5）的每个阻断点（FMA、max 时机、floor 位宽）在验证报告中逐项核对
- 对齐模式边界（out_idx=0、out_idx=out_extent-1、max(·,0) 恰好触发点）建逐位比对用例
- 索引提升点（`src × stride` 乘法前升宽整型）用大 shape 用例回归

## 6 验证输出清单

- [ ] golden 口径（中间量位宽经竞品源码逐字核对、项序）与精度分档已声明，且全文档唯一
- [ ] mere/mare 双指标与近零元素处理口径
- [ ] 边界用例表全覆盖（按命中的算子族与对齐模式）
- [ ] 端到端矩阵对 §3 命中行的逐行存在性自检（含大 shape 元素数核算）
- [ ] 非有限值（NaN/Inf/权重零×inf）分支用例齐备
- [ ] 极值值域路线（若启用补偿累加）的残余带声明
- [ ] 值依赖三处声明核对（crop_size 类）
- [ ] 适配层 capability-mapping §3 的每个待验证项在验证方案中有对应回归行（逐项标注触发时机：首轮验证 / 编译器升级 / SoC 迁移），缺任一行即验证方案不完整
- [ ] 全部「待补 / 待标定」占位行逐行登记回归义务（由谁回填、何时回填）与触发时机；**BranchCatalog.md 已产出的情况下，矩阵与性能表的 tilingKey 列不得再留「待补」**（机器可核对：BranchCatalog 存在 ⇒ 全表无「待补」字样；实测插值 r2 一般项——key 语义已定却整体留待补）
