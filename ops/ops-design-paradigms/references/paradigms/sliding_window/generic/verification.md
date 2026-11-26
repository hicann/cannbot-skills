<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# SlidingWindow 范式验证方法

> **层级**：通用知识层（generic/），语言中立。
>
> 覆盖 golden 口径、精度分档与正反向边界用例；窗口数学逐位语义与梯度守恒的验证义务同步列出。

## 1 golden 基准选择

| 输出语义 | golden 基准 | 理由 |
|---------|-----------|------|
| 正向聚合（max 及其余池化聚合类） | 竞品 CPU/参考实现 | 确定语义（tie-break 契约一致时逐位可比） |
| 正向聚合（avg） | 竞品 CPU/参考实现 | 累加项序敏感度低于插值（窗口 k² 项、fp32 累加常规值域收敛）；golden 中间量位宽仍须声明 |
| argmax mask | 竞品实现 | **tie-break 规则逐位核对**（首遇/末遇/任意三态，与 [window-math.md](window-math.md) §3 契约一致才算过） |
| 反向 A/B 分支 | 竞品实现 | 零原子确定语义，可逐位比对 |
| 反向 C 分支（原子） | 竞品 **GPU** 实现（atomicAdd） | 双方同为非确定实现，容差内可比（同 Interpolation grad 口径） |

## 2 精度标准（按 dtype 分档）

| dtype | 口径 |
|-------|------|
| int32 / int64（argmax mask） | 逐位一致 |
| float16 / bfloat16 | 最大相对误差 < 1e-3（升 fp32 累加） |
| float32 | 最大相对误差 < 1e-6；avg 除子口径一致时通常远优 |
| max 全空窗输出 | −Inf 与竞品逐位一致（**不得转 0**） |

## 3 边界用例（必测组合矩阵）

> 每行必须落成具体用例：「构造参数模板」列给出可执行参数（shape、k/s/p/d 与数据置法）。生成侧产物 Validation 的端到端矩阵须把本表作为**落位契约**逐行消费，判据如下（逐条机器可核对；max_pool_with_argmax_v3 两轮生成评审 D2/D3→R2-1/R2-2 实测缺口）：
>
> 1. **原样复制**：每条必测行须把「构造参数模板」原样复制进端到端矩阵——shape 与 k/s/p/d 逐字一致；可增补数据置法 / 期望来源列、可追加额外行，但必测行自身不得改参、不得省略；
> 2. **逐行自检清单**：Validation 须附「必测行 × 端到端矩阵行号」自检清单，每条必测行一行，登记落位行号（或缺失原因与归属），供验收逐行核对；
> 3. **禁止同域替代（机器判据）**：对矩阵逐行扫描，须同时存在——`p>0 ∧ d>1` 的行（「负起点 × dilation>1」；以 d=1 或 p=0 的行顶替＝未落位）与按 [window-math.md](window-math.md) §2 公式核算为窗口级空窗（裁剪后 `start′ ≥ end′`）的合法输入行（以 `k>输入` 报错行或空张量行顶替＝未落位）；
> 4. **按 TilingKey 分组覆盖计数**：矩阵须自带机器可读计数表（验收化要求见 §6 清单）。

| 用例 | 构造参数模板 | 验证点 | 方向 |
|------|--------------|--------|------|
| **空窗**（裁剪后 start′ ≥ end′；`pads ≤ k/2` 约束**不排除**空窗，可经 dilation 网格错位构造——[window-math.md](window-math.md) §2 反例） | `(1,1,2,2) k(2,2) s(1,1) p(1,1) d(3,3)` | max: −Inf 直写；avg: 0 / 除子口径；mask 初值口径与竞品核对；不越界不崩溃 | 正向 |
| **全 pad 窗口对**（kernel > 输入） | `(1,3,1,1) k(3,3) s(1,1) p(1,1) d(1,1)` | 窗口退化、countIncludePad 除子差异 | 正向 |
| **负起点 × dilation>1**（pad 使 start < 0 且 d > 1——while 推进与 clamp 唯一可观察分歧域） | `(1,1,7,13) k(3,3) s(2,2) p(1,1) d(2,2)` | while 推进后起点仍在格点上（clamp 错位引入窗外元素）；golden 遍历序与设计逐位对拍 | 正向 |
| **负起点（dilation=1 对照）** | `(1,1,5,5) k(3,3) s(2,2) p(1,1) d(1,1)` | while 推进与 max(·,0) 等价（对照行，不替代上一行） | 正向 |
| **dilation > 1 覆盖** | `(1,1,8,8) k(3,3) s(1,1) p(0,0) d(2,2)` | 枚举格点不重不漏；窗口面积按格点数非跨度 | 正/反向 |
| **NaN 传播**（窗口含 NaN） | `(1,1,5,5) k(2,2) s(2,2) p(0,0) d(1,1)`，数据埋 NaN（窗口内/跨窗口边界） | max: NaN 成为 winner 且 mask 指向 NaN 位；反向路由到 NaN 位 | 正/反向 |
| **平局 tie-break**（窗口内重复最大值） | `(1,1,5,5) k(3,3) s(1,1) p(1,1) d(1,1)`，数据置重复最大值 | mask 与竞品同口径（首遇）；反向梯度守恒（和 = grad_output 和） | 正/反向 |
| **除子三分档**（divisorOverride=0 / countIncludePad=1/0） | `(1,3,4,4) k(3,3) s(2,2) p(1,1) d(1,1)`，三属性档各一行 | 正反向除子**同值**（不一致即精度事故） | 反向 |
| **反窗口边界**（输入首/末行、首/末列元素） | `(1,1,6,6) k(2,2) s(2,2) p(1,1) d(1,1)`，取边界元素逐点核对 | pStart/pEnd 区间恰覆盖全部窗口、无漏路由无重路由 | 反向 |
| **stride > kernel**（窗口不重叠） | `(1,1,5,5) k(2,2) s(3,3) p(0,0) d(1,1)` | 反窗口区间退化为 ≤1 个窗口 | 反向 |
| **窗口全重叠**（stride=1） | `(1,1,4,4) k(3,3) s(1,1) p(1,1) d(1,1)` | 反窗口区间最大；梯度守恒 | 反向 |
| **ceil/floor 模式** | `(1,1,5,5) k(2,2) s(2,2) p(0,0) d(1,1)`，ceil_mode=True/False 各一行 | 输出尺寸公式与竞品一致；ceil 模式末行空窗处理 | 正向 |
| **adaptive 换算**（如适用） | `(1,1,7,7) → out(3,3)` | start/end 整数域公式与竞品一致 | 正向 |
| **大 shape**（坐标乘加超 int32） | `(1,1024,2048,2048) fp16 k(2,2) s(1,1) p(0,0) d(1,1)`（N·C·H·W = 2^32，超物理预算时按黑盒 oversized 侧表规则登记） | 宽整型提升点 | 正/反向 |
| **空输入 / 空输出**（size=0） | `(1,1,0,0)`（空输入）；`(0,3,4,4)`（空 batch → 空输出） | 1 核短路；分支 C 零初始化仍执行 | 正/反向 |

> **「空输入 / 空输出」行的规格豁免路径**：规格显式声明非退化域（每维 ≥1）**并**在校验表落实维正性检查（命名 error code，空维用例入错误路径矩阵）时，允许免测本行——须在逐行自检清单登记「豁免＋依据（校验项与 error code）」；仅规格声明而未落实校验＝未豁免（max_pool_with_argmax_v3 第二轮评审 R2-6 实测形态：声明「不支持空 tensor」却无任何校验落点）。

## 4 梯度守恒与路由验证

- **分支 A**：`sum(grad_input) == sum(grad_output)`（均匀贡献，逐点容差内）；除子口径核对（正反向同值）
- **分支 B**：每个 grad_output 元素**恰好路由一次**——mask 唯一性统计（`sort(mask) == 去重序列` 反向验证散点不重复）；反窗口枚举形态下 mask 命中次数分布采样核对
- **分支 C**：重复路由容差（原子非确定）；要求确定性时分桶预归约路径单测
- mask 索引口径：正反向同 dtype（int32/int64）同线性坐标，跨口径比对必失败

## 5 性能验证要点

- **窗口面积敏感**：耗时 ∝ outSize × 窗口格点数（正向）与 inSize × 反窗口重叠度（反向）；用例按 (k, s) 组合分档
- **dilation 访存离散度**：dilation > 1 时窗口内读跨 stride，GM 命中率下降——与 dilation=1 对照测
- **分支 B 无效扫描**：mask 命中率低的用例（大窗口少重叠）单列；线程档位（1024）收益验证
- 对照点：快除 vs 直除的坐标分解开销（正反向各一档）；分支 C 两段 launch 小任务少核路径

## 6 验证输出清单

- [ ] golden 口径（tie-break 契约 / 原子容差）与精度分档已声明
- [ ] 边界用例表全覆盖（按方向 × 分支；**分支 = 方向分支（正/反向）× TilingKey/模板分支（布局、索引 dtype 等 key 编码维度）**——仅方向行不构成全覆盖）
- [ ] §3 落位契约逐条通过：必测行原样复制进端到端矩阵、「必测行 × 落位行号」逐行自检清单已附、同域替代机器判据（`p>0 ∧ d>1` 行存在性、窗口级空窗行存在性）逐条扫描通过（第二轮评审 R2-1/R2-2 实测缺口）
- [ ] 验证矩阵自带**机器可读**的按 TilingKey 分组覆盖计数表：固定列 `TilingKey | 覆盖行数 | 用例行号清单`，产物 TilingKey 设计（组合表）登记的每个 key 各占一行（缺 key 行＝未通过）；每个 key ≥1 行，0 行的 key 不得省略、须在行内显式标注原因与用例归属；矩阵 tilingKey 列不得以「待补 / 待回填」占位收尾——key 映射以产物自身的 TilingKey 组合表为唯一事实源，设计期内即可回填。不得声明「全覆盖」而实表存在零用例 key（max_pool_with_argmax_v3 两轮评审 D4→R2-3 实测：28 行 tilingKey 全「待补」、无计数表、回填承诺无属主）
- [ ] 「空窗不发生 / 每窗 ≥1 有效采样」类保证均凭 [window-math.md](window-math.md) §2 空窗核算成文（核算过程随附、范式反例代入自检通过）——由输入域约束推定不可达＝未通过（第二轮评审 R2-2 实测：≥8 处虚假保证成文）
- [ ] 梯度守恒 / mask 唯一性验证义务登记
- [ ] 除子正反向一致性核对通过
- [ ] NaN / 平局 / 空窗 / dilation 用例齐备
- [ ] "待验证"项已列入验证义务
