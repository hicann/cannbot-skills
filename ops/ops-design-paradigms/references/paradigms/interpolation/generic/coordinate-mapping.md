<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Interpolation 范式坐标映射模型

> **层级**：通用知识层（generic/），语言中立。
>
> 核心问题：输出位置如何映射为输入域浮点源坐标，并分解为邻居索引与权重；浮点序列如何与 golden 严格同序。

## 1 输出驱动映射

正向遍历以**输出坐标系**为驱动：每个输出位置独立完成"出坐标 → 映射 → 采样"，输出元素间无依赖（scene §4 五段式的段 1–2）。反向以 grad_output 为驱动坐标系，映射同一套公式，方向换成散射。

## 2 对齐模式语义表

scale 与源坐标的计算约定（按轴独立，同一算子内多轴须同模式）：

| 模式 | scale | 源坐标 src | 说明 |
|------|-------|-----------|------|
| **align_corners** | `(in_extent - 1) / (out_extent - 1)`；`out_extent = 1` 时为 0 | `src = scale × out_idx` | 首尾像素对齐；TF/PyTorch 双端契约 |
| **half_pixel**（align_corners=False） | `in_extent / out_extent` | `src = scale × (out_idx + 0.5) - 0.5`，再 `max(src, 0)` | PyTorch 默认；max(·,0) 在全局坐标系（减 srcStart 之前）施加 |
| **TF crop 仿射** | `(y2 - y1) × (in_extent - 1) / (crop_extent - 1)` | `src = y1 × (in_extent - 1) + out_idx × scale`；`crop_extent = 1` 时取框中心 `0.5 × (y1 + y2) × (in_extent - 1)` | crop_and_resize；框坐标为归一化值，值依赖读入 |
| **显式索引** | —（无 scale） | `src` 由 idx 张量直接给出 | IndexedInterpolate；RoiSample 的 grid_sample 同理 |

- **scale 计算位置**：host 侧预计算为标量传入（不进 TilingData 数值循环）；fp32 承载，`(in-1)/(out-1)` 的除法舍入与 golden 同源
- **退化分支**：`out_extent = 1` 时 align_corners scale 为 0（全部输出映射到 src=0）；`in_extent = 1` 时邻居坍缩为单点（lerp 恒 0），走最近邻退化路径
- **srcStart/dstStart 重基**：分片场景（如 sync 系列的全局/分片坐标系）先升全局坐标再重基，`max(·,0)` 施加时机在重基**之前**——顺序错则边界列语义偏移
- **显式缩放因子属性的方向语义（scale_factor / scales_h / scales_w 类）**：竞品的显式因子是**放大倍数**（`out_extent = round(in_extent × factor)`，如 ATen upsample 系），**不是坐标映射乘子**——映射 scale 仍取实际定形后的 `in_extent / out_extent`（数值上为因子的倒数，且以定形后的维度为准重算，不直用因子原值）。把因子原值直接当 src 乘子，上采样会被映射成越界边带复制（实测评审阻塞项）。核对义务：
  - spec `attributes` 语义须写明方向（放大倍数）与换算关系；HostTiling / TilingData / 分支文档同一口径
  - 因子与 size 并存时按竞品**入口层**逐字核对（如 `torch.nn.functional.interpolate` 同时给定 `size` 与 `scale_factor` → 报错）；本项目范式方向＝**同时显式给出即属性非法（tiling 侧报错，见 kernel-entrance.md「负向行为单一契约」）**，规格须显式写明该裁定并机读编码互斥约束——「output_size 优先」「静默取一」类裁定与报错并存即缺陷（实测插值 r2 严重项）；默认值不构成「显式给出」，区分口径须落字（同上）
  - 因子缺省（未显式给出）→ 按 size 路径定形；因子非正值（≤0）→ 竞品两层语义不同（入口层 `F.interpolate` 对 `scale_factor ≤ 0` 报错；ATen kernel 层对无效 scales 存在忽略回退），规格须**锚定入口层**单一裁定并写明依据与核对来源——禁止「允许 ≥0」「仅 <0 报错」「≤0 回退 in/out 计算」三种口径在任何两处并存（实测插值 r2 建议项）
  - golden 脚本与 kernel 必须使用同一方向口径——kernel 用因子原值、golden 用倒数（或 in/out）的双口径是必失败组合

## 3 邻居分解与 clamp 防御

```
f0 = floor(src)                    # 向负无穷取整（fp32 域）
lerp = min(src, in_extent - 1) - f0   # lerp 用未 clamp 的 f0（与 golden 同序）
i0 = clamp(f0, 0, in_extent - 1)   # 防御性双向 clamp（src 可为负或超上界）
i1 = min(i0 + 1, in_extent - 1)    # i1 由 clamp 后的 i0 推导，不由 f0+1
```

- **clamp 是防御而非语义**：half_pixel 的 `max(·,0)` 已保证 src ≥ 0，但浮点边界（src = in_extent-1 的末位舍入）仍可能产生 `f0 = in_extent`；双向 clamp 兜底
- **lerp 的次序敏感**：lerp 用 `min(src, extent-1)` 减 `f0`，若先把 src clamp 再减 f0 会改变末位——与 golden 计算序逐位对齐
- **索引类型**：邻居索引中间量用宽整型（≥ int32，大 shape 升 int64）承载；`src × stride` 的地址偏移乘法前先提升类型（offset 溢出是本范式高频事故）
- **IndexedInterpolate 的索引防御**：读入的 idx 值先升宽整型再 clamp 到 `[0, extent-1]`（int64 输入含负值/大值时防 OOB；契约保证合法的算子也建议保留防御，除非语义要求报错——见 §4）

## 4 越界策略

| 策略 | 语义 | 实例 | 实现位置 |
|------|------|------|---------|
| **clamp 收边** | 越界源坐标收到边界像素，邻居权重照常 | resize 系（half_pixel 的 max + i0/i1 clamp） | kernel 坐标映射段 |
| **外插值替代** | 源坐标出输入域 → 输出置 `extrapolation_value` | crop_and_resize | kernel 段 2 判定后短路 |
| **贡献丢弃** | 邻居越界 → 该邻居项跳过（权重不归一） | upsample grad 的条件累加、roi_align 的越界 bin | kernel 段 4 |
| **契约保证** | 调用方保证索引合法，kernel 不校验 | three_interpolate（PointNet++ 上游保证） | 规格声明 + 文档标注 |

选择依据是竞品语义；契约保证类须在规格中核对竞品文档并标注风险（防御性 clamp 与契约可并存，代价是每邻居两次比较）。

## 5 浮点同序（索引数学禁融合重排）

坐标映射的浮点序列是**逐位语义**，禁止任何改变舍入路径的重排：

1. **禁 FMA 融合**：`(g + 0.5) × scale - 0.5` 中，若编译器将乘加融合为 FMA，乘积不落 fp32 寄存器，lerp 末位改变——golden 恰为 0 时相对误差爆炸（实测 mare 8e-3 → 修为 volatile 中间积强制先舍入后收敛）。依赖 `CAP_FMA_CONTROL`
2. **运算顺序与 golden 同序**：`g × scale` 与 `- srcStart` 分两步；升全局坐标（fp32 相加）与缩放（fp32 相乘）各自独立舍入
3. **floor 在 fp32 域**：golden（fp64 中间量）下 floor 的输入先舍入到 fp32 再取整，才与 kernel 逐位一致——**取整前的精度位宽是口径的一部分，须在验证方案中声明**
4. **值域**：src 计算的中间量在 ±FLT_MAX 输入下不溢出（scale ≤ 1 时天然满足；TF crop 的框坐标乘 `(extent-1)` 须核算）

## 6 高阶核（bicubic）

upsample_bicubic2d 类为 4 邻居 × 4 邻居：

- 每轴 4 邻居：`i = floor(src) + {-1, 0, 1, 2}`，权重由 cubic convolution 核 `A(x)`（分段，|x| ≤ 2）按 lerp 求值；轴间分离（先 H 后 W）
- 边界 clamp 同 §3，但 4 邻居的上侧越界（floor(src)-1 < 0）须 clamp 到 0，权重不重算
- 权重函数含 `|x|` 与分段常数（A=-0.75 系），项数 16 项——累加精度按 [weight-model.md](weight-model.md) §3 判定树从严选择
- 其余（对齐模式、同序规则）与双线性一致

## 7 设计输出清单

- [ ] 对齐模式判定（逐轴核对竞品默认值；半像素的 max 时机）
- [ ] scale 预计算位置与 dtype（host 标量入参）
- [ ] 退化分支（out_extent=1 / in_extent=1 / 最近邻模式）
- [ ] 越界策略（clamp / 外插值 / 贡献丢弃 / 契约保证）及实现位置
- [ ] 邻居索引中间量类型与偏移乘法的提升点
- [ ] 浮点同序声明（floor 位宽 / FMA 阻断点清单）
- [ ] 显式缩放因子属性的方向语义（放大倍数，映射乘子取倒数）与 recompute 换算、缺省回退与非正值报错口径核对
- [ ] 互斥属性并存裁定（同时显式给出→报错；默认值不算显式给出，区分口径落字）与因子值域单一口径，五处同向＋spec `machine_constraint` 机读编码
