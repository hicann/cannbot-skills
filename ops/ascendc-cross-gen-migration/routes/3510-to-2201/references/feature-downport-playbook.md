# 能力下沉快速路径（feature down-port）

> 适用：A5 已实现某语义，A2 只是被**门禁/工程限制**挡住，而不是算法缺失。
> 本文件只给方法与判断，具体实验数据、环境、真实路径见文末「案例索引」。

## 何时走这条路

- A5 上有实现、A2 编译能过（或只差少量接线），但运行报“参数不支持 / 仅 950 支持”；
- 同一参数在 host tiling 与 aclnn/metadata 两处被校验；
- 目标是把既有语义“放开”，不是重写计算主体。

若命中 RegBase/SIMT/A5 专有通路，仍按 SKILL.md 的 D2/D3 主流程执行。

## 四处门禁（缺一不可）

| # | 位置 | 典型症状 | 处理 |
|---|---|---|---|
| 1 | `op_host/<op>_tiling.cpp` | `maskType ... but got 0` / `residual ... unsupported` | 放开目标 dtype/quant/layout 组合，保留其他限制 |
| 2 | `<op>_metadata/op_host/op_api/aclnn_*.cpp` | 主算子放开了，metadata 仍报同样错误 | 同步放开，注意 `isArch35 && quant==0` 类条件 |
| 3 | `op_kernel/*_tilingkey.h` + 入口 `*.cpp` | 新组合没有 key，运行落不到新分支 | 按既有 bitfield 加 key，旧 key 不动；入口加 `TILING_KEY_IS` 与实例化 |
| 4 | `op_kernel/arch22/*_kernel*.h` | 编译过、结果错（边界/尾块） | 照 arch35 的 `if constexpr` 改可见范围、尾块、计数上限 |

**边界**：只动 arch22、tilingKey、两处 host 门禁；arch35 只读（接口签名必须同步时改一行）。

## 验证下限

1. 干净构建 + 安装到独立目录并 source vendor 环境；
2. 可见用例（回归路径）；
3. 至少 1 个自造边界用例（可见用例覆盖不到的新语义）；
4. 记录门禁「原条件 → 放开条件」、改动文件、失败根因。

## 常见错法

- 只改 tiling，不改 metadata → 仍被拒；
- 只加 key 不改 kernel → 走旧逻辑，精度错；
- tilingKey 手写数值与 host 计算不一致；
- 尾块/边界重复计算；
- 顺手重构 arch35，扩大回归面。

## 预算

门禁型 ≤ 120 次工具调用 / ≤ 20 min；内核扩展型 ≤ 200 次 / ≤ 40 min。超出则停止扩大调研，先交付最小可验证改动；UT/文档/性能仅在用户要求时补。

## 案例索引

| 案例 | 场景 | 数据与环境 |
|---|---|---|
| [GBSA 非因果 / 隐式残块](../experiments/gbsa.md) | `mask_type=0` + `residual_block_mode=1` 下沉 A2 | 四处门禁、tilingKey 位域、精度与 A/B 成本 |
| [SFA LSE + PA_BSND](../experiments/sfa.md) | `return_softmax_lse=True` + `layoutKv=PA_BSND` 下沉 A2 | 门禁原文、可见/隐藏用例、评测命令 |
| [BSA BSND layout](../experiments/bsa.md) | `BSND` 布局下沉 A2 | 门禁 + tilingKey + arch22 分支、ACLNN 直调配方 |
| [SFAG sinks](../experiments/sfag.md) | `sinks` 仅 950，下沉 A2 | aclnn 门禁 + tiling + arch22 kernel |
| [block_attn_res_prepare](../experiments/block_attn_res_prepare.md) | 整算子 A5-only 移植 A2 | 构建注册 + arch22 kernel/tiling |
| [MaskedCausalConv1d](../experiments/mcc.md) | 线性注意力因果卷积下沉 A2 | A2 注册 + arch22 kernel + 因果卷积参考 |

> 案例文件是“实验记录”，不是通用规范；换算子时按上面的方法重新定位门禁，不要照搬数值。
| [simple-arch22-kernel](../assets/simple-arch22-kernel.md) | 简单搬运类专用 arch22 kernel | 复用通用模板慢 20%–30% 时的最小实现模板 |
| [简单搬运路径对比](../experiments/spc.md) | 简单搬运类下沉 A2 | 单 TBuf vs TQue 双缓冲的核时对比 |
