# 案例：复杂 Attention 反向类算子迁移（注册、标量暂存区、尾块）

> 实验记录，非通用规范。

## 环境与对象

| 项 | 值 |
|---|---|
| 仓库 | `ops-transformer@ba516241` |
| 算子 | 两段式 attention residual 的第二阶段（残差更新 + RMSNorm + online softmax 合并） |
| 目标 | `--soc=ascend910b`（DAV_2201 / arch22） |
| 工具链 | CANN 9.2.0 + torch_npu 2.7.1 |

## 基线现象

- 文档：A2 不支持；仓库只有 arch35 kernel；
- 直调报 `only supports Ascend 950 and Ascend 350`。

## 失败与修复（受控）

| 阶段 | 改动 | 结果 |
|---|---|---|
| 初版 | 补 kernel + def AddConfig，但缺 `op_host/op_tiling/arch22` 注册 | 包内只有 ascend950，`getws` 报 JSON config 找不到 |
| 第二轮 | 补 `arch22/<op>_arch22_tiling.cpp`（DAV_2201 注册），最后用 910b 打包 | A2 kernel 出现，`getws=0/run=0`，但精度仅 3/12 |
| 第三轮 | 标量暂存区从“共享 Tensor 的 8 元素切片”改为“每个标量独立 TBuf、平面 64 元素” | 严格用例 23/28 |
| 第四轮 | 修尾块/非对齐 D，增加 BF16 位级对比（`bad`/`1ulp_bad`） | **28/28 严格用例通过**，h 与 FP64 golden 位级一致 |

## 成本与收益

| 轮次 | 墙钟 | tokens | 构建次数 | 结果 |
|---|---|---|---|---|
| 第一轮（含死锁 + 注册缺失） | 56.7 min | 35.5 M | 40 | 23/28 |
| 修复轮（聚焦剩余失败） | **24.4 min** | **13.7 M** | 13 | **28/28** |

外部 harness（T=8,D=64，BF16 容差）：

| 指标 | 结果 |
|---|---|
| `getws` / `run` | 0 / 0 |
| `p_max_abs` | 0.0156 |
| `h_max_abs` | 0.0246 |

## 关键教训

1. **A2 注册三件套**：`op_tiling/arch22` 注册 + `op_kernel/arch22` + def `AddConfig("ascend910b")`，缺一不可；打包必须 910b 最后，否则被 950 覆盖；
2. **设备侧标量运算**：DAV_2201 无 `expf/sqrtf`，用 `AscendC::Exp/Sqrt` 单元素向量；标量暂存区不能切成 8 元素小平面，否则“p 对、h 错”；
3. **尾块与非对齐 D** 是边界用例失败的主因；BF16 输出要用位级对比（`1ulp_bad=0`）确认舍入与累加顺序。

## 同题 A/B（最小提示词、单轮、同机同版本环境）

| 组 | 墙钟 | tokens | 自测 | 外部 ACLNN 直调 |
|---|---|---|---|---|
| A（本 Skill） | **16.8 min** | **16.7 M** | 28/28 严格用例 | 通过（p=0.0156 / h=0.0246） |
| B（裸 agent） | 29.1 min | 38.9 M | 未给出用例结果 | 通过（p=0.0156 / h=0.0246） |

结论：同题对比下 A 组耗时/tokens 约为 B 组的 **0.58×/0.43×**，且 A 组额外完成 28 例严格自测；B 组单用例可跑通但未做系统精度验证。
