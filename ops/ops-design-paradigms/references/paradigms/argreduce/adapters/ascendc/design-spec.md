# ArgReduce 设计约束（AscendC）

> **层级**：AscendC 语言适配层。设计产出的一致性检查清单与领域约束。

## 1 输入与输出定义

| 项 | 要求 |
|----|------|
| 输入 | 算子规格（或 spec.yaml）：输入 shape/dtype、dimension（含负轴）、keep_dims、indice_dtype、isMin 语义、withValue、NaN/平局/空 R 三项语义声明 |
| 输出 | 设计文档须包含：场景判定结论（含 trace）、选择语义结论（主规则/平局/NaN 逐项）、模式选择结论（[mode-mapping.md](mode-mapping.md) 决策树 + trace）、TilingKey/TilingData、资源与同步分析、验证要点 |

## 2 领域约束

1. **三项语义声明缺一不可**：平局规则（first/last）、NaN 语义（传播/按值）、空 R 输出约定。规格缺失时报告输入缺口，不得默认
2. **位置精度结论必须显式**：位置寄存器选型（与值通道等宽依据）+ 块内偏移上界核对（fp16→uint16 即 65536）+ 块基址拼接方案（int32 输出上限 2^31）；依赖浮点位置通道时必须同时声明平台依据（DAV_2201 历史口径）
3. **dtype 影响索引中间类型**：`ORIG_DTYPE_X == DT_INT64` 时索引中间类型选 int64，否则 int32
4. **GROUP_REDUCE 核数口径**：判定与 SyncAll 均用物理核数（originCoreNum），动态核数仅用于无依赖切核
5. **一套实现支撑算子族**：isMin/withValue 用模板参数，禁止按算子复制 kernel

## 3 一致性检查

| # | 检查项 | 通过标准 |
|---|--------|---------|
| 1 | 场景判定与折叠一致 | A1/R/A0 折叠正确；判定顺序 EMPTY→COPY→AR/ARA |
| 2 | 模式与场景一致 | mode ∈ 该场景的合法模式集（mode-mapping §6 对照表） |
| 3 | TilingData 字段闭环 | Host 计算来源 ↔ Kernel 消费逐字段对应（tiling-preprocess §4） |
| 4 | 平局规则三处一致 | 块内选择 / 跨块合并 / 跨核聚合的 Compare 模式同源（GE/LE 方向一致） |
| 5 | 位置偏移一次施加 | 块基址/核基址在生成侧或合并侧恰好施加一次，无重复无遗漏 |
| 6 | workspace 布局与聚合方式匹配 | AR 分支（core0 聚合）与 RA 分支（全核聚合）布局不同，见 capability-mapping |
| 7 | API 状态已核 | 用到的 API 在白名单内；标注"待验证"的结论以假设输出 |
| 8 | 精度验证独立 | 参考结果用独立 oracle（如 PyTorch 对拍），不用被测算子自证。**golden 陷阱**：`np.argmax` 的 NaN 语义与 torch **不一致**——golden 必须显式 NaN-aware 实现（全序 NaN > +Inf > 有限值 > -Inf，多 NaN 取先），禁止直接用 np.argmax |
| 9 | Issue 回修同步 | 开发期 issue（穿刺发现/编译期拦截）修复后，设计文档同步回修并更新修订记录——禁止只改代码留文档债（残留旧知识会误导后续算子与评审） |

## 4 产出格式

由调用方指定（工作流场景为 DESIGN.md 8 章结构）。范式要求设计文档中的代码块为**真实代码**（可用模板骨架填充），禁止省略占位（`// ...`）；完整性由调用方工作流自带的评审/门禁机制检查。
