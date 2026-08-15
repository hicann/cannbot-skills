# {operator_name} Open-source Alignment

> Attention / State Recurrence 专用模板。仅用于 Linear Attention / GDN / KDA / retention / RWKV / state recurrence 类算子。非此类算子不得引用本模板作为通用准出条件。

## 1. 用户数学 Contract

以下数学项是 GDN/KDA 默认示例。retention、RWKV 或其他 state recurrence 算子必须按用户公式增删行，例如 retention 可加入 decay/state，RWKV 可加入 time_mix/key_position。不要为了套模板保留不适用的数学项。

| 数学项 | 用户 contract | 是否覆盖到 golden/verify | 备注 |
|--------|---------------|--------------------------|------|
| `<按算子公式填写>` |  |  |  |
| `scale`（如适用） |  |  |  |
| mask / clamp（如适用） |  |  |  |
| dtype / cast |  |  |  |

## 2. 参考实现锁定

### 2.1 Reference Source 判定

| 字段 | 内容 |
|------|------|
| reference_source | `OPEN_SOURCE` / `USER_LOCAL` |
| 判定依据 | 用户是否显式给出“本地参考实现/实现参考/source-of-truth/pipeline 对齐”路径 |
| 用户给出的本地实现参考路径 | 仅 `USER_LOCAL` 填写；否则写“不适用” |
| 开源仓 URL | 仅 `OPEN_SOURCE` 必填；`USER_LOCAL` 也应作为对照填写 |
| clone_status | `CLONED` / `UNAVAILABLE` / `NOT_NEEDED` |
| clone 路径 | `CLONED` 时填写，例如 `tmp/open_source_refs/<repo>`；`UNAVAILABLE` 时写“不适用” |
| clone 失败原因 | 仅 `UNAVAILABLE` 填写 |
| 降级依据 | 仅 `UNAVAILABLE` 填写：仓内开源规范摘要章节、远程搜索路径、curated reference |
| commit / tag / 文件状态 | 必填 |

> 用户未显式给出本地实现参考路径时，必须使用 `OPEN_SOURCE`。用户给出的 baseline / 评测 / 性能对比路径只能填入 evaluation_baseline，禁止作为 primary reference。应尝试 clone 开源仓；clone 失败不得阻塞生成，但必须记录 `clone_status=UNAVAILABLE` 和降级依据。禁止把开发机本地仓库、历史算子目录或当前工作区外同名实现作为 primary reference。

### 2.2 Reference 清单

| 类型 | 路径 / URL | commit / 版本 | 关键文件 | 用途 |
|------|------------|---------------|----------|------|
| primary reference |  |  |  | open-source 或用户显式本地实现参考 |
| curated reference |  |  |  | 仓内 GDN/KDA 用例、mixed tolerance 精度规则、报告字段、Catlass 经验 |
| 用户本地 implementation reference |  |  |  | 仅用户显式要求按本地实现参考时填写 |
| evaluation_baseline |  |  |  | 仅评测指标/shape/报告字段，禁止作为实现参考 |

## 3. 公式到代码对齐

| 数学项 | 参考实现位置 | 本实现落点 | 采用 / 偏离 | 裁决理由 |
|--------|--------------|------------|-------------|----------|
| `<按算子公式填写>` |  |  |  |  |
| `scale`（如适用） |  |  |  |  |
| mask / clamp（如适用） |  |  |  |  |
| dtype / cast |  |  |  |  |

## 4. 文件级映射

| 参考文件 | 参考职责 | 本算子对应文件 | 采用方式 |
|----------|----------|----------------|----------|
|  | kernel pipeline |  |  |
|  | block scheduler |  |  |
|  | epilogue / finalize |  |  |
|  | workspace / flag |  |  |
|  | tiling key |  |  |

## 5. 差异裁决

| 高风险项 | 参考实现语义 | 用户 contract | 本实现裁决 |
|----------|--------------|---------------|------------|
| scale 作用位置 |  |  |  |
| mask / clamp |  |  |  |
| accumulator / cast / round |  |  |  |
| layout |  |  |  |
| varlen / partial |  |  |  |
| workspace / flag |  |  |  |
| tiling key / shape support |  |  |  |

## 6. Baseline 状态

| 项目 | 内容 |
|------|------|
| evaluation_baseline 路径 |  |
| 调用方式 |  |
| 支持 shape |  |
| 不支持 shape |  |
| 性能对比口径 |  |
