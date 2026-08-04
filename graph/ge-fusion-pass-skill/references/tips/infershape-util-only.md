# tip: C++ 推 shape 只用 InferShapeUtil，全面禁用 GeUtils::InferShape

> 📎 导航落点：`references/interface-catalog.md` §二（pass 接口，shape 推导）、`references/fusion-troubleshooting.md` §6（InferShape 是否成功）。本文件仍是该接口纪律的**唯一权威定义源**，正文未改动。迁移映射见 `tips/MIGRATION.md`。

**读者**：② 开发 / ③ 分析。

## 症状

C++ replacement 需要推导 shape 时，随手调了 `GeUtils::InferShape`，导致 shape 未正确推导或行为不一致。

## 根因

自定义融合 pass 框架里，shape 推导的对外契约是 `ge::fusion::InferShapeUtil::InferShape`。`GeUtils::InferShape` 不是这条路径的正确入口。

## 硬性做法

- C++ replacement 需要推导 shape 时**必须**使用 `ge::fusion::InferShapeUtil::InferShape`。
- **全面禁用** `GeUtils::InferShape`。
- 若 `InferShapeUtil::InferShape` 接口不可用，记录 API gap 或失败原因，**不要**回退到 `GeUtils::InferShape`。

## 自查

- 源码里是否出现过 `GeUtils::InferShape`？有→改为 `ge::fusion::InferShapeUtil::InferShape`。
- InferShape 失败时，是记录了 API gap，还是偷偷回退了 `GeUtils`？必须是前者。
- 配合 `format-sensitive-nchw.md`：InferShape 失败常常是 format 未设而非接口问题，先排 format。
