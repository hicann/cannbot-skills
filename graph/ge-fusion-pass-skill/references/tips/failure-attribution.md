# tip: 失败归因纪律——先查"这一步漏没漏"，别急着甩给环境/文档

> 📎 导航落点：`references/fusion-troubleshooting.md`（开篇诊断纪律的根）。本文件仍是该归因纪律的**唯一权威定义源**，正文未改动。迁移映射见 `tips/MIGRATION.md`。

**读者**：② 开发 / ③ 分析。来源：多次批跑复盘的共同规律——同一任务"通过版 vs 失败版"的差别，往往只是失败版**漏做了某一步**（如落 Conv2D 前漏 `SetFormat(NCHW)`），纯属 run-to-run 遗漏，而非环境或文档问题；把开发遗漏误判为"文档碎片化/环境不行"会一直修错地方。

## 症状

- pass 挂了，第一反应是"文档没写清 / 环境有问题 / 这个算子不支持"，于是去换算子、改环境、绕行。
- 但换个方式重跑有时又好了——说明不是稳定的环境/文档问题，而是某步做法漂移。

## 硬性做法

失败时按这个顺序归因，先排干净"自己漏步"再谈外部：

1. **对照硬性 tips 逐条自查**：报错先按签名匹配到对应 tip 的"自查"清单走一遍——
   - `Failed to select engine` → `es-all-no-version-rename.md`（是不是预换了版本名/没用 es_all wrapper）
   - `E50002` format → `format-sensitive-nchw.md`（中间 tensor 漏设 NCHW）
   - `attribute order has changed` → `compliant-node-builder-ir-order.md`（IR 顺序）
   - 静默不命中 → `dump-first-op-type.md`（op type 用了框架名）
   - 结果和源码对不上 → `stale-pass-artifact-cleanup.md`（残留 pass）
2. **通过版 vs 失败版逐步对比**：若有过一次成功/一个已通过的近似实现，逐步 diff，定位"少做/做反了哪一步"，而不是重写。
3. **确属环境缺失**（无 NPU / 缺 atc / 缺 es_all / 缺 npu_bridge）：按门禁 G3 如实标注"未运行"，**不伪造结果**，也不把它记成"pass 写错了"。
4. **确属文档缺失/冲突**：记录 API gap（缺哪个签名、哪两处冲突），按 `api-signature-gate.md` 走回退核实，不要靠猜。

## 反面清单（不要做）

- 见 `Failed to select engine` 就反复重试同一报错 op type，或凭经验预换版本名。
- 明知 InferShape 会失败仍把图返回出去（returning graph anyway），把失败甩给"已知限制"。
- 一挂就归因"文档不行/算子不支持"，跳过 tips 自查。

## 自查

- 这次失败，对应 tip 的自查清单是不是逐条过了？
- 是"稳定复现的外部缺失"（→ G3 标注），还是"换个做法就好的自己漏步"（→ 修做法）？
