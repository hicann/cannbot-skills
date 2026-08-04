# tip: 信任 es_all wrapper，不预换版本名（唯一决策树）

> 📎 导航落点：`references/pass-development-paradigm.md` §7（实现）。本文件仍是该决策树的**唯一权威定义源**，正文未改动。迁移映射见 `tips/MIGRATION.md`。

**读者**：② 开发 / ③ 分析。这是全套 tips 里唯一容易被写反的一条，**默认分支只有一个，回退分支严格受真实运行期证据门控**。

## 症状

- 想在 replacement 里新建某个算子（如 `es::GEMM`、`es::Conv2D`）。
- 曾经的错误做法：凭"某个 V2/V3 版本才有 kernel"的经验，把样例/需求分析文档要求的 wrapper 预先改写成 `GemmV2`/`GemmV3`/`Conv2DV2` 之类——结果 agent 主动跳过可用的 wrapper 去自行构建未注册版本，反而更容易挂。

## 根因

- `es_all` 是从**当前环境的 op proto 现场生成**的。它**暴露出来的 C++/Python wrapper 就是可信来源**——暴露即可用。
- "能建图 / 能 InferShape"（op_proto 决定）与"本 soc 能否编译运行 / 有无注册 kernel"是**两套独立的事实**，对外暴露的名字可能三处不对齐（ES 符号名 ≠ 产出节点 op type ≠ 本 soc 已注册 kernel 的 op type）。这套背景见 `kernel-registration-mismatch.md`——但**它只解释为什么存在回退分支，不构成"预换版本名"的许可**。

## 硬性做法（决策树）

```
需要新建算子节点
   │
   ├─ es_all 暴露了对应 wrapper（如 es::GEMM / es::Conv2D）？
   │     │
   │     ├─ 是 → 【默认分支，永远先走这里】直接用该 wrapper，样例/需求分析文档用什么就用什么，
   │     │        落定前不做任何"版本纠正"。不预换 V2/V3。
   │     │           │
   │     │           └─ 真实 ATC/pyatc 报 `Failed to select engine for [XxxOp]`
   │     │              （该 op type 本 soc 确无 kernel）？
   │     │                 ├─ 否 → 保持原 wrapper，继续。
   │     │                 └─ 是 → 【回退分支，仅此条件下】按报错算子改用同一算子的
   │     │                          等价实现，日志写明改动依据；不反复重试同一报错 op type。
   │     │
   │     └─ 否（es_all 未暴露你需要的 wrapper）→ 走显式手建：
   │              见 `compliant-node-builder-ir-order.md`（IR 顺序严格按 op_proto REG_OP）。
```

**禁止**：
- 在 es_all 已有 wrapper 时，因"想换版本名 / 少建常量节点"绕开它去自行构建。
- 凭经验/静态推断预换版本名（未见真实 `Failed to select engine` 就改名）。
- "明知 InferShape 会失败仍把图返回出去（returning graph anyway）"把失败甩给"已知限制"。

## 自查

- replacement 里每个新建算子，用的是不是 es_all 暴露的原始 wrapper？（列出 wrapper 名）
- 有没有在**没见到**真实 `Failed to select engine` 的情况下改过版本名？有→回退到原 wrapper。
- 若确实回退了等价实现，日志里有没有写清依据（哪条报错、原 op type、替换为哪个）？
