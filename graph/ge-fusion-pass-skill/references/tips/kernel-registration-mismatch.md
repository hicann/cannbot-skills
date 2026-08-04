# tip: "能建图 ≠ 能编译" 的根因（回退分支背景，非预换许可）

> 📎 导航落点：`references/fusion-troubleshooting.md` §6（引擎选择失败）。本文件仍是该根因背景的**唯一权威定义源**，正文未改动。迁移映射见 `tips/MIGRATION.md`。

**读者**：③ 分析 / ①（理解为什么会有回退分支）。

> **本文只解释 `Failed to select engine` 回退分支的根因，不构成"预换版本名"的许可。** 默认永远是信任 `es_all` 暴露的 wrapper——决策树见 `es-all-no-version-rename.md`，那才是要执行的规则。
>
> **防回归（有前车之鉴）**：项目里确实一度加过"落节点前必须校验 op type 的 kernel 注册、不在已注册清单内就换成已注册的 GemmV2/GemmV3/Conv2D"这套**正向**引导（连同一个收集 kernel 清单的 es-api-collector）。随后被**整体撤销**：它反而误导 agent 主动跳过可用的 `es::GEMM` wrapper 去自行构建未注册版本，首次通过率不升反降（一个批次对比实证）。所以：**不要以任何形式复活"预先按 kernel 清单换版本名/落节点前先查 kernel 注册再决定用哪个 op type"的正向规则。** 唯一允许的动作是——见到**真实运行期** `Failed to select engine` 后，才按报错回退等价实现。宿主仓库的静态校验应保留这一护栏。

## 症状

自定义 pass 用 ES API 构造的替换节点在 ATC 离线编译阶段报：

- `Failed to select engine for [XxxOp]`（典型：用了本 soc 无 kernel 的 op type）
- 或换用 Graph API/CompliantNodeBuilder 手建后报 `Failed to recover ir definitions` / `attribute order has changed`

两类报错都不提示"应改用哪个 op type""属性应按什么顺序给"，容易反复试错。

## 根因

算子的三处名字互不对齐：

```
ES API 符号名        ≠   产出节点的 op type   ≠   本 soc 已注册 kernel 的 op type
```

- **op_proto**：决定有无 IR / shape 推导 → 能建图、能 InferShape。
- **kernel 注册**（`opp/built-in/op_impl/.../<soc>/aic-*-ops-info-*.json` 顶层 key）：决定 ATC `select engine` 阶段能否为该 op type 选到执行引擎 → 能不能编译运行。
- **ES API**：某算子 C++ 头可能只暴露了未注册的那个版本，Python ES 又可能同时有多个版本。

所以"能建图 ≠ 能编译"：建图/InferShape 成功不代表本 soc 有 kernel。

## 硬性做法

- **默认**：按 `es-all-no-version-rename.md` 的决策树，信任 es_all 暴露的 wrapper，不预换。
- **仅当真实 ATC/pyatc 报 `Failed to select engine for [XxxOp]`**：确认是该 op type 本 soc 无 kernel，才按报错算子回退同一算子的等价实现；改动依据写进日志；不反复重试同一报错 op type。
- **`attribute order has changed` / `Failed to recover ir definitions`** 属于另一类（显式手建 IR 顺序错），修法见 `compliant-node-builder-ir-order.md`，不是换版本名能解决的。

## 自查

- 报错到底是 `Failed to select engine`（kernel 缺失，走本 tip 回退）还是 `attribute order has changed`（IR 顺序，走 CNB tip）？先分清再动手。
- 回退前，有没有确认这是**真实运行期**报错，而不是静态猜测？
