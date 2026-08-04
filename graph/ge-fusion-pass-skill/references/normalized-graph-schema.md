# normalized-graph.json Schema（统一图表达）

输入适配（skill 内部里程碑 R3）产出的 `normalized-graph.json` 用本 schema 表达；**验证证据阶段（内部 R4）的图 diff 共用本 schema**——baseline 与 optimized 的 normalized-graph 按节点/边同构比对，就是该阶段的图结构正确性证据。R3/R4 均是 workflow 里程碑，不是版本号。

> 本 schema 是**结构契约**，不是知识源。ONNX/AIR/pbtxt 各自的解析细节在 `"$SKILL_ROOT/scripts/adapt_input.py"` 与 `input-adaptation.md`；真实 GE op type 仍以 `PreRunBegin` dump 为准（`tips/dump-first-op-type.md`），normalized-graph 里的 `op_type` 是解析阶段的最佳值，编译后以 dump 修正。

## 设计原则

- **统一表达，不统一语义**：把 ONNX/AIR/pbtxt/dump 的图都映成同一套节点+边，但**不声称恢复了源格式没有的信息**（pbtxt 缺权重/属性 → 标 `missing`，不编造）。
- **控制边/动态端口/GE 内部 op 一等公民**：schema 有专门字段；前端表达不了时标 `unrepresentable` + 降级建议（走 ES/GE IR），不硬塞。
- **provenance 伴随**：`normalized-graph.json` 只存图结构，来源/转换命令/假设/复现级别在 `provenance.json`（见 `input-adaptation.md`）。

## 顶层结构

```json
{
  "schema_version": "1.0",
  "source": {
    "input_type": "onnx | air | pbtxt | torch_script | tf_script | ge_dump",
    "file": "<原始文件路径>",
    "sha256": "<文件 sha256>",
    "reproduction_level": "structural | semantic"
  },
  "nodes": [ { ... } ],
  "data_edges": [ { ... } ],
  "control_edges": [ { ... } ],
  "outputs": [ { ... } ],
  "unrepresentable": [ { ... } ],
  "missing": [ { ... } ]
}
```

- `reproduction_level`：
  - `semantic`——normalized-graph 能反映源模型的完整语义（ONNX/AIR 正常解析时）。
  - `structural`——只承诺结构复现，可能缺权重/属性/原始框架语义（pbtxt、残缺输入）。**pbtxt 一律标 `structural`**，不声称语义无损。

## nodes 字段

```json
{
  "id": "n0",
  "op_type": "ge:Add",
  "name": "Add_1",
  "original_op_type": "Add",
  "inputs": [ {"port": 0, "from_node": "n_in0", "from_port": 0}, ... ],
  "outputs": [ {"port": 0, "name": "out0"}, ... ],
  "attrs": { "axis": 1, "data_format": "NCHW" },
  "shape": {"out0": [2, 3]},
  "dtype": {"out0": "FLOAT"},
  "format": {"out0": "NCHW"},
  "optional_inputs_present": ["bias"],
  "unrepresentable": ["动态输入个数无法静态确定"]
}
```

- `op_type`：解析阶段得到的最优 op type。**编译后以 `PreRunBegin` dump 的 `ge:Xxx` 修正**（前端名经 GE 导入会改名，`tips/dump-first-op-type.md`）。`original_op_type` 保留源格式原名（如 ONNX 的 `Add`）。
- `attrs`/`shape`/`shape_range`/`dtype`/`format`：能从源解析到的就填；解析不到的 key 不出现，**不填 null 假装有**。GE pbtxt 的 `input_desc_*`/`output_desc_*` 会规范成 `input:<port>`/`output:<port>` 键；`shape`/`shape_range` 是维度数组，`dtype` 是 GE dtype 名，`format` 是 `{storage, origin}`。
- `optional_inputs_present`：显式列出存在的可选输入（`bias`/`offset_w` 等），与 `fragment-spec.md` §四可选输入策略对应。
- `unrepresentable`（节点级）：该节点有但 schema/前端表达不了的特性（动态输入个数、控制边、嵌套子图），逐条列 + 触发 graph 路线（`pass-development-paradigm.md` §3.1 矩阵）。

## data_edges 字段

```json
{ "from_node": "n0", "from_port": 0, "to_node": "n1", "to_port": 1 }
```

数据依赖边。端口编号与节点 `inputs`/`outputs` 对齐。

## control_edges 字段

```json
{ "from_node": "n0", "to_node": "n1" }
```

控制依赖边（无端口）。`PatternFusionPass` 不支持控制边（`interface-catalog.md` §一.2），出现控制边即触发 graph 路线。

## outputs 字段

```json
{ "name": "z", "from_node": "n_add", "from_port": 0 }
```

按图输出的原始顺序记录名称、producer 节点和输出端口。验证证据阶段用它核验 baseline 与 optimized 的最终输出数量、顺序和来源端口。源格式没有可靠输出签名时，`outputs` 为空且 `missing` 必须记录原因；**不得用无消费者的末端节点猜测输出**。

## unrepresentable（图级）

```json
[ { "element": "n3", "reason": "嵌套子图，pattern 无法表达", "degradation": "graph 路线" } ]
```

图级汇总：哪些节点/边无法用 pattern 表达，原因，退化到哪。与 `pass-development-paradigm.md` §3.1 选型矩阵对接。

## missing（图级）

```json
[ { "element": "n5.w", "reason": "pbtxt 未含权重", "assumption": "复现时用随机权重，仅验结构" } ]
```

源格式本应有但缺失的信息（pbtxt 缺权重/属性、dump 缺属性值）。**逐条记 + 复现时的补全假设**，不假装恢复。`provenance.json` 的 `assumptions` 字段汇总这些。

## 最小示例（单 Add）

```json
{
  "schema_version": "1.0",
  "source": {
    "input_type": "onnx",
    "file": "min_add.onnx",
    "sha256": "<sha256>",
    "reproduction_level": "semantic"
  },
  "nodes": [
    {"id": "n_in0", "op_type": "ge:Data", "name": "x", "outputs": [{"port": 0, "name": "x"}], "dtype": {"x": "FLOAT"}, "shape": {"x": [2, 2]}},
    {"id": "n_in1", "op_type": "ge:Data", "name": "y", "outputs": [{"port": 0, "name": "y"}], "dtype": {"y": "FLOAT"}, "shape": {"y": [2, 2]}},
    {"id": "n_add", "op_type": "ge:Add", "original_op_type": "Add", "name": "Add_0",
     "inputs": [{"port": 0, "from_node": "n_in0", "from_port": 0}, {"port": 1, "from_node": "n_in1", "from_port": 0}],
     "outputs": [{"port": 0, "name": "z"}], "dtype": {"z": "FLOAT"}, "shape": {"z": [2, 2]}}
  ],
  "data_edges": [
    {"from_node": "n_in0", "from_port": 0, "to_node": "n_add", "to_port": 0},
    {"from_node": "n_in1", "from_port": 0, "to_node": "n_add", "to_port": 1}
  ],
  "control_edges": [],
  "unrepresentable": [],
  "missing": []
}
```

## 与验证证据图 diff 的接口

验证证据阶段取 baseline（pass 未生效）与 optimized（pass 生效）两份 `normalized-graph.json`：
- 按节点名称、`op_type`、带端口 `data_edges`、控制边和 `outputs` 比对，产出图结构变化证据。
- `unrepresentable`/`missing` 透传到验证证据报告，标注"结构复现"边界。
- 整网**输出**比较（数值）不靠本 schema，靠验证证据阶段跑 baseline/optimized OM 比（`requirements-analysis-template.md` §9.3）。

对于 pbtxt 生成的最小 ONNX，`artifacts/repro/structural-isomorphism.json` 必须精确比较节点 `name`/`op_type` 与数据边 `src:port → dst:port`。只有该文件的 `status: PASSED` 才能把 reproduction 标记为“结构同构”；控制边、缺失外部边界或不可表达结构必须降级到 ES/GE IR，不得伪造通过。
