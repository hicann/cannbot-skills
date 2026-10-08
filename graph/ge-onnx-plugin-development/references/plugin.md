# 插件编写：descriptor 与三种回调

模板：`templates/plugin_parse_node.py`（一对一）、`templates/plugin_decompose.py`（一对多）、
`templates/plugin_parse_operator.py`（JSON 整体解析）。

## 1. descriptor

```python
from ge.onnx_plugin import onnx_plugin

my_op = onnx_plugin(
    source="MyOp",            # ONNX 原始算子类型（symbolic 里 域:: 后面的名字）
    domain="example.domain",  # 与 symbolic 的域一致（对齐约定见 export.md §3）
    opsets=(1,),              # 覆盖 custom_opsets 登记的版本
    target="PartitionedCall", # GE 目标算子类型，原型必须已安装并注册
)
```

- `source`/`domain` 不允许包含 `:`（origin type 分隔符）；`opsets` 元素为正整数；
- 同一 source 只能注册一个插件，不同插件 opsets 不能重叠（冲突在编译开始时报错退出）；
- 同一 descriptor 可绑定多个回调（如 parse_node + decompose 接力）；
- FrameworkType/ImplyType 由框架内部固定，不对外暴露。

## 2. 回调选型

```text
节点属性是否含 tensor/子图？
├─ 是 → parse_operator（node.attrs 读 tensor/子图属性会直接报错）
└─ 否 → 是否需要不预知属性名、整体搬运全部属性？
    ├─ 是 → parse_operator
    └─ 否 → parse_node（优先推荐）
GE 没有现成算子可一对一映射？
└─ 是 → 追加 decompose（与 parse_node 接力）
```

同一 origin 下 parse_operator 优先于 parse_node（同时绑定时只有 parse_operator 生效）。

### 2.1 parse_node：按名取属性（优先）

```python
from ge.graph import Operator
from ge.onnx_plugin import OnnxNode

@my_op.parse_node
def parse_my_op(node: OnnxNode, target: Operator) -> None:
    alpha = node.attrs.get("alpha", 1.0)  # 键 = symbolic 的属性名；值类型 = 导出后缀决定
    target.set_attr("alpha", alpha)
    target.register_input("x")            # 端口注册按需，见 §3
    target.register_output("y")
```

- 返回值必须是 None；
- `node.name`/`origin_type`/`inputs`/`outputs`/`attrs` 只读（inputs/outputs 为 tuple，
  保持 ONNX 数据名顺序与空占位）；
- `target.set_attr` 支持 bool/int/float/str 及同类列表，其他类型明确报错；
- **属性名转写**：`set_attr` 的键是 GE 目标算子 IR 原型的属性名，与 ONNX 节点属性名**不一定相同**
  （如 ONNX `LeakyRelu` 的 `alpha` 对应 GE `LeakyRelu` 的 `negative_slope`）。动手前先查证目标
  算子原型的属性名，方法：`python3 -c "import ge.es.nn as n; print(n.LeakyRelu.__doc__)"`
  （es.math 同理），或查 CANN 算子文档。属性名写错可能不报编译错，到执行阶段数值不对才暴露；
- 回调抛异常时当前节点解析失败（不影响其他节点报错），bridge 会把 Python 异常
  转换为 parser failure。

### 2.2 parse_operator：整体解析属性

```python
import json

@my_op.parse_operator
def parse_my_op(source, target) -> None:
    attrs = json.loads(source.get_attr("attribute"))
    alpha = 1.0
    for attr in attrs.get("attribute", []):
        if attr.get("name") == "alpha":
            alpha = float(attr.get("f", 1.0))
    target.set_attr("alpha", alpha)
```

source 是框架把 ONNX 节点先行物化成的算子（**只读**），全部节点属性被打包成一个 JSON 串
存于 `attribute` 键。JSON 结构：最外层只有 `attribute` 一个键，值为数组，每个元素描述一个属性
（含 `name`、`type`、存放值的键）：

```json
{
  "attribute": [
    { "f": "1.5",           "name": "alpha",  "type": 1 },
    { "i": 3,               "name": "level",  "type": 2 },
    { "s": "x",             "name": "label",  "type": 3 },
    { "t": { "...": "..." }, "name": "value",  "type": 4 },
    { "g": { "...": "..." }, "name": "branch", "type": 5 },
    { "floats": [1.5, 0.5], "name": "coeffs", "type": 6 },
    { "ints": [2, 3],       "name": "dims",   "type": 7 },
    { "strings": ["a","b"], "name": "tags",   "type": 8 }
  ]
}
```

- `type` 是 ONNX 属性类型编号：1=float 2=int 3=string 4=tensor 5=子图 6=float列表 7=int列表 8=string列表；
- 标量（`f`/`i`/`s`）的值是字符串形式，取用时转数值；列表（`floats`/`ints`/`strings`）直接是数组；
  tensor（`t`）与子图（`g`）是完整结构体字典，一般不需要逐字段解析；
- 完整字段说明见 GE 仓文档 `https://gitcode.com/cann/ge/blob/master/docs/zh/api/graph_engine_api/python/ge/onnx_plugin/OnnxPlugin/parse_operator.md`。

### 2.3 decompose：用已有算子拼子图

```python
from ge.es import GraphBuilder
from ge.es.math import Mul
from ge.es.nn import Threshold

@my_op.decompose
def decompose_my_op(source):
    alpha = float(source.get_attr("alpha"))  # 读到的是 parse_node 写入的值（属性中转）
    builder = GraphBuilder("my_op_decomposition")
    x = builder.create_input(0)
    mask = Threshold(x, threshold=alpha)
    output = Mul(x, mask)
    return builder.build_and_reset([output])  # 返回值必须是 ge.graph.Graph
```

- decompose 依赖参数解析回调的产出：source 读到的属性、端口均来自同一 descriptor 的
  parse_node/parse_operator（对动态 IO target 是必需的接力）；
- builder 在回调内部创建，不作为参数传入；
- 子图输出要和参数解析阶段注册的输出对应；子图输入不用手动指定 dtype/shape
  （GE 按原节点输入自动对齐）；
- 构图只用 `ge.es.math`/`ge.es.nn` 等 ES 生成算子包（清单见 GE 仓
 `https://gitcode.com/cann/ge/blob/master/docs/zh/user_guides/es_graph/api/es_python.md`），没有字符串型万能 `graph.op` 工厂；
- 返回非 Graph（含 None）使当前节点解析失败。

## 3. 端口注册

目标算子输入输出个数固定（静态 IR，如 Elu）时**不用注册**；个数不固定（动态 IO，
如 PartitionedCall）时**必须逐个注册**，否则 parser 连线阶段按索引找不到端口。

连线机理：GE 按位置对号入座——节点第 i 个输入接到目标算子第 i 个输入上。因此：

- **位置要占满**：按 ONNX 算子的输入顺序，每个位置都注册（必需用 `register_input`，
  可选位置用 `register_optional_input`）。只要有节点可能在第 k 个位置连数据，第 k 个输入
  就必须注册，否则 `E19999: Resolve operator IO name failed`；
- **顺序要对**：注册顺序必须与 ONNX 算子的输入顺序一致。顺序错不报编译错，
  到 shape 推导或执行阶段才暴露（数据接错端口）；
- 可选输入不连时框架不自动填值，缺省语义由目标算子定义（同一算子在 A 节点连、
  B 节点不连完全合法，不连只打告警）。

| 方法 | 用途 |
|------|------|
| `register_input(name)` | 必需输入 |
| `register_optional_input(name)` | 可选输入（输出没有可选注册） |
| `register_output(name)` | 输出 |
| `register_dynamic_input(name, count)` | 动态输入，如 Sum→AccumulateNV2 按 `len(node.inputs)` 注册并 `set_attr("N", count)` |
| `register_dynamic_output(name, count)` | 动态输出 |

## 4. 生命周期与目录组织

- `Operator` 只在 callback 内有效：callback 返回或抛异常后，其任何方法抛
  `RuntimeError`；不能保存到全局复用，不能 copy/deepcopy/pickle；
- `parse_operator`/`decompose` 的 source 只读：`get_attr`/`name`/`type` 可用，
  `set_attr` 与端口注册抛 `RuntimeError`；
- 插件目录只放插件 .py 文件：GE 扫描 `ASCEND_CUSTOM_OPP_PATH` 指向目录下的一层
  Python 文件，导出器/执行器（依赖 torch/numpy/acl）会被误当插件加载；目录内任一
  文件加载失败（如语法错误）会中断整个编译（fail-fast）；
- `ge.graph.Operator`（parser callback 期）与 `ge.graph.Node`（节点已入图后）是不同对象，
  不共享 handle。

## 5. 常见错误对照

| 症状 | 原因 |
|------|------|
| E13010/E16002 算子未注册 | domain 或 opsets 两侧不一致（按 export.md §3 对照 origin type 三段拆解） |
| E19999 Resolve operator IO name failed | 动态 IO target 没注册端口，或端口位置没占满 |
| node.attrs 取属性报错 | 属性是 tensor/子图类型，改用 parse_operator |
| Source Operator is read-only | 在 parse_operator/decompose 的 source 上调 set_attr 或端口注册 |
| Operator is only valid inside parse_node | callback 外保存并使用 Operator |
| 编译开始即报插件冲突退出 | 同 source 多插件或不同插件 opsets 重叠 |
