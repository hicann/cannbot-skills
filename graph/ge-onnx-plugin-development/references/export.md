# 导出侧：PyTorch 自定义算子导出 ONNX

模板：`templates/export_onnx.py`。

## 1. 自定义算子定义

autograd.Function + symbolic，symbolic 中用 `graph.op("域::算子名", ...)` 声明 ONNX 节点：

```python
class MyOpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        return input_tensor * 2.0  # TODO(填充): 算子的真实计算

    @staticmethod
    def symbolic(graph, input_tensor):
        return graph.op("example.domain::MyOp", input_tensor, alpha_f=1.0)
```

属性通过关键字参数后缀声明类型，后缀决定插件侧 `node.attrs` 取到的类型：

| torch 导出写法 | node.attrs 取到 | ONNX 属性类型 |
|---------------|----------------|--------------|
| `alpha_f=1.0` | `1.0`（float） | FLOAT |
| `axis_i=1` | `1`（int） | INT |
| `name_s="x"` | `"x"`（str） | STRING |
| `coeffs_f=[1.5, 0.5]` | `[1.5, 0.5]`（float 列表） | FLOATS |
| `dims_i=[1, 2]` | `[1, 2]`（int 列表） | INTS |
| `tags_s=["a", "b"]` | `["a", "b"]`（str 列表） | STRINGS |

后缀只有 `_f`/`_i`/`_s`（及 tensor 用的 `_t` 等），**没有"复数后缀"**：传标量值导出标量
属性，传列表值导出同类型列表属性，后缀不变（如 `dims_i=[1, 2]` 导出 INTS 列表）。

## 2. 导出调用

```python
torch.onnx.export(
    Model(), sample_input, "model.onnx",
    opset_version=18,                        # 标准域版本，官方算子按此导出
    input_names=["x"], output_names=["y"],
    custom_opsets={"example.domain": 1},     # 为每个用到的自定义域登记版本
)
```

## 3. domain/opset 对齐约定（最重要）

插件命中注册完全取决于「模型侧构造的 origin type」与「插件注册的 origin type」字符串精确
匹配（`<domain>::<opset>::<source>`），任一段不同都查不到：

| 环节 | 谁定 | 写什么 | 示例 |
|------|------|--------|------|
| symbolic（导出侧） | 模型作者 | 算子的 `域::算子名` | `graph.op("example.domain::MyOp", ...)` |
| custom_opsets（导出侧） | 模型作者 | 该域登记的版本号 | `custom_opsets={"example.domain": 1}` |
| onnx_plugin（插件侧） | 插件作者 | 对抄导出侧的域和版本 | `domain="example.domain", opsets=(1,)` |

规则：

- **两侧一致是硬要求**：插件 `domain` 必须与 symbolic 的域取值完全相同；插件 `opsets` 必须
  包含 custom_opsets 登记的版本。自定义域版本惯例恒为 `1`（登记 1、声明 `(1,)` 即可）。
- `ai.onnx` 允许使用（标准算子重申映射、自定义算子注册均合法）。用 `ai.onnx` 注册自定义算子时：
  插件 `domain="ai.onnx"`、`opsets` 覆盖模型的 `opset_version` 值。
- **不要在 custom_opsets 里给 `ai.onnx` 登记版本**：空 domain 与 `ai.onnx` 归一化到同一版本表键，
  custom_opsets 的登记会与 `opset_version` 互相覆盖（后写生效），把全模型标准算子
  （Relu/Conv 等）的 origin type 版本段一起改坏，报 E13010/E16002 连坐。
- 报错中的 origin type 三段拆解可快速定位差异：`example.domain::1::MyOp` =
  domain（symbolic 写的域）:: opset（custom_opsets 登记的版本）:: source（算子名）。

## 4. 导出后自检

```bash
python3 -c "import onnx; m = onnx.load('model.onnx'); print(list(m.opset_import)); \
[print(n.op_type, repr(n.domain)) for g in m.graph for n in g.node if n.domain]"
```

确认两点：每个自定义域都有对应版本条目；自定义节点的 domain 与 symbolic 写的取值一致。

注意：`torch.onnx.export` 内部依赖 onnx 包，隔离环境缺包会报
`Module onnx is not installed!`（`pip3 install onnx` 解决）。torch 2.7~2.8、onnx 1.21.0 已实测。
