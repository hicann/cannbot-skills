# 约束 + 示例

---

## 强制约束

1. 禁止从外部 JSON 加载映射配置 — 映射逻辑必须直接写在 `map_case()` 函数体中
2. 禁止硬编码 OUTPUT_NAME_MAP — 返回 dict 中 output tensor 的 key 必须与 `operator_model.outputs[*].name` 一致
3. 禁止凭直觉猜测 shape / ndim — 必须从 `S5_mapping_spec.md` 的散文描述推导，该散文本身由 `operator_model` 生成
4. 禁止 import torch / torch_npu — mapper 是纯计算模块
5. 禁止修改任何输入文件（S5/S2P1）
6. 禁止跳过 validate_config
7. `ndim` 范围和 `tensor_constraints` 必须从 `S2P1_operator_model.json` 的 `inputs[*].rank` 和 `inputs[*].shape.constraints` 提取，禁止从 tiling/infershape 源码重新推断
8. 禁止 NPU 相关依赖出现在 mapper 中
9. 禁止将 `S2P1_low_configs.json` 中的 `source`、`reason` 等元信息字段写入 mapped JSON 的 `params` 中 — 网络用例仅保留算子参数
10. `map_case()` 生成后，必须与 `operator_model` 做 L2 交叉验证：dtype/rank 范围、outputs key 一致
11. `_decompose` 允许因子 = 1，`parts = 0` 时返回空 tuple `()`
12. `seed = 42` 固定，pytest 和 TTK CSV 必须使用相同 seed

---

## ✅/❌ 示例

```python
# ✅ 映射逻辑直接 inline — ❌ 禁止从外部 JSON 加载
def map_case(case, rng=None):
    # 逻辑由 S5_mapping_spec.md 散文逐条翻译
    {tensor}_ndim = rng.randint({lo}, {hi})
    ...

# ✅ parts=0 特殊处理 — ❌ 禁止不处理 parts=0
leading_shape = () if leading_dims == 0 else _decompose(product, leading_dims)
```
