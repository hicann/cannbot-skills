# 踩坑经验

## 先探测 API，再写测试

禁止凭文档或直觉猜测 API 名称和签名。编写测试前，必须执行探测：

```bash
python -c "import torch_npu; print(hasattr(torch_npu, 'npu_xxx'))"
python -c "import torch, torch_npu; torch_npu.npu_xxx(torch.randn(1,2).npu())"
```

在代码中用 `hasattr` 守护：

```python
HAS_NPU_OP = hasattr(torch_npu, "npu_xxx") and torch.npu.is_available()
if not HAS_NPU_OP:
    pytest.skip("npu_xxx not available")
```

## Golden 函数的唯一来源是 aclnn 文档「计算公式」

1. **只读**算子 `docs/aclnn*.md` 文件中的「计算公式」这一节（通常紧跟在"接口功能"之后，以 `$$` 包裹的公式）
2. **不读**同一文档的其他节（函数原型、参数说明、示例代码等）
3. **不推**不从 kernel 源码反推公式——kernel 是参考实现，不是定义
4. 公式中每个变量的含义必须从公式本身推断，不需要读参数说明表
5. **输出-公式映射表**：编写 reference 前，必须为每个输出 tensor 显式标注其对应的公式表达式。格式（以注释形式声明在 reference 函数上方）：
   ```python
   # 输出-公式映射：
   # - {output_name_1} = {公式表达式}     ← 来自公式 {来源小节}
   # - {output_name_2} = {公式表达式}     ← 来自公式 {来源小节}
   # - {output_name_3} = {公式表达式}     ← 来自公式 {来源小节}（未显式定义，经数值验证确认）
   ```
   如果公式节未显式给出某个输出的表达式，必须通过数值交叉验证（见第 6 条）确定，禁止猜测。
6. **数值语义验证**：当公式节未显式定义某个输出的计算式时，用一个小 case 在 NPU 上运行，列出所有候选表达式并逐个计算 diff，选择 diff 最小的作为正确语义。将验证结论写入输出-公式映射表注释中。

## Reference 必须用真实 NPU 输出交叉验证

先手动跑一个小 case 对比 NPU 输出和 reference 输出，确认一致后再批量生成测试。关注点：
- **输出 shape**：NPU 返回的 shape 是否与 reference 一致（尤其注意广播、未 expand 的中间结果等）
- **输出 dtype**：每个输出的实际 dtype 是否与预期一致（尤其注意混合精度场景下中间输出与主输出 dtype 不同）
- **输出数量**：API 实际返回几个值（可能不等于接口文档描述）
- **输出语义**：每个输出 tensor 的数值是否与输出-公式映射表中声明的表达式一致。禁止只检查 shape/dtype 而跳过数值语义验证。

## 禁止假 PASS — 没验证就不能报通过

```python
# ✅ 正确：NPU 不可用时 skip，或做了实质验证才 report pass
if not HAS_NPU_OP:
    pytest.skip("NPU not available")
# ... NPU 对比 ...

# ❌ 错误：NPU 对比放在 if 里，else 分支只做了 CPU 属性检查就放行
if HAS_NPU_OP:
    torch.testing.assert_close(npu_out, ref_out)
# 没有 else → CPU 上跑了个 assert shape == expected 就 PASSED（假 PASS）
```

## 输出含 inf/nan 时，比较必须用 equal_nan=True

边界值输入（`extreme` 取 dtype 最大值）或多步算子会导致中间计算溢出，输出包含 inf/nan，这是正常测试路径。

```python
# ❌ 错误：默认 equal_nan=False，nan != nan 必然失败
torch.testing.assert_close(npu_out.cpu(), ref_out, rtol=rtol, atol=atol)

# ❌ 错误：放任 FAILED 不处理，门禁会拦截
torch.testing.assert_close(npu_out.cpu(), ref_out, rtol=rtol, atol=atol, equal_nan=True)

# ❌ 错误：降低精度标准放行（见铁律）
torch.testing.assert_close(npu_out.cpu(), ref_out, rtol=1.0, atol=1.0)
```

正确写法已在 01-code-templates.md「断言」节给出（try/except + equal_nan=True + pytest.xfail）。

## 可变输出数量的算子 — 全量解包，按模式验证

部分算子的接口支持多种输出模式（如根据可选输出参数是否为 None 决定返回不同数量的输出）。torch_npu Python API 通常固定返回所有输出 tensor。

**禁止做法**：因为当前模式不需要某个输出就跳过整个用例。

```python
# ❌ 错误：某个模式下不需要某输出，直接跳过
if mode != full_mode:
    pytest.skip("mode not supported")
```

**正确做法**：始终用全量变量承接返回值，然后根据模式只验证该模式应有的输出。

```python
# ✅ 正确：全量解包，按模式验证
outputs = npu_op(input_1, input_2, ...)
npu_out_0 = outputs[0]
npu_out_1 = outputs[1] if len(outputs) > 1 else None
npu_out_2 = outputs[2] if len(outputs) > 2 else None

# shape/dtype 检查：所有模式下都检查的输出
assert npu_out_0.shape == expected_shape_0
assert npu_out_0.dtype == expected_dtype_0

# 按模式验证对应输出
if should_verify_output_1:
    assert npu_out_1.shape == expected_shape_1
if should_verify_output_2:
    validate(npu_out_2, ref_out_2)
```

**原则**：
- API 返回几个 tensor 就解包几个，禁止丢弃或跳过
- Reference 始终按最完整模式计算（因为底层实际执行的是完整路径）
- 不同模式通过控制验证哪些输出来区分，而非控制是否执行用例
- 目标：0 skipped，所有枚举预算都用于实际验证

## DYNAMIC TensorList 注意事项

当 S5_mapping_spec.md 中存在 param_type=DYNAMIC 的 input/output 时，需额外注意：

**API 调用**：DYNAMIC 算子的 Python API 接受 `list[Tensor]` 输入，返回 `list[Tensor]` 输出（如 `torch._foreach_xxx(x_list)` 返回 `[Tensor, ...]`）。禁止对子 tensor 逐个调用 API（效率低且语义错误）。

**Reference 实现**：对 TensorList 输入逐子 tensor 计算，返回 `list[Tensor]`：
```python
# ✅ 正确：list comprehension 逐子 tensor 计算
ref_out = [reference_fn(t) for t in x_list]

# ❌ 错误：对 list 调用标量 API
ref_out = reference_fn(x_list)  # TypeError 或语义错误
```

**断言嵌套迭代**：DYNAMIC 输出为 `list[Tensor]`，需逐子 tensor 对比（见 01-code-templates.md「DYNAMIC 模式」）。禁止对 list 整体做 `assert_close`。

**空子 tensor 安全性**：空 tensor 变体（ID 含 `_empty` 后缀）的子 tensor shape 含 0 维度（如 `[0, 5]`）。`make_data` 对空 shape 返回空 tensor，`torch.randn([0, 5])` 正常工作。空 case 始终 `_data_range: "normal"`，不会触发 `with_inf`/`with_nan` 的 `view(-1)[0]` 越界（空 tensor 无元素可索引，但 `view(-1)[0]` 会抛 IndexError）。因此空 tensor 路径不经过 `with_inf`/`with_nan` 分支，安全。
