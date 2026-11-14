# custom_index_select（DSL 版）

torch.index_select 语义（按 dim 轴索引选择）。

**动态 shape 版**：
- 结构键 (data_dtype, idx_dtype, use64) 编译一次（三因子坐标分解与 rank
  无关），shape 变更**零重编译**
- **@host shape 派生**：wrapper 结构化 reshape (outer, gds, inner) /
  (outer, m, inner)，@host 从 Tensor shape 派生 inner/m/gds/total
  （cannir.extract_shape 运行期值）→ 运行期 blocks → 显式维度 view 展平；
  帧契约 3 槽零标量（原 7 槽），use64 决定派生值域（i64 / 窄化 i32）
- 视图 pack（inner 维 W 递降：整除 ∧ 首地址对齐；signed 归一位级别名进 kernel）
- 越界索引（含负值）→ 输出位级写零
- 支持dtype：18 种（11 直通 + bool/f64/complex64/complex32/fp8×3 位级别名）
- rank 0~8（0 维标量），dim 支持负值归一化
- **输入契约**：x/index 须为 NPU 张量且同设备（违反 → `ValueError`；
  wrapper 不做隐式搬移——对齐官方 example 的 device.type 校验惯例）

用法：
```python
from custom_index_select import custom_index_select
out = custom_index_select(x_npu, index_npu, dim=0)
```
