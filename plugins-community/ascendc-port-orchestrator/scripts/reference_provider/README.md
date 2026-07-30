# reference_provider — 跨代际移植与反向生成真值工具

本目录只服务插件的两个模式：

- `port_a3_to_a5`：当前承载 arch22→arch35 移植；输入由 `case_gen.py` 生成，真值由来源架构算子实测产生。
- `backward`：输入由 `backward_input_gen.py` 生成，真值由
  `autograd_backward_reference.py` 在 CPU/fp64 上计算。

## 主要文件

| 文件 | 用途 |
|---|---|
| `case_gen.py` | 生成可复现、分层覆盖的算子输入。 |
| `input_gen.template.py` | 跨代际移植参考输入模板。 |
| `backward_input_gen.py` | 生成反向算子的代表性与边界输入。 |
| `autograd_backward_reference.py` | 用 `torch.autograd.grad` 计算 CPU/fp64 梯度真值。 |
| `backward_cpu_truth.template.py` | 声明式反向真值模板。 |
| `verify.py` | 在目标 NPU 上验证生成的 AscendC 算子。 |

## 真值边界

1. 跨代际移植必须记录来源架构的参考产物和来源信息；不得用目标实现自证。
2. 反向生成必须从可微正向规格和 `BACKWARD_SPEC` 计算 CPU/fp64 真值。
3. 两个模式均须保留输入数据哈希、覆盖层级和逐 case 结果，确保可复现和可审计。
