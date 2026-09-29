# Host 约束校验：先检查，再使用

在分配资源、编译和启动之前，检查输入的类型、shape、dtype、device 与参数组合。只添加当前实现需要的约束，错误信息直接说明哪个参数不合法。

## 简单示例

下例只演示“二维 FP32、宽度为正”的检查顺序，适用于普通 Python eager 入口；这些条件是示例契约，不是通用 DSL 限制。

```python
import torch


def validate_input(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a Tensor")
    if x.ndim != 2:
        raise ValueError("x must have rank 2")
    if x.dtype != torch.float32:
        raise TypeError("x must be FP32")
    rows, columns = x.shape
    if columns <= 0:
        raise ValueError("x width must be positive")
    return rows, columns
```

按算子契约补充支持的设备、多个输入是否同设备、stride 和标量范围；明确空输入是返回空输出还是拒绝。符号 shape/Fake 路径使用目标环境支持的符号约束表达，不能直接照搬 Python 条件判断。

- `TensorSpec` 描述 dtype、维度关系与 stride；分组整除、可选输入依赖等语义条件仍需显式检查。
- 设备数据中的索引、长度等值，在约定的设备校验路径处理，并明确错误如何传回；不要默认用 `.item()` 或 `.cpu()` 引入同步。任务表边界见 [AICPU metadata](aicpu-load-balance-metadata.md)。
- 参数分类与启动核数的确定见 [编译与启动](compilation-launch.md)。

**反例：** 检查 rank 前读取 `shape[1]`；检查除数为正前做取模；用无条件 `.contiguous()` 隐藏不支持的布局。

**检查：** 合法输入正常通过；非 Tensor、错误 rank、错误 dtype、零宽度分别在使用相关数据前报错。按实际契约补充设备、布局和空输入边界。
