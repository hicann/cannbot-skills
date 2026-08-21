---
type: CATLASS DSL Debugging Guide
title: Runtime、DLPack 与正确性诊断
description: Launch、device、DLPack、数值 oracle 和 layout 不一致的排查顺序。
tags: [catlass-dsl, debug, runtime, dlpack, correctness, layout]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
sources:
  - id: runtime
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/catlass/runtime.py
    title: CATLASS DSL runtime
  - id: dlpack
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/tests/test_dlpack_bridge.py
    title: DLPack bridge tests
  - id: dlpack-runtime
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/catlass/tla/runtime.py
    title: DLPack bridge implementation
  - id: vadd
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/basic_vadd/basic_vadd.py
    title: VADD runtime and correctness harness
operator_families: [elementwise, matmul]
arch: [c310]
---

# 接口与概念

运行链将 host/device 对象转换为 kernel 参数并 launch；DLPack bridge 负责设备
tensor 的 dtype、shape、pointer 与生命周期转换。[^runtime][^dlpack]
Zero-copy binding、layout 转换和 CPU device 拒绝逻辑位于 DLPack runtime。[^dlpack-runtime]

# 用法

## 最小运行链

```python
dev_x = torch.arange(400, dtype=torch.float32, device="npu")
dev_z = torch.full_like(dev_x, float("nan"))
tla_x = tla.from_dlpack(dev_x, layout_tag=tla.arch.RowMajor)
tla_z = tla.from_dlpack(dev_z, layout_tag=tla.arch.RowMajor)

tla.initialize(device=0)
try:
    executor = tla.compile(
        kernel, tla_x, tla_z,
        arch_scope="aiv.c310",
        cache=True,
        cache_dir="./artifacts/runtime-cache",
    )
    executor(tla_x, tla_z, block_dim=1)
    torch.npu.synchronize()
finally:
    tla.finalize()
```

# 代码模式

## Oracle 与首个 mismatch

```python
expected = reference(dev_x)
matches = torch.isclose(dev_z, expected, rtol=1e-5, atol=1e-6, equal_nan=True)
if not bool(matches.all()):
    index = int((~matches).nonzero()[0])
    raise AssertionError(
        f"index={index} actual={dev_z[index].item()} "
        f"expected={expected[index].item()}"
    )
```

输出先填充 NaN 或不可能 sentinel，用于区分“计算错误”和“根本没有写回”。
整数 oracle 使用精确相等；atomic add 需要把输出初值计入 reference。[^vadd]

## Layout 诊断输入

对二维 tensor 使用 `value[m, n] = m * 1000 + n`，并选择非方形 shape，例如
`[3, 5]`。若结果转置、按固定间隔错位或每行重复，可分别定位 layout tag、
stride 或 tile coord。

# 约束

- device tensor 必须与 launch device 一致并在执行结束前存活。
- 动态 shape 的标量 extent 必须与实际 allocation 一致。
- oracle 必须覆盖 dtype 舍入、atomic 累加和 layout 的物理/逻辑解释。
- DLPack 是 zero-copy，原始 device tensor 在 launch 和同步完成前不能释放。
- CPU/NumPy DLPack producer 会被拒绝；device id 必须与 runtime 一致。

# 失败表现

| 表现 | 首查 |
| --- | --- |
| `does not implement __dlpack__` | 输入对象类型 |
| CPU/NumPy 不支持 | tensor 是否已搬到 NPU |
| `null strides` / shape metadata 错误 | DLPack producer 与 layout |
| `` `block_dim` must be an int `` | launch block 数不是整数 |
| 非法地址 | device、pointer 生命周期、extent |
| 全部仍为 sentinel | block_dim、GM 回写、同步 |
| 稳定转置/错位 | layout tag、stride、tile coord |
| 偶发 mismatch | flag/barrier、host synchronize |

# 验证方法

依次运行 identity、单 tile、非对称 shape、边界 tile 和完整 workload；记录
compile、launch、synchronize、oracle 为独立状态。本文未执行 runtime。

[^runtime]: 固定提交 runtime 参数转换与执行实现。
[^dlpack]: 固定提交 DLPack bridge 的契约测试。
[^dlpack-runtime]: 固定提交 DLPack tensor 的解析和 layout 转换。
[^vadd]: 固定提交 VADD 的 device tensor 与 reference 流程。
