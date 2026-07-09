# mul 算子（dav-3510 Reg 参考）

逐元素 `z = x * y`（float32）。**这是 Ascend950 / dav-3510 原生 `AscendC::Reg` 路径的最小参考算子**（单条 `Reg::Mul`）。

开发新的 dav-3510 Reg 算子时照抄本文件的三层结构：`__simd_vf__` 向量函数 → `__aicore__` 包装（`GetPhyAddr()` + `asc_vf_call`）→ kernel 类（`TPipe/TQue` + `DataCopyPad`，经典 AscendC 仅用于搬运/队列/同步）。

对比 `operators/add`、`operators/sqrt`：它们用**经典 AscendC 计算 API**（`AscendC::Add`/`Sqrt`），仅供 harness/CMake/test 结构参考；Reg 计算形态看本文件。

## 编译

```bash
mkdir -p build && cd build
cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..
make -j
```

## 测试

```bash
pytest test_mul.py -v
```
