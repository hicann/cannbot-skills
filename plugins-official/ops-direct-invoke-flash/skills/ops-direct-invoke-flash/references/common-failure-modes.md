# 常见失败模式

## 故障排查模板

按以下格式将条目追加到 `docs/{OP}/plans/troubleshooting.md`：

```markdown
## [Phase N.M] 简要问题标题

**现象：** 出了什么问题。
**根因：** 底层原因。
**解决方案：** 改了什么、为什么这样改。
**经验：** 一句话总结，供后续算子参考。
**预防：** 应在工作流、检查清单或设计规则中改进什么。
```

## UB 拷贝路径

- 不要把 UB 到 UB 的 `DataCopy` 当作任意长度的 `memcpy`。
- 以字节而非仅以元素来推理。
- 显式处理 `32B` 块对齐。
- `32B` 对应 `16` 个 fp16 元素或 `8` 个 fp32 元素。
- 如果逻辑长度不是块对齐的，使用补齐、掩码或专门的尾块路径。
- 仅当算术语义可接受、且不要求按位保真的拷贝语义时，才将向量运算用作搬运的变通手段。
- 如果使用向量运算变通手段，须针对 `NaN`、`Inf`、有符号零、舍入和 dtype 转换验证其行为。
- 块大小和传输数量必须按 dtype 校验：`count * sizeof(T) >= 32B`。块大小 8 对 fp32（32B）有效，但对 fp16 只产生 16B——低于 DMA 最小值。

## device 侧辅助函数调用

- `__aicore__` 函数不能调用普通的 host 侧辅助函数——编译器会拒绝它们。
- 已知违规者（失败 3 次以上）：`ceil_div`、`align_down`、`align_up`。
- **务必**在核函数代码中用内联算术替换：例如用 `(a + b - 1) / b` 代替 `ceil_div(a, b)`。
- 在任何新核函数首次编译前，grep 源码中的 host 专用辅助函数名并替换它们。
- 命名空间放在哪里无关紧要——问题在于调用约定，而非可见性。

## 测试命名冲突

- 如果 `{OP}` 与标准库符号（如 `erf`、`sinh`、`cosh`、`exp` 或 `log`）冲突，应避免在 `.asc` 中使用顶层 `namespace {OP}`。
- host 入口应置于带前缀的 `namespace op_{OP}` 内（与 `TORCH_LIBRARY(op_{OP}, ...)` 注册一致），从而避免符号冲突。

## 构建新鲜度

- 如果改了源码但结果没变，先确认构建确实重新编译了改动的文件，并核对构建日志。
- 增量构建偶尔会因时间戳问题跳过重编。结果可疑时，删除 `build/` 目录后用 `cmake -DCMAKE_ASC_ARCHITECTURES=<arch> .. && make -j` 干净重建，再去追查更深层的逻辑 bug。

## 迭代累加精度

- 参考代码常使用迭代累加（每个分块 `acc += step`），其误差会随分块数量漂移。
- 在约 560 个分块（n=65536）时，fp32 舍入误差可能超过 `1e-3`——远高于标准测试容差。
- 对于线性递增或序列生成类算子，优先使用直接计算：`base + i * step` 代替 `acc += step`。
- 在阶段 3 分析中，将源码里任何迭代累加模式都视为精度风险。

## AscendC Cast 支持矩阵

- AscendC `Cast` 并不支持所有 dtype 组合。值得注意的缺口：不支持 `float -> int8`。
- int8 输出需要以 `half` 作为源类型。使用两步链：`float -> half`（CAST_NONE）再 `half -> int8`（CAST_ROUND）。
- 在假定存在直接路径之前，务必对照 AscendC Cast 类型支持矩阵核对每一步 Cast。
- 当输出 dtype 与计算 dtype 不同时，应在阶段 4 设计类型转换链，而不是当作阶段 6 的修补。

## TBuf 生命周期

- 对未经 `pipe.InitBuffer()` 初始化的缓冲区调用 `TBuf::Get<T>()` 属于未定义行为。
- 现象：在单核上正常，在多核派发时以 ACL error 507035 失败。
- 用与其 `pipe.InitBuffer()` 相同的 `if constexpr` 条件来保护每一次 `TBuf::Get()` 调用。
- 如果某缓冲区只在一条 dtype 路径上需要，则其分配和访问都必须放在该路径的编译期分支内。

## Ascend950 Reg API 漂移

- Ascend950 Reg 核函数不得使用 `AscendC::MicroAPI`、Membase，或除 `asc_vf_call` 以外的原始 `asc_*` API。
- 经典 AscendC 的计算/类型转换/规约调用，如 `AscendC::Mul`、`AscendC::Cast`、`AscendC::ReduceSum`、`AscendC::Sigmoid`、`AscendC::Sqrt`、`AscendC::Duplicate`、`AscendC::Adds` 和 `AscendC::Muls`，会破坏 Reg 原生路径。
- 使用 `__simd_vf__` Reg 包装函数进行向量计算，并从 `__aicore__` 包装函数中通过 `asc_vf_call` 调用它们。

## Reg 尾块掩码

- Reg 掩码是元素数量，而非字节数。
- 尾块存储必须使用与计算相同的逻辑尾块掩码。带掩码的计算后接不带掩码的存储，可能覆盖补齐后的 UB 内容。
- 对 B16 数据进行 float 计算通道时，使用 float 通道计数模式，除非目标样例证明存在另一种有效形态。

## Reg 标量规约

- 将标量规约结果存入 32B 槽位，例如 `reduceElems = 32 / sizeof(float)`。
- 对规约槽位上的标量算术使用 `UpdateMask<float>(1)`。
- 避免用 `LocalTensor::GetValue()` 把 Reg 产生的标量拉入 host 风格的标量代码。应在 Reg 包装函数中使用 `Div`、`Mul` 和 `Duplicate` 完成收尾和广播。

## Reg Cast Dist 模式

- B16 输入转 float 应使用 `LoadDist::DIST_UNPACK_B16`、`Cast<float, T, trait>` 和 `StoreDist::DIST_NORM_B32`。
- float 转 B16 输出应使用 `LoadDist::DIST_NORM`、`Cast<T, float, trait>` 和 `StoreDist::DIST_PACK_B32`。
- 缺失 `CastTrait` 字段或使用错误的 dist 模式是常见的编译或精度风险。
