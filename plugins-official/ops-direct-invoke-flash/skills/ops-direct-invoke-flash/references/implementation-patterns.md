# 实现模式

## 参考算子

在 `operators/` 中选取一个已完成的算子作为结构参考，参考其：

- 单文件 `.asc` 的分区方式（device kernel / tiling / host 入口 + `TORCH_LIBRARY` 注册）
- `CMakeLists.txt` 形态（`find_package(ASC)` + torch_npu，产出 `libop_{OP}.so`）
- `test_{OP}.py` 的组织方式
- `SHAPES` / `DTYPES` 用例矩阵结构
- 核函数入口与 host 分发结构

不要照搬某个具体算子的数学逻辑。复用的是结构，而非语义。

## 可编译骨架阶段

- 创建单文件 `{OP}.asc`：device kernel + tiling + host 入口（参数个数与算子 arity 匹配）+ `TORCH_LIBRARY` 注册的骨架；kernel 留空，host 入口返回空输出张量。
- 创建 `CMakeLists.txt`：`find_package(ASC)`、`project(op_{OP} LANGUAGES ASC CXX)`、链接 torch_npu，产出 `libop_{OP}.so`。
- 从骨架阶段起就配置好 ASC 语言与 `--npu-arch` 编译选项；不要推迟到核函数实现时才加上。

## 定义文档要求

需记录：

- 公式
- 输入/输出含义
- dtype 策略
- 边界场景
- CPU 参考伪代码

实现应遵循该文档，而不是先在代码里臆造数学逻辑。

## 迭代累加模式分析

当源参考使用迭代累加（例如在按 chunk 的循环内 `acc += step`）时，需评估浮点舍入误差是否会随迭代次数增长。该模式可能出现在任何参考实现中。对于序列生成类和线性递增类算子，累计误差与 chunk 数量成正比，在大数组上可能超出 fp32 测试容差（例如约 560 次迭代后 max_diff > 0.001）。

优先采用逐元素直接计算：
- 不要每次迭代都 `workUnit += workStep`，而是计算 `base + globalIdx * step`。
- 无论元素在数组中的位置如何，这都将误差限定在一次乘加运算之内。

在阶段 3 的源分析中标记出任何迭代累加模式，并在定义文档中记录决策（迭代 vs. 直接）。

## 设计文档要求

需记录：

- 计算策略
- UB 缓冲区清单
- `liveBytesPerElem`
- 切分公式
- 向量指令序列

如果算子使用了 local 拷贝路径，需显式记录字节数、对齐假设与尾块处理。

## 测试套件要求

- 从一个接口存在性占位测试开始（断言算子注册到 `torch.ops.op_{OP}`）。
- 以 torch 作为 CPU 参考；在核函数实现之前先写好参考比对逻辑。
- 加入以 `@pytest.mark.parametrize` 对 `(shape, dtype)` 参数化的 NPU 测试，并用 `@pytest.mark.skipif(not torch.npu.is_available())` 守卫。
- 除非某个 case 明确针对边界或失败行为，否则生成的输入应保持在算子的有效定义域内。

## 核函数实现

按以下顺序实现：

1. device kernel `{OP}_kernel<T>()`（按 tile 处理）
2. tiling 函数 `calc_{OP}_tiling_params()`
3. `namespace op_{OP}` 中的 host 入口（按 dtype 分发并 launch 核函数）
4. `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL(op_{OP}, PrivateUse1, m)` 注册

每完成一个有意义的步骤后：

- 构建
- 运行测试
- 与定义文档和设计文档比对

## Ascend950 Reg API 模式

对于 Ascend950 / `dav-3510`，向量计算、类型转换（Cast）与规约应使用 `AscendC::Reg` 编写。详细参考见 [reg-api-guide.md](reg-api-guide.md)。

默认 wrapper 形态：

```cpp
__simd_vf__ inline void OpRegVf(__ubuf__ float *dstAddr, __ubuf__ float *srcAddr,
                                uint32_t count, uint32_t chunkCount)
{
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::RegTensor<float> dstReg;
    AscendC::Reg::RegTensor<float> srcReg;

    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t offset = chunk * vectorLength;
        uint32_t remaining = count - offset;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(srcReg, srcAddr + offset);
        // AscendC::Reg compute.
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(dstAddr + offset, dstReg, mask);
    }
}

__aicore__ inline void OpReg(const AscendC::LocalTensor<float> &dst,
                             const AscendC::LocalTensor<float> &src,
                             int64_t count)
{
    __ubuf__ float *dstAddr = (__ubuf__ float *)dst.GetPhyAddr();
    __ubuf__ float *srcAddr = (__ubuf__ float *)src.GetPhyAddr();
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t elementCount = static_cast<uint32_t>(count);
    uint32_t chunkCount = (elementCount + vectorLength - 1) / vectorLength;
    asc_vf_call<OpRegVf>(dstAddr, srcAddr, elementCount, chunkCount);
}
```

Reg 规约的标量应使用 32B 槽位：

```cpp
constexpr uint32_t reduceElems = 32 / sizeof(float);
AscendC::Reg::MaskReg scalarMask = AscendC::Reg::UpdateMask<float>(1);
```

将第 `i` 行存储到 `dstAddr + i * reduceElems`。对于 Reg 规约产生的值，避免使用 `LocalTensor::GetValue()`；将标量收尾与广播都保留在 Reg wrapper 内部。

Reg sigmoid 应分解为 `Muls(-1) -> Exp -> Adds(1) -> Div(1, denom)`。在 Ascend950 Reg 路径中不要调用经典的 `AscendC::Sigmoid`。

B16 类型转换（Cast）应使用 `AscendC::Reg::CastTrait`，B16 输入用 `LoadDist::DIST_UNPACK_B16`，B16 输出用 `StoreDist::DIST_PACK_B32`。对于 float 计算路径，保持 `VECTOR_REG_WIDTH / sizeof(float)` 的 lane 逻辑，除非有本地范例证明存在另一种合法形态。

## 关键约定

- `DataCopyPad` 的 `blockLen` 以字节计，而非元素数。
- `PipeBarrier<PIPE_V>()` 应遵循真依赖与冒险顺序；在每个向量操作后都加一个虽然安全，但偏保守。
- UB 到 UB 的 `DataCopy` 并不是任意长度的 `memcpy`。
- 在计算 DMA 块大小或传输元素数时，对每一种受支持的 dtype 评估 `count * sizeof(T)`。某个元素数对 `float` 满足 32B DMA 最小值（8 个元素），但对 `half` 可能低于最小值（需要 16 个元素）。以 `32 / sizeof(T)` 作为最小元素数。
- 在 `CMakeLists.txt` 中，`find_package(ASC REQUIRED)` 必须在 `project(op_{OP} LANGUAGES ASC CXX)` 之前调用。
- device kernel 模板在 host 入口处按 dtype 实例化（如 `{OP}_kernel<float>` / `<half>` / `<int32_t>`），由 dtype 分发驱动，无需单独的显式实例化声明。
- 在 host 分发代码中使用 RAII 管理 device 侧内存。
- 在 Ascend950 Reg 路径中，`asc_vf_call` 是唯一允许的原始 `asc_*` 调用。
- 在 Ascend950 Reg 路径中，经典 AscendC 的计算/类型转换/规约调用应替换为 `AscendC::Reg` wrapper。

## 类型转换（Cast）支持矩阵

AscendC 的 `Cast` 并不支持所有源到目标的类型对。特别是：

- `float -> int8_t` **不受支持**。请使用两步链：`float -> half`（CAST_NONE），再 `half -> int8_t`（CAST_ROUND）。
- 在假定存在直接路径之前，务必对照 AscendC 文档核对每一步 Cast。

当输出 dtype 与计算 dtype 不同时，在设计阶段（阶段 4）规划 Cast 链，并在 UB 缓冲区清单中记录每个中间缓冲区。

## TBuf 生命周期规则

- 每个 `TBuf::Get<T>()` 调用都必须由与对应 `pipe.InitBuffer()` 相同的 `if constexpr` 条件守护。
- 对未初始化的 `TBuf` 调用 `Get()` 是未定义行为。它在单核测试中可能看似正常，但在多核分发下会破坏 pipe 管理（ACL error 507035）。
- 当某缓冲区仅在特定 dtype 路径下需要时，`InitBuffer()` 和所有 `Get()` 调用都必须位于同一个编译期守护内。
