# Ascend950 Reg API 指南

在生成或重构 Ascend950 / `dav-3510` 算子时，请参考本指南。

## 策略

- 向量计算、类型转换（Cast）与规约代码使用 `AscendC::Reg`。
- 不要使用 `AscendC::MicroAPI`。
- 不要使用 Membase API、shim、兼容层或回退方案。
- 除 `asc_vf_call` 外，不要使用裸 `asc_*` C API。
- 经典 AscendC 张量 API 仅作集成与数据搬运辅助使用。`AscendC::TPipe`、`TQue`、`TBuf`、`GlobalTensor`、`LocalTensor`、`DataCopyPad`、`DataCopy` 与 `PipeBarrier` 允许用于队列、buffer、DMA 与顺序控制；计算 / 类型转换（Cast）/ 规约应通过 Reg 封装完成。
- **可配置的 VF 融合上限：** 默认情况下，对一个 `__simd_vf__` 函数内部融合的 VF 计算指令数量没有硬性上限。如果用户指定了融合上限 `N`（通过 skill 请求、`STATE.md` 或设计文档），则每个 `__simd_vf__` 最多融合 `N` 条 VF 计算指令（load/store 与掩码设置不计入）；超出的较长指令链应拆分为多个 `__simd_vf__` 封装，分别通过独立的 `asc_vf_call` 调用串联，在存在数据依赖时用 `PipeBarrier<PIPE_V>()` 隔开。原因：较长的融合链会增大寄存器压力，并在 `dav-3510` 上劣化编译器调度；合理的上限取决于算法与调优目标，因此该上限交由用户配置，而非硬编码。

## 标准代码结构

Reg 计算应采用三层结构：

1. 一个 `__simd_vf__` 函数，接收 `__ubuf__` 指针，声明 `AscendC::Reg::RegTensor<T>` 与 `AscendC::Reg::MaskReg`，加载寄存器、计算、并存储寄存器。
2. 一个 `__aicore__` 封装，接收 `AscendC::LocalTensor` 值，通过 `GetPhyAddr()` 转换它们，计算 `uint32_t` 的元素数 / 分块数，并调用 `asc_vf_call<...>()`。
3. 核函数类或 harness 层，负责 GM/UB 搬运、切分、队列以及 `PipeBarrier<PIPE_V>()`。

默认分块循环：

```cpp
constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
uint32_t elementCount = static_cast<uint32_t>(count);
uint32_t chunkCount = (elementCount + vectorLength - 1) / vectorLength;
```

向量函数内部：

```cpp
for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
    uint32_t offset = chunk * vectorLength;
    uint32_t remaining = count - offset;
    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(srcReg, srcAddr + offset);
    // Reg compute here.
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(dstAddr + offset, dstReg, mask);
}
```

掩码以元素数为单位，而非字节数。尾块的 load、计算与 store 都必须使用尾块掩码。

## 允许的 Reg 模式

在自创新模式之前，优先使用下列已验证的模式：

- 类型：`AscendC::Reg::RegTensor<T>`、`AscendC::Reg::MaskReg`、`AscendC::Reg::CastTrait`。
- 掩码：`AscendC::Reg::CreateMask`、`AscendC::Reg::UpdateMask`。
- Load/store：`LoadAlign`、`LoadUnAlignPre`、`LoadUnAlign`、`StoreAlign`。
- 计算：`Duplicate`、`Add`、`Adds`、`Sub`、`Mul`、`Muls`、`Div`、`Max`、`Exp`、`Sqrt`。
- 规约：`AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM, ...>`。
- 类型转换（Cast）：`AscendC::Reg::Cast<dst_t, src_t, trait>`。

当存在本地的目标范例时，沿用其 include 风格、launch 结构、切分结构体与文件布局。不要把旧算子中任何非 Reg 的计算路径拷贝进 Ascend950 Reg 代码。

## 常用模板

标量规约的输出应占用一个 32B 槽位：

```cpp
constexpr uint32_t reduceElems = 32 / sizeof(float);
AscendC::Reg::MaskReg scalarMask = AscendC::Reg::UpdateMask<float>(1);
// Store row i's scalar at dstAddr + i * reduceElems.
```

不要通过 `LocalTensor::GetValue()` 把规约后的标量值取回，再从 C++ 标量数组进行广播。应将收尾计算保留在 Reg 中：加载标量槽位，在 `scalarMask` 下做除法或乘法，然后用 `Duplicate(..., regScalar, fullMask)` 进行广播。

Sigmoid 应拆解为 Reg 操作：

```cpp
AscendC::Reg::Muls(denomReg, srcReg, -1.0f, mask);
AscendC::Reg::Exp(denomReg, denomReg, mask);
AscendC::Reg::Adds(denomReg, denomReg, 1.0f, mask);
AscendC::Reg::Duplicate<float>(oneReg, 1.0f, mask);
AscendC::Reg::Div(dstReg, oneReg, denomReg, mask);
```

B16 输入转 float：

```cpp
static constexpr AscendC::Reg::CastTrait trait = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + offset);
AscendC::Reg::Cast<float, T, trait>(dstReg, srcReg, mask);
AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(dstAddr + offset, dstReg, mask);
```

Float 转 B16 输出：

```cpp
static constexpr AscendC::Reg::CastTrait trait = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(srcReg, srcAddr + offset);
AscendC::Reg::Cast<T, float, trait>(dstReg, srcReg, mask);
AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(dstAddr + offset, dstReg, mask);
```

当后一个 Reg 封装要消费前一个封装产生的数据时，在两者之间使用 `PipeBarrier<PIPE_V>()`。

## 评审清单

- 除 `asc_vf_call` 外，不存在 `AscendC::MicroAPI`、Membase 或裸 `asc_*` 调用。
- Ascend950 Reg 路径中不存在经典 AscendC 的计算 / 类型转换（Cast）/ 规约调用。
- 每个向量函数都有显式的掩码与尾块处理。
- 每个 Reg 封装都通过 `GetPhyAddr()` 把 `LocalTensor` 转换为 `__ubuf__`。
- 类型转换（Cast）路径指定了 `CastTrait`、`LoadDist` 与 `StoreDist`。
- 规约标量使用 32B 槽位与 `scalarMask`。
- 较大的逻辑张量已切分，使每次 Reg 调用的元素数都能放进 `uint32_t`。
- 若已为该算子指定 VF 融合上限 `N`，则每个 `__simd_vf__` 函数融合的 VF 计算指令数 ≤ `N`；否则跳过此项检查。
