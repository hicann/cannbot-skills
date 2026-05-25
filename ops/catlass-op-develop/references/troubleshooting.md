# Troubleshooting — 常见问题排查

| 现象 | 可能原因 | 排查方法 |
|------|---------|---------|
| 编译找不到 BlockMmad 特化 | DispatchPolicy 与 ArchTag 不匹配 | 确认 `dispatch_policy.hpp` 中定义 |
| Epilogue 模板参数不匹配 | BlockEpilogue 槽位签名错误 | `rg "template.*Epilogue" catlass/include/catlass/epilogue/block/` 确认 |
| DeviceGemm 在 op_kernel 中报错 | 用了 host 侧适配器 | 只能用 `Kernel{}(params)` |
| L0C 容量超限 | CType 用 fp32 时 `m0*n0*4 > L0CSize` | 减小 L0TileShape 的 M/N |
| GELU/SILU 激活无用 | 用了 BasicMatmul 而不是 MatmulActivation | 有 Epilogue → MatmulActivation |
| 精度差 | CType 用 half 而非 float | 累加用 fp32 (`GemmType<float, ...>`) |
| 编译找不到 Epilogue 头文件 | include 路径或命名空间错误 | Epilogue 在 `catlass/include/catlass/epilogue/`，不在 `gemm/` |
| ComputeType 与 Tile 模板不匹配 | DType 的 Element 与 Tile 要求不一致 | 检查 `TileElemWise*` 的 `ComputeType_` 与 `CType` 匹配 |
| bias / scale tensor 越界 | UB 布局与 BIAS/FB 区域重叠 | 确认 catlass ArchTag 的 BIAS_SIZE / FIXBUF_SIZE |
| 编译器含 catlass 的内部 static_assert 报错 | Tile/Block 组件选择与 kernel 类型不兼容 | 查看报错信息，通常有详细说明 |
