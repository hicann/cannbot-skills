# conv3dtranspose — Conv3DTranspose 参考实现

本示例提供转置卷积的 Tiling、Kernel 和 TilingData。计算路径与 [conv3ddx](../conv3ddx/README.md) 对齐，加入 output_padding、目标形状及可选 bias；直调与 ACLNN 工程复用本目录源码。

## 1 示例范围

| 项目 | 实现内容 |
|---|---|
| 数值绑定 | x → dedy、filter → weight、y → dedx；数据同为 FP16 或 BF16 |
| 布局 | `fmt` 统一指定 x、filter、y 的 NCDHW 或 NDHWC 布局 |
| 形状 | 逻辑 3D；目标形状结合 output_padding 与形状余量 rho 校验 |
| output_padding | `qd/qh/qw` 为非负空间分量，每轴满足 `q < stride` 或 `q < dilation` |
| bias | 可选一维 FP32 向量，长度为输出通道数 `cin`；通过 BT 参与首次 MMAD |
| Tiling | small-shape、standard、l1-resident，包含 bias 驻留判定 |
| 权重准备 | `loadB2=1/2/3`：转置加逆序、仅逆序、仅转置 |
| 写回 | 有贡献块正常写回；无贡献块经 `ComputeZero/ComputeForBias` 写零或 bias |

本表描述本示例源码的实现范围；各分支知识由 [input 索引](../../input/index.md) 组织。

本示例的 loadB2=1/2/3、key 和 TilingData 是配套的私有 ABI；3 表示单位核仅转置。新生成工程的三实参声明对该场景使用 loadB2Condition=1，见 [模板绑定](../../input/tilingkey-templates.md)。复用本示例时保留其 Host/Kernel/共享头和分发配套，不能只替换 key。

## 2 文件组织

```text
conv3dtranspose/
+-- README.md
+-- conv3dtranspose_tiling.h        Host 规格、求解结果与求解接口
+-- conv3dtranspose_tiling.cpp      形状校验、三种 Tiling 策略与 bias 驻留
+-- conv3dtranspose_tiling_data.h   Host/Device 共用字段与资源约束
`-- conv3dtranspose_kernel.cpp      Cube 主通路、BT 通路与设备入口
```

## 3 参数传递

1. 按逻辑 NCDHW 轴义填写 `Conv3DtSpec`，包含目标形状、`qd/qh/qw` 和 `hasBias/biasElems`。外部格式、bias 的 dtype 与维度由调用侧核实。
2. 调用 `Conv3DtSolveTiling`。返回 0 后，`Conv3DtTiling` 提供启动核数、模板分发信息及参数负载。
3. 将 `result.payload` 中的 `sizeof(Conv3DtTilingData)` 字节拷入设备 TilingData 缓冲区；Kernel 的 `tiling` 指向负载起点。
4. 按数据类型与 `loadB2` 实例化 `conv3dt_entry<T, LOADB2>`，以 `blockDim` 启动。bias 指针与 `hasBiasFlag` 保持一致；无 bias 时传空指针。设备入口按 AIC-only 核型接入。

Host 将 q/rho 折入目标形状和反向补边，Kernel 按统一窗口规则计算。分发信息留在 Host；资源查询与平台库 ABI 的接入要求同 conv3ddx。

## 4 代码与知识对应

| 代码 | 知识 |
|---|---|
| q/rho、反向补边与初始化标志 | [前置处理](../../../../generic/preprocessing.md) |
| 三种 Tiling 策略与共享字段 | [Tiling](../../../../generic/input/tiling/tiling.md) |
| bias 驻留、`LoadBiasToBT` 与首次 MMAD | [bias BT 通路](../../input/bias-bt-path.md) |
| 两路搬运与输出寻址 | [指令映射](../../input/instruction-mapping.md) |
| `ComputeZero/ComputeForBias` 与事件收尾 | [同步实现](../../input/sync-implementation.md) |

接入后按 [验证设计](../../../../generic/input/verification.md) 检查目标形状、q/rho、bias 广播与尾块，并用非零预填验证无贡献块和重复执行。直调接入使用 `ascendc-direct-invoke-template`，ACLNN 接入使用 `ascendc-registry-invoke-template`。
