# conv3ddx — Conv3DBackpropInput 参考实现

本示例提供输入梯度算子的 Tiling、Kernel 和 TilingData。直调与 ACLNN 工程复用同一份计算实现，启动、注册和构建由对应工作流接入。

## 1 示例范围

| 项目 | 实现内容 |
|---|---|
| 数值绑定 | weight、dedy → dedx；三者同为 FP16 或 BF16 |
| 布局 | `fmt` 统一指定 weight、dedy、dedx 的 NCDHW 或 NDHWC 布局 |
| 规格 | 逻辑 3D 形状；1D/2D 先按前置处理规则升维 |
| 属性 | 逐轴 stride、dilation、两侧 pad 和 groups；最终通过形状、指令与资源校验 |
| Tiling | small-shape、standard、l1-resident，包含容量回调与多核任务划分 |
| 权重准备 | `loadB2=1/2/3`：转置加逆序、仅逆序、仅转置 |
| 写回 | 有贡献块经 Fixpipe 写回，无贡献块由 `StoreZeroBlock` 覆盖有效输出 |

本表描述本示例源码的实现范围；生成其他分支时按 [input 索引](../../input/index.md) 读取对应知识。

本示例的 loadB2=1/2/3、key 和 TilingData 是配套的私有 ABI；3 表示单位核仅转置。新生成工程的三实参声明对该场景使用 loadB2Condition=1，见 [模板绑定](../../input/tilingkey-templates.md)。复用本示例时保留其 Host/Kernel/共享头和分发配套，不能只替换 key。

## 2 文件组织

```text
conv3ddx/
+-- README.md
+-- conv3ddx_tiling.h        Host 规格、求解结果与求解接口
+-- conv3ddx_tiling.cpp      参数校验、平台查询与三种 Tiling 策略
+-- conv3ddx_tiling_data.h   Host/Device 共用字段、粒度与资源约束
`-- conv3ddx_kernel.cpp      两路搬运、乘加、写回与设备入口
```

## 3 参数传递

1. 按逻辑 NCDHW 轴义填写 `Conv3DdxSpec`。外部 `input_size` 的轴序转换在调用侧完成。
2. 调用 `Conv3DdxSolveTiling`。返回 0 后，`Conv3DdxTiling` 提供 `blockDim、tilingKey、templateId、loadB2` 和参数负载。
3. 将 `result.payload` 中的 `sizeof(Conv3DdxTilingData)` 字节拷入设备 TilingData 缓冲区；Kernel 的 `tiling` 指向负载起点。
4. 按数据类型与 `loadB2` 实例化 `conv3ddx_entry<T, LOADB2>`，以 `blockDim` 启动。设备入口按 AIC-only 核型接入，完整输出写回由 Kernel 完成。

Host 分发信息留在 Host，Kernel 只消费 TilingData。资源由平台接口查询，并与共享头中计算布局及 buffer 的约束核对；平台库的 C++ ABI 配置由调用工程保持一致。

## 4 代码与知识对应

| 代码 | 知识 |
|---|---|
| `ValidateAndNormalize` | [前置处理](../../../../generic/preprocessing.md) |
| 三种策略、容量回调与任务参数 | [Tiling](../../../../generic/input/tiling/tiling.md) |
| `EnsureA1/EnsureB1`、L0 装载与 Fixpipe | [指令映射](../../input/instruction-mapping.md) |
| 任务循环、窗口、权重逆序与输出寻址 | [计算流程](../../../../generic/input/compute-flow.md) |
| 队列与事件交接 | [同步实现](../../input/sync-implementation.md) |

接入后按 [验证设计](../../../../generic/input/verification.md) 检查模板与权重分支、分组寻址、尾块、负反向边界和非零预填下的重复执行。直调接入使用 `ascendc-direct-invoke-template`，ACLNN 接入使用 `ascendc-registry-invoke-template`。
