# conv3ddw — Conv3DBackpropFilter 参考实现

本示例提供权重梯度算子的 Tiling、Kernel 和 TilingData，采用 MN_StreamK 计算路径。直调与 ACLNN 工程复用同一份实现，启动、注册和构建由对应工作流接入。

## 1 示例范围

| 项目 | 实现内容 |
|---|---|
| 数值绑定 | x、dedy 同为 FP16 或 BF16；dedw 为 FP32，保持正向核坐标顺序 |
| 布局 | x/dedy 配对 NCDHW 或 NDHWC；dedw 选 NCDHW、NDHWC 或 DHWCN |
| 规格与属性 | 逻辑 3D，groups=1；1D/2D 先按前置处理规则升维 |
| Tiling | MN_StreamK；基本块、L1 父段与小任务调整，选择每个输出任务归约完整 K 的计划 |
| 搬运 | 普通 H/W 窗口、完整宽度、baseN 为完整通道窗口 |
| 归约 | 输出任务固定 co/ci/kd，归约 batch/dout/ho/wo，按 65536 阈值分段 |
| 初始化与写回 | MIX AIV 清零并通知 AIC；AIC 计算后以 FP32 Fixpipe 原子累加 |

本表描述本示例源码的实现范围；窗口、切宽及其他执行分支由 [filter 索引](../../filter/index.md) 组织。最终参数通过 Host 的形状、L0/L1 与 API 约束检查。

## 2 文件组织

```text
conv3ddw/
+-- README.md
+-- conv3ddw_tiling.h        Host 规格、平台能力与求解接口
+-- conv3ddw_tiling.cpp      参数校验、平台查询与 MN_StreamK 求解
+-- conv3ddw_tiling_data.h   Host/Device 共用字段
+-- conv3ddw_kernel.h        搬运、乘加、初始化、写回与同步
`-- conv3ddw_kernel.cpp      MIX 设备入口
```

## 3 参数传递

1. 按逻辑 NCDHW 轴义填写 `Conv3DdwSpec`。`xFormat` 指定 x/dedy 布局，`yFormat` 指定 dedw 布局；filter_size 在 Host 解析为核尺寸和目标通道。
2. 调用 `Conv3DdwSolveTiling`；已有平台能力时可调用 `Conv3DdwSolveTilingWithCaps`。返回 0 后得到 `Conv3DdwTilingData`，失败原因写入 error 缓冲区。
3. 将完整 TilingData 拷入设备参数缓冲区，按输入数据类型实例化 `conv3ddw_entry<T>`，以 `usedCoreNum` 启动。
4. 设备入口保持 `KERNEL_TYPE_MIX_AIC_1_2` 与 `__schedmode__(1)`。所有参与单元先执行初始化协议，再允许无任务 AIC 返回。

`Bound` 字段单位为元素，`*Bytes` 为字节。`cin1G/cout1G` 在本示例表示实际组内通道数。L1 按父段 Bound 分配，L0 按基本块及 buffer 份数分配；队列和事件交接与所选 SDK 配对。

## 4 代码与知识对应

| 代码 | 知识 |
|---|---|
| `CheckAxis` 与规格校验 | [前置处理](../../../../generic/preprocessing.md) |
| `Solver/SetStepK4SplitMN/AdjustSmallCaseBaseBlock` | [MN_StreamK Tiling](../../../../generic/filter/tiling/templates/mn-streamk/tiling.md) |
| `LoadToA1/LoadToB1/CopyInMte1` | [指令映射](../../filter/instruction-mapping.md) |
| `ProcessTask/ComputeDepth/CopyOut` | [计算流程](../../../../generic/filter/compute-flow.md) |
| `InitializeOutput`、队列和事件 | [同步实现](../../filter/sync-implementation.md) |

接入后按 [验证设计](../../../../generic/filter/verification.md) 检查输入量化、独立权重梯度 Golden、通道及 K 尾块、分段累加与重复初始化。直调接入使用 `ascendc-direct-invoke-template`，ACLNN 接入使用 `ascendc-registry-invoke-template`。
