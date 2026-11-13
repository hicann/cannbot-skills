# conv3ddw_direct — 卷积权重梯度直调示例

本示例将 [filter 范式](../../filter/index.md) 落地为自包含的 Host 求解器、共享参数、Cube Kernel 和 `<<<>>>` 启动接口。代码组织和注释沿用 conv3ddx 示例；数学绑定为 **x、dedy → FP32 dedw**，保持正向核坐标顺序。

## 1 规格与计算路径

| 项目 | 示例契约 |
|---|---|
| 输入/输出 | x、dedy 同类型 FP16 或 BF16；dedw 为 FP32 |
| 布局 | x/dedy 配对 NCDHW 或 NDHWC；dedw 独立选 NCDHW、NDHWC、DHWCN |
| 形状 | 统一五维规格；已升维的问题保留原有 D/H/W 轴义 |
| 属性 | groups=1、普通正向权重梯度；output_padding=0、enable_hf32=false |
| 求解 | MN_StreamK；Host 求解基本块、L1 父段及小任务调整，接收自然无跨核 K 归约的计划 |
| 搬运 | 普通 H/W 窗口、完整宽度、baseN 为完整通道窗口；最终参数通过 L0/L1 与 API 检查 |
| 归约 | 输出任务固定 co/ci/kd；完整归约 batch/dout/ho/wo，按 65536 阈值分段 |
| 初始化/写回 | MIX AIV 清零 → 汇合通知 → AIC 计算 → FP32 Fixpipe 原子累加；每次调用重新初始化 |

范式中的拆核、切宽、非整窗等知识由相应生成实现展开。本示例选择上述一条完整执行路径，Host 对需要其他路径的规格明确返回错误。源码中没有逐 case 的形状或 Tiling 查询表。

## 2 文件组织

```text
conv3ddw_direct/
+-- README.md
`-- op_kernel/
    +-- conv3ddw_launch.h          Host 逻辑规格、资源和两种 dtype 的 ABI
    +-- conv3ddw_tiling_data.h     Host/Device 共用参数；Bound 单位为元素
    +-- conv3ddw_tiling.cpp        校验、平台查询与纯 Host 求解
    +-- conv3ddw_kernel.h          两路搬运、计算、写回、初始化与同步
    `-- conv3ddw_kernel.cpp        MIX 入口与两个 launch 包装
```

`Conv3DdwSpec` 按逻辑 NCDHW 轴义填写。外部格式只影响寻址，不能用 reshape 代替格式转换。filter_size 在 Host 解析为核尺寸与目标通道；设备入口保留对应参数槽，数值计算只消费 x、dedy。直调将完整 `Conv3DdwTilingData` 拷入 GM 后传给 launch。

L1 队列按 Bound 分配两份父段，L0A/L0B 使用所选 SDK 的两个物理半区，L0C 按最终基本块与份数分配。当前 SDK 的 TPipe 管理固定 M_MTE1 事件的初始令牌和末次等待；生成类只管理轮内交接。

## 3 构建与调用

本目录保存算子实现与直调 ABI，编译和链接由调用工程接入。设备源与 Host 求解器分别编译，再链接到宿主程序：

```bash
source <CANN>/set_env.sh
mkdir -p <build>
bisheng -std=c++17 --npu-arch=<target> -xasc -fPIC \
    -I<CANN>/compiler/ascendc/include -I<CANN>/include -Iop_kernel \
    -c op_kernel/conv3ddw_kernel.cpp -o <build>/conv3ddw_kernel.o
g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC \
    -I<CANN>/include -Iop_kernel \
    -c op_kernel/conv3ddw_tiling.cpp -o <build>/conv3ddw_tiling.o
bisheng <caller_main.cpp> <build>/conv3ddw_kernel.o <build>/conv3ddw_tiling.o \
    -I<CANN>/include -Iop_kernel -L<CANN>/lib64 \
    -lascendcl -lruntime -lplatform -lregister -ltiling_api -lstdc++ -lm -ldl \
    -o <build>/test_conv3ddw
```

设备源由 bisheng `-xasc --npu-arch=<target>` 编译。Host 平台查询源使用 `_GLIBCXX_USE_CXX11_ABI=0`，与 libplatform 接口配对。MIX 入口声明 `__schedmode__(1)`；运行时从 platformInfo 查询资源，核数和容量不作为几何查表常量。

```cpp
Conv3DdwTilingData td{};
char error[256]{};
if (Conv3DdwSolveTiling(&spec,&td,error,sizeof(error))!=0) return Error(error);
CopyHostToDevice(tilingGm,&td,sizeof(td));
if (spec.dtype==CONV3DDW_FP16)
    LaunchConv3DdwFp16(td.usedCoreNum,stream,x,nullptr,dedy,dedw,nullptr,tilingGm);
else
    LaunchConv3DdwBf16(td.usedCoreNum,stream,x,nullptr,dedy,dedw,nullptr,tilingGm);
```

上面的错误处理和内存拷贝函数由调用工程提供。主计算不调用内置卷积、外部算子仓代码或 CPU 回退。

## 4 代码与知识对应

| 实现 | 对应知识 |
|---|---|
| CheckAxis / 规格校验 | [preprocessing §8](../../../../generic/preprocessing.md#8-filter-分支的规格归一) |
| Solver / SetStepK4SplitMN / AdjustSmallCaseBaseBlock | [MN_StreamK Tiling](../../../../generic/filter/tiling/templates/mn-streamk/tiling.md) |
| LoadToA1 / LoadToB1 / CopyInMte1 | [指令映射](../../filter/instruction-mapping.md) |
| ProcessTask / ComputeDepth / CopyOut | [计算流程](../../../../generic/filter/compute-flow.md) |
| InitializeOutput / 队列和事件 | [同步实现](../../filter/sync-implementation.md) |
