# conv3ddw_aclnn — 卷积权重梯度 ACLNN 注册示例

本示例复用 [直调示例](../conv3ddw_direct/README.md) 的纯 Host 求解器和 Kernel 主体，提供 OpDef、值依赖推导、TilingFunc、MIX 注册入口及手写 ACLNN L2/L0 接口。数值类型、规格和执行路径与直调示例一致。

## 1 文件组织

```text
conv3ddw_aclnn/
+-- op_graph/
|   `-- conv3ddw_proto.h              算子原型；运行包由 OpDef 生成注册信息
+-- op_host/
|   +-- conv3ddw_def.cpp                   OpDef 注册
|   +-- Conv3ddw.h                         输入输出、属性和平台配置
|   +-- conv3ddw_infershape.cpp       filter_size 元素值推导形状；输出 FP32
|   `-- arch35/
|       +-- conv3ddw_solver.cpp       与直调一致的纯 Host 求解
|       `-- conv3ddw_tiling.cpp       context 规格、平台查询、字段与启动发布
+-- op_kernel/
|   +-- conv3ddw.cpp                  注册入口；模板绑定 MIX_AIC_1_2
|   `-- arch35/
|       +-- conv3ddw_struct.h         模板参数与 key
|       +-- conv3ddw_tiling_data.h    共享参数
|       +-- conv3ddw_launch.h         规格与求解声明
|       `-- conv3ddw_kernel.h         与直调一致的计算主体
`-- op_api/
    +-- aclnn_conv3ddw.h/.cpp         两段式接口与参数校验
    `-- conv3ddw.h/.cpp               L0 常量装载与 AICore 启动记录
```

本目录提供算子实现源码，工程骨架由 `ops/ascendc-registry-invoke-template` 提供。共享主体不从相邻示例或参考算子仓 include 文件。

## 2 接口与注册契约

| 项目 | 约定 |
|---|---|
| 输入顺序 | x、filter_size、out_backprop |
| 数值类型 | x/out_backprop 同为 FP16 或 BF16，y 为 FP32 |
| 格式组合 | 两种配对特征格式 × 三种输出格式 × 两种输入类型，共 12 组 |
| filter_size | L2 接收 aclIntArray，转换为 executor 持有的 int32 ND 常量；值按输出格式解释 |
| 推导 | `InputsDataDependency({1})`，五个目标尺寸均为正；Tiling 再核对输出形状与常量值 |
| 属性顺序 | strides、pads、dilations、groups、data_format、enable_hf32、output_padding |
| 调用域 | 连续五维张量、groups=1、output_padding=0、enable_hf32=false，最终计划满足直调示例相同准入 |
| 核类型 | 编译实例 MIX_AIC_1_2，Host `SetScheduleMode(1)`；统一清零和通知后才计算 |
| key | MN_StreamK 枚举值 2，对应本参数列表下标 1；用 GET_TPL_TILING_KEY 计算，块大小不编码入 key |

L2 第一段使用 `L2_DFX_PHASE_1` 建立调用上下文后创建 executor，复制属性、装载常量并记录算子；第二段使用 `L2_DFX_PHASE_2` 后提交。L0 使用 `L0_DFX`。这些入口还承担线程上下文初始化，不能删成只有 launcher 宏的包装。

用户归约 workspace 为零；ACLNN 为形状常量和运行时资源计算的 workspace 由第一段返回，调用方按返回字节数申请并保持到执行完成。

## 3 构建与调用

将本目录源码接入 `ops/ascendc-registry-invoke-template` 工程，在生成工程中构建和安装：

```bash
source <CANN>/set_env.sh
cd <generated-project>
bash build.sh --soc=<compute-unit>
bash build/custom_opp_*.run --quiet --install-path=<install>
source <install>/vendors/Conv3ddw/bin/set_env.bash
```

生成工程编入 Host 求解器、注册入口和手写 `op_api`，以 `conv3ddw.cpp` 为设备编译入口。`OPS_ACLNN_GEN=0` 保留手写接口，安装头使用 `op_api/aclnn_conv3ddw.h`，API 库链接 `libopapi.so`。构建和打包覆盖全部 dtype/format 设备二进制；目标 API、资源和 MIX 能力由实际 SDK 与 platformInfo 核对。

```cpp
uint64_t workspaceSize=0;
aclOpExecutor *executor=nullptr;
auto status=aclnnConv3ddwGetWorkspaceSize(x,filterSize,dedy,strides,pads,dilations,
    1,dataFormat,false,nullptr,dedw,&workspaceSize,&executor);
if (status!=ACLNN_SUCCESS) return status;
void *workspace=AllocateWorkspace(workspaceSize);
status=aclnnConv3ddw(workspace,workspaceSize,executor,stream);
SynchronizeBeforeRelease(stream,workspace);
```

上面的内存和同步包装由调用工程实现；两段式的错误码都须检查。生成代码不调用内置卷积替代自身计算。
