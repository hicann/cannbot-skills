# conv3ddx_aclnn — Conv3DBackpropInput 注册算子参考实现（aclnn 注册形态，终版）

> 与 `conv3ddx_direct` 共享同一求解器与 kernel 主体（Host 求解器 / kernel 类 / 校验层），交付形态为完整注册算子包：GEIR 原型 + 图推导 + OpDef + 值依赖推导 + TilingFunc + registry 入口 + **手写 ACLNN L2 契约** + 构建接线。支持完全相同的规格范围；注册接入以本示例为准，工程骨架见仓内技能 `ops/ascendc-registry-invoke-template`。本目录为全工作流（62 任务，需求→设计→实现→验证→调优）验收终版的逐字副本。

## 支持范围

与 direct 完全一致（输入梯度场景、fp16/bf16、dedy/dedx NCDHW/NDHWC 配对、1D/2D/3D 升维统一、三 tiling 模板、三 weight 变换、主计算完整写回、Host 校验层拒绝非法规格），见 [direct README 规格覆盖表](../conv3ddx_direct/README.md)。注册形态新增：filter 三布局独立（NCDHW/NDHWC/DHWCN）、input_size 值依赖（int32/int64）、OpDef 24 个 dtype×format 组合、手写 L2 接口（十二形参，含 `out` 张量与 `aclnnConv3ddxTensorGetWorkspaceSize` 张量形态变体）。

## 实测记录（Ascend950DT_9592，CANN 9.3.0，全工作流验收终版）

- **功能**：TTK 全量批 **626/626 strict_pass**（L0 27 + L1 599，torch golden 基准）；L2 负向 13/13 全拒
- **白盒**：三分支（key 16777217/18/19）各 1024 例上板 100% 通过（含 DHWCN/混合布局配对与 4 dtype 组合）
- **GEIR 图模式**：34 例 × src-static/src-dyn/bin-static/bin-dyn 四组合全绿；负向 13/13 拒绝
- **Tiling UT**：60 例全绿（21 正向 oracle 手推 + 39 负向日志锚点，含重钉修复记录）

## 文件结构

```text
conv3ddx_aclnn/
+-- op_graph/
|   +-- conv3ddx_proto.h                GEIR 原型（REG_OP 三输入一输出 + 六属性）
|   `-- conv3ddx_graph_infer.cpp        GE 图推导（FE 布局归一化消歧；undeclared-filter
|                                       归一化由注册层构建接线承载，见下「构建」注记）
+-- op_host/
|   +-- conv3ddx_def.cpp                   OpDef 注册
|   +-- Conv3ddx.h                         输入输出、属性和平台配置
|   +-- conv3ddx_infershape.cpp         InferShape + InferDataType 链式注册（613 行：
|   |                                   InputsDataDependency({0}) + 值依赖推导 + 布局消歧）
|   `-- arch35/
|       +-- conv3ddx_tiling.cpp         ascend950 唯一 tiling 编译单元（转发）
|       +-- conv3ddx_tiling_arch35.h    CompileInfo（平台资源入图携带）
|       `-- conv3ddx_tiling_arch35.cpp  标准 TilingFunc（1915 行；含 3 轮调优保留的
|                                       host-only 装载守卫优化；与 direct 求解器同底本）
+-- op_kernel/
|   +-- conv3ddx_arch35.cpp             registry 入口（GET_TILING_DATA_WITH_STRUCT + AIC 守卫）
|   `-- arch35/
|       +-- conv3ddx_kernel.h           与 direct kernel 同主体（1139 行；无贡献块走
|       |                                   ComputeZeroBlock：零权重单次 MMAD + Fixpipe 通路）
|       +-- conv3ddx_struct.h           TPL 声明 + 三 key 实例（核类型 AIC_ONLY，不含 dtype/format）
|       `-- conv3ddx_tiling_struct.h    TilingData POD（Host/Device 同一份）
+-- op_api/                             手写 ACLNN L2 契约（负载交付，非脚手架残留：
|   +-- aclnn_conv3ddx.h/.cpp          源工程构建关闭 autogen 桩生成后编入；十二形参
|   |                                   两段式 + Tensor 形态变体）
|   `-- conv3ddx.cpp/.h                L1 层（单算子执行入口）
+-- test_aclnn_conv3ddx.cpp            aclnn 两段式 Host 入口（手写契约版；spec boundary
                                        同族退化几何，解析可推 golden）
`-- README.md
```

> 本示例只收算子实现代码（单一交付路径 ascend950/arch35）；包构建骨架（CMakeLists /
> build.sh / arch22 空跑桩）不纳入，见 `ops/ascendc-registry-invoke-template`。

## 与 direct 的 diff（迁移要点）

| 维度 | direct | aclnn |
|------|--------|-------|
| Kernel 入口 | `Launch*` 六入口值传参 | `__global__ conv3ddx` 六 GM 参数 + `GET_TILING_DATA_WITH_STRUCT` |
| 入口命名 | launch 表任意命名 | OpType snake_case（`Conv3ddx` → `conv3ddx`，`opFile`/`opInterface` 绑定） |
| **核类型** | bisheng `-xasc` 按 AIC-only 编译 | **必须 TPL SEL 末位声明 `ASCENDC_TPL_KERNEL_TYPE_SEL(AIC_ONLY)`**，未声明按 VectorCore 编译空跑 |
| Tiling 装载 | `ReadTilingFromGM` 手工按 28B 路由头偏移拷贝 | `REGISTER_TILING_DEFAULT` + `GET_TILING_DATA_WITH_STRUCT`（框架 tiling 槽整体读入） |
| Host 求解 | 宿主 main 直接调 `Conv3DdxSolveTiling`（platformInfo 运行时查询） | 标准 TilingFunc 从 context 取参数（CompileInfo 与运行时查询交叉核对） |
| 属性 | 宿主结构体/CLI 传参 | OpDef Attr（strides/pads/dilations/groups/data_format/enable_hf32） |
| dtype/format 实例 | Launch 表 6 实例（2 dtype × 3 loadB2） | `ASCENDC_TPL_SEL` 3 key 实例（key = 2^24 + loadB2），dtype/format 由构建按 24 组合外层宏注入 |
| filter 布局 | NCDHW / NDHWC 二选 | 三布局独立（NCDHW/NDHWC/DHWCN，独立于 data_format） |
| 推导 | 不需要（宿主直接构造规格） | **必须链式注册 + 值依赖**（`InputsDataDependency({0})`；y.shape 只能来自 input_size 元素值，不可从 out_backprop 反解） |
| L2 接口 | 无 | **手写契约**（十二形参含 `out` 张量；另有 `aclnnConv3ddxTensorGetWorkspaceSize` 张量形态供图模式常量馈入；构建关闭 autogen 桩生成） |
| 校验出口 | 返回码（宿主判定） | `OP_LOGE` 稳定锚点 + `ge::GRAPH_FAILED`（不路由任何 tilingKey） |
| 调用 | `conv3ddx_launch(stream, …)` | `aclnnConv3ddxGetWorkspaceSize` + `aclnnConv3ddx` |

## 构建、测试与已知边界

- **构建/安装**：走 `ops/ascendc-registry-invoke-template` 骨架——`bash build.sh --soc=ascend950` → `build/custom_opp_*.run` → `--install-path=<work>/dev-package` → `source …/set_env.bash`。两处**源工程构建接线**随骨架落地时需从源工程 `op/op_host/CMakeLists.txt` 抄录：① aclnn 桩生成关闭（`OPS_ACLNN_GEN=0`，手写 op_api 契约编入）；② autogen ini 后处理（DHWCN 组重排最前 + ND-filter 接受组合——GEIR undeclared-filter 的 FE 回退归一化，证据链见源工程 geir-run 记录）
- **验证通路**：TTK kernel 模式挂载安装包（GEIR 注册链路：op_proto → tiling .so → kernel 二进制）；用例/golden/判定口径同 [direct README 测试说明](../conv3ddx_direct/README.md) 与范式 [verification](../../../../generic/input/verification.md)
- **已知缺陷（带证放行登记，源工程 `.workflow/geir-gate-waiver-authorization.md）**：
  ① fp16 + DHWCN filter 在 **GEIR 图执行**派发下 B2 装载错位（同形状 aclnn 直调与 bf16 路径均正确——图模式该组合会静默算错，交付文档 known-defects 登记在册）；② 动态 shape 图模式下运行期 tiling 拒绝 ND filter desc（响亮报错）。修复后应将 2 用例从登记册回迁 GEIR 套件复跑
- **已知边界**：单一交付路径 ascend950/arch35（arch22 空跑桩与构建骨架留在源工程，不纳入本示例）；负右边界 `woEnd` 复跑点同 direct README「已知边界」
