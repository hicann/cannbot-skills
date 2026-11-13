# conv3dtranspose_aclnn — Conv3DTranspose 注册算子参考实现（aclnn 注册形态，转置卷积变体，终版）

> 与 `conv3dtranspose_direct` 共享同一求解器与 kernel 主体，提供注册算子实现源码（GEIR 原型 + 图推导 + OpDef + 值依赖推导 + TilingFunc + registry 入口 + **手写 ACLNN L2 契约**）。支持范围与 [conv3dtranspose_direct](../conv3dtranspose_direct/README.md) 完全一致（含 bias BT 通路、output_padding、显式 input_size 双模式、三 tiling 模板、三 weight 变换）。工程骨架由 `ops/ascendc-registry-invoke-template` 提供。

## 实测记录（Ascend950DT_9592，CANN 9.3.0，全工作流验收终版）

- **功能**：TTK 全量批 **626/626 strict_pass**（L0 26 + L1 600，torch golden 基准）；L2 负向 19/19 全拒
- **Tiling UT**：83/83 全绿（含 rho>0 / bias 满载/重载 / 奇 32B 块 / K 尾块 / 负右边界）
- **GEIR 负向**：18/18 × 双组合全拒
- **GEIR 正向**：33/40（7 例 rho>0 为 GE 编译期内置 tiling 限制——`TbeOpTilingPyInterfaceOld` 路由到内置 Conv3DTransposeV2 的形状推断不含 rho 余量；本算子运行期 tiling 对同几何全绿，带证登记）
- **代码审查**：7 项 findings 全部关闭（F-01 GetInputDesc+回退通道 / F-02 具名常量 / F-03 响亮拒绝+断言 / F-04 容量校验 / F-05/06 登记 / F-07 删除 PIPE_ALL）

## 与 conv3ddx_aclnn 的差异（转置卷积变体迁移要点）

| 维度 | conv3ddx_aclnn | conv3dtranspose_aclnn | 范式依据 |
|------|---------------|----------------------|---------|
| OpDef 组合 | 24 组合（filter 3 布局 × fp16/bf16 × int32/int64 × dedy 2 布局） | **72 组合**（filter 3 布局 × x 2 布局 × fp16/bf16 × int32/int64 × bias fp32/fp16）+ fix-3 布线（DHWCN 优先回退序 + ND 接受组合） | preprocessing §2.1 + Interface.md |
| 输入 | 3 必选（input_size/filter/out_backprop） | **5 槽**（input_size/x/filter/bias 可选/offset_w 保留可选）+ R4 安装树脚本补全第 5 槽 | preprocessing §2.2 + bias-bt-path §1 |
| 值依赖 | InputsDataDependency({0}) | 同 + **双模式**（显式逐轴 rho∈[0,s) / 全零哨兵属性推导 rho=0） | preprocessing §4.2 |
| 推导 | InferShape 链式 + 布局消歧 | 同 + **rho 校验**（rho=Ly−(base+q)，违例响亮拒绝） | preprocessing §4.2/§4.3 |
| Tiling 求解 | 三模板 + loadB2 + b⁺=Keff−1−p⁺+rho | 同 + **转置几何 b⁺=Keff−1−p⁺+q+rho** + **initOutputFlag 转置条件表**（增补 qD/qH>0） + **CalcBiasFullLoadFlag** + bias 字段 | preprocessing §4.3/§5.2/§5.3 + scene §2.7 |
| Kernel | 三 loadB2 变换 + ComputeZeroBlock | 同 + **bias BT 通路**（FullLoadBias/LoadBiasToBT + 带 BT 首次 MMAD 初值 + ComputeForBias 无贡献块写 bias）+ **ComputeZero**（零初始化 MMAD 替代标量写零）+ **域断言**（CONV3DTRANSPOSE_FIELD_ASSERT） | bias-bt-path 全文 + sync §1.2 |
| L2 契约 | 十二形参手写（含 out 张量） | 同构 + **bias 张量形参**（可选 null=无 bias）+ **offset_w 保留槽** | Interface.md |
| TilingData | 无 bias 字段 | **+ bias 槽数据**（biasBytes/biasUse/isBiasFullLoad/biasAllocBytes 等）+ 转置几何字段（rhoD/H/W, qD/H/W） | tiling_data.h |
| 构建接线 | fix-3 24 组合布线 | fix-3 **72→74 组合**布线（含 CONV3DTRANSPOSE_OPS_INFO_WIRING 开关——UT 注册表直调路径需关闭）+ R4 reserved-slot 三层安装树脚本 | geir 修复轮 + fix-findings |

## 文件结构

```text
conv3dtranspose_aclnn/
+-- op_graph/
|   +-- conv3dtranspose_proto.h              (91)  GEIR 原型（REG_OP 5 输入 1 输出 + 8 属性）
|   `-- conv3dtranspose_graph_infer.cpp     (792)  GE 图推导（FE 布局归一化消歧）
+-- op_host/
|   +-- conv3dtranspose_def.cpp            OpDef 注册
|   +-- Conv3DTranspose.h                  输入输出、属性和平台配置
|   +-- conv3dtranspose_infershape.cpp     (1017)  InferShape + InferDataType 链式注册
|   +-- arch35/
|   |   +-- conv3dtranspose_tiling.cpp       (24)  ascend950 唯一 tiling 编译单元（转发）
|   |   +-- conv3dtranspose_tiling_arch35.h  (46)  CompileInfo（平台资源入图携带）
|   |   `-- conv3dtranspose_tiling_arch35.cpp (2363) 标准 TilingFunc（F-01 GetInputDesc+回退通道）
|   `-- arch22/conv3dtranspose_tiling.cpp          多架构脚手架空跑桩
+-- op_kernel/
|   +-- conv3dtranspose_arch35.cpp          (119)  registry 入口（GET_TILING_DATA_WITH_STRUCT + AIC 守卫）
|   +-- arch35/
|   |   +-- conv3dtranspose.h             (1788)  kernel（三 loadB2 + BT 通路 + ComputeZero + 域断言）
|   |   +-- conv3dtranspose_tiling_data.h  (154)  TilingData POD（+bias/转置字段 + 具名对齐常量）
|   |   `-- conv3dtranspose_tiling_key.h   (151)  TPL 声明 + 三 key 实例（AIC_ONLY）
|   `-- arch22/ + conv3dtranspose_arch22.cpp       多架构脚手架空跑桩
+-- op_api/                                 手写 ACLNN L2 契约（负载交付，OPS_ACLNN_GEN=0）
+-- test_aclnn_conv3dtranspose.cpp            aclnn 两段式 Host 入口
`-- README.md
```

## 构建说明

将本目录源码接入 `ops/ascendc-registry-invoke-template` 骨架，在生成工程中执行构建、安装和环境加载。Host 仅编译 `arch35/conv3dtranspose_tiling.cpp` 转发入口；Kernel 编译 `conv3dtranspose_arch35.cpp`。设置 `OPS_ACLNN_GEN=0`，编入手写 `op_api`、安装对应公共头，并链接 `libopapi.so`；图推导源码随图原型库打包。

GEIR 接入由生成工程补齐两项配置：① `CONV3DTRANSPOSE_OPS_INFO_WIRING` 控制的 ops-info 后处理（DHWCN 优先、追加两个 ND filter 接受组合，72→74；UT 关闭）；② reserved-slot 安装树处理，补齐 `offset_w` 第 5 槽。

## 已知边界

1. **GEIR 正向 7 例 rho>0**：GE 编译期 `TbeOpTilingPyInterfaceOld` 路由到内置 Conv3DTransposeV2 tiling（形状推断不含 rho），非本算子缺陷（运行期全绿）——带证登记，同 conv3ddx geir 豁免先例
2. **UT 注册表直调**：需 `CONV3DTRANSPOSE_OPS_INFO_WIRING=OFF`（布线 74 组合与二进制 72 注册不一致时 gert dispatch 全拒）
