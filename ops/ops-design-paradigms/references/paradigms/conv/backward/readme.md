# 卷积反向范式档案

本目录组织非量化浮点三维卷积的输入梯度、权重梯度和转置卷积生成知识。设计入口为 [patterns.md](patterns.md)，通用知识与 Ascend C 适配按 input、filter 分支组织。

## 1 典型算子

| 算子 | 数值输入 → 输出 | 分支 | 示例 |
|---|---|---|---|
| Conv3DBackpropInput / Conv3DDX | weight、dedy → dedx | input | [直调](adapters/ascendc/example/conv3ddx_direct/README.md)、[ACLNN](adapters/ascendc/example/conv3ddx_aclnn/README.md) |
| Conv3DTranspose | x、filter、可选 bias → y | input | [直调](adapters/ascendc/example/conv3dtranspose_direct/README.md)、[ACLNN](adapters/ascendc/example/conv3dtranspose_aclnn/README.md) |
| Conv3DBackpropFilter / Conv3DDW | x、dedy → dedw | filter | [直调](adapters/ascendc/example/conv3ddw_direct/README.md)、[ACLNN](adapters/ascendc/example/conv3ddw_aclnn/README.md) |

`filter` 是权重梯度知识分支的目录名。Conv3DTranspose 的输入张量 filter 仍按其接口语义绑定到 input 分支的 weight。

## 2 支持范围

| 维度 | 范式约定 |
|---|---|
| 规格来源 | 根据正向卷积参数建立梯度问题，或直接消费反向算子规格；Transpose 按自身形状契约绑定 |
| 计算类型 | 非量化浮点；数值输入采用 FP16 或 BF16，分支内部保持要求的同类型关系；dedw 输出 FP32 |
| 外部布局 | 特征张量采用配对的 NCDHW 或 NDHWC；权重及 dedw 的布局组合以分支契约为准 |
| 卷积属性 | stride、dilation、各轴两侧 pad、groups 按分支形状关系和最终 Tiling 准入条件校验；Transpose 另有 output_padding、目标形状和可选 bias 契约 |
| 维度归一 | 以三维问题设计和执行；1D/2D 规格按 [前置处理](generic/preprocessing.md) 升维并保留轴语义 |
| 开发语言与交付 | Ascend C；分别提供直调和 ACLNN 源码示例，由对应调用工程接入 |
| 平台能力 | 从 platformInfo 和目标 SDK 获取资源及 API 能力，按最终布局、Tiling、搬运和同步约束判断可实现性 |

input 采用 standard、small-shape、l1-resident 三种 Tiling 模板；filter 以 MN_StreamK 为计算主线，窗口、搬运与写回分支按其契约展开。范式知识与具体示例的实现范围分别在 [input 索引](generic/input/index.md)、[filter 索引](generic/filter/index.md) 和各示例 README 中说明。

### 目录与路由

`conv/backward/` 按卷积族和计算方向组织目录，路由键为 `ConvBackward`。[routes.yaml](../../routes.yaml) 的 `routes.paradigms.ConvBackward` 显式指向本目录的 patterns.md，沿用通用范式路由；目录层数不参与标签匹配。

```text
op.paradigms contains ConvBackward
    -> conv/backward/patterns.md
        -> generic/index.md + adapters/ascendc/index.md
            -> input/index.md   (Conv3DDX / Conv3DTranspose)
            -> filter/index.md  (Conv3DDW)
```

`ConvBackward` 是 `op.paradigms` 标签；`op.category` 仍从既有分类中按语义选取。例如使用 Contraction 分类时，保留该分类必需的 Contraction 标签，并添加 ConvBackward：

```yaml
op:
  category: Contraction
  paradigms: [Contraction, ConvBackward]
  language: ascendc
```

两级公共索引说明共享术语、规格和能力，各分支索引维护自己的计算流、Tiling、Kernel、同步和验证知识。

## 3 后续扩展方向

- 按新增场景补充 filter 执行分支的直调和 ACLNN 示例，并完善形状、布局及归约覆盖。
- 随目标 SDK 更新核对搬运、矩阵计算、写回和同步契约，通过平台能力映射接入实现。
- 新增类型、布局或语言适配时，同步更新分支准入、索引、示例和验证规则。
