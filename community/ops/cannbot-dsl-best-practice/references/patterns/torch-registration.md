# Torch 注册：复用公开入口的薄适配

需要 `torch.ops`、Meta/Fake 或图模式接入时使用。公共计算与程序生命周期沿用 [整体实现](../overall-implementation.md)；新增 Torch 支持只增加 schema、注册和参数映射，Kernel 与编译参数契约的构建逻辑保持集中维护。

Torch 注册装饰器由 PyTorch 提供；NPU dispatch、Meta/Fake 和图模式支持按实际 PyTorch 与设备后端核对。

## 接入结构伪码

以下描述职责与数据流，注册 API、dispatch key 和符号约束表达以目标 PyTorch 与设备后端版本为准。

```text
公开入口：operator(public_inputs, public_attributes) → public_outputs

定义 schema:
    参数名称、顺序、类型、默认值、可选性
    返回数量与顺序、mutation 与 alias

注册设备实现(schema_args):
    显式映射 schema 参数到公开入口参数
    return operator(映射后的参数)

注册 Meta/Fake 实现(schema_args):
    检查抽象执行环境可表达的元数据约束
    output_specs = 从输入元数据与属性推导输出描述
    return 按目标注册机制构造抽象输出
```

设备实现复用公开入口的校验、资源准备与程序调用。Meta/Fake 只描述输出，不调用公开计算入口、分配真实设备资源或读取设备数据。Meta 与 Fake 是不同接入机制，按目标环境选择，避免重复注册冲突。

## 共享公开契约与输出描述

| 对齐项 | 需要保持一致的内容 |
|---|---|
| 参数 | keyword 名称、默认值、可选性与标量含义；名称不同时显式映射 |
| 返回 | Tensor 数量、顺序、可选结果及嵌套结构 |
| 输出属性 | shape、dtype、stride 及对应抽象环境的 device 语义 |
| 副作用 | 输入是否被修改、输出是否与输入共享存储 |

输出推导复杂时，将其抽取为纯元数据函数，由真实分配与抽象描述共用。符号 shape 路径保留符号表达，使用该环境支持的约束检查；数据依赖的输出 shape 需要专门支持，不能读取设备值来补齐推导。

### 核心例子：返回有效视图时保留真实 stride

假设局部契约返回二维输出，但底层每行分配 `padded_cols` 个元素，只暴露前 `cols` 列：

```text
物理分配：storage.shape = (rows, padded_cols)，行连续
真实返回：output = storage[:, :cols]
共同描述：shape = (rows, cols)
          stride = (padded_cols, 1)
          dtype = 公开输出类型
          与输入无 alias

Meta/Fake 按共同描述构造抽象输出
```

若只按 `(rows, cols)` 构造普通连续 Tensor，抽象输出的行 stride 会变为 `cols`，与真实返回不一致。描述推导应来自实现采用的 padding 规则，而非复制输入属性。

## 注册对象与图模式

namespace、schema、注册对象生命周期和 dispatch key 按当前工程维护；导入后注册对象应在所需期间保持有效。设备实现直接委托公开入口，避免维护另一套计算或编译缓存。

需要图模式时，按公开支持范围增加追踪接入，并保持参数、返回与副作用契约。自动求导和动态 shape 各自需要实现依据与验证；注册成功不自动提供这些能力。

**反例：** 设备注册漏传非默认参数；schema 宣称函数式调用而实现修改输入；Meta/Fake 使用 `empty_like(input)` 描述不同布局的输出；图替换遗漏一个返回值；在注册文件复制 TensorSpec 或 Kernel 实现。

**检查：** 直接调用与 dispatcher 调用在默认、非默认和可选参数下结果一致；真实与抽象输出的 shape、dtype、stride、结构及 alias 一致；仅对公开要求的图模式、符号 shape 和梯度路径执行相应验证。
