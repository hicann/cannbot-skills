# Launcher 开发指导

本文是按需加载的 host 侧方法，供 Step 3 设计 PLAN、Step 4 实现冻结设计时使用。它不提供 Basic、Grouped、MX 或融合 recipe，也不规定固定工程树；参数语义、Tiling 合同和验证标准以当前 DESIGN/PLAN 为准，具体项目文件可由 Step 4 在实现中补充。

## 1. Launcher 职责

Launcher 负责：

1. 从项目合同解析逻辑 shape、dtype、layout、属性和运行时 dispatch；
2. 按 DESIGN/PLAN 的 Tiling/Params 合同生成或装载 TilingData；
3. 分配和释放 host/device buffer，按物理数据合同执行 H2D/D2H；
4. 用 DESIGN 冻结的 Kernel entry、GM 参数顺序、grid/core、workspace 和 stream 启动设备 Kernel；
5. 输出 PLAN 指定的设备结果和运行记录。

Launcher 不生成随机输入、不计算 CPU Golden、不判定精度、不选择候选、不切换备选、不读取 golden 文件决定行为。输入生成、物理转换、Golden 和比较由 PLAN 指定的项目资料负责。

## 2. 输入合同与文件清单

PLAN 必须先冻结：

```text
argument_schema
logical_to_physical_conversion
buffer_size_and_alignment_rules
input/output artifact paths
tiling provider
kernel ABI and optional arguments
dispatch/specialization rules
workspace lifecycle
```

`target_file_manifest` 与 `data_and_golden_wiring` 提供 Launcher、数据和 CLI 的初始计划，不由本文默认固定名称或数量。Step 4 可以在项目根内补充文件和脚本，并将实际路径记录到 `execution_record`。读取时按设计字节数执行 exact-size 校验，拒绝缺失、截断、非有限或越界输入。

## 3. ACL 会话与资源生命周期

按项目的设备上下文包装创建/销毁：

```cpp
ACL_CHECK(aclInit(nullptr));
ACL_CHECK(aclrtSetDevice(deviceId));
ACL_CHECK(aclrtCreateContext(&context, deviceId));
ACL_CHECK(aclrtCreateStream(&stream));

// H2D -> kernel launch -> synchronize -> D2H

ACL_CHECK(aclrtDestroyStream(stream));
ACL_CHECK(aclrtDestroyContext(context));
ACL_CHECK(aclrtResetDevice(deviceId));
ACL_CHECK(aclFinalize());
```

真实工程必须对每个成功分配的 host/device/workspace 资源建立异常路径和释放顺序。`deviceId`、stream 数量和内存 flag 由项目合同决定，不能作为场景默认。

## 4. 逻辑数据到物理 buffer

对每个 Tensor 分开计算：

```text
logical shape/dtype
storage dtype/packed bytes
layout pattern (ND/NZ/other)
transpose/packing/padding
alignment and physical shape
row/plane stride and offset units
valid/tail range
```

ND buffer 通常按有效逻辑元素计算，块化/packed layout 必须按物理块、C0、padding 和打包单位计算。不要把逻辑 shape、Copy extent、UB pitch 和 GM row bytes 混为一个 size。量化 scale、group metadata、broadcast operand、workspace 和输出分别建立 size/offset 合同。

## 5. Tiling 与 Params

host Tiling 先逐字段对照 Investigation 的 Scheduler Params Semantic Contract：

1. 若已有项目 Tiling Engine 的类型、字段、单位、合法域、交叉约束和输出 ABI 完全兼容，PLAN 可授权复用或最小适配；
2. 若 Blaze 源码没有 host tiling，但 device 侧语义、合法域和 ABI 已闭合，PLAN 可授权项目 Engine 直接返回当前 partition 已证明的固定合法控制值；
3. 需要改变 partition、增加 specialization 或缺少合法域时回 Step 2/3；Launcher 不猜值；
4. TilingData、Kernel Params、Block/Scheduler/Epilogue Params 和 host 参数逐字段检查类型、顺序、单位和生命周期；
5. `usedCoreNum`、grid、workspace 大小和 dispatch flag 只能来自 DESIGN/PLAN 的冻结合同。

## 6. ABI 合同、签名骨架与启动绑定

Step 3 先在接口合同中冻结 `abi_mapping_draft`，再以每个选定 Blaze 组装方案的同一真实 witness 在 `matmul_base_analysis.abi_bindings[]` 中冻结准确 `kernel_abi_contract`。Launcher 不重新解释这些合同，只把与当前 `design_binding_ref` 对应的映射编译为 PLAN action。

### 6.1 合同表和交叉表

每个 `abi_bindings[]` 记录至少包含 `design_binding_ref`、`partition_ids`、`assembly_witness_ref` 及其 `kernel_abi_contract`：入口修饰符/链接、入口符号、模板与具体 specialization、有序 GM 参数及方向/可空语义、TilingData/Params、workspace、grid/usedCore、Wrapper、dispatch、final/partial 生命周期和每项 `source_ref`。

`abi_crosswalk` 对每个逻辑参数、输出和必要辅助 ABI 对象建立一行，并为每行提供稳定 `crosswalk_row_id`：

| 合同列 | 必填内容 |
|---|---|
| 逻辑对象 | `crosswalk_row_id`、参数或辅助 ABI 对象的 ID、角色、必选/可选语义 |
| 物理表示 | buffer、storage dtype/layout、字节范围、offset/stride 单位、有效范围 |
| host 接线 | Launcher 参数、所有者、分配/释放生命周期、H2D/D2H 方向 |
| device 接线 | Wrapper 参数、有序 Kernel GM 参数或入口绑定、可空条件 |
| 控制接线 | TilingData/Params 字段，或有证据的 `not_applicable`；workspace、grid/usedCore、stream/dispatch |
| 消费者 | Block、Kernel、Epilogue 或最终输出消费者，以及 final/partial 时机 |
| 证据 | 同一真实 witness 的 `source_refs` 和 DESIGN 合同 ID |

不同 `design_binding_ref` 的 crosswalk 行不能互换或拼接。场景只可用 `abi_crosswalk_delta` 追加自身 operand、输出、Params、Wrapper 或同步接线；不能重写基础 MatMul 行。所有实际 buffer、启动和 Wrapper action 都必须引用对应 binding 的 crosswalk 行。

### 6.2 从真实 witness 形成签名骨架

签名骨架是 DESIGN 中的可审计声明形态，不是可直接复制的实现代码。它的结构固定为：

```cpp
<observed linkage and entry modifier>
void <observed entry symbol>(
    <ordered GM parameter 0>,
    ...,
    <observed TilingData/Params parameter(s)>,
    <observed workspace or control parameter(s)>) {
  <observed Wrapper or selected Kernel invocation>;
}
```

只有真实 witness 已明确的 token 才能替换尖括号中的描述。不得因本文骨架自行加入 `__cube__`、`__mix__`、`GM_ADDR`、模板参数、GM 参数顺序或 Wrapper 调用方式；任何一个未闭合的 token 都是源码事实缺口，不是 Launcher 的自由选择。

### 6.3 启动绑定和闭合门禁

PLAN 的启动绑定必须以合同表逐项写明：

```text
entry modifier and symbol
selected template/specialization
ordered GM arguments and physical byte rules
TilingData address/size and Params mapping
workspace address/size/lifetime
grid/usedCore and dispatch/stream
Wrapper/entry invocation binding
source-backed signature skeleton ref
```

启动前必须证明每个必需逻辑对象都可追到设备消费者，且输出路径、optional/null 语义、物理字节/offset 单位、Tiling/Params、workspace、grid/usedCore 和 Wrapper/entry 全部闭合。Blaze 源码事实缺失时返回 Step 2 的一次补充调查；接口或冻结合同本身缺失时回 Step 3；仅 PLAN 文件映射或实现不完整时由 Step 4 在项目内补齐并记录。需求本身不明确时等待用户澄清。在任何一种语义或 ABI 缺口未解决前，不允许启动 Kernel。

## 7. Kernel 启动与 ABI

启动前依次校验：

- required/optional GM 参数顺序、方向、空值语义和物理字节数；
- TilingData/Params 地址和大小、workspace、grid/core、stream 和 entry modifier；
- layout/transpose/dtype/shape/dispatch 是否在 DESIGN 支持域；
- Buffer 不重叠、offset/stride 不越界，融合额外输入和输出满足 broadcast/生命周期合同；
- custom 文件、wrapper 和 build target 都在 PLAN 白名单。

不要从 Kernel 名称或相似工程猜测 `__cube__`、`__mix__`、模板参数、ratio 或参数顺序。实际 entry/Wrapper 必须来自 Investigation 和 DESIGN 的 concrete witness。

## 8. Dispatch 设计

只有需求要求 runtime 变化时才建立 dispatch。Step 3 必须决定：

```text
dispatch_axes
compile_time_specializations
runtime flags/enums
partition coverage per specialization
rejection for unsupported combinations
```

Launcher 可以采用分层 dispatch 或显式表，但不能自动生成所有 layout × transpose × dtype 的笛卡尔积。每个 specialization 的 Tiling/Params/ABI 和支持域要有独立 source refs；无证据的组合返回明确错误。

## 9. 数据生成、Golden 和结果输出

Launcher 与数据/Golden 组件的边界固定为：

```text
data producer: logical values -> physical buffers
launcher: buffers -> device Kernel -> raw output
golden/verifier: logical Golden + raw output -> threshold/nonfinite decision
```

五模式、known-C、SplitM、slot、broadcast、量化 metadata 或诊断输出只有在 DESIGN/PLAN 声明时才进入 Launcher 参数。Launcher 不现场生成 Golden，也不因 golden 失败自行调整阈值或重试备选。

## 10. 构建和安全边界

PLAN 提供项目 include、CMake target、Tiling library、Kernel wrapper 和 launcher source 的初始计划；Step 4 可以在项目根内调整或补充实现。三个官方源码区和 Skill Asset 原文件只读；编译错误涉及官方区时只读采集证据，并按版本问题回 Step 1、源码事实问题回 Step 2 或设计问题回 Step 3，不能直接打补丁。

每次构建/运行记录：

```text
project contract IDs
source/源码版本一致性
Tiling/Params values and units
actual grid/usedCoreNum
entry specialization
buffer sizes/addresses (non-sensitive summary)
result artifact and checkpoint
```

清理和最终回归按照 PLAN `cleanup_contract`、`validation_checkpoints` 和 `execution_record` 中的实际临时产物执行。本文只提供 host 方法，不定义交付命令、固定脚本、固定数据目录或固定文件名。
