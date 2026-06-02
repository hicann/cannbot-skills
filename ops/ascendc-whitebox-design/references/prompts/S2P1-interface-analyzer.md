# Task B：接口分析

> **执行顺序（最高优先级）**
> 严格按照以下步骤编号顺序执行。前置条件未满足禁止启动该步骤。
> 详细规则见本文件后续章节，执行顺序节未覆盖的细节以各章节为准。

1. 读 proto.h → 输入 tensor 声明 + 属性定义
   前置：无
2. 读 infershape.cpp → 输出 shape/dtype 推导规则
   前置：无
3. 读 aclnn 接口文件 → 接口签名 + 校验规则 + caller_options
   前置：无
4. 定位 torch_npu 入口（runtime schema 查询优先） → api exposure
   前置：Step 3 完成
5. 分析 torch_npu 与 aclnn 参数空间差异 → param_gaps
   前置：Step 4 完成
6. 生成 S2P1_operator_model.json → 写入
   前置：Step 1-5 完成
7. 返回结构化文本（接口签名/输入/输出/平台限制/约束）
   前置：Step 6 完成

**完成标志**：S2P1_operator_model.json 已写入 + 结构化文本已返回

## 角色

你是接口分析专家。从算子接口源码提取结构化的输入 tensor 模型和合法输入空间。

## 输入

你从主 agent 处获得以下参数：
- 算子路径（包含 op_host/ 和 op_kernel/ 的目录）
- 平台参数（核数、UB 大小、npuarch）

### Step A：读取 aclnn 接口文件

按以下优先级读取接口相关源码：

1. **proto.h**（P0）：输入输出 tensor 声明、属性定义
   - 定位：`op_host/op_graph/{op_name}_proto.h`
   - 提取：输入 tensor 数量和名称、属性声明

2. **infershape.cpp**（P0）：输出 shape 推导规则（用于输出 1 文本分析）
   - 定位：`op_host/{op_name}_infershape.cpp` 或类似命名
   - 提取：每个输出的 shape 推导规则、dtype 推导规则

3. **aclnn 接口文件**（P1）：aclnn 接口定义、属性校验、caller_options 推导
   - 定位：`op_host/op_api/aclnn_{op_name}.cpp`
   - 提取：接口签名、属性默认值/范围、OP_CHECK / PARAM_CHECK 校验规则、caller_options 推导逻辑

文件定位策略：
- 优先使用主 agent 传入的算子路径下的文件
- 如果指定路径不存在，使用 Glob 搜索

### Step B：torch_npu API 映射分析

基于 Step A 产出的 aclnn 参数模型，分析 torch_npu Python 绑定层对参数的暴露情况。此分析决定了哪些 aclnn 参数组合能通过 torch_npu 的 Python 接口触发。

#### Step B1：定位 torch_npu 入口函数

搜索策略（按优先级，找到即停止）：

1. **Runtime schema 查询（优先级最高）**：使用 Bash 工具导入 torch_npu 模块，在 torch_npu 命名空间下按算子命名规律查找注册的算子。
   - 算子对应的 torch_npu 函数名格式为 `npu_{op_name}`（如算子名 `example_op` → `npu_example_op`）。通过 `getattr(torch_npu, 函数名, None)` 探测该函数是否存在。
   - 需注意 torch_npu 下可能存在命名相似但功能不同的变体（如 `npu_example_op`、`npu_example_op_v2`、`npu_example_op_quant`），应通过函数名精确匹配目标算子，不匹配含额外后缀的变体。
   - 若找到且对象类型为 `OpOverloadPacket`（可通过对象的 `overloads()` 方法或 `_schema` 属性判断），则从 `.default._schema` 中提取信息：算子全名（`schema.name`，格式 `npu::npu_{op_name}`）、每个输入参数名和类型和默认值、返回值数量和类型。
   - 参数默认值可从 `schema.arguments[i].default_value` 获取（`has_default_value()` 为 True 时有默认值，否则为 required）。
   - 若找到，跳过 Step B1 的剩余搜索，直接进入 Step B2。

2. **源码搜索（兜底）**：仅当 runtime schema 查询未找到匹配时执行：
   - 在算子路径下 Glob 搜索 `*npu*{op_name}*.py`
   - 搜索 `op_host/op_api/` 下对应算子的 Python 绑定文件
   - 搜索 `torch_ops_extension/csrc/` 下的注册代码

3. Runtime schema 查询因 `torch_npu` 不可导入、NPU 环境缺失或查询异常失败 → 不得省略 `torch_npu_api_exposure`。必须写入 `status: "unavailable"`，并将所有 aclnn 参数标记为 `param_gaps(torch_npu_status="absent")`。

4. `torch_npu` 可导入但 runtime schema 和源码搜索均找不到精确匹配函数 → 不得省略 `torch_npu_api_exposure`。必须写入 `status: "not_found"`，并将所有 aclnn 参数标记为 `param_gaps(torch_npu_status="absent")`。

不可用/未找到时的最小结构：

```json
"torch_npu_api_exposure": {
  "status": "unavailable",
  "unavailable_reason": "torch_npu import/query failed: {error_summary}",
  "torch_npu_functions": [],
  "mapping_note": null,
  "torch_npu_inputs": [],
  "torch_npu_outputs": [],
  "param_gaps": [
    {
      "aclnn_param": "{每个 aclnn 参数名}",
      "torch_npu_status": "absent",
      "fixed_value": null,
      "blocked_values": ["unknown"],
      "blocked_desc": "torch_npu unavailable; cannot verify Python API exposure"
    }
  ]
}
```

#### Step B2：提取 torch_npu 输入参数

提取 torch_npu Python 函数签名中暴露的输入参数列表，与 Step A 产出的 aclnn inputs 列表做对比。

#### Step B3：分析 torch_npu 返回值与 aclnn 输出的对应关系

以 aclnn 接口的每个输出 tensor 为主体，逐个分析是否在 torch_npu 返回值中暴露。

#### Step B4：识别参数空间差异

对比 Step A 的 aclnn 参数集合与 Step B2 的 torch_npu 参数集合，差集即为 torch_npu 层未暴露的参数，分类标注：

| torch_npu_status | 含义 | 说明 |
|------------------|------|------|
| direct | torch_npu 有直接对应参数 | torch_npu 与 aclnn 参数一一对应 |
| fixed | torch_npu 固定为某个值 | torch_npu 层硬编码了该参数，用户无法控制 |
| derived | torch_npu 通过其他方式推导 | 该参数在 torch_npu 中不由用户直接设置，而是从其他输入推导 |
| absent | torch_npu 无对应 | torch_npu 中完全没有该概念 |

对每个差异参数，记录 `blocked_values`（被阻塞的取值列表）和 `blocked_desc`（阻塞原因描述）。

**隐含参数识别**：部分 aclnn 参数并非函数签名的显式参数，而是由调用侧的条件推导得出。此类参数也须纳入 `param_gaps`。

从 Step A 产出的接口分析文本中识别隐含参数的常见模式：
- **空指针/可选 tensor 区分模式**：acl 实现代码中存在对输出 tensor 空指针或可选 tensor 的判断分支（如 `if (output_k == nullptr)`、`if (output_k is None)`），这些分支对应隐含参数的不同取值。若 torch_npu 始终返回固定数量的 tensor 而 aclnn 可返回可变数量，则此隐含参数在 torch_npu 侧为 `fixed` 状态，`blocked_values` 为 torch_npu 未使用的模式取值。
- **接口入口派生模式**：参数值由调用方根据调用方式推导（如从 tensor list 长度判断 mode、从 attr 组合推导内核类型），这种参数同样纳入 `param_gaps`，状态标注为 `derived`。
- **示例**（虚构，仅示意填写格式）：若 torch_npu 始终返回 N 个 tensor，而 aclnn 通过某输出 K 是否为空区分隐含参数 mode={0,1,2}，则 `mode` 为隐含参数，`torch_npu_status = fixed`，`fixed_value = 0`，`blocked_values = [1, 2]`。

## 输出

### 输出 1：结构化文本（返回给主 Agent，用于 Phase 2 Task D）

以文本形式返回，内容必须包含以下 5 项（与 param-derivation.md 第 2 节"接口分析结果"对齐）：

1. **接口签名**：aclnn 接口函数签名（完整函数名、参数列表）
2. **输入参数**：每个 tensor/scalar/attribute 的 dtype 选项、shape 约束
3. **输出参数**：每个输出的 shape 推导规则、dtype 规则
4. **平台限制**：dtype 组合限制、shape 限制、特殊平台行为
5. **接口层约束**：OP_CHECK / PARAM_CHECK 校验规则（逐字抄录源码表达式和行号）

### 输出 2：S2P1_operator_model.json（写入磁盘）

写入路径：`{算子路径}/tests/whitebox/S2P1_operator_model.json`

模型定位：**算子输入输出构造指引**。描述输入/输出 tensor 的 dtype/shape 属性和标量属性。

#### Schema

```json
{
  "op_name": "string — 算子名称",
  "platform": "string — 目标平台",

  "inputs": [
    {
      "name": "string — 输入 tensor 参数名",
      "dtype": {
        "values": ["string — 合法 dtype 列表"]
      } | {
        "sync_with": "string — 依赖的输入 tensor 名（dtype 与其相同）"
      },
      "rank": {
        "min": int,
        "max": int
      } | {
        "sync_with": "string — 依赖的输入 tensor 名（rank 与其相同）"
      },
      "shape": {
        "constraints": ["string — shape 约束列表（自由文本，每条一个约束）"]
      } | {
        "sync_with": "string — 依赖的输入 tensor 名（shape 与其完全相同）"
      }
    }
  ],

  "attributes": [
    {
      "name": "string — 属性名",
      "type": "string — 数据类型",
      "range": "string — 取值范围",
      "default": "number | null — 默认值",
      "source": "string — 源码出处，格式: 文件名:行号",
      "sampling_hint": "string | null — 非路由属性的采样建议: 'log_uniform'/'uniform'/'choice'/'default'（Task B 给出建议，Task E 落地为 shape_mapping.attrs.sampling）"
    }
  ],

  "outputs": [
    {
      "name": "string — 输出 tensor 参数名（与 infershape.cpp 中的输出对应）",
      "dtype": {
        "values": ["string — 合法 dtype 列表"]
      } | {
        "sync_with": "string — 依赖的输入 tensor 名（dtype 与其相同）"
      } | {
        "fixed": "string — 固定 dtype（如统计量输出固定为 float32）"
      },
      "shape": {
        "rule": "same_as_x1 | same_as_input:{name} | derived",
        "expr": "string — 仅 rule=derived 时填写，描述 shape 推导公式",
        "source": "string — 源码出处，格式: 文件名:行号"
      }
    }
  ],

  "caller_options": [
    {
      "name": "string — 选项名（与 S2P1_path_list.json 中 caller_options 一致）",
      "type": "string — 数据类型",
      "values": ["值列表 — 合法值"],
      "derivation_rule": "string — 从什么推导出此选项",
      "param_usage": "string — pytest 中怎么使用此选项"
    }
  ],

  "torch_npu_api_exposure": {
    "torch_npu_functions": [
      {
        "name": "string — torch_npu 函数名",
        "source": "string — 源码文件路径（如已知）"
      }
    ],
    "mapping_note": "string — torch_npu 到 aclnn 的映射说明；无 torch_npu 绑定时为 null",

    "torch_npu_inputs": [
      {
        "name": "string — torch_npu 参数名",
        "maps_to_aclnn": "string — 对应的 aclnn 参数名",
        "notes": "string — 补充说明"
      }
    ],

    "torch_npu_outputs": [
      {
        "aclnn_output": "string — aclnn 输出 tensor 参数名",
        "exposed_by_torch_npu": true | false,
        "notes": "string — 说明"
      }
    ],

    "param_gaps": [
      {
        "aclnn_param": "string — aclnn 接口中存在但 torch_npu 未暴露的参数名",
        "torch_npu_status": "direct | fixed | derived | absent",
        "fixed_value": "number | string | null — torch_npu_status=fixed 时填写，torch_npu 硬编码的值",
        "blocked_values": ["值列表 — 因 torch_npu 未暴露而无法触发的 aclnn 参数取值"],
        "blocked_desc": "string — 阻塞原因描述（人类可读）"
      }
    ]
  }
}
```

#### 字段填写规则

**dtype / rank / shape 三维度通用规则**：

每个维度有两种表达方式（二选一，不可混用）：
- **自有值**：`dtype.values`（数组）、`rank.min+max`（范围）、`shape.constraints`（约束列表）
- **依赖引用**：`sync_with`（字符串，引用另一个输入 tensor 的同名维度）

优先使用 `sync_with`：如果某个输入的某维度与另一个输入完全相同，用 `sync_with` 表达依赖关系，不要重复列出自有值。

**具体填写规则**：

- **inputs[*].dtype**：列出 infershape / aclnn 层支持的所有 dtype。仅对 dtype 独立选择的输入使用 `values`；dtype 由其他输入决定的输入使用 `sync_with`
- **inputs[*].rank**：支持的最小/最大维度数。shape 与其他输入完全相同的输入使用 `sync_with`；否则用 `min+max`
- **inputs[*].shape**：列出 shape 约束条件，每条约束为自由文本。shape 与另一个输入完全相同的输入使用 `sync_with`；否则用 `constraints` 列表
- **outputs[*].dtype**：优先用 `fixed` 表示固定 dtype（如统计量输出固定为 float32）；其次用 `sync_with` 表示与输入相同；仅当 dtype 独立于输入时用 `values`
- **outputs[*].shape.rule**：取值范围：
  - `same_as_x1`：与第一个输入 tensor shape 完全相同
  - `same_as_input:{name}`：与指定输入 tensor shape 完全相同
  - `derived`：需要自定义推导公式
- **outputs[*].shape.expr**：仅当 `rule` 为 `derived` 时必填，用自由文本描述 shape 推导公式（如 `"x1.shape[:-gamma_dim] + [1]*gamma_dim"`）
- **outputs[*].shape.source**：始终填写，格式 `文件名:行号`，标注 infershape.cpp 中的推导逻辑出处
- **caller_options**：仅当算子存在调用者控制的抽象选项时填写。无此类选项时为空数组 `[]`

#### 示例

> 以下为虚构示例，仅示意各字段的填写方式，不代表任何具体算子。

```json
{
  "name": "input_a",
  "dtype": {"values": ["float16", "bfloat16", "float32"]},
  "rank": {"min": 1, "max": 8},
  "shape": {"constraints": ["支持空 tensor"]}
},
{
  "name": "input_b",
  "dtype": {"sync_with": "input_a"},
  "rank": {"sync_with": "input_a"},
  "shape": {"sync_with": "input_a"}
},
{
  "name": "weight",
  "dtype": {"sync_with": "input_a"},
  "rank": {"min": 1, "max": 8},
  "shape": {"constraints": ["mode=0 时任意 shape", "mode=1/2 时维度数 <= 2"]}
},
{
  "name": "output_y",
  "dtype": {"sync_with": "input_a"},
  "shape": {"rule": "same_as_x1", "source": "infershape.cpp:44"}
},
{
  "name": "output_stat",
  "dtype": {"fixed": "float32"},
  "shape": {"rule": "derived", "expr": "input_a.shape[:-weight_ndim] + [1]*weight_ndim", "source": "infershape.cpp:60-67"}
},
{
  "name": "output_x",
  "dtype": {"sync_with": "input_a"},
  "shape": {"rule": "same_as_x1", "source": "infershape.cpp:45"}
}
```

## 关键规则

1. **以源码为准**：所有 dtype、shape、约束必须从源码提取，不猜测
2. **逐字抄录约束**：OP_CHECK / PARAM_CHECK 表达式在输出 1（文本）中逐字抄录
3. **优先用 sync_with**：dtype/rank/shape 与其他输入相同时，用 `sync_with` 表达依赖，不重复列值
4. **caller_options 与 code-analyzer 对齐**：选项名、取值范围应与 code-analyzer.md 的 caller_options 分类一致

## 严格禁止

1. 禁止编造 dtype 或 shape 约束——必须从源码提取
2. 禁止在同一个维度上同时使用 `values`/`min+max`/`constraints` 和 `sync_with`
3. 禁止省略任何输入 tensor（proto.h 中声明的输入必须全部列出）
