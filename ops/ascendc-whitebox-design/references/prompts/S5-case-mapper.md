# Step 5 Case Mapper — 参数组合 → 合法 Tensor 构造配置

> **执行顺序（最高优先级）**
> 严格按照以下步骤编号顺序执行。前置条件未满足禁止启动该步骤。
> 详细规则见本文件后续章节，执行顺序节未覆盖的细节以各章节为准。
> 步骤编号与 workflow.md Step 5（5a/5b/5c/5d）一一对应。

5a. 生成 S5_case_mapper.py + S5_verify_mapper.py，运行产出 S5_mapped_cases_path.json（路径覆盖）
    前置：S2P2_cases.json + S2P1_operator_model.json 已就绪
5b. 映射网络用例 → S5_mapped_cases_network.json
    前置：5a 完成
5c. 合并 + 空 tensor 补全
    前置：5b 完成
    - 过滤路径覆盖 case：剔除任意输入 tensor 元素数 > 1 亿的 case
    - 读取 S5_mapped_cases_path.json + S5_mapped_cases_network.json
    - 空 tensor 补全
    - 合并路径 + 网络 + 空 → S5_mapped_cases_low.json（所有输入全 normal）
5d. data_range 拆分展开
    前置：5c 完成
    - 读取 S5_mapped_cases_low.json → expand_high → S5_mapped_cases_high.json

**完成标志**：S5_mapped_cases_path.json + S5_mapped_cases_network.json + S5_mapped_cases_low.json + S5_mapped_cases_high.json 已写入

## 1. 角色

将 S2P2_cases.json 的抽象参数组合映射为具体的 tensor 构造配置（shape + dtype），产出 `S5_case_mapper.py` 供 Step 6（pytest）和 TTK 模块消费。算子特定的 shape 逻辑全部来自 `S2P1_operator_model.json` 的 `shape_mapping` 节。

---

## 2. 输入 / 输出

| 输入 | 用途 |
|------|------|
| `S2P2_cases.json` | 抽象参数组合（mapper 运行时 `json.load` 读取） |
| `S2P1_operator_model.json` | shape_mapping（唯一真相来源）+ operator_model（验证用），mapper 运行时加载 |

### S2P2_cases.json 读取策略（强制）

子 agent 在验证映射逻辑时，**禁止**使用 Read 工具无限制读取 S2P2_cases.json 全文。
必须使用 Read 工具的 `offset=1, limit=30` 参数仅读取前几条用例作为验证样本。
若样本用例的映射结果正确，即可认为 shape_mapping 逻辑验证通过，全部用例由生成的 S5_case_mapper.py 脚本运行时通过 `json.load` 批量处理。子 agent 可自行把握样本数量，以覆盖主要 group 为准。

输出：`S5_case_mapper.py`（纯计算模块）→ 运行后生成 `S5_mapped_cases_path.json`

---

## 3. shape_mapping 概览

> **详细字段定义见** `{skill_base}/references/shape_mapping_schema.md`

| 节 | 关键字段 |
|----|---------|
| `dtype` | `param`, `values` |
| `shape_params` | `role`(leading_product/trailing_size/ndim/mode_switch), `default`, `group_defaults` |
| `ndim` | `source`(random/from_param), `range`, `tensor_constraints` |
| `inputs` | `rule`(decompose/sync_with/fixed/optional), `align_trailing_with`, `ndim_fallback` |
| `outputs` | `rule`(same_as/derived/fixed), `derived.expr`, `dtype_override` |
| `attrs` | `param`, `type`, `coerce`, `sampling` |

### dtype 传递规则

| rule | dtype 来源 |
|------|-----------|
| decompose | `dtype_str`（`sm["dtype"]["param"]`） |
| sync_with | 与目标 tensor 相同 |
| fixed | `spec["dtype"]` 或回退 `dtype_str` |
| optional | 存在时同 decompose，否则不适用 |
| same_as | 与目标 tensor 相同 |
| derived | `spec["dtype_override"]` 或回退 `dtype_str` |

---

## 4. 产物 1：S5_case_mapper.py

### 4.1 运行时加载

sm 和 model 必须从 `S2P1_operator_model.json` 运行时加载，禁止硬编码：

```python
with open("S2P1_operator_model.json") as f:
    _data = json.load(f)
sm = _data["shape_mapping"]
model = {"inputs": _data["inputs"], "outputs": _data["outputs"]}
```

`shape_mapping.outputs` 的 key 必须与 `operator_model.outputs` 的 name 一致。若不一致，以 `operator_model` 为准调整 `shape_mapping` key。

### 4.2 数据结构

```python
# tensor 规格（plain dict，无 dataclass，无 torch 依赖）
tensor_spec = {"shape": [int, ...], "dtype": "str"}

# 单个 case 的输出
case_output = {
    "id": "case00000",
    "params": dict,
    "tensors": {
        "inputs":  {name: tensor_spec, ...},
        "outputs": {name: tensor_spec, ...}
    }
}

# S5_mapped_cases_path.json 整体结构
{"cases": [case_output, ...]}
```

### 4.3 关键函数

| 函数 | 用途 |
|------|------|
| `_prime_factors(n)` | 返回质因数列表（如 49 → [7, 7]） |
| `_balanced_decompose(num, parts)` | **乘积分解**：将 num 分解为 parts 个整数因子，满足 `math.prod(result) == num`，各因子尽量接近。算法：① 对 num 做质因数分解得到 primes；② 初始化 parts 个桶为 [1]*parts；③ 用最小堆，每次弹出最小桶，乘以最大的质因数，反复直到 primes 用完。不变式：`math.prod(result) == num`，不允许因子为 0。示例：`_balanced_decompose(128, 3)` → `(4, 4, 8)`（4×4×8=128） |
| `_balanced_decompose_nontrivial(num, parts)` | 同上但要求每个因子 > 1。若质因数不足 parts 个（如 num=7, parts=5 → 质因数 [7] 仅有 1 个），无法满足全部 > 1，返回含 1 的降级结果（如 `(1,1,1,1,7)`），调用方按 ndim_fallback 处理 |
| `_resolve_param(case, param_name, sp)` | 按 `case[param] > group_defaults[group] > default` 优先级解析参数 |
| `_eval_when(expr, case)` | 按 schema 的 AST 白名单求值 when 表达式，违规表达式返回 False 并记录错误 |
| `_apply_ndim_constraints(lo, hi, constraints, case)` | 对 ndim 范围应用 tensor_constraints 过滤 |
| `map_case(case, rng, sm, model)` | 核心映射：单个 case → tensor 构造配置 |
| `validate_config(cfg, case, sm, model)` | 动态校验，返回错误列表（空 = pass） |
| `load_mapped_configs(cases_file, model_file, seed=42)` | 批量映射：加载 JSON → 逐 case 映射 → 校验 |
| `main()` | 入口：调用 load_mapped_configs → 写 S5_mapped_cases_path.json |

### 4.4 map_case 流程

```
输入: case (dict), rng (Random), sm (shape_mapping), model (operator_model)

1. 解析 dtype
   dtype_str = case[sm["dtype"]["param"]]

2. 解析 attrs — 若 case 中包含 spec["param"] 对应的 key → 直接取 case[key]；
   否则 → 按 spec["sampling"] 采样注入：

   注入规则:
   - spec 含 "sampling":
       strategy="log_uniform" → 10 ** random.uniform(log10(lo), log10(hi))
       strategy="uniform"    → random.uniform(lo, hi)
       strategy="choice"     → random.choice(values)
       strategy="default"    → spec.get("default", 0)
   - spec 不含 "sampling":    取 spec.get("default")，无 default 则跳过

   attrs = {name: _sample_attr(case, name, spec) for name, spec in sm["attrs"].items()}

3. 解析 shape_params
   对每个需要的 param:
     val = _resolve_param(case, param_name, sm["shape_params"])
   优先级: case[param] > shape_params[group_defaults][group] > shape_params[default]

4. 确定各 tensor 的 ndim
   a. base_lo, base_hi = sm["ndim"]["range"]
   b. 预计算所有 tensor 的 ndim 约束范围：
      遍历 sm["ndim"]["tensor_constraints"]，对每个 tensor 调用 _apply_ndim_constraints
   c. 主 tensor（含 decompose.leading 的输入）确定 ndim：
      lo = max(base_lo, 该 tensor约束下界)
      若该 tensor 的 decompose 有 align_trailing_with → lo = max(lo, 被对齐 tensor 的约束下界)
      若 decompose.leading 存在且 product > 1 → lo = max(lo, len(trailing) + 1)
      if lo > hi: lo = hi
      主 tensor ndim = rng.choice(range(lo, hi + 1))
   d. 其余 decompose tensor 确定各自 ndim：
      lo, hi = 该 tensor的约束范围（来自步骤 b）
      hi = min(hi, 主 tensor ndim)    # align_trailing_with 要求 ndim ≤ 主 tensor ndim
      若 decompose 无 leading → ndim 固定为 len(trailing)，不随机
      若 decompose 有 leading → ndim = rng.choice(range(lo, hi + 1))

5. 构造 inputs（按 sm["inputs"][name]["rule"] 分支）
   - decompose:
     a. trailing_shape = 对每个 trailing 元素取 param 值
     b. product = _resolve_param(case, decompose["leading"]["param"], sm["shape_params"])  # 仅 leading 存在时
     c. leading_dims_count = eval(parts_expr, {"ndim": 该 tensor 的 ndim, "trailing": trailing_shape})
     d. if leading_dims_count == 0: leading_shape = ()
        else: leading_shape = _balanced_decompose(product, leading_dims_count)
     e. shape = leading_shape + trailing_shape
   - sync_with: shape + dtype 复制目标 tensor
   - fixed: 直接使用 spec 中的 shape/dtype
   - optional: 按 case[param] 条件判断，存在则构造，否则置 None

6. 构造 outputs（按 sm["outputs"][name]["rule"] 分支）
   - same_as: shape + dtype 复制指定 tensor
   - derived: 按 schema 的 AST 白名单求值 expr（各 tensor 的 shape 和 ndim 变量）+ dtype_override
   - fixed: 直接使用 spec 中的 shape/dtype

7. 组装返回值
   return {
      "params": {**case, **resolved_shape_params, **attrs},
     "tensors": {"inputs": inputs, "outputs": outputs}
   }
    注意: attrs 通过 _sample_attr 注入后，合并入返回值 params（{**case, **attrs}）
```

### 4.5 validate_config 规则来源

| 来源 | 推导的校验 |
|------|-----------|
| `model.inputs[*].dtype.values` | 输入 dtype ∈ 合法值列表 |
| `model.inputs[*].rank.min/max` | 输入 ndim ∈ 范围 |
| `model.outputs[*].dtype.fixed` | 输出 dtype == 固定值 |
| `sm.inputs.*.align_trailing_with` | `src.shape == target.shape[-len(src.shape):]` 且 `src.ndim ≤ target.ndim` |
| `sm.ndim.tensor_constraints` | 条件 ndim min/max |

签名: `validate_config(cfg, case, sm, model)` → `list[str]`（空列表 = pass）

---

## 5. 产物 2：S5_verify_mapper.py

### 4 层验证

| 层 | 内容 |
|----|------|
| L1 | `validate_config()` — 规则由 schema 动态推导 |
| L2 | operator_model 交叉验证 — dtype/rank/shape |
| L3 | source_constraints 交叉验证 — 语义约束，API 矛盾标记 warn |
| L4 | NPU e2e — 调用算子 API 检查输出 shape |

**通用规则**：NPU 不可用 → SKIP；API 限制 → warn；其他 → fail。

采样：10 random cases，seed=42，≤20 时全量。

### 闸门

| 结果 | 处理 |
|------|------|
| 0 fail | PASS → 进入 Step 6 |
| ≥1 fail | FAIL → 修复，最多 3 轮 |

---

## 6. 约束

### 6.1 严格禁止

1. 禁止硬编码 SHAPE_MAPPING / operator_model — 必须运行时从 S2P1_operator_model.json 加载
2. 禁止硬编码 OUTPUT_NAME_MAP — shape_mapping.outputs 的 key 必须与 operator_model.outputs 的 name 一致
3. 禁止硬编码 shape / ndim — 按 shape_mapping schema 动态计算
4. 禁止 import torch / torch_npu — mapper 是纯计算模块
5. 禁止修改任何输入文件（S5/S2P1）
6. 禁止跳过 validate_config
7. 禁止凭直觉猜测 shape 逻辑 — 必须读 shape_mapping
8. 禁止 NPU 相关依赖出现在 mapper 中
9. 禁止将 S2P1_low_configs.json 中的 `source`、`reason` 等元信息字段写入 mapped JSON 的 `params` 中——网络用例仅保留 shape_mapping 中定义的算子参数，禁止透传来源链接或推断理由到任何输出字段

### 6.2 实战规则

**规则 1（退化维度）**：`_balanced_decompose` 产生含 1 的维度（如 `(1,1,1,128)`）是合法输入，不需要特殊处理。

**规则 2（parts=0）**：`leading_dims_count == 0` 时，`leading_shape = ()`，**禁止**调用 `_balanced_decompose(product, 0)`（会报错）。最终 `shape = () + trailing_shape`。

**规则 3（seed）**：`load_mapped_configs` 固定 seed=42，pytest 和 TTK CSV 必须使用相同 seed。

**规则 4（验证闭环）**：脚本生成后必须依次确认：
1. 运行脚本，退出码 0
2. S5_mapped_cases_path.json 生成，case 数量 == S2P2_cases.json
3. 0 validation errors
4. 抽查样本用例的 tensor 构造合法性（ndim、乘积分解、对齐），样本来源于使用 Read 工具 `limit` 参数读取的前几条用例

**规则 4a（样本验证）**：验证映射逻辑时，仅通过 Read 工具的 `limit` 参数读取 S2P2_cases.json 的前几条用例（约 30 行），不读取完整文件。样本数量由子 agent 自行把握，以覆盖主要 group 为准。

---

## 7. case 维度展开

> 在 5c 合并完成后执行。含空 tensor 补全 + data_range 展开。

### 设计原则

- `_data_range` 存储在 `tensors.inputs.{name}._data_range`，per-input 独立
- **low 档位**：全量 path + 空 tensor + 全网络，所有输入全 normal
- **high 档位**：path × data_range + 空 tensor (normal) + network × data_range
- 空 tensor 不参与 data_range 交叉，固定 `_data_range: "normal"`
- 空 tensor 的 `params["_group"]` 写为最终空输入名（如 `input_a_input_b_empty`），TTK remark 自动提取
- 每个输入 tensor 元素数 ≤ 100,000,000（1 亿）。超出的 path case 直接丢弃

### 空 tensor shape 展开

> 函数导出在 S5_case_mapper.py 中，由 S5_merge_expand.py 调用。

**目标**：从 1 个随机 path case 模板生成空 tensor 变体，要求最大化维度多样性。禁止将整个 shape 全零化——应利用 shape_params 的 role（leading_product / trailing_size）定位可空化的维度区域，仅局部置零，保留非零维度的结构信息。

**Phase 1 — 场景发现**：从 shape_params 中按 role 分类空场景（leading_product → leading 区域、trailing_size → trailing 区域），确定每个场景影响哪些 input tensor。

**Phase 2 — 变体生成**：对每个场景，从模板 shape 中定位对应维度区域，生成 1 个 single_zero 变体（仅一个位置置 0）。所有场景遍历完后，追加 1 个 all_zero 变体（所有输入 tensor 的全部维度置 0，兜底）。跨场景去重。修改后按依赖图传播（sync_with / align_trailing_with）并更新 outputs，同时将 params 中受 shape 变更影响的字段同步为与修改后 shape 一致的值。

**不变约束**：`_data_range = "normal"`、`_group` 命名规则、元数据字段、`supplement_empty_cases(sm, path_cases, seed)` 签名与返回格式。

### low 档位：全 normal

```python
def set_all_data_range(case, dr):
    for spec in case["tensors"]["inputs"].values():
        if spec is None:
            continue
        spec["_data_range"] = dr
    return case
```

### high 档位：one-hot + 全统一

```python
import copy

NON_NORMAL = ["zero", "extreme", "negative", "tiny_pos",
              "all_ones", "near_zero", "with_inf", "with_nan"]

def expand_high(cases):
    """one-hot + 全统一 展开。"""
    expanded = []
    for c in cases:
        input_names = [n for n, s in c["tensors"]["inputs"].items() if s is not None]
        nc = copy.deepcopy(c)
        nc["id"] = f"{c['id']}_all_normal"
        set_all_data_range(nc, "normal")
        expanded.append(nc)
        for dr in NON_NORMAL:
            nc = copy.deepcopy(c)
            nc["id"] = f"{c['id']}_all_{dr}"
            set_all_data_range(nc, dr)
            expanded.append(nc)
        for inp in input_names:
            for dr in NON_NORMAL:
                nc = copy.deepcopy(c)
                nc["id"] = f"{c['id']}_{inp}_{dr}"
                for name, spec in nc["tensors"]["inputs"].items():
                    if spec is None:
                        continue
                    spec["_data_range"] = dr if name == inp else "normal"
                expanded.append(nc)
    return expanded
```

### 输出文件格式（强制）

> **low 文件（`S5_mapped_cases_low.json`）必须使用缩进 + 选择性压缩格式**：
> - `json.dumps(indent=2)` 后，对 `shape`、`dtype`、`id`、`_group`、`_data_range` 字段压缩为单行
> - high 文件（`S5_mapped_cases_high.json`）保持紧凑格式，不做缩进

```python
import re

_COMPACT_KEYS = ("shape", "dtype", "id", "_group", "_data_range")

def _compact_json(text):
    for key in _COMPACT_KEYS:
        text = re.sub(
            rf'("{key}"): \[\s*\n((?:\s+[^\n]+,\n)*\s+[^\n]+\n\s*)\]',
            lambda m: f'{m.group(1)}: [{", ".join(l.strip().rstrip(",") for l in m.group(2).strip().splitlines() if l.strip())}]',
            text, flags=re.MULTILINE
        )
        text = re.sub(
            rf'("{key}"): ("[^"]*"|\d+(?:\.\d+)?(?:e[+-]?\d+)?)',
            rf'\1: \2',
            text, flags=re.MULTILINE
        )
    return text
```

### 主流程

```python
def main():
    import json, copy, math, random
    from S5_case_mapper import supplement_empty_cases

    with open("S2P1_operator_model.json") as f:
        sm = json.load(f)["shape_mapping"]
    with open("S5_mapped_cases_path.json") as f:
        path_cases = json.load(f)["cases"]
    with open("S5_mapped_cases_network.json") as f:
        net_cases = json.load(f)["cases"]

    # 5c: 过滤（元素数 > 1 亿）
    path_cases = [c for c in path_cases
                  if all(v is None or math.prod(v["shape"]) <= 100_000_000
                         for v in c["tensors"]["inputs"].values())]

    # 5c: 空 tensor 补全
    empty_cases = supplement_empty_cases(sm, path_cases)

    # 5c: 合并路径 + 网络 + 空 → low（all normal）
    combined = path_cases + net_cases + empty_cases
    low = [set_all_data_range(copy.deepcopy(c), "normal") for c in combined]
    with open("S5_mapped_cases_low.json", "w") as f:
        raw = json.dumps({"cases": low}, indent=2, ensure_ascii=False)
        f.write(_compact_json(raw))

    # 5d: high — 基于 low 做 data_range 展开
    high = expand_high(path_cases) + empty_cases + expand_high(net_cases)
    with open("S5_mapped_cases_high.json", "w") as f:
        json.dump({"cases": high}, f, ensure_ascii=False)
```

- ID 格式：
  - 全统一：`case00001_all_zero`、`network00001_all_extreme`
  - one-hot：`case00001_{input_name}_zero`、`case00001_{input_name}_with_inf`
  - 空 tensor：`case00081_{input_name}_empty`（`_group`=`{empty_inputs_joined}_empty`）
- `_data_range` 写在 `tensors.inputs.{name}._data_range`，per-input 独立
- 展开**不涉及 map_case 的 rng**，仅复制已映射的 case
- 空 tensor 的 `_group` 由最终全零输入名拼接，ttk-converter 的 `params["_group"]` → remark 规则自动生效

---

## 8. ✅/❌ 示例

```python
# ✅ 运行时加载 schema → ❌ 禁止硬编码 SM / model / ndim
with open("S2P1_operator_model.json") as f:
    sm = json.load(f)["shape_mapping"]
ndim = rng.choice(range(sm["ndim"]["range"][0], sm["ndim"]["range"][1] + 1))

# ✅ parts=0 特殊处理 → ❌ 禁止不处理 parts=0
leading_shape = () if leading_dims_count == 0 else _balanced_decompose(product, leading_dims_count)
```
