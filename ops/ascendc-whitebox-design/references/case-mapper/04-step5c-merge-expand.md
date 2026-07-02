# Step 5c：合并 + 空 tensor 补全 + data_range 展开

> **前置条件**：Step 5b 完成（S5_mapped_cases_network.json 已写入）

---

## 概述

| 输入文件 | 输出文件 |
|---------|---------|
| `S5_mapped_cases_path.json`、`S5_mapped_cases_network.json` | `S5_merge_expand.py`、`S5_mapped_cases_low.json`、`S5_mapped_cases_high.json` |

通过 `S5_merge_expand.py` 依次完成：合并 path + network + 空 tensor → low.json（all normal）→ data_range 交叉展开 → high.json。

---

## S5_merge_expand.py

脚本分为**通用部分**（照搬）和**算子专属部分**（`supplement_empty_cases`，见 §supplement_empty_cases 策略）。子 agent 先照搬通用部分，再根据算子 param_type 生成 `supplement_empty_cases` 并插入脚本，最后按 §执行 中的步骤依次运行。

```python
"""S5_merge_expand.py — 合并 + 空 tensor 补全 + 元素数过滤 + data_range 展开"""
import json, copy, math, os, re, sys

out_dir = os.path.dirname(os.path.abspath(__file__))

# === 共享工具函数 ===

NON_NORMAL = ["zero", "extreme", "negative", "tiny_pos",
              "all_ones", "near_zero", "with_inf", "with_nan"]

_DOMAIN_EXCLUDE = {
    "positive":     {"zero", "negative", "near_zero", "with_nan"},
    "non_negative": {"negative", "near_zero", "with_nan"},
    "non_zero":     {"zero", "near_zero", "with_nan"},
}

def _allowed_ranges(value_domain):
    """根据 value_domain 返回允许的 NON_NORMAL 标签列表。"""
    if not value_domain:
        return list(NON_NORMAL)
    vd_type = value_domain["type"]
    if vd_type == "range":
        lo, hi = value_domain.get("min"), value_domain.get("max")
        excluded = {"extreme", "with_nan", "with_inf"}
        if lo is not None and lo > 0:
            excluded.add("zero")
            excluded.add("negative")
        elif lo is not None and lo == 0:
            excluded.add("negative")
        elif hi is not None and hi < 0:
            excluded.add("zero")
        if hi is not None and hi < 1e-7:
            excluded.add("tiny_pos")
        if (lo is not None and lo > 1) or (hi is not None and hi < 1):
            excluded.add("all_ones")
        return [dr for dr in NON_NORMAL if dr not in excluded]
    excluded = _DOMAIN_EXCLUDE.get(vd_type, set())
    return [dr for dr in NON_NORMAL if dr not in excluded]

def set_all_data_range(case, dr):
    for spec in case["tensors"]["inputs"].values():
        if spec is None:
            continue
        if isinstance(spec, list):
            for sub in spec:
                sub["_data_range"] = dr
        else:
            spec["_data_range"] = dr
    case.pop("_data_range", None)
    return case

def expand_high(cases):
    """one-hot + 全统一展开。按 _value_domain 过滤不兼容 data_range。"""
    if not cases:
        return []
    expanded = []
    input_names = [n for n, s in cases[0]["tensors"]["inputs"].items() if s is not None]
    for c in cases:
        nc = copy.deepcopy(c)
        nc["id"] = f"{c['id']}_all_normal"
        set_all_data_range(nc, "normal")
        expanded.append(nc)
        all_allowed = set(NON_NORMAL)
        for spec in c["tensors"]["inputs"].values():
            if spec is None:
                continue
            s = spec[0] if isinstance(spec, list) else spec
            all_allowed &= set(_allowed_ranges(s.get("_value_domain")))
        for dr in sorted(all_allowed):
            nc = copy.deepcopy(c)
            nc["id"] = f"{c['id']}_all_{dr}"
            set_all_data_range(nc, dr)
            expanded.append(nc)
        for inp in input_names:
            s = c["tensors"]["inputs"][inp]
            s0 = s[0] if isinstance(s, list) else s
            allowed = _allowed_ranges(s0.get("_value_domain"))
            for dr in allowed:
                nc = copy.deepcopy(c)
                nc["id"] = f"{c['id']}_{inp}_{dr}"
                for name, spec in nc["tensors"]["inputs"].items():
                    if spec is None:
                        continue
                    target_dr = dr if name == inp else "normal"
                    if isinstance(spec, list):
                        for sub in spec:
                            sub["_data_range"] = target_dr
                    else:
                        spec["_data_range"] = target_dr
                expanded.append(nc)
    return expanded

_COMPACT_KEYS = ("shape", "dtype", "id", "_group", "_data_range", "_value_domain")

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

# === 算子专属：supplement_empty_cases（根据 param_type 生成，见下方 §策略） ===

def supplement_empty_cases(mapped_path_cases, seed=42):
    """从 1 个随机 path case 模板生成空 tensor 变体。
    REQUIRED 和 DYNAMIC 分支实现不同，子 agent 根据算子 param_type 选择对应模板生成。
    """
    pass  # 子 agent 根据 §supplement_empty_cases 策略 替换此实现

# === 5c 入口：合并 + 空 tensor + 过滤 → low ===

def _max_numel(spec):
    """计算 tensor spec 的最大元素数（DYNAMIC 取各子 tensor 最大值）"""
    if spec is None:
        return 0
    if isinstance(spec, list):
        return max((math.prod(sub["shape"]) for sub in spec), default=0)
    return math.prod(spec["shape"])

def merge_low():
    with open(os.path.join(out_dir, "S5_mapped_cases_path.json")) as f:
        path_cases = json.load(f)["cases"]
    with open(os.path.join(out_dir, "S5_mapped_cases_network.json")) as f:
        net_cases = json.load(f)["cases"]

    path_cases = [c for c in path_cases
                  if all(_max_numel(v) <= 100_000_000
                         for v in c["tensors"]["inputs"].values())]
    net_cases = [c for c in net_cases
                 if all(_max_numel(v) <= 100_000_000
                        for v in c["tensors"]["inputs"].values())]

    empty_cases = supplement_empty_cases(path_cases)

    vd_map = {}
    model_path = os.path.join(out_dir, "S2P1_operator_model.json")
    if os.path.exists(model_path):
        with open(model_path) as f:
            op_model = json.load(f)
        for inp in op_model.get("inputs", []):
            vd = inp.get("value_domain")
            if vd:
                vd_map[inp["name"]] = vd

    def _attach_vd(case):
        for name, spec in case["tensors"]["inputs"].items():
            if spec is None:
                continue
            vd = vd_map.get(name)
            if not vd:
                continue
            if isinstance(spec, list):
                for sub in spec:
                    sub["_value_domain"] = vd
            else:
                spec["_value_domain"] = vd
        return case

    combined = path_cases + net_cases + empty_cases
    low = []
    for c in combined:
        c = set_all_data_range(copy.deepcopy(c), "normal")
        _attach_vd(c)
        low.append(c)

    with open(os.path.join(out_dir, "S5_mapped_cases_low.json"), "w") as f:
        raw = json.dumps({"cases": low}, indent=2, ensure_ascii=False)
        f.write(_compact_json(raw))

# === 5d 入口：data_range 展开 → high ===

def expand_high_main():
    with open(os.path.join(out_dir, "S5_mapped_cases_low.json")) as f:
        low_cases = json.load(f)["cases"]

    def _is_empty(c):
        cid = c.get("id", "")
        return cid.endswith("_empty")

    empty_cases = [c for c in low_cases if _is_empty(c)]
    path_cases = [c for c in low_cases if not _is_empty(c) and c["params"].get("_group") != "network"]
    net_cases = [c for c in low_cases if c["params"].get("_group") == "network"]

    high = expand_high(path_cases) + empty_cases + expand_high(net_cases)

    with open(os.path.join(out_dir, "S5_mapped_cases_high.json"), "w") as f:
        json.dump({"cases": high}, f, ensure_ascii=False)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "5d":
        expand_high_main()
    else:
        merge_low()
```

---

## 逻辑说明

### 过滤规则

剔除任意输入 tensor 元素数 > 1 亿的 path/network case（规则相同）。DYNAMIC（list[dict]）取各子 tensor 元素数最大值。

### supplement_empty_cases 策略

> 函数 `supplement_empty_cases` 定义在 S5_merge_expand.py 中（算子专属部分）。
> 从 1 个随机 path case 模板生成空 tensor 变体。

**通用约束**：
- 签名：`supplement_empty_cases(mapped_path_cases, seed=42)`
- 返回：`list[dict]`，格式同 mapped_path_cases 元素（含 id/params/tensors）
- 每个空 case 的 `_data_range = "normal"`
- ID 后缀：REQUIRED 统一为 `_empty`（含全零兜底）；DYNAMIC 为 `_empty`（位置变体）或 `_all_empty`（全零兜底）。`_is_empty` 通过 `endswith("_empty")` 检测，兼容两种

#### REQUIRED 分支（单 tensor）

> REQUIRED 空 tensor 实现高度算子专属，无通用代码模板。子 agent 根据 S5_mapping_spec.md 和 operator_model 生成具体实现。

**策略**：

Phase 1 — 可空化维度识别：从 S5_mapping_spec.md §shape 构造参数 中，读取每个 shape 参数的控制关系（参数名 → 控制的 tensor → 控制的 dim 索引），确定哪些 input tensor 的哪些维度可以被置零。const input（值和 shape 均固定的 tensor，不随 case 参数变化）不参与空化。

Phase 2 — 变体生成：
- 对每个可空化的（tensor, dim）组合，从 1 个随机 path case 模板生成 1 个变体：仅将该维度置 0，其余维度保持原值
- 跨变体去重（按所有 input shape 签名去重）
- 追加 1 个全零兜底变体：仅将 Phase 1 识别的可空化 tensor 全维度置 0，const input 保持原值
- 按 sync_with 传播零化到关联 tensor，更新 outputs
- 将 params 中受 shape 变更影响的字段同步为与修改后 shape 一致的值

**不变约束**：
- 签名：`supplement_empty_cases(mapped_path_cases, seed=42)`
- 返回：`list[dict]`，格式同 mapped_path_cases 元素
- 每个空 case 的 `_data_range = "normal"`
- 设置 `params["_group"]`（如 `"{affected_key}_empty"`）
- ID 后缀统一为 `_empty`（含全零兜底），供 `_is_empty` 检测

#### DYNAMIC 分支（TensorList）

> DYNAMIC 空 tensor 不逐子 tensor 逐维度枚举（子 tensor 数量可达 50，逐维度枚举导致场景爆炸），改为按子 tensor 在列表中的位置采样。

策略（从 1 个随机 path case 模板生成变体）：
- 筛选 ndim>0 的子 tensor 索引（跳过 scalar，scalar 无法置零单个维度）
- first/middle/last：取筛选后索引列表的首/中/末位置各 1 个变体，将该子 tensor 全维度置 0
- partial：取 ndim 最高的子 tensor，将其最大维度值置 0（保留其余维度结构，仅 ndim≥2 时生成）
- all_empty：所有子 tensor 全维度置 0（兜底，ID 后缀 `_all_empty`）
- 位置去重：若首/中/末指向同一索引（仅 1 个非 scalar 子 tensor），只生成 1 个变体
- sync_with 传播：全维度置 0 → 目标对应位置全维度置 0；partial → 目标同维度置 0
- _group：DYNAMIC 空 case 不设置 _group（继承模板值），空 tensor 检测依赖 ID 后缀（见通用约束），不依赖 _group

**DYNAMIC 代码模板**（子 agent 替换 `name`/`out` 变量值（L226-227 引号内的 {name}/{out}）+ `{target}`/`{target_out}` 文本替换（L272 等引号内的占位符））：

> **输出推导假设**：以下模板假设输出 shape 规则为 `same_as_input`（02 §4）。若算子输出规则为 `derived` 或 `fixed`，子 agent 需修改模板中 outputs 更新逻辑。

```python
def supplement_empty_cases(mapped_path_cases, seed=42):
    if not mapped_path_cases:
        return []
    import random, copy
    rng = random.Random(seed)
    template = rng.choice(mapped_path_cases)

    name = "{name}"
    out = "{out}"

    max_id = 0
    for c in mapped_path_cases:
        cid = c.get("id", "case00000")
        num = int(cid.replace("case", ""))
        if num > max_id:
            max_id = num

    # Phase 1：筛选 ndim>0 的子 tensor 索引
    valid_indices = [i for i, sub in enumerate(template["tensors"]["inputs"][name])
                     if len(sub["shape"]) > 0]

    positions = []
    seen = set()
    for label, pos in [("first_empty", valid_indices[0] if valid_indices else -1),
                       ("middle_empty", valid_indices[len(valid_indices) // 2] if valid_indices else -1),
                       ("last_empty", valid_indices[-1] if valid_indices else -1)]:
        if pos >= 0 and pos not in seen:
            positions.append((label, pos, "full"))
            seen.add(pos)

    if valid_indices:
        partial_idx = max(valid_indices,
                          key=lambda i: len(template["tensors"]["inputs"][name][i]["shape"]))
        partial_ndim = len(template["tensors"]["inputs"][name][partial_idx]["shape"])
        if partial_ndim >= 2:
            positions.append(("partial_empty", partial_idx, "partial"))

    # Phase 2：变体生成
    empty_cases = []
    idx = 0
    for label, tensor_idx, mode in positions:
        variant = copy.deepcopy(template)
        sub = variant["tensors"]["inputs"][name][tensor_idx]
        if mode == "full":
            sub["shape"] = tuple([0] * len(sub["shape"]))
        else:  # partial: 最大维度值置零
            shape = list(sub["shape"])
            max_dim = shape.index(max(shape))
            shape[max_dim] = 0
            sub["shape"] = tuple(shape)
        # 更新 outputs（same_as_input 同步）
        variant["tensors"]["outputs"][out][tensor_idx]["shape"] = sub["shape"]
        # sync_with 传播（无 sync_with 时删除此段）
        if "{target}" in variant["tensors"]["inputs"]:
            target_sub = variant["tensors"]["inputs"]["{target}"][tensor_idx]
            if mode == "full":
                target_sub["shape"] = tuple([0] * len(target_sub["shape"]))
            else:
                target_shape = list(target_sub["shape"])
                if len(target_shape) > max_dim:
                    target_shape[max_dim] = 0
                    target_sub["shape"] = tuple(target_shape)
            if "{target_out}" in variant["tensors"]["outputs"]:
                variant["tensors"]["outputs"]["{target_out}"][tensor_idx]["shape"] = target_sub["shape"]
        variant["id"] = f"case{max_id+1+idx:05d}_{name}_{label}"
        variant["_data_range"] = "normal"
        idx += 1
        empty_cases.append(variant)

    # all_empty 兜底
    all_empty = copy.deepcopy(template)
    for sub in all_empty["tensors"]["inputs"][name]:
        sub["shape"] = tuple([0] * len(sub["shape"]))
    for sub in all_empty["tensors"]["outputs"][out]:
        sub["shape"] = tuple([0] * len(sub["shape"]))
    if "{target}" in all_empty["tensors"]["inputs"]:
        for sub in all_empty["tensors"]["inputs"]["{target}"]:
            sub["shape"] = tuple([0] * len(sub["shape"]))
        if "{target_out}" in all_empty["tensors"]["outputs"]:
            for sub in all_empty["tensors"]["outputs"]["{target_out}"]:
                sub["shape"] = tuple([0] * len(sub["shape"]))
    all_empty["id"] = f"case{max_id+1+idx:05d}_{name}_all_empty"
    all_empty["_data_range"] = "normal"
    empty_cases.append(all_empty)

    return empty_cases
```

### 空 tensor 在 low / high 中的传递

- **low**：由 `supplement_empty_cases` 生成后合并，同其他 case 一样设 `_data_range: "normal"`
- **high**：按 `id` 后缀（`_empty`）从 low 中分离，直接保留（不经过 `expand_high`）

### low 档位：全 normal

所有输入 tensor `_data_range` 统一设为 `"normal"`。

### 输出格式

| 文件 | 格式 |
|------|------|
| `S5_mapped_cases_low.json` | 缩进 + 选择性压缩（`_compact_json` 压缩 shape/dtype/id/_group/_data_range） |
| `S5_mapped_cases_high.json` | 紧凑格式，无缩进 |

---

## 输出示例

以下示例展示 `_compact_json()` 的实际压缩效果，供子 agent 生成脚本时参考。

### 示例：S5_mapped_cases_low.json 压缩前 vs 压缩后

`_compact_json()` 将 5 个 key（`shape`、`dtype`、`id`、`_group`、`_data_range`）从多行缩进展平为单行。

**压缩前**（JSON indent=2 展开形式）：

```json
{
  "cases": [
    {
      "id": "case00000",
      "params": {
        "_group": "{group_id}",
        "{param_1}": "value_1",
        "{param_2}": "value_2"
      },
      "tensors": {
        "inputs": {
          "{tensor_a}": {
            "shape": [
              4,
              8,
              16
            ],
            "dtype": "float16"
          }
        },
        "outputs": {
          "{tensor_out}": {
            "shape": [
              4,
              8,
              16
            ],
            "dtype": "float16"
          }
        }
      }
    }
  ]
}
```

**压缩后**（low.json 实际写入格式；5 个 key 展平为单行，其他结构不变）：

```json
{
  "cases": [
    {
      "id": "case00000",
      "params": {
        "_group": "{group_id}",
        "{param_1}": "value_1",
        "{param_2}": "value_2"
      },
      "tensors": {
        "inputs": {
          "{tensor_a}": {"shape": [4, 8, 16], "dtype": "float16"}
        },
        "outputs": {
          "{tensor_out}": {"shape": [4, 8, 16], "dtype": "float16"}
        }
      }
    }
  ]
}
```

**压缩规则**：
- 被压缩的 key：`shape`、`dtype`、`id`、`_group`、`_data_range`
- **不被压缩**的 key（如 `params`、`tensors`、各 tensor 名）保持原有缩进
- `S5_mapped_cases_low.json` 中所有 case 的 `_data_range` 均为 `"normal"`

---

## 约束

- `_data_range` 存储在 `tensors.inputs.{name}._data_range`，per-input 独立（`set_all_data_range` 统一写入，同时清除 5a 遗留的顶层 `_data_range`）
- DYNAMIC 输入（list[dict]）的每个子 tensor 均写入 `_data_range`，同一个 TensorList 内所有子 tensor 的值相同（语义上共享同一个 data_range）
- 空 tensor 不参与 data_range 交叉，固定 `_data_range: "normal"`
- 空 tensor 通过 `id` 后缀（`_empty`）检测，不依赖 `_group` 字段
- 每个输入 tensor 元素数 ≤ 100,000,000（1 亿）。超出的 path/network case 直接丢弃

---

## 执行

子 agent 须按以下顺序完成：

1. 将上方通用部分写入 `{whitebox}/S5_merge_expand.py`
2. 根据算子 param_type 生成 `supplement_empty_cases`（REQUIRED 或 DYNAMIC 分支），替换脚本中的 `pass` 占位
3. 运行 `python S5_merge_expand.py` → `S5_mapped_cases_low.json`
4. 运行 `python S5_merge_expand.py 5d` → `S5_mapped_cases_high.json`

---

## 检查清单

- [ ] `S5_merge_expand.py` 已写入 `{whitebox}/`
- [ ] `supplement_empty_cases` 已根据算子 param_type 正确生成（REQUIRED 或 DYNAMIC 分支）
- [ ] `python S5_merge_expand.py` 退出码 0
- [ ] `S5_mapped_cases_low.json` 已写入（缩进 + 压缩格式）
- [ ] `python S5_merge_expand.py 5d` 退出码 0
- [ ] `S5_mapped_cases_high.json` 已写入（紧凑格式）
- [ ] 空 tensor 变体数量 > 0
