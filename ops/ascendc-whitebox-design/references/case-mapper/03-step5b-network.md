# Step 5b：映射网络用例 → S5_mapped_cases_network.json

> **前置条件**：Step 5a 完成（S5_case_mapper.py + S5_mapped_cases_path.json 已写入）
>
> **映射规则**：5b 读 `S5_mapping_spec.md`（算子侧规格）+ `S2P1_low_configs.json`（网络侧结构），自行生成映射规则并写回 `S5_mapping_spec.md` §网络用例映射。

---

输入文件清单见 [00-execution-order.md](00-execution-order.md)。

## 1. 目标

将 `S2P1_low_configs.json` 映射为与 `S2P2_cases.json` 格式一致的 `S2P2_network_cases.json`，然后调用 `S5_case_mapper.main()` 复用 5a 全部管道，附加元数据后输出 `S5_mapped_cases_network.json`。

读 `S5_mapping_spec.md` §输入 tensor 的 `（DYNAMIC）` 标注识别 DYNAMIC 输入。读 `S2P1_low_configs.json` 前 3 条确认字段结构，生成映射规则并写回 `S5_mapping_spec.md` §网络用例映射。

## 2. 数据流

```
S5_mapping_spec.md（算子侧：§dtype ~ §验证规则）
  +
S2P1_low_configs.json（网络侧：config 字段结构）
  │  5b 分析两侧，生成映射规则
  │  写回 S5_mapping_spec.md §网络用例映射（可审查）
  ▼
S2P2_network_cases.json（可审查）
  │  S5_case_mapper.main(network_cases_file, network_out_file, id_prefix="network")
  ▼  ── 内部经 load_mapped_configs → map_case → validate_config ──
S5_mapped_cases_network.json
  │  每条附加 _source / _reason 元数据
```

## 3. 映射推导

### 3.0 生成映射规则

读 `S5_mapping_spec.md`（§dtype ~ §验证规则，算子侧）+ `S2P1_low_configs.json` 前 3 条（网络侧），分析两侧字段对应关系，生成映射规则并写回 `S5_mapping_spec.md` §网络用例映射。规则包括：
- 每个 config 字段对应的算子参数名及映射方式（直接 / 乘积 / 透传）
- DYNAMIC 输入的 tensor_count 来源（config 提供 / 固定值）
- 缺失参数的默认值

后续 §3.1-3.6 的模式用于将生成的规则翻译为代码。

### 3.1 直接映射

config 中某字段直接对应算子参数。对应：`case["{算子参数名}"] = cfg["{字段路径}"]`。

### 3.2 乘积映射

多个 config 字段的乘积对应一个算子参数。对应：`case["{算子参数名}"] = cfg["{字段1}"] * cfg["{字段2}"]`。

### 3.3 dtype 映射

读 S5_mapping_spec.md §dtype 获取 dtype 控制参数名和合法值，对照 config 的 dtypes 字段确定映射。值直接透传。

### 3.4 attr 补全

对照 S5_mapping_spec.md §shape 构造参数 + §属性，对 low_config 中缺失的算子参数取 default 值，不随机采样。

### 3.5 tensor_count 映射（DYNAMIC 专用）

读 S5_mapping_spec.md §输入 tensor 中标注 `（DYNAMIC）` 的输入，对照 config 字段确定 tensor_count 来源：

- **config 提供**：config 中有对应字段（如 `shapes.tensor_count`）→ `case["{name}_count"] = cfg["{字段路径}"]`
- **固定值**：config 中无对应字段 → `case["{name}_count"] = {N}`（N 在 [min, min(max, 50)] 范围内选取）

### 3.6 _group

所有网络用例的 `_group` 固定为 `"network"`。

## 4. 代码模板

```python
def map_network_to_path_cases(configs_file):
    """
    将 S2P1_low_configs.json 转换为 S2P2_network_cases.json
    子 agent 从 §3.0 生成的映射规则翻译生成
    """
    with open(configs_file) as f:
        configs = json.load(f)

    cases = []
    for config in configs:
        cfg = config
        case = {}

        # ============================================================
        #  以下内容由子 agent 翻译生成
        # ============================================================

        # 直接映射
        case["{算子参数名}"] = cfg["{语义名}"]

        # 乘积映射
        case["{算子参数名}"] = cfg["{语义名1}"] * cfg["{语义名2}"]

        # dtype 映射
        case["{算子dtype_param}"] = cfg["{语义名}"]

        # attr 补全（不采样，取 default）
        case.setdefault("{算子参数名}", {default})

        # tensor_count 映射（DYNAMIC 专用）
        case["{name}_count"] = cfg["{语义名}"]  # 或固定值

        # _group 固定
        case["_group"] = "network"

        # ============================================================

        cases.append(case)

    return cases
```

## 5. 生成 S2P2_network_cases.json + 调 main

```python
import json, os
from S5_case_mapper import main

out_dir = os.path.dirname(os.path.abspath(__file__))
low_configs_file = os.path.join(out_dir, "S2P1_low_configs.json")
network_cases_file = os.path.join(out_dir, "S2P2_network_cases.json")
network_out_file = os.path.join(out_dir, "S5_mapped_cases_network.json")

# 1. 语义→算子参数，写 S2P2_network_cases.json（可审查）
cases = map_network_to_path_cases(low_configs_file)
with open(network_cases_file, "w") as f:
    json.dump(cases, f, indent=2)
print(f"Written {len(cases)} cases to S2P2_network_cases.json")
# → 人工审查: 字段名、_group、乘积值、attr 默认值

# 2. 复用 5a 管道
main(network_cases_file, network_out_file, id_prefix="network")

# 3. 附加元数据
low_configs = json.load(open(low_configs_file))
with open(network_out_file) as f:
    mapped = json.load(f)
for i, mc in enumerate(mapped["cases"]):
    mc["_source"] = low_configs[i].get("source")
    mc["_reason"] = low_configs[i].get("reason")
with open(network_out_file, "w") as f:
    json.dump(mapped, f, indent=2)
print(f"Written S5_mapped_cases_network.json")
```

## 6. 输出

- `S2P2_network_cases.json` — 中间产物，与 `S2P2_cases.json` 格式一致，所有字段为算子参数名，`_group: "network"`
- `S5_mapped_cases_network.json` — 最终输出，与 `S5_mapped_cases_path.json` 格式一致，每条额外包含 `_source` / `_reason`

> **验证说明**：`main()` 内部已对每条 network case 执行 L1 校验（`validate_config`）。L2-L4 验证（`S5_verify_mapper.py`）仅在 5a 对 path cases 执行——其目的是验证 mapper 函数本身的正确性，5a 通过后即可认为 `map_case`/`validate_config` 逻辑可靠，network cases 复用同一函数无需重复验证。
