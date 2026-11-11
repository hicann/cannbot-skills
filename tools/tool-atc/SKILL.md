---
name: tool-atc
description: "ATC（Ascend Tensor Compiler）模型转换：框架网络模型或单算子描述文件转 om 离线模型，含参数选型与报错排查。触发场景：用户提到 atc、模型转换、onnx/caffe/tensorflow 转 om、单算子转离线模型、转换报错时使用。"
---

# ATC 模型转换工具

ATC（Ascend Tensor Compiler）是 CANN 体系下的模型转换工具：开源框架网络模型或 Ascend IR 单算子描述文件（JSON）→ 适配 AI 处理器的离线模型（`.om`）。转换过程含算子调度优化、权重数据重排、内存优化。

> **目录定位**：atc 属 CANN 工具链，位于 `tools/` 下（与 asys 等维测工具并列）；同时是算子开发工作流「算子/模型 → 离线模型 → 设备验证」的必经环节。

> **官方文档提炼**：官方文档常用重要内容已提炼为 [references/atc_official_digest.md](references/atc_official_digest.md)（简介/各框架命令/单算子转换/版本配套/报错/联网查找指引）；麒麟平台侧的 ATC 使用指南见 [references/atc_tools_guide.md](references/atc_tools_guide.md)。本文件未覆盖的细节可联网访问官方仓查找（指引见 digest 文末）。

## 工作流程

1. **环境准备与验证**：先 `source CANN 安装目录下的 set_env.sh`，确认 `atc` 可用（[脚本](scripts/find_atc.sh)可定位并自检）
2. **解析场景**：判断转换路线——框架网络模型（`--model`）或单算子描述文件（`--singleop`）
3. **生成命令**：结合 [references/](references/) 文档生成 atc 命令，解释 `--soc_version` 等关键参数
4. **真实环境验证**：执行转换，以转换日志、生成的 om 模型路径作为验证结果；报错时按 [troubleshooting](references/atc_examples.md) 排查

## 输出要求

- 针对场景给出 atc 命令（含 `--soc_version` 等关键参数解释）与生成的 om 模型路径
- 验证通过的转换附真实执行回显与日志要点
- 未经真实环境验证的命令明确标注"未验证，来源：官方文档"
- 禁止编造命令参数；不确定的参数语义查 [atc_official_digest.md](references/atc_official_digest.md)、联网查官方仓或 `atc --help`

## 两条转换路线

### 路线 1：框架网络模型 → om

```bash
atc --model=model.onnx --framework=5 \
    --soc_version=<soc_version> \
    --input_shape="input:1,3,224,224" \
    --output=./model_out
```

`--framework`：0 Caffe / 1 MindSpore / 3 TensorFlow / 5 ONNX；Caffe 需 `--weight`。
完整框架示例见 [atc_official_digest.md](references/atc_official_digest.md)「各框架转换命令」与 [atc_tools_guide.md](references/atc_tools_guide.md)。

### 路线 2：单算子描述文件 → 单算子 om（功能验证）

```bash
atc --singleop=op_list.json --soc_version=<soc_version> --output=<目录>
```

- 描述文件为 JSON 数组（`op`/`input_desc`/`output_desc`/`attr`，输入输出含 `format`/`type`/`shape`）
- `--output` 此处为**目录**；`--singleop` 只能配合有限参数集合（见 [atc_quickref.md](references/atc_quickref.md)）
- 描述文件可用 [scripts/gen_singleop_json.py](scripts/gen_singleop_json.py) 生成；格式样例见 [assets/singleop_json_example.json](assets/singleop_json_example.json)
- 详细配置见 [atc_official_digest.md](references/atc_official_digest.md)「单算子模型转换」

## 使用前必读的七条约束

1. **`--soc_version` 必须与运行环境芯片型号一致**，否则转换出的模型无法运行。查询方法见 [atc_quickref.md](references/atc_quickref.md)
2. **转换不依赖 NPU 设备**；环境变量统一由 `set_env.sh` 提供，手动 export 单个变量（如 `ASCEND_OPP_PATH`）会遗漏必要变量
3. **版本配套**：运行环境 CANN 版本 ≥ 转换环境（兼容 4 个版本周期）；8.5.0+ 转换必须安装与目标芯片匹配的 ops 算子包；动态 shape/虚拟化实例场景要求两端版本完全一致（否则加载报 E19999）
4. **`--precision_mode` 默认 `force_fp16`**：原图 fp32 算子会被强制降精度为 fp16 并可能插入 Cast 节点。保持原图精度用 `--precision_mode=must_keep_origin_dtype`（推荐用新版 `--precision_mode_v2=origin`）
5. **模型量化**（`--compression_optimize_conf` 等）与算子融合（`--fusion_switch_file`）相关异常先查官方 FAQ（[atc_official_digest.md](references/atc_official_digest.md) 报错速查与 FAQ 索引）与 precision 参数文档，勿凭经验猜测
6. **vendor 安装与 atc 执行须同一 set_env 会话**：跨会话环境漂移导致 proto 加载不一致，行为差异难排查
7. **转换成功 ≠ 设备端可运行**：端侧场景 atc rc=0 仅代表 host 转换链路通过，设备端运行时栈仍可能拒载；验证须 A/B 对照 + 设备侧输出文件与日志取证

## 转换自检清单

```bash
source <install>/set_env.sh       # 每次使用前
atc --help                        # 全量参数（以实际环境输出为准）
bash scripts/find_atc.sh          # 定位 atc + 版本自检
```

转换成功标志：无 `ERROR`/`EZ` 报错且 om 产物生成；验证模型信息可用 `atc --mode=6 --om=<model>.om`。

## 参考文档索引

| 文档 | 内容 | 来源 |
|------|------|------|
| [atc_official_digest.md](references/atc_official_digest.md) | 官方文档常用重要内容提炼：ATC 简介与工作流程、环境准备、各框架转换命令、版本配套、单算子转换（JSON 格式/参数闭集/依赖约束）、报错速查、联网查找指引 | 官方文档提炼 |
| [atc_tools_guide.md](references/atc_tools_guide.md) | Kirin 平台 ATC 转换指南：环境准备、ONNX/AIR/单算子转换、常用参数与约束 | 项目整理（参考 Kirin 实践） |
| [atc_quickref.md](references/atc_quickref.md) | 高频参数速查（framework/input_shape/soc_version/precision_mode/动态 shape/调试），参数清单来自实际环境 atc --help 实录 | 官方文档 + 工具实录 |
| [atc_examples.md](references/atc_examples.md) | 场景示例、命令序列（脱敏占位符）、典型回显、报错排查（含 E20002 opp 完整性判据） | 官方文档 + 工具实录 |
| [assets/singleop_json_example.json](assets/singleop_json_example.json) | 单算子描述文件格式样例（顶层数组、type/format 字符串标量） | 端侧设备验证可转换 |

## 脚本工具

| 脚本 | 用途 |
|------|------|
| `scripts/find_atc.sh` | 定位 atc 可执行文件、读取 CANN 版本、检查内置算子配置目录 |
| `scripts/gen_singleop_json.py` | 生成单算子描述 JSON（dtype/format 校验），输出转换命令与产物命名提示 |

## 与其他 Skill 的关系

- `tool-hdc`：om 转出后推送到设备执行、结果拉回等环节使用该 skill
- `ascendc-precision-debug`：ATC 编译链路引入精度差异（算子融合/降精度/format 转换）的调试方法

## 信息来源

- 官方文档提炼：gitcode.com/cann/ge `docs/zh/user_guides/atc_tools`（见 atc_official_digest.md；更多细节按其文末指引联网查找）
- Kirin 端侧实践参考：gitcode.com/cann/cann-recipes-harmony-infer（命令与实践参考，非逐字复制）
- 参数清单另以实际环境 `atc --help` 实录交叉核对；与文档有差异时以工具输出为准
- 回显示例见 [atc_examples.md](references/atc_examples.md)「典型回显」；其中未标注为示例的命令使用前需按官方文档核对
