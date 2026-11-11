# Kirin 平台 ATC 转换指南

本文件面向 Kirin 端侧算子开发，汇总 ATC 在 Kirin 环境下的常用转换流程、参数选型与排错入口。命令语义以 CANN 官方 ATC 文档和实际环境 `atc --help` 输出为准；更完整的参数说明见 [atc_quickref.md](atc_quickref.md) 与 [atc_official_digest.md](atc_official_digest.md)。

## 1. 环境准备

1. 安装包含 ATC 的 CANN toolkit，确认 `${install_path}/cann/bin/atc` 可执行。
2. 在每次转换的同一个 shell 会话中加载环境变量：

```bash
# 路径以实际安装位置为准
source <CANN安装根>/set_env.sh
```

3. 自检 atc 版本与可用性：

```bash
atc --help | head
bash scripts/find_atc.sh
```

> 不要只手动 export 单个变量（例如 `ASCEND_OPP_PATH`）；`set_env.sh` 还会设置 proto、算子库等转换所需变量。

## 2. 转换路线

### 2.1 ONNX 网络模型转 om

```bash
atc --model=/path/to/model.onnx \
    --framework=5 \
    --output=/path/to/out/model_name \
    --soc_version=<soc_version>
```

- `--model`：ONNX 模型路径。
- `--framework=5`：输入框架为 ONNX。
- `--output`：输出 om 的路径与文件名前缀。
- `--soc_version`：目标芯片型号，必须以设备实际型号为准。

### 2.2 AIR 模型转 om

```bash
atc --model=/path/to/model.air \
    --framework=1 \
    --output=/path/to/out/model_name \
    --soc_version=<soc_version>
```

- `--framework=1`：输入为 AIR 格式。
- 其余参数含义与 ONNX 路线一致。

### 2.3 单算子描述文件转单算子 om

单算子路线用于快速验证单个算子的功能。描述文件是 JSON 数组，需给出算子的输入、输出、属性和 dtype/format/shape。

```bash
atc --singleop=/path/to/op_list.json \
    --output=/path/to/out_dir \
    --soc_version=<soc_version>
```

- `--singleop`：单算子描述文件路径。
- `--output`：这里是**目录**，不是文件名。
- 描述文件可用 [scripts/gen_singleop_json.py](../scripts/gen_singleop_json.py) 生成；格式样例见 [assets/singleop_json_example.json](../assets/singleop_json_example.json)。
- `--singleop` 只接受有限的参数组合，完整依赖约束见 [atc_official_digest.md](atc_official_digest.md)。

## 3. Kirin 常用参数

| 参数 | 作用 | 使用建议 |
| --- | --- | --- |
| `--soc_version` | 指定目标芯片 | 必须与设备实际型号一致，否则 om 无法运行 |
| `--input_shape` | 指定输入 shape | 动态 shape 场景需与运行端能力匹配 |
| `--input_format` | 指定输入排布 | 默认值随框架变化，跨框架转换时显式确认 |
| `--precision_mode` | 精度模式 | 默认 `force_fp16`；需要保留原图精度时改用 `must_keep_origin_dtype` 或 `--precision_mode_v2=origin` |
| `--output_type` | 指定输出 dtype | 仅在模型输出类型需要调整时使用 |
| `--out_nodes` | 指定输出节点 | 调试中间结果或裁剪网络时使用 |
| `--log` | 转换日志级别 | 排错时提高日志级别 |
| `--mode=6` | 查看 om 信息 | 转换完成后核对模型输入输出 |

## 4. 常见约束

1. **版本配套**：运行环境 CANN 版本应不低于转换环境；8.5.0 及以上通常需要安装与目标芯片匹配的 ops 算子包。
2. **精度模式**：`force_fp16` 可能引入 Cast 节点或降低 fp32 精度；精度敏感场景先确认 `precision_mode`。
3. **单算子输出**：`--singleop` 的 `--output` 是目录；转换成功后到目录下查找生成的 om。
4. **转换成功不等于设备可运行**：Kirin 端侧还要经过 OMC/HAP 部署，设备运行时可能因 dtype、算子支持度或版本不配套拒载；需以设备侧日志和输出文件为准。
5. **排错入口**：先看转换日志中的 `ERROR`/`EZ` 错误码，再查 [atc_examples.md](atc_examples.md) 与 [atc_official_digest.md](atc_official_digest.md) 的报错速查。

## 5. 参考

- CANN ATC 官方文档：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/atctool/docs/zh/user_guides/atc_tools/overview/learning_guide.md>
- Kirin 端侧实践参考：<https://gitcode.com/cann/cann-recipes-harmony-infer>
- 参数实录：`atc --help`
