<!--
来源：CANN 官方 ATC 文档
- 在线：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/atctool/docs/zh/user_guides/atc_tools/overview/learning_guide.md
- 仓库：https://gitcode.com/cann/ge/tree/master/docs/zh/user_guides/atc_tools
本文件为官方文档常用重要内容的离线提炼，以官方最新版本为准。
联网查找：需要更详细内容时按文件末尾「联网查找指引」访问官方仓库。
-->

# ATC 官方文档提炼（常用重要内容）

## ATC 简介与工作流程

ATC（Ascend Tensor Compiler）是 CANN 体系下的模型转换工具，将开源框架网络模型或 Ascend IR 单算子描述文件（JSON）转换为适配 AI 处理器的离线模型（`.om`）。转换过程含算子调度优化、权重数据重排、内存使用优化。

两条路线（官方《ATC 概述》）：

1. **框架网络模型**：Parser 解析为中间态 IR Graph → 图准备/拆分/优化/编译 → om → 板端经模型加载/执行接口推理
2. **单算子描述文件**：JSON 经单算子编译 → 单算子 om → 板端经单算子模型加载接口（如 `aclopSetModelDir`）、执行接口（如 `aclopExecuteV2`）验证

关键概念：GE（Graph Engine，图编译运行控制中心）；数据排布格式 NCHW/NHWC/NC1HWC0（卷网络注意 `--input_format`）。

## 环境准备

```bash
# 安装 Toolkit 后，每次使用前 source（官方方式，禁止手动 export 单个变量替代）
source <CANN安装根>/set_env.sh

which atc && atc --help    # 验证环境
```

转换在 host 侧进行，不依赖 NPU 设备；aarch64 环境转换耗时可能明显偏长（官方 FAQ）。

## 各框架转换命令（官方 quick_start 原例）

`--framework`：0 Caffe / 1 MindSpore / 3 TensorFlow / 5 ONNX。

```bash
# ONNX
atc --model=$HOME/module/resnet50_official.onnx --framework=5 --output=$HOME/module/out/onnx_resnet50 --soc_version=<soc_version>

# TensorFlow（注意 --input_format 默认 NHWC）
atc --model=$HOME/module/resnet50_tensorflow_1.7.pb --framework=3 --output=$HOME/module/out/tf_resnet50 --soc_version=<soc_version>

# Caffe（必带 --weight）
atc --model=$HOME/module/resnet50.prototxt --weight=$HOME/module/resnet50.caffemodel --framework=0 --output=$HOME/module/out/caffe_resnet50 --soc_version=<soc_version>

# MindSpore（.air）
atc --model=$HOME/module/resnet50_export.air --framework=1 --output=$HOME/module/out/resnet50_air --soc_version=<soc_version>
```

- 静态 shape 模型若未自带完整输入信息需 `--input_shape="输入名:n,c,h,w"`（多输入分号分隔，整体加双引号）
- 动态 shape：`--dynamic_batch_size="1,2,4,8"` / `--dynamic_image_size` / `--dynamic_dims`
- 输出节点定制：`--out_nodes="node:0"`；输出类型：`--output_type`

## 版本配套（官方兼容性说明）

| 规则 | 说明 |
|------|------|
| 运行环境 ≥ 转换环境 | 运行环境 CANN 版本不低于转换 om 时的版本 |
| 向下兼容 4 个版本周期 | 低版本转换的 om 可在高版本环境运行 |
| 必须装匹配 ops 包 | 8.5.0 及之后版本，转换必须安装与目标芯片匹配的 ops 算子包，否则编译失败 |
| 完全一致场景 | 动态 shape 算子、昇腾虚拟化实例：两端版本必须一致（否则加载报 E19999） |

## 单算子模型转换（--singleop）

### 描述文件格式（Ascend IR JSON）

顶层为 OpDesc 数组：`op`（算子类型）/ `name`（可选）/ `input_desc` / `output_desc` / `attr`；输入输出项含 `format` / `type` / `shape` / `name`（可选）。type 取值如 `float16`/`float32`/`int32`/`bool`；常用 format：`ND`、`NCHW`、`NC1HWC0`。

官方 Add 示例（ND）：

```json
[
  {
    "op": "Add",
    "name": "add",
    "input_desc": [
      {"format": "ND", "shape": [3,3], "type": "int32"},
      {"format": "ND", "shape": [3,3], "type": "int32"}
    ],
    "output_desc": [
      {"format": "ND", "shape": [3,3], "type": "int32"}
    ]
  }
]
```

带属性示例（NCHW，节选）：`"attr": [{"name": "strides", "type": "list_int", "value": [1,1,1,1]}, ...]`。
各 Format/多输入/带 attr 的完整分场景样例见官方《配置文件样例》（联网查找指引见文末）。

### 转换命令与参数闭集

```bash
atc --singleop=$HOME/singleop/add.json --output=$HOME/singleop/out/op_model --soc_version=<soc_version>
```

`--output` 此处为**目录**。`--singleop` 场景只能配合官方按产品列出的有限参数集合（`--output`、`--soc_version` 必填）。Atlas A2 / A3 / Ascend 950PR&950DT 闭集（官方 `--singleop` 文档）：

`--output`、`--soc_version`、`--precision_mode`、`--precision_mode_v2`、`--op_select_implmode`、`--optypelist_for_implmode`、`--enable_small_channel`、`--sparsity`、`--allow_hf32`、`--log`、`--debug_dir`、`--op_compiler_cache_mode`、`--op_compiler_cache_dir`、`--atomic_clean_policy`、`--customize_dtypes`、`--is_weight_clip`

Atlas 推理/训练系列略有差异（如增加 `--core_type`/`--aicore_num`，950 增加 `--jit_compile`），以官方 `--singleop.md` 各产品条目为准。整网参数（如 `--input_shape`）不适用于 `--singleop`。

### 依赖约束（官方）

- 仅使用 TBE 算子（不查 AI CPU）时需设置并事后 unset：
  `export ASCEND_ENGINE_PATH=${INSTALL_DIR}/lib64/plugin/opskernel/libfe.so:${INSTALL_DIR}/lib64/plugin/opskernel/libge_local_engine.so:${INSTALL_DIR}/lib64/plugin/opskernel/librts_engine.so`
- 板端推理需调用单算子模型加载接口（如 `aclopSetModelDir`）+ 执行接口（如 `aclopExecuteV2`）

## 高频参数要点

完整参数速查表见 [atc_quickref.md](atc_quickref.md)（含取值、默认值、易错约束）。最关键四条：

| 参数 | 要点 |
|------|------|
| `--soc_version` | **必配**，必须为运行阶段芯片型号。查询：A2/A3/950 用 `npu-smi info -t board -i <id>`（A3 加 `-c <chip_id>`）取 Chip Name_NPU Name；其他产品 `npu-smi info` 的 Name 前加 Ascend |
| `--precision_mode` | 默认 `force_fp16`（fp32/bf16 强制降 fp16，可能插 Cast）；保持原图精度用 `must_keep_origin_dtype`；与 `--precision_mode_v2`（官方推荐，默认 `fp16`，保持原图精度用 `origin`）互斥 |
| `--mode` | 0 默认转换；1 转 JSON；3 仅 pre-check；5 ge dump 转 JSON；6 显示模型信息（`--om` 配合）；30 转 execute-om |
| `--log` / `--op_debug_level` | 排障标配：`--log=debug`；op_debug_level 1~4 生成 CCE 文件/保留 kernel（详见 quickref） |

## 报错速查

| 报错 | 处理 |
|------|------|
| `EZ3003: No supported Ops kernel and engine are found` | 官方 Solution：检查 OPP 组件安装；实际排查顺序见 [atc_examples.md](atc_examples.md)（含典型回显） |
| `E19999: The opp version of the model does not match...` | 动态 shape/虚拟化场景两端 opp 版本不一致 → 两端同版本，重转模型（官方 FAQ《算子库包版本问题导致加载单算子失败》） |
| `The required parameter --xxx for ATC is empty` | 参数为空或值未加引号 |
| Caffe `layers` 结构不支持 | 转换为 layer 格式 |
| 量化类失败（校准集不匹配/不支持量化的层/build graph failed 等） | 官方 FAQ 专章（见下方索引） |
| aarch64 转换慢 | 官方 FAQ 已知现象，可换 x86_64 环境 |

官方 FAQ 全目录（atc_tools/FAQ/）：aarch64 转换慢、AIPP 视频格式判断、算子对应关系查询、量化层不支持/校准集不匹配/输入过大/build 失败、算子库包版本问题（E19999）、`nca load model failed, ret=254`。

## 联网查找指引

需要本文件未覆盖的细节时，访问官方仓库（勿整体 clone，按文件取用）：

```bash
# 网页浏览
https://gitcode.com/cann/ge/tree/master/docs/zh/user_guides/atc_tools/<子目录>

# 原始文件直接抓取（示例：某参数说明）
https://raw.gitcode.com/cann/ge/raw/master/docs/zh/user_guides/atc_tools/CLI_options/--singleop.md
```

| 想查什么 | 位置 |
|---------|------|
| 某参数完整说明 | `CLI_options/--<参数名>.md` |
| 单算子 JSON 各场景配置样例 | `sinlgeop_model_convert/config_file_example.md`、`singleop_desc_intro.md` |
| 快速入门完整流程 | `overview/quick_start.md`、`overview/environment_setup.md` |
| 高级功能（AIPP、动态 shape、专题） | `AIPP/`、`basic_feature/`、`custom_network_modify_caffe/`、`custom_network_modify_tf/` |
| FAQ 原文 | `FAQ/<条目>.md` |
