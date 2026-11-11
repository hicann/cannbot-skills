<!--
来源说明：
1. 参数语义整理自官方文档（gitcode.com/cann/ge CLI_options/，提炼与核对见 atc_official_digest.md）。
2. 参数清单与取值经实际环境 atc --help 实录交叉核对（标注"实录"）。
   与目标环境 atc --help 有差异时以工具输出为准。
-->

# ATC 高频参数速查

## 参数分组总览（实录）

```
atc 参数
├── General      --mode / --raw_ge_options
├── Input        --model / --weight / --framework / --input_shape / --dynamic_*
├── Output       --output / --output_type / --json / --check_report
├── Target       --soc_version / --core_type / --aicore_num
├── Feature      --out_nodes / --insert_op_conf / --external_weight ...
├── Model Tuning --fusion_switch_file / --buffer_optimize / --oo_level ...
├── Op Tuning    --precision_mode / --precision_mode_v2 / --jit_compile ...
└── Debug        --op_debug_level / --log / --debug_dir / --dump_mode ...
```

## 高频参数表

### 输入输出（必会）

| 参数 | 说明 |
|------|------|
| `--model` | 原始模型文件路径 |
| `--weight` | 权重文件（**Caffe 必选**） |
| `--framework` | 0 Caffe；1 MindSpore；3 TensorFlow；5 ONNX |
| `--input_shape` | 静态输入 shape 或动态 shape range，多节点分号分隔，整体加双引号：`"input:1,3,224,224"`；模型未自带完整输入信息时必填，建议显式指定 |
| `--input_format` | 输入数据格式（NCHW/NHWC 等；Caffe/MindSpore/ONNX 默认 NCHW，TensorFlow 默认 NHWC） |
| `--output` | 输出路径前缀（不加后缀，自动补 .om）；`--singleop` 时为**目录** |
| `--out_nodes` | 指定输出节点：`"node_name1:0;node_name2:0"` |
| `--mode` | 0 默认转换；1 转 JSON 查看（配 `--dump_mode=1` 带 shape）；3 仅 pre-check；5 ge dump txt 转 JSON；6 显示模型信息；30 原图转 execute-om（实录） |
| `--json` / `--om` | `--mode=1` 的输出 json；`--mode=6` 配合 `--om` 查看模型信息 |

### 目标芯片（必会）

| 参数 | 说明 |
|------|------|
| `--soc_version` | **必配**。必须为模型运行阶段所用 AI 处理器型号。取值查询见下节 |
| `--core_type` | AiCore（默认）/ VectorCore；含 Cube 算子只能 AiCore |
| `--aicore_num` | 指定 AI Core 数量 |

### soc_version 取值查询（官方文档口径）

- Atlas A2/A3、Ascend 950 系列：在装卡服务器执行 `npu-smi info -t board -i <id>`（A3 加 `-c <chip_id>`），取 **Chip Name_NPU Name**（如 `Ascendxxx_1234`）
- 其他产品（310p/310b/910 等）：`npu-smi info` 的 Name 前加 `Ascend`
- 麒麟平台：型号以设备实际为准（映射见 npu-arch）
- 同一 soc_version 的不同产品可共用一次转换；版本配套规则见 SKILL.md 约束 3

### 精度（高频坑区）

| 参数 | 说明 |
|------|------|
| `--precision_mode` | `force_fp16`（**默认**，fp32/bf16 强制选 fp16）/ `force_fp32` / `cube_fp16in_fp32out` / `allow_fp32_to_fp16` / `must_keep_origin_dtype`（保持原图精度）/ `allow_mix_precision` 系列（实录） |
| `--precision_mode_v2` | 新版：`fp16`（默认）/ `origin` / `cube_fp16in_fp32out` / `mixed_float16` / `mixed_bfloat16` 等；**与 --precision_mode 互斥，官方推荐优先 v2** |
| `--keep_dtype` | 按配置文件保持个别算子精度（`must_keep_origin_dtype` 时不生效） |
| `--op_precision_mode` | 算子级精度配置 .ini 路径 |
| `--op_select_implmode` | high_performance（默认）/ high_precision 等算子实现选择 |

### 动态 shape

| 参数 | 说明 |
|------|------|
| `--dynamic_batch_size` | 动态 batch：`"1,2,4,8"` |
| `--dynamic_image_size` | 动态分辨率：`"h1,w1;h2,w2"` |
| `--dynamic_dims` | 通用动态维度：`"dim1_v1,dim1_v2;dim2_v1"` |

### 调试（排障必会，实录）

| 参数 | 说明 |
|------|------|
| `--log` | 日志级别：debug / info / warning / error / null（默认） |
| `--op_debug_level` | TBE 算子调试：1 生成 CCE 文件与 Python-CCE 映射 json；2 另加 CCE 编译 -O0-g；3 保留 kernel 文件（.o/.json）；4 保留 kernel 并另生成 CCE 与 UB 融合描述 json |
| `--debug_dir` | 算子编译中间文件目录（默认 ./kernel_meta） |
| `--op_compiler_cache_dir/mode` | 算子编译缓存路径与开关（disable/enable/force） |
| `--save_original_model` | true 时同时输出原始模型（仅见于实际环境 atc --help 实录，官方文档未列出，以实际 `--help` 为准） |
| `--check_report` | pre-check 报告文件（默认 check_result.json） |

### 优化（按需）

| 参数 | 说明 |
|------|------|
| `--fusion_switch_file` | 融合规则开关配置（图融合/UB 融合）——算子融合异常排查入口 |
| `--disable_reuse_memory` | 1 = 关闭内存复用 |
| `--external_weight` | 1 权重落文件；2 全部权重单文件 |
| `--jit_compile` | 0 二进制优先；1（默认）在线编译；2 静态在线 + 动态二进制 |
| `--deterministic` | 1 = 确定性计算 |
| `--compression_optimize_conf` | 量化压缩相关配置 |
| `--insert_op_conf` | AIPP 等前置算子插入配置 |

## 参数间约束（易错）

1. `--precision_mode` 与 `--precision_mode_v2` **互斥**
2. `--singleop` 只能配合官方文档列出的有限参数集合（必含 `--output`、`--soc_version`；按产品略有差异，见 [atc_official_digest.md](atc_official_digest.md)「单算子模型转换」），`--input_shape` 等整网参数不适用
3. `--input_shape` 中节点名必须与模型实际输入名一致（可用 netron 或 `--mode=3` pre-check 确认）
4. 路径参数中含分号会解析异常
5. 量化相关报错优先查官方 FAQ（[atc_official_digest.md](atc_official_digest.md) 报错速查与 FAQ 索引）
