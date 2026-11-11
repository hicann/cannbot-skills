<!--
来源说明：
1. 「场景示例」命令来自官方文档（atc_official_digest.md 提炼源与 atc_tools_guide.md）。
2. 「命令序列」已去除个人路径/设备信息并抽象化为占位符。
3. 「典型回显」为命令输出示例；未在当前环境复验的命令，使用前按官方文档核对。
-->

# ATC 场景示例、实战命令序列与报错排查

## 场景示例（官方文档口径）

### ONNX 模型转 om 并指定 soc_version

```bash
atc --model=resnet50.onnx --framework=5 \
    --soc_version=<soc_version> \
    --input_shape="input:1,3,224,224" \
    --output=./out/resnet50
```

- `--soc_version` 必须与运行环境芯片型号一致（查询见 atc_quickref.md）
- `--output` 生成 `./out/resnet50.om`
- 验证模型信息：`atc --mode=6 --om=./out/resnet50.om`

### Caffe / TensorFlow / MindSpore 转换

```bash
atc --model=model.prototxt --weight=model.caffemodel --framework=0 --soc_version=<soc> --output=./m
atc --model=model.pb --framework=3 --soc_version=<soc> --input_format=NHWC --input_shape="input:224,224,3" --output=./m
atc --model=model.air --framework=1 --soc_version=<soc> --output=./m
```

### 单算子描述文件转换

```bash
# op_list.json 生成：python3 scripts/gen_singleop_json.py --op Add \
#   --input x1:16,16:float16 --input x2:16,16:float16 --output y:16,16:float16 -o op_list.json
atc --singleop=op_list.json --soc_version=<soc_version> --output=./op_model
```

JSON 格式与分场景配置样例见 [atc_official_digest.md](atc_official_digest.md)「单算子模型转换」。

## 命令序列（脱敏占位符）

场景：自定义算子（vendor 包安装后）转换单算子离线模型并部署验证。

```bash
# 0. 环境准备与自检
source <install>/set_env.sh
which atc && atc --help | head -5

# 1. 单算子描述文件（op 为已注册算子名：内置或 vendor 自定义）
#    <op>.json：[{"op":"<OpName>","input_desc":[{"format":"ND","type":"<dtype>","name":"x1","shape":[...]}，...],"output_desc":[...],"attr":[]}]
atc --singleop=<op>.json --soc_version=<soc_version> --output=<out_dir> [--op_debug_level=1]

# 2. onnx 路线（模型含该自定义算子时）
atc --model=<model>.onnx --framework=5 --soc_version=<soc_version> --output=<out_dir>/<name>

# 3. fp32 输入保持原始精度（避免默认 force_fp16 自动降精度插 Cast）
atc --singleop=<op>.json --soc_version=<soc_version> --output=<out_dir> \
    --precision_mode=must_keep_origin_dtype
# 或 onnx 路线同参数

# 4. 转换产物推板执行 → 见 tool-hdc skill
```

### 经验条目

1. **fp32 输入转换失败先查精度模式**：默认 `force_fp16` 下 FE 精度适配 pass 可能把 fp32
   输入降精度为 fp16 并插入 `trans_Cast` 节点；若目标平台无该 Cast 内核即报 EZ3003。
   解决：`--precision_mode=must_keep_origin_dtype`（或 v2 的 `origin`）。
2. **改 vendor 算子包代码后必须卸载重装**：安装器对已存在 vendor 目录保留旧文件，覆盖安装不生效。
   另一隐蔽根因：**CPack staging 目录（build/_CPack_Packages）增量构建不刷新**——`.run` 时间戳是新的
   但包内 `dynamic/<arch>/<op>_tiling_key.h` 等源码仍是旧版（TPL SEL 行数不变），atc 侧按陈旧 SEL
   推断 tiling key 集合，与 kernel 实际返回不符（典型报错 `KeyError: '1'` 或次生 EZ3003）。
   判别：`grep -c <SEL宏> build/_CPack_Packages/.../<op>_impl/dynamic/<arch>/<op>_tiling_key.h`
   行数须与工程源码一致；处置：`rm -rf build` 全量重编 → `scripts/uninstall.sh`（位于
   `opp/vendors/<vendor>/scripts/`）→ `.run --quiet --force` 重装。
   行为变化提醒：proto 放行新 dtype（如 fp16）后，fp32 输入可能从「不插 Cast」变「插 Cast」
   （force_fp16 精度模式下的自动降精度行为，见经验 1 的排查思路）。
3. **单算子产物名含 shape/dtype 枚举**，修改描述文件后重新转换产物文件名会变化，
   设备侧启动参数引用的模型文件名要同步更新。
4. **隔离实验定位报错归属**：最小 singleop JSON 单独转换，区分「算子问题」vs「环境/图优化问题」。
   日志配合限制：`ASCEND_PROCESS_LOG_PATH` 指向目录须**提前创建**；atc **主进程**的图优化日志
   （如 trans_Cast 插入点）不保证落盘（子进程日志可见），因此 FE 行为排查首选最小隔离实验：
   singleop ↔ onnx 两路线对照、逐 dtype 对照、纯 Cast 图对照（"2 个 fp32 输入 ↔ 2 个 Cast"
   模式吻合度定位法）。
5. **vendor 安装与 atc 执行须同一 set_env 会话**：装完 vendor 的 shell 与跑 atc 的 shell 必须
   source 过同一 `set_env.sh`，跨会话环境漂移导致 proto 加载不一致（行为差异难排查）。

## 单算子端侧产物实战补充

- **产物扩展名**：部分端侧平台单算子转换产物扩展名为 `.omc`（非 `.om`）；部署脚本以 glob
  获取产物名，不要硬编码。
- **singleop 产物名 dtype 编码对照**（ge::DataType 枚举值）：fp32=0 / fp16=1 / int8=2 /
  int32=3 / uint8=4 / int16=6 / uint16=7 / uint32=8 / bool=12。产物名形如
  `0_<Op>_0_2_2_6_...omc`（首段=dtype 编码），核对产物与描述文件是否一致时可用。
- **onnx 路线约束**：opset 11 + ir_version=6（高版本 ir 会被 parser 拒绝）；value_info 的
  shape 严格按算子规格构造（超规格报 E90003）；输出 dtype 与算子一致，不引入额外节点；
  算子名须与 ONNX 标准算子同名（无 onnx_plugin 时）或工程提供 framework/onnx_plugin。
- **fp32/bf16 等高精度输入**两路线均须 `--precision_mode=must_keep_origin_dtype`（force_fp16
  默认会插 trans_Cast，见经验 1）。
- **转换成功 ≠ 设备端可运行**：atc rc=0 且无错误码只代表 host 侧转换链路通过；设备端运行时栈
  （图校验、张量尺寸查询等）仍可能拒载某些 dtype 或图形态（实例：无符号整型 dtype 可能被设备
  图校验拒绝，`aa start` 返回成功但不产生输出，设备 hilog 可取证）。端侧验证应保留 A/B 对照
  （同链路已知可运行 dtype 复推）与设备日志手段，失败时先定界 host 转换 vs 设备运行时。

## 典型回显

### 示例 1：内置算子 singleop 转换（工具链内置算子库不含该算子时）

命令：`atc --singleop=add_16x16_f16.json --output=/tmp/atc_verify --soc_version=<soc>`

```text
ATC start working now, please wait for a moment.
...
ATC run failed, Please check the detail log, Try 'atc --help' for more information
[PID: xxx] 20xx-xx-xx-xx:xx:xx.xxx Unsupported_Operator(EZ3003): No supported Ops kernel and engine are found for [Add], optype [Add].
        Possible Cause: The operator is not supported by the system. Therefore, no hit is found in any operator information library.
        Solution: 1. Check that the OPP component is installed properly. 2. Submit an issue to request for the support of this operator type.
TraceBack (most recent call last):
        Failed to select engine for [Add][Add].[FUNC:operator()][FILE:engine_place.cc][LINE:149]
```

**结论**：EZ3003 表示目标平台的算子信息库不含该算子（或插入的 Cast 无内核）。
转换自定义算子前需先安装对应 vendor 算子包（含 op_proto/op_api），或使用内置算子库完整的工具链。
排查顺序：① OPP 组件是否安装完整 → ② 目标 soc 的算子信息库是否覆盖该算子 → ③ vendor 包是否安装。

### 示例 2：atc --help 正常输出（工具可用的判据）

```text
ATC start working now, please wait for a moment.
usage: atc <args>
generate offline model example:
atc --model=./alexnet.prototxt --weight=./alexnet.caffemodel --framework=0 --output=./domi --soc_version=<soc_version>
generate offline model for single op example:
atc --singleop=./op_list.json --output=./op_model --soc_version=<soc_version>
===== Basic Functionality =====
...
```

## 报错排查速查

| 报错 | 根因 | 处理 |
|------|------|------|
| `EZ3003: No supported Ops kernel and engine are found`（见上方典型回显） | 目标平台的算子信息库不含该算子/插入的 Cast 无内核 | ① 确认 OPP 完整 ② 确认 soc 算子库覆盖 ③ vendor 包安装 ④ fp32 场景加 `--precision_mode=must_keep_origin_dtype` |
| `E20002: Invalid_Environment_Variable ... ASCEND_OPP_PATH is invalid ... during FE initialization`（伴生 `Failed to build OpsStoreInfo Vector`、GELib init failed，内置算子同样失败） | 工具链包的 opp 不完整（`opp/built-in` 缺 `op_graph`（旧版 `op_proto`）、`op_impl`、`framework` 子目录） | 换装 opp 完整的工具链包；判别法：`ls <包>/opp/built-in` 看上述子目录是否齐备 |
| `E10032`（singleop json 解析失败） | 描述文件 `type`/`format` 字段写成数组 | **必须字符串标量**（小写即可："float32"/"uint32"/"int8"…），参见 scripts/gen_singleop_json.py |
| `E90003: Tiling func of op_type <X> failed`（failure details 常为空） | 成因①：模型 shape 超出算子规格（如 1D 输入但规格限 2D）；成因②：报 `Template constraint`——目标 dtype 无对应 kernel 模板实例（TPL SEL 行未放行） | 成因①：onnx value_info/singleop shape 严格按 spec.yaml 构造；成因②：对照正常工程的 TPL SEL 行数定位缺失实例，随 kernel 实例化扩展放行 |
| `get_op_tiling.py ... KeyError: '<N>'`（N=tiling key 数字） | vendor 包内 kernel 头文件陈旧（staging 增量构建不刷新），atc 扫到的 tiling key 集合与 kernel 实际返回不符 | `rm -rf build` 全量重编 + 卸载重装 vendor（见上文经验条目） |
| `E19999: The opp version of the model does not match...` | 动态 shape/虚拟化场景两端 opp 版本不一致 | 两端同版本，重转模型 |
| `The required parameter --xxx for ATC is empty` | 参数为空或值未加引号 | 补全参数；含特殊字符的值加双引号 |
| `contains "layers" structures ... unsupported by ATC` | Caffe 旧版 layers 格式 | 转换为 layer 格式 |
| FE 初始化失败 | 工具链与目标平台不配套 | 换配套工具链（麒麟平台见 atc_tools_guide.md 环境准备） |
| input_shape 节点名不匹配 | 节点名与模型输入名不一致 | netron 或 `--mode=3` pre-check 确认 |
| 量化类报错（build graph failed 等） | 校准集/层支持/输入大小等 | 查官方 FAQ（[atc_official_digest.md](atc_official_digest.md) FAQ 索引） |

更多 FAQ 见官方文档（[atc_official_digest.md](atc_official_digest.md) FAQ 索引与联网查找指引）。
