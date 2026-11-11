---
name: ops-kirin-device-verify
description: "麒麟（Kirin）端侧设备验证：串联算子编译、ONNX/golden 生成、atc 转 OMC、HAP 与 hdc 部署、设备推理、输出回传与精度比对的全流程。触发场景：算子在麒麟设备上做端到端验证、HAP 推理、output0.bin 精度比对时使用。"
---

# Kirin 端侧算子麒麟设备验证

在 Kirin 系列端侧设备上验证自定义算子：host 侧完成编译安装与 OMC 转换后，经 HAP 推理应用在设备端执行算子，回传输出与 golden 比对得出结论。全流程 8 个步骤，host 侧步骤（1-4）与设备侧步骤（5-8）可分段暂停/恢复。

> **目录定位**：面向所有需要在 Kirin 端侧设备上验证算子的场景（算子开发验证、单算子功能验证、自定义算子在麒麟设备上的推理测试等），可独立使用，也可作为各类算子开发工作流的端侧验证环节被调用。本 skill 聚焦端到端流程编排与实战问题规避，工具级细节见 `tool-atc` 与 `tool-hdc`。

## 变量约定

文档中所有环境相关取值以变量表示，使用前按实际环境赋值：

| 变量 | 含义 | 获取/默认方式 |
| --- | --- | --- |
| `${WORK_DIR}` | 工作目录 | 用户实际路径 |
| `${SKILL_DIR}` | 本 skill 安装目录 | 如 `.opencode/skills/ops-kirin-device-verify` |
| `${OP_NAME}` | 目标算子名 | 如演示样例 `add_custom`（官方样例，配套 assets/ 模板） |
| `${SOC_VERSION}` | 目标芯片型号 | 由设备查询确定，映射见 `npu-arch` skill |
| `${SOC_TOKEN}` | 构建系统使用的小写 soc 值 | `echo ${SOC_VERSION} \| tr '[:upper:]' '[:lower:]'` |
| `${SN}` | 设备序列号 | `hdc list targets` 获取 |
| `${BUNDLE_NAME}` | HAP 推理应用 bundle 名 | **必须** `hdc shell bm dump -a` 核实，不设默认值 |
| `${DEVDIR}` | 设备端应用下载目录 | 设备侧 `bm dump` 的 dataDir，或用户指定 |
| `${VENDOR_PKG}` | 编译产物 vendor 包 | `find build build_out -maxdepth 1 -name '*.run'` |
| `${HAP_PATH}` | HAP 推理应用文件 | 用户提供 |
| `${HOST_ARCH}` | 宿主 CPU 架构 | `uname -m` |
| `${CANN_VERSION}` | CANN 版本（仅用于描述，不参与路径拼接） | `$ASCEND_HOME_PATH/compiler/version.info` 的 `Version` 字段 |

> 命令来源标注约定：**[官方文档]** = 官方文档出处（来源 URL 见「信息来源」节）；**[已验证]** = 已在麒麟设备环境验证过的经验（详见 references/device_verify_examples.md 对应 P/E 编号）。

## 工作流程

### 步骤 0：环境确认（按需）

先结合调用上下文判断环境是否已检查并达标（如此前已执行 `ascendc-kirin-env-check`、或会话中已确认 atc/vendor/设备链路可用，则跳过本步骤）；未检查或不达标时，调用 `ascendc-kirin-env-check` 确认 CANN/工具链环境，再运行本 skill 预检脚本：

```bash
bash ${SKILL_DIR}/scripts/check_onboard_env.sh                # vendor 装入 + HAP 就绪 + hdc 自检
bash ${SKILL_DIR}/scripts/check_onboard_env.sh --skip-device  # 跳过 hdc 连接自检
bash ${SKILL_DIR}/scripts/check_onboard_env.sh --hap <hap文件>
```

预检项：vendor 已装入 opp、HAP 文件存在（设备侧特有项）。**[已验证]** host/CANN 环境检查委托 `ascendc-kirin-env-check`；hdc 定位与连接自检委托 `tool-hdc`（`scripts/find_hdc.sh`），本脚本不重复实现。

> [已验证 P8] 设备链路可能离线或需人工建立连接。**先把 host 侧步骤 1-4 全部完成**，再走设备链路步骤 5-8；本流程各步骤可独立暂停/恢复。

### 步骤 1：编译目标算子

按工程形态二选一（soc 参数大小写以构建脚本要求为准）：

```bash
# 形态一：自定义算子工程（提供 build.sh 的工程）[已验证]
cd ${WORK_DIR}/operators/${OP_NAME}
bash build.sh --soc=${SOC_TOKEN}        # 小写形式；产物在 build/ 下
# → build/*.run（vendor 包：op_proto/tiling/kernel），文件名随包形态/架构生成

# 形态二：ops-math 等官方仓形态 [官方文档 §2]
bash build.sh --pkg --soc=${SOC_VERSION} --ops=${OP_NAME} -j16   # 大写形式；产物在 build_out/ 下
```

> [已验证 P2] 工具链对 tiling key 有查重：TPL 宏参数须用**字面数字**，符号常量名内嵌数字（如 `XXX_TPL_MODE_F32`）会被误解析导致 `There is duplicated number ...` 构建失败。

### 步骤 2：安装编译产物（vendor 包）

将 vendor 包装入当前 source 的工具链 opp（若同时存在多个 CANN 包，注意**只认 source 的那个**）：

```bash
# 改码后必须先卸载再重装（安装器对已存在 vendor 目录"跳过升级"不覆盖文件）[已验证 P3]
bash ${ASCEND_HOME_PATH}/opp/vendors/<vendor_name>/uninstall.sh 2>/dev/null || true
VENDOR_PKG=$(find ${WORK_DIR}/operators/${OP_NAME}/build -maxdepth 1 -name '*.run' -print -quit)
bash "${VENDOR_PKG}" --install
# 安装校验报 undefined symbol: mmDladdr 时属误报，追加 --force 绕过（libmmpa 运行时提供）：
# bash "${VENDOR_PKG}" --install --force
```

成功判据：`${ASCEND_OPP_PATH}/vendors/<vendor_name>/` 下出现新鲜的 op_proto/tiling/kernel 文件（核对文件时间戳）。**[已验证 P3，见 references/device_verify_examples.md]**

> [官方文档 §4] 若后续在 host 侧跑样例/ST 需要 aclnn 运行时，另需 `export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/<vendor_name>/op_api/lib:${LD_LIBRARY_PATH}`；纯 atc 转换依赖 proto 库自动加载，无需此步。装 vendor 的 shell 与跑 atc 的 shell 必须是 source 过同一 set_env 的会话 **[已验证 P10]**。

### 步骤 3：生成测试用例（ONNX 模型 + 输入数据 + golden）

**先检查复用**：结合当前会话上下文与工作目录，检查是否已存在目标算子的输入/输出 bin 数据及 onnx 模型；核对算子名/shape/dtype 与当前目标一致才可复用；不一致或不存在则按下方方式生成。**[已验证]**

```bash
# a) ONNX 模型：复制官方样例模板后按目标算子修改（opset=11、ir_version=6 保持不变）[官方样例]
cp ${SKILL_DIR}/assets/create_onnx.py ./create_onnx_${OP_NAME}.py
# 编辑：节点名/属性、输入输出 value_info 的 dtype 与 shape（shape 严格按算子规格，见下）
python3 create_onnx_${OP_NAME}.py    # 产出 .onnx

# b) 输入数据 + golden：按算子规格生成（seed 固定可复现，支持特殊值注入）
python3 ${SKILL_DIR}/scripts/gen_cases.py \
    --golden-op <numpy逐元素函数，二元如less/add、一元如is_finite> --shape 128,512 --dtype float16 \
    --seed 97 --special nan,inf,equal,ulp,spread,zpair --out-dir ./case_data
```

特殊值注入种类（`--special`，逗号分隔，float 专用）：`nan`/`inf`（NaN/±Inf）、`equal`（跨输入等值对）、`ulp`（单 ulp 相邻对）、`spread`（±10 大差异）、`zpair`（+0/-0 对）；`--special-ratio` 控制比例。目标算子不在内置清单或非逐元素算子时，**复制 scripts/gen_cases.py 后按其语义修改** golden 计算与注入逻辑（参考脚本定位）。

硬约束：

- onnx 的 `opset_import.version=11`、`ir_version=6`，高版本 ir 会被工具链 parser 拒绝 **[已验证]**
- onnx value_info 的 shape/dtype **必须与算子规格一致**（有 spec.yaml 以其为唯一真值源，否则以用户提供的算子输入输出规格为准）：shape 不符会触发 tiling 模板约束失败 `E90003 Tiling func of op_type ... failed` **[已验证 P5]**
- golden 数据形态与设备端输出 output0.bin 同构：bool 输出以 uint8(0/1) 存盘，其他 dtype 按实际 dtype 存盘 **[已验证]**
- 官方样例模板（assets/）：`create_onnx.py`（onnx 构造）、`gen_data.py`（裸 bin 生成）；单算子描述文件格式样例由 `tool-atc` 维护（`assets/singleop_json_example.json`）

### 步骤 4：atc 转 OMC（默认 onnx 路线，支持 singleop 路线）

调用 `tool-atc` skill 完成转换；路线、参数与报错排查均见该 skill。

```bash
source <CANN安装根>/set_env.sh    # 每个新 shell 必须重新 source，路径以实际安装为准 [已验证 P10]

# 路线一（默认）：ONNX 模型 [tool-atc skill]
atc --model=./${OP_NAME}.onnx --framework=5 --soc_version=${SOC_VERSION} \
    --output=./model/${OP_NAME}_f16

# 路线二（备选）：单算子描述文件，无 onnx 或需精确控制图结构时使用 [官方文档 §6]
#   格式样例见 tool-atc 的 assets/singleop_json_example.json：顶层数组、type/format 为字符串标量
atc --singleop=${OP_NAME}.singleop_json --soc_version=${SOC_VERSION} \
    --output=./model --op_debug_level=1 &>atc.log
```

**fp32/bf16 等高精度输入必须追加** `--precision_mode=must_keep_origin_dtype`（否则默认 `force_fp16` 自动降精度插入 trans_Cast，host 工具链对 Kirin 平台无 Cast 内核 → `EZ3003`）**[已验证 P4]**：

```bash
atc --model=./${OP_NAME}_fp32.onnx --framework=5 --soc_version=${SOC_VERSION} \
    --output=./model/${OP_NAME}_fp32 --precision_mode=must_keep_origin_dtype
```

要点：

- singleop json 顶层为**数组**，`type`/`format` 必须是**字符串标量**（写成数组报 `E10032`）**[已验证 P10]**
- 产物命名：singleop → `0_<Op>_1_2_<shape编码>_<...>.omc`（自动生成含 dtype/rank/shape 编码）；onnx → `<--output 指定名>.omc` **[已验证]**
- 成功判据：omc 文件生成且日志无 EZ3003/E90003/E10032
- 工具链要求：包内需具备完整 `opp`（`opp/built-in` 下 `op_graph`/`op_impl`/`framework` 齐备）；若缺子目录，atc FE 初始化报 `E20002` **[已验证 P1，根因与判据见 tool-atc 报错排查]**

### 步骤 5：HAP 准备（下载 → 安装 → 核实）

调用 `tool-hdc` skill；连接、安装、传输、shell 与错误码均见该 skill。

```bash
# a) 准备 HAP（由用户提供，或从官方样例仓获取；raw 链接对二进制返回 403，需 git clone）
# HAP_PATH=<HAP 文件路径>

# b) 安装 [官方文档 §7.1]
hdc -t ${SN} install ${HAP_PATH}

# c) 强制核实 bundle 名（文档与样例源码可能拼写不一致，以设备输出为准）[官方文档 §7.2]
hdc -t ${SN} shell bm dump -a     # 从输出确认 ${BUNDLE_NAME}
```

> 卸载 HAP（`hdc -t ${SN} uninstall ${BUNDLE_NAME}` [官方文档 §7.5]）属**破坏性操作**，执行前须用户确认。HAP 版本/路径/包名一律以变量与用户确认值为准，不硬编码。

### 步骤 6：推送文件到设备

将输入 bin 与 omc 传输至设备端目录；输入文件须**先用例前缀改名**再推送（设备端同名会覆盖，如 x1.bin）**[已验证 P10]**：

```bash
mv ./case_data/x1.bin ./case1_x1.bin && mv ./case_data/x2.bin ./case1_x2.bin   # 改名防覆盖
hdc -t ${SN} file send ./case1_x1.bin ${DEVDIR}/case1_x1.bin [官方文档 §7.3]
hdc -t ${SN} file send ./case1_x2.bin ${DEVDIR}/case1_x2.bin
hdc -t ${SN} file send ./model/${OP_NAME}_f16.omc ${DEVDIR}/${OP_NAME}_case1.omc
```

自校验：输出字节 = 元素数 × dtype 字节数（如 bool [128,512] = 65536B），不符立即报警 **[已验证]**。

### 步骤 7：启动推理

**每次执行前必须 force-stop**：应用进程存活时再次 `aa start` 会复用已启动 ability，新参数被忽略、output0.bin 不更新（但返回 start ability successfully）**[已验证 P6]**：

```bash
hdc -t ${SN} shell aa force-stop ${BUNDLE_NAME}          # 前置，强制
hdc -t ${SN} shell aa start -a EntryAbility -b ${BUNDLE_NAME} \
    --ps path case1_x1.bin,case1_x2.bin --ps omPath ${OP_NAME}_case1.omc
```

参数协议（源自 HAP 推理应用源码 HAPTesting.py）**[已验证]**：`--ps path` 多输入**逗号分隔**字符串（官方文档样例为单输入，多输入以本协议为准）；`--ps omPath` 为单 omc 文件名；输出固定写 `output0.bin` 于 `${DEVDIR}`。启动后轮询确认 output0.bin 时间戳已更新再回传。

### 步骤 8：回传输出与精度比对

```bash
hdc -t ${SN} file recv ${DEVDIR}/output0.bin ./output0_case1.bin   # output0.bin 固定名，recv 后立即改名 [官方文档 §7.4]
# bool 输出（如比较类算子，golden 为 uint8；显式 bitwise 模式，反转=0 即 PASS）：
python3 ${SKILL_DIR}/scripts/compare_output.py \
    ./output0_case1.bin ./case_data/golden.bin --shape 128,512 --dtype uint8 --mode bitwise
# 浮点输出（golden 为 float16/float32）：
python3 ${SKILL_DIR}/scripts/compare_output.py \
    ./output0_case1.bin ./case_data/golden.bin --shape 128,512 --dtype float16 \
    --mode tolerance --rtol 1e-3 --atol 1e-3
```

比对判定标准（`--mode` 由用户按输出类型显式指定，不依赖 dtype 推断）：

- **bitwise 模式**：按位模式逐元素比较（bool/整数/浮点均按字节位模式），反转元素比例必须为 0 即 PASS；浮点 NaN payload 不同或 ±0 差异会判 FAIL，语义一致请用 tolerance **[已验证]**（适合 bool/整数等需全一致的输出）
- **tolerance 模式**：rtol/atol 按算子规格对应的 dtype 精度标准配置，脚本默认 1e-3，**容差必须显式输出留痕**；NaN/Inf 与 golden 同位同值视为一致（Inf 须同号）
- 输出字节自校验不符（元素数 != shape 期望值）直接报警终止比对

## 输出要求

验证完成后必须输出：

1. **每步产物路径**：vendor 包、onnx、输入 bin、golden、omc、设备端回传 output
2. **最终结论**：PASS / FAIL（bool 类注明"bitwise 全一致，反转 N 个"；浮点/整数类注明不匹配元素数与占比）
3. **所用容差**：rtol/atol 或 bitwise 判定依据
4. **完整命令序列**：实际执行的全部命令（按步骤编号，变量已代入真实值），供复现

## 参考文档索引

| 文档 | 内容 | 来源 |
| --- | --- | --- |
| [references/device_verify_examples.md](references/device_verify_examples.md) | 端到端参考结论、P1-P10 问题清单、依赖清单（通用 atc/hdc 命令分别见 tool-atc / tool-hdc） | 端到端参考结论与问题清单 |
| [assets/create_onnx.py](assets/create_onnx.py) | onnx 单算子模型构造模板（opset11/ir6） | 本项目模板 |
| [assets/gen_data.py](assets/gen_data.py) | 裸 bin 输入/golden 生成模板 | 本项目模板 |
| [../../tools/tool-atc/assets/singleop_json_example.json](../../tools/tool-atc/assets/singleop_json_example.json) | 单算子描述文件格式样例（由 tool-atc 维护） | tool-atc |
| [../ascendc-kirin-env-check/references/kirin_platform_guide.md](../ascendc-kirin-env-check/references/kirin_platform_guide.md) | 麒麟环境前提、CANN 包与 kirin-bisheng 安装 | ascendc-kirin-env-check |

## 脚本工具

| 脚本 | 用途 | 说明 |
| --- | --- | --- |
| [scripts/check_onboard_env.sh](scripts/check_onboard_env.sh) | 设备侧特有项预检（步骤 0） | 只检查 vendor 装入与 HAP；host/CANN 委托 ascendc-kirin-env-check，hdc 委托 tool-hdc；支持 `--skip-device`/`--hap` |
| [scripts/gen_cases.py](scripts/gen_cases.py) | 输入数据 + golden 生成参考脚本（步骤 3） | seed 可复现；内置常见逐元素 numpy 函数；`--special` 通用特殊值注入；其他算子复制后按其语义修改 |
| [scripts/compare_output.py](scripts/compare_output.py) | 输出 vs golden 比对参考脚本（步骤 8） | `--mode bitwise/tolerance` 由用户显式指定；bitwise 含反转元素数与余弦相似度，tolerance 含 rtol/atol 与差异点定位 |

## 与其他 Skill 的关系

- **ascendc-kirin-env-check**：步骤 0 按需调用完成麒麟平台环境确认；环境前提与 CANN 包 / kirin-bisheng 安装以其 `references/kirin_platform_guide.md` 为准。
- **npu-arch**：确认目标芯片型号与架构代号映射（${SOC_VERSION} 取值依据）。

## 信息来源

- 官方文档：`https://gitcode.com/cann/cann-recipes-harmony-infer`（`docs/ascend_ops_guide.md`、`docs/atc_tools_guide.md`）
- 官方样例模板：`ops/ascendc/src/add_custom/test/{create_onnx,gen_data}.py`（比对脚本为本 skill 自研沉淀，见 scripts/compare_output.py）
- HAP 推理应用：`https://gitcode.com/funna2000/kirin_cann_test`（参数协议源码 HAPTesting.py）
- 端到端参考结论与问题清单：references/device_verify_examples.md（端到端参考结论与问题清单）
