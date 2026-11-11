# 端侧设备验证实操参考

端到端链路参考：host 侧 atc 转 OMC → HAP 推理 → 输出回传 → 与 numpy golden 逐位比对。

## 1. 前置条件与变量

| 依赖 | 取值/来源 | 备注 |
| --- | --- | --- |
| 工具链 | CANN 完整包（含 ATC 所需完整 opp 与目标平台 simulator；常见发布形态 mobile-station，包名形如 `Ascend-cann-toolkit_${cann_version}_linux-${HOST_ARCH}-mobile-station.run`，获取与安装见 [kirin_platform_guide.md](../../ascendc-kirin-env-check/references/kirin_platform_guide.md)） | 需具备完整 `opp/built-in` 与目标平台 simulator（P1）；安装路径以 source 后 `$ASCEND_HOME_PATH` 为准 |
| 环境变量 | `source <CANN安装根>/set_env.sh` | 每个新 shell 必须重 source（P10） |
| vendor 包 | `${VENDOR_PKG}`（`build/*.run`） | 改码后必须 uninstall→重编→重装（P3） |
| Python | numpy / onnx；跨机自动化可选 paramiko | 数据生成、onnx 构造 |
| 设备 | Kirin 端侧设备（型号以 hdc 查询为准） | SN 用 `hdc list targets` 获取 |
| hdc | USB 直连或按环境配置 | `hdc list targets` 验证在线 |
| HAP 推理应用 | `${HAP_PATH}`（用户提供，或从官方样例仓获取） | raw 链接对二进制 403，须 `git clone --depth 1`（P9） |
| singleop json 样例 | 与目标算子/规格匹配的 singleop json | 格式样例由 tool-atc 维护（`assets/singleop_json_example.json`） |

## 2. 参考用例与结论

| 用例 | 规格 | 转换路线 | omc 产物 |
| --- | --- | --- | --- |
| 用例1 | fp16 [256,1024] | atc `--singleop` | 自动命名（含 dtype/rank/shape 编码） |
| 用例2 | fp16 [128,512] | onnx → atc `--framework=5` | `${OP_NAME}_f16_2d.omc` |
| 用例3 | fp32 [128,512] | onnx → atc `--framework=5` + `--precision_mode=must_keep_origin_dtype` | `${OP_NAME}_fp32.omc` |

比对结论：三用例均 PASS，**bitwise 全一致，0 反转**，覆盖特殊值注入点（NaN/±0/±Inf/等值对/±10/单 ulp）。

## 3. 命令序列（按步骤）

> 通用 atc/hdc 命令不在此重复：ATC 转换见 `tool-atc`，hdc 连接/安装/传输/推理见 `tool-hdc`。本节只保留偏差、产物与结论。

### 3.1 工具链与 vendor 安装

```bash
# 装载后 source（路径以实际安装为准）
source <CANN安装根>/set_env.sh

# vendor 安装（改码后流程：uninstall → 删 build 重编 → 重装）【P3】
bash "${VENDOR_PKG}" --install
# 安装校验报 undefined symbol: mmDladdr → 误报，--force 绕过（libmmpa 运行时提供）
# bash "${VENDOR_PKG}" --install --force
```

### 3.2 用例数据生成（E1）

设计原则（已沉淀为 scripts/gen_cases.py）：seed 固定可复现；每用例随机底板 + `inject_specials` 注入 12% 语义边界值（NaN 于 x1/x2 各半、等值对含 ±0、±10 大差异双向、±Inf、±0 对、单 ulp `np.nextafter`）；golden 按算子语义 numpy 计算（记录算子为二元比较类，`bool → uint8(0/1)` 直接 tofile，与设备输出同构可比对）。

### 3.3 atc 转换（E3）

命令与参数见 `tool-atc`；转换结论：

- 路线A（`--singleop`）：产物按 dtype/rank/shape 自动命名，如 `0_<Op>_1_2_256_1024_...omc`
- 路线B（onnx，`--framework=5`）：产物名由 `--output` 指定
- fp32 输入必须追加 `--precision_mode=must_keep_origin_dtype`（P4）
- 成功回显末行：`ATC run success, welcome to the next use.`

### 3.4 HAP 安装与文件推送（E4/E5）

hdc 命令见 `tool-hdc`；结论：

- bundle 名以 `hdc shell bm dump -a` 输出为准（见 P7）
- 设备目录按 `${DEVDIR}` 取设备侧实际路径
- 多用例文件必须加用例前缀，避免设备端同名覆盖（P10）

### 3.5 推理与回传（E6/E7）

hdc 命令见 `tool-hdc`；关键偏差：

- 每次推理前必须 `aa force-stop`，否则应用存活时 `aa start` 复用 ability、新参数被忽略、`output0.bin` 不更新（P6）
- 多输入用 `--ps path` 逗号分隔（HAPTesting.py 协议；官方 guide 样例为单输入）
- `output0.bin` 为固定名，recv 后立即改名，避免多用例覆盖

输出字节数自校验：bool 输出 = 元素数 ×1B（用例1 262144=256×1024、用例2/3 65536=128×512），均精确匹配。

### 3.6 精度比对（E8）

比对脚本输出指标（已沉淀为 scripts/compare_output.py）：文件名/shape/dtype、相对误差（含 eps 防零除）、绝对误差、余弦相似度、**反转元素数量与比例（bool 输出核心验收指标）**、平均误差、第一个差异点（flat 索引+双方值）、最大差异点。验收：bool 输出按 spec 容差 **bitwise 全一致（反转=0 即 PASS）**；浮点指标仅作留痕。

典型回显格式（节选）：

```
============================================================
文件名(output)     : output0_case1.bin
shape              : (256, 1024)  dtype: bool(uint8)
元素总数           : 262144
------------------------------------------------------------
反转元素数量       : 0
反转元素比例       : 0.000000
第一个差异点       : 无
最大差异点         : 无
------------------------------------------------------------
结论               : PASS (bitwise 全一致)
============================================================
```

## 4. 问题清单 P1-P10（背景 → 根因 → 解决）

### P1 opp 不完整，atc 无法初始化
- 现象：atc FE 初始化失败，报 `E20002 ASCEND_OPP_PATH invalid` + `Failed to build OpsStoreInfo Vector` + GELib init failed；内置算子 singleop 同样失败
- 判定：`ls $ASCEND_OPP_PATH/built-in`，`op_graph`（旧版 `op_proto`）、`op_impl`、`framework` 缺任一即 opp 不完整（见 tool-atc 报错排查 E20002）
- 解决：安装 opp 完整的 CANN 包；与 vendor 包无关

### P2 tiling key 查重，构建失败
- 现象：重编 vendor 报 `There is duplicated number in ASCENDC_TPL_DECL_UINT DTYPE_MODE! Duplicated List: [...]`
- 根因：`template_tiling.py` 的 `extract_num` 按字面数字解析 TPL 宏参数，符号常量名内嵌数字（`XXX_TPL_MODE_F32`→32）污染值列表
- 解决：tiling key 宏 DECL/SEL 改字面数字（语义等价），constexpr 常量保留供 C++ 代码用

### P3 vendor 安装保留旧文件 + 安装校验误报
- 现象：①改码重装后 `$ASCEND_OPP_PATH/vendors/<vendor>/` 内仍是旧头文件（atc 复现旧报错）；②安装报 `undefined symbol: mmDladdr` 拒装
- 根因：①安装器对已存在 vendor 目录按版本判断"跳过升级"不覆盖文件；②安装器独立 dlopen 校验 proto 库，mmDladdr 在 libmmpa.so（运行时提供），校验环境缺载入链属误报
- 解决：改码后必须 `uninstall.sh → 删 build 重编 → 重装`；安装用 `--force` 跳过校验

### P4 fp32 转 omc 报 EZ3003
- 现象：fp32 输入（singleop 与 onnx）报 `EZ3003: No supported Ops kernel and engine are found for [trans_Cast_0/1], optype [Cast]`；fp16 输入不插 Cast
- 根因：`--precision_mode` 默认 `force_fp16`，FE 将 fp32 降精度并插入 trans_Cast；目标平台算子库无 Cast 内核时即报 EZ3003。自检：fp32 输入数与 trans_Cast 数对应；`atc --help` 核对 `--precision_mode` 默认值
- 解决：转换追加 `--precision_mode=must_keep_origin_dtype`

### P5 onnx 1D shape 触发模板约束失败
- 现象：onnx 输入声明为 1D（如 [4]），报 `E90003 Tiling func of op_type ... failed`（failure details 为空）
- 根因：算子规格限定 2D，tiling 模板校验 1D 非法
- 解决：onnx value_info 严格按 spec 声明 shape

### P6 `aa start` 应用存活时不重新执行推理
- 现象：再次启动后 output0.bin 时间戳未更新（仍是上一次产物），但返回 `start ability successfully`
- 根因：应用进程存活时再次 `aa start` 复用已启动 ability，新 `--ps` 参数被忽略、不重新推理
- 解决：每次执行前 `hdc shell aa force-stop <bundle>` 再 start

### P7 bundle 名口径不一致
- 现象：官方 guide 与样例源码的 bundle 名可能拼写不一致
- 解决：以 `hdc shell bm dump -a` 输出为准
- 启示：bundle 名核实设为强制步骤，不信任文档默认值

### P8 设备链路不在线
- 现象：设备链路拒绝连接或未建立，命令阻塞
- 解决：阻塞点前移——先完成所有本地环节（数据/转换/打包/比对脚本），就绪后走设备链路
- 启示：设备链路设计为可暂停/恢复的独立阶段（本 skill 步骤 5-8）

### P9 gitcode raw 链接 403
- 现象：直接下载 HAP 二进制 raw 链接返回 403（"暂不支持预览"）
- 解决：用 `git clone --depth 1 <样例仓>` 取文件（raw 对二进制不可用）

### P10 执行层小坑（备忘）
- singleop json 的 type/format 写成数组 → `E10032` json 解析失败，须为字符串标量
- `ASCEND_PROCESS_LOG_PATH` 指向目录须提前创建；atc 主进程图优化日志（trans_Cast 插入点）不保证落盘，排查 FE 行为优先用隔离实验（singleop vs onnx、逐 dtype 对照）而非日志
- 多用例文件在设备侧须加用例前缀（x1.bin 会被覆盖），output0.bin 固定名须每次 recv 后立即改名
- 装 vendor 的 shell 与跑 atc 的 shell 必须是 source 过同一 set_env 的会话
- 排查 FE 降精度行为可用 `atc --help` 核对 `--precision_mode` 默认值（force_fp16）
