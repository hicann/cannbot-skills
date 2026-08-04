# 输入适配指南（input-adaptation）

用户只提供模型或 dump 时，skill 如何识别目标图结构、产出统一证据，并在证据允许范围内生成最小复现与验证链路。这是输入适配阶段（skill 内部里程碑 R3，不是版本号）的知识层；可执行逻辑在 `"$SKILL_ROOT/scripts/adapt_input.py"`。

> 谁读：① 需求分析（输入只给模型/dump 时先跑 `adapt_input.py`）；② 开发（按 normalized-graph 框定匹配）；③ 验证（按输入类型选触发编译路线）。
>
> 产物 schema 见 `normalized-graph-schema.md`；目标片段语义见 `fragment-spec.md`；触发编译步骤见统一 skill 阶段三“验证顺序与证据”。

## 一、输入分流（§8.2 扩展）

| 输入 | 首选处理 | 能力边界 |
|---|---|---|
| 原始 Torch/TF/GE IR 脚本 | 原样复用并生成 dump | 不改写用户脚本，除非用户明确要求 |
| ONNX | `adapt_input.py inventory` 解析节点/边/属性/shape；`compile` 走 `atc --framework=5` | 不强制转 AIR；真实 GE op type 仍以 `PreRunBegin` 为准（`tips/dump-first-op-type.md`） |
| AIR | `compile`：C++ pass 用 `atc --mode=0 --framework=1`；Python pass 用 `pyatc` | `compile-evidence.json` 记录 OM、dump、完整日志；整网输出仍需分别跑 baseline/optimized OM 比较（`requirements-analysis-template.md` §9.3） |
| GE dump pbtxt | `adapt_input.py inventory` 提取目标结构 + `repro` 生成最小复现 | pbtxt 可能缺权重和属性，**只承诺结构复现（structural），不承诺原网语义无损恢复** |
| TensorFlow PB | 已有 `.pb` 时 `compile` 走 `atc/pyatc --framework=3`；脚本生成和 TF 在线路线另行探测 | 生成 `.pb` 仍需 TF1；在线执行仍需 `npu_bridge`；缺运行时则只标对应步骤 `NOT_RUN` |

`adapt_input.py detect` 自动识别输入类型（按扩展名 + 文件魔数（magic bytes））；`inventory` 按类型走对应解析器（ONNX 用 onnx 库、pbtxt 用文本解析）。AIR 不伪造可解析图，直接交给 `compile` 走 ATC/pyATC；TensorFlow PB 直接交给 `framework=3`，不要求重新导入 TensorFlow。

## 二、统一产物（§8.3）

| 产物 | 产出命令 | 内容 |
|---|---|---|
| `input-inventory.json` | `detect` / `inventory` | 输入类型、文件 sha256、节点/边/子图概览、缺失信息、status |
| `normalized-graph.json` | `inventory` | 统一节点/数据边/控制边/端口/属性/shape/dtype/format（schema 见 `normalized-graph-schema.md`） |
| `artifacts/repro/` | `repro` | skill 生成的最小复现脚本/模型，**绝不覆盖用户 `data/`**（文件所有权规则，见 §四） |
| `structural-isomorphism.json` | `repro <pbtxt>` | 原 pbtxt 与生成 ONNX 的节点、带端口数据边精确比较；仅 `PASSED` 才能声称结构同构 |
| `provenance.json` | `repro` / `provenance` | 来源、转换命令、工具版本、env 快照、假设、复现级别 |
| `compile-evidence.json` | `compile` | 实际 argv、返回码、OM、`PreRunBegin`、自定义 pass dump、完整日志路径 |
| `requirements-analysis.md` | 统一 skill 阶段一 | 引用上述证据，标 `structural` 或 `semantic` reproduction |

解析产物的 status 用 `OK` / `NOT_RUN`；同构和编译证据用 `PASSED` / `FAILED` / `NOT_RUN`，均在降级或失败时附 `reason`。

### 编译命令

编译必须提供完整 `soc_version`，并由脚本默认开启 `DUMP_GE_GRAPH=1` 与 `DUMP_GRAPH_LEVEL=1`。脚本只收集**本轮新生成或变更**的 OM/dump，避免把旧文件误作本轮证据。

```bash
# C++ pass + AIR
python3 "$SKILL_ROOT/scripts/adapt_input.py" compile model.air \
  --pass-language cpp --soc-version <完整值> \
  --work-dir artifacts/compile-air --output artifacts/compile-air/model

# Python pass + AIR（必须在同进程加载 pass）
python3 "$SKILL_ROOT/scripts/adapt_input.py" compile model.air \
  --pass-language python --py-pass-path src/my_pass.py \
  --soc-version <完整值> --work-dir artifacts/compile-air-py

# 已有 TensorFlow PB（无需由 skill 生成 PB）
python3 "$SKILL_ROOT/scripts/adapt_input.py" compile matmul_add.pb \
  --pass-language cpp --soc-version <完整值> --work-dir artifacts/compile-tf-pb
```

`compile-evidence.json` 只有同时包含 OM、`PreRunBegin` 和自定义 pass 阶段 dump 时才标基础 `PASSED`；缺环境或缺文件分别如实标 `NOT_RUN` / `FAILED`。这仍只证明编译和 pass 阶段被触发。要把“目标 pass 真命中并生效”纳入门禁，必须提供目标 `--pass-name` 和结构化 events 或 `fusion_result.json`，并追加 `--require-pass-effect`：

```bash
python3 "$SKILL_ROOT/scripts/adapt_input.py" compile model.air \
  --pass-language cpp --pass-name MatMulAddPass \
  --events artifacts/pass-events.jsonl \
  --fusion-result artifacts/fusion_result.json \
  --require-pass-effect --pass-load-root "$ASCEND_OPP_PATH/vendors/pass_so_dir/custom_fusion_passes" \
  --soc-version <完整值> --work-dir artifacts/compile-air
```

没有结构化生效证据时，基础编译结果不会被伪装成完整融合结论。

## 三、降级矩阵

| 缺失条件 | 降级行为 | 标注 |
|---|---|---|
| onnx 库不在 | ONNX 解析失败 | inventory `NOT_RUN` + reason；normalized-graph `NOT_RUN` |
| atc 不在 | 无法触发编译 | 阶段三对应步骤标 `NOT_RUN` |
| 无 `.air` 文件 / 缺 atc 或 pyatc / 缺 `ge.passes` 或 Python pass bridge / 缺完整 soc | AIR 无法直接编译 | `compile-evidence.json` 标 `NOT_RUN` + 原因；不转成其他格式 |
| pbtxt 缺权重/属性 | 只结构复现 | `reproduction_level: structural` + `missing[]` 记录 + provenance 记补全假设 |
| 控制边/GE 内部 op/动态端口前端表达不了 | 走 ES/GE IR | normalized-graph `unrepresentable[]` + 退化 graph 路线（`pass-development-paradigm.md` §3.1） |

**降级铁律**：探测不到就标 `NOT_RUN` + 真实原因，绝不伪造节点/边/dump/结论（与门禁 G3 一条心）。

## 四、文件所有权规则

- `artifacts/repro/`：skill 生成的最小复现落这里。`adapt_input.py repro` 默认写到 `./artifacts/repro/`。
- **禁止覆盖用户 `data/`**：`repro` 命令检查目标目录不得落在 `data/` 内，命中即拒。
- 复现脚本/模型只新增到 `artifacts/repro/`，不动用户原始文件（fixture 测试断言原文件 sha256 不变）。

## 五、provenance 字段语义

`provenance.json` 记录复现可复跑性：

- `source_file` / `sha256`：原始输入及其 hash。
- `input_type` / `reproduction_level`：`semantic`（完整语义，ONNX/AIR 正常）或 `structural`（仅结构，pbtxt/残缺）。
- `transform_commands`：把原始输入变成复现物用的可执行命令（`cp`、`adapt_input.py repro ...`），能原样复跑。
- `compile_command_templates`：从复现物进入 ATC/pyATC 的命令模板；实际执行的 argv、日志和生成物以 `compile-evidence.json` 为准。
- `structural_isomorphism`：pbtxt 路线的同构状态和证据路径；非 `PASSED` 时不得声称模型已结构复现。
- `tool_versions`：python/onnx/mindspore/atc/pyatc、`ge.passes` 和 Python pass bridge 的版本、模块或可执行路径（缺失标 `NOT_AVAILABLE`）。
- `env`：`ASCEND_HOME_PATH`/`ASCEND_OPP_PATH`/`GE_REPO_PATH` 等快照。
- `assumptions`：所有补全假设（pbtxt 缺权重→随机权重、缺属性→不承诺语义等），逐条记。
- `status`：`OK` / `NOT_RUN`（+reason）。

## 六、AIR 路线的环境约束（重要）

输入适配阶段（内部 R3）§8.5 验收“AIR 能通过 atc/pyatc 直接生成 OM”需要真实 `.air`、完整 `soc_version`、可用设备及本轮日志。运行前通过 `bash "$SKILL_ROOT/scripts/check_env.sh"` 与 `python3 "$SKILL_ROOT/scripts/check_runtime.py"` 记录当前 CANN 根、`atc`/`pyatc`、`ge.passes` 和 Python pass bridge；任一项缺失时，`compile-evidence.json` 必须标为 `NOT_RUN` 并说明原因。

- C++ AIR 仅在输入确认是 AIR 时使用 `atc --framework=1 --mode=0`；将 ONNX 传给 `--framework=1` 的报错只能说明输入类型不匹配，不能作为 AIR 成功证据。
- Python AIR 需要 `pyatc`、`ge.passes` 与 Python pass bridge 均真实可用；命令、OM、GE dump 和 stdout/stderr 必须保存到当前 case 的 `artifacts/`。
- 只存在临时目录或无法复核的历史命令、计数和 hash 一律视为 `NOT_RUN`，不得作为本 case 的能力或验收结论。

`compile` 的 C++/Python 分流、argv、OM/dump/log 收集可由 fixture 覆盖其格式契约，但 fixture 不替代真实运行证据。ATC/pyatc 编译成功只证明模型可编译、pass 可被触发，**不替代 baseline/optimized OM 的整网输出比较**（§9.3）。
