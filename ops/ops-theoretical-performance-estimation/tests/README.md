# ops-theoretical-performance-estimation 自验证（ST）

两层验证，职责分明：

| 层 | 验什么 | 依赖 | 何时跑 |
|---|---|---|---|
| **offline/** | 计算流解释 + 引擎本身对不对（结构/合法性自洽） | 无（纯软件） | 每次改 flow/解释器/引擎，当门禁 |
| **calibration/** | 标定常量 ⟷ 上板 bench 实测一致（防凑数） | 无（读已记录 bench CSV + 跑引擎） | 当门禁（已并入 `run.py`）；bench 本身周期性上板重采 |
| **onboard/** | 理论性能上限 vs 950 实测（模型精度，独立验证集） | 950 板 + 真实 kernel | 周期性采集 + 提交后回归 |

以上为 Roofline 路径。OpModel 的验证脚本随子目录保留，按
[版本与验证](../ops-opmodel-toolkit/references/distribution.md)运行；
`OPMODEL_SKILL_DIR` 指向 `ops-opmodel-toolkit` 子目录。其 wheel 冒烟验证不并入
`tests/run.py`，原有离线门禁无需安装 OpModel。两套行为评测分别保留在父、子目录的
`evals/evals.json` 中。

> 三层分工：bench（calibration/bench_results）**派生**常量 → onboard 算子**独立验证** → calibration 检查**锁死**「常量必须有 bench 背书」。详见 `calibration/README.md`。

## 验收口径

`theoretical_time_us` 是 roofline **理论性能上限**（可达最快）；按时延即理论时延，是实测时延的下界。
判定围绕「实测是否超越了理论性能上限」：

```
理论时延 ≤ 实测时延 ≤ cp4_target(=理论/target_efficiency, 默认 0.6)   ← 健康
eff = 理论 / 实测 ∈ [target_efficiency, 1]
```

- **measured < 理论·(1-3%) → BELOW_FLOOR**：实测时延比理论时延还低（实测性能超过理论上限），
  只能是模型把可达上限算低了（某项硬件能力被低估）。**最该抓的模型 bug**，典型是中等 shape 的
  L2 驻留带宽（技术文档 #1）。
- **eff ∈ [0.6,1] → OK**：实测时延落在 [理论时延, CP4 目标] 之间，达标。
- **eff < 0.6 → LOW_EFF**：实测时延远高于理论时延（实测性能远低于上限）。两种可能——kernel 未优化
  （真性能问题，非模型 bug）或 flow 漏建了 work（理论时延偏低）。用 profiling 占比与 case 的
  bytes/指令对比来区分。

## offline 层（无需板子）

```bash
python3 tests/run.py                              # ← 一把跑完 offline 全部，聚合退出码（CI/pre-commit 门禁）
# 或单跑：
python3 tests/offline/test_builders.py            # 遍历所有 <op>_cases.json：build→引擎，断言自洽
python3 tests/offline/test_design_conformance.py  # builder 是否忠实实现计算流（按选中模版独立展开校验）
python3 tests/offline/test_negative.py            # 负向：喂坏输入，断言各门禁确实会咬
python3 tests/offline/test_public_instruction_rates.py  # asc-devkit 公开性能表的 300 个组合与别名
python3 tests/offline/test_overlap_model.py       # 引擎 overlap 标定回归
python3 tests/offline/test_engine_knobs.py        # 引擎各建模旋钮回归（带宽/争用/显式开销/填充/bound/均衡）
```

- `test_builders.py`（**自洽性**）：自动发现 `references/examples/<name>_cases.json`，按约定配
  `references/examples/<name>_flow.json`，经通用解释器 flow_runtime 路由+解释。断言不抛异常（引擎强校验
  op/dtype、UB 可行）、memory 字节合理、指令非空、bound 合法、cp4≥理论。验「能不能跑、跑出来合不合理」。
- `test_negative.py`（**门禁会咬**）：喂坏输入断言对应层报错——perf_testcase 校验（未知 dtype / 符号维 / 非白名单 tuning / case_id 重复 / 未知字段）、引擎强校验（未知指令 / 不支持 op·dtype / Cast 非法对 / elements 缺失 / UB 不可行）、flow_runtime 路由（attr 谓词命中、无命中、强制不存在模版）。防止某次改动让门禁变成永远 PASS。
- `test_design_conformance.py`（**设计符合性**）：验「**计算流→评估脚本** 这一步翻译对不对」。
  每个算子的计算流是仓内正式文件 `references/examples/<op>_flow.json`（算子=多模版，每模版一条 copy_in/vf/copy_out 流，
  既是唯一真源也是金标准）。测试经解释器得 case dict（含路由到的模版名），再**独立**按该模版展开期望（低精度补 Cast、
  可选输入按 when 取舍），断言指令流逐条相符、元素数 full=N/reduced=行数、gm_read=Σ copy_in 字节（同名两次=重读两次）、
  gm_write=Σ copy_out 字节。独立展开 vs flow_runtime 解释器互为实现。缺 flow.json 直接判失败。
- `test_public_instruction_rates.py`：用独立摘录的 `fixtures/asc_devkit_instruction_rates.csv` 校验 300 个公开指令/类型组合，另验修正项的 dtype 别名及兼容指令映射。fixture 标明 asc-devkit 文档和修订版本；未覆盖类型与特殊模式不据此宣称通过。
- `test_overlap_model.py` / `test_engine_knobs.py`（**引擎回归**）：前者守 overlap 标定；后者逐档验 GM 带宽口径 / 读写争用 / 显式固定开销 / 流水填充排空 / bound 分类 / 尾块负载均衡的数值行为符合各自公式（参考值从硬件 JSON 自推）。

## onboard 层（当前环境为 950）

1. 在已配置好的 Ascend950 环境中，按 `onboard/collect_onboard.md` 准备 CANN 和结果目录：
   `gen_ttk_csv.py` 由同源 shape 生成 TTK kernel CSV → `python3 -m ttk kernel -b=release` → 读取被测实现对应的设备侧 Task Duration；
   或跑 `kernels/<op>/` 的 `<<<>>>` 直调 kernel 取 `RESULT ... us=..`。已在 Ascend950 跑通。
2. 把实测写成一份 CSV（格式见该文档）。**实测 CSV 是 point-in-time 板上快照，不随 skill 提交**——
   存到 skill 外（PR/评审记录/个人目录）。skill 只提供可复现的 kernel + 对照工具。
3. 对照：

```bash
python3 tests/onboard/compare.py --file /path/to/measurements/add_measured.csv     # 对照一份 CSV
python3 tests/onboard/compare.py --dir /path/to/measurements     # 扫一批（跳过 _ 开头）
```

存在 BELOW_FLOOR 时 `compare.py` 退出码 1，可作精度回归门禁。其判定逻辑（`classify()` + 退出码）由 `tests/offline/test_compare.py` 用合成数据自测（无需板子、无需实测快照，已纳入 `run.py`）。

## 当前覆盖（算子族）

| 族 | 算子 | 压的模型路径 |
|---|---|---|
| 归约/Norm | softmax, rms_norm, layer_norm | reduce 引擎、多级/双归约、广播、低精度 fp32 内算、`row_split_2pass` 非全载 |
| 归约(纯 reduce) | reduce_sum | 单归约、reduce 单发资源 |
| 融合 Norm | add_rms_norm | add+rmsnorm 融合流、多流 L2 并存标注 |
| Elementwise 纯 ALU | add | mte-bound 判定、双发 ALU、动态输入(repeat) |
| Elementwise+超越 | gelu(tanh), swiglu | **超越(Exp/Reciprocal)+ALU overlap（标定核心）** |
| Vector-bound 链 | vchain | 多指令链 vector-bound、bf16 低精度 Cast 路径 |
| 类型转换 | cast | conversion 快/慢路径、读写字节不对称(bound 翻转) |

共 10 个 example 算子（`references/examples/<op>_flow.json` + `<op>_cases.json`）。shape 均覆盖小/中/大三档（对应 固定开销 / L2 驻留 / HBM 带宽 三区间）；softmax/rms_norm/layer_norm 各有非全载用例触发 `row_split_2pass`（x 读两次）。onboard kernel 覆盖其中 add_rms_norm/reduce_sum/rms_norm/swiglu/vchain。

## 扩一个算子进 ST

写 `references/examples/<op>_flow.json` + `<op>_cases.json`（不写任何 .py），offline 的 `test_builders.py`/`test_design_conformance.py` 自动纳入，上板后 `compare.py --file` 对照。完整步骤与 flow 编写契约见 `references/builder_template.md`「加一个新算子的步骤」，不在此重复。
