---
name: ops-knowledge-optimization-ingest
description: 对单个 AscendC / 昇腾 NPU 算子做受控劣化挖矿，把实测得到的性能退化机制编译成 cannbot-knowledge OKF 优化点卡，摄入共享知识库 runbooks/operator-optimization/；当需要从算子劣化中沉淀实测优化经验并入库时使用。
disable-model-invocation: true
argument-hint: category/op --dataset_dir PATH [--knowledge-root PATH]
allowed-tools: Read, Glob, Grep, Write, Bash, Agent
---

# ops-knowledge-optimization-ingest — 单算子劣化优化点生产者

对单个 AscendC 算子做受控「劣化 + 经验抽取」：每轮寻找一个新的性能退化机制，使算子比当前版本慢至少目标阈值，同时保持可编译运行；把发现的机制记录为原始经验，再**编译成 OKF 优化点卡（`OPT-N`）摄入共享知识库**，供后续检索、RAG 与性能优化使用。

产出**解耦的两层知识**：
- **原始层** `experience_lib` JSON —— 挖矿轨迹（含 `code_diff` / 前后 profile 等富字段），只作原始记录与下游 RAG，不当作知识正文。
- **curated 层** `<knowledge-root>/runbooks/operator-optimization/single-op-degradation.md` —— 从原始层编译的**实测优化点库**（扁平 `OPT-N`，跨算子单一共享、增量合并）；**这才是摄入知识库的知识**，根因与收益取自**本地 msprof 实测** profile，`置信度` 记「已验证(独立eval)」。`--knowledge-root` 缺省 = 仓库根。

> **评估在本地 NPU 上完成**（`msprof` 采集，非远程服务）：`single_op_evaluate.py` 先用算子工程自带的 build 步骤编译，再用 `msprof` 包裹 run 步骤采集 `task_duration_us`(=score) 与 aic-metrics。需真实 Ascend NPU + CANN + `msprof`/`npu-smi`（先 `source` CANN `set_env.sh`）。

## 工作流程
1. 收集输入：`op_category` / `op_name`、`dataset_dir`、`threshold`、`experience_lib`、`work_root`、`max_round`、`--knowledge-root`；确认本地 NPU + `msprof` 可用（评估在本机采集）。
2. 使用 `resolve_source.py` 定位算子源码，创建工作副本和原始基线备份，后续只修改工作副本。
3. 使用 `single_op_evaluate.py` 评估工作副本，生成 `baseline_eval.json`。
4. 读取经验库并剥离大字段 `code_diff`，用于判断本轮机制是否与已有经验重复。
5. 建立 `progress.json`，按 `max_round × {degrade, refactor, structured_tiling_degrade}` 驱动所有工作单元。
6. 每个工作单元只做一处改动，统一评估后按阈值判定是否达标；达标时结合前后 profile 写入原始经验记录（`experience_lib`）。
7. 只有 `degrade` 模式达标才推进下一轮基线；`refactor` 和 `structured_tiling_degrade` 只产出经验，不改主基线。
8. 挖矿收尾前检查 `progress.json` 无 `todo`，并重新读取 `experience_lib` 确认新增经验已写入。
9. 编译入库：把本次新增的经验记录编译成 `OPT-N` 卡增量合并进 `single-op-degradation.md`，补 `index.md`、写 `log/` 条目、刷新知识图谱，最后跑校验脚本。详见 `references/workflow.md`「编译进知识库」。

## 脚本工具
- `scripts/resolve_source.py` - 根据 `dataset_dir`、算子类别和算子名定位源码目录，支持常见两级或三级目录布局。
- `scripts/single_op_evaluate.py` - 本地评估：build 算子工作副本 + `msprof` 包裹 run 采集，生成含 `score`(task_duration_us) 和 `performance_metrics` 的 JSON。支持 `--op_dir`/`--build-cmd`/`--run-cmd`/`--device`/`--full`/`--quick`/`--kernel-match`。
- `scripts/validate_degradation_knowledge.py` - 校验 `single-op-degradation.md` 的 `OPT-N` 结构、引用闭合、悬空锚与坏实践字段；编译入库后必跑。

## 参考资料
- `references/workflow.md` - 完整劣化流程、三种探索模式、经验记录格式、本地评估（msprof）配置，以及「编译进知识库」的字段映射与治理三件套。
- `STRUCTURE-runbook.md` - `single-op-degradation.md` 的 `OPT-*` / `CT-*` / `AP-*` 卡骨架（编译入库时严格对齐）。
