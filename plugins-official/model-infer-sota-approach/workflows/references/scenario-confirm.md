# 确认推理场景与性能目标（操作细则）

供主 agent 在步骤 1 使用。这一步是**交互式**的、由主 agent 直接做（subagent 不与用户对话）：先从仓库把功课做足，再用结构化提问跟用户把"优化哪个场景、目标是什么、用什么尺子判定"敲定。**只锁定场景**——不写 Dashboard、不跑推理、不采集。

## 原则

- **先勘察后提问**：别空手问用户；先从仓库找出候选场景，带着候选去确认。
- **一次问清**：能批量确认的用一次结构化提问（AskUserQuestion）摆出来，不逐条盘问。
- **前置条件优先**：高阶流程在已有可运行 baseline 之上做；没有就先去建，别硬启动。

## 1. 自动勘察候选场景（先做功课）

从仓库收集，整理成候选清单：

- **模型目录 `cann-recipes-infer/models/<model>/`**：
  - `infer.sh` 里的 `YAML_FILE_NAME` —— 当前默认跑哪个场景。
  - `config/*.yaml` —— 每个 YAML 文件名编码一个场景：phase（`prefill` / `decode`）、卡数（`rank_16` …）、并行（`16ep` / `16tp` / `32dp` / `32sp` …）、量化（`a8w8` / `a8w8c8` …）。逐个列出可选场景。
  - `README.md` —— 推荐场景说明。
- **已有产物**：`progress.md`、`optimization-analysis/`、`baseline/baseline_metadata.json` —— 判断是否已有 baseline、是否定过场景、之前优化到哪；`progress.md` 在则从中直接取模型 / 权重 / 部署配置等前置信息作场景定义默认值，不重新推导。

候选清单每条 = `{YAML 名, phase, 卡数, 并行, 量化, 是否已有 baseline}`。

## 2. 前置条件检查

确认模型已框架适配、`infer.sh` 能跑通，且有可复现精度的 baseline（或至少能现采）。

**不满足时**（代码缺失 / 跑不通 / 无 baseline）：停在这一步，告诉用户高阶流程的前置未满足，先走 `model-infer-migrator` 或 `model-infer-optimize` 阶段 0 建好 baseline 再回来。不要在没 baseline 时硬启动后续流程。

## 3. 和用户确认（结构化提问）

带着候选清单，用一次 AskUserQuestion 把下面敲定（已能从仓库确定的项给默认值让用户确认，只对真正缺的追问）：

- **Q1 优化哪个场景**：从候选 YAML 里选一个，或用户自定义。这一项决定 phase / 卡数 / 并行 / 量化，候选 YAML 直接作为选项。
- **Q2 workload 侧重**：decode 单步时延 / prefill 吞吐 / 长序列 / batch·并发 / 整体。它决定后续 profiling 采哪个阶段、候选发现往哪使劲。
- **Q3 精度 / 功能判定口径**：默认"贪心逐 token 与 baseline 对齐 + 可读 / 不重复 / 非全零 / 不提前 EOS"；量化或生成类模型按模态调整（口径细节见 [`scenario-setup.md`](scenario-setup.md)）。让用户确认或改。
- **Q4 性能目标（可选）**：给"具体数值目标（TPOT / 吞吐 / E2E 时延）"和"无硬目标、尽量榨"两类。无目标不阻塞，标为可选。
- **Q5 输出归档目录**：默认 `optimization-analysis/<case>/`，确认或改。

## 4. 处理"只给了泛化目标"

用户只说"帮我优化 XX 模型 / 提速"时，不要直接开盘问、也不要擅自决定：主 agent 从第 1 步候选里挑最可能的场景（通常是 `infer.sh` 当前的 YAML，或 README 主推场景）作为**推荐**，连同其它候选一起摆给用户在 Q1 里选。

## 5. 锁定并交棒

确认后把结果整理成**场景定义**：

- 模型 / case、代码目录、推理入口
- 选用 YAML（标明 phase / 卡数 / 并行 / 量化）
- workload 侧重
- 精度 / 功能判定口径
- 性能目标（或标"可选"）
- 输出归档目录
- `progress.md`（共享状态文件）路径：找得到就记录其实际位置；找不到就按本插件 `progress_template.md` 约定创建一份后记录。该路径后续作为 `<progress_path>` 下传给各 subagent，位置不在 skill 里钉死。

这份场景定义是步骤 2 `scenario` subagent 的输入——它据此构造可复现输入、跑基线、落 `scenario.md`（执行细节见 [`scenario-setup.md`](scenario-setup.md)）。本步不写 Dashboard（Dashboard 在候选发现完成后才初始化）、不跑推理。

## 输出

锁定的场景定义（可直接喂给 scenario subagent）+ 前置条件结论（满足 / 需先建 baseline）。
