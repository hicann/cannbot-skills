# scenario subagent 操作细则

供 scenario subagent 使用。目标是**建立可复现的推理输入 + 跑通精度基线 + 定一把可机判的判定口径**，不做任何性能优化、不改优化代码。下面以本仓 `cann-recipes-infer/models/<model>/` 的标准结构为准；个别模型入口命名略有差异，按实际为准。

## 0. 工作目录纪律

所有命令先 `cd cann-recipes-infer/models/<model>/` 再跑，不要从仓库根目录直接执行，避免日志和产物落到错误位置。

> 路径约定：文中 `cann-recipes-infer/...`（如 `cann-recipes-infer/executor/scripts/set_env.sh`、`cann-recipes-infer/dataset/default_prompt.json`）是相对仓库根的**文件位置**（`init.sh` clone 的参考仓），用于"去哪找 / 改这个文件"；`cd` 进模型目录后命令里出现的 `infer.sh`、`config/<scenario>.yaml`、`infer.py` 是相对当前模型目录的。两类路径不要拼接。

## 1. 锁定推理入口与场景

- 入口固定是 `bash infer.sh`：它 source `cann-recipes-infer/executor/scripts/set_env.sh` 和 `cann-recipes-infer/executor/scripts/function.sh`，把 `config/<YAML_FILE_NAME>` 作为配置，调 `launch` 启动。
- **选场景 = 选 YAML**。`config/` 下的 YAML 文件名编码了关键场景维度，例如 `decode_r1_rank_16_16ep_a8w8.yaml` = decode 阶段 / 16 卡 / 16 EP / W8A8，`prefill_r1_rank_32_32dp_32ep_a8w8.yaml` = prefill / 32 卡 / DP+EP。按要优化的 workload 选对应 YAML，改 `infer.sh` 里的 `YAML_FILE_NAME`，或直接 `python infer.py --yaml_file_path config/<scenario>.yaml`。
- 多机场景：节点 IP 和 cann_path 在 `cann-recipes-infer/executor/scripts/set_env.sh`，每个节点都要执行 `bash infer.sh`。

## 2. 构造可复现输入

输入由两部分决定：YAML 里的 runner 设置（input_len / seq_len、batch、output_len 等）+ 提示词来源（`cann-recipes-infer/dataset/default_prompt.json` 或自定义）。`infer.py` 通过 `generate_prompt(runner_settings)` 把两者组合成 `preset_prompts`。

复现要点，逐项固定并记录：

- **固定提示词**：用 `cann-recipes-infer/dataset/default_prompt.json`，或写死一个自定义 prompt，把原文记进场景记录。
- **固定形状**：input_len / seq_len、batch、output_len（max_new_tokens）写进 YAML 并记录。
- **固定解码**：用贪心 / 确定性解码（确认 YAML 没开采样，如 `do_sample: false` 或 temperature=0），保证逐 token 可复现。随机种子一般已在 `infer.py` 固定（`torch.manual_seed(42)` + `torch.npu.manual_seed_all(42)`），但仍要确认没有别处引入随机性。
- **阶段一致**：prefill 场景用 prefill YAML、decode 场景用 decode YAML，不要混；这一步定的阶段要和后续 profiling 采集的阶段一致。

## 3. 跑通基线并采样输出

```bash
cd cann-recipes-infer/models/<model>
# 把 infer.sh 的 YAML_FILE_NAME 指向目标场景，然后：
bash infer.sh
# 或显式指定：
python infer.py --yaml_file_path config/<scenario>.yaml
```

`infer.py` 默认先 warmup 再正式 generate；取正式 generate 段的输出。第一次先确认无 crash、吐字正常，再把生成的文本 / token 采样存下，作为基线输出样本。

## 4. 建立精度 / 功能判定口径

按模态选一把**可机判**的尺子，写进场景记录，后续所有 round 都用同一把：

- **LLM（贪心可复现）**：以基线输出的 token ids / 文本做逐字对比；附最低可用性门槛——可读、不重复、非全零、不提前 EOS。
- **MoE / 量化模型**：同上；量化基线允许与浮点有界误差，记录可接受的误差范围。
- **图像 / 视频生成**：固定 seed 和 step 数，比对输出图 / 帧的关键指标或可视一致；记录采样器和 step。

不要只写"看起来对"——要给出能自动判定的规则（逐 token 一致 / 误差阈值 / 指标范围）。

## 5. 落盘场景记录

落到 `optimization-analysis/<case>/scenario.md`，至少包含：

- 模型 / case、代码目录、推理入口、选用的 YAML（标明 phase / 卡数 / 并行 / 量化）
- 输入：prompt 原文或来源、input_len / seq_len、batch、output_len、解码参数
- 基线运行命令（可直接复制重跑）
- 关键环境：`set_env.sh` 的 cann_path / IPs（多机）、torch_npu 版本、卡数
- 基线输出样本（截断，够复核即可）
- 判定口径（机判规则 + 阈值）
- 阻塞项（如有）

## 6. 阻塞处理

- 跑不通时只回报阻塞原因 + 需要补的信息（权重路径、卡数、缺依赖等），不要绕过或伪造基线。
- 若模型还没适配进框架、或没有可运行 baseline，属于本流程的前置条件未满足：建议先走 `model-infer-migrator` 或 `model-infer-optimize` 阶段 0 建立 baseline，再回到本流程。

## 只回传

场景记录文件路径 / 输入样本（或构造脚本）路径 / 基线推理命令 / 精度口径摘要 / 阻塞项（如有）。
