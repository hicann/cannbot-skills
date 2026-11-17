# CCU Skill 维护与评测

本文供 `hccl-ccu-best-practice` 的维护者使用，说明 Skill 工具回归、知识来源基线维护和行为评测。使用 Skill 开发通信算法时，按 [算法验证参考](../communication/hccl-ccu-best-practice/references/validation.md) 核对构建、功能、选路和性能证据。

## 工具回归

在 `cannbot-skills` 仓库根目录执行：

```bash
CCU_SKILL_DIR="$PWD/communication/hccl-ccu-best-practice"
python3 -B -m unittest discover -s "$CCU_SKILL_DIR/tests" -v
python3 "$CCU_SKILL_DIR/scripts/check_launch_contract.py" \
  "$CCU_SKILL_DIR/assets/launch-contract.example.json"
```

## 来源基线维护

确认目标 HCCL/HCOMM 仓库和算子后，用实际路径复核来源，例如：

```bash
python3 "$CCU_SKILL_DIR/scripts/inspect_sources.py" \
  --hccl-root /path/to/hccl --hcomm-root /path/to/hcomm \
  --op all_gather --check-baseline
```

`--check-baseline` 用于维护者复核本 Skill 调研的文件版本；不同版本的漂移不是产品代码错误，重新核验技术契约后再更新来源基线。不得只刷新哈希来消除漂移状态。

任务可先使用单独的已核验知识基线副本，无需修改已安装 Skill 的默认文件。核对受影响方法、实际能力/资源及文件字节，记录源码身份、差异和核验依据后，将副本交给来源与 sync/arith 检查器的 `--baseline`；调用方绑定副本 SHA-256 并独立检视。来源报告新增修订时保留旧报告，基线和报告变化均按调用方工作流失效及重绑定规则处理。

## 行为评测与效率记录

行为任务见 [evals.json](../communication/hccl-ccu-best-practice/evals/evals.json)，维护者记录实际评测结果，未执行不得计为通过。

| 观测项 | 记录方式与合格边界 |
|---|---|
| 正确性 | 已知算术/同步/断言和框架方法缺陷的识别与源码依据；提速不能降低原有判断正确性 |
| 重复调研 | 记录手工查阅的文件/区段及用途，统计无新增依赖时重复展开MATCH契约的次数；只读哈希校验与Reviewer必要独立核验不算冗余 |
| 无关阅读 | 是否读取未用算子、可选历史样本或无关任务类型参考；新依赖和真实漂移允许定向增量，不设固定文件数上限 |
| 重复产物 | 完整参数表/矩阵/技术推导跨文件复写处数；引用绑定及必要独立证明不算复写，不用字数硬限制省略证明 |
| 工具与读写量 | 实际调用数、原始输出/生成正文字符数；区分源码/工具验证与模型重写文字的成本 |
| 耗时 | 有事件记录时分别计源码查阅、推理、命令执行、写作与收口；仅有mtime时标近似区间，不把无落盘窗口都称为纯推理 |

同场景对比修订前后记录，并注明源码/缓存/矩阵变化。本次工具回归通过或文字评测预期不能换算为已实测节省分钟数。
