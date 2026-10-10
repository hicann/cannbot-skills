# 基线证据与回归比较

每轮 `run.sh` 在日志父目录下创建唯一 `run_<时间>_<随机串>/`，不覆盖历史批次：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 计划用例、阶段、源码标识、归档包身份、当前安装库哈希、测试配置 |
| `results.jsonl` | 每用例完整算法集合、首末算法、实际算法覆盖配置、mpirun 退出码、Checker 汇总、原始日志路径/哈希、测试二进制哈希 |
| `result.txt` | 便于阅读的结果，包含首末算法；不代替 JSON 证据 |
| `<序号>_<用例>.log` | 完整原始日志 |

`--log-dir` 选择父目录，`--role ordinary|baseline|directed|regression` 声明固定角色，`--phase` 仅作自定义标签；不是手写 run-id。
基线/回归角色必须提供 `--artifact`，并在结束时自动检查证据完整性；定向角色必须提供 `--expect-algo` 和非空 `HCCL_ALGO`。不得用标签命名规避角色要求，旧记录不根据标签自动推断角色。
普通测试可不提供归档；`compare` 和构建门禁不接受 ordinary 记录作为基线。`--dry-run` 永远不能作为基线。

## 运行前先保存包

归档工具位于 `hccl-aicpu-best-practice/scripts/archive_package.py`，仅复制已有产物，不构建、不安装、不切源码。
它要求成功退出码、构建日志、源码快照标识、构建环境标识及该包的 host/device 参考库；拒绝覆盖已有归档目录，也拒绝放回原包输出目录、仓内 build/build_out。自定义清理目录另传 `--clean-root`。
源码快照须包含未提交 diff 和新增文件，不能只填写 HEAD。

```bash
python3 <hccl-aicpu-best-practice路径>/scripts/archive_package.py \
  --repo <HCCL源码根> --host-library <该包的libhccl.so> --device-library <该包的libscatter_aicpu_kernel.so> \
  --package <已构建包.run> --build-log <完整构建日志> --exit-code-file <构建退出码文件> \
  --source-id '<commit+工作区快照哈希>' --environment-id '<CANN版本+工具链+构建配置标识>' \
  --build-command 'bash build.sh --pkg --full' --output <任务产物目录>/baseline-v1
```

输出 `artifact.json`、归档包、日志及退出码。候选包使用不同目录，例如 candidate-v1、candidate-v2。
切回已有版本时安装归档包并按 hccl-vm 部署 device 库，**不重新构建相同版本**。

归档时参考库必须来自该安装包或同次打包的最终 staging 产物（注意 strip/签名等后处理）。`--artifact` 会在启动测试前核对归档包内容哈希及现场 host/device 库与参考哈希是否一致，旧包身份配上新库会立即失败。工具不自行解包证明参考库来源，调用方仍须保存其打包/提取依据，不能拿当前安装的任意库充当旧包参考库。

## 运行与检查

```bash
unset HCCL_ALGO
export HWLOC_COMPONENTS=-gl,-opencl
bash <hccl-test-tool-hvm路径>/run.sh -m AI_CPU -o allgather -d int32 -s 64,1M -t 112,118 \
  --role baseline --phase new-selector --artifact <任务产物目录>/baseline-v1/artifact.json \
  --env HCCL_USE_NEW_SELECTOR=1 --log-dir <任务产物目录>/tests
```

同时确认通信域未设置覆盖配置。`--source-id` 可由归档自动填充；显式填写时必须与归档一致。
自定义测试程序用 `--test-bin-dir <目录>`，工具记录实际二进制哈希。

```bash
# 首次候选构建之前：旧包已归档，检查计划是否跑齐、算法名与完整日志是否保留。
python3 <hccl-test-tool-hvm路径>/evidence.py check <baseline运行目录> --require-role baseline
# 额外要求所有用例通过；相同基线 FAIL 不会自动豁免。
python3 <hccl-test-tool-hvm路径>/evidence.py check <baseline运行目录> --require-role baseline --require-pass
# 候选包按相同矩阵和配置运行（role 改为 regression，artifact 指向候选包），然后比较。
python3 <hccl-test-tool-hvm路径>/evidence.py compare <baseline运行目录> <regression运行目录>
```

比较器要求两边角色分别为 baseline 和 regression，按用例键比较配置、测试程序身份、全部选中算法及结果，不按文件行序比较。
默认与新 selector 无配置是两组独立基线，分别比较；包和源码身份允许变化，构建环境与测试条件应匹配。
证据完整的 FAIL 可以记录，但 `--require-pass` 和 `compare` 均失败；继续定位，不将其计入通过数。

缺项处理顺序：从该批次原始日志/完整 stdout/归档恢复 → 确实缺失时安装缓存旧包补测 →
无可信包或构建输入改变才重建。恢复旧日志时另存完整运行目录与配置证据，不修改现有日志或编造缺字段。
完整旧证据若仅缺 role，应核对原始命令与配置后在独立副本迁移，保存原记录及迁移依据；不为补角色元数据重编或重测，也不能仅凭标签推断。旧格式结果只有 PASS/FAIL 时不能自动补出包身份或算法名；不要把不完整旧记录当作新 schema 的有效基线。

当前解析器以完整 `CHECKER_RUN_SUMMARY`（含单算子与大图数量）及一次成功 mpirun 为通过证据，
任何失败汇总、缺少退出标记、多个 mpirun 或算法命中不一致均判 FAIL。若 Checker 版本改变输出格式，先更新解析器并回归，不退回仅搜索任意 `Checker Success`。

工具离线回归：`python3 -B <hccl-test-tool-hvm路径>/tests/test_evidence.py`；不运行 HCCL ST/UT、编译安装或实际 Checker。

多批次验证按 [测试覆盖计划](test-plan.md) 提前编排，由 plan.py 串行执行并自动汇总；基线完整性仍在首次候选构建前通过父层 [workflow.py 门禁](../../hccl-aicpu-best-practice/references/build-verification.md#23-预检与候选构建入口) 检查。

失败或缺失用例的计划生成、证据补齐及切包前门禁见 [定点补测与切包前检查](recovery.md)。
