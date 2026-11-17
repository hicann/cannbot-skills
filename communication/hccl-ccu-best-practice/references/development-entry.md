# CCU 开发入口：先复用契约，只补查差异

## 最短阅读路径

1. 读取用户需求和已有设计；只定位本次 executor、模式和接口。按 [算子索引](operator-index.md) 只读本算子接入卡；覆盖AllGather、AllReduce、ReduceScatter、Broadcast、Reduce、Scatter。AllGather Sole另有框架方法卡，其他算子不套该卡；不再从Mesh/NHR全文提炼已有字段。
2. 首次运行 `inspect_sources.py`，其 `knowledge.cards` 给出各卡的 MATCH/CHANGED/UNVERIFIED 和文件范围。后续角色只需运行 `check_knowledge.py --hccl-root ... --hcomm-root ... --op <算子目录名> --summary`，核对当前字节并复用记录；无需再生成全算子 inventory。
3. MATCH：在卡的模式/executor/分支范围内直接使用已总结结论；HEAD 变化但文件字节相同不单独使卡失效。
4. CHANGED：用下方 `--explain` 查看经基线哈希核验的差异区段与符号位置，只补查受影响契约；UNVERIFIED：只定位缺失来源。新增依赖、不同 executor/拓扑、旧布局或卡未覆盖的算子，定向补查对应部分。不要因为一张卡变化而全仓重研。
5. 按算法需要展开 API/同步、实现或验证参考。新 Ring/Tree 可以独立设计，不要求先找到同算法蓝本。需要借鉴 Ring 样本时使用已修复版本，不从任务目录复制未经核对的旧实现。

`check_knowledge.py`先验证`--op`对应HCCL算子目录存在；目录别名或拼写错误返回2，不自动纠正。输出`operator.checked_cards`与`operator.status`说明是否核验算子卡：CHECKED、NOT_SELECTED（显式只选其他卡）或NO_CARD（该算子暂无卡）。默认检查无算子卡时返回1，补查后可显式选择公共/实际依赖卡，但不能据此声称核验了算子接入。返回1也表示至少一张卡需要补查，不是任务/引擎失败；可选历史Ring样本缺失不阻止独立设计，仍复用其余MATCH卡。

```bash
python3 "$SKILL_DIR/scripts/check_knowledge.py" \
  --hccl-root "$HCCL_ROOT" --hcomm-root "$HCOMM_ROOT" --op reduce_scatter \
  --cards reduce-scatter-sole sync-limits generator-arith \
  --summary --explain --diff-lines 60
```

`--cards`只选实际使用卡，不以缩小范围代替未覆盖依赖检查；可选Ring样本仅在借鉴时加载。`review_hint`在基线Git对象可用且字节符合记录时给DIFF_AVAILABLE，缺对象/哈希不符明确BASELINE_UNAVAILABLE，不联网拉取也不改基线。变更区段/附近符号只是导航，不证明变更与契约无关；卡仍为CHANGED/UNVERIFIED。

`knowledge-baseline.json` 的卡是明确范围的来源记录，原 `source-map.json` 仍是较早的接口/样本来源索引。两者范围和调研版本不同；旧来源报告的 changed 不等于当前卡不能使用。既不通过刷新哈希掩盖差异，也不重复调查已 MATCH 且适用的结论。

## 一次形成、跨角色复用的接入记录

将下表放在既有 DESIGN 的来源/接入依据中，不新增一组重复报告：

| 字段 | 内容 |
|---|---|
| 已加载卡 | Skill 路径/版本、card-id、校验输出/哈希 |
| 本次范围 | 算子、实际 executor/方法/分支、CCU 模式和拓扑入口 |
| 可复用结论 | 字段单位/偏移、资源消费、接口/选路与构建链路 |
| 补查结果 | 文件/函数、实际字节哈希、变化点及结论 |
| 剩余缺口 | 影响实现或验收的具体问题；没有则写无并直接进入设计 |

Developer 继承这份记录和冻结契约，只调查实现新增依赖或变化部分。Reviewer 独立核对关键算法和源码对应关系，首轮覆盖全变更；复审核对 delta 和受影响证明，引用未变化的首轮证据，不重新写一套架构调查。Verifier 消费冻结矩阵与已有命令/证据契约，不重做算法设计。短交接列输入身份、产物、变化和缺口即可，不复制整篇手册。

数据流证明只维护在每template的spec，完整参数/同步/算术映射只维护在JSON。其他报告引用路径、字段/阶段和哈希，记录自己的判断、代码落点、反例或差异；不复制完整原表。需要展开的关键证明照常保留，不以篇幅限制省略正确性依据。

## 首次构建前消除已知缺口

- 算术能力：复用 `generator-arith` 卡并按目标生成器核对实际lowering；V1拒绝MUL/SUB，未知代际使用交集；在同步附件的compute_ops声明并检查。
- 同步预算：按 [实际配额与检查器](api-and-synchronization.md#设计期同步契约检查) 检查 channel 接收表、XN、Notify slot/mask 和去重边界；每个声明对照实现。
- 运行环境：按 [验证参考](validation.md#0-在冻结矩阵前核对运行条件) 核对将加载的库与可信构建产物，确定 CCU 模式可用；源码 HEAD 或符号存在不能代替二进制身份。
- 测试可执行性：按接入卡核对 count/size、root/reduceType 和 loop 可达性；六算子先按 [输入模型](test-input-preflight.md) 计算实际量，非零root/非sum单独核对runner；先算宿主机内存，再决定矩阵。不可表达、不可达和容量受限分开标明，不把缺运行证据写成 PASS。

本入口减少重复查询和已知返工，不免除新算法的数据流/同步证明、独立评审与真实验证。

## 来源报告修订

首次完整来源报告可用 `inspect_sources.py ... --op <目录名> --output "$TASK_DIR/source-report.json"`。重新设计时选择全新 `source-report-r<正整数>.json`，例如 `--output "$TASK_DIR/source-report-r2.json"`；旧报告保留原名和字节作历史归档，不删除、不覆盖。由调用方的设计记录绑定本次文件名与实际 SHA-256，按原失效流程重新核对下游报告。报告是设计输入，不由生产实现者覆盖成候选快照。

仓内只放行当前算子 `src/ops/<op>/docs/<task-id>/` 下上述两类文件名，task-id 沿用安全目录名，父目录须已创建；其他输出可放仓外。所有修订均以排他创建写入，符号链接按实际落点检查。

调用方已提供重新核验的知识基线时，给 `inspect_sources.py`、`check_knowledge.py`、`check_contracts.py` 和独立 sync/arith 检查器传入同一 `--baseline /path/to/reviewed-knowledge-baseline.json`。来源报告的 `knowledge.baseline_sha256` 标识实际使用的基线；调用方同时绑定基线文件、来源报告和核验依据。来源/能力继续按当前文件字节检查，不通过只更新哈希来消除漂移。
