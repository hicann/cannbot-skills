# 自定义算法的实现入口与来源校验

本文消费已确认的算法设计和 Dataflow Spec，只说明实现前如何核对接口来源、选择实现骨架及定位缺口。Ring、Tree 等调度语义由 `hccl-aicpu-design` 给出；不要求仓内存在同类实现。完整构建和双轨验收仍按 Plugin 与验证能力契约执行。

## 1. 最小输入与阅读路径

1. 消费已确认的算法、通信域/拓扑、支持模式、dtype/reduceOp、选路范围和 Dataflow Spec；缺少设计结论时返回输入缺口，不从历史实现猜测。
2. 只读 [executor 输入契约](11-operator-contracts.md) 中目标算子及其绑定 executor 的条目。
   设计中的逻辑阶段不能自动推出目标版本的 scratch、线程数或资源接口。
   AllGather Parallel 使用[四阶段契约](12-allgather-parallel.md)，不重新从 Mesh/NHR 全文提炼布局。
3. 查 [API 按动作导航](02-api-reference.md#0-按开发动作选择接口)，只展开本算法用到的接口。
4. 用 [五参数说明](09-template-data-params.md) 对照已确认 Spec 的数据流和实际 executor 赋值。
   只有涉及对应资源/同步问题时，再展开 `03-resource-model.md` / `08-thread-sync-patterns.md` 的相关节。

不默认扫描所有 template、完整读 NHR/Mesh、阅读全部参考或深入 HCOMM 内部实现。
若要修改/继承某个已有 template，需阅读被改方法及其调用关系；复用整段实现时阅读对应完整函数。

## 2. 先校验已有知识，只调查缺口

本 Skill 的算子卡和接口导航附有 [源码基线](source-baseline.json)。路径相对调用方提供的仓根，
不含维护者机器上的绝对路径。只读校验：

```bash
python3 "$SKILL_DIR/scripts/check_sources.py" --repo "$HCCL_REPO" \
  --hcomm-repo "$HCOMM_REPO" --op all_reduce
```

- `MATCH`：清单内文件内容未变，复用其已总结契约。仍确认本次选择的 executor/分支确实落在卡片范围。
- `CHANGED`：按输出文件、卡片中的函数名定向核对；缺文件时定位新位置。不回退成全仓调研。
- `UNVERIFIED`：例如未提供 HCOMM 仓，不能宣称其头文件已核验；按实际使用的接口定向取得证据。
- 新 executor、不同运行模式或新接口超出卡片范围：只补充这些差异，不要求找到同算法蓝本。

AllGather Parallel 校验追加 `--contract all-gather-parallel`。扩展卡单独标注基准 commit，
不影响其他算子的读取范围；共同文件变化仍须定向核对。将本次输出留在任务目录并引用到 Spec，
不以“执行了 preflight”代替契约来源校验。

脚本比较工作区实际字节，包含未提交修改；仓 HEAD 变化但清单文件未变不会单独使卡片失效。
它不分析传递依赖、拓扑支持、已安装 HCOMM 二进制或算法正确性。新增依赖、编译条件及部署版本仍需核对。
`workflow.py preflight` 继续负责任务身份与选路入口，不以此脚本替代。

在现有 Spec 的「元信息/依据」保存一个简短契约记录：

| 字段 | 记录内容 |
|---|---|
| 适用范围 | 算子、executor 类/方法/分支、运行模式、输入输出布局 |
| 证据 | 本次源码身份、基线校验结果、额外依赖文件及内容哈希 |
| 结论 | 字段单位、buffer 落位、资源消费方法、所需接口 |
| 未覆盖 | 需要定向核对的具体问题；无缺口则直接设计 |

调研中发现的接入风险在同一记录中列明“问题 → 修复或不影响的依据 → 验证方式”。
首次构建前关闭影响已声明路径的缺口，尤其是 GetRes 消费、成本参数透传、注册与实际选择入口。

后续角色复用同一记录和未变化的依据，不重复架构调查；Reviewer 仍独立检查本次算法的关键结论。
维护者只有核对受影响的契约并更新文档后，才更新基线哈希；不能通过刷新哈希把未知行为标成已验证。

## 3. 算子骨架与算法调度分离

`assets/barebone.{h,cc}.tpl` 提供可生成的类和接口，不预置 Mesh、归约、rank 数对应的线程数或切片规则。
生成器负责两个源文件和两处 CMake，不负责实现算法：

```bash
python3 "$SKILL_DIR/scripts/new_template.py" --repo "$HCCL_REPO" \
  --op all_reduce --class InsTempAllReduceRing --pattern barebone --dry-run
```

核对预览后去掉 `--dry-run`。算法属性须匹配目标仓的 `AlgoType` 和选路映射；
未识别的自定义名字默认 `UNKNOWN`，不自动伪装成 Mesh，也不假设存在 `AlgoType::RING`。
需要新枚举时按父层注册/选路契约接入，不能仅给生成器传一个不存在的值。

骨架在资源和数据流未实现时返回错误，成本候选为空；scratch 的 `0` 是待替换占位。
完成 Spec 中的资源、预算、边界和数据流后才解除错误。生成成功或静态检查通过都不代表算法可运行。

## 4. 自定义实现的接口核对

算法阶段、片归属、逻辑 peer、同步和资源预算由输入 Spec 确定。实现时可从目标版本 `channel.h` 的 `CreateChannelRequestByRankId` 定向核对申请接口，但协议/拓扑覆盖须检查实现与调用条件；不能虚构 `CalcChannelRequestRing`，也不能照搬 Mesh 全连接申请。使用 `SendRecv*` 时按 wrapper 的握手语义核对发送、接收 channel 的实际配对。静态规则不能证明死锁自由；边界覆盖、构建和目标通信域的功能验证继续执行。

## 5. 可选历史实现

用户指定借鉴既有实现时，可按[历史样例索引](13-historical-examples.md)定位固定源码对象；历史代码和测试结论不能替代当前设计、来源核对或功能验收。调研委派、交接格式和效果度量由上层 Plugin 与评测维护，不在本实现参考中编排。
