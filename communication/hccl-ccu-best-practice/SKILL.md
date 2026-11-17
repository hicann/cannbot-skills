---
name: hccl-ccu-best-practice
description: 依据目标 HCCL/HCOMM 源码设计、实现、修改或检视 CCU 通信算法，覆盖 template、CCU kernel、executor/selector 接线、启动参数、资源与同步。用于新增 CCU 算法、修改数据流或定位 CCU 实现问题；不适用于 AICPU/AIV 算法或单纯调用 HCCL API。
---

# HCCL CCU 通信算法开发

本Skill提供CCU技术能力，工作流角色、状态与最终裁决归调用方。`SKILL_DIR`为本目录，`HCCL_ROOT`/`HCOMM_ROOT`来自用户或已确认环境；生成源码写目标HCCL仓，记录放调用方指定目录，不猜测仓路径。

## 按任务阅读

先读 [开发入口](references/development-entry.md)，按实际需要展开；不默认加载全部参考。

| 技术任务 | 必读 | 按需 |
|---|---|---|
| 新算法/数据流设计 | [算法设计](references/algorithm-design.md)；[算子索引](references/operator-index.md) 中本算子的接入卡 | [API/同步](references/api-and-synchronization.md)中的所用原语、[验证预检](references/validation.md#0-在冻结矩阵前核对运行条件) |
| 实现或局部修复 | 已冻结设计/契约、[实现步骤](references/implementation.md) | [短代码范式](references/implementation-patterns.md)、受影响原语；改变偏移/收发/同步/资源/参数时补设计 |
| 独立评审/定位接口 | 原始需求、所涉设计、当前源码与原始证据 | 对应契约章节；形成独立结论后核对开发解释，不重写通用调查 |
| 构建/功能/性能验证 | [验证方法](references/validation.md)与冻结矩阵 | 已安装测试工具的对应能力；不重做算法设计 |

只读解释/评审不修改生产源码、不运行构建。AICPU/AIV和单纯HCCL API调用不走本Skill。

## 确认来源后复用

首次来源检查选择目标算子目录名，按算子索引映射，例如AllReduce：

```bash
python3 "$SKILL_DIR/scripts/inspect_sources.py" \
  --hccl-root "$HCCL_ROOT" --hcomm-root "$HCOMM_ROOT" --op all_reduce --summary
```

需要原始证据时不加summary，将完整stdout保存为source-report.json，后续只读摘要/差异。`knowledge.cards`的MATCH且适用结论直接复用；CHANGED/UNVERIFIED按开发入口的 `check_knowledge.py --cards ... --summary --explain` 定向核对。可选历史样本未使用时不加载；旧source-map漂移不使所有当前知识卡失效。新增依赖另核对，不能刷新哈希掩盖差异。完整分层仅在需要时读 [架构](references/architecture.md)。

重新设计按 [来源报告修订](references/development-entry.md#来源报告修订) 创建新文件并保留旧证据；调用方选择已核验的新知识基线时，来源与契约检查统一传入 `--baseline`，并由调用方绑定文件与哈希。

## 设计与实现的关键约束

1. 区分host构造指令图与CCU运行期。动态Variable使用目标支持的DSL；API存在/编译成功不证明生成器可翻译。按 [算术能力](references/api-and-synchronization.md#生成器算术能力) 映射实际rep，未知代际采用交集。
2. 注册期CcuKernelArg与运行期uint64_t启动槽不同。host顺序、LoadArg、launch槽数、FastLaunch缓存/派生字段必须一致。
3. channel→peer顺序、die/resGroup、kernel和资源数量与消费一致。本地完成Event和远端Notify分别证明；按 [实际配额](references/api-and-synchronization.md#设计期同步契约检查) 检查接收表、去重边界、上界及复用，零片早退也证明活性。
4. HCCL通过自身HCOMM dlsym适配层调用已支持接口，不引入HCOMM私有头或legacy依赖。CCU源码按实际template/kernel CMake接线，不套AICPU device登记。
5. 新算法独立推导数据流/同步，不要求先找到同类蓝本。默认选路变化需用户要求；显式路径、支持范围与测试断言可达性按实际链路核对。

调用方已有分层产物时，用 [Dataflow Spec模板](assets/dataflow-spec-template.md) 写每template的技术证明，参数与同步/算术值集中在JSON；报告引用并记录自身核验与差异。独立完整方案可用 [设计模板](assets/design-template.md)，不为一次任务重复生成两份完整设计。

## 静态检查与交付

设计冻结前、实现后按真实声明运行统一入口：

```bash
python3 "$SKILL_DIR/scripts/check_contracts.py" \
  --launch /path/to/task/launch-contract.json --sync /path/to/task/sync-contract.json \
  --hcomm-root "$HCOMM_ROOT"
```

此入口执行launch/sync/arith三个原检查器，返回各退出码、契约哈希及原始输出；任一失败整体失败，其他项仍运行。独立检查器继续可用。声明consistent不证明C++映射完整、同步活性、部署身份或功能正确，必须对应实际源码；独立评审仍自行验证关键结论。

验证前确认库身份、工具可表达域、断言可观测性和内存预算；六个标准算子按`--op`用 [输入预检](references/test-input-preflight.md) 直接计算实际count/bytes与等效用例，通信域编号不当rank数。功能通过须有实际目标算法命中与Checker证据，性能须有同环境对照；VM功能不证明真机性能。缺环境时说明未验证范围，不用静态/dry-run补PASS。交付列源码身份、变更、设计、真实命令/结果、证据与剩余缺口。

调用方要求功能验证后的 estimate 时，按 [验证参考](references/validation.md#4-性能估算与实测) 复用已安装 HVM Skill 的独立估算流程；绑定候选包、两仓身份、CCU 模式和本次算法命中，分别报告估算与功能证据。缺失或失败如实返回调用方，不用真机实测要求代替已要求的仿真估算。
