# Broadcast Ring 复盘回归案例

来源：用户提供的 2026-09-22 开发—验证—检视报告。以下伪代码是行为评测输入，
不定义 HCCL API。运行人工评测时，只向评审者提供请求、输入和相关 skill；将预期留给评测人评分。
评测不运行伪代码，不据此声称已完成真实 HCCL 功能验证。

## 自动化部分

`python3 scripts/test_review.py` 覆盖轮转环正反例、Broadcast/未知形态转人工核对、
构件语义优先于所在算子目录，以及围栏解析和 S03 真不一致检出。已接入 `selftest.sh`。
开发侧 `test_checks.py` 覆盖同类围栏解析；formal-new 的 `tests/test_deliverables.py`
覆盖 ignored Spec、文件缺失、哈希变化和只读检查。这些测试不证明下面的控制流语义。

## 人工语义评测输入

请求统一为：“评审以下算法及其支持性约定，列出有证据的缺陷与尚未验证的性质。”

### T1 · 支持性时序

```text
Spec: 不支持 symmetric memory
入口: param.symm=false; Selector(param); param.symm=QuerySymm(); Execute(param)
注册: supportSymm=false
Template: 不检查 param.symm，直接通信
```

预期：指出选路消费早于真实赋值，显式命中后缺少有效阻断；引用时序与后果。
对照变体：赋值移到 Selector 前且所有入口走同一有效过滤时，不要求无条件再加模板守卫。

### T2 · 模式枚举与资源

```text
enum Mode { OPBASE, OFFLOAD, ACLGRAPH }
scratch = mode == OPBASE ? 1 : 0
remoteAccess = mode == OFFLOAD
执行: remoteAccess ? 使用远端 output : 使用 scratch
入口: 三态均可达
```

预期：ACLGRAPH 申请零 scratch 却使用 scratch。对照变体改为只在 OFFLOAD 返回零后不再报此缺陷；
不将该表达式推广到其他资源模型。

### T3 · 链式广播端点与依赖

```text
N=4，环序=[2,0,3,1]，root=3
按环序推导 prev/next
root: 从 input 发给 next
其他 rank: 等前驱接收完成，再向 next 转发；仅 next==root 时停止发送
```

预期：路径 3→1→2→0，共三跳；不能因没有本地 step 循环或轮转 chunk 公式报 R02/R03。
分别注入三个变体：中间 rank=2 提前停止发送；末端继续发回 root；中间 rank 未等接收完成就转发。
预期分别指出断链、错误终点导致未配对通信风险、未完成数据被使用；读写 wrapper 的实际后果仍须真仓核验。

### T4 · 同步作用域

```text
Spec: repeat { preSync; transfer; postSync }
代码: preSync; repeat { transfer }; postSync
```

预期：静态计数相同仍应指出契约结构不一致；根据缓冲复用和线程依赖判断后果，不能直接断定必然死锁。
对照变体代码与 Spec 均在 repeat 外时不报位置不一致。

### T5 · 交付完整性

```text
.gitignore: *.md
交付计划: kernel.cc + spec.md
Git index: 只有 kernel.cc
工作目录: 两文件都存在
```

预期：发现 Spec 未纳入 Git 交付；存在文件不等于交付完整。说明可由作者显式纳入版本控制，
或依据约定改为单独归档；不自动暂存，不把日志等 artifact 一律要求提交。

## 评分与维护

每例分别记录：命中预期根因、证据正确、未误报对照变体、明确自动/人工覆盖边界。
记录实际耗时与未完成项，不预填“通过”。新增脚本规则前先确认适用条件，再补正反例；
脚本未实现控制流分析时，不把人工案例描述为自动化检出能力。
