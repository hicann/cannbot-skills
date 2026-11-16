# 接入故障诊断

按症状定位，只读取相关接口或产物章节。先保留最小复现、环境报告和原始错误。

## 目录

- [症状入口](#症状入口)
- [加载和注册](#加载和注册)
- [编译或声明失败](#编译或声明失败)
- [数值与重放](#数值与重放)
- [日志取证](#日志取证)

## 症状入口

| 症状 | 首先核对 | 深入读取 |
|---|---|---|
| .so 未加载 | OPP 根、os/arch、依赖库、ABI、路径有效性 | build-verify.md |
| 已加载但回调未触发 | 原型与 type、注册宏、版本门控、优先级、选中能力 | interfaces.md |
| Python 注册失败 | 类型标注、attrs 签名、零参构造、Portable 成对 | python-track.md |
| binary 加载失败 | host/device 格式、真实入口、magic、SoC | language-launch.md |
| 声明式编译失败 | bin/size、stream_id、args 返回值、图形态/端侧限制 | interfaces.md |
| 首轮正确后续错误 | 静态 Execute 录制、刷新、辅流 event、缓存复用 | scenarios.md |
| 编译机正常、部署机失败 | 外部文件、Portable、Python 源码和运行依赖 | scenarios.md、python-track.md |
| OM 加载成功、执行失败 | 执行序列与 OM 形态是否匹配、config handle、V1/RT2 分派 | build-verify.md |

## 加载和注册

1. 列出实际 ASCEND_CUSTOM_OPP_PATH 条目，核对目录或 `.py` 是否存在。
2. 检查根 libcust_opapi.so 或 op_graph/lib/<os>/<arch> 路线；不要只看文件名。
3. Linux 用 readelf/nm/ldd 检查依赖和符号，保留真实 dlopen 错误。
4. 核对 REG_OP 的 TU 是否参与编译、实现 op type 是否对应构图节点。
5. 新宏与旧库混用时检查 GetGeVersion 门控；kBottom+kCustom 不能作为普通默认注册。

当前注册导出可查 `GetRegisteredCustomOpCreator*`；不要只查 static 局部变量 custom_op_register，它不要求出现在动态符号表。导出 ABI 的签名与数量核对目标 custom_op.h，勿用错误 ctypes 返回类型探测。
离线 OM 加载报 `[CUSTOM OP] pull creator ABI symbols are incomplete.` 时，先确认交付件链接了静态库 `libcustom_op_registry_static.a`；该项属预防性验收，见 [OPP 与构建](build-verify.md#opp-与构建)。

ArgsUpdater 优先于 AnnotatedArgsOp 是正常策略，不是“声明式偶发失效”；Python capability 的 INFO 未实现提示也不自动等于当前所需能力缺失。

## 编译或声明失败

- 两参 AddLaunch 的返回值是状态码，不能按 token 的 UINT32_MAX 检查。
- stream_id 必须是当前主流或合法申请的辅流，不能留默认 0 假定节点总在流 0。
- C++ bin 长度为 0、AppendArg 失败、未知 shape 被算成 block_dim 都应在编译阶段定位。
- device ELF 检查只能用于该格式；共享库或第三方封装需用对应 loader 核对。
- Python 模板编译器/include/arch 参数是目标工具链相关信息，先检查环境，不不断重试错误参数。
- MIX/FFTS 只看到源码注解或某个默认 magic 不足以决定整条路径可用性。

## 数值与重放

从一个固定、已支持的 dtype/shape 开始，比较全部输出，随后再测边界和并发：

- binary/key：两个同shape不同dtype/attrs节点，两个不同shape，源码修改后的重编译。
- ABI：参数顺序/宽度、隐含 grid/FFTS、workspace、真实入口。
- 尾块/空张量：按真实 kernel 契约支持或明确拒绝，不能只 ceil block_dim。
- 静态执行：验证 Execute 是否发生在分发阶段，不能以回调日志推定每轮 Host 分支执行。
- 换址：交替两套输入多轮，检查刷新或内存中转；首轮正确不代表后续安全。
- 辅流：主→辅→主依赖、event reset 与生命周期、当前模型流句柄。
- shape：unknown维不等于上界；按输出实际shape裁剪并检查元信息回填。

源码与实测依据分开记录。不能为了得到“预期机制”而修改已经正确的输出逻辑，需要性能优化时先用图和日志证明实际走到的分支。

## 日志取证

仅在需要时启用，避免无条件收集全部图快照：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=1
```

按日志文本检索，日志编译进的行号可能与本地源码不同：

| 证明目标 | 检索线索 |
|---|---|
| 路径/加载 | value of env ASCEND_CUSTOM_OPP_PATH、Find ... so files、begin dlopen |
| 注册 | register custom operator creator for |
| 引擎选择 | Assigning DNNEngine ... to node（DEBUG 级） |
| 基本 task | generate basic custom kernel task |
| 刷新模式 | args_refresh_strategy、ArgsUpdater operator detected |
| 中转节点 | MemConflict、INSERT、Identity、MemcpyAsync |
| RT2 每轮执行 | ExecuteCustomOpImpl、EagerArgsHandler（DEBUG 级） |

根目录 libcust_opapi.so 未命中后子目录加载成功时，前一条 realpath WARNING 可是正常探测。
声明式分支可从生成图/task和具体函数调用确认，不能把函数名本身假定为一定打印的日志。

`ACL_LOG_*` 宏在行首自动打印函数名，文案里通常又写一遍 API 名，例如 `aclmdlExecuteV2Impl: start to execute aclmdlExecuteV2`，因此 `grep -o <api> | wc -l` 会把调用数算成两倍；计数用 `grep -c "start to execute <api>"`，并按时间戳核对轮次。
V1 分发与 dlopen 是 INFO 级（dlopen 只打 so 路径、不含算子名）；RT2 执行侧细节是 DEBUG 级。
