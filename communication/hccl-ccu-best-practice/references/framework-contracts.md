# AllGather Sole：框架实际调用契约

仅适用于当前 `InsV2AllGatherSoleExecutor` 的 CCU 分支。来源由 `all-gather-framework` 卡逐文件锚定；先校验卡，其他 executor 重新核对自己的调用方。下表不是所有 CCU 算法的统一接口。

| 实现点 | 调用方与实际约束 | 设计/实现时直接回答 |
|---|---|---|
| 带参构造 | executor `CalcRes` 用 `(param, topoInfo->userRank, algHierarchyInfo.infos[0])` 构造 template | userRank 到局部 rank/channel 的映射；空层级的处置 |
| 默认构造 | executor `FastLaunch` 用 `make_unique<InsAlgTemplate>()`，随后只传 param 与缓存上下文 | 重放不能依赖带参构造留下的成员；从上下文恢复必要字段 |
| `CalcRes` | executor 直接调用 template；后续注册消费声明的 kernel/channels/die/resGroup | 申请与消费数量一致，注册对象生命周期有效 |
| `KernelRun` | executor 的 `OrchestrateLoop` 构造本轮参数并调用 | 字节/元素单位、已加基址偏移、整次 stride 与本轮 slice；详见接入卡 |
| `FastLaunch` | context 保存 threads、kernelSubmitInfos，另传本次 buffer | launch 前缀、缓存 metadata、地址派生字段与 token 必须对应本次 ABI |
| `GetRes` / `GetThreadNum` | 通用基类接口：GetRes默认报不支持，GetThreadNum返回0；当前Sole executor未直接调用二者，重放线程数取templateAlgRes.threads.size()（threads来自resCtx） | 按实际消费者决定是否覆盖；不能因继承GetThreadNum=0就断言Sole必然零线程。仍核对实际资源与submitInfo |
| `CalcScratchMultiple` | executor 据此计算 loop 上限，基类默认值为0 | 本算法是否用 scratch；零不代表没有 CcuBuffer 等资源 |
| `Describe` | 基类声明为纯虚方法 | 提供真实算法描述 |
| 静态 `CalcCostCoeff` | executor 调 `InsAlgTemplate::CalcCostCoeff`；基类默认返回空。`cost_model.cc` 的成本注册路径对空结果记录 uncalibrated 并跳过算法 | 目标路径消费成本模型时必须返回有依据的非空项；显式 HCCL_ALGO 不替代此条件。使用当前 helper 与真实任务模型，不抄固定系数 |
| `CreateChannelRequestByRankId` | `channel.cc` 函数入口执行 `channels.clear()` | 多 peer 分别收集后合并；固定消费顺序。是否去重由协议决定，不统一去重所有算法 |

源码定位（均相对 HCCL 根）：

- `src/ops/all_gather/algorithm/executor/ins_v2_all_gather_sole_executor.cc`：`CalcCostCoeff`、`CalcRes`、`OrchestrateLoop`、`FastLaunchSaveCtx`、`FastLaunch`。
- `src/ops/op_common/algorithm/template/ccu_alg_template_base.{h,cc}`：默认/纯虚方法及默认资源行为。
- `src/ops/op_common/selector/cost_model.cc`：调用 executor 成本方法与空结果剔除。
- `src/ops/op_common/algorithm/executor/channel/channel.cc`：建链输出容器语义。

设计只登记本次所用方法、实现选择和偏离；不把整表复制到 DESIGN/LOG。对应代码范式见 [短代码范式](implementation-patterns.md)。

基类的 `CalcRes`、`KernelRun`、`FastLaunch` 也默认返回 `HCCL_E_INTERNAL`；“不是纯虚函数”不能作为继承默认实现的依据。
