# apace 路线代码审查验收条件

> Reviewer 在 Step 4 逐项检查。违反任意红线项 = FAIL。
> 四大约束的详细论述见 [`architecture.md`](architecture.md) §10。

## 红线项

| # | 检查项 | 验收条件 |
|---|--------|---------|
| R1 | 无 `__schedmode__` | 代码中不含 `schedmode` 或 `core_ratio` |
| R2 | 有 `KERNEL_TYPE_MIX_AIC_1_1` | 每个入口函数都含核配比声明 |
| R3 | 入口变体与参考算子一致 | PUT=4 dtype 变体入口于 `kernel_launcher.h`；AG=单入口于 impl.h |
| R4 | `block/` `tiling/` 未修改 | 与官网仓原始文件完全一致 |
| R5 | CrossCore flag idx 配对 | AIV `WaitFlag` idx == AIC `SetFlag` idx |
| R6 | CommContext 与引擎匹配 | UDMA 模式有 `__gm__ CommContext*`；HCCL windows 无 |
| R7 | 无 asc-devkit matmul | 代码中不含 `AscendC::Matmul` |
| R8 | 无 HCCL 高阶 API | 代码中不含 `Hccl::` |

## 常见 FAIL 原因

| 现象 | 根因 | 修复方向 |
|:---|:---|:---|
| 代码中含 `__schedmode__(1)` | 误加调度属性 | 删除，核配比由 `KERNEL_TYPE_MIX_AIC_1_1` 保证 |
| 代码中含 `Hccl::AllReduce` | 误用 HCCL 高阶 API | 改用 `CollectiveComm` 四段式 API |
| 代码中含 `AscendC::Matmul` | 误用 asc-devkit 接口 | 替换为 `Blaze::Gemm::Block::BlockMmad` |
| `block/` 或 `tiling/` 有改动 | 误改共享层 | 恢复共享层文件，只在 `kernel/<op>/` 下改 |
| 精度对不上但无报错 | flag idx 不配对 / splitKNum 配置错 | 核对 flag 编排和 splitKNum 规则 |
| 死锁（aclError:507015） | schedmode 或 flag 不配对 | 检查无 schedmode；检查 CrossCore flag idx 配对 |
| DESIGN.md 与代码不一致 | localMatmul 等参数变更后未同步文档 | 同步 DESIGN.md |
