# 迁移专属故障模式

> 按"现象 → 根因 → 处置"组织。阶段 3/4/5 出错时先查这里，再查 `ascendc-runtime-debug` / `ascendc-precision-debug` / `ascendc-crash-debug`。

## 1. 编译期

| 现象 | 根因 | 处置 |
|---|---|---|
| `__VEC_SCOPE__`/`RegTensor`/`MicroAPI` 未定义 | 残留 RegBase 代码进入 arch22 编译 | 按 `api-migration-recipes.md` §2 重写，不是加 include |
| `FixpipeParamsArch3510` 未声明 | A5 Fixpipe 结构体残留 | 改 `FixpipeParamsV220`，字段逐项重算 |
| `CrossCoreSetFlag<4,...>` 模板失败 | 模式 4 仅 A5 | 改模式 2 或普通事件 |
| 大量 `undefined reference` 指向 arch35 | CMake/`SUPPORT_TILING_DIR`/include 路径残留 | 全仓 `grep arch35`，改目录与 CMake |
| `DT_FLOAT8_*`/`hifloat8` 未定义 | A2 ge 枚举/头文件无该 dtype | 按 dtype 决策去掉或替换；同步改 kernel 模板 |
| 编译通过但运行"算子未注册" | `AddConfig` 字符串 ≠ `--soc`；config 目录缺失 | 三处对齐：`--soc`、`AddConfig`、`config/<soc>` |
| 模板实例化爆炸导致 OOM | A5 的 dtype 组合仍全开 | 先按 A2 支持的 dtype 收敛列表再编 |

## 2. 运行期

| 现象 | 根因 | 处置 |
|---|---|---|
| `UB out-of-range`（507035 / error 340） | tile 仍按 A5 预算；40KB SIMT 预留未删；buffer 峰值算错 | `a2_ub_budget.py` 重算；删 40KB 预留；按峰值申请 |
| L0C 溢出 | `baseM×baseN` 按 A5 的 256KB L0C 设计 | 减半 base 块或分段 Fixpipe（A2 L0C 128KB） |
| 结果整体错位 / 数值乱 | L0A 分形 NZ→ZZ 未重算；`LoadData` 参数照抄 | 用 16×16 最小矩阵打通，逐参数核对 |
| 结果部分行/块错 | L0C→UB 直通改为 GM 中转后，workspace 大小或偏移没改 | 检查 `GetWorkspaceSizes`、每 tile 偏移、`DataCopy` 长度 |
| 死锁 / 卡死 | BufferID 隐式同步被删但未补 `SetFlag/WaitFlag`；flagId 冲突 | 逐点补事件配对；统一 flagId 分配表 |
| 偶发错误（重复跑不一致） | 竞争、workspace 复用未同步、跨核顺序假设 | 重复执行定位；加必要同步；禁用不确定的并行假设 |
| 精度大范围偏差 | Cast 舍入/饱和映射错；尾块越界；索引错误 | 最小 shape 二分；打印中间张量比对 |
| 仅在 subnormal 域偏差 | A5 FTZ 规避（eps/algo）未删干净，或删了但用例没覆盖 | 检查残留 `PRECISION_*`/eps；补 subnormal 用例 |
| 特定 dtype 输出全 0 | dtype 迁移时转换缺失（如 fp8 当 bf16 读） | 补显式转换；禁止 reinterpret |
| 改了代码但行为完全不变 / 特性开关无效 | 公共 torch API 回退到内置算子，custom 包未被调用 | 改用 ACLNN 直调验证（`verification-harness.md`）；看日志里是否有 vendor 报错文案 |
| 文档/静态扫描说“A2 不支持”，实际直调能跑通 | 文档版本差异、目录名误导（`arch35`/`arc35`）、扫描只看了目录 | 先构建 + 直调验证缺口（`stage_1` Step 1.0）；能跑就改做性能/边界任务，不要按迁移做 |
| 直调报 dtype 不符（如 `DT_INT32 should be DT_UINT64`） | ACL dtype 枚举写错（INT64=9、UINT64=10、BF16=27） | 按 `verification-harness.md` 的枚举表核对，别按常见枚举猜 |
| 结果整体错乱但不报错 | 辅助张量 dtype 与文档不符（如 `blockSparseMask` 应为 INT8 却传 INT32） | 按文档逐项核对 dtype/shape；用最小 case 对齐 CPU 参考 |
| 仓库 pytest 跑不起来（缺 pytest/pandas/openpyxl/tensorflow） | 离线环境依赖不全 | 同版本拷贝 site-packages；仅 dtype 映射用途可用 numpy stub；仍不行则直调 ACLNN |
| 单算子构建超过 1 小时 | 大算子 kernel 模板组合多；每次删 build 重编 | 保留 `build/` 增量；只在切 soc/arch 时清理；文档改动不触发重编 |
| 性能远低于预期（一倍以上） | 多绕一次 GM（L0C→GM→UB）；未开双缓冲；tile 过小 | 按 `performance-tuning-a2.md` 顺序排查 |

## 3. 流程性坑（最常导致返工）

| 坑 | 后果 | 正确做法 |
|---|---|---|
| 把 `arch35/` 改名成 `arch22/` | 目录名说谎，后续维护者误判 | 内容重写后再建目录 |
| 把 SIMT 函数体注释掉留空 | 功能静默缺失，精度测试可能"通过"（输出未写） | 明确 SIMD/AICPU/不支持三选一 |
| 保留 `#if __NPU_ARCH__ == 3510` 死分支 | 死代码腐化，误导后续读者 | 删除；确需跨代则改成两端都可编 |
| 直接复制 `config/ascend950/` 到 `ascend910b/` | 字段（dtype/format/key）与 A2 不匹配 | 从 A2 现有算子复制再改 |
| 用 A5 耗时作为 A2 优化目标 | 目标本身错误 | 只用 A2 实测与 A2 基线 |
| 只看编译通过就进入性能 | 精度未过的性能数据无意义 | 先 GATE-4 |
| 为"让测试过"放宽阈值/删用例 | 交付即埋雷 | 根因分析；条件接受需用户签字 |
| 忽略尾核/尾块 | 大数据量下偶发越界或结果错 | 专门设计非整除 shape 用例 |
| 把 workspace 当 UB 用 | 性能幻觉（实际走 GM） | 区分 UB/workspace，按 `TPosition` 确认 |
| 同一轮改多处再测 | 无法归因 | 一次一改，A/B 记录 |
| 反复 `rm -rf build` / 每改一行就重建 | 构建次数翻倍，耗时明显超过裸 agent | 批量改完再构建；S 类 ≤3 次、M 类 ≤6 次；只有切换 arch 或缓存损坏才清理 |
| 简单算子直接复用 A5 通用 kernel 类 | 功能对但核时比专用实现慢 20%–30% | 简单搬运/scatter 类优先写专用 arch22 kernel；用 msprof 核时对比后再定 |
| 改共享 tiling/头文件触发另一 soc 全量重编 | 日志出现大量 ascend950 的 `opc.py`，会话被拖到超时 | 改动限制在目标 soc 专属文件；或接受代价但保留 A2 包先评测 |
| 新建 arch22 文件但只 include 通用类 | 形式上有专用 kernel，核时与通用路径相同甚至更慢 | 专用文件内必须自己写搬运/计算主循环；验收以核时为准 |
| 小拷贝用 TQue 双缓冲/批量预取 | 每 token 固定开销大，2KB 拷贝反而比单 TBuf 慢 30%+ | 小 token 用单 TBuf + DataCopyPad + PipeBarrier |
| 用户磁盘配额满 | 编译报 `Disk quota exceeded`/`No space left`，被误判为代码问题 | 构建前 `quota -s`，清理旧 build 或 TMPDIR 指到 /tmp |
| 简单算子复用大 tiling 结构/入口 | 同一算法实测 11.0 µs vs 专用小结构 8.9–9.7 µs | A2 定义小的专用 tiling 结构 + 独立 kernel 入口，只读少量字段 |
| 用 `sleep 90–180` 轮询后台构建 | 单次工具调用就是 1–3 min，累计可占会话 80% 时间 | 前台构建 + timeout；或后台构建后先做别的，用 pgrep 查一次 |
| 全盘 `find /` / 全盘 grep | 单条命令可达 2 min，且输出大 | 限定仓库/安装目录并加 -maxdepth、| head |
| TQue 深度不足 + 长生命周期 AllocTensor | kernel 启动后卡在同步（无报错、不返回），首个 AllocTensor 之后死锁 | 常驻数据（query/参数）用独立 TBuf；TQue 只放逐行逐块数据，确保 depth ≥ 同时存活 tensor 数 |
| 手工在 gen 目录调 opc.py/bisheng | `cstdint not found` / `libplatform.so` 找不到 | 统一用 build.sh；先 source set_env、带 CPLUS_INCLUDE_PATH/LD_LIBRARY_PATH |
| 一次构建编多个算子 | 产物/配额翻倍，放大 OOM 概率 | 只编目标算子（--ops=<op>） |
| 构建被 kill（Error 137）后反复 clean 重试 | 每次仍 OOM，浪费时间 | 先降 -j、错开其他构建会话，再重试 |
| 缺 `op_tiling/arch22` 注册文件 | `--soc=ascend910b` 构建成功但包内只有 ascend950，运行报 JSON config 找不到 | 补齐 arch22 tiling 注册（DAV_2201）+ AddConfig + 最后用 910b 打包 |
| 最后一步用 `--soc=ascend950` 打包 | build_out 被 950 覆盖，A2 二进制丢失 | 打包顺序固定：910b 必须最后 |
| 设备代码调用 `expf/sqrtf` 等 libm 标量函数 | 编译报未声明，或链接报 `undefined symbol: expf` | 改用 AscendC::Exp/Sqrt/Log 单元素向量指令 |
| 工具链坏了还反复重跑 build.sh | 每次同错，浪费时间 | 构建前先编译空文件自检 cc/c++，先修环境 |
| 把一个 LocalTensor 切成多个 8 元素标量小平面做 SetValue/GetValue | 数据通路对、标量加权通路（h）全错，难定位 | 每个标量独立 TBuf，平面 ≥64 float，按 S_V/V_S 顺序 |
| 稀疏注意力只验 `attn_out` | PA_BSND/fp16/mode3 的 attn_out 与 TND/bf16/lse 的 softmax_sum 可能单独不达标 | 对 attn_out、softmax_max、softmax_sum 分别统计 matched_ratio |
| 反向算子以 `getws/run=0` 当精度通过 | 反向梯度输出可能全错但接口正常返回 | 必须跑仓库 pytest 或梯度 golden 比对（SFAG 实测） |
| 反向算子梯度全错但接口正常（SFAG 实测 dq matched 0.009、d_sinks 0.25） | 仅改 host 门禁/白名单不会修复数值；sinks/dSinks 归约路径与参考实现不一致 | 对照同族 arch22 反向实现（如 `sparse_flash_mla_grad`）逐输出比对 dq/dk/dv/dq_rope/dk_rope/d_sinks |
