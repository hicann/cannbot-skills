# STATE.md 模板

<!-- 本模板必须与 SKILL.md 中的各阶段保持同步。 -->

将此文件作为 `docs/{OP}/STATE.md` 的起始模板。

```markdown
# {OP} Ascend C 实现

来源：{SOURCE_DESCRIPTION}
来源类型：{SOURCE_TYPE}
创建时间：{DATE}
目标芯片：SocVersion={SOC_VERSION} / NpuArch={NPU_ARCH}（阶段 0 由 scripts/detect_soc.py 探测）

## 阶段 2：可编译骨架
- [ ] 创建单文件 `{OP}.asc`：device kernel + tiling + host 入口 + `TORCH_LIBRARY` 注册的骨架（kernel 留空，host 入口返回空输出张量）
- [ ] 创建 `CMakeLists.txt`（`find_package(ASC)` + torch_npu，产出 `libop_{OP}.so`）
- [ ] 创建 `test_{OP}.py`，包含接口存在性占位测试
- [ ] 验证 `cmake/make` 构建成功且 `pytest test_{OP}.py -v` 可运行
- [ ] 提交

## 阶段 3：算子定义文档
- [ ] 分析算子源码并提取数学语义
- [ ] 若提供了源代码：检查是否存在迭代累加模式——评估精度风险
- [ ] 若为公式/描述：与用户确认公式并补全缺失的 I/O 细节
- [ ] 编写 `docs/{OP}/{OP}_definition.md`
- [ ] 子 Agent 评审：数学正确性（判定：PASS/FAIL/CONCERN）
- [ ] 子 Agent 评审：语义与边界场景（判定：PASS/FAIL/CONCERN）
- [ ] 润色并提交

## 阶段 4：算子设计文档
- [ ] 编写 `docs/{OP}/{OP}_design.md`
- [ ] 记录任何 UB-to-UB 拷贝路径：字节数、`32B` 块对齐、按位拷贝要求、最小缓冲区大小、尾块处理
- [ ] 对每个块大小或传输计数常量，验证对所有支持的 dtype（fp16、fp32）满足 `count * sizeof(T) >= 32`
- [ ] 若输出 dtype 与计算 dtype 不同，依据 implementation-patterns.md § 类型转换（Cast）支持矩阵 设计 Cast 链
- [ ] 若面向 Ascend950 / `dav-3510`：记录 Reg 包装函数、掩码/尾块处理、32B 规约标量槽位、`CastTrait`/dist 模式以及禁用 API 的规避
- [ ] 子 Agent 评审：UB 预算与切分数学（判定：PASS/FAIL/CONCERN）
- [ ] 子 Agent 评审：指令序列与依赖顺序（判定：PASS/FAIL/CONCERN）
- [ ] 若面向 Ascend950 / `dav-3510`：子 Agent 评审 Reg API 合规性（判定：PASS/FAIL/CONCERN）
- [ ] 润色并提交

## 阶段 5：测试套件
- [ ] 在 `test_{OP}.py` 中定义 `SHAPES` 用例矩阵（最少：小/大/非对齐形状，边界场景取值）与 `DTYPES`（每个支持的 dtype）
- [ ] 编写接口存在性测试（断言算子注册到 `torch.ops.op_{OP}`）
- [ ] 检查 host 入口命名是否与 C/C++ 标准库符号冲突（使用 `namespace op_{OP}`）
- [ ] 编写以 `@pytest.mark.parametrize` 参数化、torch 作 CPU 参考的 NPU 测试（`@pytest.mark.skipif(not torch.npu.is_available())` 守卫）
- [ ] 验证 `pytest` 可运行；骨架应导致 NPU 测试失败
- [ ] 提交

## 阶段 6：核函数实现
- [ ] 实现 device kernel `{OP}_kernel<T>()`（`__global__ __aicore__`，按 tile 处理 UB 数据）
- [ ] 实现 tiling 函数 `calc_{OP}_tiling_params()`（返回 numBlocks/blockLength/tileSize）
- [ ] 若面向 Ascend950 / `dav-3510`：用 `__simd_vf__` + `AscendC::Reg` + `asc_vf_call` 包装函数实现向量计算/类型转换/规约
- [ ] 若面向 Ascend950 / `dav-3510` 且为该算子指定了 VF 融合上限 `N`：每个 `__simd_vf__` 函数融合 ≤ `N` 条 VF 计算指令（更长的链拆分为多个包装函数，通过各自独立的 `asc_vf_call` 串接）。在设计文档中记录所选的 `N`；若未指定上限则跳过。
- [ ] 若面向 Ascend950 / `dav-3510`：扫描禁用 API（`AscendC::MicroAPI`、Membase、除 `asc_vf_call` 外的裸 `asc_*`、经典 AscendC 计算/类型转换/规约）
- [ ] 若面向 Ascend950 / `dav-3510`：验证掩码为元素计数、尾块存储已加掩码、规约使用 32B 标量槽位，且 Reg 产生的标量不经由 `GetValue()` 取出
- [ ] 验证每个 `TBuf::Get<T>()` 调用都由与其对应 `pipe.InitBuffer()` 相同的 `if constexpr` 条件守护
- [ ] 验证 device 侧代码中使用的所有辅助函数在 device 侧编译上下文中均有效
- [ ] **构建前关卡：** 在 `__aicore__` 函数内对核函数 `.asc` 执行 grep，查找 `ceil_div`、`align_down`、`align_up` 以及任何其他仅限 host 的工具调用——编译前替换为内联算术
- [ ] 实现 `namespace op_{OP}` 中的 host 入口（按 dtype 分发并 launch `{OP}_kernel<dtype>`）
- [ ] 添加 `TORCH_LIBRARY(op_{OP}, m)` schema 与 `TORCH_LIBRARY_IMPL(op_{OP}, PrivateUse1, m)` 绑定
- [ ] 构建通过
- [ ] 所有测试在本地通过
- [ ] 提交

## 阶段 7：验证
- [ ] 验证核函数源码包含完整实现（而非阶段 2 的骨架）
- [ ] 在 NPU 上完整构建（`cmake/make`）并运行 `pytest test_{OP}.py -v`
- [ ] 检视构建/测试日志
- [ ] 全部用例通过
- [ ] 若构建产物疑似过期，删除 `build/` 后干净重建
- [ ] 若有需要则迭代

## 阶段 7.5：黑/白盒测试门禁
Harness 配置：`harness.test_gate = {on|off}`

如果 `harness.test_gate` 为 `off`：
- [ ] 记录按配置跳过测试门禁
- [ ] 提交跳过记录

如果 `harness.test_gate` 为 `on`：
- [ ] 按照 `ascendc-st-design` skill 执行黑盒用例生成与执行
- [ ] 按照 `ascendc-whitebox-design` skill 执行白盒用例生成与执行
- [ ] 在 `test-harness/` 下记录真实测试产物、结果和日志
- [ ] 运行 `validate_test_gate.py` 并确认 `STATUS: PASSED`
- [ ] 若有精度失败，回到阶段 6 修复后重新验证
- [ ] 提交

## 阶段 8：最终文档
- [ ] 更新 `docs/index.md`
- [ ] 更新 `AGENTS.md`
- [ ] 验证所有 STATE.md 复选框均已勾选
- [ ] 最终提交
```
