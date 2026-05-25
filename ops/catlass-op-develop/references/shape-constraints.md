# Catlass Kernel 测试 Shape 运行期约束

> 这是 **catlass kernel 代码本身**对运行期 shape 的硬约束（避免 AIV UB 越界）。具体怎么把约束写进 TEST.md / 测试用例 / golden 生成由测试 agent 决定，本 skill 仅提示约束本身。

---

## Δ1：固定 COMPUTE_LENGTH 的 Tile Epilogue 与小 M / N 不兼容

Catlass Matmul + 固定 `COMPUTE_LENGTH` 的 Tile Epilogue 时：

- **不宜**用过小的 M、N（如个位数）——容易与尾块 / 向量长组合触发 AIV UB 越界
- **宜选** L1 分块 M/N 的整数倍（常见 M 取 128 倍数、N 取 256 倍数，以 catlass `GemmShape` 为准）

向调用方提的需求：测试用例至少有一组 shape 满足 L1 分块整数倍。

---

## Δ2：dtype 覆盖

catlass 模板按 dtype 实例化（fp16 / bf16 / fp32 / int8）。**每个**设计阶段列出的合法分支至少一组测试用例，确保 catlass kernel 的每个实例化路径都过编译与精度。

向调用方提的需求：测试矩阵覆盖到设计阶段产出的"合法组合"全集。

---

## Δ3：精度阈值

与通用 fp16/bf16 GEMM 标准一致，无 catlass 专属放宽规则。详见 `ops-precision-standard` skill。
