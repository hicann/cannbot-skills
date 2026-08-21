# Optimization 入库标准

适用于 `knowledge/optimization/`；先执行[公共入库标准](common-entry-standards.md)。该目录记录可跨
算子复用的候选，不记录算子全貌或战绩。

## 强制内容

每个候选必须包含：

1. **适用范围**：operator family、SoC、dtype/layout、shape/tile 和瓶颈前提。
2. **可证伪假设**：要减少的计算、字节、等待或串行阶段，以及预期 metric。
3. **变换步骤**：可执行代码/伪代码、任务/地址/slot 变化和单轴实验顺序。
4. **保持条件**：算法、tail/mask、layout、所有权、同步和累加/写回精度。
5. **资源代价**：片上容量、workspace、寄存器、block、重复计算和编译风险。
6. **失败与回退**：编译、容量、正确性、噪声或负载失衡的停止条件。
7. **验证合同**：完整正确性、同配置 benchmark/profile、噪声门槛和 fresh best 复测。

## 拒绝条件

- 从 fast 名称、单个 shape、单次延迟、ratio 或截图推断收益。
- candidate 改变输出、精度、layout、mask、launch 或调用侧工作。
- 把 lowering/runtime 故障当成优化经验，或把单算子结果外推到整个 family。

## 验收清单

- [ ] 前提、假设、步骤、代价、metric 和回退能组成一次单轴实验。
- [ ] 源码启发与项目实测分开，实测结论有直接 evidence。
