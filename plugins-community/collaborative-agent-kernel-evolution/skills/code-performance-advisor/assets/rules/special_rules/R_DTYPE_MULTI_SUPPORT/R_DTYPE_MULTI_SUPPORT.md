# 规则名称：多数据类型支持与模板化设计

## 1. 需求场景 (Requirement)
- **业务背景**：算子需要支持多种数据类型（FP16/BF16/FP32）以适配不同精度的模型训练和推理场景，通过低精度数据类型减少内存带宽压力，同时保持计算精度。
- **形状/数据类型上下文**：支持 `T.FP16`, `T.BF16`, `T.FP32` 等多种输入输出数据类型。

## 2. 模式描述 (Pattern)
- **优化思路**：通过 C++ 模板类结合 Tiling Key 机制实现编译期数据类型分发，每种数据类型编译生成最优代码路径，避免运行时类型判断开销。
- **目标**：支持低精度类型获得 2 倍内存带宽提升，编译期确定类型实现零运行时开销。

## 3. 性能损耗因果链 (Inference / Physics)
- **因果说明**：低精度数据类型（FP16/BF16）相比 FP32 占用内存减半，理论上可获得 2 倍内存带宽提升。
- **事实桥接**：
  - 模板化设计 -> 编译期类型特化 -> 零运行时分支开销
  - Tiling Key 分发 -> 精确匹配类型组合 -> 最优执行路径
  - 低精度存储 -> 减少 GM 带宽占用 -> 降低 MTE 瓶颈

## 4. 触发信号 (Triggers — 与原始 profiling 表头对应)
- **需要查看的字段（以 `op_summary` 原始表头为准）**：
  - `Data Type`（数据类型）
  - `aic_mte2_ratio` / `aiv_mte2_ratio`（搬运单元占比，低精度可降低）
  - `Task Duration(us)`（整体耗时）
- **如何解读（定性）**：
  - 如果算子仅支持 FP32，但模型使用 FP16/BF16 训练，需要额外的类型转换开销
  - `aic_mte2_ratio` 较高表明内存带宽受限，低精度可缓解此问题
  - 使用标签 `T.*` 标注当前支持的数据类型范围

## 5. 动作实现 (Action)
- **参考代码位置**：`code_snippets/case1/` (adaptive_avg_pool3d), `case2/` (batch_norm_v3), `case3/` (apply_adam_w_v2)
- **实施步骤（示例性）**：
  1. 将算子实现类改为模板类：`template <typename T> class KernelOp`
  2. 在 OpInfo 文件中声明支持的数据类型组合
  3. 在 Tiling 阶段根据数据类型设置 Tiling Key（如：FP16=1, FP32=2, BF16=3）
  4. 在 Kernel 入口使用宏分发到不同模板实例：`if (TILING_KEY_IS(1)) { KernelOp<half> op; }`
  5. 针对不同数据类型选择合适的 RoundMode（FP16 用 CAST_NONE, BF16 用 CAST_RINT/CAST_ROUND）

## 6. 约束与副作用 (Constraints)
- **编译开销**：模板实例化增加编译时间和二进制大小，每种类型组合生成一份代码
- **适用场景**：`S.TransferDominated`, `S.MteBusy`（内存带宽受限场景收益明显）
- **不适用场景**：计算密集型算子（`S.LowCubeUtil`），低精度收益有限

## 7. 验证逻辑 (Verification)
- **验证原则**：低精度类型的内存带宽占用下降，整体性能提升
- **推荐验证项**：
  - `Task Duration(us)`：FP16/BF16 相比 FP32 期望下降 20-50%（取决于内存瓶颈程度）
  - `aic_mte2_ratio` / `aiv_mte2_ratio`：期望呈下降趋势（搬运占比减弱）
  - `HBM Bandwidth Utilization`：期望提升（单位时间传输更多有效数据）
- **验证方法**：对比同一 Shape 下 FP32 vs FP16/BF16 的性能数据，确认精度损失在可接受范围内

## 标签
- Domain: `U.Vector`, `U.Mix`, `U.Cube`
- Symptom: `S.TransferDominated`, `S.MteBusy`, `S.MemoryBound`
- Context: `T.FP16`, `T.BF16`, `T.FP32`
