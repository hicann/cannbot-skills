# ascendc-algorithm-check

**Tagline**: AscendC 算法完整性与数学不变量检查

**Triggers**:
- 当 `cake-code-review` 主 skill 进入 Phase 3（Algorithm Correctness Check）时自动调用
- User 明确说"检查算法是否写完整"、"检查数学不变量"时独立使用

---

## 1. Introduction

`ascendc-algorithm-check` 负责检查 AscendC 算子实现是否覆盖了算法级必需步骤，而不是只看 API 使用或结构安全。

**本阶段负责**:
- 归一化链是否完整
- 分母安全和关键数学不变量是否建立并被消费
- 迭代算法的状态更新是否完整
- 归约结果是否被后续逻辑一致消费

**本阶段不负责**:
- API 黑名单和最佳实践 dispatch（交给 API precheck）
- 红线/安全/结构问题扫描（交给 `ascendc-review.md`）
- 编译、UT、ST 执行（交给 `ascendc-verify.md`）
- 更深数值参考对比、精度分析、性能评估（交给 `ascendc-evaluation`）

---

## 2. Check Categories

### 2.1 Normalization-chain completeness

检查 `max/sub -> exp -> reduce-sum -> divide/scale` 或同类归一化链是否缺步骤、跳步骤、错消费。

**常见信号**:
- 已计算稳定化偏移，但后续未做归一化
- 已做 `ReduceSum`，但结果未参与 divide/scale
- 只做了局部 tile 归一化，未完成全局合并

**示例**:
- Softmax: `ReduceMax` 和 `Exp` 存在，但 `ReduceSum` 结果没有被最终输出消费

### 2.2 Denominator safety and invariants

检查分母是否来自已建立的不变量，以及这些不变量是否在代码或文档中闭环。

**常见信号**:
- 分母来自归约结果，但看不到 `> 0`、finite、non-empty 等前提建立点
- 分母 guard 和算法前提相互矛盾
- 仅在 verify 阶段期待测试发现分母问题

**示例**:
- Softmax/SinkhornKnopp 中的 sum/norm 必须来自前序有效归约，而不是默认假设永远合法

### 2.3 Iterative update completeness

检查迭代算法是否遗漏状态刷新、双边更新、收敛变量维护、buffer 切换、残差回写。

**常见信号**:
- 循环里只更新一侧状态，另一侧更新缺失
- 迭代次数变量存在，但核心 update 没有落地
- 新旧状态比较、收敛判断、swap/copy 回写不完整

**示例**:
- SinkhornKnopp: row/col scaling 其中一条更新链缺失，导致迭代只执行半步

### 2.4 Reduction result consumption consistency

检查归约结果在所有使用路径上是否被一致读取、广播、回写和复用。

**常见信号**:
- `ReduceSum`/`ReduceMax` 写入临时 tensor 但后续未读取
- 不同 branch 对同一归约结果消费方式不同
- 只消费 tile 局部结果，遗漏跨 tile 合并结果

### 2.5 End-to-end algorithm closure

检查输入假设、核心计算、输出约束是否形成完整闭环。

**常见信号**:
- 中间量已算出，但最终输出少一步恢复或投影
- 文档声明支持某种数学约束，代码路径未体现
- 重要中间变量存在但与最终结果断链

---

## 3. Review Procedure

1. 识别算子属于单步归一化、迭代归一化、规约-广播、投影/校正中的哪一类
2. 画出最小算法链: 输入 -> 中间不变量 -> 归约/更新 -> 输出
3. 对照源码确认每个必要步骤都有实现和消费点
4. 记录缺失步骤、断链点、被破坏的不变量
5. 以 issue 列表输出给 `ascendc-fix.md`

---

## 4. Output Format

### JSON issue format

```json
{
  "id": "ALG-01",
  "rule": "Algorithm Correctness Check - normalization-chain completeness",
  "severity": "P1",
  "location": "op_kernel/example.h:210-248",
  "code_snippet": "ReduceSum(sumBuf, expBuf, tmp, nt);",
  "description": "已生成归一化分母，但结果未被最终 divide/scale 消费，算法链在输出前断开",
  "fix_suggestion": "补齐 reduce result 的读取与 divide/scale 消费路径，保持归一化链闭环"
}
```

### Markdown report fragment

```markdown
### ALG-01: 归一化链不完整

**严重性**: P1 - 严重
**违反规范**: Algorithm Correctness Check - normalization-chain completeness
**代码位置**: `op_kernel/example.h:210-248`

#### 问题分析
- 已完成稳定化和指数变换
- 已生成分母归约结果
- 最终输出没有消费该分母，归一化链断开

#### 修复建议
- 补齐 reduce result 的读取和 divide/scale
- 若分母依赖前序不变量，在生成点或消费点显式说明约束
```

---

## 5. Common Findings

1. **只做一半归一化链**
   - 例如只做到 `Exp` 或只做到 `ReduceSum`

2. **把分母安全问题推迟到 verify**
   - verify 只能证明编译/UT/ST 结果，不能替代算法推理

3. **迭代算法缺半步更新**
   - 常见于 row/col、left/right、old/new 双缓冲切换不完整

4. **归约结果产生后无人消费**
   - 临时 tensor 正确生成，但算法结果仍走旧路径

---

## 6. Known Limitations

1. 本阶段依赖源码和文档推理，无法替代完整数值评估
2. 若算法定义本身不清晰，需要结合算子文档或设计说明判断
3. 若需要 reference 数值比对或精度分析，转到 `ascendc-evaluation`
