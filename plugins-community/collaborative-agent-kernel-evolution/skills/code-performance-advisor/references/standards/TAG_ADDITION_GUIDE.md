# Tag Addition Guide - 新标签添加工作流

> **设计原则**：Tag taxonomy 是 rule matching 的索引系统，每个新tag的引入都会影响匹配精度和维护成本。因此需要严格的准入与验证机制。

---

## Part 1: 准入判断（When to Add a New Tag）

### ✅ 应该添加新tag的场景

1. **新瓶颈模式发现**
   - 通过多个算子的profiling数据，发现了现有symptom tag无法表达的共性瓶颈
   - 例如：发现一类新的"AICORE与MTE竞争"模式，现有 `S.MteBusy` 不够精确

2. **新算子族涌现**
   - 一类新算子在多个项目中被反复使用，且优化模式独特
   - 例如：MoE (Mixture of Experts) 算子族，需要 `O.MoE` 来区分其特殊的负载均衡优化

3. **新硬件架构引入**
   - 新一代Ascend芯片带来新的架构特性
   - 例如：910D引入的新特性需要 `C.Arch.910D` 及配套的context tags

4. **语义补全需求**
   - 现有tag存在语义对称性缺失（如有 `Low` 无 `High`）
   - 且确实有多个实际算子需要该对称tag
   - 例如：已有 `S.LowVecUtil`，发现多个算子需要 `S.HighVecUtil`（如需限制并行度避免资源争抢）

### ❌ 不应该添加新tag的场景

1. **单一算子特有现象**
   - 只有一个算子出现，无法验证通用性
   - **解决方案**：在rule的描述中用自然语言说明，不新增tag

2. **现有tag可组合表达**
   - 如 `S.MemoryBound` + `C.Tile.Small` 已能精确匹配
   - 无需引入 `S.SmallTileMemoryBound`

3. **过度细分**
   - 如将 `S.LowComputeUtil` 细分为 `S.LowComputeUtil.Level1`, `S.LowComputeUtil.Level2` ...
   - **问题**：增加匹配复杂度，降低rule复用性

4. **命名不规范**
   - 如 `S.slow_vector`（应为CamelCase）
   - 如 `S.VectorIsSlow`（冗余is，应为 `S.SlowVector`）

---

## Part 2: 新Tag设计规范

### 2.1 命名规范（Naming Convention）

**格式**：`{Prefix}.{Descriptor}`

**Prefix规则**：
- `U.` - Unit（执行单元）
- `O.` - Operator（算子族）
- `T.` - Type（数据类型）
- `S.` - Symptom（症状/瓶颈）
- `C.` - Context（上下文/约束）

**Descriptor规则**：
- **CamelCase**，首字母大写
- **语义清晰**：优先使用领域通用术语（如 `MatMul` 而非 `MM`）
- **避免冗余**：不用 `is/has/with` 等（tag本身即断言）
- **保持简洁**：一般2-4个单词，最多不超过5个单词

**示例**：
```
✅ Good:
  S.HighCubeUtil          # 清晰、简洁
  O.SparseAttention       # 使用领域术语
  C.Layout.NHWC           # 层次化命名（对于复杂context）

❌ Bad:
  S.CubeUtilizationIsHigh  # 冗余is
  O.SA                     # 缩写不清晰
  S.HighCubeUtilAndLowVecUtil  # 过于复杂，应拆分为两个tag
```

### 2.2 语义一致性（Semantic Consistency）

**对称性原则**：
- 如果引入 `High`，应评估是否需要对称的 `Low`
- 如果引入 `Start`，应评估是否需要对称的 `End`

**层次性原则**：
- Context tag可以有层次：`C.Arch.910B`, `C.Arch.910B2`
- 但不超过3层，避免过度嵌套

**互斥性标注**：
- 同一维度的tag应在文档中标注互斥关系
- 例如：`U.Cube`, `U.Vector`, `U.Mix` 是互斥的（一个算子只能有一个）

---

## Part 3: 新Tag添加工作流（Workflow）

### Step 1: 提案与验证（Proposal & Validation）

**创建提案文档**：
```bash
# 在临时目录创建提案
mkdir -p references/standards/tag_proposals
cat > references/standards/tag_proposals/PROPOSAL_<tag_name>.md <<EOF
# Tag Proposal: <tag_name>

## 1. 动机（Motivation）
为什么需要这个tag？现有tag无法表达什么？

## 2. 使用场景（Use Cases）
列出至少3个实际算子案例，说明该tag的适用场景。

## 3. 定义（Definition）
精确定义该tag的触发条件（如profiling阈值、代码模式等）。

## 4. 影响分析（Impact Analysis）
- 是否与现有tag冲突？
- 是否需要同步添加对称tag？
- 预计影响多少条rule？

## 5. 命名合理性（Naming Rationale）
为什么选择这个名字？是否符合命名规范？

EOF
```

**验证checklist**：
- [ ] 至少3个实际算子案例
- [ ] 定义明确（可用profiling metrics或代码pattern描述）
- [ ] 命名符合规范（CamelCase，语义清晰）
- [ ] 不与现有tag冲突或重复
- [ ] 已考虑语义对称性

### Step 2: 更新Taxonomy（Update tag_taxonony.md）

**格式**：
```markdown
* `<TagName>`: <简短描述（一句话）>。
```

**示例**：
```markdown
### 2. Symptom 标签（瓶颈原语 / Matching）
* **计算瓶颈**
    * `S.LowComputeUtil`: 通用低利用率。
    * `S.LowCubeUtil`: Cube 低利用率。
    * `S.LowVecUtil`: Vector 低利用率。
+   * `S.HighVecUtil`: Vector 高利用率（需限制并行避免争抢）。
    * `S.ScalarBound`: 标量瓶颈。
```

**变更管理**：
```bash
# 在commit message中标注tag变更
git add references/standards/tag_taxonony.md
git commit -m "feat(taxonomy): Add S.HighVecUtil for over-utilization scenarios

- Use cases: 3 operators with vec_ratio > 95% show resource contention
- Symmetric to existing S.LowVecUtil
- Enables rules for parallelism throttling

Closes: #<issue_number>
"
```

### Step 3: 验证向后兼容性（Backward Compatibility Check）

**运行验证脚本**：
```bash
# 验证所有现有tag仍然有效
python3 scripts/analysis_engine/tag_validator.py

# 预期输出：
# ✅ Success: All 23 rule(s) have valid tags
```

**如果验证失败**：
1. 检查是否误删了旧tag
2. 检查拼写错误
3. 回滚并修复taxonomy

### Step 4: 创建示例Rule（Create Example Rule）

**创建至少一条使用新tag的rule**：
```bash
mkdir -p assets/rules/R_<NEW_RULE_NAME>
```

**Rule结构**：
```
R_<NEW_RULE_NAME>/
├── R_<NEW_RULE_NAME>.md              # Rule描述
└── R_<NEW_RULE_NAME>_tags.json      # Tag定义（必须包含新tag）
```

**tags.json示例**：
```json
{
  "rule_id": "R_<NEW_RULE_NAME>",
  "domain_tags": ["U.Vector", "O.Activation"],
  "symptom_tags": ["S.HighVecUtil"],  // 新tag
  "context_tags": ["C.Arch.910B"],
  "required_tags": ["S.HighVecUtil"]   // 新tag作为必需条件
}
```

### Step 5: 更新测试用例（Update Test Cases）

**添加单元测试**：
```python
# tests/test_tag_validator.py
def test_new_tag_validation(self):
    """Test validation of newly added tag."""
    # 验证新tag在taxonomy中
    self.assertIn("S.HighVecUtil", self.valid_tags)

    # 验证新tag可以正常匹配
    with tempfile.NamedTemporaryFile(mode='w', suffix='_tags.json', delete=False) as f:
        test_data = {
            "rule_id": "TEST_NEW_TAG",
            "symptom_tags": ["S.HighVecUtil"]
        }
        json.dump(test_data, f)
        temp_path = Path(f.name)

    try:
        is_valid, invalid_tags, _ = validate_tag_file(temp_path, self.valid_tags, verbose=False)
        self.assertTrue(is_valid)
        self.assertEqual(len(invalid_tags), 0)
    finally:
        temp_path.unlink()
```

**运行测试**：
```bash
# 单元测试
python3 tests/test_tag_validator.py

# 端到端测试
bash tests/e2e_test.sh fastgelu
```

### Step 6: 更新文档（Update Documentation）

**需要更新的文档**：
1. **CHANGELOG.md** - 记录tag变更
2. **SKILL.md** - 如果是重要tag，更新skill说明
3. **subskills/code_tag.md** - 更新tagging示例（如有必要）

**CHANGELOG示例**：
```markdown
## [Unreleased]

### Added
- **Tag Taxonomy**: Added `S.HighVecUtil` for vector over-utilization scenarios
  - Use case: Operators with vec_ratio > 95% showing resource contention
  - Symmetric to existing `S.LowVecUtil`
  - Enables rules: R_THROTTLE_VECTOR_PARALLELISM

### Changed
- Updated `tag_validator.py` to include new tag detection logic
```

---

## Part 4: Tag生命周期管理

### 4.1 Tag弃用流程（Deprecation）

**何时弃用**：
- Tag被更精确的新tag替代
- Tag使用率极低（<3条rule使用）且无通用性
- Tag语义不清导致误用

**弃用步骤**：
1. 在taxonomy中标记 `(Deprecated)` 而不删除
   ```markdown
   * `S.OldTag`: (Deprecated, use S.NewTag instead) 旧定义。
   ```
2. 更新所有使用该tag的rule
3. 保留至少一个release周期（让用户迁移）
4. 在下一个major version中删除

### 4.2 Tag重命名流程（Renaming）

**避免重命名**：Tag重命名会破坏外部引用，尽量通过alias解决。

**如果必须重命名**：
1. 添加新tag
2. 保留旧tag并标记 `(Alias for NewTag)`
3. 同时支持两个tag至少2个release
4. 逐步迁移所有rule
5. 弃用旧tag

---

## Part 5: 自动化检查（Automation）

### 5.1 Pre-commit Hook

**自动阻止无效tag提交**：
```bash
# .git/hooks/pre-commit
#!/bin/bash
python3 ${CLAUDE_SKILL_DIR}/scripts/analysis_engine/tag_validator.py
if [ $? -ne 0 ]; then
    echo "❌ Tag validation failed. Fix tags or update taxonomy before commit."
    exit 1
fi
```

### 5.2 CI/CD集成

**GitHub Actions示例**：
```yaml
name: Tag Validation

on: [push, pull_request]

jobs:
  validate-tags:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Validate Tags
        run: |
          python3 ${CLAUDE_SKILL_DIR}/tests/test_tag_validator.py
          bash ${CLAUDE_SKILL_DIR}/tests/e2e_test.sh fastgelu
```

---

## Part 6: FAQ

**Q: 我怎么知道是typo还是真的需要新tag？**

A: 运行validator，它会自动判断：
```bash
python3 scripts/analysis_engine/tag_validator.py
```
输出会明确标注：
- `🔍 Likely a TYPO` - 很可能是拼写错误
- `🆕 This looks like a NEW TAG` - 可能是新tag需求

**Q: 可以直接在code_tag subskill中创造新tag吗？**

A: **不可以**。所有新tag必须先经过此工作流添加到taxonomy，validator会阻止未注册的tag。

**Q: 如果我不确定tag是否需要，但想试验？**

A: 创建 `tag_proposals/` 提案，收集3个以上实际案例后再正式添加。

**Q: Tag命名冲突怎么办？**

A: 使用层次化命名，如：
- `C.Arch.910B` vs `C.Arch.910B2`
- `O.Attention` vs `O.SparseAttention`

**Q: 添加新tag后现有rule评分会变吗？**

A: 会。如果新tag与现有rule的symptom/context匹配，评分会重新计算。建议添加新tag后重跑benchmark。

---

## Part 7: 案例研究（Case Study）

### 案例1：成功案例 - 添加 `S.MteBusy`

**背景**：
- 发现多个算子在profiling中显示MTE单元繁忙但compute util不低
- 现有 `S.TransferDominated` 不够精确（它指总体搬运主导，而非MTE单元瓶颈）

**流程**：
1. 收集了5个算子案例（mhc_post, layernorm, softmax, ...）
2. 定义：`mte_ratio > 60%` 且 `compute_util < 80%`
3. 添加到taxonomy的 "搬运瓶颈" 类别
4. 创建3条rule使用该tag
5. 测试验证：准确匹配了所有5个算子

**结果**：成功区分了"搬运主导"和"MTE单元瓶颈"两类问题。

### 案例2：失败案例 - 尝试添加 `S.LowComputeUtil.Moderate`

**背景**：
- 想区分"略低"（50-70%）和"极低"（<30%）的compute util

**问题**：
1. 过度细分：增加了rule匹配复杂度
2. 阈值主观：不同场景的"略低"定义不同
3. 可替代：用 `C.ComputeUtil.Range` context tag + 自然语言描述即可

**结果**：提案被拒绝，改用现有tag组合方案。

---

## 总结：新Tag添加原则

1. **必要性优先**：没有3个以上实际案例，不添加新tag
2. **语义明确**：定义必须可量化（profiling metrics）或可验证（code pattern）
3. **向后兼容**：每次变更必须通过validator测试
4. **文档同步**：taxonomy、CHANGELOG、tests必须同步更新
5. **渐进迭代**：优先用提案试验，验证后再正式添加

**记住**：Tag taxonomy是系统的"类型系统"，每个新tag都是一个"类型"的引入，需要像对待API变更一样谨慎。
