# Hypothesis Template

> 复制本文件到 `retro/YYYYMMDD_短描述.md`，填写后执行 `scripts/validate_hypothesis.sh` 校验。
> 通过后由 `scripts/build_index.sh` 自动提升至 `hypotheses/` 并更新 INDEX.md。

---

```yaml
---
id: H{xx}                    # 由 build_index.sh 分配，写 retro 时填 H_NEW
title: {一句话描述，≤25字}
symptom: {TAXONOMY.md symptom 合法值}
when: {TAXONOMY.md when 合法值}
root_cause: {TAXONOMY.md root_cause 合法值}
evidence: {TAXONOMY.md evidence 合法值}
escalate_to: {null | mssanitizer | msaicerr}
source: {ascendc-debug.md | mssanitizer-helper | msaicerr-helper | retro/日期}
---
```

## triggers
<!-- 必填，≥2条，描述触发此 hypothesis 的具体现象 -->
- 现象1
- 现象2

## read_target
<!-- evidence=code 时必填；描述 CC 应读哪些文件哪些位置 -->
- `op_host/{op_name}_custom.cpp` → grep `{关键词}`
- `kernel/{op_name}.cpp` → 看 `{函数名}` 中的 `{代码段}`

## code_pattern
<!-- 必填，展示 bug 代码的典型形态，用代码块 -->
```cpp
// ❌ 错误写法
{bug_code}
```

## fix_template
<!-- 必填，给出正确代码，用代码块 -->
```cpp
// ✅ 正确写法
{fixed_code}
```

## verify_cmd
<!-- 必填，如何验证修复有效 -->
- 步骤1
- 步骤2

## notes
<!-- 可选，踩坑经验、易混淆点、来源说明 -->
