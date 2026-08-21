---
type: CATLASS DSL Learned Result
title: {{title_json}}
description: {{description_json}}
tags: {{tags_json}}
status: stable
generated: {{generated_json}}
verified: {{verified_json}}
sources: {{sources_json}}
operator_family: {{operator_family_json}}
topic: {{topic_json}}
arch: {{arch_json}}
versions: {{versions_json}}
applicability: {{applicability_json}}
result_status: {{result_status_json}}
kernel_sha256: {{kernel_sha256_json}}
---

# {{operator_family}}：{{topic}}

## 适用条件

- 架构：{{arch}}
- CATLASS DSL 版本条件：{{catlass_dsl_version}}
- CANN 版本条件：{{cann_version}}
- Shape：{{shape}}
- Dtype：{{dtype}}
- Layout：{{layout}}
- 仓库集成：{{repository_integration}}

## 可证伪假设

{{hypothesis}}

## 实际修改

{{actual_change}}

## 正确性与性能

- correctness_before：`{{correctness_before}}`
- correctness_after：`{{correctness_after}}`
- performance_before：`{{performance_before}}`
- performance_after：`{{performance_after}}`
- profiling_observation：`{{profiling_observation}}`

## 已验证结果

{{result}}

- 结论状态：{{status}}
- Kernel SHA-256：`{{kernel_sha256}}`

## 原始证据链接

{{evidence}}

> 本条目只总结证据直接支持的结论；链接失效时不得继续引用为已证实经验。
