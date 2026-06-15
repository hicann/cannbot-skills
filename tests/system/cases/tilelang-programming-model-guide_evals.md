---
skill_name: tilelang-programming-model-guide
---

# Case 1: Developer 模式与 Expert 模式对比

## Config
- Max Tokens: 80000
- Ascend Platform: A2

## Prompt

在 TileLang-Ascend 中，Developer 模式和 Expert 模式有什么区别？我应该如何选择？

## Expected Output

回复应从五个维度对比两种模式：内存分配（Developer 用 T.alloc_shared/T.alloc_fragment，Expert 用 T.alloc_L1/T.alloc_ub/T.alloc_L0A/L0B/L0C）、作用域（Developer 自动分离 Cube/Vector，Expert 手动 T.Scope）、同步（Developer 自动插入，Expert 手动 T.barrier_all/T.set_flag/T.wait_flag）、CV 交互（Developer 消除 workspace+vid 片上直连，Expert 显式 GM workspace）、pass_configs（Developer 全部开启，Expert 全部关闭）。混合模式可在 Developer 主体中混合少量 Expert API。

## Expectations
- [contains] Developer
- [contains] Expert
- [contains] T.alloc_shared
- [contains] T.alloc_fragment
- [contains] T.Scope

---

# Case 2: pass_configs 配置指南

## Config
- Max Tokens: 80000
- Ascend Platform: A2

## Prompt

TileLang-Ascend 有哪几个 Ascend 专用 pass_configs 开关？纯 Vector 算子和 Developer GEMM 分别应该怎么配置？

## Expected Output

回复应列出四个 Ascend 专用 pass_configs：TL_ASCEND_AUTO_SYNC（自动核内同步）、TL_ASCEND_MEMORY_PLANNING（自动内存规划）、TL_ASCEND_AUTO_CV_COMBINE（自动 CV 分离）、TL_ASCEND_AUTO_CV_SYNC（自动核间同步）。纯 Vector 算子只需开启 AUTO_SYNC + MEMORY_PLANNING；Developer GEMM/CV 融合需开启全部四个。Expert 极致性能模式全部关闭并手动控制。

## Expectations
- [contains] TL_ASCEND_AUTO_SYNC
- [contains] TL_ASCEND_MEMORY_PLANNING
- [contains] TL_ASCEND_AUTO_CV_COMBINE
- [contains] TL_ASCEND_AUTO_CV_SYNC
- [contains] threads=2
