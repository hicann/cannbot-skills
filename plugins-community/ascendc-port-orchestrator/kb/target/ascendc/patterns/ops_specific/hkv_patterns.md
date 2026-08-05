# HKV 哈希表算子专属 Patterns

> 从 HierarchicalKV source→AscendC 翻译对比中提取。
> 仅适用于 HKV 类哈希表算子（lock/evict/digest/bucket 操作）。
>
> **通用化的部分已移入 [general_patterns.md](../PATTERN_INDEX.md)  <!-- legacy: general_patterns.md was consolidated -->**（P-P13 协作组规约、P-P16 Value 搬运、P-P17 前缀和聚合、F-P5 warp 对齐）。
> 本文件仅保留 HKV 专属内容。
>
> **完整代码示例**: 见 [HKV_TRANSLATION_PATTERN_ANALYSIS.md<!-- legacy: HKV_TRANSLATION_PATTERN_ANALYSIS.md was removed -->

---

## 精度/功能类

### H-F-P1: 两阶段提交协议（Lock-Key → Unlock-Key 分离）

**严重度**: **严重** | **来源**: HKV AI 版 bug

`find_or_insert_ptr` 返回 value 指针让调用方写入 → key 位置必须保持 LOCKED 直到外部 `unlock_key` 内核。

**反模式** (AI 在 lock 后立即 unlock):
```cpp
Simt::AtomicExch(current_key_ptr, key);  // BUG: 立即解锁 → 调用方写 value 时数据竞争
```

**正确模式**: 不在此解锁。调用方写完 value 后，用 `find_or_insert_ptr_unlock_key` 内核统一解锁。

---

### H-F-AP1: 已有 key 时冗余 digest 更新

**严重度**: 低

找到已存在 key (DUPLICATE) 时应使用 `update_score_only`，不需要 `update_with_digest`（digest 是 key 的指纹，key 没变 digest 不变）。

---

## 性能类

### H-P-P1: Digest 向量化查找（vcmpeq4）

**严重度**: 高 | **来源**: HKV 手写版

```cpp
// 4 个 1-byte digest 同时比较（1 条 32-bit load + vcmpeq4）
VecD_Comp target = digests_from_hashed<K>(hashed_key);
VecD_Comp probe = *(reinterpret_cast<__gm__ VecD_Comp*>(digests_ptr));
uint32_t cmp = vcmpeq4(probe, target);
// 只有 digest 匹配 (1/256) 时才读完整 8-byte key
```

bucket_capacity=128: 128 次 8-byte 比较 → 32 次 4-byte 比较。

**注意**: `vcmpeq4` 是 HKV 项目的软件实现（types.h），不是硬件指令。

---

### H-P-P2: 读写分离——纯查找路径无需 AtomicCas

**严重度**: 中

当 `update_score=false`（纯查找），创建独立 kernel 去掉 AtomicCas/AtomicExch/ScoreFunctor，只用 `ldg_l2nc_l1c` 只读访问。原子操作开销远大于普通 load。

---

### H-P-P3: 协作组大小模板参数化

**严重度**: 低

```cpp
// 反模式: constexpr int32_t GROUP_SIZE = 16;  // 硬编码
// 正确: template <..., int32_t GROUP_SIZE = 16>
// 支持 DISPATCH_GROUP_SIZE 宏在不同负载率下选择 2/4/8/16/32
```

---

### H-P-AP1: 单线程淘汰遍历

**严重度**: 高

```cpp
// 反模式: 单线程遍历整个 bucket 找最小 score
for (uint32_t i = 0; i < bucket_capacity; i++) { ... }
```

**修复**: 使用协作组并行遍历 + shuffle 规约（通用 P-P13）。
