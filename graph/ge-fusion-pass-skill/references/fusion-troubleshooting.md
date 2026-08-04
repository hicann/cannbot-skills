# 融合 pass 诊断树导航（fusion-troubleshooting）

融合 pass「没生效」「报错了」时，先分清**三层**再动手：**pass 有没有被加载 → 有没有被执行 → 有没有命中**。三层混淆会把「没加载」误诊成「pattern 写错」。

> 谁读：③ 验证诊断（主线）；② 开发在「写错还是没生效」之间判断时回看。
>
> 本文是**诊断树的导航层**：给出每个节点的现象 + 一句提示 + 指向权威 tip / 统一 skill 阶段三“诊断、性能与交付”。**报错原文 / 根因表 / 修法正文都在各 tip 与阶段三里，本文不复制**。
>
> 诊断纪律的根是 `tips/failure-attribution.md`：失败先按报错匹配到对应 tip 的「自查」清单逐条过——大多数挂点是「自己漏了某一步」（漏设 NCHW、用了框架名、残留 pass），不是环境 / 文档问题；确属环境缺失才按 G3 标「未运行」。

```
pass 没生效 / 报错
   │
   ├─ 1. pass 是否被加载？        ── 否 → 加载 / 安装 / 环境变量问题
   ├─ 2. pass 是否被执行？        ── 否 → 注册阶段 / 装饰器 / 触发编译命令问题
   ├─ 3. 是否产生候选与匹配？     ── 否 → op type / 输入个数 / 边界问题
   ├─ 4. 守卫为何拒绝？           ── MeetRequirements 返回 false（dtype/shape/attr）
   ├─ 5. replacement 是否成功？   ── 否 → 建图 / format / IR 顺序问题
   ├─ 6. InferShape / format / engine 是否成功？ ── 否 → shape 推导 / format / kernel 问题
   └─ 7. 图与输出是否正确？       ── 否 → dump 对比 / 残留 pass 污染
```

---

## 1. pass 是否被加载？

**现象**：run.log 里连 pass begin 的 `std::cout`/`print` 都没有；C++ 没有任何 pass 日志；Python 无 pass 日志且无 Python 报错。

**方向**：先确认产物在不在加载路径、加载机制对不对，再看注册宏 / 阶段。**加载基线**——跨轮次残留的 `.so`/`.py` 会被一并加载、污染下一次编译——见 `tips/stale-pass-artifact-cleanup.md`（只清理本轮 build 产物，共享 vendor 目录默认只读、只盘点不删；无法证明加载集合已隔离时标注「加载基线未确认」）。独立 C++ pass 加载根如无官方文档或实测证据，按未确认处理并在本 case 的 `requirements-analysis.md` §10 记录确认计划。

具体报错与修法对照（C++ `.so` 路径、Python `ASCEND_GE_PY_PASS_PATH` 的几类常见失败、装饰器遗漏、`name` 重名等）见统一 skill 阶段三“诊断、性能与交付”。

## 2. pass 是否被执行？

**现象**：模块被 import / `.so` 被 dlopen，但无 pass 日志；或 GE 报 pass 执行失败但代码「看起来成功了」。

**方向**：
- Python：漏了 `@register_fusion_pass` / `@register_decompose_pass` 装饰器，或 `stage` 不对，或 `name` 与已注册 pass 重名。
- **Python `run()` 被判失败**：GE 报 pass 执行失败但代码「看起来成功」→ `run()` 返回了 `0` 或 `False`（**假值即失败**，成功要返回 `True`/`None`）。返回值语义与 C++ 相反，见 `interface-catalog.md` §二。
- `PatternFusionPass` / `DecomposePass` 误重写了 `run()` → 类定义时即抛 `TypeError`（见 `interface-catalog.md` §二）。

## 3. 是否产生候选与匹配？

**现象**：pattern/遍历静默不命中、fusion 跳过、首次 ATC 看似成功却没融合；无报错；dump 拓扑无变化、run.log 里 `meet requirements`/`replacement` 未出现。

**方向**（按命中率从高到低）：
1. **框架名 ≠ GE 导入后 op type**（最常见的静默失败原因，不报错）→ `tips/dump-first-op-type.md`。
2. **可选输入导致输入个数对不上**（`bias`/`offset_w` 在，pattern 却按不带建）→ `fragment-spec.md` §四。
3. pattern 输入个数 / 输出边界与真实图不一致 → `interface-catalog.md` §一.2。

## 4. 守卫为何拒绝？

**现象**：`MeetRequirements` 被调用但返回 false，run.log 里 `meet requirements (false, reason=...)`。

**方向**：守卫读的是 dtype / shape / attr / Const 值。常见是 dtype/shape 不符预期、或 Const 值匹配无容差（严格相等）。匹配严格度（`PatternMatcherConfig` vs `MeetRequirements`）见 `interface-catalog.md` §一.4；守卫写在 `MeetRequirements`、不在 `Replacement` 里返 null 的写法见 `example-map.md`（`4_add_zero_pass` 行）。**注意**：依赖 InferShape 后真实 shape 的守卫须注册在 `kAfterInferShape`（见 `pass-development-paradigm.md` §5）。

## 5. replacement 是否成功？

**现象**：`Got null replacement graph`；或建图后报 IR / format 类错误。

**方向**：
- **format 未设**（中间 tensor `format=ND`）→ `tips/format-sensitive-nchw.md`（E50002）。
- **手建节点 IR 顺序错**（`attribute order has changed` / `Failed to recover ir definitions`）→ `tips/compliant-node-builder-ir-order.md`。
- replacement 输入顺序与 pattern 边界不一致、可选输入缺失时断边 → `fragment-spec.md` §四 replacement 侧。

## 6. InferShape / format / engine 是否成功？

**现象**：`InferShape ... failure`；`Not_Supported_Format(E50002)`；`Failed to select engine for [XxxOp]`。

**方向**：
- **InferShape 失败**：C++ 推 shape 用了 `GeUtils::InferShape`（错）或 format 未设 → `tips/infershape-util-only.md`（只用 `InferShapeUtil::InferShape`）+ `tips/format-sensitive-nchw.md`（InferShape 失败常是 format 而非接口问题，先排 format）。
- **format 报错（E50002）** → `tips/format-sensitive-nchw.md`。
- **引擎选择失败**（`Failed to select engine`）：该 op type 本 soc 无 kernel → `tips/es-all-no-version-rename.md`（按报错算子回退等价实现，**不预换版本名、不反复重试同一报错 op type**）；根因背景见 `tips/kernel-registration-mismatch.md`（「能建图 ≠ 能编译」）。

## 7. 图与输出是否正确？

**现象**：pass 生效了（dump 有变化、日志齐全），但图结构不对、或输出数值/shape 不对；或「dump 里的变化对不上当前源码」。

**方向**：
- **dump 前后对比**：文件名随注册阶段与 CANN 版本变化，找错文件会误判「pass 没跑」。文件名表、大小写不敏感匹配、采集级别见统一 skill 阶段三“验证顺序与证据”；典型变化对照（MatMul+Add→GEMM、Add(x,0) 删除、grouped Conv 拆分等）与日志关键字清单见 `tips/dump-log-diff-checklist.md`。
- **残留 pass 污染**：dump 出现「非预期的替换效果」、和当前源码对不上 → 先按 `tips/stale-pass-artifact-cleanup.md` 排查残留 pass，再怀疑本次代码。
- **输出正确性**（baseline vs optimized 整网最终输出比较）属验证证据阶段；本诊断树只到「pass 是否按预期改图」一层。

---

> 诊断时若踩到 `references/` 尚未覆盖的**新**坑（新现象 + 根因 + 验证过的修法），用户可明确请求统一 skill 的阶段四沉淀；优先更新本文档对应节点的导航落点，再决定是否新开 tip（见统一 skill 阶段四）。迁移映射见 `tips/MIGRATION.md`。
