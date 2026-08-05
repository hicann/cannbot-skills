---
type: reference
description: Curated URL index for hiascend.com CANN documentation — agents use this as starting point for hardware/API queries instead of guessing URLs
---

# hiascend.com 文档 URL 注册表

> **Purpose**: hiascend.com 使用 JS 懒加载导航，WebFetch 无法读取内容，playwright 也看不到完整 TOC。
> 本文件是人工 + agent 共同维护的 **已知有效 URL 列表**，是所有 agent 查询公开文档的起点。
>
> **Rule for agents**: 遇到硬件/API 问题时，先查本文件找最相关的 URL，用 `browser_navigate` + JS `innerText` 提取读取内容。找到新有价值的页面时，追加到本文件。
>
> **⚠️ Two URL trees — pick correct one**:
> - 编程指南（架构规格 / best practices / 流水线概念）: `.../opdevg/Ascendcopdevg/atlas_ascendc_*.html`
> - **API 参考（TPosition/TBuf/MrgSort/WholeReduce 等语言+指令规格）**: `.../API/ascendcopapi/atlasascendc_api_07_*.html`
> Historical failure: researcher/optimizer agents kept searching only the `opdevg` tree and missed all API-ref content, forcing escalation to internal queries for info that was actually public.

Root URL for CANN 9.0.0-beta.2: `https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/`

---

## 硬件架构规格 (programming-guide tree, `/opdevg/Ascendcopdevg/`)

| 页面 | URL (relative) | 内容摘要 | 覆盖问题 |
|------|-----|---------|---------|
| 架构规格 TOC | `atlas_ascendc_10_0009.html` | 各版本架构规格目录 | — |
| **NPU架构版本351x** | `atlas_ascendc_10_00065.html` | **Ascend950PR架构**: AIC/AIV结构, UB bank conflict规则 (2R+2W per group), AIV↔L1硬通道, SIMD Register File (RegTensor/MaskReg/UnalignReg/AddrReg), SSBuffer, CrossCoreSetFlag (flagId 0-10, max-count 15, 模式 0/1/2/4), Aux Scalar, FIXP→UB | IQ-1部分, IQ-3硬件层 |
| NPU架构版本300x | `atlas_ascendc_10_00064.html` *(未验证)* | 上一代架构 | — |
| NPU架构版本220x | *(待发现)* | Ascend910B架构对比 | — |
| 分形要求 | `atlas_ascendc_10_0099.html` | L0A/L0B/L0C/L1 分形格式要求 | — |

## 性能优化 Best Practices (programming-guide tree)

| 页面 | URL (relative) | 内容摘要 |
|------|-----|---------|
| Best Practices TOC | `atlas_ascendc_best_practices_10_00010.html` | 性能优化总览（knowledge-maintain `--learn` 默认 URL）|

---

## API 参考（API-reference tree, `/API/ascendcopapi/`）

> **When to use this tree**: any question about AscendC language构造、数据结构、指令规格、dtype 支持、align 要求、TPosition 合法性 等 API 级细节。

### TOC / entry

| 页面 | URL (relative) | 用途 |
|------|-----|------|
| API 总览 | `atlasascendc_api_07_0003.html` | AscendC API 索引根，展示整个 API 树结构 |

### 编程模型 / 数据结构

| 页面 | URL (relative) | 内容摘要 | 覆盖问题 |
|------|-----|---------|---------|
| **TPosition** | `atlasascendc_api_07_0174.html` | 枚举定义：GM/VECIN/VECOUT/VECCALC/**A1/A2/B1/B2/C1/C2/CO1/CO2**/LCM/SPM/SHM/TSCM/C2PIPE2GM/C2PIPE2LOCAL；每项含义说明表 | IQ-3 |
| **TBuf 简介** | `atlasascendc_api_07_0161.html` | TBuf 作为临时变量存储，**可以设置为任意 TPosition 逻辑位置**；TBuf 不支持队列操作；Get 获取 Tensor 无需释放 | IQ-3 |
| TBuf 构造函数 | `atlasascendc_api_07_0162.html` *(未抓取)* | TBuf 构造签名 | — |
| TBufPool 简介 | *(07_01xx 系列，未记录具体号)* | "管理 Unified Buffer / L1 Buffer 物理内存，主要用于多 stage ... 物理内存不足的场景" — L1 明确作为 TBufPool 可管理资源 | IQ-3 |

### 矢量归约 / 排序原语 (Memory矢量计算 → 归约计算 / 排序组合)

| 页面 | URL (relative) | 内容摘要 | 覆盖问题 |
|------|-----|---------|---------|
| **WholeReduceMax** | `atlasascendc_api_07_0079.html` | A5 dtype: half/float (Atlas 350 also u16/s16/u32/s32); max elements/iter 128(16b)/64(32b)/32(64b); repeatTime [0,255]; ReduceOrder: VALUE_INDEX / INDEX_VALUE / ONLY_VALUE / ONLY_INDEX; index 以 dst dtype 存储, reinterpret_cast 读取 | IQ-2 |
| WholeReduceMin | `atlasascendc_api_07_0080.html` *(未抓取，同 Max)* | 同 WholeReduceMax 但求最小 | IQ-2 |
| WholeReduceSum | `atlasascendc_api_07_0081.html` *(未抓取，同 Max)* | 同 WholeReduceMax 但求和 | IQ-2 |
| BlockReduceMax | `atlasascendc_api_07_0082.html` *(未抓取)* | 按 32B block (= 8 half 或 8 float) 归约 | IQ-2 |
| **MrgSort** | `atlasascendc_api_07_0232.html` | 4-way merge；8B (score,index) pair；score dtype half/float；max 4095 elements/queue；repeatTimes 1-255；TPosition VECIN/VECCALC/VECOUT | IQ-2 |
| ProposalConcat / Extract / RpSort16 / MrgSort4 / Sort32 | 排序组合 API 系列 *(未抓取具体号)* | 排序 pipeline 配件 | IQ-2 |

### 高阶 API (高阶API → 排序操作)

| 页面 | URL (relative) | 内容摘要 |
|------|-----|---------|
| TopK | *(未抓取)* | 高阶 TopK 包装 |
| Sort | *(未抓取)* | 高阶 Sort 包装 |
| Concat / Extract | *(未抓取)* | 归并/提取 |

### 同步控制

| 页面 | URL (relative) | 内容摘要 |
|------|-----|---------|
| CrossCoreSetFlag | *(未抓取; 359x 架构页已引用 API 签名 SetFlag<模板>, flagId)* | AIC↔AIV 同步 set |
| CrossCoreWaitFlag | *(未抓取)* | AIC↔AIV 同步 wait |

---

## 发现新页面时的追加格式

```markdown
| 页面标题 | `atlas[_ascendc|ascendc_api]_...html` | 一行内容摘要 | 回答了哪个问题（IQ-N / 新发现） |
```

写入时注意：**文件名前缀决定哪棵树**：
- `atlas_ascendc_10_*.html` → programming-guide tree
- `atlasascendc_api_07_*.html` → API-reference tree

---

## 如何高效读取页面内容

```javascript
// hiascend.com 是 JS 渲染的，必须用 playwright browser_navigate + evaluate
// WebFetch 返回的是空壳 HTML，不含实际文档内容

// 正确姿势：
// 1. browser_navigate(url)
// 2. browser_wait_for(time=2)
// 3. browser_evaluate: () => document.querySelector('main').innerText
//    或: () => document.body.innerText (fallback)

// 定位特定 spec 时可以用正则 slice：
// () => {
//   const txt = document.querySelector('main').innerText;
//   const idx = txt.lastIndexOf('WholeReduceMax\n');  // skip nav sidebar duplicate
//   return txt.slice(idx, idx + 3000);
// }
```

---

## 已知 gap / 待扫描

- `DataCopy` UB↔L1 operand surface: 确认 UB (VECCALC) ↔ A1/B1 的具体 API 签名页
- `CrossCoreSetFlag` / `CrossCoreWaitFlag` API ref 具体页号
- 220x 架构页（若存在）以建立 A5 vs 910B 公开文档对比
- API ref 中对 `TBuf<TPosition::A1>` 在纯 AIV kernel 中的是否有明确的编译器行为说明
