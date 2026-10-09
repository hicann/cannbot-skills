---
type: operator
title: SparseFlashMLA 完整接口与一般 SWA 契约
description: 主算子和 metadata 的完整 Python/ACLNN 接口，以及一般 SWA 的布局、stride、有效长度、sink 和 LSE。
tags:
- catlass-cpp
- sparse-flash-mla
- api
- contract
- metadata
- aclnn
- SWA
- ragged
- stride
- sink
status: draft
operator_families:
- sparse-flash-mla
architectures:
- atlas-a2-a3
- ascend910b
generated:
  by: swa-general-final
verified: []
sources:
- id: api-computation-contract
  resource: knowledge/operator/sparse-flash-mla/computation.md
  title: 参数对应的计算语义
  kind: contract
- id: swa-contract-interface-contract
  resource: knowledge/operator/sparse-flash-mla/interface.md#api
  title: 两个公开接口及声明
  kind: contract
- id: swa-contract-computation-contract
  resource: knowledge/operator/sparse-flash-mla/computation.md
  title: 流式舍入与位置语义
  kind: contract
- id: swa-contract-metadata-contract
  resource: knowledge/operator/sparse-flash-mla/metadata.md
  title: 任务表生产消费协议
  kind: contract
- id: swa-contract-diagnostic-contract
  resource: knowledge/operator/sparse-flash-mla/validation.md#suite
  title: 诊断 ABI 与新增能力的验证边界
  kind: contract
- id: swa-contract-pytorch-fake-dispatch
  resource: knowledge/operator/sparse-flash-mla/workflow.md#materials
  title: 诊断脚本的 Fake/Meta 分派约束
  kind: material
---

本页按主题合并知识；下列目录进入各主题的起点，各通用章节下保留完整细节。
通过知识工具 get 时使用文件路径，不把章节锚点传给 get。

- [完整公开接口](#api)
- [一般 SWA 合法域](#swa-contract)

# 接口与概念

<a id="api"></a>

<a id="api--接口与概念"></a>

## 完整公开接口

从[计算特征](computation.md)进入本页，在生成 host、wrapper 或 kernel 前读取；接口声明直接写在本页 §5，两段式调用和 tensor 访问约束见[接口契约](workflow.md#materials--接口参考)。这里给出可核对的完整参考接口，而不是简化的 Q/K/V 示例。算法识别依据调用者提供的标杆中的选择集合和归一化关系，不依据下列函数名。

<a id="swa-contract"></a>

<a id="swa-contract--接口与概念"></a>

## 一般 SWA 合法域

本条目用于 **SWA 单模式的一般输入**。主算子仅选择原始 KV 的右下因果窗口，K=V；同时生成可独立调用的 `SparseFlashMlaMetadata`，主算子实际消费其输出。公开参数槽保持完整，CSA/HCA、原始 KV 稀疏计算等域外请求明确拒绝；域内尚未实现的组合属于待完成缺口，不能靠缩小声明域完成任务。本条目对功能范围的限定优先于三模式通用知识中的全模式验收要求，不改变成对交付要求。

一般输入意味着从运行时 shape、dtype、stride、长度和页表推导任务与地址，不将 case ID、N1=4、B=1、S1=S2=4096 或固定 metadata 表写成合法输入白名单。用户指定的 SWA 测试子集只是回归集合；它不能定义整个支持域，也不能证明尚未覆盖的布局已经正确。

区分公开声明、arch22 的实际实现包络、用户 golden 的数学语义。数学上可定义不等于旧实现支持，旧实现的缺口也不能静默替换用户标杆。下文自包含地列出已归纳的真实边界与本次修正要求，生成者无需读取外部实现。`verified: []` 不代表生成工程已经完成 NPU 验证。

本任务完整域由三类能力清楚划分：第一类为910B/arch22公开兼容域（D512、N2=1、幂次N1、127/0窗口、四布局、实际batch/长度、必需sink及LSE开关）；第二类为按当前golden与输入ABI要求实现的修正能力（TND的有效长度与存储边界分离、KV实际正stride、全量O/LSE零行）；第三类为诊断ABI可表达但不自动升级为正式接口要求的扩展（例如缺失sink）。前两类属于本任务实现与验证范围，第三类必须显式声明支持或返回清晰的扩展不支持错误，不能混称原arch22已支持。[^swa-contract-diagnostic-contract]

# 用法

<a id="api--用法"></a>

## 完整公开接口

**成对交付：要求生成 `sparse_flash_mla` 时，必须在同一次任务中同步生成 `sparse_flash_mla_metadata`，包含实际实现、可调用入口、相应构建产物与联调验证。该要求不依赖是否提出旧接口兼容。** 下表决定采用哪一层调用形式，不改变这两个算子的交付范围；metadata 算法与表协议见[配套调度知识](metadata.md)。

| 用户输入 / 交付要求 | 应冻结的公开契约 |
|---|---|
| 要求生成 `sparse_flash_mla`（新实现或替换） | 同步生成 `sparse_flash_mla_metadata`；固定两个公开入口、任务表协议与生产/消费关系，按下列交付形式实现 |
| 提供 Python 标杆，要求相同调用方式 | 保留实际入口、构造状态、参数名/顺序/默认值、返回结构；用下文 §2 映射，不自动换成旧 ACLNN ABI |
| 只有等价公式，未指定接口 | 由公式的外部依赖确定完整接口；下文 §3–4 提供现有共享 KV 实例的接口参考，记录采用与不适用项 |
| 要求替换现有 SparseFlashMla / PyTorch 调用 | 完整保留 §3 的两个 schema、默认值、可选参数槽和输出，按 §4 对接 metadata；内部 Catlass 工程名不改变公开符号 |
| 要求 ACLNN 算子包互通 | 交付 §5 的四个导出符号及注册、形状推导、AICPU metadata 集成，核对目标包的头文件；需要 Python 调用时再附带 §3 wrapper |
| 只要求 Catlass 直调 | 公开 host 入口仍覆盖已冻结的输入输出；生成本算子时同时提供可独立调用的 metadata 入口和实现。device Params/tiling 可重设计，但必须实际消费配套算子输出；§6 作参数去向参考，直调交付不等同于 ACLNN 包 |

在 docs/design.md 留一份接口映射表：`源参数/输出 → 公开入口 → host/tiling/device 去向 → dtype/shape/可空条件 → 实现与验证状态`。构造函数保存的 scale/layout/mask 也是外部输入。测试样例恰好为 None 或未使用的模式，不是删除参数的依据。声明存在但当前硬件不支持的组合应有明确校验，不能静默丢弃参数后声称完整兼容。

以下签名为声明参考，代码中的省略号不代表实现。选择的交付层应实现实际行为；要求生成 `sparse_flash_mla` 时，两者都要实现，不能用省略包装层来省略配套算子。仅借用计算机制开发其他算子的公式任务，仍按其自身公开契约确定交付层。

<a id="swa-contract--用法"></a>

## 一般 SWA 合法域

先读[完整接口](interface.md#api)，冻结 main 28 参数、metadata 27 参数的 Python schema，以及选定交付层的四个 ACLNN 符号。ACLNN 声明以[接口知识 §5](interface.md#api--5-aclnn-两段式-c-接口)中的代码块为参考，不提供可直接 include 的头文件；实际包互通仍需核对目标 CANN 头文件、导出和链接。

| SWA 调用字段 | 冻结与接线规则 |
|---|---|
| `q`, `ori_kv`, `metadata` | 正式SWA实际必需；Q/KV同为FP16或BF16，metadata为INT32 `[1024]` |
| `sinks` | 正式arch22兼容域必需、非空FP32 `[N1]`；公开schema的Optional标记及默认None不改变这一调用要求。诊断可空字段属于单独扩展，见下文 |
| `cmp_kv`, 两组 `sparse_indices` | SWA 取 None；非空时在公开入口明确拒绝，不静默切模式或忽略 |
| `ori_block_table` | PA 必需，INT32 `[B,max_pages_per_batch]`；非 PA 不需要 |
| `cmp_block_table`, `cu_seqlens_cmp_kv`, `seqused_cmp_kv`, `cmp_residual_kv` | 保留槽；SWA无压缩KV，不消费这些值进行attention。合法辅助tensor可传入，不能统一按“非空”拒绝；逐槽shape/dtype及实际接线见下表 |
| `ori_topk_length`, `cmp_topk_length` | arch22保留槽，None或空tensor可表示未启用；非空tensor明确拒绝。不能把该规则推广到其他cmp辅助槽 |
| `cmp_ratio` | SWA 规范值为 1；case 配置中的 None 由 adapter 转为 1 |
| mask/window | ori mask4、左127、右0；arch22 main 兼容调用用 `cmp_mask_mode=3`，见下文 |
| `softmax_scale` | 接受有限的 Python float/C++ double，包括0、负值和超过FP32有限范围的值；实际传入，不硬编码 1/√512，不作用于 sink。参数合法性与计算中的溢出分开处理 |
| `topk_value_mode` | 公开兼容值和默认值为1，生成正式入口明确验证1；SWA无TopK计算，不把它当模式选择器。原arch22特性检查未对该字段执行与A5相同的值检查，不据漏检宣称其他值有支持语义 |
| `return_softmax_lse` | 实现开/关两种值，均返回二元组；关闭时设备 wrapper 的 LSE 为 FP32 `[0]` |
| metadata 模式属性 | 显式 `has_ori_kv=True, has_cmp_kv=False, ori_topk=0, cmp_topk=0`；两个 has 的默认值均为 True，不能依赖默认值 |

**cmp辅助槽逐项规则：** `cmp_kv=None`和无压缩索引决定SWA模式；辅助控制tensor存在本身不切换模式。公开合法的辅助值可以保留并验证而不进入数学，不能把“未使用”误写成“必须None”。

| 辅助槽 | 正式接口表示 | arch22已知行为与本次接线 |
|---|---|---|
| `cmp_block_table` | INT32 `[B,max_pages]`，可省略 | SWA不取cmp页、不读取其页号；既有main只在压缩分支执行cmp页表专项校验。新入口保持合法辅助表可传并记录未使用，不要求它与不存在的cmp存储构成有效页映射 |
| `cu_seqlens_cmp_kv` | INT32 `[B+1]`，可省略 | main对已传tensor检查rank/dtype和B+1；SWA数学不消费。metadata在`has_cmp_kv=False`时不据其值创建压缩任务 |
| `seqused_cmp_kv` | INT32 `[B]`，可省略 | main校验已传tensor的类型和batch长度；SWA不以其覆盖原始KV长度，不要求不存在的cmp容量 |
| `cmp_residual_kv` | INT32 `[B]`，可省略 | 合法非空tensor可通过main校验；SWA不以它改变原始窗口，metadata不创建cmp依赖。它并非SWA必需输入 |
| 两个`topk_length` | 保留Optional槽 | arch22兼容域允许None/空tensor，拒绝非空tensor；这与前四项不同 |

前四项默认None；实际传入时按公开rank/dtype/shape契约处理，不把入口漏检当作允许任意畸形输入的承诺。其值不得触发host attention、读取不存在的cmp KV或改变SWA输出。测试应分别覆盖None、合法非空及错误rank/dtype/长度；既有入口的接受行为、生成入口的规范校验和数学未使用理由分别记录。一般已传辅助tensor的空tensor行为不能从topk_length的特例外推。

**sink与诊断扩展：** 正式兼容入口拒绝缺失或空sinks，逐launch读取本次FP32 `[N1]`。诊断descriptor允许以空指针表达无sink，只说明ABI预留该状态；若实现支持，必须单独定义无sink的max/sum初始化、empty-row行为和独立参考，不可用全0 sink代替，也不能将未经适配、实际必需sinks的用户golden冒充无sink参考。诊断通过不能证明正式arch22支持缺失sink。[^swa-contract-diagnostic-contract]

**cmp mask适配：** case的`cmp_mask_mode=None`表示没有压缩计算；arch22 main仍要求属性值3。因此兼容adapter仅将该缺省转为3，main与metadata使用同一属性；metadata在`has_cmp_kv=False`时不检查此压缩分支。不得将显式非法值偷偷改成3；新wrapper的默认值和接受域写入`docs/api.md`。这是参数适配，不是添加CSA/HCA。

metadata 的 head、batch、layout、窗口和长度从同一次 main 调用获得。TND 的 B 是 `len(cu_seqlens_q)-1`，不是 T1；BSND 的 max_seqlen 是容量；PA 有效长度来自 seqused，不是 page_size；TND 总 token 数不是各 batch 最大长度。调度输入改变后重新产表。两个算子各有 executor/workspace，并保持任务表就绪与消费的 stream 依赖。

**公开接口与内部调度表分开冻结。** 新生成的两算子默认成对使用，不要求与旧手写 metadata 字节互通。公开 INT32 `[1024]` 和参数槽保留，内部可以使用版本化 header、batch 有效长度/存储基址、任务区间及实际 tile 信息；生成/消费端必须来自同一轮并验证容量、完整初始化、非法版本拒绝。若用户明确要求旧包互通，才必须严格遵守 [metadata 页](metadata.md) 的旧 FA/FD 字段协议。旧 Mbase=256 分区是该协议的实现选择，不是新成对算子的性能限制；不应一面改小 Mtile，一面按旧表把任务误解释。正式与诊断入口复用同一规划规则，诊断接口不能偷偷采用不同数学或不受容量限制的私有表来掩盖正式入口缺口。

1024-word容量不意味着batch数量只能等于剩余表长除以某个batch记录大小。原接口没有由这种私有编码产生的112等batch上限；若将全部batch信息塞入表而拒绝更多batch，这是生成实现的功能缺口。更通用的表只保存版本、tile、总任务和每core起止游标，空间随core数增长；每batch的存储前缀/有效长度继续由main的公开tensor槽提供，或由本轮wrapper整理到明确计入workspace和执行成本的控制区。任务游标必须能跨batch恢复，不能依赖每轮从零线性扫描全部batch。诊断ABI的长度在host，因此如kernel需要这些值，launch负责把本次数据传入其拥有的设备控制区；异步kernel不得解引用host数组。小batch内嵌可作为快路径，但应有容量外的合法回退或如实保留未完成状态，不能将自选私有表上限描述成SWA契约。

# 代码模式

<a id="api--代码模式"></a>

## 完整公开接口

<a id="api--算子算法"></a>
## 算子算法

流式标杆入口（`RUN_MODE=1`）的构造函数有29个外部参数，`forward`有17个（均不计self）：

```python
class GeneralizedSFA:
    def __init__(
        self,
        layout_q,
        layout_kv,
        q_type,
        ori_kv_type,
        cmp_kv_type,
        B,
        S1,
        S2,
        T1,
        N1,
        N2,
        D,
        K,
        block_num1,
        block_num2,
        block_size1,
        block_size2,
        cu_seqlens_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        ori_topk_length,
        cmp_topk_length,
    ):
        ...

    def forward(
        self,
        q,
        ori_k,
        ori_sparse_indices,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        sinks,
        template_idx,
        cmp_k=None,
        cmp_sparse_indices=None,
        seqused_q=None,
        ori_topk_length=None,
        cmp_topk_length=None,
        return_softmax_lse=False,
    ):
        ...
```

| 标杆字段 / 行为 | 对接时要保留或明确转换的内容 |
|---|---|
| `q`, `ori_k`, `cmp_k` | 对应设备的 `q`, `ori_kv`, `cmp_kv`；本例 K=V，不增加独立 V 或 q_rope/k_rope 输入 |
| 构造状态中的 `softmax_scale`, `cmp_ratio`, mask/window/layout | 必须传到主算子的属性/tiling；不能只照抄 forward 形参而遗漏构造状态 |
| `N1/N2/D`, `B`, `K` | 对应 metadata 的 head/batch/TopK 属性；K 是索引容量，不是内核里再执行一次 TopK 排名 |
| `template_idx=0/1/2` | 标杆分别是 SWA / CFA / SCFA（窗口 / 窗口+压缩前缀 / 窗口+压缩索引）；旧实现相应称 SWA / HCA / CSA，不是额外的公开 ACLNN 参数 |
| 各组 `cu_seqlens_*`, `seqused_*`, `cmp_residual_kv` | 前缀和、有效长度和压缩余数是不同输入。标杆压缩可见边界使用 Lo，旧实现使用 Lc*ratio+residual；依照[计算知识](computation.md)裁决差异，接口相似不证明计算相同 |
| `ori_sparse_indices`, `ori_topk_length`, `cmp_topk_length` | 此 forward 将对应中间量置 None，没有启用这些功能。保留入口参数并记录事实；要求扩展这些功能时需另有有效标杆，不能宣称该文件已覆盖 |
| 页表与分页 KV | forward 实际使用整理后的逻辑 KV；`gen_data` 另产出分页张量/页表给设备。不能把 CPU 逻辑 KV 误当成 PA 物理页数组 |
| `attn_out` | 标杆 BSND 返回 q dtype；TND helper 分配 FP32，形状仍与 q 相同。旧设备接口返回 q dtype，兼容标杆的 wrapper 要显式处理输出 dtype |
| `softmax_lse` | 标杆关闭时为 None；BSND 为 FP32 `[B,N2,S1,G]`；TND helper 最后转置为 FP32 `[T1,N2,G]`。旧设备接口 TND 是 `[N2,T1,G]`，关闭时是空 tensor；比较前要按目标契约转换，不能只比较 numel |

`gen_data` 的 `input`/`metadata_input` 字典不是完整的公开 schema：`input` 没有 metadata、cmp_ratio、topk_value_mode、return_softmax_lse 等全部字段；`metadata_input` 的 N1/N2/D/B/K 是测试层命名，不能原样 `**dict` 调用 PyTorch 接口。缺项从原始 params、实际张量和选中模式补齐，不从 API 默认值猜测。

<a id="api--3-pytorch-的两个完整接口"></a>
## 3. PyTorch 的两个完整接口

以下为 `torch.ops.cann_ops_transformer` 的调用形式。`*` 后必须用关键字；Python 内部 dispatcher 函数允许位置参数，不代表公开 schema 也允许。main 有 **18 个 tensor 输入（含 q）+ 10 个属性 = 28 个参数**；metadata 有 **9 个 tensor 输入 + 18 个属性 = 27 个参数**。

```python
def sparse_flash_mla(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    softmax_scale=1.0,
    cmp_ratio=1,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q='BSND',
    layout_kv='BSND',
    topk_value_mode=1,
    return_softmax_lse=False,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...

def sparse_flash_mla_metadata(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    *,
    cu_seqlens_q: Optional[torch.Tensor]=None,
    cu_seqlens_ori_kv: Optional[torch.Tensor]=None,
    cu_seqlens_cmp_kv: Optional[torch.Tensor]=None,
    seqused_q: Optional[torch.Tensor]=None,
    seqused_ori_kv: Optional[torch.Tensor]=None,
    seqused_cmp_kv: Optional[torch.Tensor]=None,
    cmp_residual_kv: Optional[torch.Tensor]=None,
    ori_topk_length: Optional[torch.Tensor]=None,
    cmp_topk_length: Optional[torch.Tensor]=None,
    batch_size: Optional[int]=None,
    max_seqlen_q: Optional[int]=None,
    max_seqlen_ori_kv: Optional[int]=None,
    max_seqlen_cmp_kv: Optional[int]=None,
    ori_topk: Optional[int]=None,
    cmp_topk: Optional[int]=None,
    cmp_ratio: Optional[int]=None,
    ori_mask_mode: Optional[int]=None,
    cmp_mask_mode: Optional[int]=None,
    ori_win_left: Optional[int]=None,
    ori_win_right: Optional[int]=None,
    layout_q: Optional[str]=None,
    layout_kv: Optional[str]=None,
    has_ori_kv: Optional[bool]=None,
    has_cmp_kv: Optional[bool]=None,
) -> torch.Tensor:
    ...
```

两个符号由同一 `sparse_flash_mla.py` builder 注册、同一 C++ 扩展导出。不能只生成一个主函数，或把 metadata 函数改成无对外入口的辅助代码后声称兼容两个算子。metadata 还注册了 tensor 全为 None 时的 dispatcher fallback，仍调用 NPU 实现；只注册有 tensor 的派发路径会遗漏这种合法调用形式。

metadata 的 Python dispatcher 在调用 C++ 前对 None 做以下归一化；这些是默认值，不保证默认组合可运行：

| 属性 | None 转换值 |
|---|---|
| `batch_size`, `max_seqlen_q`, `max_seqlen_ori_kv`, `max_seqlen_cmp_kv`, `ori_topk`, `cmp_topk` | 0 |
| `cmp_ratio` | 1 |
| `ori_mask_mode`, `cmp_mask_mode` | 0 |
| `ori_win_left`, `ori_win_right` | -1 |
| `layout_q`, `layout_kv` | `"BSND"` |
| `has_ori_kv`, `has_cmp_kv` | True（不是自动检查主算子输入） |

<a id="api--主接口的参数语义与形状"></a>
### 主接口的参数语义与形状

记 `G=N1/N2`，T1/T2/T3 分别为 q/ori/cmp 的打包 token 数，Ko/Kc 为两组独立的索引容量。

| 参数 | dtype / shape / 作用 |
|---|---|
| `q` | FP16/BF16；BSND `[B,S1,N1,D]`，TND `[T1,N1,D]` |
| `ori_kv`, `cmp_kv` | 与 q 同 dtype，共享 K/V；BSND `[B,S2或S3,N2,D]`，TND `[T2或T3,N2,D]`，PA_BBND `[各自page_num,各自page_size,N2,D]` |
| `ori_sparse_indices`, `cmp_sparse_indices` | INT32；BSND `[B,S1,N2,Ko或Kc]`，TND `[T1,N2,Ko或Kc]`；batch 内逻辑索引，不是物理页号，索引顺序/重复/-1 行为由计算契约决定 |
| `ori_block_table`, `cmp_block_table` | INT32 `[B,各自max_pages_per_batch]`；各组独立逻辑页→物理页映射。PA 布局需要对应页表；非分页不传 |
| `cu_seqlens_q`, `cu_seqlens_ori_kv`, `cu_seqlens_cmp_kv` | INT32 `[B+1]`，含首项 0；TND 的 batch 存储边界。不能改成不含 0 的累计长度或 B 个普通长度 |
| `seqused_q`, `seqused_ori_kv`, `seqused_cmp_kv` | INT32 `[B]`，每 batch 的有效计算长度；不替代 TND 存储基址。传递到主算子和 metadata 的值必须一致，是否可省略按布局和目标实现判断 |
| `cmp_residual_kv` | INT32 `[B]`；旧压缩路径用其恢复 Lc*cmp_ratio+residual，不是 TopK 尾块长度；有 cmp、cmp_mask_mode=3 且 ratio≠1 时该实现要求传入 |
| `ori_topk_length`, `cmp_topk_length` | INT32；BSND `[B,S1,N2]`，TND `[T1,N2]`；逐 query/KV head 的实际槽数，与容量 Ko/Kc 分开。保留两个槽；arch22 当前 cmp_topk_length 非空受限，不能因 signature 是 Optional 就承诺支持任意非空值 |
| `sinks` | FP32 `[N1]`；每 query head 的额外 logit，只参与共同分母，不乘 scale。旧设备实现要求实际提供；数学无 sink 与传全 0 不等价 |
| `metadata` | INT32 `[1024]`；旧接口需传 metadata 算子的输出，不是配置字典或 tiling buffer；ABI 结构见[调度协议](metadata.md) |
| `softmax_scale` | Python float / C++ double，默认 1.0；不自动换成 1/√D |
| `cmp_ratio` | 整数，默认 1；压缩 token 与原 token 位置换算比例，不在算子内生成压缩 KV |
| `ori_mask_mode`, `cmp_mask_mode` | 整数，默认 0；枚举语义为 0 无 mask、3 右下因果、ori 的 4 窗口；合法组合还受 SoC/模式约束 |
| `ori_win_left`, `ori_win_right` | 整数，默认 -1；相对因果位置的窗口属性。保留原始值；不能把默认 -1 自动改为 0，具体语义需按标杆/设备契约区分 |
| `layout_q`, `layout_kv` | 默认 BSND；旧实现接受 `(BSND,BSND)`、`(TND,TND)`、`(BSND,PA_BBND)`、`(TND,PA_BBND)`。测试文件的别名如 PA_ND 必须在已核实等价后由适配层映射，不能发明新的设备枚举 |
| `topk_value_mode` | 整数，默认 1，该实现支持 1；不是 K，也不是 template_idx |
| `return_softmax_lse` | bool，默认 False；只决定 LSE 的计算/有效性，不改变 Python 返回 tuple 的长度 |

<a id="api--输出与包装层"></a>
### 输出与包装层

- `attn_out`：与 q 相同 shape/dtype，主接口返回值第 0 项。
- `softmax_lse`：FP32，开启时 BSND `[B,N2,S1,G]`、TND `[N2,T1,G]`；关闭时该版本真实 C++ wrapper 返回 shape `[0]` 的 tensor，仍占 tuple 第 1 项。
- metadata：INT32 `[1024]` tensor，由调用路径确定 NPU device；使用与主输入一致的 device。
- `None`、空 tensor、零值 tensor 不可混为一种缺省输入。Python Optional 表示签名允许传 None，host 校验仍可能在某模式要求该 tensor 非空。
- 该快照的 Python Fake/Meta 实现仍有关闭 LSE 时 `[]` 标量占位、取 KV head 仅访问 ori_kv 等与 C++ 路径不同的写法。生成 Fake/Meta 时按已冻结的真实输出契约保持一致，不能照抄这些差异；至少核对 rank/dtype/device/两个返回值。

<a id="api--4-两个算子的接线与模式"></a>
## 4. 两个算子的接线与模式

metadata **不接收 q/KV 数据、索引内容、页表、sinks、softmax_scale**。它接收长度、索引容量、head 信息和模式来生成任务表；这些缺席是原接口设计，不是需要自行补入的参数。

| metadata 属性 | 从同一次主算子调用推导 / 传递 |
|---|---|
| `num_heads_q`, `num_heads_kv`, `head_dim` | q 的 N1、存在的 KV 的 N2、q 的 D |
| `batch_size` | BSND 的 B；TND 由 cu_seqlens_q 的长度减 1 得到并按该 API 约定传递，不能把 T1 当 B |
| `max_seqlen_q`, `max_seqlen_ori_kv`, `max_seqlen_cmp_kv` | BSND 需要相应 tensor 的序列轴容量；TND/PA 按该版本长度推导规则准备，不能把 T2/T3 总长或 page_size 当每 batch 最大长度 |
| `ori_topk`, `cmp_topk` | 对应 sparse_indices 存在时取其最后一维 Ko/Kc；无对应稀疏路径取 0。两者不是同一个全局 K |
| `has_ori_kv`, `has_cmp_kv` | 对应 KV 实际是否存在；务必显式传，尤其 SWA 的 has_cmp_kv=False |
| 9 个可选 tensor | 三组 cu_seqlens、三组 seqused、cmp_residual_kv、两组 topk_length；与主接口保持相同内容/解释 |
| `cmp_ratio`、两组 mask、两组 window、两组 layout | 与主接口属性一致；metadata 不接受 softmax_scale、topk_value_mode、return_softmax_lse |

普通 arch22 / 910B 路径的接线（不含 ori_sparse 扩展）：

| 计算特征 | ori_kv / cmp_kv | cmp_sparse_indices / metadata.cmp_topk | cmp_ratio | ori_mask_mode / cmp_mask_mode |
|---|---|---|---|---|
| 仅窗口（SWA） | 有 / 无 | None / 0 | 1 | 4 / 3（arch22 main 校验要求；cmp 无计算） |
| 窗口+压缩前缀（标杆 CFA，旧 HCA） | 有 / 有 | None / 0 | 128 | 4 / 3 |
| 窗口+压缩索引（标杆 SCFA，旧 CSA） | 有 / 有 | 有 / Kc，1..8192 | 4 | 4 / 3 |

这三行 `ori_win_left/right=127/0`、`ori_topk=0`，D=512、N2=1，N1 取 1..128 中的 2 的幂；CSA Kc 不要求 512 整倍数。Kc=0 在前缀模式有意义，不是 CSA 的合法索引宽度。两条压缩路径准备 residual；PA 为每组存在的 KV 准备页表和实际长度，TND 准备相应前缀和。源码还有 ori_sparse 输入槽及其他 SoC 分支，不用这三行抹掉完整签名，也不把它们当作所有数学标杆的限制。

调用顺序：准备输入及模式属性 → 调 metadata → 将输出 tensor 传到主算子 → 消费两个返回值。在同一 stream 上维持先后依赖；跨 stream 须有完成事件。长度、K、模式等调度输入改变后重新生成 metadata。metadata 有自己的 executor/workspace；不能用主算子的 executor 替代它。

<a id="api--5-aclnn-两段式-c-接口"></a>
## 5. ACLNN 两段式 C 接口

以下声明已从历史接口参考内嵌到知识页，不作为编译头文件；`aclTensor`、`aclOpExecutor`、`aclnnStatus` 和 `aclrtStream` 仍由目标 SDK 提供。main GetWorkspaceSize 共 **32** 个参数，metadata GetWorkspaceSize 共 **30** 个；两个 Execute 都是 4 个参数。C 接口没有 Python 默认参数，所有位置必须正确占位。若调用者给出的目标接口版本不同，先记录差异并以目标包实际 ABI 冻结本轮契约。

```cpp
aclnnStatus aclnnSparseFlashMlaGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *oriKvOptional,
    const aclTensor *cmpKvOptional,
    const aclTensor *oriSparseIndicesOptional,
    const aclTensor *cmpSparseIndicesOptional,
    const aclTensor *oriBlockTableOptional,
    const aclTensor *cmpBlockTableOptional,
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensOriKvOptional,
    const aclTensor *cuSeqlensCmpKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedOriKvOptional,
    const aclTensor *sequsedCmpKvOptional,
    const aclTensor *cmpResidualKvOptional,
    const aclTensor *oriTopkLengthOptional,
    const aclTensor *cmpTopkLengthOptional,
    const aclTensor *sinksOptional,
    const aclTensor *metadataOptional,
    double softmaxScale,
    int64_t cmpRatio,
    int64_t oriMaskMode,
    int64_t cmpMaskMode,
    int64_t oriWinLeft,
    int64_t oriWinRight,
    char *layoutQOptional,
    char *layoutKvOptional,
    int64_t topkValueMode,
    bool returnSoftmaxLse,
    const aclTensor *attnOutOut,
    const aclTensor *softmaxLseOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnSparseFlashMla(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

__attribute__((visibility("default"))) aclnnStatus aclnnSparseFlashMlaMetadataGetWorkspaceSize(
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
    const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedOriKvOptional,
    const aclTensor *sequsedCmpKvOptional, const aclTensor *cmpResidualKvOptional,
    const aclTensor *oriTopkLengthOptional, const aclTensor *cmpTopkLengthOptional, int64_t numHeadsQ,
    int64_t numHeadsKv, int64_t headDim, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenOriKv,
    int64_t maxSeqlenCmpKv, int64_t oriTopk, int64_t cmpTopk, int64_t cmpRatio, int64_t oriMaskMode,
    int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight, const char *layoutQOptional,
    const char *layoutKvOptional, bool hasOriKv, bool hasCmpKv, const aclTensor *metaData, uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnSparseFlashMlaMetadata(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

历史 metadata 头文件对两个导出声明标了 default visibility；目标构建须验证真实导出属性和符号。注意 main 的 layout 是 `char *`，metadata 是 `const char *`；metadata 的 C 顺序先 9 个 tensor 再 numHeadsQ，而 Python 顺序先 3 个 head 属性，不能直接复制位置实参列表。`attnOutOut`、`softmaxLseOutOptional`、`metaData` 虽然类型写 `const aclTensor *`，仍是调用者预分配的输出描述符，不是输入数据。

分别执行 `GetWorkspaceSize → 分配相应 workspace（大小为0按接口要求处理）→ Execute`，检查每一步状态码，保持 tensor、workspace 和依赖的生命周期到设备使用完成。C++ 编译与链接应使用同一目标包的真实头文件和库；这里的声明不能替代实际导出符号验证。

<a id="api--6-host-到-device-的参数去向"></a>
## 6. host 到 device 的参数去向

内部device入口由当前实现设计，不需要继承某个旧kernel的模板开关或指针实参顺序。host到device必须保持以下依赖：

| 依赖 | 消费阶段与含义 |
|---|---|
| query、原始/压缩KV | QK与PV；分支选择决定实际使用哪一路KV |
| 两路稀疏索引、有效TopK长度 | 构造本query实际参与的KV集合，长度不能超过索引容量 |
| 两路页表 | 分别将原始/压缩KV逻辑位置转换为物理地址 |
| cuSeqlens、seqUsed、cmpResidual | batch边界、有效长度、压缩后的可见范围与因果位置 |
| sinks | 每head的softmax分母项，按实际输入读取 |
| metadata | 本次任务表，由配套metadata入口产出并由主核消费 |
| attentionOut、可选softmaxLse | 当前公开契约约定的输出布局与dtype |
| workspace、tiling/Params | 中间存储与host选择的执行策略；分别推导容量、生命周期和解释协议 |

scale/ratio/mask/window/layout/TopK 容量、shape/stride 等由 host 校验后编码到 Params/tiling/模板分支；没有同名 device 指针不等于可以丢弃。metadata 是运行任务表，tiling 是当前 kernel 的形状/策略配置，workspace 是中间存储，三者不可合并理解。原注册算子使用 `GetUserWorkspace(workspace)`；新 Catlass 直调若传独立 user workspace，就直接消费该指针，不套注册算子 workspace 偏移。

Catlass 内部可以改为结构化 Params；公开入口到每个必需计算依赖必须可追踪。首轮为了验证只实现一种布局/模式时，应保留完整公开契约并报告尚未实现的组合，不能将简化 launch 函数当成全功能接口完成交付。

<a id="api--7-生成后的接口验收"></a>
## 7. 生成后的接口验收

按本次选择的交付层检查，不用数值通过一条 case 代替接口验证。生成 `sparse_flash_mla` 的完成前提是两个算子的实现及对应构建产物齐全，并通过“本次生成的 metadata 入口 → 实际任务表 → 本次生成的主算子入口”的联调；只复用已有 metadata 实现不算配套算子生成完成：

1. 逐项比对入口签名：名称、位置/keyword-only、默认值、可选性、返回个数。复用本页实例时检查 main 28、metadata 27、C ABI 32/30/4/4；换成等价标杆时以其真实入口为准。
2. 从实际公开入口调用窗口/前缀/稀疏模式，覆盖 TND+TND、TND+PA、BSND+PA 以及契约要求的 BSND+BSND；metadata 使用真实输出贯通主算子，不能替换成常量零表。
3. 两种 LSE 开关都验证 tuple 长度和 tensor shape/dtype；TND 明确检查 N2/T1 轴序。非零 sink、非默认 scale、ragged 长度、不同 ori/cmp 页表与页大小应实际影响预期计算。
4. 稀疏容量至少含 1537 和 8192，并检查 metadata.cmp_topk 与 indices.shape[-1] 相等；可选参数在合法情况下分别测试 None/有值。对目标分支不支持的非空输入明确拒绝，不默默忽略。
5. 要求包互通时验证四个 ACLNN 符号；要求 Python 调用时验证两个 torch.ops schema、对应扩展导出与 tensor 全为 None 的 metadata 派发，并检查真实运行和 Fake/Meta 输出一致。仅 kernel 编译成功、直接调用 C++ helper、只产生输出 bin，都不能证明公开接口完整。
6. 接口和计算契约都通过后，按[计算知识](computation.md)测性能；需要的布局整理和 metadata 时间进入端到端口径。

<a id="api--接口验收"></a>
## 接口验收

逐参数检查公开schema、wrapper、host校验、tiling和device去向，不用签名存在代替实际支持。每个可选参数至少覆盖缺省、合法非缺省和非法组合；目标硬件不支持的组合明确拒绝。分别从metadata入口产表、主入口消费表，再执行成对调用，检查两项输出、关闭LSE、TND轴次序和dtype。

按[分阶段开发](development.md#stages)推进，并用[用例与验收](validation.md#coverage)实现测试adapter。公开接口的限制属于工程契约；等价数学表达输入的差异需显式映射，不用另一架构功能或未执行的helper推断当前入口支持能力。

<a id="swa-contract--代码模式"></a>

## 一般 SWA 合法域

<a id="swa-contract--算子算法"></a>
## 算子算法

记 batch b 内有效长度为 Lq、Lkv，query 的 batch 内序号为 t、head 为 h。令 `G=N1/N2`、KV head `n=floor(h/G)`；arch22 合法 N2=1，保留此映射以免混淆 head 和 token。右下对齐位置与可见列为：

\[
c=L_{kv}-L_q+t,\qquad J_t=[\max(0,c-127),\;\min(L_{kv},c+1)).
\]

左闭右开区间是包含当前因果位置的 **128 token 窗口**。`Lkv-Lq` 来自每个 batch 的有效长度，不是容量差或 TND 存储基址差。若 `t<Lq-Lkv`，golden 直接令 O、LSE 为 0，不能改成仅含 sink 的 `LSE=sink`。存储 padding `t>=Lq` 不计算，golden 输出为 0。

对有效行，实数数学为：

\[
z_j=\alpha\langle q_{b,t,h},kv_{b,j,n}\rangle,\quad
Z=e^{s_h}+\sum_{j\in J_t}e^{z_j},\quad
O=\frac{\sum_{j\in J_t}e^{z_j}kv_{b,j,n}}{Z},\quad LSE=\log Z.
\]

sink只影响分母，等价零value的额外logit；正式兼容域必须读取实际sinks tensor。数学上的无sink与`sink=0`不等价，但无sink并非本段正式域公式的默认输入；若实现诊断扩展，按上文单独定义和验证。

**流式舍入顺序是契约的一部分。** 用户 golden 默认 `RUN_MODE=1`，每行先设 `m=s_h, l=1, u=0`。每个逻辑 KV 分片递推：

\[
m'=\max(m,\max_j z_j),\quad a=e^{m-m'},\quad p_j=e^{z_j-m'},
\]
\[
l'=a l+\sum_j p_j\;\text{(FP32)},\qquad
u'=a u+\operatorname{PV}_{FP32}(\operatorname{cast}_{q\_dtype}(p),KV),
\]
\[
O=\operatorname{cast}_{q\_dtype}(u/l),\qquad LSE=m+\log(l+10^{-10}).
\]

**cast 的 P 是未归一化 `exp(z-m)`，不是 `p/l`。** row_sum 使用 cast 前的 FP32 值，PV 使用 cast 到 FP16/BF16 后的值进行 FP32 累积；sink 只初始化一次。不能改成先归一化再 cast，也不能使用 `RUN_MODE=0` 的 `sinks_softmax` 替代。golden 逻辑分片上限 512；当前 128 窗口每行最多一片。改变分片或让其他 query 的 score 参与本行 max 会改变数值路径。

<a id="swa-contract--scale-的公开类型与数值边界"></a>
### scale 的公开类型与数值边界

入口按收到的double值检查有限性；Python Meta、native桥、正式host和内部planner采用同一接受域。`isfinite(static_cast<float>(scale))`不是等价检查：有限double转换成FP32可能成为无穷，这不能作为额外参数拒绝条件。也不能自行要求scale为正、夹紧到FP32最大值，或改成默认scale。tiling中的表示与设备乘法顺序属于数值设计，应与用户golden中标量参与FP32计算的实际规则一致，不能以入口缩域代替实现。

可构造明确的有限输出反例：BSND、B=S1=S2=N1=N2=1、D512，Q全1、KV全-1、sink=0、scale=1e100。当前golden的FP32分数为负无穷，sink仍给出分母1，未归一化P为0，因此O、LSE均为有限的0；FP16/BF16都应实际验证。令scale为-1e100且Q全-1，仍可得到同类负无穷分数。有效Q长度为0时，即使scale极大，也必须按零行契约写O/LSE为0。这些例子说明拒绝所有FP32不可表示的scale会丢掉有明确有限结果的合法输入。

另一些极值组合会让golden本身产生NaN，例如FP32零分数乘溢出的scale，或softmax出现正无穷减正无穷。分别记录“接口接受”“golden是否有限”“设备是否有限”和适用的比较结果；不改golden、不静默忽略非有限元素，也不把两边均NaN记为普通精度PASS。至少覆盖两dtype、scale为0/非默认有限值/正负巨大有限值、有效行与零行，并单独验证NaN和±Inf参数在入口被拒绝。有限输出探针必须经过正式metadata→main及完整O/LSE检查；CPU黄金观察或planner接受测试不证明设备极值精度。

<a id="swa-contract--数据路径与存储层级"></a>
## 数据路径与存储层级

布局决定存储地址，窗口决定可见集合。以下地址以元素为单位，base 指向逻辑首元素；若从 allocation 基址计算，还需加 storage_offset。

| 布局 | shape 与地址 |
|---|---|
| Q BSND | `[B,S1,N1,D]`；`base+b*sB+t*sS+h*sN+d*sD`，连续 stride 为 `(S1*N1*D,N1*D,D,1)` |
| Q TND | `[T1,N1,D]`；`base+(cu_q[b]+t)*sT+h*sN+d*sD`，连续 stride 为 `(N1*D,D,1)` |
| KV BSND | `[B,S2,N2,D]`；`base+b*sB+j*sS+n*sN+d*sD` |
| KV TND | `[T2,N2,D]`；`base+(cu_kv[b]+j)*sT+n*sN+d*sD` |
| KV PA_BBND | `[page_count,page_size,N2,D]`；`page=table[b,floor(j/page_size)]`、`r=j%page_size`，地址为 `base+page*sPage+r*sS+n*sN+d*sD` |

布局对为 BSND+BSND、TND+TND、BSND+PA_BBND、TND+PA_BBND。逻辑页号与物理页号不同；无效表项不构成可读页。CPU golden 的逻辑 KV 不能当成 PA 物理页，物理页偶然递增也不能省略查表。

有效页号可在同batch或不同batch重复，物理页是只读输入，不要求独占。容量校验是每batch表项足以覆盖有效逻辑长度且被访问页号落入实际物理存储，不能用`sum(ceil(Lkv/pageSize)) <= physicalPageCount`替代。重复映射仍保留每个逻辑token的因果位置与softmax权重；跨batch共享前缀不改变各自的Q/KV有效长度和窗口。

| stride 类别 | 原arch22边界 | 本任务新实现要求 |
|---|---|---|
| 连续Q/KV | 四布局基础路径，Q按连续token/head寻址 | 全部支持，不能从某个优化tile限制合法长度 |
| PA KV页间padding | host解析stride0，Cube页拷贝消费实际页stride；页内token/head/D连续。页长为16倍数且在`[16,1024]` | 读取真实正页stride，支持非恒等页表、跨页窗口和有界尾块 |
| BSND KV batch间padding | Cube读取实际batch stride，但host获取stride数组时会按普通布局检查第0轴连续，存在入口与设备能力不一致 | 新生成的成对入口与主核贯通真实正batch stride；不得把旧入口冲突当作此次允许拒绝的理由 |
| TND KV token间padding | 原TND设备路径按连续`N2*D`寻址，没有一般token stride消费能力 | 按实际正token stride寻址，或正式wrapper显式整理并完整计时；这是本次输入ABI修正能力，不能称旧kernel已支持 |
| Q非连续、KV页内非连续、负stride、重叠view | 没有已确认的一般支持保证；某轴漏检不证明正确寻址 | 不自动纳入完整arch22域；依据冻结公开契约明确拒绝，或作为显式新增能力实现并验证。不得用这些扩展的不支持拒绝上面三种KV合法首轴stride |

正stride仍须满足非重叠存储、内层连续、实际分配范围和64位偏移可表示等要求，不表示任意正整数均合法。入口获得原生stride、内部descriptor传递、搬运ld/page stride、输出对象stride和测试保留stride必须形成闭环。旧host在部分SDK上只能用storage shape推导连续stride，这是能力退化，不可对非连续输入伪造连续值。

**singleton轴不要求规范stride数值。** 大小为1的轴只取索引0，它的正stride不会改变实际元素地址；例如KV的N2=1时，head stride可以是其他正值，不能因为不等于512就拒绝。判定允许的“首轴padding、内轴连续”时，从末轴向内累计期望连续跨度：始终验证stride为正；仅在该轴size>1时要求stride等于当前连续跨度；累计跨度乘以该轴size，不乘它的实际stride。首轴也遵循这个地址原则：只有首轴size>1时才要求stride至少覆盖一个内层切片，以避免相邻切片重叠。BSND的B=1、TND只有一个存储token、PA只有一个物理页及页表B=1时，不能因首轴stride小于内层跨度就拒绝；实际可访问地址仍须在底层存储内，正stride和溢出检查仍适用。不将singleton特例推广为任意多元素轴可不连续。空tensor与零有效长度另按各自契约判定。

Python Meta、正式host和诊断descriptor必须使用相同规则。Meta可以基于shape/stride推断并检查契约，不读取长度tensor内容；运行时值检查留给正式入口。用真实Torch Meta调用覆盖BSND、TND及两种PA组合中的N2=1、单token等singleton内轴，并覆盖B=1/单token/单物理页的singleton首轴，改变其正stride且保持其他轴和底层存储合法；非singleton D/token不连续、负/零stride、多元素首轴重叠作为拒绝对照。再用实际设备tensor保留这些stride进行正式计算，核对所有输出和输入未被改写。AST或stub测试只证明所执行谓词，不等价于已注册Meta dispatch或设备验证。

Meta/Fake入口不能仅调用`empty_like(q)`返回形状。先验证可由描述符确定的合同：layout配对、Q/KV rank与dtype、D=512、N2=1、合法head档、支持的stride，以及必需输入和可选tensor的dtype/shape关系，再分配O与LSE；关闭LSE只改变第二输出的形状，不能跳过输入验证。复用纯shape/stride校验函数可避免Meta与运行时包装逐渐分叉。Meta没有数据时不读取cu/used/page内容、不调用`.item()`、不创建真实NPU控制区；这部分值域检查在正式入口执行。使用实际注册的Meta与Fake dispatcher，分别验证LSE开关下的合法输入，以及错误rank、D、dtype、head、缺失必需输入和不支持的view；“合法形状能返回输出”不能代替拒绝语义一致。

**无Tensor的metadata还要验证factory分派。** 其所有Optional Tensor槽均为None时，只有标量属性，不能从输入推断Fake输出设备。已核对的PyTorch 2.6路径中，只有独立`impl(..., "Meta")`且返回空meta tensor仍可能进入`wrap_meta_outputs_with_default_device_logic`，在`_find_common_device`中因`common_device is None`失败；直接调用该Meta函数成功不能证明注册Fake调用成功。此失败属于抽象设备推断，不是metadata数值或真实设备分配失败。[^swa-contract-pytorch-fake-dispatch]

保持main 28参数、metadata 27参数的原schema，使用公开`torch.library.register_fake`为两个算子注册静态实现。该API提供Fake专用分派并同时注册Meta kernel；应替换原来的独立Meta注册，不能在已有Meta kernel上重复注册，也不能添加dummy Tensor或device参数来规避无Tensor调用。抽象函数复用描述符校验，只构造固定INT32 `[1024]`等无数据输出；禁止进入native桥、读取`.item()`、获取真实设备/stream或分配真实设备控制区。eager无Tensor调用的正式路径仍须独立保留和验证。

无Tensor metadata的抽象输出可使用`meta` placement；它表达形状/dtype，不承诺eager调用也在meta设备生成任务表。Fake main消费该抽象metadata时仍按Q的Fake设备构造O/LSE，不把metadata的抽象placement误判成真实混设备输入；真实执行的设备归属和任务表生产消费另由正式入口检查。

验证必须经过已注册的`torch.ops`，并让任何native入口访问立即报错：在`FakeTensorMode`中覆盖全部Tensor槽None的`B=0`与`B=2` metadata，检查FakeTensor类型、INT32 `[1024]`及约定的抽象placement；把`B=2`返回值交给合法Fake main，检查输出随Q的设备、shape/dtype及LSE开关，同时保留all-None错误head拒绝。`B=0`在此验证metadata factory，不据此放开main的B>0要求。再覆盖真实Meta/Fake的四布局与静态拒绝矩阵，记录原schema完整、无native调用和无真实设备执行；这些结果不替代NPU数值验收。

`ori_stride_padding/cmp_stride_padding/both_stride_padding` 只是 case 标签，须检查实际 shape、strides 和 storage_offset。SWA 没有 cmp_kv，cmp-only 标签可能无作用；B=1 的 batch stride padding 也可能不可观测。新接口应消费支持的真实 strides，或执行已记录的 wrapper 整理，不能伪造连续 stride。host 在 stride API 缺失时按 storage shape 推连续值，仅对真实连续数据安全。

本任务adapter的一个TND KV实例将token stride从512扩为2048。它检验上述一般地址能力，不能成为2048专用分支，也不能因旧TND路径的缺口拒绝该输入或在测试侧偷偷改连续；其他合法正token stride同样按运行时参数处理。

<a id="swa-contract--batchragged-与有效长度"></a>
### batch、ragged 与有效长度

下表规定本任务生成实现的长度语义；其中TND的seqused优先属于按golden修正的能力，不是原arch22 main已正确消费的保证。

| 布局 | 存储边界 | 有效计算长度 |
|---|---|---|
| BSND | 每 batch 固定容量 S1/S2 | seqused 有值取 `[b]`，否则取容量 |
| TND | INT32 `[B+1]` 的 cu，首0、非递减、末项与打包容量一致；基址取 `cu[b]` | golden 和 metadata 优先 seqused，缺省取 `cu[b+1]-cu[b]` |
| PA KV | 页表与页大小决定可寻址容量 | 必需 `seqused_ori_kv: INT32[B]`，不以页容量替代有效长度 |

seqused 均为非负 INT32 `[B]`，不超过所属存储区容量；TND 不超过各 batch 的 cu 差值。某 batch 的有效长度可为 0；B 必须大于 0，旧 main 还要求 TND 的 T1 大于 0，不将全空 tensor 自动当作合法调用。PA 有效长度必须有足够表项和合法物理页覆盖。

**TND原行为与本次修正：** 标杆和metadata优先seqused，缺省取cu差值；原arch22 SWA main在TND分支只用cu差值，即使入口接收seqused也不代表计算使用它。此次生成实现必须保存`storageBase=cu[b]`与`effectiveLength=seqused[b]`（缺省为cu差值），用前者定位、后者调度和计算右下对齐窗口，并让本轮metadata与main一致。Q和KV的effective/storage分离分别验证；未实现此要求应记录功能未完成，不能以旧缺口为由缩域。若调用者的选定用例没有更短seqused，须另加定向用例才能覆盖该分支。

**零输出要求：** 无效前缀、BSND padding及TND已分配但不参与计算的位置，O和启用的LSE都按golden为0。旧实现存在显式无效前缀清零路径，但常规初始化是否覆盖所有零工作/padding不能由此推断；本次实现须给每个应零元素指定写者，用NaN预填输出验证，不依赖分配器恰好清零。

<a id="swa-contract--分核策略与基本块切分"></a>
## 分核策略与基本块切分

按[metadata 协议](metadata.md)冻结生产/消费双方的 M/S2 块单位、无效前缀和任务末端。按 batch、head group、有效 query 产任务，按有效窗口估成本；decode、长短 prefill、ragged 可选不同合法调度，不按 case ID 选固定表。N1=128 的旧 metadata 有切 G 分支和核数限制，不能从 N1=4 的单位直接外推。

每个有效 query/head 恰好一个最终写者；无效行和 padding 置零另有覆盖。未实现 FD 时产合法 FA-only 表并验证主核实际消费任务区间。split-S2 须归并 m/l/u 且整行 sink 一次；普通 SplitK 求和错误。

<a id="swa-contract--流水排布同步关系与数值精度"></a>
## 流水排布、同步关系与数值精度

QK/PV 使用实际 CATLASS C++ Cube 组件，mask、exp、FP32 sum、sink 和最终归一化进入 Vector 阶段。物理尾列在 max/sum 前屏蔽为负无穷、PV 权重置0；补齐 head/query 行不写逻辑输出。记录 KV/P 的生产、消费、槽复用事件与释放顺序。布局整理、metadata 生成或任何 host attention 不得被移出端到端计时。

# 约束

<a id="api--约束"></a>

## 完整公开接口

公开参数、模式和有效域按冻结契约逐项实现；未支持的组合明确拒绝，不静默删参。

<a id="swa-contract--约束"></a>

## 一般 SWA 合法域

<a id="swa-contract--cann正式包的构建闭环"></a>
## CANN正式包的构建闭环

复用目标SDK的通用customize构建模板时，保留其公共工具依赖。9.2.0-beta.2 的 `tools/op_project_templates/ascendc/customize/cmake/util` 含指向 `ascendc/common/util` 的相对符号链接；搬到独立工程时应在原SDK位置解引用复制（如`cp -aL`），或完整保留正确目录关系。只复制链接会使`ascendc_gen_options.py`等工具消失，后续向dangling symlink拷贝也会失败。不得以旧工程autogen目录补洞。

按[公共 CANN 构建工具准备](development.md#sdk-build)在生成工程内暂存目标 SDK 的工具，取得已解引用的 `cmake/` 和哈希清单。该步骤只暂存公共CMake工具：有效customize文件优先，common/util补齐缺失工具；它不生成kernel、host或注册实现。生成工程自行接入该cmake目录，仍需提供本轮自己的CMakeLists、op_host和op_kernel。发现未知断链或越出SDK的链接时先修正SDK路径，不能拿以前的生成工程补齐。每次正式候选使用独立stage/build/install目录，避免新源码或安装覆盖让先前计时无法对应到二进制。

**host的C++标准以真实编译命令为准。** customize模板的`cmake/intf.cmake`可在`intf_pub`的`INTERFACE_COMPILE_OPTIONS`中显式传播`-std=c++11`；仅设置`CMAKE_CXX_STANDARD=17`或目标`CXX_STANDARD`不能证明最终生效，冲突的显式`-std=`还可能按命令顺序覆盖。读取本轮暂存模板及消费它的目标属性，开启verbose构建或检查`compile_commands.json`，核对op_host注册、op_api及共享planner各翻译单元的最终标准。当前opdev头中`enable_if_t`、较新constexpr等报错时，先检查是否实际按C++11编译，不删SDK声明或降级自己的公共类型来掩盖配置。

需要C++17时，在本工程引入`intf_pub`之后，针对已确认的那一项传递选项删除冲突或替换为C++17，保留其他接口选项；也可采用效果等价且实际命令可验证的目标配置。只改本轮工程，不修改全局SDK，不用无差别文本替换影响设备OPC参数或其他工具链。AICPU交叉编译、Torch扩展和AI Core编译分别检查各自标准、ABI与支持参数，不能因host成功就推定所有目标一致。重新配置/构建受影响目标并保留真实命令和结果；配置阶段的属性打印不能替代已执行的编译证据。

自定义编译选项应先核对SDK模板解析器。此版本同一算子的多次 `add_ops_compile_options` 会使后行覆盖前行的选项；把include路径、auto-sync配置、架构宏和可选诊断宏合并到同一个调用。OPC的 `--op_debug_config` 允许项与底层编译器宏不是一层：9.2.0-beta.2接受oom、dump_cce、dump_bin、dump_loc、ccec_O0、ccec_g、check_flag、sanitizer，不能据二进制feature字段自行添加printf。诊断构建与用于性能验收的release构建分别保存。

host普通构建与设备`binary`目标分别执行，核对生成和安装的每个非空设备object一致。`op_build registration.so output_dir --aicpu`在此SDK产生`aicpu_kernel.ini`，不是最终注册JSON；使用同SDK `tools/op_project_templates/op_project_tmpl/cmake/util/aicpu_parser_ini.py`的实际CLI转换，再核对op name、tensor类型、RunCpuKernel、kernelSo及engine等字段。不要猜JSON格式，也不要把缺产物当作成功。四个公开ACLNN符号、内部注册、AICPU配置、设备object和Python schema必须对应同一轮。

独立AICPU库跟随当前公共cpukernel模板选择匹配的hcc、头文件、context/protobuf库和ABI。9.2.0-beta.2普通非MINRC分支是目标hcc加ABI1，头来自OPP的`aicpu_kernel/inc`，库优先来自`${ASCEND_AICPU_PATH}/opp/built-in/op_impl/aicpu/aicpu_kernel/lib/Ascend`的`libcpu_kernels_context.a`与`libascend_protobuf.a`；SoC分目录、MINRC及context回退按实际模板判断。保留对应`--whole-archive`边界、`-Bsymbolic`和protobuf的`--exclude-libs`设置，不将`lib64/libaicpu_context.a`加`libbase_ascend_protobuf.a`视为普遍要求。其他组合须用实际符号、头文件及运行库证明ABI一致；详细路径和检查见[动态工程与正式部署](development.md#engineering--设备编译与正式注册的贯通检查)。

链接成功、同架构host新进程`RTLD_NOW`加载并查找`RunCpuKernel`、真实custom scheduler上的正式metadata小用例是三份不同证据。保留ELF依赖/未定义符号、库哈希和真实dlerror；未确认提供者的securec符号不能视为已解析。host加载失败能定位该host的依赖缺口，但设备scheduler环境可能不同，不能据此单独认定设备失败根因；host成功同样不能替代Compute到达、同步和完整输出验证。

该SDK的producer/converter还有一个可定位的格式差异：`op_build --aicpu`会输出无前缀的`workspaceSize=0`，而其INI converter逐项按`分组.字段`解析，`CUSTAICPUKernel`校验要求`opInfo.workspaceSize`。遇到这个已确认组合时保留原INI与哈希，生成独立转换输入，仅把这一个键改为`opInfo.workspaceSize=0`，其余字段原样保留，再调用官方converter。若出现未知无前缀键、重复字段、非预期workspace，返回错误并核对SDK，不能笼统删除无法解析行或手写一个固定JSON冒充注册结果。转换成功后仍需真实AICPU执行，成功产包不是运行成功的证据。

真实验证按成本递增排查：编译后实际dlopen，Python schema注册与Meta形状检查，独立metadata执行及主核联调，然后原ATK双标杆和正式device计时。CPU/Meta检查不能替代NPU执行。空tensor合法性应由是否存在零维度决定，不能用各轴shape*stride的和判断是否有元素；非空数据必须有地址。LSE和PA table同样检查合法物理stride与非重叠存储，不能只校验Q/KV。

| 项目 | arch22兼容边界及本任务交付要求 |
|---|---|
| dtype | q、ori_kv、设备 O 同为 FP16/BF16，FP32 累积和 LSE |
| head | `N1 ∈ {1,2,4,8,16,32,64,128}`、`N2=1`、`G=N1/N2`；通用数学映射不授权 N2>1 |
| D | Q/KV 都为512，全部D参与QK/PV，无独立RoPE槽 |
| 长度 | 实际batch/shape/长度不限选定用例，不允许私有表造成小batch上限；TND seqused优先按标杆修正，存储边界独立保存；整数和可寻址容量需校验 |
| mask/window | ori mask4、左127、右0；任意窗口、mask0/3、任意N1属于其他架构或数学能力 |
| 模式 | 仅原始窗口；CSA/HCA/ori sparse 在 main 与 metadata 各自拒绝 |
| PA | 两种Q布局；表与有效长度必需，page_size为16倍数且 `[16,1024]`，页内连续 |
| stride | PA实际页stride属于已有能力；BSND batch stride入口接线与TND token stride为本次明确贯通/修正要求；不据此承诺任意Q或页内view |
| sink | 正式域必需非空FP32 `[N1]`；诊断可空sink是单独扩展 |
| 输出 | O与q同形状同dtype；开启LSE为BSND `[B,N2,S1,G]`、TND `[N2,T1,G]`；关闭 `[0]` |

golden 的 TND O helper 分配 FP32，LSE helper 最后为 `[T1,N2,G]`，关闭时 None。adapter 显式转换设备结果到冻结比较形式并记录转换，不改 golden；不能用 numel 相同代替 shape/dtype/轴次序检查。完整槽位不代表支持任意非空参数，默认组合也不一定合法。

# 失败表现

<a id="api--失败表现"></a>

## 完整公开接口

metadata 先报错只能证明该入口拒绝；主入口须独立验证。错误 shape、槽复用或长度单位会造成漏任务、越界或数值错误。

<a id="swa-contract--失败表现"></a>

## 一般 SWA 合法域

因果位置误用 t 会破坏 decode 和 Q/KV 长度不等的 batch；TND 把 seqused 当存储步长会串 batch。P先归一化再cast、漏sink、用cast后P求分母会造成系统性误差。只支持N1=4或B=1、按case ID白名单、忽略metadata固定launch、压缩请求返回SWA结果、任意stride按连续访问，都不能以选定用例通过宣称完成泛化。

# 验证方法

<a id="api--验证方法"></a>

## 完整公开接口

按[验证清单](validation.md#coverage)分别调用新 metadata、新主算子及组合调用；核对所有输出、任务覆盖和实际运行日志。

<a id="api--分核策略与基本块切分"></a>
## 分核策略与基本块切分

任务坐标与块单位见[metadata 协议](metadata.md)；query/head 与 S2 切片不能用物理容量替代有效长度。

<a id="api--数据路径与存储层级"></a>
## 数据路径与存储层级

完整参数去向覆盖 wrapper、host、metadata、tiling 和 device；两组 KV 的长度、页表和页大小分别传递。

<a id="api--流水排布同步关系与数值精度"></a>
## 流水排布、同步关系与数值精度

metadata 就绪后主算子才能消费；每次长度/模式/调度输入改变重新生成，不能复用旧任务表。device 生命周期与舍入顺序见[开发知识](development.md#stages)。

[^api-computation-contract]: [参数对应的计算语义](computation.md)。

<a id="swa-contract--验证方法"></a>

## 一般 SWA 合法域

当前验证集和 ID 范围以调用者输入为准。按 dtype、Q/KV 布局、head 比、batch 数、query/KV 长度、有效长度和窗口宽度构造覆盖矩阵；ID 只作报告定位，不得进入合法性判断或 kernel 分支。多 batch ragged 输入必须让 `seqused_q` 和 KV 有效长度实际影响各 batch 地址与窗口；标为某模式的 case 还须核对其非空输入是否真正走了该模式，不能只看名称。

一般SWA还须提供定向覆盖：两种PA及非恒等页表、真实页stride padding；BSND实际batch stride、TND实际token stride；TND多batch且Q/KV的seqused分别小于存储长度；128窗口边界、Q/KV长度不等、Lq>Lkv无效前缀；零有效长度batch和整批零工作；非默认scale、非零sink、LSE开关和轴序。TND有效长度/stride和零行修正能力单独标记，不冒称旧kernel已通过。分别从两个入口验证CSA/HCA、N2>1、非幂次N1、错误dtype/长度/页表及域外stride的拒绝；metadata先失败不能代替main拒绝证据。cmp辅助槽另测合法非空不改变SWA输出；正式main验证缺失/空sink拒绝，诊断无sink扩展结果单列。

精度遵守用户双标杆和既定容差；性能同输入、同计时边界逐例测量，默认最低比值0.8（手写基线耗时 / 生成实现耗时），每条达标，不能用平均数遮蔽慢例。通过指定子集后仍逐项报告一般域已验证与未验证维度，不能宣称三模式全功能。

本页已经给出隔离生成所需的窗口、数值路径、布局地址、实际边界和修正要求。验收以本轮冻结的公开契约、golden和真实运行结果为准；不要求生成者读取外部手写源码，不以接口漏检或未验证设备分支扩张支持声明。

[^swa-contract-interface-contract]: [两个公开接口及声明](interface.md#api)。
[^swa-contract-computation-contract]: [流式舍入与位置语义](computation.md)。
[^swa-contract-metadata-contract]: [任务表生产消费协议](metadata.md)。
[^swa-contract-diagnostic-contract]: [SWA诊断ABI与完整输出验证](validation.md#suite)；诊断预留状态不等于正式兼容入口已经支持。
[^swa-contract-pytorch-fake-dispatch]: 诊断脚本通过标准 PyTorch Fake/Meta 注册路径推断 device、shape 和 dtype；具体内部函数随目标 PyTorch 版本变化，必须在目标环境运行同类探针，不把内部实现当作自定义注册 API。材料边界见 `workflow.md#materials`。
