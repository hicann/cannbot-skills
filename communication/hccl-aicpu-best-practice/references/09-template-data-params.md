# TemplateDataParams 关键参数的量化说明

> 来源：用户提供的 `ZZ/skill-dev/template-data-params.md`，于 2026-09-09 迁入。源码路径相对用户确认的 `${HCCL_ROOT}`；使用时核对当前 executor 的赋值分支与 template 消费方式。本文是本子 Skill 五参数语义的知识真源。

阅读顺序：§1–§4 理解公式、落位与单位，§5 看非对称布局，§6 量化检查，§7 适用边界，§8 Ring 漏项案例。

`TemplateDataParams`连接executor与template：executor描述本阶段的数据布局，template按该布局定位输入和输出数据。其中，4个stride参数描述输入、输出各自的二维切片寻址步长，`repeatNum`描述repeat维度的长度。

本文根据当前仓库的赋值和地址计算整理，适用于采用规则切片寻址的template。特殊template、中转buffer和变长数据的适用边界见第7节。

结构体定义见template_utils.h（`${HCCL_ROOT}/src/ops/op_common/algorithm/template/template_utils.h`）中的`TemplateDataParams`。

## 1. 统一地址公式

定义以下符号：

| 符号 | 含义 |
| --- | --- |
| $i$ | 切片对应的算法rank索引，通常为当前子通信域内的索引，不一定等于全局rank |
| $r$ | repeat索引，$0 \le r < R$ |
| $B_{\mathrm{in}}$ | 输入buffer地址加`inBuffBaseOff` |
| $B_{\mathrm{out}}$ | 输出buffer地址加`outBuffBaseOff` |
| $L$ | 本次处理的切片字节数，通常为`sliceSize`；尾块按实际长度处理 |

输入和输出切片的起始地址分别为：

$$
A_{\mathrm{in}}(r,i)
=B_{\mathrm{in}}+iS_{\mathrm{in}}+rT_{\mathrm{in}}
$$

$$
A_{\mathrm{out}}(r,i)
=B_{\mathrm{out}}+iS_{\mathrm{out}}+rT_{\mathrm{out}}
$$

对应的代码表达式为：

```cpp
inputOffset = inBuffBaseOff
    + i * inputSliceStride
    + r * inputRepeatStride;

outputOffset = outBuffBaseOff
    + i * outputSliceStride
    + r * outputRepeatStride;
```

这描述的是切片位置，不能单独表达算子的归约关系。例如ReduceScatter还需要把不同来源rank的对应数据归约到目标切片。

实现依据见AllGather Mesh（`${HCCL_ROOT}/src/ops/all_gather/algorithm/template/aicpu/ins_temp_all_gather_mesh_1D.cc`）的`LocalDataCopy`和`PostLocalCopy`。

### 1.1 输出落位先看 executor

`B_out` 是 `outputPtr + outBuffBaseOff`，不保证是用户输出。以下是当前 `${HCCL_ROOT}/src/ops/scatter/algorithm/executor/` 的已核对赋值；换版本须重读对应函数。`HCCL_BUFFER` 段把 `outputPtr` 指向 CCL buffer，最终写用户输出的是 `OUTPUT` 段。

| executor / 段 | `outBuffType` | `outBuffBaseOff` 与 `hcclBuffBaseOff` | 赋值位置 |
| --- | --- | --- | --- |
| `ins_v2_scatter_sole_executor.cc` / sole | `OUTPUT` | 分别为 `processedDataCount * dataTypeSize_` / `0` | `:212, :242–243` |
| `ins_v2_scatter_parallel_executor.cc` / intra0 → inter0 → inter1 → intra1 | `HCCL_BUFFER` → `OUTPUT` → `HCCL_BUFFER` → `OUTPUT` | 两个 `HCCL_BUFFER` 段均同为 `scratchOffset`；两个 `OUTPUT` 段分别为 `dataOffset` / `scratchOffset` | `:742–746, :772–776, :802–806, :832–836` |
| `ins_v2_scatter_sequence_executor.cc` / intra → inter | `HCCL_BUFFER` → `OUTPUT` | intra 两者均为 `0`；inter 分别为 `processedDataCount * dataTypeSize_` / `0` | `:187, :202, :244–245, :270–271` |
| `scatter_sequence_aicpu_executor_3level.cc` / level0 → level1 → level2 | 通常 `HCCL_BUFFER` → 条件选择 `HCCL_BUFFER`/`OUTPUT` → `OUTPUT`；跳层时 level0 也可为 `OUTPUT` | level0 两者均为 `0`；level1 为 `HCCL_BUFFER` 时两者均为 `0` | `:248–249, :278–279, :359–397` |
| `ins_v2_scatter_omnipipe_executor.cc`、`ins_v2_scatter_omnipipe_2d_executor.cc` / `In2HCCLBuff`、`HCCLBuff2HCCLBuff` | `HCCL_BUFFER` | `outBuffBaseOff` 从 `stepSliceInfo` 继承；不能仅凭类型断言它等于 `hcclBuffBaseOff` | 前者 `:375–382, :406–411`；后者 `:350–357, :381–386` |

做本 rank 切片的 scratch→out 回搬前，先核对所有绑定 executor 的段表、两侧指针、base offset 和 repeat/rank stride。若 `outBuffType == HCCL_BUFFER`，再核对上一跳是否已直写目标槽位：已在位时可跳过自拷贝，错位时仍需搬移。把分流和在位/错位的理由写在拷贝分支附近。`ins_temp_scatter_ring.cc:233–253` 的 `KeepChunk` 是“上一跳 `RecvWrite` 已写到 `hcclBuffBaseOff + rpt * sliceSize`，故跳过”的具体例子，不是所有 `HCCL_BUFFER` 段的通用早退。

## 2. 5个参数的精确定义

| 参数 | 符号 | 量化定义 | 单位 |
| --- | --- | --- | --- |
| `inputSliceStride` | $S_{\mathrm{in}}$ | 固定repeat，输入侧相邻算法rank切片的起始地址差 | 字节 |
| `outputSliceStride` | $S_{\mathrm{out}}$ | 固定repeat，输出侧相邻算法rank切片的起始地址差 | 字节 |
| `inputRepeatStride` | $T_{\mathrm{in}}$ | 固定算法rank，输入侧相邻repeat切片的起始地址差 | 字节 |
| `outputRepeatStride` | $T_{\mathrm{out}}$ | 固定算法rank，输出侧相邻repeat切片的起始地址差 | 字节 |
| `repeatNum` | $R$ | 本次template调用处理的repeat数量，索引为$0,\ldots,R-1$ | 次 |

对输入或输出任意一侧，在相邻索引存在时：

$$
S=A(r,i+1)-A(r,i)
$$

$$
T=A(r+1,i)-A(r,i)
$$

**stride是起点到起点的距离，包含切片自身长度。** 例如切片长度64B，下一个切片起点相距256B，则stride为256B，中间空隙为192B。

`repeatNum`表示本次调用内的逻辑重复组数，不一定等于executor的外层分块循环次数，也不表示通信算法的step数。

当`repeatNum=1`时，唯一的repeat索引为0，两个RepeatStride不影响上述地址公式，通常设为0。结构体中`repeatNum`默认值为0，常规循环中不能把默认值当作一次执行。

## 3. Slice与repeat是两个独立维度

对输入或输出任意一侧，相对基址的偏移如下：

| repeat / slice | $i=0$ | $i=1$ | $i=2$ |
| --- | --- | --- | --- |
| $r=0$ | 0 | $S$ | $2S$ |
| $r=1$ | $T$ | $T+S$ | $T+2S$ |
| $r=2$ | $2T$ | $2T+S$ | $2T+2S$ |

假设单片长度为$L$，以下两种紧凑布局都合法：

| 物理布局 | SliceStride | RepeatStride |
| --- | --- | --- |
| `[repeat][slice][片内数据]`，每个repeat有$N$片 | $L$ | $NL$ |
| `[slice][repeat][片内数据]`，每个slice有$R$个repeat | $RL$ | $L$ |

因此，不存在通用约束`RepeatStride = rankSize * SliceStride`，也不要求`RepeatStride >= SliceStride`。参数应由实际物理布局推导。

仓内AllGather Parallel（`${HCCL_ROOT}/src/ops/all_gather/algorithm/executor/ins_v2_all_gather_parallel_executor.cc`）的`GenTemplateAlgParamsInter0`使用：

$$
S_{\mathrm{in}}=S_{\mathrm{out}}=DN_0,\qquad
T_{\mathrm{in}}=T_{\mathrm{out}}=D,\qquad R=N_0
$$

其中，$D$为每个rank的完整数据块字节数，$N_0$为Level0的rank数量。该布局中repeat沿较小步长移动。

## 4. 常见算子的参数取值

设$D$为每个rank的完整逻辑数据块大小：AllGather中指每rank输入块大小，ReduceScatter中指每rank输出块大小。设$L$为本轮实际处理的切片字节数，可能有$L<D$。

| 场景 | inputSliceStride | outputSliceStride | inputRepeatStride | outputRepeatStride | repeatNum |
| --- | --- | --- | --- | --- | --- |
| 普通单层AllGather，本地输入仅有自己的块 | 0 | $D$ | 0 | 0 | 1 |
| 普通单层ReduceScatter，本地输出仅保留自己的块 | $D$ | 0 | 0 | 0 | 1 |

### 4.1 AllGather

本地算法rank为$i$时，其输入切片起始地址为：

$$
A_{\mathrm{in}}(0,i)=B_{\mathrm{in}}
$$

在输出中属于该rank的位置为：

$$
A_{\mathrm{out}}(0,i)=B_{\mathrm{out}}+iD
$$

`inputSliceStride=0`表示本地输入地址不随算法rank索引变化，不表示不读取输入。

赋值依据见AllGather Sole（`${HCCL_ROOT}/src/ops/all_gather/algorithm/executor/ins_v2_all_gather_sole_executor.cc`）中的`OrchestrateLoop`。

### 4.2 ReduceScatter

输入中属于目标算法rank $i$的切片起始地址为：

$$
A_{\mathrm{in}}(0,i)=B_{\mathrm{in}}+iD
$$

该rank的最终结果写入自身本地输出：

$$
A_{\mathrm{out}}(0,i)=B_{\mathrm{out}}
$$

`outputSliceStride=0`表示结果写到各进程自身的本地输出基址，不表示所有rank向同一物理地址写入。

赋值依据见ReduceScatter Sole（`${HCCL_ROOT}/src/ops/reduce_scatter/algorithm/executor/ins_v2_reduce_scatter_sole_executor.cc`）。

### 4.3 分块长度不等于布局步长

本轮只处理$L$字节，并不意味着rank之间的stride变成$L$。当用户buffer仍按完整块$D$排列时，stride必须保持$D$，本轮在块内的位置由base offset体现。

典型关系为：

$$
\mathrm{sliceSize}=\mathrm{currDataCount}\times\mathrm{dataTypeSize}
$$

$$
\mathrm{baseOffset}=\mathrm{processedDataCount}\times\mathrm{dataTypeSize}
$$

切片长度决定本轮处理多少数据，stride决定不同逻辑切片在buffer中的位置。

## 5. 5个参数同时参与的示例

AllGather Sequence AICPU（`${HCCL_ROOT}/src/ops/all_gather/algorithm/executor/ins_v2_all_gather_sequence_executor_aicpu.cc`）的`GenIntraTemplateParams`在普通中转buffer路径下使用：

$$
S_{\mathrm{in}}=0,\quad S_{\mathrm{out}}=D,\quad
T_{\mathrm{in}}=L,\quad T_{\mathrm{out}}=DN_0,\quad R=N_1
$$

其中，$N_0$为Level0的rank数量，$N_1$为Level1的rank数量。CCU或对称内存分支会调整输入布局，不适用这里的输入参数取值。

假设：

- $N_0=4$，$N_1=3$。
- 每个rank完整块大小$D=1024$B。
- 本轮每片处理$L=256$B。
- 当前算法rank为$i=2$。

参数为：

```cpp
inputSliceStride   = 0;
outputSliceStride  = 1024;
inputRepeatStride  = 256;
outputRepeatStride = 4096;
repeatNum          = 3;
```

相对于各自基址，当前rank的切片位置为：

| repeat | 输入偏移 | 输出偏移 | 处理长度 |
| --- | --- | --- | --- |
| 0 | 0B | 2048B | 256B |
| 1 | 256B | 6144B | 256B |
| 2 | 512B | 10240B | 256B |

输入中3片连续存放；输出中分别落到3个组内的rank 2位置。

若误将`outputRepeatStride`设为$N_0L=1024$B，后两片将写到3072B和4096B。即使没有越界，数据归属也已错误，会导致结果校验失败。

## 6. 正确性的量化检查

### 6.1 与期望布局一致

先根据算子语义、rank映射和buffer布局，独立推导每个实际访问切片的期望偏移$E(r,i)$，再检查：

$$
E(r,i)=b+iS+rT
$$

其中，$b$是对应的base offset。输入和输出分别检查，不能只检查两侧stride是否相等。

若相邻索引的地址差不是常数，单靠4个stride就不能完整表达该布局，需要拆分调用或使用额外的位移信息。

### 6.2 访问范围合法

对每个实际访问的切片，应满足：

$$
0\le b+iS+rT
$$

$$
b+iS+rT+L_{r,i}\le C
$$

其中，$C$是对应buffer容量，$L_{r,i}$是包含尾块处理后的实际切片长度。实现检查时还需避免乘法、加法发生整数溢出。

### 6.3 独立输出不重叠

对同一buffer中必须分别保存的两个不同输出切片，若长度均为$L$，应满足：

$$
\left|(i-i')S_{\mathrm{out}}+(r-r')T_{\mathrm{out}}\right|\ge L
$$

对于变长切片，直接检查两个半开区间是否相交：

$$
E(r,i),E(r,i)+L_{r,i})
$$

此条件只针对必须独立保存的输出；有意归约到同一位置的访问不能直接套用。输入数据允许按算法语义复用。

### 6.4 覆盖次数与单位正确

- `repeatNum`等于本次调用需要处理的逻辑重复组数，避免少处理或多处理。
- 4个stride的单位均为字节，元素数应乘`dataTypeSize`后再用于字节寻址。
- 按完整元素处理时，地址偏移和切片长度应满足相应元素对齐要求。
- 只枚举template实际访问的索引。例如AllGather的本地输入通常只读取自身算法rank对应的切片，不能假定每个本地buffer都访问完整的二维索引集合。
- 原地操作还需检查读写时序，确保输出不会提前覆盖尚未消费的输入；地址范围正确并不足以保证这一点。

## 7. 适用边界

这5个参数描述寻址布局与覆盖范围，还需结合`sliceSize`、`tailSize`、base offset、算法rank映射和归约语义才能判断template整体正确性。

### 7.1 中转buffer可能使用独立布局

[AllGather Mesh（`${HCCL_ROOT}/src/ops/all_gather/algorithm/template/aicpu/ins_temp_all_gather_mesh_1D.cc`）的部分中转buffer路径使用紧凑布局：

$$
S_{\mathrm{scratch}}=\mathrm{sliceSize},\qquad
T_{\mathrm{scratch}}=\mathrm{sliceSize}\times\mathrm{templateRankSize}
$$

因此，不能把输入或输出stride机械套到所有内部buffer。

### 7.2 特殊template可能复用字段

ReduceScatter OmniPipe DPU（`${HCCL_ROOT}/src/ops/reduce_scatter/algorithm/template/aicpu/ins_temp_reduce_scatter_omnipipe_mesh_1d_dpu.cc`）中的拷贝循环使用`repeatNum`控制次数，却按`inputSliceStride`和`outputSliceStride`递增地址：

```cpp
for (auto i = 0; i < tempAlgParams.repeatNum; ++i) {
    // 省略DataSlice构造和LocalCopy调用，仅展示偏移关系。
    inputOffset = inBuffBaseOff + i * inputSliceStride;
    outputOffset = outBuffBaseOff + i * outputSliceStride;
}
```

这种实现需要按实际消费方式理解字段，不能只凭名称套用第1节的二维模型。

### 7.3 变长和分step布局需要额外信息

`TemplateDataParams`还包含`allRankSliceSize`、`allRankDispls`、`sendCounts`、`recvCounts`和`stepSliceInfo`等字段。使用这些信息的template可能具有变长或分step布局，应结合其具体地址计算逐一核对。

这5个参数错误通常会导致错位读取、结果覆盖或数据遗漏，表现为结果或精度校验失败；它们本身不描述浮点舍入误差，也不能单独保证归约数值精度。

## 8. Ring 漏掉输入 rank 项：PR 2901

来源：[HCCL PR 2901](https://gitcode.com/cann/hccl/pull/2901/discuss)，
[当前 diff](https://gitcode.com/cann/hccl/pull/2901.diff)，2026-09-09 查阅。
用户说明旧实现漏掉下式最后一项；PR diff 展示的是新增 Ring 文件的修复后版本，不是旧错行到新行的差异。
目标文件：`${HCCL_ROOT}/src/ops/all_gather/algorithm/template/aicpu/ins_temp_all_gather_ring.cc`，`CopyLocalData`。

```cpp
const u64 inputOffset = tempAlgParams.buffInfo.inBuffBaseOff
    + rpt * tempAlgParams.inputRepeatStride
    + tempAlgParams.inputSliceStride * myAlgRank;
```

当 Ring 是先 Mesh 后 Ring 的 inter 阶段时，inputPtr 可以指向上一阶段的 OUTPUT。
AllGather Parallel executor 的 `GenTemplateAlgParamsInter0` 使用
`ISS=OSS=D*N0, IRS=ORS=D, RPT=N0`，不能沿用普通单层 AllGather 的 `ISS=0` 假设。

量化例：`D=1024B, N0=4, myAlgRank=2, base=0`。根据 `[server][local-rank][D字节]`
布局独立推导，本 server 2 的数据从第 8 个 D 块开始，四个 repeat 应读取第 8/9/10/11 块：

| rpt | 正确输入偏移 | 漏 rank 项的偏移 |
|---|---|---|
| 0 | 8192 | 0 |
| 1 | 9216 | 1024 |
| 2 | 10240 | 2048 |
| 3 | 11264 | 3072 |

漏项导致读取其他算法 rank 的数据；后续 Ring 传输即使正确，也在传播错误输入。
`ISS=0` 或 `myAlgRank=0` 会掩盖错误。因此通用 Ring 至少要验证两者均非零的路径，
并令输入内容区分 rank、repeat 和片内位置；仅检查 rank 0 或全相同输入不能提供足够证据。
Ring 的 `step` 决定 sendIdx/recvIdx，`rpt` 选择逻辑组，二者分别列出，不能互换。

## 9. 可执行的量化样例

spec 第 5 章记录参数赋值来源、算法 rank 映射，并放入以下格式的 `layout-check` JSON。
`formula` 是拟实现/实际代码的字节偏移（含 base，但不含 buffer 指针值）；
`expected` 是从物理布局独立推导的数值。`vars` 中使用非负整数，公式只允许变量、整数、`+ - *` 和括号。
`length`、`capacity`、`element_size` 均以字节计；容量与 base 必须属于同一实际 buffer。

```layout-check
{
  "cases": [
    {"name": "ring inter rank2 repeat0", "formula": "b + r*IRS + i*ISS", "vars": {"b": 0, "r": 0, "IRS": 1024, "i": 2, "ISS": 4096}, "expected": 8192, "length": 256, "capacity": 12288, "element_size": 4},
    {"name": "ring inter rank2 repeat1", "formula": "b + r*IRS + i*ISS", "vars": {"b": 0, "r": 1, "IRS": 1024, "i": 2, "ISS": 4096}, "expected": 9216, "length": 256, "capacity": 12288, "element_size": 4},
    {"name": "ring inter rank2 repeat2", "formula": "b + r*IRS + i*ISS", "vars": {"b": 0, "r": 2, "IRS": 1024, "i": 2, "ISS": 4096}, "expected": 10240, "length": 256, "capacity": 12288, "element_size": 4},
    {"name": "ring inter rank2 repeat3", "formula": "b + r*IRS + i*ISS", "vars": {"b": 0, "r": 3, "IRS": 1024, "i": 2, "ISS": 4096}, "expected": 11264, "length": 256, "capacity": 12288, "element_size": 4}
  ]
}
```

设计阶段运行 `python3 "$SKILL_DIR/../hccl-aicpu-design/scripts/check_layout.py" path/to/spec.md`；设计 Skill 的 `check_spec.py` 也调用该检查。
把上例 formula 的 `+ i*ISS` 删除会失败，即使所有旧偏移仍在容量内。

每个实际适用的布局分别填写 input/output/scratch 样例，覆盖零 stride、非零 base、`L<D`、
`RepeatStride<SliceStride`、多 repeat、尾片以及 userRank 不等于算法 rank 的场景。
不支持的场景注明原因；量化样例中的 rank/repeat 必须属于该阶段实际访问的索引集合。

检查器只验证有限样例的偏移等式、无符号 64 位算术范围、容量和元素对齐，
不自动证明期望值正确、输出切片互不重叠、完整覆盖、rank 映射、原地读写时序或 C++ 与公式相同。
这些仍按 §6 人工复核，并由 Plugin 按共享验证契约执行双轨验收；特殊字段消费方式按 §7 写实际公式。
