# Ascend950 (dav-3510) 实测平台坑全集（20 个，每个都有真实事故）

> 全部来自 FA-SageAttention 开发实践，按类别组织。
> 每条：症状 → 根因 → 解法。写代码前通读一遍可省数天。

## A. 编译器 / API 层

### P1 AIV 上标量 GM store 静默失败
- **症状**: `(__gm__ uint32_t*)pr[4] = x` 在 AIC 上正常落盘，在 AIV 上**完全不落盘**（无报错）
- **解法**: AIV 侧 GM 进度标记必须 `Duplicate(UB 值) + DataCopyPad ≥1KB`

### P2 count-API Muls/Adds 的 dst 寻址
- **症状**: `Muls(tensor + off, ...)` 编译错或解析到错误 overload
- **解法**: 一律 `Muls(tensor[off], ...)`（operator[]）

### P3 StoreUnAlign 需左值
- **症状**: `StoreUnAlign(compAddr + rowOff, ...)` 报 no matching function
- **解法**: 先存局部 `__ubuf__ float* cAddr = compAddr + rowOff;`

### P4 RegTensor 赋值
- **症状**: `t0 = f0`（RegTensor 拷贝）编译失败
- **解法**: 分支写法避免（if/else 各自完整路径）

### P5 device 标量 fp 除法无法 lower
- **症状**: `error in backend: Do not know how to split...`
- **解法**: SageDivScr（8 元素 UB scratch 走向量 Div，count≥8、禁原地）

### P6 `__simd_vf__` 参数限制
- **症状**: 参数必须 `__ubuf__` 地址空间；不能传 `__gm__*`
- **解法**: 需 GM 参数的 helper 用普通 `CATLASS_DEVICE` 函数

### P7 `__vector__` kernel 无 sub-block
- **症状**: merge kernel 中 `GetSubBlockIdx()` 恒返回 0 → 行 64-127 无人算、输出半零
- **解法**: `sub = GetBlockIdx() & 1; task = GetBlockIdx() >> 1; 步长 = GetBlockNum() >> 1`

### P8 MulDstAdd 语义
- **症状**: `MulDstAdd(dst, a, b)` 以为是 `dst += a·b`，实际是 **`dst = a·dst + b`**（乘的是累加器！）
- **解法**: 累加语义用 `Mul(t, wk, p); Add(acc, acc, t);`

### P9 事件方向（HardEvent 方向性）
- **症状**: `SetFlag<V_MTE3>; WaitFlag<V_MTE3>` 想等 MTE3 拷贝落盘——方向反了（V 置位、MTE3 等，不等拷贝完成）
- **解法**: 等 MTE3 完成用 `PipeBarrier<PIPE_ALL>`（或 SetFlag<MTE3_V>+异地 Wait）。
  **kernel 退出不保证 MTE3 排空**，partial 写后必须 fence

### P10 小 DataCopyPad 写不可靠
- **症状**: <1KB 的 UB→GM 写偶发不落盘；32B 必失败
- **解法**: ≥1KB；scale/μ 用 ×8 冗余凑尺寸。512B GM→UB 读可靠

## B. L0 / catlass 组件层

### P11 FAIQK 与 FAIPV 的 Shape 语义相反
- **症状**: `BlockMmadTla` 的 L1TileShape 在两个 block 里读法不同——FAIQK 读 (M,N,K)，FAIPV 读 (M,K,N)
- **根因**: 传错导致静态槽距 ≠ 运行时实际 → L0A 槽溢出踩踏在飞 mmad
- **症状实例**: PV `Shape<128,64,128>` 使 K_static=64 而运行时 blockK=128 → L0A 槽距 16KB < 实需 32KB → 第二次 A-load 覆盖第一个在飞 mmad → 输出 cols 0-63 恒坏 ≈V 列均值
- **解法**: 见不变量 I1——垫 QK K_static=256 使两 block 槽距都是 32KB

### P12 L0A/L0B 槽距错位竞态
- **症状**: P=1 对、P≥2 挂死或错数
- **根因**: 两 block 的 L0A/L0B 槽都从 0 布局且共享事件 id {0..3}，槽距不等时事件计数错配
- **解法**: I1；L0C 由 cursor 天然分区不冲突

### P13 kernel 内 tiling 逐字段拷贝
- **症状**: 新增 tiling 字段后部分 case 输出全零（设备读到栈垃圾 → P=0）
- **根因**: kernel 里 `td` 是从 GM 逐字段拷贝的，漏拷新字段
- **解法**: 加字段必须同步加拷贝行

### P14 CMake 头文件追踪
- **症状**: 改了 .asc include 的头，重编后行为不变（跑的旧二进制）
- **解法**: 头文件加入 `add_executable` 的 sources

## C. 流水 / 同步层

### P15 AIC 上 PipeBarrier<PIPE_ALL>
- **症状**: 必挂死（AIC 管集合不含 V/MTE3）
- **解法**: 删；AIC 只用事件

### P16 游标 post-increment
- **症状**: 步序错乱 [1,2,0]；t0 单块因 wrap 恰好回 0 而"假通过"
- **解法**: c.jCur（I4）；split-K 下 isFirst 判据用 jStart 非 0

### P17 状态链槽位
- **症状**: 2 块 0.75、3 块 0.99、4 块 0.89 的渐进劣化
- **根因**: step s 读了同槽（未更新）的 lastMax/sumL
- **解法**: I5——读 (s-1)%N 槽写 s%N 槽（lmR/lmW/sumR/sumW 分离）

### P18 MM2 双 set surplus
- **症状**: 4 块起 O 的 rows 32-63 损坏
- **根因**: RS 同时 set MM2{7,8} 两 id → surplus 让下下个 PV 提前通过等待，fixpipe 覆盖还在被读的 oUb
- **解法**: I3——只 set 本步槽 `MM2_INTRA_EVT[m2]`

### P19 pL1 槽复用无 handshake
- **症状**: 步数 > 槽数时挂死
- **根因**: SM(s+N) 覆盖 pL1[m3] 时 AIC PV(s) 仍在读 L1
- **解法**: I8——PV_CONSUMED flag (id 12-15) + P-Cast 前 WaitFlag<MTE3_V>(m3Prev)

## D. 数据 / 精度层

### P20 ×8 冗余 scale 索引漏乘
- **症状**: 1 块 0.9999、2 块 0.999、4 块 0.885 渐进劣化（拟合显示全部用了块 0 的 scale）
- **根因**: `gmSK.GetValue(idx)` 读的是 float 下标，×8 冗余布局必须 `idx*8`
- **解法**: I2；取证法见 fa-sageattention-precision.md §dScale 拟合

## E. 其他

### P21 D=64 行距硬编码
- **症状**: D=64 崩溃 507015
- **根因**: RS 相位 `r*128` 硬编码，而 D=64 时 PV fixpipe 写 oUb 是紧凑 [64,64]（行距 64）；
  且 2D GM 写 (128B blockLen + srcStride) 组合触发设备错误
- **解法**: Rescale/DivL 加 D 参数（行距=D），Cast 只转 SH·D，GM 写 1D 连续

### P22 merge 连续小拷贝 + 复用 event id
- **症状**: merge 读到的 m/l 部分为零/垃圾
- **解法**: works 连续 → 一次性整块拷贝 + PipeBarrier

### P23 bf16 生成器
- **症状**: gen_data 声称 bf16 实际生成 fp16 数据
- **解法**: fp32 >> 16 截断路径

### P24 GQA Q̂ 加载 stride 反了
- **症状**: GQA 下崩溃或错数
- **解法**: srcStride=(H-1)·D（GM 侧跳行），dstStride=0（UB 紧凑）

### P25 因果 mask 反向
- **症状**: causal 输出错
- **根因**: `Sub(t0, c0, thr)` 方向反了（有效位置被 mask）；golden 的 causal 约定还有 (sk-sq) 偏移
- **解法**: `Sub(t0, thr, c0)` + 阈值 lim-1

### P26 host JSON bool 解析
- **症状**: `atoll("true")` = 0 → causal 永远关
- **解法**: 前瞻匹配 "true"/"false"

### P27 stale o.bin 欺骗（元坑，最危险）
- **症状**: kernel 挂死被 timeout 杀掉后 o.bin 不更新 → 验证脚本读旧文件 → 每次"结果"一模一样的假象 → 不知情连续发射挂死 kernel → 污染全部 8 张卡
- **解法**: **每次运行必查 rc + o.bin mtime**（srun.sh 固化）；timeout 后绝不解读任何输出

### P28 python 批量编辑脚本静默失败
- **症状**: replace 字符串不匹配时不报错不写入；assert 中断导致文件整体未写、实验代码残留
- **解法**: 每次编辑后 grep 验证标记；**编辑后验括号平衡**（一次失衡导致 string.h 假错 + stale binary 连锁）

---

## F. 新版 catlass 适配坑（2026-08 重建上板时踩到，全部已修）

> 背景：源码丢失后按本 skill 重建，但重建环境装的是 **gitcode 最新 catlass**（比原开发时的旧版新）。
> 新版 catlass 的 `BlockMmadTla` 行为变了，重建代码沿用旧设计会直接挂死/编译错。以下 6 条是重建→22/22 PASS 修的全部。

### P29 CATLASS_ARCH 未设 → catlass ascend950 头被 #if 跳过
- **症状**: 一上来连锁编译错（`ScaleGranularity` 未声明、`PackedTileCopyTla` 模板缺省不匹配）
- **根因**: catlass 头按 `CATLASS_ARCH==3510` 门控 ascend950 专用实现（含 `ScaleGranularity` 枚举）；CMakeLists 没设，catlass 默认 2201(AtlasA2) → ascend950 分支全跳过
- **解法**: CMakeLists 加 `target_compile_definitions(sage_fa PRIVATE CATLASS_ARCH=3510)`（dav-3510↔3510 映射）

### P30 aclrtMalloc 的 void* 不能转 kernel 形参 GM_ADDR
- **症状**: `cannot initialize a parameter of type '__gm__ uint8_t *' with an lvalue of type 'void *'`
- **根因**: `aclrtMalloc` 给 `void*`，kernel 形参 `GM_ADDR`=`__gm__ uint8_t*`；ASC 不做 `void*→__gm__ uint8_t*` 隐式转换（cann-samples 用 `uint8_t* dX` 裸指针）
- **解法**: host 的 `DevMem.p` 用 `uint8_t*`（非 `void*`）+ `aclrtMalloc((void**)&p, ...)` 显式 cast

### P31 AIC 重复 wait MM1/MM2（新版 catlass 内部已 wait）★最隐蔽
- **症状**: kernel 挂死（卡5/卡2 均超时，看门狗 ~27s 释放）
- **根因**: 新版 catlass `BlockMmadTla`（FAIQK:223 / FAIPV:197）**内部自己 `CrossCoreWaitFlag(MM1_RES_INTRA_EVENT[taskId])` / `MM2_RES_INTRA_EVENT[taskId]`**，flag ID 与本算子全同（{9,10}/{7,8}）；重建代码 AIC 又手动 wait 同一组 → 双重 wait：第一次消费 flag、catlass 内部第二次 wait 永久阻塞
- **解法**: 删 AIC 侧手动 `CrossCoreWaitFlag MM1/MM2`（保留 AIV 侧 set，catlass 等的就是它）
- **判别**: grep 本仓 catlass include 的 FAIQK/FAIPV 实现源，确认内部已含 `CrossCoreWaitFlag(MM*_RES_INTRA_EVENT[...])`（组件库源可直接核验）

### P32 PV_CONSUMED 用循环前预置（冷 set 不传播）
- **症状**: 首次 SM 等 PV_CONSUMED 永卡 → 看门狗
- **根因**: 循环前在 AIC Init 上 `CrossCoreSetFlag<PIPE_FIX>` 预置 PV_CONSUMED，但此时 FIX 管道无活动，**预置的 set 不传播**（与 MM1/MM2 在 PIPE_V 上的预置不同——后者 V 管道同样冷却生效，故 MM1/MM2 bootstrap 可用，PV_CONSUMED 在 FIX 上不可）
- **解法**: 改 **step 守卫**：pL1 槽首次填充(step<槽数)跳过 wait、仅复用(step≥槽数)才 wait（仿 MTE3_V 的 `if it>=3` 守卫），无需 bootstrap
- **注意**: 单 buffer P-scale 与 4-slot pL1 不冲突——SM(s) it=s+1 写、PV(s) it=s+2 读、SM(s+4) it=s+5 才覆盖，PV 读在覆盖前 3 步，且已有 PV_CONSUMED 守卫同 SM 步保护

### P33 merge 紧凑化只 copy sub0（sub1 漏合 → inf）
- **症状**: case0(causal 1024, splitK=4) qblock0 的 **sub1(rows 64-127) 全 inf**，其余 sub1 finite 但少合 shard
- **根因**: `Muls(wSlab[nValid*512], wSlab[k*512], 1.0f, 128)` 只 copy 128 floats（sub0 的 m|l@offset0），**漏 sub1 的 m|l@offset+256**；SageMergeW 对 sub1 读未紧凑化的 slot（含空 shard 的 garbage po）→ qblock0 仅 shard3 非空、sub1 读到空 shard → l*=0 → DivL inf
- **取证**: 看 o.bin 行模式——sub0 全 finite、sub1 全 inf → 直指 merge 的 sub 分支
- **解法**: copy **512 floats 全 slot**（sub0 m|l + pad + sub1 m|l + pad 都搬）

### P34 bf16 输入被当 fp16 读（host 未转）
- **症状**: bf16 case cos≈0.28（非 inf，是错值）
- **根因**: host 把 bf16 输入的位直接喂 kernel（kernel 量化只认 `half`），bf16 位模式被误读成 fp16 → 量化全错
- **解法**: host 读入后若 bf16，**先 bf16→fp16(RNE)** 再喂 kernel（同 numpy `astype(float16)`）；emu/verify 的 bf16→fp32 路径不变

### P35 "看门狗慢" 误判（实为每进程设备开销，非卡顿 flag）
- **症状**: 单次 srun 7~27s，疑似卡顿 flag/看门狗
- **根因**: 实测 host 逐阶段计时——`aclrtSetDevice`≈1.8s + `aclrtResetDevice`/`aclFinalize`≈4.6s 的**每进程设备初始化/销毁开销**（驱动+固件加载），kernel 本身仅 ~4ms；之前 27s 是卡被多次挂死后 setDevice/reset 变慢
- **解法**: 用 `--bench`（进程内预热+多 iter 均摊）测真实 kernel 用时；持久进程（ACLNN/torch_npu）后该开销消失
- **教训**: 计时前先分阶段定位，勿把设备开销当 kernel 卡顿

### P36 FP8 MX PV 集成墙（path B 未完成，留作后续）
- **症状**: 把 PV 换 `MmadFAIPVMx` 编译错（`is_one_of<half, __fp8e4m3...>` 断言失败）
- **根因**: catlass FP8 cube 要求 **L1 内 P/V 就是 fp8(E4M3)**，不是 half；`CopyGmToL1B` 不自动 half→fp8 转换（DataCopy no match）。即"pL1 仍 half、catlass load 时转"的假设错
- **正解**: 用 ≤100 行探针 kernel 实测 catlass FP8 MX 组件的**真实 fp8 数据流**（`EpilogueAscend950FASoftmax` 怎么产 fp8 P+pScale、V 怎么 half→fp8 load、确切类型），照实测复刻，**勿靠 build error 盲猜类型**
- **教训**: catlass MX FP8 API 复杂，build-error 猜类型低效；先通读能跑的参考实现再写
