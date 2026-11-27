# Winograd Tiling 模板：变换域求导与驻留规划

本模板将 [scene §2.8](../../../scene.md#28-winograd-等价模型f23-变换域求导) 的 3×3 权重梯度改写为 16 个变换坐标上的矩阵乘加。Host 选择 tile 档和驻留方向；Kernel 统一计算运行时通道块。分支记号见 [language §1.2](../../../language.md#12-winograd-模板记号变换域求导)，执行与同步见 [winograd-implementation](../../../../../adapters/ascendc/filter/winograd-implementation.md)。

## 1 准入与选择顺序

模板按注册编号从小到大尝试：**DLoad(8) → Winograd(9) → MN_StreamK(10)**。准入不满足时继续下一个；规格校验或已选模板求解失败不等于继续回退。

| 条件 | 当前约定 |
|---|---|
| 格式与类型 | x、dedy、dedw 全为 NCDHW、FP32；hf32 是 FP32 的计算模式，由 enable_hf32/hf32Flag 表达 |
| 平台能力 | 通过当前实现的架构与平台门禁；hf32 或 kd=3 使用额外平台准入，不能从容量或支持 FP32 推定 |
| 核 | kd 为 1 或 3，kh=kw=3 |
| stride/dilation | D/H/W 全为 1；因此合法 output_padding 为 0 |
| D 轴 pad | kd=1 时前后均为 0；kd=3 时前后均为 1 |
| H/W pad | 前后分别对称，padUp<kh、padLeft<kw；实现还检查 pad≤8，组合后实际为 0、1、2 |
| 分组 | groups=1 |
| kd=3 空间尺寸 | di/hi/wi/dout/ho/wo 均不小于 6 |
| 归约规模 | `kAxis=batch*dout*ceilDiv(ho,2)*ceilDiv(wo,2)`；FP32 上限 512000，hf32 上限 256000 |

这些阈值是当前选择规则，不是片上容量或性能保证。平台能力记录绑定目标 SDK、实现门禁及 MIX 配比，硬件容量另从 platformInfo 取得。`pad<kernel` 保证每个核深度存在有效贡献，**不消除 D 轴边缘的无效步，也不保证每个 tile 块均含有效采样**。

## 2 tile 档与运行时通道块

`tileH=ceilDiv(ho,2)`、`tileW=ceilDiv(wo,2)` 表示输出 2×2 tile 的行列数；`th/tw` 表示一次处理多少行列的 tile。

| tile 档 | FP32 的 th×tw | winogradTilingFlag |
|---|---|---|
| B16H8W8_B32H4W8 | 4×8 | 1 |
| B16H4W16_B32H2W16 | 2×16 | 2 |

分别比较 `clusters=ceilDiv(tileH,th)*ceilDiv(tileW,tw)`，选块数较少者；相等时选 2×16。clusters 是覆盖块数，边缘块的有效 tile 另行截断。

Kernel 在入口调用一次 `CalRtSingleShapeBlock`，随后以同一份 `rtTiling` 传给 BlockIterator 和 AivFwdTransformer：

- singleShapeCin 固定为编译档 64。
- 按两轴均为 64 估算块数 T。T≤启动核数/2 或 T>启动核数时，singleShapeCout 取 64。
- 其余情况按 **16 → 32 → 48** 尝试，取首个满足 `ceilDiv(cout,Cout档)*ceilDiv(cin,64)≤启动核数` 的值；都不满足则取 64。

L1/L0/UB 按编译最大档预算；本轮实际 m/n 使用运行时块及尾长。运行时缩小 Cout 不等于缩小物理槽跨度。

## 3 驻留方向与水位线

ci>co 时驻留 dedy，否则驻留 x。驻留侧变换结果写 GM 的 nk1c1k0c0 区，AIC 经 MTE2 取入；流式侧由 AIV 经 MTE3 直接写 L1。

```text
驻留 x: x -> 变换 -> GM 驻留区 -> L1                 --+
          dedy -> 变换 ---------------> L1           --+--> 16 点乘加

驻留 dedy: dedy -> 变换 -> GM 驻留区 -> L1           --+
          x -> 变换 ---------------> L1              --+--> 16 点乘加
```

驻留 x 时主块沿 Cout 方向遍历，驻留 dedy 时沿 Cin 方向遍历。`residentCBound>watermarkResidentC` 触发补充变换；当前 kd 内水位递增，**切换 kd 后清零**。kd=3 的尾轮按 kd 分子轮，各子轮重新建立驻留内容。

## 4 K 坐标、分段与尾轮

区分三种长度：

| 长度 | 定义与用途 |
|---|---|
| kAxis | batch×dout×tileH×tileW；数学 tile 总数，Host 检查上限 |
| cuttableK | `dout*batch*ceilDiv(tileH,th)`；可分核的 K 单位数，一个单位走完全部 W 块 |
| tiles.elements | 当前 tile 块的有效高×宽；一次单点 MMAD 的 k |

```text
固定 kd：cuttableK 坐标 = (od, batchIdx, hBlock)
                                  |
                                  v
                         wBlock = 0 ... wSteps-1
                                  |
                  id = od + kd - padFront
                                  |
                有效：变换 / 搬运 / 16 点乘加
                无效：跳过数据与乘加，保留必要握手
```

`SegmentTileKIterator` 每走一个 tile 块计数一次；每段最多 `32768/(th*tw)` 个**块迭代步**，完整块对应最多 32768 个 tile。段内在 L0C 累加；段末 Fixpipe→UB→逆变换，首段覆写结果，后续段原子累加。不能将段间合并写成继续保留同一份 L0C。

cuttableK>1 只表示允许尾轮切 K。尾块存在时，`kGroups=min(启动核数/每个kd的尾块数,cuttableK)`；各组处理相同尾块的不同 K 区间。分组部分和先写 tailGm，`TailInterleaveWrite` 再按 Cout 划分最终写回责任，每个负责者汇总全部 kGroups。

kd=3 的尾轮每子轮采用“写部分和 → 全核屏障 → 聚合写回 → 全核屏障”；两处屏障分别保护可读性和下一子轮的槽复用。无效 D 段与闲置核的同步规则见适配层。

## 5 workspace 地址与预算

设备按以下顺序使用用户 workspace，系统保留段由 SDK 单独处理：

```text
+---------------+---------------------------------------+-------------------+
| 区域          | 设备起点（字节）                      | 长度（字节）      |
+---------------+---------------------------------------+-------------------+
| transX        | 0                                     | Xbytes            |
| transDy       | Xbytes                                | DYbytes           |
| nk1c1k0c0     | Xbytes + DYbytes                      | residentBytes     |
| tailGm        | Xbytes + DYbytes + residentBytes      | tailBytes         |
+---------------+---------------------------------------+-------------------+
```

- `cBytesX=alignUp(cin*sizeof(SrcT),32)`，dedy 同理。
- `Xbytes=batch*cBytesX*di*hi*wi`；`DYbytes=batch*cBytesDy*dout*ho*wo`。
- `k1=ceilDiv(tileH,th)*ceilDiv(tileW,tw)`，驻留区 `NK1C1K0C0::Shape.k0=16*th*tw`；后者不同于 TilingData 中的物理分形 k0=8。
- `residentBytes=batch*dout*min(cBytesX,cBytesDy)*k1*k0`。
- `tailBytes=coreNum*64*64*3*3*sizeof(float)`；64 为编译通道块上限，不是平台容量。

四区前缀和必须与设备指针偏移一致。驻留区的 n 索引为 `od*batch+batchIdx`，不是原始 batch；转置区的 D 长度分别使用 di、dout。字节数乘积先用足够宽的整数计算。

## 6 key、字段与启动

key 为 `GET_TPL_TILING_KEY(TPL_STREAM_K,0,0,winogradTilingFlag,residentFlag)`。Winograd 由非零 winogradTilingFlag 选中，**没有独立的模板编号位取值**；residentFlag=0 驻留 x、=1 驻留 dedy。

Host 发布原始几何、属性、hf32Flag、coreNum 和共享结构；本模板不消费 MN_StreamK 的 base*/step*/Bound 求解结果。tile 档和驻留方向经 key 传递，rtTiling 在设备入口一次计算。blockDim 使用本模板的 coreNum，MIX 类型与 AIV 配比按目标构建元数据绑定。

Winograd 在公共 MN_StreamK 清零前分发：由每个输出块首段覆写建立基值，后续段累加；调用前无需复用 MN_StreamK 的整输出预清零协议。

## 7 产出校验

- [ ] 自然选择顺序、准入、kAxis 上限与目标能力门禁一致。
- [ ] tile 档比较的是 clusters；rtTiling 只调整 Cout，所有消费者使用同一份值。
- [ ] cuttableK、块迭代步、单次 MMAD 的 k 分别记录；kd 与 od 的角色明确。
- [ ] 水位线按 kd 重建；无效 D 步保持必要握手，全无效段的逆变换输入为零。
- [ ] workspace 区域顺序、D 长度、对齐与前缀和一致。
- [ ] 段间合并、尾轮聚合和两处 kd 屏障完整，输出只覆盖有效 co/ci/kd/kh/kw。
- [ ] Golden 使用直接求和；FP32/hf32 模式、源码版本和设备结果分别记录。
