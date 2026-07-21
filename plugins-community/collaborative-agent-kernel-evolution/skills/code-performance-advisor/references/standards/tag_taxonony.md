标签体系 (Taxonomy) 简洁版

标签体系分为三层：Domain（硬门禁）→ Symptom（瓶颈）→ Context（精细化）。

### 1. Domain 标签（硬门禁 / Gating）
* **U.*（执行单元）**
    * `U.Cube`: 主要在 Cube 单元执行。
    * `U.Vector`: 主要在 Vector 单元执行。
    * `U.Mix`: 混合执行（Cube + Vector）。
    * `U.DMA`: 纯搬运算子。
    * `U.CPU`: CPU 侧执行算子。
* **O.*（算子族）**
    * `O.MatMul`: 矩阵乘法族。
    * `O.Conv`: 卷积族。
    * `O.Pooling`: 池化族。
    * `O.Norm`: 归一化族（LayerNorm/DeepNorm 等）。
    * `O.Loss`: 损失函数族。
    * `O.Optim`: 优化器族（AdamW/EMA 等）。
    * `O.Compare`: 比较类（Equal/LessEqual 等）。
    * `O.Condition`: 条件判断类（IsFinite/IsInf 等）。
    * `O.Broadcast`: 广播类（add_bias_broadcast/where_broadcast）。
    * `O.Index`: 索引/采样类（gather/scatter/index_*）。
    * `O.Resize`: 插值/缩放类（resize/upsample/grid_sample）。
    * `O.Mask`: Mask/Select 类（MaskedSelect/Tril/Triu）。
    * `O.Sort`: 排序/TopK 类。
    * `O.TensorCreation`: 张量创建类（Arange/Eye）。
    * `O.TensorMove`: 张量搬移/序列类（ReverseSequence/FeedsRepeat）。
    * `O.Foreach`: 列表/批处理算子（Foreach*）。
    * `O.Activation`: 激活函数族（Gelu/Swish/SwiGLU）。
    * `O.Reduce`: 归约族。
    * `O.Elementwise`: 逐元素族。
    * `O.Attention`: 注意力相关族。
    * `O.Reorder`: 变换/重排类。
    * `O.Fused`: 融合算子族。
    * `O.DataCopy`: 纯搬运/拷贝类。
    * `O.General`: 兜底类型。
* **T.*（数据类型）**
    * `T.FP16` / `T.BF16` / `T.FP32` / `T.FP64` / `T.INT8` / `T.INT32` / `T.INT64` / `T.BOOL`

### 2. Symptom 标签（瓶颈原语 / Matching）
* **计算瓶颈**
    * `S.LowComputeUtil`: 通用低利用率。
    * `S.LowCubeUtil`: Cube 低利用率。
    * `S.LowVecUtil`: Vector 低利用率。
    * `S.ScalarBound`: 标量瓶颈。
* **搬运瓶颈**
    * `S.MemoryBound`: 内存受限。
    * `S.TransferDominated`: 搬运主导。
    * `S.DmaOverhead`: DMA 开销高。
    * `S.MteBusy`: MTE 单元繁忙（拷贝流水压力高）。
    * `S.CacheMiss`: 缓存缺失。
    * `S.LowHbmUtil`: HBM 带宽利用率低。
    * `S.StridePenalty`: Stride 访存惩罚。
    * `S.LocalCopyRedundant`: Local/UB 内部冗余搬运。
* **流水瓶颈**
    * `S.PipeStall`: 流水线停顿。
    * `S.HighScalarRatio`: 标量占比高。
    * `S.IcacheMiss`: 指令缓存 miss。

### 3. Context 标签（排序与解释 / Refining）
* **规模/形状**
    * `C.K.Small`: K 维度较小（阈值由配置定义）。
    * `C.K.Large`: K 维度较大（阈值由配置定义）。
    * `C.MN.Small`: M/N 都较小（阈值由配置定义）。
    * `C.Batch.Small`: Batch 较小（阈值由配置定义）。
    * `C.Tile.Small`: Tile 较小（阈值由配置定义）。
    * `C.Reduce.LastDim`: 归约发生在最后一维。
    * `C.Sequence.Ragged`: 变长序列相关算子。
* **布局**
    * `C.Layout.NCHW`: NCHW 布局。
    * `C.Layout.NHWC`: NHWC 布局。
    * `C.Layout.FRACTAL`: Fractal 布局。
    * `C.Layout.General`: 无法识别时的通用布局。
* **架构**
    * `C.Arch.910B`: Ascend 910B。
    * `C.Arch.910B2`: Ascend 910B2。
    * `C.Arch.910D`: Ascend 910D。
    * `C.Arch.910C`: Ascend 910C。

* **内存与对齐（Ascend 相关）**
    * `C.Align.256B`: 256B 对齐约束。
    * `C.L0.Capacity`: L0 缓冲容量约束。
    * `C.L1.Capacity`: L1 缓冲容量约束。
    * `C.UB.Capacity`: UB 缓冲容量约束。
    * `C.L2.Shared`: L2Cache 共享约束。

