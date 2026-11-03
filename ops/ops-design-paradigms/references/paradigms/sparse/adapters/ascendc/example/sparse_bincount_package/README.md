# SparseBincount

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                     |     √    |
| <term>Atlas A3系列产品</term>    |    √     |
| <term>Atlas A2系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：统计稀疏张量中每个值（bin index）的出现次数或加权累加和。支持 1D 和多维模式，支持 binary_output 模式。

- 计算公式：

$$
\text{1D: } output[v] += w_i \quad (\text{if } v < size)
$$

$$
\text{Multi-dim: } output[batch][v] += w_i \quad (\text{if } v < size \text{ and } batch < dense\_shape[0])
$$

## 参数说明

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>稀疏张量的索引，2D tensor，shape (N, R)。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>输入</td>
      <td>稀疏张量的值（bin index），1D tensor，shape (N,)。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dense_shape</td>
      <td>输入</td>
      <td>稀疏张量的稠密形状，1D tensor，shape (R,)。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>size</td>
      <td>输入</td>
      <td>bin 的数量（标量 tensor），shape (1,)。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>输入</td>
      <td>权重，1D tensor，shape (N,) 或 (0,)。为空时每个出现计为1。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>binary_output</td>
      <td>属性</td>
      <td>是否二值输出。True时出现过的bin设为1，False时输出计数或加权累加。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>统计结果。1D时shape=[size]，多维时shape=[dense_shape[0], size]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- indices 为 2D tensor，shape (N, R)；values、dense_shape、size、weights 均为 1D tensor
- indices 行数(N) 必须等于 values 元素数(N)
- indices 列数(R) 必须等于 dense_shape 元素数(R)
- indices[:, j] 必须在 [0, dense_shape[j]) 范围内
- values[i] 必须在 [0, size) 范围内
- weights 非空时，元素数必须与 values 元素数(N) 相等
- size 为非负标量
- weights 为空（元素数为0）时，每个出现计为 1.0
- binary_output = True 时，出现过的 bin 设为 1.0，忽略权重

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|--------|------|
| 图模式调用 | [test_geir_sparse_bincount](./examples/test_geir_sparse_bincount.cpp) | 通过[算子IR](./op_graph/sparse_bincount_proto.h)构图方式调用SparseBincount算子。 |
