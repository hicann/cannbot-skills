# ThreeInterpolate

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

- 算子功能：基于3个最近邻的加权线性特征插值。给定已知特征点的特征描述符、待插值点的3个最近邻索引和对应权重，对每个待插值点的每个特征通道，计算3个最近邻特征的加权和。

- 计算公式：

$$
y[b, n, c] = \sum_{k=0}^{2} weight[b, n, k] \times features[b, idx[b, n, k], c]
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
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
      <td>features</td>
      <td>输入</td>
      <td>已知特征点集合，shape=(B, M, C)。B=batch, M=已知点数, C=通道数。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>idx</td>
      <td>输入</td>
      <td>3个最近邻索引，shape=(B, N, 3)。N=待插值点数，每行3个索引值，取值范围[0, M)。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>3个最近邻权重，shape=(B, N, 3)。与idx对应，通常由距离倒数归一化得到。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>插值后的特征，shape=(B, N, C)。B=batch, N=待插值点数, C=通道数。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- features和weight必须同dtype（均为float32或均为float16），且y的dtype与features一致。
- idx可独立选择int32或int64。
- 所有输入tensor必须为连续（contiguous）格式。
- features、idx、weight均必须为3维tensor；features与idx、weight的B维必须一致，idx与weight的N维必须一致，idx与weight的最后一维必须为3。
- idx值必须在[0, M)范围内，由调用方保证不越界。
- 当输出非空（B、N、C均大于0）时，M必须大于0；M=0时idx无合法取值，算子将报错。
- 维度上限：B、M、C、N均不超过2^32-1，且N×C不超过2^32-1，B×N×3与B×M×C不超过2^64-1。
- 支持空tensor：当B、N、C中任一为0时，输出为空tensor，不执行计算；features的M维为0但输出非空的场景不支持，算子将报错。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>图模式</td>
    <td><a href="./examples/arch35/test_geir_three_interpolate.cpp">test_geir_three_interpolate</a></td>
    <td>通过<a href="./op_graph/three_interpolate_proto.h">算子IR</a>构图方式调用ThreeInterpolate算子，参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成编译和验证。</td>
  </tr>
</tbody>
</table>
