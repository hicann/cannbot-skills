# RandomUniformV2

## 产品支持情况

| 产品                                                               | 是否支持 |
| :----------------------------------------------------------------- | :------: |
| Ascend 950PR&950DT系列产品                                        |    √    |
| Atlas A3系列产品                        |    ×    |
| Atlas A2系列产品 |    ×    |
| Atlas 200I/500 A2推理产品                                         |    ×    |
| Atlas推理系列产品                                                 |    ×    |
| Atlas训练系列产品                                                 |    ×    |

## 功能说明

- 算子功能：返回均匀分布的随机数列。

## 参数说明

<table style="table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 140px">
  <col style="width: 140px">
  <col style="width: 180px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>shape</td>
      <td>输入</td>
      <td>输出张量的形状。</td>
      <td>INT64、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入</td>
      <td>偏移值。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>属性</td>
      <td>随机数种子。</td>
      <td>INT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seed2</td>
      <td>属性</td>
      <td>随机数种子。</td>
      <td>INT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>属性</td>
      <td>指定输出的数据类型。</td>
      <td>INT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>生成的随机数序列。</td>
      <td>FLOAT16、FLOAT、BF16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输出</td>
      <td>偏移值。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码                                                                       | 说明                                                                                     |
| ---------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| 图模式调用 | [test_geir_random_uniform_v2](./examples/test_geir_random_uniform_v2.cpp) | 通过[算子IR](./op_graph/random_uniform_v2_proto.h)构图方式调用RandomUniformV2算子。 |
