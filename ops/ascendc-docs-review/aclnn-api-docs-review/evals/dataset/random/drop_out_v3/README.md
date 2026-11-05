# DropOutV3

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                     |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                             |    ×     |

## 功能说明

- 算子功能：训练过程中，按照概率p随机将输入中的元素置零，并将输出按照1/(1-p)的比例缩放。
- 计算公式：
  $$
  out_i=\begin{cases}0,&\text { with probability }p \\\frac{1}{1-p}input_i,&\text { with probability }1-p\end{cases}
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
      <td>input</td>
      <td>输入</td>
      <td>输入元素。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>noise_shape</td>
      <td>输入</td>
      <td>预留参数，入参请用空指针代替。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>p</td>
      <td>输入</td>
      <td>元素置零的概率，取值范围为[0, 1]。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>输入</td>
      <td>随机数的种子，影响生成的随机数序列。</td>
      <td>INT64、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入</td>
      <td>随机数的偏移量，它影响生成的随机数序列的位置。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出数据。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>输出</td>
      <td>bit类型并使用UINT8类型存储的mask数据。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| aclnn接口 | [test_aclnn_drop_out_v3](./examples/arch35/test_aclnn_drop_out_v3.cpp) | 通过[aclnn_drop_out_v3](docs/aclnnDropoutV3.md)接口方式调用DropOutV3算子。 |
| 图模式调用 | [test_geir_drop_out_v3](./examples/arch35/test_geir_drop_out_v3.cpp)   | 通过[算子IR](./op_graph/drop_out_v3_proto.h)构图方式调用DropOutV3算子。 |
