# NPUGetFloatStatus

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                     |     √    |
| <term>Atlas A3系列产品</term>   |    ×     |
| <term>Atlas A2系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    √     |
| <term>Atlas推理系列产品</term>                               |    √     |
| <term>Atlas训练系列产品</term>                               |    √     |

## 功能说明

* 算子功能：读取NPU硬件浮点溢出状态寄存器，并在检测到溢出时将溢出标志写回输入`addr`（side effect），输出`data`固定为全零。该算子常与NPUAllocFloatStatus配合使用，用于混合精度训练中loss scaling溢出状态的读取场景。本仓交付<term>Ascend 950PR&950DT系列产品</term>实现（暂未支持溢出探测的空实现）：不修改输入`addr`，输出`data`固定全零；其余支持产品的实现由canndev仓交付。

- 计算公式：

  $$
  \text{data} = \text{zeros}(8, \text{dtype}=\text{float32})
  $$

  当检测到溢出（status != 0）时，同时执行写回：

  $$
  \text{addr} = \text{ones}(8, \text{dtype}=\text{float32})
  $$

  Ascend 950上暂未支持溢出探测，status恒为0，故`addr`不被修改，`data`恒为全零。

- 示例：

  ```text
  输入addr：
  tensor([0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float32)
  输出data：
  tensor([0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float32)
  ```

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
      <td>addr</td>
      <td>输入</td>
      <td>接收溢出标志写回的Tensor，固定shape为(8,)。检测到溢出时被写为全1（Ascend 950空实现中恒不写回）。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>data</td>
      <td>输出</td>
      <td>NPU浮点状态Tensor，固定shape为(8,)，包含8个float32零值，不承载状态信息。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 该算子有一个输入`addr`、一个输出`data`，无属性。
- 输入`addr`的shape固定为(8,)，dtype仅支持float32，format仅支持ND。
- 输出`data`的shape固定为(8,)，不可变更。8个元素对应NPU硬件的溢出状态标志位（共32字节）。
- 输出`data`的dtype仅支持float32，不支持其他数据类型。
- 输出`data`的format仅支持ND。
- 输出`data`的值始终为全零（8个float32零值），不承载状态信息；溢出状态通过side effect写回输入`addr`。
- 在Ascend 950上为空实现：overflowStatus固定为0，不修改输入`addr`，输出固定全零，无并行归约，天然确定性。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/test_geir_npu_get_float_status.cpp">test_geir_npu_get_float_status</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
