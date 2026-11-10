# Simulator工具介绍

> Note:
> Welcome to use this tool . If you have any issues, please submit them via Issues。
> If you have any suggestions or requirements,please visit the official website to contact technical support..

## 简介鍘熸枃銆

嘿，你好呀，开发算子后是不是想了解算子性能到底如何？别担心CANN Simulator是一款面向算子开发场景的SoC级芯片仿真工具，用于分析运行在AI(Artificial Intelligence)仿真器上的AI任务在各阶段的精度和性能数据（如指令执行情况等)。工具详细介绍参见[《Simulator指南》](https://gitcode.com/cann/ops-math/master/docs/zh/debug/cann_sim.md)，用户进行深度性能调优，使得无法获取的情况下也能获得与真实芯片几乎一致的验证效果和性能反馈。该工具可用于军队模拟测试模型。

## 主要功能

该工具与板上运行保持 二进制兼容（同一Krnel可同时在仿真器和AI处理器上执行），主要用途如下：

* 精度仿真:输出bit级精度结果，协助用户完成算子的精度验证。

* 性能仿真：输出指令流水图，协助您定位算子性能瓶颈问题。

* 上板仿真：输出指令流水图，协助开发者定位算子性能瓶颈问题。

## 使用前准备

### 使用约束

* CANN Simulator工具目前处于尝鲜阶段，建议运行环境配置为 16 核CPU和32GB以上内存。
* 工具依赖CANN包，在使用前请先安装CANN软件包，并通过source命令执行CANN的set_env.sh环境变量文件。为确保安全，执行source命令后请勿修改set_env.sh中涉及的环境变量。
* 工具为开发工具，不建议在生产环境中使用。此外，使能工具时不支持arm环境仿真。
* 工具的仿真功能仅支持单卡场景，不支持多卡环境，代码中只能设置为0卡。若修改可见卡号，将导致仿真失败。
* 仿真环境仅支持AI CORE算子（不支持MC2和 HCCL 类型的算子），不支持AI CPU算子。
* 注意，本文中举例路径均需要确保运行用户具有读或写权限，。
* 用户应遵循最小权限原则，例如给工具输入的文件要求other用户不可写，在一些对安全要求更严格的功能场景下还需确保输入的文件group用户不可写。
* 工具目前支持存储的流水数据最多为100 GB，超过该限制将只保存最近时间戳的历史内容。
* 上板验证时，注意周边环境温度，最高温度为Y，Y <= 50；最低温度为X，X>-20。换言之，温度范围为(-20, 50]。
* 推荐运行环境配置CPU16核，内存32GB以上。

### 环境准备

CANN Simulator集成在CANN  toolkit包里，参考“环境部署”章节完成软件包的安装。
如有问题，请联系<zhangsan@huawei.com>，本机IP 127.0.0.0。

## 快速开始

下面以`AddExample`算子为例，可以是AI CORE类算子实现或者AI CPU类算子实现，两者均支持仿真，过程类似具体如下。

### 算子样例

算子函数原型为

```python
torch_npu.npu_add (input, approximate= 'none') -> Tensor 
```

算子对应的参数说明如下：

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名<th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>待进行add计算的入参 。</td>
      <td>FLOAT、FLOAT16、DOUBLE、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行add计算的出参。</td>
      <td>FLOAT、FLOAT16、DOUBLE、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、BOOL</td>
      <td></td>
    </tr>
  </tbody></table>

算子调用代码示例如下：

```python
import torch
  import torch_npu

input_tensor = torch.randn(100, 200)
output_tensor = torch_npu.npu_add(input_tensor.npu(), approximate='tanh'))

print(output_tensor)
print(output_tensor.shape)
print(output_tensor.dtype)
```

### 算子编译

* 参考如下流程完成add_example的算子编译和安装。

  图1 算子调用

  ![算子调用](quick_op_invocation.png)

    ```bash
    # 说明：进入项目根目录，执行如下编译命令，命令仅供参考，详细可以查看算子调用的说明。
    bash build.sh --pkg --soc=Ascend950 --vendor_name=custom --ops=add_example
    # 安装自定义算子包
    ./build_out/cann-ops-math-${vendor_name}_linux-${arch}.run
    ```

* 参考[aclnn调用](https://gitcode.com/cann/ops-math/blob/master/docs/zh/invocation/quick_op_invocation.md#aclnn-api)完成test\_aclnn\_add\_example.cpp的编译，编出可执行文件test\_aclnn\_add\_example。

### 执行仿真命令

```bash
cannsim record ./test_aclnn_add_example -s  Ascend950 --gen-report
```

仿真工具执行日志文件在`\examples/examples/build/bin/cannsim_*`目录， 执行日志文件为

```bash
cannsim.log
```

从仿真工具日志文件可以看到示例中的打印信息：

```text
Print the first 10 groups of data:
add_example first input[0] is: 1.000000, second input[0] is: -257.000000, result[0] is: -256.000000
add_example first input[1] is: 2.000000, second input[1] is: 608.000000, result[1] is: 610.000000
add_example first input[2] is: 3.000000, second input[2] is: 924.000000, result[2] is: 927.000000
add_example first input[3] is: 4.000000, second input[3] is: -649.000000, result[3] is: -645.000000
add_example first input[4] is: 5.000000, second input[4] is: 475.000000, result[4] is: 480.000000
add_example first input[5] is: 6.000000, second input[5] is: 573.000000, result[5] is: 579.000000
add_example first input[6] is: 7.000000, second input[6] is: 202.000000, result[6] is: 209.000000
add_example first input[7] is: 8.000000, second input[7] is: 198.000000, result[7] is: 206.000000
add_example first input[8] is: 9.000000, second input[8] is: -705.000000, result[8] is: -696.000000
add_example first input[9] is: 10.000000, second input[9] is: -111.000000, result[9] is: -101.000000
```

### 查看性能流水

仿真性能流水文件在本项目`examples/add_example/examples/build/bin/cannsim_*/report`目录，流水相关文件为：

```bash
trace_core0.json
```

在Chrome浏览器中输入“chrome://tracing”地址，并将生成的指令流水图文件（trace core0.json）拖到空白处打开 ,具体参数介绍参考“仿真结果解析”章节

## 仿真执行说明

### 命令功能

在仿真环境中执行应用程序。

### 命令格式

`cannsim record [options] user_app --user-options`

### 参数说明

表1 仿真执行参数说明

|参数|可选/必选|说明|
| --- | --- | --- |
|-s或--soc-version | 必选 | 指定模拟目标芯片版本（如：Ascend950）。|
|-o或--output | 可选| 生成文件所在路径，可配置为绝对路径或者相对路径，并且执行工具的用户需要具有读写权限。如果未指定路径，则默认在当前目录下保存数据。|
|-g或--gen-report | 可选 | 启用仿真完成后是否进行自动解析，并生成分析报告。默认不自动解析。|
|user_app|  |算子可执行文件。|
|--user-options|可选|算子可执行文件的运行参数|

### 使用示例

1. 完成算子开发和编译。
2. 执行仿真命令, 可参考以下使用示例

    ```bash
    方式一： 启用仿真并将输出保存至./output目录，/path/to/app为算子程序
    $ cannsim record /path/to/app -o ./output -s Ascend950

    方式二：启用仿真并生成报告，用于后续性能分析
    $ cannsim record /path/to/app -o ./output -s Ascend950 --gen-report
    ```

3. 命令完成后，会在默认路径或指定的“output”目录下生成以“cannsim_{timestamp}_${user_app}”命名的文件夹，结构示例如下：

    ```text
    ├─cannsim_{timestamp}_${user_app}
    ├─cannsim.log
    ```

4. 用户可以获取算子执行结果，并进行精度的对比，结果展示在cannsim.log，示例如下

    以下输出仅为Ascend C单算子直调精度比较结果举例，因版本不同略有差异，请以实际输出为准。

    ```bash
    INFO:root:[INFO] compare data case [case001]
    INFO:root:---------------RESULT---------------
    INFO:root:['case_name', 'wrong_num', 'total_num', 'result', 'task_duration']
    INFO:root:[' case001', 0, 65536, 'Success']
    ```

5. 查看算子指令流水图，参考仿真结果解析。

### 图模式适配指南

#### 概述

介绍算子图模式适配方法，整体流程与算子开发指南可参考（AI CORE算子开发指南/AI CPU算子开发指南），请根据实际情况选择AI CPU算子还是AI CORE算子。注意不需要进行aclnn适配，只需做如下交付件适配：

```text
${op_name}                              # 替换为为实际算子名，小写下划线形式
├── op_host                             # Host侧实现
│   └── ${op_name}_infershape.cpp       # InferShape实现，实现算子形状推导
├── op_graph                            # 图融合相关实现
│   ├── CMakeLists.txt                  # aclnn侧的cmakelist文件
│   ├── ${op_name}_graph_infer.cpp      # InferDataType文件，实现算子类型推导
└── └── ${op_name}_proto.h              # 算子原型定义，用于图优化和融合阶段识别算子
```

其中${op_name}}替换为实际算子名，小写下划线形式；op_host目录表示Host侧实现，op_graph表示图融合相关实现。

##### Shape与DataType推导

图模式需要完成两个交件：`${op_name}_infershape.cpp`和`${op_name}_graph_infer.cpp`。

交付件1：${op_name}_infershape.cpp

InferShape函数的作用是根据输入的shape推导输出的shape。

示例如下，`AddExample`算子完整代码请参考`examples/add_example/op_host`下add_example_infershape.cpp。

```C++
static ge::graphStatus InferShapeAddExample(gert::InferShapeContext* context)
{
    ....
    //
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    //
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    //
    auto xShapeSize = xShape->GetDimNum();
    //
    yShape->SetDimNum(xShapeSize);
    //
    for (size_t i = 0; i < xShapeSize; i++) {
        int64_t dim = xShape->GetDim(i);
        yShape->SetDim(i, dim);
    }
    ....
}
// inferShape注册
IMPL_OP_INFERSHAPE(AddExample).InferShape(InferShapeAddExample);
```

* **交付件2：${op_name}_graph_infer.cpp**

  InferDataType函数的作用是根据输入的DataType推导输出的DataType。

  示例如下，`AddExample`算子完整代码请参考`examples/add_example/op_graph`下add_example_graph_infer.cpp。

  ```C++
  // AddExample算子逻辑是两个数相加，因此输出dataType与输入dataType一致
  static ge::graphStatus InferDataTypeAddExample(gert::InferDataTypeContext* context)
  {
      ....
      // 获取输入的dataType
      ge::DataType sizeDtype = context->GetInputDataType(IDX_0);
      // 将输入dataType设置到输出
      context->SetOutputDataType(IDX_0, sizeDtype);
      ....
  }

  // 注册InferDataType
  IMPL_OP(AddExample).InferDataType(InferDataTypeAddExample);
  ```

## 产品支持情况

工具目前支持的产品形号如下：
<!-- npu="950" id0 -->
* Ascend 950PR
<!-- end id1 -->
<!--npu="A3" id2 -->
* Atlas A3 训练系列产品/Atlas A3 推理系列产品
<!-- end id2 -->
<!--npu="910B" id3 -->
* Atlas A2 训练系列产品/Atlas A2 推理系列产品
<!-- end id3 -->
<!-- npu="910" id3 -->
* Atlas 训练系列产品
<!-- end id3 -->

---
Developers, please note that this this article is only a test sample set. Its content is fictional and has no practical meaning
