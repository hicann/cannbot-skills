# 内容冗余性【正反例】

## 表格与列表内容冗余

### 反例

|类型|说明|
|---|---|
|数据类型|数据类型支持bf16、fp32、fp64。|
|Shape|维度支持1~8维，其中out最后一维必须是输入x最后一维的1/2。|
|数据格式|数据格式支持ND。|
|空Tensor|输出Tensor不支持空Tensor。|

输出参数说明：

- 数据类型说明：数据类型支持bf16、fp32、fp64。

- Shape说明：维度支持1~8维，其中out最后一维必须是输入x最后一维的1/2。

- 数据格式说明：数据格式支持ND。

- 空Tensor说明：输出Tensor不支持空Tensor。

**以不同形式或者格式等对输出参数做说明，因此是内容冗余的。**

## 语句与代码段注释冗余

### 反例

```text
${op_name}                              # 替换为为实际算子名，小写下划线形式
├── op_host                             # Host侧实现
│   └── ${op_name}_infershape.cpp       # InferShape实现，实现算子形状推导
├── op_graph                            # 图融合相关实现
│   ├── CMakeLists.txt                  # aclnn侧的cmakelist文件
│   ├── ${op_name}_graph_infer.cpp      # InferDataType文件，实现算子类型推导
└── └── ${op_name}_proto.h              # 算子原型定义，用于图优化和融合阶段识别算子
```

其中${op_name}替换为实际算子名，小写下划线形式；op_host表示Host侧实现，${op_name}_infershape.cpp表示InferShape实现；op_graph表示图融合相关实现，${op_name}_graph_infer.cppb表示InferDataType文件，${op_name}_proto.h表示算子原型定义。

**代码段中的注释已经解释了含义，在正文中又以普通文本格式对代码或者目录等做了解释，因此是内容冗余的。**

## 列表与代码段注释冗余

### 反例

```text
${op_name}                              # 替换为为实际算子名，小写下划线形式
├── op_host                             # Host侧实现
│   └── ${op_name}_infershape.cpp       # InferShape实现，实现算子形状推导
├── op_graph                            # 图融合相关实现
│   ├── CMakeLists.txt                  # aclnn侧的cmakelist文件
│   ├── ${op_name}_graph_infer.cpp      # InferDataType文件，实现算子类型推导
└── └── ${op_name}_proto.h              # 算子原型定义，用于图优化和融合阶段识别算子
```

- ${op_name}替换为实际算子名，小写下划线形式。
- op_host表示Host侧实现。
- ${op_name}_infershape.cpp表示InferShape实现。
- op_graph表示图融合相关实现。
- ${op_name}_graph_infer.cppb表示InferDataType文件。
- ${op_name}_proto.h表示算子原型定义。

**代码段中的注释已经解释了含义，在正文中又以无序列表格式对代码或者目录等做了解释，因此是内容冗余的。**
