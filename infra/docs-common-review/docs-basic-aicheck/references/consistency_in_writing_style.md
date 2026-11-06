# 写作风格一致性【正反例】

## 标点符号习惯一致性

### 正例

1. 数据类型说明：数据类型支持bf16、fp32、fp64。
2. Shape说明：维度支持1~8维，其中最后一维必须是输入x最后一维的1/2。
3. 数据格式说明：数据格式支持ND。
4. 空Tensor说明：输入Tensor支持空Tensor。

### 反例

1. 数据类型说明：数据类型支持bf16、fp32、fp64。
2. Shape说明：维度支持1~8维，其中最后一维必须是输入x最后一维的1/2
3. 数据格式说明：数据格式支持ND
4. 空Tensor说明：输入Tensor支持空Tensor。

## 无序列表/有序列表渲染效果一致性

### 正例

- <font color="blue">**数据类型说明**：数据类型支持bf16、fp32、fp64。</font>

- <font color="blue">**Shape说明**：维度支持1~8维，其中最后一维必须是输入x最后一维的1/2。</font>

- <font color="blue">**数据格式说明**：数据格式支持ND。</font>

- <font color="blue">**空Tensor说明**：输入Tensor支持空Tensor，输出Tensor不支持。</font>

### 反例

- <font color="blue">**数据类型说明**：数据类型支持bf16、fp32、fp64。</font>

- **Shape说明**：维度支持1~8维，其中最后一维必须是输入x最后一维的1/2。

- <font color="blue">**数据格式说明**：数据格式支持ND。</font>

- <font color="blue">空Tensor说明：输入Tensor支持空Tensor，输出Tensor不支持。</font>
