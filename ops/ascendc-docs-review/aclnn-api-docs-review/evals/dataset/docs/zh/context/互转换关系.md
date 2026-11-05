# 互转换关系

当一个API（如aclnnAdd、aclnnMul等）**输出的aclTensor数据类型**与**输入的数据类型**推导后的计算类型不一致时，API内部就会将计算结果转换成输出类型对应的数据类型。

数据类型转换需要满足以下规则，不满足规则的将不能进行转换，调用API时会出现参数校验失败。

- 浮点类型：ACL\_FLOAT16、ACL\_FLOAT、ACL\_DOUBLE、ACL\_BF16。
- 整数类型：ACL\_INT8、ACL\_UINT8、ACL\_INT16、ACL\_UINT16、ACL\_INT32、ACL\_UINT32、ACL\_INT64、ACL\_UINT64。
- 复数类型：ACL\_COMPLEX64、ACL\_COMPLEX128。
- 整数类型间可以转换，也支持往浮点、复数类型转换。
- 浮点类型间可以转换，也支持往复数类型转换。
- 复数类型间可以转换。
- BOOL支持往整数、浮点、复数类型转换。

除了以上场景，其他场景的转换均不支持。
