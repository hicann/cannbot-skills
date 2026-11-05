# 数据类型

通过**aclCreateTensor**接口创建aclTensor时，支持的全量数据类型参见[《acl API（C）》](https://hiascend.com/document/redirect/CannCommunityCppApi)中“数据类型及其操作接口>aclDataType”。
两段式接口参数说明时，为了方便描述，支持的数据类型将采用如下简写形式。

**表1**  数据类型简写表

|   原始数据类型    | 简写形式（不区分大小写） |
| :---------------: | :----------------------: |
|     ACL_FLOAT     |      FLOAT或FLOAT32      |
|    ACL_FLOAT16    |         FLOAT16          |
|     ACL_INT8      |           INT8           |
|     ACL_INT32     |          INT32           |
|     ACL_UINT8     |          UINT8           |
|     ACL_INT16     |          INT16           |
|    ACL_UINT16     |          UINT16          |
|    ACL_UINT32     |          UINT32          |
|     ACL_INT64     |          INT64           |
|    ACL_UINT64     |          UINT64          |
|    ACL_DOUBLE     |     DOUBLE或FLOAT64      |
|     ACL_BOOL      |           BOOL           |
|    ACL_STRING     |          STRING          |
|   ACL_COMPLEX64   |        COMPLEX64         |
|  ACL_COMPLEX128   |        COMPLEX128        |
|     ACL_BF16      |      BF16或BFLOAT16      |
|     ACL_INT4      |           INT4           |
|     ACL_UINT1     |          UINT1           |
|   ACL_COMPLEX32   |        COMPLEX32         |
|   ACL_HIFLOAT8    |         HIFLOAT8         |
|  ACL_FLOAT8_E5M2  |       FLOAT8_E5M2        |
| ACL_FLOAT8_E4M3FN |      FLOAT8_E4M3FN       |
|  ACL_FLOAT8_E8M0  |       FLOAT8_E8M0        |
|  ACL_FLOAT6_E3M2  |       FLOAT6_E3M2        |
|  ACL_FLOAT6_E2M3  |       FLOAT6_E2M3        |
|  ACL_FLOAT4_E2M1  |       FLOAT4_E2M1        |
|  ACL_FLOAT4_E1M2  |       FLOAT4_E1M2        |
