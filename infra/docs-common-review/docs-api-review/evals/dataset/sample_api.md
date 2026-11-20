# demoQueryDeviceStatus

## 功能说明

查询当前线程绑定的 Device 状态。支持在 DVPP 场景下获取 Device 的运行状态码，状态码含义参见 demoStatus 枚举。

## 产品支持情况

Atlas A3系列产品：支持，Ascend 950PR&950DT系列产品：支持。

## 函数原型

```
int32_t demoQueryDeviceStatus(int32_t deviceId, demoStatus *status);
```

## 参数说明

| 参数名 | 说明 |
| ---- | ---- |
| deviceId | Device 的编号，从0开始。 |
| status | 状态码变量的指针。 |

## 约束说明

1. 本接口需在 Device 初始化完成后调用。
2. deviceId 取值需大于等于0。

## 调用示例

```
//查询Device状态
demoStatus status;
int32_t ret = demoQueryDeviceStatus(0, &status);
```
