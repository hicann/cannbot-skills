# CCU 短代码范式

仅在对应知识卡MATCH且适用时复用；片段不是可直接宣称功能通过的算法。不要把Skill目录加入生产源码依赖，按本算法需要复制/改写局部函数。

## V1运行期stride乘host常量

`generator-arith`卡覆盖Variable赋值、ADD与生成器；[ccu-patterns.hpp](../assets/ccu-patterns.hpp) 的 `EmitScaledStride` 用host常量的二进制分解生成ASSIGN/ADD链：

```cpp
AscendC::ccu::Variable offset; // 独立句柄，不写 Variable offset = stride
cannbot_ccu_example::EmitScaledStride(stride, hostIndex, offset);
```

`hostIndex`在强转uint16_t前必须证明范围；helper的判断/位移只操作host整数，不产生Variable SHL。`result`不得与stride共用句柄；API的拷贝构造和移动赋值是句柄复制，`Var doubled; doubled = stride;`才是此处需要的值复制。factor=0生成0，其他因子最多展开16位。按实际运行期stride上界证明乘积及base+offset不溢出且在注册区间内；如果stride也在host可得，可改为host预乘并同步维护launch/cache ABI。算术契约登记实际ASSIGN/ADD映射。

工具回归验证helper的host展开与模拟Variable值/别名语义；维护者另用目标HCOMM头做语法实例化。它们不代替真实CCU翻译、设备执行或算法验证。

## 多peer建链：分开收集再合并

适用 `all-gather-framework` 的建链方法。以下为一次每peer的请求，置于返回HcclResult的函数中；peers已经按kernel消费顺序构造：

```cpp
std::vector<HcclChannelDesc> merged;
for (u32 peer : peers) {
    std::vector<HcclChannelDesc> perPeer;
    CHK_RET(CreateChannelRequestByRankId(comm, param, myRank, peer, perPeer, 1));
    merged.insert(merged.end(), perPeer.begin(), perPeer.end());
}
```

建链入口clear的是perPeer，不会抹掉已收集请求。N=2的prev/next是否合并由协议决定，应在生成peers时显式处理；需要同peer多端口/多channel的算法不能按peer盲目去重。请求可能返回多条描述，不能默认merged下标等于peer下标；登记并核对实际channel→peer/die映射与kernel消费顺序。

## launch区间与缓存metadata

以下只是两槽ABI示意，偏移已是字节且属于各自注册buffer：

```cpp
// cache[0..1] 为launch地址，cache[2..3]只保存基址偏移。
cache[0] = inputBase + cache[2];
cache[1] = outputBase + cache[3];
// launchCount仍为2，不传4项缓存，也不传sizeof(cache)。
```

实际算法依本次launch-contract的字段名/顺序更新全部required_updates，包括alias等派生量；保留cache key和token失效依据。参数表集中在JSON，LOG只记录实现落点及差异，Reviewer保留自己的核验结论与证据，不再复制多份完整参数表。

缓存带长度头时，以`cache_launch_offset`声明启动区间（Reduce当前为1）；物理缓存更新索引与LoadArg索引分开，按 [启动参数契约](implementation.md#4-启动参数及-fastlaunch) 核对。默认0兼容原前缀布局。
