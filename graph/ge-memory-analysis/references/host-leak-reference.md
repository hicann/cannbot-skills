# Host 内存泄漏排查参考（TMT）

Host 内存泄漏是**通用问题**，排查方法与普通 C/C++ 程序一致，不依赖 GE 特有工具。

## 1. 工具选型

| 工具 | 泄漏检测 | 越界检测 | 性能 | 动态开关 | 免编译 | 适用场景 |
|------|---------|---------|------|---------|--------|---------|
| **TMT** | ✅ | ❌ | 高 | ✅ | ✅（LD_PRELOAD） | **首选**：已构建产物/现网进程/长时间运行业务，增量定位泄漏栈 |
| ASan | ✅ | ✅ | 中 | ❌ | ❌（需重编译） | 可重编译时；泄漏+越界一起查 |
| valgrind | ✅ | ✅ | 低 | ❌ | ✅ | 短生命周期小程序，兼容 ASan 不可用时 |

## 2. TMT 使用（https://gitcode.com/tangqunzhang/tmt）

```bash
# 1. 安装（root）
# 方式一：git clone
git clone https://gitcode.com/tangqunzhang/tmt.git /home/tmt && cd /home/tmt && make && make install
# 方式二（clone 不可用时）：浏览器打开 https://gitcode.com/tangqunzhang/tmt 页面点 "ZIP" 下载
# （zip 直链为 JS 动态生成，无法 curl 直接下载），上传服务器后：
# unzip tmt-main.zip && mv tmt-main /home/tmt && cd /home/tmt && make && make install

# 2. 环境变量
export TMT_WATCH_SIZE=1024,4294967295     # 统计的内存大小范围，默认值；抓不到泄漏再改 [1,1024) 由大到小分批抓
export LD_PRELOAD=/home/tmt/libtmt.so     # 预加载，被测对象免编译

# 3. 统计点控制（推荐内嵌到业务代码，从稳定运行的第2次迭代开始统计避免资源初始化误报）
#    c++:  system(("tmt clear " + std::to_string(getpid())).c_str());
#          system(("tmt start " + std::to_string(getpid())).c_str());
#          ... 业务迭代 N 次 ...
#          system(("tmt dump " + std::to_string(getpid())).c_str());
#    也可外部命令: tmt clear/start/dump <pid>（启动后执行，时机不如内嵌准，可能误报）

# 4. 看结果
ls /home/tmt/log/tmt_malloc_stats_<可执行名>_<pid>.log
```

**结果解读**（log 文件按序含）：
- 各大小段内存分配/释放/泄漏块数（不受 watch size 限制，看总量趋势）
- 总泄漏大小 vs 观测泄漏大小（后者只算 watch size 范围）
- **泄漏调用栈**（相同栈合并，含 address/size/count）→ 定位泄漏点
- **统计点后才释放的调用栈** → 区分"释放时机不对"（业务持有过久/池未收缩）vs "最终未释放"（真泄漏）

**常见问题**：共享内存报错 → root 执行 `/home/tmt/cleanup.sh` 清理。

**验证工具**：tmt 仓库自带 sample（`sample/sample_main.cc`，含泄漏/复用场景），make 时自动编译，执行验证：
```bash
cd /home/tmt && make && TMT_WATCH_SIZE=1024,4294967295 LD_PRELOAD=/home/tmt/libtmt.so bin/sample
# 结果在 /home/tmt/log/tmt_malloc_stats_sample_<pid>.log
```

## 3. 备选工具

1. ASan 构建：`bash build.sh --asan`，复现后看 LeakSanitizer 报告（可同时查越界）
2. valgrind：`valgrind --leak-check=full <app>`（性能低，小程序用）
3. 崩溃伴随泄漏 → 用 gdb 加载 core dump 分析（`gdb <可执行文件> <core文件>`，bt 看崩溃栈）

## 4. GE 侧常见 host 泄漏参考点

| 泄漏点 | 典型现象 |
|--------|---------|
| OM 加载路径 host buffer | host 内存随加载次数增长 |
| Session/图定义等 host 对象未销毁 | 进程常驻内存持续增长 |

如需修改 GE 运行时内存源码，请先阅读 GE 仓 [rt2_runtime.md](https://gitcode.com/cann/ge/blob/master/docs/zh/design/constraints/rt2_runtime.md)（RT2 动态 shape）或 [known_shape_runtime.md](https://gitcode.com/cann/ge/blob/master/docs/zh/design/constraints/known_shape_runtime.md)（V1 静态）。
