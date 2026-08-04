# tip: pass 框架类都在 ge::fusion，禁 using namespace fusion

> 📎 导航落点：`references/interface-catalog.md` §二/§三（构图接口 / 命名空间）。本文件仍是命名空间纪律的**唯一权威定义源**，正文未改动。迁移映射见 `tips/MIGRATION.md`。

**读者**：② 开发 / ①。

## 症状

写 C++ pass 时用了 `using namespace fusion;` 或把 `FusionBasePass`/`PatternFusionPass` 当成全局 `fusion` 命名空间下的类，导致找不到符号 / 编译报未声明。

## 根因

`FusionBasePass`、`PatternFusionPass`、`DecomposePass`、`SubgraphRewriter`、`SubgraphBoundary`、`Pattern`、`MatchResult`、`InferShapeUtil` 等 pass 框架 API 定义在 **`ge::fusion`**，不是全局 `fusion`。

## 硬性做法

包含相应头文件并使用 `ge::fusion` 限定：

```cpp
#include "ge/fusion/pass/pattern_fusion_pass.h"  // PatternFusionPass 场景
#include "ge/fusion/pass/decompose_pass.h"       // DecomposePass 场景
#include "ge/fusion/pass/fusion_base_pass.h"     // FusionBasePass 场景

using namespace ge;
using namespace ge::fusion;
```

也可全限定：`ge::fusion::PatternFusionPass`、`ge::fusion::GraphUniqPtr` 等。

**禁止** `using namespace fusion;`。

## 自查

- 源码里有没有 `using namespace fusion;`？有→删掉，改 `ge::fusion` 或全限定。
- pass 框架类是不是都通过 `ge::fusion::` 或 `using namespace ge::fusion;` 访问？
