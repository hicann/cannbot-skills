---
skill_name: ascendc-docs-search
eval_mode: text
---

# Case 1: API 变体搜索（关键能力）

## Config
- Max Tokens: 100000

## Prompt

我在用 ReduceMax API，发现文档里好像有好几个不同版本的 ReduceMax，怎么确认应该用哪个？

## Expected Output

回复应说明 Ascend C 存在 240+ 个带数字后缀的 API 变体（如 `ReduceMax.md`、`ReduceMax-35.md`、`ReduceMax-92.md`），不同变体的函数签名、参数和功能可能完全不同。应说明必须先用 `find "$ASC_DEVKIT_DIR/docs/api/" -name "ReduceMax*.md"` 列出所有变体，然后逐一查阅每个变体的函数签名和参数说明确认功能差异，最后选择符合需求的版本。应强调禁止只读单个文件就下结论。

## Expectations

- [contains] 变体
- [contains] ReduceMax
- [contains] find
- [contains] 签名

---

# Case 2: 搜索示例代码

## Config
- Max Tokens: 100000

## Prompt

我想找一个双缓冲（double buffer）+ 流水线的 Ascend C 实现参考，应该去哪里找？

## Expected Output

回复应推荐查看 `$ASC_DEVKIT_DIR/examples/00_introduction/01_add/basic_api_memory_allocator_add/` 目录，这是官方的高性能模板示例，包含双缓冲+流水线的标准实现。应说明该示例位于示例代码目录的 `00_introduction` 分类下，是入门高性能算子开发的推荐参考。

## Expectations

- [contains] 双缓冲
- [contains] basic_api_memory_allocator_add
- [contains] ASC_DEVKIT_DIR

---

# Case 3: 本地资源不足时使用在线搜索兜底

## Config
- Max Tokens: 100000

## Prompt

我在本地文档里找不到关于 Ascend C 临时内存申请（TempTensor）的详细说明，有没有更详细的官方文档？

## Expected Output

回复应说明当本地 `$ASC_DEVKIT_DIR` 文档未覆盖相关内容时，可以使用在线搜索兜底。应推荐使用 `ascend_search_client.py` 脚本搜索华为昇腾社区文档，示例命令如 `python skills/ascendc-docs-search/scripts/ascend_search_client.py "Ascend C 临时内存申请" --max_results 5`，并说明可使用 `ascend_content_fetcher.py` 获取详细内容。应提及推荐使用中文关键词搜索，版本过滤建议使用 8.5.0。

## Expectations

- [contains] ascend_search_client.py
- [contains] 在线
- [contains] 华为昇腾社区

---

# Case 4: 正向看护-多 skill 环境下正确触发目标 skill

## Config
- Max Tokens: 120000
- Distractor skills: ascendc-docs-gen;ascendc-api-best-practices;ascendc-tiling-design;ascendc-env-check

## Prompt

我想查一下 Ascend C 中 Exp 和 Log API 的参数说明和使用示例，应该用什么工具搜索？

## Expected Output

回复应正确激活 ascendc-docs-search skill，说明通过 `$ASC_DEVKIT_DIR/docs/api/` 搜索 Exp 和 Log 的 API 文档，使用 `find` 命令查找所有变体，并在 `$ASC_DEVKIT_DIR/examples/` 中搜索使用示例。即使在 ascendc-docs-gen、ascendc-api-best-practices 等相似 skill 共存的环境下，也应正确选择 ascendc-docs-search。

## Expectations

- [skill_activated] ascendc-docs-search
- [contains] Exp
- [contains] Log
- [contains] ASC_DEVKIT_DIR

---

# Case 5: 边界场景-模糊查询时主动追问

## Config
- Max Tokens: 80000

## Prompt

帮我找个 API

## Expected Output

回复应识别到用户查询过于模糊，主动追问具体需要查找哪个 API、想了解什么信息（函数签名、参数说明、使用示例、兼容性等）。不应在缺少具体 API 名称或搜索目标的情况下直接执行搜索或输出大量无关文档列表。

## Expectations

- [not_contains] find "$ASC_DEVKIT_DIR
- [contains] 哪个
