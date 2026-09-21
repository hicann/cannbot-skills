# `kb/` — 插件本地运行时资料

官方 AscendC 知识已经迁移到独立的 `cannbot-knowledge` 仓，由其安装器准备索引并
提供 `knowledge-query`。port 插件通过项目配置解析该知识仓，不再把插件目录当作
官方知识库或查询根。

本目录只保留两类插件本地资料：

- `shared/`：直接注入 agent brief 的规则、协议和方法论；它们不是可检索知识卡。
- `okf/reference/porter/`：尚由 port 插件拥有的少量模板和手册资产；不得把外部知识
  的副本重新搬回这里。

运行时产生的新经验只能写入用户知识层，默认根目录为
`$ASCENDC_PORT_USER_KB`（未设置时使用 `~/.ascendc-port/user_kb`）。插件目录和外部
官方知识仓在运行时都必须保持只读。

维护约束见 [`CONVENTIONS.md`](CONVENTIONS.md)。
