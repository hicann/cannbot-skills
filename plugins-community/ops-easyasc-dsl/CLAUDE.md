# ops-easyasc-dsl

Read `AGENTS.md` first — it is the working contract for this plugin.

Routing map:

- `agent/ROUTER.md` — progressive-disclosure router; pick the route that
  matches the task before opening anything else.
- `agent/common-language.md` — fixed terminology baseline; read it in full
  before any playbook.
- `agent/playbooks/` + `agent/references/` — one playbook, then only the
  focused references it points to.

Before reading `easyasc/`, `doc/`, `doc_cn/`, or `agent/example/`, restore
them from the delivery archives:

```bash
bash agent/scripts/init.sh
```
