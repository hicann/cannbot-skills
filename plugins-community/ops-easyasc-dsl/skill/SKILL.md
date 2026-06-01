---
name: ops-easyasc-dsl
description: easyasc DSL to AscendC workflow. Author, debug, and validate Ascend NPU kernels written in the easyasc Python DSL, then lower them to AscendC runtime.
---

# ops-easyasc-dsl

This is the user-facing entrypoint for the easyasc DSL to AscendC skill. The reusable workflow itself lives under `agent/`; this file only points you at it.

## Before you read the archived runtime/docs or examples

Parts of this repository ship as archives to keep the delivered skill bundle small.

Run the initialization script once before reading runtime/doc content or running examples:

```bash
bash agent/scripts/init.sh
```

`agent/scripts/init.sh` restores:

- `easyasc/` — the DSL Python package and codegen runtime
- `doc/` and `doc_cn/` — long-form English and Chinese documentation
- `agent/example/` — curated kernel examples and manual demo programs

The script is idempotent; only missing trees are restored. Archives live under `agent/assets/`:

- `agent/assets/ops-easyasc-dsl-runtime.tar.gz` — the `easyasc/`, `doc/`, and `doc_cn/` payload
- `agent/assets/ops-easyasc-dsl-example.tar.gz` — the `agent/example/` payload

## Where to go next

Start with the router:

- `agent/ROUTER.md` — progressive-disclosure routing layer

Preferred read order:

1. `agent/ROUTER.md` — pick the matching route
2. one playbook from `agent/playbooks/`
3. one focused reference from `agent/references/`
4. one source example from `agent/example/kernels/` (after `init.sh`)

## Top-level layout

- `skill/` — this skill entrypoint
- `agent/` — the callable easyasc DSL to AscendC workflow used by this skill
  - `agent/scripts/` — repository-maintenance scripts, including `init.sh`
  - `agent/assets/` — archived runtime/docs and example payloads
  - `agent/example/` — restored kernel examples and manual demos (on-demand)

## Environment guidance

The delivered scripts prefer environment variables over fixed install paths. Set them to match your CANN installation, for example:

```bash
export ASCEND_HOME_PATH=/path/to/your/cann/install
# optional conda env name (example only, not required)
# conda activate torch210npu
```

Avoid hardcoding machine-specific absolute paths when editing delivered scripts.
