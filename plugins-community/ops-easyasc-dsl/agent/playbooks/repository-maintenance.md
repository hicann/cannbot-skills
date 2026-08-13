# Repository Maintenance

Use this workflow for framework, stub, parser, simulator, test, tool, catalog,
or documentation changes. Kernel implementation uses an authoring workflow.

## 1. Establish the owner and baseline

Identify the smallest source owner and its public behavior. Before editing,
record the focused test or checker result that demonstrates the current state.

For runtime behavior, inspect in this order:

1. public facade and relevant stub under `easyasc/stub_functions/`;
2. parser/lowering under `easyasc/parser/`;
3. simulator execution under `easyasc/simulator/`;
4. focused tests and sample kernels.

Do not infer a device rule from a sample alone. Samples demonstrate supported
usage; stubs, lowering, simulator, and tests own behavior.

## 2. Define the compatibility boundary

State whether the change affects:

- Python public imports or kernel syntax;
- static validation, simulator runtime validation, or generated code;
- device-specific behavior;
- generated indexes or documentation routes.

Prefer device-specific checks over global restrictions. Preserve an existing
public API unless the requested change explicitly alters it.

## 3. Implement with focused tests

Add the smallest guard at the owning layer and a runtime guard when values can
remain dynamic until simulator execution. Test valid boundaries, first-invalid
boundaries, negative values where relevant, and unaffected devices.

Parser validation should fail before code generation when the conflict is
knowable. Error messages must name the invalid value or identifier, the device
or scope, and the supported alternative.

## 4. Update owner documentation

Change stable facts only after implementation tests pass. Keep ownership narrow:

- device behavior: `agent/references/facts-device-runtime.md` or `constraints/`;
- authoring safety: `agent/references/authoring-preflight.md`;
- implementation lookup: `agent/references/code-paths.md`;
- repository layout: `agent/references/repo-map.md`;
- kernel/tool selection: the corresponding Markdown catalog;
- public structure: `README*`, `doc/`, `doc_cn/`, and `agent/example/testcases/README.md`.

Do not duplicate facts in the router, glossary, diary, or generated index.
Generate indexes from their Markdown owner.

## 5. Verify the maintenance surface

Run the focused pytest set, then the relevant repository gates:

```bash
python agent/scripts/check_kernel_catalog.py --fail-on-warning
python agent/scripts/build_agent_index.py --check
python agent/scripts/check_agent_docs.py
```

Inspect the final diff for stale paths, duplicate ownership, unexplained
warnings, and unrelated changes. Do not add a new CI workflow for a check that
can run through existing pytest or tool gates.
