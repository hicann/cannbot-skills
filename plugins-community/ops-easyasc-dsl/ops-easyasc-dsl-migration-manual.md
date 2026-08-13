# ops-easyasc-dsl Migration Manual

Use this guide to migrate validated EasyASC content from:

- source: `<source-easyasc-root>`
- target: `<cannbot-skills-root>/plugins-community/ops-easyasc-dsl`

The source repository owns the current EasyASC content. The target repository
owns the delivery layout. Migration therefore follows one rule:

> Preserve the target structure; rebuild its managed content from the source.

The target's existing file contents are not a compatibility boundary. Files may
be added, replaced, renamed, or deleted in large numbers as long as the target
structure, entrypoints, archive contract, and fresh-user workflow remain valid.

This source-side manual is authoritative. A same-name copy in the target is a
legacy snapshot until this file is migrated over it.

## 1. Migration contract

### 1.1 What must remain stable

The following are structural contracts:

- the plugin remains at `plugins-community/ops-easyasc-dsl/`;
- `SKILL.md` (plugin root) remains the user-facing entrypoint;
- `agent/ROUTER.md` remains the task router;
- `agent/common-language.md` remains the fixed terminology baseline;
- target-facing playbooks and references remain under `agent/`;
- delivered maintenance tools remain under `agent/scripts/`;
- generated indexes remain under `agent/index/`;
- the two payload archives remain under `agent/assets/`;
- `agent/scripts/init.sh` restores the archived payloads;
- runtime/docs restore to `easyasc/`, `doc/`, and `doc_cn/`;
- examples restore below `agent/example/`.

These paths and roles must survive. Their old bytes do not need to survive.

### 1.2 What may change

Within the protected structure, migration may:

- replace an existing file with the current source version;
- delete a stale target file that no longer has a source owner;
- add new source files and subdirectories;
- treat a source rename as a target deletion plus addition;
- rewrite target entry documents for target paths and archive-backed usage;
- rebuild catalogs, generated indexes, and both archives;
- remove obsolete target-only semantic surfaces.

For example, an obsolete target-only documentation role is not a protected
structural anchor. It may be deleted when the current EasyASC routing model no
longer uses it. Likewise, keeping `agent/playbooks/` does not require keeping
every historical playbook inside it.

### 1.3 What requires a separate design change

Do not change these incidentally during content migration:

- the plugin root or the root `SKILL.md` entrypoint location;
- the archive-backed delivery model;
- archive names or `init.sh` restore destinations;
- `agent/scripts/`, `agent/index/`, or `agent/assets/` ownership;
- the four `agent/example/` content classes;
- the parent repository's plugin installation model.

If one of those needs to change, describe and review it as a packaging or
plugin-architecture change rather than hiding it inside a migration.

## 2. Protected target structure

The checked-in target must retain this shape:

```text
ops-easyasc-dsl/
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── README_CN.md
├── ops-easyasc-dsl-migration-manual.md
├── requirements.txt
├── sitecustomize.py
├── wsl_setenv.sh
├── skill/
│   └── SKILL.md
└── agent/
    ├── ROUTER.md
    ├── common-language.md
    ├── assets/
    │   ├── ops-easyasc-dsl-runtime.tar.gz
    │   └── ops-easyasc-dsl-example.tar.gz
    ├── index/
    ├── playbooks/
    ├── references/
    └── scripts/
        └── init.sh
```

Files such as `agent/common-language.md`, `agent/diary.md`, individual
playbooks, individual references, mapped tools, catalogs, and generated JSON
belong to content zones. They should exist when required by the current source
workflow, but their historical target versions are not structural anchors.

After `bash agent/scripts/init.sh`, the expanded shape must be:

```text
ops-easyasc-dsl/
├── easyasc/
├── doc/
├── doc_cn/
└── agent/
    └── example/
        ├── kernels/
        ├── demo/
        ├── projects/
        └── testcases/
```

The expanded trees are working copies of archived content. Under the current
packaging model they are removed again before commit; the rebuilt archives are
the checked-in payload.

## 3. Content ownership

Treat the target as a set of independently managed zones. Never mirror the
source repository onto the target plugin root in one operation.

| Target zone | Content owner | Migration behavior |
| --- | --- | --- |
| `SKILL.md`, `agent/scripts/init.sh`, archive names and restore contract | target integration | Preserve their roles and paths; rewrite content when source routing or payloads change. |
| Root README/agent guidance and visible `agent` docs | current source semantics plus target path adaptation | Reconcile to the current source document set; delete superseded target-only semantic files. |
| `agent/scripts/*.py` | mapped source `tools/*.py` | Replace from source and adapt target root calculation; retain documented target-only integration scripts. Delivered inside the runtime archive (restored by `init.sh`), not as tracked plain files. |
| `agent/scripts/tools_summary.md` | target quick-reference integration | Regenerate or rewrite from the delivered tool surface. |
| `agent/index/*.json` | Markdown catalogs and index generator | Regenerate; never preserve or hand-edit stale JSON. |
| runtime archive | source `easyasc/`, `doc/`, `doc_cn/` | Reconcile full selected trees, then repack. |
| example archive | mapped source examples/projects/tests | Reconcile full selected trees under the four target containers, then repack. |
| local scratch and workstation files | no target owner by default | Exclude unless deliberately productized. |

For each managed zone, the desired result is:

```text
mapped current source content
+ explicitly listed target-integration files
- stale target content
```

A target-only file is retained only when its target-specific purpose is stated.
"It was already there" is not a retention reason.

## 4. Source-to-target mapping

| Source path | Target path | Rule |
| --- | --- | --- |
| `easyasc/` | restored `easyasc/` | Preserve the source-relative subtree inside the runtime archive. |
| `doc/` | restored `doc/` | Preserve the source-relative subtree inside the runtime archive. |
| `doc_cn/` | restored `doc_cn/` | Preserve the source-relative subtree inside the runtime archive. |
| `kernels/` | restored `agent/example/kernels/` | Preserve the path relative to `kernels/`. |
| `examples/` | restored `agent/example/demo/` | Use an explicit demo mapping; keep device grouping meaningful. |
| `projects/` | restored `agent/example/projects/` | Preserve the path relative to `projects/`. |
| `testcases/` | restored `agent/example/testcases/` | Preserve the path relative to `testcases/`. |
| `tools/*.py` | `agent/scripts/*.py` | Adapt repository-root resolution and target-facing paths. |
| `agent/ROUTER.md`, `agent/common-language.md`, `agent/playbooks/`, `agent/references/`, `agent/diary.md` | same semantic paths under target `agent/` | Reconcile the source-owned document set without overwriting target packaging directories. |
| `agent/references/examples/*-catalog.md` | same target paths | Treat Markdown catalogs as the human-owned index source. |
| `agent/index/*.json` | same target paths | Rebuild with the target-side generator. |
| root `README*`, `AGENTS.md`, `CLAUDE.md`, `requirements.txt`, `sitecustomize.py`, `wsl_setenv.sh` | same visible paths | Rewrite source content for archive-backed target paths. |
| this manual | target root, same name | Replace the target legacy copy. |

Target-only integration paths with no literal source counterpart include:

- `SKILL.md`;
- `agent/scripts/init.sh`;
- `agent/assets/ops-easyasc-dsl-runtime.tar.gz`;
- `agent/assets/ops-easyasc-dsl-example.tar.gz`;
- `agent/scripts/tools_summary.md`.

Generated indexes are also target outputs rather than files to copy blindly.

Concrete mappings include:

- `tools/select_kernel_example.py` ->
  `agent/scripts/select_kernel_example.py`;
- `kernels/a5/matmul/matmul_float_mmad.py` ->
  `agent/example/kernels/a5/matmul/matmul_float_mmad.py`;
- `examples/torchapi/torchapimgr_demo.py` ->
  `agent/example/demo/a5/torchapimgr_demo.py`;
- `projects/a5/gdn_fwd/` ->
  `agent/example/projects/a5/gdn_fwd/`;
- `testcases/simulator/trace/test_trace.py` ->
  `agent/example/testcases/simulator/trace/test_trace.py`.

## 5. Migration workflow

### Step 1: Record the baseline

Work from clean source and target worktrees. Record:

- the source commit being migrated;
- the target branch and commit;
- the current target structural anchors;
- archive member lists;
- relevant source checks and target parent-repository checks.

Useful read-only commands include:

```bash
git status --short
tar -tzf agent/assets/ops-easyasc-dsl-runtime.tar.gz
tar -tzf agent/assets/ops-easyasc-dsl-example.tar.gz
```

Do not infer archive contents from the source layout or an older manual.

### Step 2: Declare the migration scope

Build a reviewed inventory for every affected managed zone:

- source path;
- mapped target path;
- add, replace, rename, or delete action;
- target-specific adaptation, if any;
- validation owner.

The scope may be one subsystem or the full delivered surface. Within the
declared scope, reconcile to the desired current state instead of retaining
unreviewed target leftovers.

Classify files into:

- protected target integration;
- visible managed content;
- archived runtime/docs content;
- archived example content;
- generated content;
- excluded local-only content.

### Step 3: Restore archived content

From the target plugin root:

```bash
bash agent/scripts/init.sh
```

`init.sh` restores only missing trees. Before using a restored tree as a
baseline, confirm it came from the current checked-in archive and does not
contain leftovers from an earlier migration.

### Step 4: Reconcile the visible agent surface

Rebuild the source-owned semantic surface:

- root README and agent guidance;
- `agent/ROUTER.md`;
- `agent/common-language.md`;
- `agent/playbooks/`;
- `agent/references/`;
- `agent/diary.md` when current workflow rules require it;
- catalogs and templates.

Do not copy the entire source `agent/` directory over the target `agent/`
directory. That would endanger target-owned `assets/`, `scripts/`, and
`index/`.

Within the semantic subtrees:

- add new current source files;
- replace changed files;
- delete files removed or superseded in the source;
- repair every route and owner link in the same change.

Obsolete target-only documentation roles must not survive merely because their
directories exist in the old target.

Rewrite target entry documents for this reading path:

```text
SKILL.md -> agent/ROUTER.md -> common-language baseline
         -> one playbook -> focused reference -> restored source/example
```

### Step 5: Reconcile delivered tools

For each delivered source tool:

1. map `tools/X.py` to `agent/scripts/X.py`;
2. replace the old target implementation;
3. change source-root calculation for the deeper target path;
4. update commands and links to target paths;
5. preserve the required CANN license header;
6. update the tool catalog and `tools_summary.md`.

A typical source root expression:

```python
ROOT = Path(__file__).resolve().parent.parent
```

usually becomes:

```python
ROOT = Path(__file__).resolve().parent.parent.parent
```

Do not delete `agent/scripts/init.sh`: it is target integration, not a mapped
source tool. Delete other target-only tools when they have no documented
target-specific role, together with their catalog, summary, and index entries.

### Step 6: Reconcile runtime and documentation payloads

Treat restored `easyasc/`, `doc/`, and `doc_cn/` as replaceable content
zones:

- copy new and changed source files;
- delete stale target files absent from the selected source scope;
- preserve source-relative directory structure;
- keep English and Chinese entry maps consistent;
- preserve environment-driven behavior;
- keep C/C++ resource and delivered script license headers valid.

Do not preserve an old runtime or document merely to reduce diff size.

### Step 7: Reconcile examples, projects, and tests

Populate the four target containers:

- `kernels/` -> `agent/example/kernels/`;
- `examples/` -> `agent/example/demo/`;
- `projects/` -> `agent/example/projects/`;
- `testcases/` -> `agent/example/testcases/`.

Kernel, project, and testcase mappings preserve their relative subtrees. Demo
mapping may deliberately convert a source grouping such as `torchapi/` into a
device-oriented target grouping, but that mapping must be explicit.

Delete stale target examples inside the declared scope. Add new source
categories without moving the four target container roots.

After changing kernels or tools:

- update their Markdown catalogs;
- update the short human selector material where applicable;
- regenerate JSON indexes;
- verify every catalog path against the restored target tree.

### Step 8: Repair target-facing references

Search both visible files and restored payloads for:

- source-only `tools/`, `kernels/`, `examples/`, `projects/`, or
  `testcases/` paths;
- removed playbooks or references;
- absolute source repository paths;
- user home directories, host aliases, work IDs, or IP addresses;
- private notes under `tmp/<task>/`;
- machine-specific CANN installation paths.

Use target paths in all delivered commands. Replace private locations with
placeholders such as `<repo-root>`, `<remote-a5-host>`,
`<ascend-cann-root>`, and `<python-bin>`.

### Step 9: Regenerate owned outputs

From the target plugin root, run the migrated target-side tools:

```bash
python agent/scripts/build_agent_index.py
python agent/scripts/check_kernel_catalog.py --fail-on-warning
python agent/scripts/build_agent_index.py --check
python agent/scripts/check_agent_docs.py
```

If a checker is newly migrated, adapt it to the target layout before treating
its result as authoritative. Never hand-edit generated JSON to make a check
pass.

### Step 10: Rebuild archives

Remove generated junk from the restored payloads, then rebuild the archives
from the target plugin root:

```bash
find easyasc doc doc_cn agent/example -name '.DS_Store' -delete
find easyasc doc doc_cn agent/example -name '._*' -delete
find easyasc doc doc_cn agent/example -name '__pycache__' -type d -prune -exec rm -rf {} +
find easyasc doc doc_cn agent/example \( -name '*.pyc' -o -name '*.pyo' \) -delete

if tar --version 2>&1 | head -1 | grep -qi bsdtar; then
  archive_owner_args=(--uid 0 --gid 0 --uname root --gname root)
else
  archive_owner_args=(--owner=0 --group=0 --numeric-owner)
fi

COPYFILE_DISABLE=1 tar "${archive_owner_args[@]}" \
  -czf agent/assets/ops-easyasc-dsl-runtime.tar.gz \
  easyasc doc doc_cn
COPYFILE_DISABLE=1 tar "${archive_owner_args[@]}" \
  -czf agent/assets/ops-easyasc-dsl-example.tar.gz \
  agent/example
```

Fixed archive ownership is part of the privacy boundary. Without it, tar may
record the local account and group that performed the migration even when file
contents are clean. Python bytecode and AppleDouble files are excluded for the
same reason: both may retain workstation paths or metadata even when the source
text has already been sanitized.

Confirm the archive roots:

```bash
tar -tzf agent/assets/ops-easyasc-dsl-runtime.tar.gz | sed -n '1,40p'
tar -tzf agent/assets/ops-easyasc-dsl-example.tar.gz | sed -n '1,40p'
tar -tvzf agent/assets/ops-easyasc-dsl-runtime.tar.gz | sed -n '1,10p'
tar -tvzf agent/assets/ops-easyasc-dsl-example.tar.gz | sed -n '1,10p'
```

The runtime archive must contain `easyasc/`, `doc/`, and `doc_cn/`. The
example archive must contain all four `agent/example/` containers. The verbose
listing must not contain a personal account, employee identifier, or local
machine group in its owner metadata. Also scan the decompressed archive payload,
including binary strings, rather than assuming that a clean visible worktree
proves the archive is clean.

### Step 11: Validate a fresh-user state

Return to the delivered archive-only shape:

```bash
rm -rf -- easyasc doc doc_cn agent/example
```

Run the restore flow again:

```bash
bash agent/scripts/init.sh
```

Then verify:

- every protected structural anchor exists;
- all required restored roots exist;
- Router and entry documents resolve;
- indexes are fresh;
- catalogs resolve to restored files;
- migrated tools run from `agent/scripts/`;
- focused runtime, simulator, parser, kernel, project, and testcase checks pass;
- no source-layout path leaked into target-facing commands.

After validation, remove the restored trees again before commit when retaining
the archive-only packaging model.

### Step 12: Validate in the parent repository

If the result will be proposed to `cannbot-skills`:

- update the parent `CHANGELOG.md`;
- follow `docs/CONTRIBUTING.md`;
- use Bash 4 or newer for the parent test harness;
- run the fast parent gate from `<cannbot-skills-root>`.

```bash
bash tests/run-tests.sh --fast
```

Record environment-only failures separately from failures caused by migrated
content. Do not hide an old target-content failure by weakening the structural
or content checks.

## 6. Excluded by default

Do not migrate these source development surfaces unless they are deliberately
productized:

- `tmp/` and other scratch directories;
- `.claude/`, `.vscode/`, and `.pytest_cache/`;
- `machine_specs.md`;
- local runner scripts such as `b.sh` and `r.sh`;
- workstation-specific test configuration;
- generated caches, logs, build outputs, and temporary probes.

An excluded source file must not be retained from an old target copy as an
accidental substitute.

## 7. Completion criteria

Migration is complete only when:

- the protected target paths and their roles remain intact;
- no source top-level layout has replaced the target packaging layout;
- managed content matches the declared mapped source scope;
- stale target semantic files have been removed or explicitly justified;
- target-only integration files are listed and still functional;
- superseded target-only designs do not remain accidentally;
- runtime/docs and example archives contain the rebuilt desired state;
- `agent/example/` still contains `kernels/`, `demo/`, `projects/`, and
  `testcases/`;
- all target-facing links and commands use target paths;
- catalogs and generated indexes match restored content;
- privacy and machine-path scans are clean;
- archive payloads and owner metadata contain no local account or machine data;
- fresh `init.sh` restore and focused validation succeed;
- expanded archived trees are absent from the final commit;
- parent contribution and fast-test requirements are handled when proposing the
  migration upstream.

The size of the diff is not a completion criterion. A large, reviewed content
replacement is valid; preserving obsolete files merely to keep the diff small
is not.
