import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/catlass-dsl-knowledge/scripts/record_knowledge.py"


def load_module():
    spec = importlib.util.spec_from_file_location("catlass_dsl_knowledge", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(project), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(project), check=True)
    subprocess.run(["git", "config", "user.name", "CATLASS DSL Test"], cwd=str(project), check=True)
    (project / "tracked.txt").write_text("base", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=str(project), check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=str(project), check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(project),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    return project, commit


def valid_entry(project, commit, topic="block-shape"):
    evidence = project / ".catlass-dsl" / "runs" / topic / "result.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text('{"correctness":"passed"}', encoding="utf-8")
    return {
        "operator_family": "matmul",
        "topic": topic,
        "arch": ["c310"],
        "versions": {"catlass": "repository-commit", "cann": "environment-record"},
        "applicability": {
            "shape": "M=128,N=128,K=128",
            "dtype": "float16",
            "layout": "ND",
            "repository_integration": "approved contract",
        },
        "hypothesis": "A single tile change reduces the approved metric.",
        "actual_change": "Changed only the candidate tile shape.",
        "correctness_before": {"status": "passed", "summary": "baseline passed", "reason": ""},
        "correctness_after": {"status": "passed", "summary": "candidate passed", "reason": ""},
        "performance_before": {"status": "passed", "summary": "10 us", "reason": ""},
        "performance_after": {"status": "passed", "summary": "9 us", "reason": ""},
        "profiling_observation": {
            "status": "not_run",
            "observation": "",
            "reason": "profiler unavailable",
        },
        "result": "Correctness passed and the measured latency improved.",
        "status": "条件有效",
        "evidence": [
            {"kind": "test", "path": ".catlass-dsl/runs/{}/result.json".format(topic)}
        ],
        "kernel_sha256": "a" * 64,
    }


def frontmatter(text):
    assert text.startswith("---\n")
    header, _body = text[4:].split("\n---\n", 1)
    return header


def test_builtin_bundle_has_exact_okf_v02_structure():
    root_index = (ROOT / "knowledge/index.md").read_text(encoding="utf-8")
    assert 'okf_version: "0.2"' in root_index
    for directory in ("dsl", "operator", "debug", "profiler", "optimization", "learned"):
        assert "]({}/index.md)".format(directory) in root_index
    assert not (ROOT / "knowledge/index.yaml").exists()
    assert {path.name for path in (ROOT / "knowledge").iterdir()} == {
        "index.md", "query-vocabulary.yaml", "dsl", "operator", "debug", "profiler",
        "optimization", "learned"
    }
    assert {path.name for path in (ROOT / "knowledge/operator").iterdir()} == {
        "index.md",
        "vector",
        "matmul",
        "flash-attention",
        "sparse-attention",
        "linear-attention",
        "convolution",
    }
    for path in (ROOT / "knowledge").rglob("*.md"):
        if path.name in {"index.md", "log.md"}:
            continue
        header = frontmatter(path.read_text(encoding="utf-8"))
        assert any(line.startswith("type:") and line.split(":", 1)[1].strip() for line in header.splitlines())


def test_static_concepts_have_pinned_sources_and_footnotes():
    source_prefixes = (
        "https://gitcode.com/cann/catlass/blob/"
        "7b574fb3547e76bff47c8514b07741d123a2766b/",
        "https://gitcode.com/cann/catlass/blob/"
        "81da64bca9da5c782f6589541b967456d4fdc4c7/",
        "https://gitcode.com/cann/catlass/blob/"
        "6ccf88e89723b65461e9921047c7970a71b67b42/",
        "https://gitcode.com/m0_53222058/catlass/blob/"
        "e533d4e2aee145e5e5863c2933f95aaf66bab859/",
        "https://gitcode.com/Ascend/msdebug/blob/"
        "77f50d2388c58b3b73279da604fc953ebb21676b/",
        "https://gitcode.com/Ascend/msopprof/blob/"
        "b362f30e7a49ccc5fb80f93f2026332f6001bb82/",
        "https://gitcode.com/cann/ops-transformer/blob/"
        "90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/",
        "https://gitcode.com/cann/ops-nn/blob/"
        "39a50f12554f00809f09eaf0b8a0675477879a4e/",
        "https://gitcode.com/cann/cann-samples/blob/"
        "928d8dfa322731f576b697c9ec997d34abd810b7/",
        "https://github.com/flashserve/flash-linear-attention-npu/blob/"
        "4c07565db3ab0f4bfc1dd20857154688883427ce/",
        "https://github.com/flashserve/flash-linear-attention-npu/blob/"
        "1b2ed3e13a446337d69ab5efbaf64af216adbf06/",
        "project-evidence:",
    )
    allowed_types = {
        "CATLASS DSL API Reference",
        "CATLASS DSL Programming Concept",
        "CATLASS DSL Operator Example",
        "CATLASS DSL Debugging Guide",
        "CATLASS DSL Profiling Guide",
        "CATLASS DSL Optimization Guide",
    }
    for directory in ("dsl", "operator", "debug", "profiler", "optimization"):
        for path in (ROOT / "knowledge" / directory).rglob("*.md"):
            if path.name == "index.md":
                continue
            raw = path.read_text(encoding="utf-8")
            header, body = raw[4:].split("\n---\n", 1)
            metadata = yaml.safe_load(header)
            assert metadata["type"] in allowed_types
            for field in ("title", "description", "tags", "status", "generated", "verified", "sources"):
                assert metadata[field]
            for source in metadata["sources"]:
                assert source["resource"].startswith(source_prefixes)
                assert "[^{}]".format(source["id"]) in body
                assert "[^{}]:".format(source["id"]) in body
            if any(source["resource"].startswith("project-evidence:") for source in metadata["sources"]):
                continue
            for heading in (
                "# 接口与概念",
                "# 用法",
                "# 代码模式",
                "# 约束",
                "# 失败表现",
                "# 验证方法",
            ):
                assert heading in body
            if directory == "operator":
                for heading in (
                    "## 算子算法",
                    "## 分核策略与基本块切分",
                    "## 数据路径与存储层级",
                    "## 流水排布、同步关系与数值精度",
                ):
                    assert heading in body
            assert "git-source:" not in raw
            assert "seed" not in raw.casefold()


def test_core_api_concept_covers_complete_fixed_revision_export_surface():
    text = (ROOT / "knowledge/dsl/core-api.md").read_text(encoding="utf-8")
    exported = {
        "TlaCoreAPIError", "dsl_user_op", "arch", "vec", "mask",
        "create_mask", "update_mask", "tile_view", "make_tensor",
        "make_tensor_like", "copy", "print", "flag", "cross_flag",
        "cross_core_set_flag", "cross_core_wait_flag", "set_flag",
        "wait_flag", "pipe_barrier", "local_mem_bar", "mutex",
        "mutex_guard", "mutex_lock", "mutex_unlock", "range",
        "range_constexpr", "cube", "vector", "mmad", "full", "arange",
        "add", "sub", "mul", "max", "min", "div", "where", "squeeze",
        "bitwise_not", "bitwise_and", "bitwise_or", "bitwise_xor",
        "exp", "log", "sqrt", "abs", "neg", "interleave",
        "deinterleave", "gather", "ReductionOp", "cmp", "make_ptr",
        "allocate", "recast_ptr", "make_shape", "make_coord",
        "make_stride", "make_layout", "IndexTree", "_Pointer",
        "VectorSSA", "MaskSSA", "LocalmemAllocator",
    }
    assert not [name for name in sorted(exported) if "`{}`".format(name) not in text]
    for marker in (
        "compute_order=M_FIRST",
        "hf32_mode=HF32_DISABLE",
        "f8e4m3fn/f8e5m2",
        "i8×i8 到 i32",
        "get_capacity_in_bytes(mem_scope)",
    ):
        assert marker in text


def test_device_printing_concept_has_executable_source_patterns():
    text = (ROOT / "knowledge/debug/device-printing.md").read_text(encoding="utf-8")
    for phrase in (
        "tla.print(format_string, *scalar_values)",
        "def print_scalar",
        "def print_expression",
        "def print_gm",
        "def print_ub",
        "def print_dynamic",
        "print_scalar.dump_mlir(type_args=type_args)",
        "allocation + 8",
        "1 到 262,112 个元素",
        "dynamic-shaped tensors require an explicit length",
        "不要求访问 `sources` URL",
    ):
        assert phrase in text
    assert "python debug_print.py" not in text
    assert "python print_tensor.py" not in text


def test_every_static_concept_is_offline_and_detail_complete():
    required_details = {
        "dsl/tensor-layout-memory.md": ("load(params=None)", "L0C", "resident_bytes"),
        "dsl/copy-and-sync.md": ("cross_flag", "CopyUbToGmParams", "local_mem_bar"),
        "dsl/compile-and-runtime.md": ("from_dlpack", "JitExecutor", "kernel_binary_path"),
        "dsl/python-control-flow.md": ("range_constexpr", "loop-carried", "短路"),
        "dsl/dynamic-layout-and-dlpack.md": (
            "mark_layout_dynamic",
            "mark_compact_shape_dynamic",
            "block_num",
        ),
        "dsl/simt-and-scalar-access.md": (
            "thread_idx",
            "thread_block_dim",
            "tensor indexing",
        ),
        "dsl/extern-ascendc.md": (
            "@tla.extern",
            "tla.Pointer",
            "tla.call_extern",
        ),
        "operator/vector/basic-vadd.md": ("def vadd", "update_mask", "torch.allclose"),
        "operator/vector/vector-ops.md": ("CastParams", "tla.gather", "ReductionOp.ADD"),
        "operator/matmul/basic-mmad.md": ("grid_m", "l1_buf", "CopyL0C2DstParams"),
        "operator/matmul/mixed-and-sync-variants.md": (
            "AtomicMode.ADD",
            "mutex_guard",
            "cross_flag",
        ),
        "operator/matmul/advanced-matmul.md": (
            "Batched matmul",
            "Grouped matmul",
            "mark_layout_dynamic",
        ),
        "operator/flash-attention/flash-attention-infer.md": (
            "Online Softmax",
            "actual_q_seqlen",
            "PagedAttention",
        ),
        "debug/environment-and-build.md": ("TLA_DSL_PREBUILT_ASCENDNPU_IR", "mlir-tblgen", "undefined symbol"),
        "debug/ir-and-lowering.md": ("dump_mlir", "FileCheck", "frontend lowering"),
        "debug/runtime-and-correctness.md": ("from_dlpack", "torch.npu.synchronize", "sentinel"),
        "debug/aicerror-exception-localization.md": (
            "ASCEND_DUMP_SCENE",
            "ascend info summary",
            "register read $PC",
        ),
        "profiler/collection-workflows.md": (
            "msprof op",
            "--kernel-name",
            "OPPROF_{timestamp}_XXX",
        ),
        "profiler/metrics-and-bottlenecks.md": (
            "OpBasicInfo.csv",
            "PipeUtilization.csv",
            "ResourceConflictRatio.csv",
        ),
        "profiler/visualization-analysis.md": (
            "Roofline",
            "MemoryDetail",
            "instrTimeLine",
        ),
        "profiler/simulator-analysis.md": (
            "msprof op simulator",
            "core*_code_exe.csv",
            "trace.json",
        ),
        "optimization/tiling-and-capacity.md": ("L1 A double", "bytes_for", "L0C"),
        "optimization/buffering-and-data-movement.md": ("buf0_ready", "released0", "GM round trip"),
        "optimization/pipeline-and-synchronization.md": ("Barrier 收窄", "unit_flag=0b11", "mmad_done"),
        "optimization/evidence-gates.md": ("metric_path", "observed_improvement", '"trials"'),
    }
    for relative, markers in required_details.items():
        text = (ROOT / "knowledge" / relative).read_text(encoding="utf-8")
        assert text.count("```") >= 4, relative
        assert all(marker in text for marker in markers), relative
        assert "python debug_print.py" not in text
        assert "python print_tensor.py" not in text
        assert "# 失败表现" in text
        assert "# 验证方法" in text


def test_initialize_copies_bundle_without_overwriting(tmp_path):
    knowledge = load_module()
    target = tmp_path / ".catlass-dsl/knowledge"
    copied = knowledge.initialize(ROOT / "knowledge", target)
    assert "index.md" in copied
    assert (target / "dsl/core-api.md").is_file()
    target_index = target / "index.md"
    target_index.write_text("owned", encoding="utf-8")
    knowledge.initialize(ROOT / "knowledge", target)
    assert target_index.read_text(encoding="utf-8") == "owned"


def test_record_writes_okf_concept_and_regenerates_index(tmp_path):
    knowledge = load_module()
    project, commit = git_project(tmp_path)
    knowledge.initialize(ROOT / "knowledge", project / ".catlass-dsl/knowledge")
    result = knowledge.record_entry(valid_entry(project, commit), project, "2026-07-28")
    concept = Path(result["path"])
    text = concept.read_text(encoding="utf-8")
    header = frontmatter(text)
    assert result["okf_version"] == "0.2"
    assert "type: CATLASS DSL Learned Result" in header
    assert "generated:" in header
    assert "verified:" in header
    assert "sources:" in header
    assert "project-evidence:" in header
    assert "result_status:" in header
    assert "../../../.catlass-dsl/runs/block-shape/result.json" in text
    index = (project / ".catlass-dsl/knowledge/learned/index.md").read_text(encoding="utf-8")
    assert "[matmul: block-shape](2026-07-28-matmul-block-shape.md)" in index


def test_record_is_append_only(tmp_path):
    knowledge = load_module()
    project, commit = git_project(tmp_path)
    knowledge.initialize(ROOT / "knowledge", project / ".catlass-dsl/knowledge")
    entry = valid_entry(project, commit)
    knowledge.record_entry(entry, project, "2026-07-28")
    with pytest.raises(FileExistsError):
        knowledge.record_entry(entry, project, "2026-07-28")


def test_record_rejects_missing_evidence_and_invalid_arch(tmp_path):
    knowledge = load_module()
    project, commit = git_project(tmp_path)
    entry = valid_entry(project, commit)
    entry["evidence"][0]["path"] = "missing.json"
    assert any("普通文件" in error for error in knowledge.validate_entry(entry, project))
    entry = valid_entry(project, commit, "other")
    entry["arch"] = ["unknown"]
    assert any("c310" in error for error in knowledge.validate_entry(entry, project))


def test_batch_record_preflights_existing_targets_without_partial_write(tmp_path):
    knowledge = load_module()
    project, commit = git_project(tmp_path)
    knowledge.initialize(ROOT / "knowledge", project / ".catlass-dsl/knowledge")
    knowledge.record_entry(
        valid_entry(project, commit, "pipeline-depth"), project, "2026-07-28"
    )
    index = project / ".catlass-dsl/knowledge/learned/index.md"
    original_index = index.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="pipeline-depth"):
        knowledge.record_entries(
            [
                valid_entry(project, commit, "block-shape"),
                valid_entry(project, commit, "pipeline-depth"),
            ],
            project,
            "2026-07-28",
        )

    assert not (
        project
        / ".catlass-dsl/knowledge/learned/2026-07-28-matmul-block-shape.md"
    ).exists()
    assert index.read_text(encoding="utf-8") == original_index


def test_batch_record_rolls_back_files_after_mid_batch_failure(tmp_path, monkeypatch):
    knowledge = load_module()
    project, commit = git_project(tmp_path)
    knowledge.initialize(ROOT / "knowledge", project / ".catlass-dsl/knowledge")
    index = project / ".catlass-dsl/knowledge/learned/index.md"
    original_index = index.read_text(encoding="utf-8")
    original_write = knowledge._write_exclusive
    calls = []

    def fail_second_write(destination, content):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError("injected write failure")
        original_write(destination, content)

    monkeypatch.setattr(knowledge, "_write_exclusive", fail_second_write)
    with pytest.raises(OSError, match="injected write failure"):
        knowledge.record_entries(
            [
                valid_entry(project, commit, "block-shape"),
                valid_entry(project, commit, "pipeline-depth"),
            ],
            project,
            "2026-07-28",
        )

    for topic in ("block-shape", "pipeline-depth"):
        assert not (
            project
            / ".catlass-dsl/knowledge/learned"
            / "2026-07-28-matmul-{}.md".format(topic)
        ).exists()
    assert index.read_text(encoding="utf-8") == original_index


def test_learned_template_is_an_okf_concept():
    template = (ROOT / "skills/catlass-dsl-knowledge/templates/learned-entry.md").read_text(
        encoding="utf-8"
    )
    header = frontmatter(template)
    assert "type: CATLASS DSL Learned Result" in header
    for marker in (
        "{{title_json}}",
        "{{description_json}}",
        "{{generated_json}}",
        "{{verified_json}}",
        "{{sources_json}}",
    ):
        assert marker in header
    assert "metadata_json" not in template


def test_cli_resolves_builtin_bundle_and_reports_okf_version(tmp_path):
    project, _commit = git_project(tmp_path)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "initialize", "--project-root", str(project)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "passed"
    assert payload["okf_version"] == "0.2"
    assert (project / ".catlass-dsl/knowledge/index.md").is_file()
    assert (project / ".catlass-dsl/knowledge/query-vocabulary.yaml").is_file()
    assert not (project / "knowledge").exists()
    queried = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "query",
            "--project-root",
            str(project),
            "--text",
            "make_shape",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert queried.returncode == 0, queried.stderr
    query_payload = json.loads(queried.stdout)
    assert query_payload["results"][0]["path"].startswith(
        ".catlass-dsl/knowledge/"
    )
    assert query_payload["count"] <= 20
    assert query_payload["total_count"] >= query_payload["count"]
    assert query_payload["query"]["normalized_text"] == "make shape"
    assert query_payload["suggestions"] == []
    assert query_payload["results"][0]["tags"]
    assert query_payload["results"][0]["verified"]
    assert query_payload["results"][0]["sources"]
    assert "body_snippets" in query_payload["results"][0]


def test_compact_query_snippets_and_get_form_progressive_retrieval(tmp_path):
    knowledge = load_module()
    project, _commit = git_project(tmp_path)
    knowledge.initialize(ROOT / "knowledge", project / ".catlass-dsl/knowledge")
    queried = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "query",
            "--project-root",
            str(project),
            "--text",
            "make_shape",
            "--limit",
            "1",
            "--compact",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert queried.returncode == 0, queried.stderr
    query_payload = json.loads(queried.stdout)
    match = query_payload["results"][0]
    assert set(match) == {
        "path",
        "type",
        "title",
        "description",
        "status",
        "score",
        "matched_fields",
        "matched_terms",
        "body_snippets",
    }
    assert 0 < len(match["body_snippets"]) <= knowledge.QUERY_SNIPPET_LIMIT
    assert all(snippet["line"] > 1 for snippet in match["body_snippets"])
    assert all(len(snippet["text"]) <= knowledge.QUERY_SNIPPET_CHARS for snippet in match["body_snippets"])

    fetched = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "get",
            "--project-root",
            str(project),
            "--path",
            match["path"],
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert fetched.returncode == 0, fetched.stderr
    get_payload = json.loads(fetched.stdout)
    assert get_payload["concept"]["path"] == match["path"]
    assert get_payload["concept"]["metadata"]["sources"]
    assert "make_shape" in get_payload["concept"]["body"]

    with pytest.raises(ValueError, match="安全项目相对路径"):
        knowledge.get_concept(project / ".catlass-dsl/knowledge", "../outside.md")


def test_validate_query_reindex_and_batch_record(tmp_path):
    knowledge = load_module()
    project, commit = git_project(tmp_path)
    knowledge.initialize(ROOT / "knowledge", project / ".catlass-dsl/knowledge")
    validated = knowledge.validate_bundle(project / ".catlass-dsl/knowledge")
    assert validated["okf_version"] == "0.2"
    assert validated["count"] > 0
    matches = knowledge.query_bundle(
        project / ".catlass-dsl/knowledge", operator_family="matmul", text="tile"
    )
    assert matches
    entries = [
        valid_entry(project, commit, "block-shape"),
        valid_entry(project, commit, "pipeline-depth"),
    ]
    result = knowledge.record_entries(entries, project, "2026-07-28")
    assert result["count"] == 2
    index = (project / ".catlass-dsl/knowledge/learned/index.md").read_text(encoding="utf-8")
    assert "block-shape" in index
    assert "pipeline-depth" in index
    assert knowledge.reindex(project / ".catlass-dsl/knowledge")["status"] == "passed"


def test_hybrid_query_normalizes_gdn_aliases_and_ranks_primary_operator():
    knowledge = load_module()
    report = knowledge.query_bundle_report(ROOT / "knowledge", text="CHUNK_GDN2")
    assert report["query"] == {
        "normalized_text": "chunk gated-delta-rule",
        "normalized_operator_family": None,
        "match_mode": "exact",
    }
    assert report["results"][0]["path"] == (
        "operator/linear-attention/gdn/chunk-gated-delta-rule.md"
    )
    assert report["results"][0]["score"] > 0
    assert "operator_families" in report["results"][0]["matched_fields"]
    assert report["results"][0]["matched_terms"] == [
        "chunk", "gated-delta-rule"
    ]

    hyphenated = knowledge.query_bundle_report(ROOT / "knowledge", text="chunk-gdn2")
    assert hyphenated["results"][0]["path"] == report["results"][0]["path"]

    recurrent = knowledge.query_bundle_report(ROOT / "knowledge", text="recurrent_gdn2")
    assert recurrent["results"][0]["path"] == (
        "operator/linear-attention/gdn/recurrent-gated-delta-rule.md"
    )


def test_operator_family_alias_is_a_hard_filter_and_never_matches_source_urls():
    knowledge = load_module()
    report = knowledge.query_bundle_report(
        ROOT / "knowledge", operator_family="chunk_gdn2", arch="c310"
    )
    assert report["query"]["normalized_operator_family"] == "gated-delta-rule"
    assert report["results"]
    assert all(
        "/linear-attention/gdn/" in result["path"]
        or result["path"]
        == "operator/linear-attention/kda/chunk-gated-delta-rule-fwd-h.md"
        for result in report["results"]
    )
    assert not any("convolution" in result["path"] for result in report["results"])
    assert [result["path"] for result in report["results"]] == sorted(
        result["path"] for result in report["results"]
    )


def test_hybrid_query_relaxes_text_but_not_structured_filters_and_honors_limit():
    knowledge = load_module()
    report = knowledge.query_bundle_report(
        ROOT / "knowledge",
        concept_type="CATLASS DSL Operator Example",
        arch="c310",
        text="chunk nonexistent-term",
        limit=2,
    )
    assert report["query"]["match_mode"] == "relaxed"
    assert report["total_count"] >= len(report["results"]) == 2
    assert all(
        result["type"] == "CATLASS DSL Operator Example"
        for result in report["results"]
    )
    assert all(result["score"] >= 0 for result in report["results"])


def test_hybrid_query_suggests_known_family_without_fuzzy_auto_match():
    knowledge = load_module()
    report = knowledge.query_bundle_report(ROOT / "knowledge", text="gdn3")
    assert report["results"] == []
    assert report["suggestions"]
    assert report["suggestions"][0]["normalized_operator_family"] == (
        "gated-delta-rule"
    )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("schema_version: \"1\"", "schema_version"),
        ("      - gdn2", "重复别名"),
        ("  gated-delta-rule:", "别名冲突"),
        ("gated-delta-rule", "规范算子族不存在"),
    ],
)
def test_validate_rejects_invalid_query_vocabulary(tmp_path, replacement, message):
    knowledge = load_module()
    bundle = tmp_path / "knowledge"
    shutil.copytree(ROOT / "knowledge", bundle)
    vocabulary = bundle / "query-vocabulary.yaml"
    original = vocabulary.read_text(encoding="utf-8")
    if message == "schema_version":
        vocabulary.write_text(original.replace(replacement, 'schema_version: "2"'), encoding="utf-8")
    elif message == "重复别名":
        vocabulary.write_text(original.replace(replacement, replacement + "\n" + replacement), encoding="utf-8")
    elif message == "别名冲突":
        vocabulary.write_text(
            original + "\n  matmul:\n    aliases:\n      - gdn\n",
            encoding="utf-8",
        )
    else:
        vocabulary.write_text(original.replace(replacement, "missing-family"), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        knowledge.validate_bundle(bundle)


@pytest.mark.parametrize(
    ("filters", "expected_path"),
    [
        ({"text": "make_shape"}, "dsl/core-api.md"),
        ({"text": "VADD"}, "operator/vector/basic-vadd.md"),
        ({"operator_family": "matmul", "text": "unit_flag"}, "operator/matmul/basic-mmad.md"),
        ({"text": "编译错误"}, "debug/environment-and-build.md"),
        ({"text": "aicerror"}, "debug/aicerror-exception-localization.md"),
        ({"text": "msopprof"}, "profiler/collection-workflows.md"),
        ({"text": "ResourceConflictRatio"}, "profiler/metrics-and-bottlenecks.md"),
        ({"text": "指令流水图"}, "profiler/visualization-analysis.md"),
        ({"tags": ["simulator", "profiler"]}, "profiler/simulator-analysis.md"),
            ({"text": "tiling"}, "optimization/tiling-and-capacity.md"),
            (
                {"text": "scaleKL1Ratio"},
                "optimization/matmul-memory-and-layout.md",
            ),
            (
                {"text": "prefetch_distance"},
                "optimization/matmul-pipeline-and-preload.md",
            ),
            (
                {"text": "tail_tasks / core_count"},
                "optimization/multicore-load-balancing.md",
            ),
            (
                {"text": "SubBlock 映射"},
                "optimization/vector-register-and-cv-path.md",
            ),
            (
                {"text": "legal_pairs"},
                "optimization/bandwidth-template-selection.md",
            ),
            (
                {"text": "task 级 Q/O"},
                "optimization/attention-pipelines.md",
            ),
            (
                {"text": "跨 group 连续排核"},
                "optimization/matmul-and-grouped.md",
            ),
            (
                {"text": "binary fold"},
                "optimization/regbase-and-reduction.md",
            ),
            (
                {"text": "grid-stride"},
                "optimization/simt-irregular.md",
            ),
            (
                {"text": "run_middle_tile_no_tail_branch"},
                "optimization/scalar-codegen.md",
            ),
            (
                {"text": "quant_ready"},
                "optimization/moe-communication.md",
            ),
            ({"tags": ["synchronization"], "arch": "c310"}, "dsl/copy-and-sync.md"),
        ({"text": "range_constexpr"}, "dsl/python-control-flow.md"),
        ({"text": "mark_layout_dynamic"}, "dsl/dynamic-layout-and-dlpack.md"),
        ({"tags": ["simt"]}, "dsl/simt-and-scalar-access.md"),
        ({"text": "Grouped matmul"}, "operator/matmul/advanced-matmul.md"),
        ({"operator_family": "flash-attention"}, "operator/flash-attention/flash-attention-infer.md"),
    ],
)
def test_query_finds_source_backed_knowledge(filters, expected_path):
    knowledge = load_module()
    results = knowledge.query_bundle(ROOT / "knowledge", **filters)
    assert expected_path in {result["path"] for result in results}


def test_validate_rejects_invalid_actor_status_source_footnote_and_index(tmp_path):
    knowledge = load_module()
    bundle = tmp_path / "knowledge"
    shutil.copytree(ROOT / "knowledge", bundle)
    concept = bundle / "dsl/core-api.md"
    original = concept.read_text(encoding="utf-8")

    concept.write_text(original.replace("process:catlass-dsl-source-extract", "unknown actor", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="producer/version"):
        knowledge.validate_bundle(bundle)

    concept.write_text(original.replace("status: stable", "status: current", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="status"):
        knowledge.validate_bundle(bundle)

    concept.write_text(original.replace("https://gitcode.com/", "git-source:legacy/", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="source scheme"):
        knowledge.validate_bundle(bundle)

    concept.write_text(original.replace("[^api]:", "[^missing]:", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="source footnote"):
        knowledge.validate_bundle(bundle)

    concept.write_text(original, encoding="utf-8")
    root_index = bundle / "index.md"
    root_index_original = root_index.read_text(encoding="utf-8")
    root_index.write_text(
        root_index_original.replace("dsl/index.md", "dsl/", 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="子目录链接必须显式指向 index.md"):
        knowledge.validate_bundle(bundle)
    root_index.write_text(root_index_original, encoding="utf-8")

    index = bundle / "dsl/index.md"
    index.write_text(index.read_text(encoding="utf-8") + "\n- [missing](missing.md)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="索引链接不存在"):
        knowledge.validate_bundle(bundle)


def test_skill_documents_okf_contract():
    skill = (ROOT / "skills/catlass-dsl-knowledge/SKILL.md").read_text(encoding="utf-8")
    for phrase in (
        "Open Knowledge Format v0.2",
        "knowledge/index.md",
        "type",
        "sources",
        "generated",
        "verified",
        "stale_after",
        "project-evidence:",
        "knowledge/learned/index.md",
        "内置",
        "knowledge/operator/",
        "算子算法",
        "分核策略与基本块切分",
        "数据路径与存储层级",
        "流水排布、同步关系与数值精度",
        "<子目录>/index.md",
        "knowledge/profiler/",
        "--compact",
        "get --path",
        "同一锁内预检全部目标",
    ):
        assert phrase in skill
    assert "seed" not in skill.casefold()


def test_skill_routes_concept_writes_to_split_entry_standards_references():
    skill = (ROOT / "skills/catlass-dsl-knowledge/SKILL.md").read_text(encoding="utf-8")
    references_dir = ROOT / "skills/catlass-dsl-knowledge/references"
    expected_references = {
        "common-entry-standards.md",
        "operator-entry-standards.md",
        "optimization-entry-standards.md",
        "dsl-entry-standards.md",
        "profiler-entry-standards.md",
        "debug-entry-standards.md",
        "learned-entry-standards.md",
    }

    for phrase in (
        "## 入库标准路由",
        "references/common-entry-standards.md",
        "references/operator-entry-standards.md",
        "references/optimization-entry-standards.md",
        "references/dsl-entry-standards.md",
        "references/profiler-entry-standards.md",
        "references/debug-entry-standards.md",
        "references/learned-entry-standards.md",
        "必须先完整阅读并执行",
        "再按目标目录完整阅读并执行",
        "创建、修改、迁移或审核",
        "所有相关目录",
    ):
        assert phrase in skill

    assert {path.name for path in references_dir.glob("*.md")} == expected_references
    assert not (references_dir / "knowledge-entry-standards.md").exists()

    common = (references_dir / "common-entry-standards.md").read_text(
        encoding="utf-8"
    )
    for phrase in (
        "# CATLASS DSL 知识公共入库标准",
        "## 证据能力与目录选择",
        "## OKF 与来源门禁",
        "## 静态 concept 的公共正文",
        "## 事实、推断和实测分层",
        "## 索引与演进",
        "## 统一入库工作流",
        "## 统一验收问题",
        "knowledge/operator/",
        "knowledge/optimization/",
        "knowledge/learned/",
        "knowledge/dsl/",
        "knowledge/profiler/",
        "knowledge/debug/",
    ):
        assert phrase in common

    specialized_requirements = {
        "operator-entry-standards.md": (
            "# Operator 入库标准",
            "目标只有一个",
            "## 取证范围",
            "host tiling/路径选择",
            "kernel 入口",
            "目标架构实现",
            "不复述完整接口",
            "## 核心内容",
            "路径",
            "任务",
            "数据与同步",
            "成本与观测",
            "候选与验证",
            "单一优化轴",
            "预期 profiler 变化",
            "## 删除规则",
            "完整签名",
            "源码逐文件复述",
            "提高并行度",
            "减少访存",
            "没有对象、条件、代价",
            "## 验收",
            "可证伪、可回退",
        ),
        "optimization-entry-standards.md": (
            "# Optimization 入库标准",
            "适用范围",
            "可证伪假设",
            "失败与回退",
            "验证合同",
        ),
        "dsl-entry-standards.md": (
            "# DSL 入库标准",
            "取证优先级",
            "可执行模式",
            "验证层",
        ),
        "profiler-entry-standards.md": (
            "# Profiler 入库标准",
            "两类 concept",
            "工具合同",
            "样本有效性",
        ),
        "debug-entry-standards.md": (
            "# Debug 入库标准",
            "症状指纹",
            "最小复现",
            "诊断树",
        ),
        "learned-entry-standards.md": (
            "# Learned 入库标准",
            "唯一写入入口与字段合同",
            "correctness_after.status",
            "kernel_sha256",
            "追加与冲突",
        ),
    }
    for filename, phrases in specialized_requirements.items():
        reference = (references_dir / filename).read_text(encoding="utf-8")
        assert "(common-entry-standards.md)" in reference
        for phrase in phrases:
            assert phrase in reference
