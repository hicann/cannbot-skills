# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Declarative records shared by the autoresearch framework.

The field tables below are the schema: they keep defaults visible in one place
while avoiding behavior in configuration containers. ``make_dataclass`` gives
callers the same constructor, equality, representation, and ``asdict`` support
as handwritten dataclasses.
"""

from dataclasses import field, make_dataclass
from typing import Optional


class _FactoryDefault:
    def __init__(self, constructor):
        self.constructor = constructor


def _factory(container):
    return _FactoryDefault(container)


def _record(name: str, schema: list[tuple], *, bases: tuple = ()):
    return make_dataclass(
        name,
        [
            (
                spec[0],
                spec[1],
                field(default_factory=spec[2].constructor),
            )
            if len(spec) == 3 and isinstance(spec[2], _FactoryDefault)
            else spec
            for spec in schema
        ],
        bases=bases,
        namespace={"__module__": __name__},
    )


def _add_required(schema: list[tuple], **annotations) -> None:
    schema.extend(annotations.items())


def _add_defaults(schema: list[tuple], annotation, **values) -> None:
    schema.extend((name, annotation, value) for name, value in values.items())


def _add_factories(schema: list[tuple], annotation, **constructors) -> None:
    schema.extend(
        (name, annotation, _factory(constructor))
        for name, constructor in constructors.items()
    )


_AGENT_SCHEMA: list[tuple] = []
_add_defaults(
    _AGENT_SCHEMA,
    int,
    max_consecutive_failures=10,
    max_no_edit_turns=3,
    max_turns_multiplier=8,
    chars_per_token=3,
    editable_file_truncate=8_000,
    system_context_file_truncate=15_000,
    system_context_total_truncate=40_000,
    system_fundamentals_max_chars=20_000,
    plan_max_chars=4_000,
    finish_hint_threshold=2,
    log_arg_truncate=500,
    log_result_truncate=1_000,
    cumulative_diff_truncate=10_000,
    smoke_output_limit=2_000,
    raw_output_tail=2_048,
    llm_max_tokens=8_192,
    thinking_budget=8_000,
)
_add_defaults(
    _AGENT_SCHEMA,
    float,
    call_timeout=120.0,
    retry_initial_backoff=5.0,
    retry_max_backoff_rate_limit=120.0,
    retry_max_backoff_other=60.0,
)
_add_defaults(_AGENT_SCHEMA, int, llm_max_retries=5)
_add_defaults(_AGENT_SCHEMA, float, llm_connection_check_timeout=15.0)
_add_defaults(_AGENT_SCHEMA, int | None, context_limit=150_000)
_add_defaults(_AGENT_SCHEMA, float, compression_threshold=0.75)
_add_defaults(
    _AGENT_SCHEMA,
    int,
    microcompact_min_chars=200,
    microcompact_keep_recent=1,
    compact_min_messages=4,
    compact_max_retries=3,
    compact_diagnosis_truncate=2_000,
)
_add_defaults(_AGENT_SCHEMA, float, compact_post_check_ratio=0.9)
_add_defaults(
    _AGENT_SCHEMA,
    int,
    compact_max_failures=3,
    compact_emergency_keep_rounds=1,
    compact_keep_recent_rounds=3,
    compact_op_summary_max_tokens=500,
    compact_plan_analysis_max_tokens=1_500,
    compact_kernel_sanity_cap=80_000,
    compact_rebuild_kernel_cap=20_000,
    compact_rebuild_ranking_cap=8_000,
    compact_plan_raw_fallback_chars=6_000,
    replanning_max_idle_turns=2,
    eval_feedback_tail=1_000,
    log_raw_output_truncate=4_096,
    history_summary_last_n=10,
    ranking_description_truncate=100,
    ranking_error_truncate=120,
    compact_ranking_max_entries=5,
    skill_block_max_chars=8_000,
    skill_block_top_k=5,
    skill_keyword_max_per_item=5,
)
_add_defaults(_AGENT_SCHEMA, float, skill_narrow_timeout=30.0)
_add_defaults(
    _AGENT_SCHEMA,
    int,
    plan_item_rationale_min_chars=30,
    plan_item_rationale_max_chars=400,
    min_items_per_plan=3,
    skill_inject_max_chars=6_000,
    diagnose_suggest_threshold=3,
    subagent_code_truncate=8_000,
    subagent_result_truncate=10_000,
    subagent_max_iterations=15,
)
_add_defaults(
    _AGENT_SCHEMA,
    str,
    session_dir="agent_session",
    heartbeat_file="RUNNING",
)

AgentConfig = _record("AgentConfig", _AGENT_SCHEMA)
AgentConfig.__doc__ = "Framework-level limits and retry policy for an agent run."


def _guardrail_defaults() -> dict:
    return {"content": [], "diff": [], "diff_any": []}


_TASK_SCHEMA: list[tuple] = []
_add_required(_TASK_SCHEMA, name=str, description=str)
_add_defaults(
    _TASK_SCHEMA,
    Optional[str],
    dsl=None,
    framework=None,
    backend=None,
    arch=None,
)
_add_factories(_TASK_SCHEMA, dict, dsl_config=dict)
_add_defaults(_TASK_SCHEMA, Optional[str], eval_script=None)
_add_factories(_TASK_SCHEMA, list[str], editable_files=list)
_add_defaults(_TASK_SCHEMA, int, eval_timeout=600)
_add_defaults(_TASK_SCHEMA, str, primary_metric="score")
_add_defaults(_TASK_SCHEMA, bool, lower_is_better=True)
_add_defaults(_TASK_SCHEMA, float, improvement_threshold=0.0)
_add_factories(_TASK_SCHEMA, dict, constraints=dict)
_add_defaults(_TASK_SCHEMA, Optional[str], smoke_test_script=None)
_add_defaults(
    _TASK_SCHEMA,
    int,
    smoke_test_timeout=10,
    import_timeout=15,
    max_patch_size=15_000,
)
_add_factories(
    _TASK_SCHEMA,
    dict,
    forbidden_patterns=_guardrail_defaults,
)
_add_defaults(_TASK_SCHEMA, Optional[str], program_file=None, ref_file=None)
_add_factories(_TASK_SCHEMA, list[str], context_files=list)
_add_defaults(_TASK_SCHEMA, bool, git_push=False)
_add_defaults(_TASK_SCHEMA, Optional[str], git_branch=None)
_add_defaults(_TASK_SCHEMA, int, max_rounds=30)
_add_factories(_TASK_SCHEMA, AgentConfig, agent=AgentConfig)
_add_factories(_TASK_SCHEMA, dict, metadata=dict)

TaskConfig = _record("TaskConfig", _TASK_SCHEMA)
TaskConfig.__doc__ = "Complete declarative configuration for one optimization task."


class _CommitBehavior:
    hash: Optional[str]
    nothing_to_commit: bool

    @property
    def committed(self) -> bool:
        return self.hash is not None

    @property
    def ok(self) -> bool:
        return any((self.committed, self.nothing_to_commit))


CommitResult = _record(
    "CommitResult",
    [
        ("hash", Optional[str], None),
        ("nothing_to_commit", bool, False),
        ("error", Optional[str], None),
    ],
    bases=(_CommitBehavior,),
)
CommitResult.__doc__ = "Outcome of a repository commit attempt."


class _EvaluationBehavior:
    metrics: dict

    def get_metric(self, key: str, default=None):
        return dict.get(self.metrics, key, default)


EvalResult = _record(
    "EvalResult",
    [
        ("correctness", bool),
        ("metrics", dict, _factory(dict)),
        ("error", Optional[str], None),
        ("raw_output", str, ""),
    ],
    bases=(_EvaluationBehavior,),
)
EvalResult.__doc__ = "Result of one framework evaluation."


RoundRecord = _record(
    "RoundRecord",
    [
        ("round_num", int),
        ("description", str),
        ("result", EvalResult),
        ("accepted", bool),
        ("commit_hash", Optional[str], None),
        ("duration_sec", float, 0.0),
        ("constraint_violations", list[str], _factory(list)),
    ],
)
RoundRecord.__doc__ = "Persistent summary of one optimization round."
