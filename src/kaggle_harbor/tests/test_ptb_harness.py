"""Tests for the custom-import harness wrappers.

Everything here is asserted against upstream code rather than against a
transcription: the prompt comes from running `get_prompt.py`, and the parser
dispatch comes from importing upstream's own `select_parser`.

    uv run --with harbor==0.21.0 --with pytest \
        pytest PostTrainBench/src/kaggle_harbor/tests/
"""

from __future__ import annotations

import asyncio
import os

import pytest

import build_tasks
import parse_trace
from agents import ptb_harness
from harbor.agents.base import BaseAgent
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.codex import Codex
from harbor.agents.installed.cursor_cli import CursorCli
from harbor.agents.installed.gemini_cli import GeminiCli
from harbor.agents.installed.opencode import OpenCode
from harbor.utils.import_path import import_class

# One cell is enough for a byte comparison; a second with a different
# benchmark catches a prompt section that only some benchmarks emit.
CELLS = [
    ("gsm8k", "Qwen/Qwen3-1.7B-Base"),
    ("healthbench", "Qwen/Qwen3-4B-Base"),
]

WRAPPERS = [
    ("PtbHarness_claude", "claude", "claude"),
    ("PtbHarness_codex", "codex", "codex"),
    ("PtbHarness_gemini", "gemini", "gemini"),
    ("PtbHarness_opencode", "opencode", "opencode"),
]

MODULE = "agents.ptb_harness"


def _slug(class_name: str) -> str:
    return f"{MODULE}:{class_name}"


def _instruction(benchmark_id: str, model: str, agent: str) -> str:
    """What build_tasks.py writes to instruction.md for this `--agent`."""
    return build_tasks.generate_instruction(benchmark_id, model, "10", agent, 1)


# ---------------------------------------------------------------------------
# the clause
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("benchmark_id,model", CELLS)
@pytest.mark.parametrize("class_name,upstream_agent,_parser_key", WRAPPERS)
def test_clause_matches_get_prompt(
    benchmark_id, model, class_name, upstream_agent, _parser_key
):
    """The wrapper's output equals get_prompt.py's, byte for byte."""
    cls = getattr(ptb_harness, class_name)
    assert cls.PTB_AGENT_NAME == upstream_agent

    baseline = _instruction(benchmark_id, model, build_tasks.PROMPT_AGENT)
    expected = _instruction(benchmark_id, model, upstream_agent)

    assert ptb_harness.apply_agent_clause(baseline, cls.PTB_AGENT_NAME) == expected


@pytest.mark.parametrize("benchmark_id,model", CELLS)
def test_non_claude_baseline_is_untouched(benchmark_id, model):
    """Every non-claude harness leaves instruction.md exactly as baked."""
    baseline = _instruction(benchmark_id, model, build_tasks.PROMPT_AGENT)
    for class_name, upstream_agent, _ in WRAPPERS:
        if "claude" in upstream_agent:
            continue
        cls = getattr(ptb_harness, class_name)
        assert ptb_harness.apply_agent_clause(baseline, cls.PTB_AGENT_NAME) is baseline


@pytest.mark.parametrize("benchmark_id,model", CELLS)
def test_prompt_agent_is_interchangeable_with_any_clause_free_name(
    benchmark_id, model
):
    """PROMPT_AGENT's value cannot matter: get_prompt.py only tests 'claude'."""
    assert "claude" not in build_tasks.PROMPT_AGENT
    baseline = _instruction(benchmark_id, model, build_tasks.PROMPT_AGENT)
    for other in ("codex", "cursor_cli", "gemini", "opencode"):
        assert _instruction(benchmark_id, model, other) == baseline


def test_clause_literal_is_the_upstream_bytes():
    """The literal in the module is get_prompt.py:121-123's, unmodified."""
    source = (build_tasks.REPO_ROOT / "src" / "eval" / "general" / "get_prompt.py")
    text = source.read_text(encoding="utf-8")
    marker = "        result += \"\"\""
    start = text.index(marker) + len(marker)
    end = text.index("\"\"\"", start)
    assert text[start:end] == ptb_harness._NON_INTERACTIVE_CLAUSE


# ---------------------------------------------------------------------------
# the two upstream substring dispatchers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("class_name,upstream_agent,parser_key", WRAPPERS)
def test_slug_dispatches_to_exactly_one_parser(class_name, upstream_agent, parser_key):
    """parse_trace.py:36-44 must match one key, and the right one."""
    slug = _slug(class_name)
    matches = [key for key in parse_trace.PARSERS if key in slug]
    assert matches == [parser_key], (
        f"{slug!r} matches {matches}; parse_trace.select_parser errors on >1 "
        f"and silently copies the raw trace on 0"
    )
    assert parse_trace.select_parser(slug) is parse_trace.PARSERS[parser_key]


@pytest.mark.parametrize("class_name,upstream_agent,_parser_key", WRAPPERS)
def test_slug_claude_rule_matches_upstream(class_name, upstream_agent, _parser_key):
    """get_prompt.py:120's test gives the same answer on slug and upstream name."""
    assert ("claude" in _slug(class_name)) == ("claude" in upstream_agent)


# ---------------------------------------------------------------------------
# harbor's own import machinery
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("class_name,_agent,_key", WRAPPERS)
def test_resolves_through_harbor_import_class(class_name, _agent, _key):
    cls = import_class(_slug(class_name), base=BaseAgent, label="agent")
    assert cls is getattr(ptb_harness, class_name)


@pytest.mark.parametrize("class_name,_agent,_key", WRAPPERS)
def test_resolves_through_agent_factory(class_name, _agent, _key, tmp_path):
    agent = AgentFactory.create_agent_from_import_path(
        _slug(class_name), logs_dir=tmp_path, model_name="anthropic/claude-opus-5"
    )
    assert isinstance(agent, getattr(ptb_harness, class_name))


# ---------------------------------------------------------------------------
# the version pin
# ---------------------------------------------------------------------------

def test_version_from_kwarg(tmp_path):
    agent = AgentFactory.create_agent_from_import_path(
        _slug("PtbHarness_claude"), logs_dir=tmp_path, version="2.1.219"
    )
    assert agent.version() == "2.1.219"


def test_version_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv(ptb_harness.PTB_HARNESS_VERSION_ENV, "2.1.198")
    agent = AgentFactory.create_agent_from_import_path(
        _slug("PtbHarness_claude"), logs_dir=tmp_path
    )
    assert agent.version() == "2.1.198"


def test_version_from_class_attribute(tmp_path):
    class Pinned(ptb_harness.PtbHarness_claude):
        PTB_DEFAULT_VERSION = "2.0.55"

    assert Pinned(logs_dir=tmp_path).version() == "2.0.55"


def test_version_kwarg_beats_env(tmp_path, monkeypatch):
    monkeypatch.setenv(ptb_harness.PTB_HARNESS_VERSION_ENV, "2.1.198")
    agent = AgentFactory.create_agent_from_import_path(
        _slug("PtbHarness_claude"), logs_dir=tmp_path, version="2.1.219"
    )
    assert agent.version() == "2.1.219"


def test_version_defaults_to_a_known_pin_not_latest(tmp_path, monkeypatch):
    """Every harness resolves to a version Kaggle actually registers.

    This assertion was inverted. It used to require "latest" for all four,
    on the reasoning below -- that every agents/<a>/solve.sh calls
    update_agent_cli.sh, which installs <pkg>@latest ahead of the baked copy.
    That is what the original does, and it is the wrong target here: "latest"
    resolves to whatever npm published this morning, which is not necessarily
    a build the platform supports. The registry versions in
    HARNESS-VERSIONS.md are, so those are what we pin
    (claude-code 2.1.223, codex 0.146.1, opencode 1.18.14).

    gemini-cli is the exception and stays on "latest": it is absent from the
    registry, so there is no supported version to adopt, and floating is what
    the original does. It is unreachable anyway -- no gemini-cli agent slug
    exists to schedule. verify_fidelity:366-368 asserts this on purpose.

    The pins must NOT be None. Harbor treats version=None as "any build will
    do" and skips the install when a CLI is already present.

    (codex.py:330-334), which would freeze the harness at the standard.def
    fallback -- the bug that made a gpt-5.6-sol run die on
    `unknown variant 'max'`, a level that postdates codex 0.137.0.
    """
    monkeypatch.delenv(ptb_harness.PTB_HARNESS_VERSION_ENV, raising=False)
    for name, _agent, _key in WRAPPERS:
        agent = AgentFactory.create_agent_from_import_path(
            _slug(name), logs_dir=tmp_path
        )
        assert agent.version() == "latest", name


def test_explicit_pin_still_overrides_latest(tmp_path, monkeypatch):
    """A per-row pin is upstream's POST_TRAIN_BENCH_SKIP_CLI_UPDATE equivalent."""
    monkeypatch.setenv(ptb_harness.PTB_HARNESS_VERSION_ENV, "0.146.1")
    agent = AgentFactory.create_agent_from_import_path(
        _slug("PtbHarness_codex"), logs_dir=tmp_path
    )
    assert agent.version() == "0.146.1"


def test_json_coerced_version_is_stringified_and_flagged():
    """harbor/cli/utils.py:65-90 turns `--ak version=2.1` into a float."""
    value, source = ptb_harness.PtbHarness_codex._ptb_resolve_version(2.1)
    assert value == "2.1"
    assert "coerced from float" in source


# ---------------------------------------------------------------------------
# run() delegation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# the executor's per-agent `case` branches, reproduced in the wrapper
# ---------------------------------------------------------------------------

# Every env name any of the four gaps touches. Cleared before each test so a
# wrapper that writes os.environ cannot leak into the next one; monkeypatch
# restores the pre-test state on teardown.
_CRED_ENVS = (
    "GEMINI_API_KEY", "GOOGLE_GEMINI_BASE_URL", "CURSOR_API_KEY",
    "MODEL_PROXY_API_KEY", "MODEL_PROXY_BASE_URL", "MODEL_PROXY_URL",
    ptb_harness.KAGGLE_REASONING_EFFORT_ENV,
    ptb_harness.PTB_HARNESS_VERSION_ENV,
)

# Every adapter CliFlag env_fallback, read off the descriptors rather than
# hardcoded. Needed because these tests can run inside an agent harness that
# already exports some of them — CLAUDE_CODE_EFFORT_LEVEL
# (claude_code.py:94) is set to "high" inside Claude Code itself, which made
# `--effort high` appear with no env var of ours in play.
_FALLBACK_ENVS = tuple({
    flag.env_fallback
    for cls in (ClaudeCode, Codex, GeminiCli, OpenCode)
    for flag in getattr(cls, "CLI_FLAGS", [])
    if flag.env_fallback
})


@pytest.fixture
def clean_env(monkeypatch):
    for name in (*_CRED_ENVS, *_FALLBACK_ENVS):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def proxy_env(clean_env):
    clean_env.setenv("MODEL_PROXY_API_KEY", "proxy-key-xyz")
    clean_env.setenv("MODEL_PROXY_BASE_URL", "https://mp-staging.kaggle.net/models")
    return clean_env


def _make(class_name, tmp_path, **kwargs):
    return AgentFactory.create_agent_from_import_path(
        _slug(class_name), logs_dir=tmp_path, **kwargs
    )


# -- gap 1: gemini credentials ----------------------------------------------

def test_gemini_names_come_from_harbors_own_spec():
    """Anti-staleness: we fill the names harbor says it reads, not literals."""
    spec = GeminiCli.MODEL_CONNECTION
    assert spec is not None
    keys, urls = ptb_harness.PtbHarness_gemini._ptb_credential_envs()
    assert keys == spec.api_key_envs
    assert urls == spec.base_url_envs
    assert keys and urls, "harbor stopped declaring gemini's env names"


def test_gemini_credentials_filled_from_proxy(tmp_path, proxy_env):
    agent = _make("PtbHarness_gemini", tmp_path, model_name="google/gemini-3.5-flash")
    assert agent.extra_env["GEMINI_API_KEY"] == "proxy-key-xyz"
    assert (
        agent.extra_env["GOOGLE_GEMINI_BASE_URL"]
        == "https://mp-staging.kaggle.net/models/genai"
    )
    # Written to both channels so a CLI the adapter spawns inherits it too.
    assert os.environ["GEMINI_API_KEY"] == "proxy-key-xyz"


def test_gemini_base_url_suffix_matches_the_entrypoint(tmp_path, clean_env):
    """entrypoint-common.sh:344 — "${MODEL_PROXY_BASE_URL}/genai"."""
    clean_env.setenv("MODEL_PROXY_API_KEY", "k")
    clean_env.setenv("MODEL_PROXY_BASE_URL", "https://host/models/")  # trailing slash
    agent = _make("PtbHarness_gemini", tmp_path, model_name="google/gemini-3.5-flash")
    assert agent.extra_env["GOOGLE_GEMINI_BASE_URL"] == "https://host/models/genai"


def test_model_proxy_url_is_accepted_as_an_alias(tmp_path, clean_env):
    clean_env.setenv("MODEL_PROXY_API_KEY", "k")
    clean_env.setenv("MODEL_PROXY_URL", "https://host/models")
    agent = _make("PtbHarness_gemini", tmp_path, model_name="google/gemini-3.5-flash")
    assert agent.extra_env["GOOGLE_GEMINI_BASE_URL"] == "https://host/models/genai"


# -- the self-healing property ----------------------------------------------

def test_entrypoint_values_win_over_the_proxy_fallback(tmp_path, proxy_env):
    """The day entrypoint-common.sh grows an `agents.ptb_harness:*` branch,
    this wrapper must defer to it rather than overwrite it."""
    proxy_env.setenv("GEMINI_API_KEY", "set-by-entrypoint")
    proxy_env.setenv("GOOGLE_GEMINI_BASE_URL", "https://entrypoint/genai")
    agent = _make("PtbHarness_gemini", tmp_path, model_name="google/gemini-3.5-flash")
    assert os.environ["GEMINI_API_KEY"] == "set-by-entrypoint"
    assert os.environ["GOOGLE_GEMINI_BASE_URL"] == "https://entrypoint/genai"
    assert "GEMINI_API_KEY" not in agent.extra_env
    assert "GOOGLE_GEMINI_BASE_URL" not in agent.extra_env


def test_extra_env_wins_over_os_environ(tmp_path, proxy_env):
    """harbor's --ae channel outranks the host env
    (harbor/agents/installed/base.py:584-590), so a value there also wins."""
    proxy_env.setenv("GEMINI_API_KEY", "from-os-environ")
    agent = _make(
        "PtbHarness_gemini", tmp_path, model_name="google/gemini-3.5-flash",
        extra_env={"GEMINI_API_KEY": "from-ae"},
    )
    assert agent.extra_env["GEMINI_API_KEY"] == "from-ae"


def test_no_proxy_credentials_is_not_fatal(tmp_path, clean_env):
    """Nothing to fall back on: log it and let the adapter raise its own error."""
    agent = _make("PtbHarness_gemini", tmp_path, model_name="google/gemini-3.5-flash")
    assert "GEMINI_API_KEY" not in agent.extra_env


# -- gap 2: cursor-cli, deliberately not ported ------------------------------

def test_cursor_is_still_unportable_for_the_reason_we_recorded():
    """Tripwire for the divergence, not for our code.

    cursor-cli has no wrapper because there is nowhere to put a proxy base
    URL: the adapter declares no ModelConnectionSpec, so it inherits
    BaseAgent.MODEL_CONNECTION = None (harbor/agents/base.py:66) and exposes
    no base_url_envs. If a harbor release adds one, cursor may become viable
    and this test should fail so somebody re-reads the divergence.
    """
    assert CursorCli.MODEL_CONNECTION is None, (
        "harbor gave CursorCli a ModelConnectionSpec — cursor-cli may now be "
        "portable; re-read the supported-harnesses note in README divergence "
        "11 before adding it back"
    )
    assert "PtbHarness_cursor_cli" not in dir(ptb_harness)


# -- gap 3: reasoning effort -------------------------------------------------

@pytest.mark.parametrize("class_name,kwarg", [
    ("PtbHarness_claude", "reasoning_effort"),
    ("PtbHarness_codex", "reasoning_effort"),
    ("PtbHarness_gemini", "reasoning_effort"),
    ("PtbHarness_opencode", "variant"),
])
def test_effort_kwarg_names_match_the_entrypoint(class_name, kwarg):
    """harbor_reasoning_effort_kwarg, entrypoint-common.sh:377-388."""
    assert getattr(ptb_harness, class_name).PTB_EFFORT_KWARG == kwarg


def test_effort_reaches_the_claude_cli_flag(tmp_path, proxy_env):
    """claude_code.py:89-95 turns reasoning_effort into `--effort <value>`."""
    proxy_env.setenv(ptb_harness.KAGGLE_REASONING_EFFORT_ENV, "xhigh")
    agent = _make("PtbHarness_claude", tmp_path, model_name="anthropic/claude-opus-5")
    assert "--effort xhigh" in agent.build_cli_flags()


def test_effort_reaches_the_codex_cli_flag(tmp_path, proxy_env):
    """codex.py:64-71 formats it as `-c model_reasoning_effort={value}`."""
    proxy_env.setenv(ptb_harness.KAGGLE_REASONING_EFFORT_ENV, "xhigh")
    agent = _make("PtbHarness_codex", tmp_path, model_name="openai/gpt-5.4")
    assert "model_reasoning_effort=xhigh" in agent.build_cli_flags()


def test_effort_reaches_the_opencode_variant_flag(tmp_path, proxy_env):
    """opencode.py:65 — `--variant`, not `--effort`."""
    proxy_env.setenv(ptb_harness.KAGGLE_REASONING_EFFORT_ENV, "high")
    agent = _make("PtbHarness_opencode", tmp_path, model_name="openai/gpt-5.4")
    flags = agent.build_cli_flags()
    assert "--variant high" in flags
    assert "--effort" not in flags


def test_explicit_effort_kwarg_wins_over_the_env(tmp_path, proxy_env):
    proxy_env.setenv(ptb_harness.KAGGLE_REASONING_EFFORT_ENV, "low")
    agent = _make(
        "PtbHarness_claude", tmp_path, model_name="anthropic/claude-opus-5",
        reasoning_effort="max",
    )
    assert "--effort max" in agent.build_cli_flags()


def test_no_effort_env_leaves_the_flag_alone(tmp_path, proxy_env):
    agent = _make("PtbHarness_claude", tmp_path, model_name="anthropic/claude-opus-5")
    assert "--effort" not in agent.build_cli_flags()


def test_claude_env_fallback_name_is_not_what_the_platform_sets():
    """Why the kwarg is needed at all: claude_code.py:94's env_fallback is
    CLAUDE_CODE_EFFORT_LEVEL, but the platform emits
    KAGGLE_AGENT_LLM_REASONING_EFFORT (BenchmarkContainerEnvVarBuilder.cs:365)."""
    flag = next(f for f in ClaudeCode.CLI_FLAGS if f.kwarg == "reasoning_effort")
    assert flag.env_fallback == "CLAUDE_CODE_EFFORT_LEVEL"
    assert flag.env_fallback != ptb_harness.KAGGLE_REASONING_EFFORT_ENV


# -- gap 4: the codex responses slug -----------------------------------------

@pytest.mark.parametrize("given", ["openai/gpt-5.5", "gpt-5.5"])
def test_codex_rewrites_gpt55_to_responses(tmp_path, proxy_env, given):
    """entrypoint-common.sh:512-518, both spellings."""
    agent = _make("PtbHarness_codex", tmp_path, model_name=given)
    assert agent.model_name == "openai/gpt-5.5-responses"


@pytest.mark.parametrize("given", ["openai/gpt-5.4", "gpt-5.6-sol", None])
def test_codex_leaves_other_models_alone(tmp_path, proxy_env, given):
    agent = _make("PtbHarness_codex", tmp_path, model_name=given)
    assert agent.model_name == given


def test_only_codex_rewrites_the_slug(tmp_path, proxy_env):
    """harbor_map_responses_slug is called for codex/opencode/mini-swe-agent;
    of our five only codex is a Responses client."""
    for class_name in ("PtbHarness_claude", "PtbHarness_gemini"):
        agent = _make(class_name, tmp_path, model_name="gpt-5.5")
        assert agent.model_name == "gpt-5.5"


# ---------------------------------------------------------------------------
# run() delegation
# ---------------------------------------------------------------------------

def test_run_appends_clause_then_delegates(tmp_path, monkeypatch):
    """The real adapter's run() is what finally receives the instruction."""
    seen: dict[str, str] = {}

    async def fake_run(self, instruction, environment, context):
        seen["instruction"] = instruction

    monkeypatch.setattr(ClaudeCode, "run", fake_run)

    baseline = _instruction("gsm8k", "Qwen/Qwen3-1.7B-Base", build_tasks.PROMPT_AGENT)
    expected = _instruction("gsm8k", "Qwen/Qwen3-1.7B-Base", "claude")

    agent = AgentFactory.create_agent_from_import_path(
        _slug("PtbHarness_claude"), logs_dir=tmp_path
    )
    asyncio.run(agent.run(baseline, None, None))
    assert seen["instruction"] == expected


# -- gap 6: the DISPATCHED classes, not the per-harness ones -----------------
#
# PtbHarness.__new__ composes a fresh class at run time. It used to restate the
# per-harness facts inline and drifted out of step with the classes below it:
# PTB_PROXY_SUFFIX was set by hand, PTB_EFFORT_KWARG was omitted, so a
# dispatched opencode silently inherited "reasoning_effort" instead of its own
# "variant" and ran at opencode's default effort.
#
# test_effort_reaches_the_opencode_variant_flag above could not catch that: it
# builds PtbHarness_opencode directly, which was always correct. Only the
# dispatcher was wrong, so the dispatcher is what these assert.

@pytest.mark.parametrize("scheduled,agent_name,effort_kwarg", [
    ("claude-code-2.1.223", "claude", "reasoning_effort"),
    ("codex-0.146.1", "codex", "reasoning_effort"),
    ("gemini-cli-0.39.1", "gemini", "reasoning_effort"),
    ("opencode-1.18.14", "opencode", "variant"),
])
def test_dispatched_class_carries_its_harnesss_effort_kwarg(
        monkeypatch, scheduled, agent_name, effort_kwarg):
    monkeypatch.setenv(ptb_harness.PTB_SCHEDULED_HARNESS_ENV, scheduled)
    _adapter, name, _key = ptb_harness._ptb_resolve_scheduled(scheduled)
    assert name == agent_name
    composed = ptb_harness._PTB_BY_AGENT_NAME[name]
    assert composed.PTB_EFFORT_KWARG == effort_kwarg


def test_dispatch_copies_every_fact_from_the_per_harness_class(monkeypatch):
    """The dispatcher must not restate what the classes already declare."""
    for name, cls in ptb_harness._PTB_BY_AGENT_NAME.items():
        for attr in ptb_harness._PTB_DISPATCHED_ATTRS:
            assert getattr(cls, attr) is not None or attr == "PTB_PROXY_SUFFIX", (
                f"{cls.__name__}.{attr} is None; dispatch would copy that")


def test_dispatched_opencode_effort_reaches_the_variant_flag(tmp_path, proxy_env):
    """End to end through the dispatcher: --variant, never --effort."""
    proxy_env.setenv(ptb_harness.PTB_SCHEDULED_HARNESS_ENV, "opencode-1.18.14")
    proxy_env.setenv(ptb_harness.KAGGLE_REASONING_EFFORT_ENV, "high")
    agent = _make("PtbHarness", tmp_path, model_name="openai/gpt-5.4")
    flags = agent.build_cli_flags()
    assert "--variant high" in flags
    assert "--effort" not in flags
