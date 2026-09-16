"""Upstream-faithful wrappers around Harbor's real agent harnesses.

Why this module exists
======================

Upstream's agent prompt is **agent-name dependent**. `get_prompt.py:120-123`:

    if 'claude' in args.agent:
        result += \"\"\"
    You are running in a non-interactive mode. ...
    \"\"\"

A Harbor task has one static `instruction.md`, baked at build time and shared
by every harness that runs the task, so the conditional cannot live in the
file. `build_tasks.py` therefore bakes the **non-claude** prompt (the `--agent`
it is given must not contain `claude`; verify_fidelity.py asserts it) and this
module re-applies upstream's rule at run time, per harness.

Each class below is a *thin subclass* of the real Harbor adapter. It overrides
`run()` only to transform the instruction string, then delegates to
`super().run(...)` — every line of the real harness still executes. Nothing
else is replaced or reimplemented.

Selecting one
=============

Harbor resolves `--agent module.path:ClassName` through
`harbor/utils/import_path.py:import_class` (colon separator, see
`IMPORT_PATH_FORMAT` at line 8), reached from
`harbor/agents/factory.py:create_agent_from_import_path`.

On Kaggle the import path is registered as a **HarnessVersion's HarborSlug**
and scheduled against the Agent that references it. The platform then emits
it as `KAGGLE_AGENT_HARNESS`
(kaggleazure/Kaggle.Services.Benchmarks/Helpers/BenchmarkContainerEnvVarBuilder.cs:361),
`harbor_resolve_agent` exports it as `AGENT`
(container/harbor-base/entrypoint-common.sh:247-264) and `harbor_exec_run`
passes it as `--agent` (:564-565).

⚠️ NOT via a task version's `OverrideHarnessVersionCustomSlug`. That is also
passed straight to `--agent` (:246-251), but it is a task-version property,
and `CreateBenchmarkTaskFromHarborKaggleDatasetsHandler.cs:190-195` forces
`CandidateType = ModelVersions` whenever it is set — which would pin one
harness across all 28 tasks and collapse a sweep whose rows vary both model
and harness.

The slug to register is, verbatim:

    agents.ptb_harness:PtbHarness_claude
    agents.ptb_harness:PtbHarness_codex
    agents.ptb_harness:PtbHarness_gemini
    agents.ptb_harness:PtbHarness_opencode

`agents/` is staged at the task-definition root by build_tasks.py, which is
what `harbor_apply_custom_import_pythonpath`
(container/harbor-base/entrypoint-common.sh:531-541) prepends to PYTHONPATH.

Why the class names are spelled that way
========================================

The `--agent` string feeds **two** upstream substring dispatchers, so the slug
itself is data, not decoration:

1. `get_prompt.py:120` — `if 'claude' in args.agent`.
2. `src/trace_parsing/parse_trace.py:36-44` — `select_parser()` substring-
   matches the name against `{claude, codex, cursor, gemini, opencode}`;
   >1 match is a hard `SystemExit`, 0 matches silently copies the raw trace.

Each slug above therefore contains its upstream agent name in lower case and
matches **exactly one** parser key, and contains `claude` if and only if the
upstream agent name it stands for does. `agents`, `ptb_harness` and
`PtbHarness` contain no parser key; `opencode` does not contain `codex`. The
test suite asserts all of this against upstream's real `select_parser`.

The version pin
===============

Upstream pins exact CLI builds per leaderboard row — eight
`@anthropic-ai/claude-code@`, eight `@openai/codex@`, four
`@google/gemini-cli@` and one `CURSOR_VERSION` across `containers/*.def` — so
a row is not fully specified without one.

On the Agent route above the platform supplies it:
BenchmarkContainerEnvVarBuilder.cs:362 emits
`KAGGLE_AGENT_HARNESS_VERSION = agent.HarborVersionSlug`, and
`harbor_append_agent_version` (entrypoint-common.sh:444-450) turns that into
`--ak version="..."`, which reaches `BaseInstalledAgent.__init__`
(harbor/agents/installed/base.py:521) as usual. On that route
`_ptb_resolve_version` is a pass-through and adds nothing.

It exists for the routes where nothing supplies it: a plain local
`harbor run`, and the custom-slug route, whose branch at
BenchmarkContainerEnvVarBuilder.cs:246-251 **blanks** the variable ("there is
no corresponding harness version") so the adapter would install its default
(newest) build. Three sources, highest precedence first.

Not wrapped: cursor-cli
=======================

Out of scope — the model proxy does not support it, and there is no hook to
point it at one anyway: `CursorCli` declares no `MODEL_CONNECTION` (the
identifier does not occur in harbor/agents/installed/cursor_cli.py, so it
inherits `BaseAgent.MODEL_CONNECTION = None`, harbor/agents/base.py:66), and
its run command (cursor_cli.py:879-883) passes no endpoint flag and reads no
base-URL variable. Checked, not assumed.
"""

from __future__ import annotations

import os
from typing import Any

from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.codex import Codex
from harbor.agents.installed.gemini_cli import GeminiCli
from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

# get_prompt.py:121-123, byte-for-byte. The leading and trailing newlines are
# part of the literal upstream appends and are load-bearing.
_NON_INTERACTIVE_CLAUSE = """
You are running in a non-interactive mode. So make sure every process you are running finishes before you write your last message.
"""

# Env var read by _ptb_resolve_version. Deliberately not
# KAGGLE_AGENT_HARNESS_VERSION: that name belongs to the platform and is the
# one the custom-slug branch blanks. Same PTB_ prefix the port already uses
# for its own knobs (build_tasks.py:318, :325, :342, :344).
PTB_HARNESS_VERSION_ENV = "PTB_HARNESS_VERSION"

# The platform's reasoning-effort value, emitted for every agents-candidate
# run at BenchmarkContainerEnvVarBuilder.cs:365. harbor_reasoning_effort_kwarg
# (entrypoint-common.sh:380-396) translates it into the agent's own kwarg
# name, but switches on `$AGENT` by exact name, so a custom import path gets
# the "no reasoning kwarg; ignoring" warning at :415 instead. Re-done below.
KAGGLE_REASONING_EFFORT_ENV = "KAGGLE_AGENT_LLM_REASONING_EFFORT"

# The credentials the entrypoint's unconditional block cannot spell for us.
# Both names are accepted: the executor exports MODEL_PROXY_BASE_URL
# (entrypoint-common.sh:307-308 hard-errors without it), while the host .env
# and the starter template also use MODEL_PROXY_URL
# (kaggle-benchmark-harbor-starter-template/agents/antigravity_agent.py:121-127).
PROXY_KEY_ENVS = ("MODEL_PROXY_API_KEY",)
PROXY_URL_ENVS = ("MODEL_PROXY_BASE_URL", "MODEL_PROXY_URL")


def apply_agent_clause(instruction: str, ptb_agent_name: str) -> str:
    """Reproduce get_prompt.py:120-123 on an already-generated instruction.

    `instruction` is `instruction.md`, i.e. get_prompt.py's stdout after the
    `$( )` capture at run_task.sh:75-76 stripped the trailing newline `print`
    added and `echo` put exactly one back (build_tasks.py:119-122).
    """
    # get_prompt.py:120 — the rule, verbatim, on upstream's own agent name.
    if 'claude' not in ptb_agent_name:
        return instruction
    # Undo run_task.sh:76's capture to recover get_prompt.py's `result` ...
    result = instruction.rstrip("\n")
    # ... apply get_prompt.py:121-123 to it ...
    result += _NON_INTERACTIVE_CLAUSE
    # ... and re-apply the capture (build_tasks.py:119-122).
    return result.rstrip("\n") + "\n"


class _PtbHarness:
    """Mixin placed before the real adapter in every subclass's MRO.

    Reproduces, per harness, the four things the executor entrypoint does for
    a built-in agent name but not for a custom import path — plus the clause
    rule that is the reason this module exists at all. Every one of them
    defers to a value already present, so if the entrypoint ever grows an
    `agents.ptb_harness:*` branch this class quietly stops doing anything.
    """

    # Upstream's `--agent` string for this harness — the name of its
    # `agents/<name>/` directory in the PostTrainBench checkout. This is the
    # value get_prompt.py:120 tests, and the value that must be passed to
    # `build_tasks.py --agent` for the baked instruction.md to match.
    PTB_AGENT_NAME: str = ""

    # "latest", because that is what upstream runs.
    #
    # Every one of upstream's four agent solve.sh scripts opens with
    #
    #     bash /home/ben/update_agent_cli.sh <cli>
    #
    # and that script does `npm install -g --prefix "$HOME/.local"
    # "<pkg>@latest"` (src/utils/update_agent_cli.sh:56), into a prefix
    # run_task.sh puts AHEAD of the baked one on PATH. The opt-out,
    # POST_TRAIN_BENCH_SKIP_CLI_UPDATE, is commented out in example.env:31, so
    # updating is the default and pinning is the exception. The versions in
    # standard.def (codex 0.137.0) are the OFFLINE FALLBACK — used only when
    # the npm update fails ("falling back to pinned", :60) — not the version
    # upstream runs.
    #
    # This port replaced solve.sh with harbor's harness, which dropped the
    # update call with it: `update_agent_cli.sh` is still staged into the image
    # at /home/ben/update_agent_cli.sh and is simply never invoked. Left
    # unpinned, harbor does NOT install the newest build either — with
    # `version=None` it only checks whether the CLI exists at all
    # (`_installed_codex_satisfies_version`, codex.py:330-334) and skips the
    # install when it does, so the baked fallback becomes permanent. That
    # froze all four harnesses at their fallback versions, and it is how a
    # 2-hour gpt-5.6-sol run died at startup on
    # `unknown variant 'max'` — `max` postdates codex 0.137.0.
    #
    # "latest" reproduces upstream exactly: every adapter renders it as
    # `@latest` (codex.py:354, gemini_cli.py:137, opencode.py:105,
    # claude_code.py:442), and for the two that check before installing the
    # string compare against a real version never matches, so the install
    # always runs. The baked copy stays as the same offline fallback upstream
    # keeps.
    #
    # ⚠️ TRADEOFF, inherited rather than introduced: runs are no longer
    # byte-reproducible, because two runs a week apart can pick up different
    # CLI builds. That is true of upstream too. Upstream's own lever for
    # reproducibility is POST_TRAIN_BENCH_SKIP_CLI_UPDATE=1; here the
    # equivalent is a per-row pin, e.g.
    #
    #     class PtbHarness_claude_2_1_219(PtbHarness_claude):
    #         PTB_DEFAULT_VERSION = "2.1.219"
    #
    # whose slug still contains exactly one parser key. `--ak version=` and
    # PTB_HARNESS_VERSION both still override this.
    #
    # No subclass sets this: the scheduled agent slug names the build and the
    # platform passes it through (see PtbHarness_gemini). An earlier revision
    # pinned a version per harness; that was reverted once the slug was
    # confirmed to carry it, so this base value is the only one in play.
    PTB_DEFAULT_VERSION: str | None = "latest"

    # The kwarg this harness calls reasoning effort, from
    # harbor_reasoning_effort_kwarg (entrypoint-common.sh:380-396). None means
    # the harness has no such knob and the value is dropped, which is what
    # :413-416 does ("not fatal ... the agents without the knob shouldn't
    # abort the sweep").
    PTB_EFFORT_KWARG: str | None = "reasoning_effort"

    # Base-URL suffix on MODEL_PROXY_BASE_URL for this harness's dialect, when
    # the harness needs a base URL the entrypoint's unconditional block
    # (entrypoint-common.sh:312-324) does not already export under the name
    # the harness reads. None means nothing to do.
    PTB_PROXY_SUFFIX: str | None = None

    # Credential env names to fill. Normally derived from the harness's own
    # MODEL_CONNECTION so a harbor rename follows automatically; only set
    # explicitly by a harness that has no MODEL_CONNECTION at all.
    PTB_API_KEY_ENVS: tuple[str, ...] = ()
    PTB_BASE_URL_ENVS: tuple[str, ...] = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        version, version_source = self._ptb_resolve_version(
            kwargs.pop("version", None)
        )
        model_note = self._ptb_rewrite_model(kwargs)
        effort_note = self._ptb_inject_effort(kwargs)
        # `version` is BaseInstalledAgent's own constructor kwarg
        # (harbor/agents/installed/base.py:521, stored at :580), so this is
        # the adapter's normal path, not a new mechanism. The effort kwarg
        # goes in the same way — CLI_FLAGS kwargs are auto-extracted at
        # base.py:167-170 and resolved at :570, both inside this super() call,
        # so it has to be in `kwargs` before we get there.
        super().__init__(*args, version=version, **kwargs)
        # self.logger is set by BaseAgent.__init__ (harbor/agents/base.py:82),
        # so everything below has to come after the super() call.
        self.logger.info(
            "PTB harness %s (upstream agent %r): model=%r version pin %r (%s)",
            type(self).__name__,
            self.PTB_AGENT_NAME,
            self.model_name,
            version,
            version_source,
        )
        for note in (model_note, effort_note, *self._ptb_fill_credentials()):
            self.logger.info("PTB: %s", note)

    # -- the version pin ----------------------------------------------------

    @classmethod
    def _ptb_resolve_version(cls, kwarg_value: Any) -> tuple[str | None, str]:
        """Pick the CLI build to pin, and say where it came from."""
        if kwarg_value is not None:
            if isinstance(kwarg_value, str):
                return kwarg_value, "--ak version="
            # harbor's parse_kwargs runs json.loads on every --ak value
            # (harbor/cli/utils.py:65-90), so an unquoted `version=2.1`
            # arrives as the float 2.1 and `version=1` as the int 1. str()
            # cannot recover what json.loads discarded (2.10 -> 2.1), so the
            # coercion is accompanied by a warning rather than done silently.
            return str(kwarg_value), (
                f"--ak version= (coerced from {type(kwarg_value).__name__}; "
                f"pass it quoted, --ak version=\"...\", as "
                f"harbor_append_agent_version does at "
                f"entrypoint-common.sh:449)"
            )
        env_value = os.environ.get(PTB_HARNESS_VERSION_ENV)
        if env_value:
            return env_value, PTB_HARNESS_VERSION_ENV
        if cls.PTB_DEFAULT_VERSION:
            return cls.PTB_DEFAULT_VERSION, "PTB_DEFAULT_VERSION"
        return None, "unpinned — the adapter installs its default build"

    # -- the model slug -----------------------------------------------------

    def _ptb_rewrite_model(self, kwargs: dict[str, Any]) -> str:
        """Hook for harbor_map_responses_slug. Base class rewrites nothing."""
        return f"model slug: {kwargs.get('model_name')!r} (no rewrite for this harness)"

    # -- reasoning effort ---------------------------------------------------

    def _ptb_inject_effort(self, kwargs: dict[str, Any]) -> str:
        """Reproduce harbor_reasoning_effort_kwarg (entrypoint-common.sh:380-396).

        The platform emits KAGGLE_AGENT_LLM_REASONING_EFFORT unconditionally
        for an agents candidate (BenchmarkContainerEnvVarBuilder.cs:365); the
        entrypoint drops it for a custom import path. No validation here, on
        purpose: entrypoint-common.sh:401-403 says "We do no validation in the
        executor and assume that configuration is correct for the particular
        harness", so a bad value must reach the adapter and be rejected there.
        """
        effort = os.environ.get(KAGGLE_REASONING_EFFORT_ENV)
        if not effort:
            return f"reasoning effort: {KAGGLE_REASONING_EFFORT_ENV} unset"
        if self.PTB_EFFORT_KWARG is None:
            # entrypoint-common.sh:413-416, the empty-kwarg branch.
            return (
                f"reasoning effort: {effort!r} ignored — {self.PTB_AGENT_NAME} "
                f"has no reasoning kwarg"
            )
        if self.PTB_EFFORT_KWARG in kwargs:
            return (
                f"reasoning effort: {self.PTB_EFFORT_KWARG}="
                f"{kwargs[self.PTB_EFFORT_KWARG]!r} already supplied; "
                f"{KAGGLE_REASONING_EFFORT_ENV}={effort!r} not applied"
            )
        kwargs[self.PTB_EFFORT_KWARG] = effort
        return f"reasoning effort: {self.PTB_EFFORT_KWARG}={effort!r}"

    # -- credentials --------------------------------------------------------

    @classmethod
    def _ptb_credential_envs(cls) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The API-key and base-URL env names this harness reads.

        Taken from the harness's own ModelConnectionSpec
        (harbor/agents/model_connection.py:133-147) rather than from string
        literals here, so a harbor rename is followed automatically and the
        assertions below fail loudly instead of the wrapper silently setting a
        name nothing reads.
        """
        spec = cls.MODEL_CONNECTION
        if spec is None:
            # Only legitimate for a harness with no spec at all; the subclass
            # must then say which names it reads. Asserted per class in the
            # tests so a harbor upgrade that adds a spec is caught.
            return cls.PTB_API_KEY_ENVS, cls.PTB_BASE_URL_ENVS
        assert hasattr(spec, "api_key_envs") and hasattr(spec, "base_url_envs"), (
            f"{cls.__name__}: harbor's ModelConnectionSpec no longer exposes "
            f"api_key_envs/base_url_envs; ptb_harness.py must be updated"
        )
        return spec.api_key_envs, spec.base_url_envs

    def _ptb_lookup(self, *names: str) -> str | None:
        """First non-empty value, extra_env before os.environ, in name order.

        Same shape as the starter template's
        agents/antigravity_agent.py:112-128. extra_env first because that is
        harbor's own `--ae` channel and BaseInstalledAgent._env_sources
        (harbor/agents/installed/base.py:584-590) ranks it above os.environ.
        """
        for name in names:
            value = self._extra_env.get(name) or os.environ.get(name)
            if value:
                return value
        return None

    def _ptb_set(self, name: str, value: str) -> None:
        """Publish a credential without overwriting one that already exists.

        Written to both channels: harbor's own env source
        (BaseInstalledAgent._env_sources, base.py:584-590), which is what the
        adapters resolve through, and os.environ, which is what a CLI the
        adapter spawns inherits. Belt and braces; neither is overwritten.
        """
        self._extra_env[name] = value
        os.environ.setdefault(name, value)

    def _ptb_fill_credentials(self) -> list[str]:
        """Fill in what entrypoint-common.sh's per-agent `case` would have.

        Defers entirely to any value already set, so the day the entrypoint
        gains an `agents.ptb_harness:*` branch this becomes a no-op.
        """
        key_envs, url_envs = self._ptb_credential_envs()
        notes: list[str] = []

        for name in key_envs:
            if self._ptb_lookup(name):
                notes.append(f"credentials: {name} already set; left alone")
                continue
            proxy_key = self._ptb_lookup(*PROXY_KEY_ENVS)
            if not proxy_key:
                notes.append(
                    f"credentials: {name} unset and no "
                    f"{'/'.join(PROXY_KEY_ENVS)} to fall back on"
                )
                continue
            self._ptb_set(name, proxy_key)
            notes.append(f"credentials: {name} <- {PROXY_KEY_ENVS[0]}")

        if self.PTB_PROXY_SUFFIX is None:
            return notes

        for name in url_envs:
            if self._ptb_lookup(name):
                notes.append(f"credentials: {name} already set; left alone")
                continue
            proxy_url = self._ptb_lookup(*PROXY_URL_ENVS)
            if not proxy_url:
                notes.append(
                    f"credentials: {name} unset and no "
                    f"{'/'.join(PROXY_URL_ENVS)} to fall back on"
                )
                continue
            # entrypoint-common.sh:344 — "${MODEL_PROXY_BASE_URL}/genai".
            value = proxy_url.rstrip("/") + self.PTB_PROXY_SUFFIX
            self._ptb_set(name, value)
            notes.append(f"credentials: {name} <- proxy + {self.PTB_PROXY_SUFFIX}")

        return notes

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        # Signature copied from harbor/agents/base.py:205-210 and matched by
        # all four adapters (claude_code.py:1601, codex.py:1333,
        # gemini_cli.py:780, opencode.py's run).
        transformed = apply_agent_clause(instruction, self.PTB_AGENT_NAME)
        if transformed != instruction:
            self.logger.info(
                "PTB: get_prompt.py:120 clause APPLIED (upstream agent %r "
                "contains 'claude'); harness=%s version=%s",
                self.PTB_AGENT_NAME,
                self.name(),
                self.version(),
            )
        else:
            self.logger.info(
                "PTB: get_prompt.py:120 clause NOT applied (upstream agent %r "
                "does not contain 'claude'); harness=%s version=%s",
                self.PTB_AGENT_NAME,
                self.name(),
                self.version(),
            )
        return await super().run(transformed, environment, context)


# Four of the five harnesses scratch.md §7 lists ("claude-code, codex,
# cursor-cli, opencode, gemini-cli"), each paired with the upstream
# `agents/<name>/` directory it stands for. cursor-cli is deliberately
# absent — see "Why there is no cursor-cli wrapper" above. Real classes from
# AgentFactory._AGENT_MAP (harbor/agents/factory.py:29-42).

class PtbHarness_claude(_PtbHarness, ClaudeCode):
    # Credentials: nothing to do. ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL are
    # in the entrypoint's unconditional block (entrypoint-common.sh:316-317),
    # which every agent name reaches.
    # Effort: `reasoning_effort` per entrypoint-common.sh:384-388; the adapter
    # turns it into `--effort` (claude_code.py:89-95). That CliFlag also has
    # env_fallback="CLAUDE_CODE_EFFORT_LEVEL", but nothing in the executor
    # sets that name — the platform sets KAGGLE_AGENT_LLM_REASONING_EFFORT —
    # so the fallback does not cover us and the kwarg is still needed.
    PTB_AGENT_NAME = "claude"


class PtbHarness_codex(_PtbHarness, Codex):
    # Credentials: OPENAI_API_KEY / OPENAI_BASE_URL are unconditional
    # (entrypoint-common.sh:313-314).
    # Effort: `reasoning_effort` (entrypoint-common.sh:384-388) ->
    # `-c model_reasoning_effort={value}` (codex.py:64-71).
    PTB_AGENT_NAME = "codex"

    def _ptb_rewrite_model(self, kwargs: dict[str, Any]) -> str:
        """harbor_map_responses_slug, entrypoint-common.sh:512-518.

        One `if` for one model, not a slug map: the proxy picks its backend
        from the slug, so a Responses call against plain gpt-5.5 lands on the
        wrong one. In scope because commit.sh:52 and :65 both run
        `agent_config=gpt-5.5` against codex.
        """
        model = kwargs.get("model_name")
        if model in ("openai/gpt-5.5", "gpt-5.5"):
            kwargs["model_name"] = "openai/gpt-5.5-responses"
            return (
                f"model slug: rewriting {model!r} -> 'openai/gpt-5.5-responses' "
                f"(entrypoint-common.sh:512-518)"
            )
        return f"model slug: {model!r} (no rewrite)"


class PtbHarness_gemini(_PtbHarness, GeminiCli):
    # gemini_cli.py:52-57 declares api_key_envs=("GEMINI_API_KEY",) and
    # base_url_envs=("GOOGLE_GEMINI_BASE_URL",); neither is in the
    # unconditional block, both are in the name-gated branch at
    # entrypoint-common.sh:342-345. Without them the CLI dies with "When using
    # Gemini API, you must specify the GEMINI_API_KEY environment variable"
    # (observed) or, with only the key, talks to
    # generativelanguage.googleapis.com instead of the proxy (:339-341).
    PTB_AGENT_NAME = "gemini"
    PTB_PROXY_SUFFIX = "/genai"  # entrypoint-common.sh:344
    # No PTB_DEFAULT_VERSION here or on any sibling: the scheduled slug names
    # the build and the platform passes it. gemini-cli additionally has no
    # agent slug at all (antigravity-sdk is the supported Gemini harness), so
    # it is unreachable regardless.


class PtbHarness_opencode(_PtbHarness, OpenCode):
    # Credentials: MODEL_CONNECTION is ModelConnectionSpec(passthrough=True)
    # with no api_key_envs/base_url_envs (opencode.py:53), and the
    # unconditional block already exports GOOGLE_GENERATIVE_AI_API_KEY /
    # GOOGLE_BASE_URL / OPENAI_* / ANTHROPIC_* (entrypoint-common.sh:312-324).
    # Nothing to fill, and the empty tuples make _ptb_fill_credentials a no-op.
    # Effort: opencode calls it `variant`, not `reasoning_effort`
    # (entrypoint-common.sh:377-383; the CliFlag is opencode.py:65).
    PTB_AGENT_NAME = "opencode"
    PTB_EFFORT_KWARG = "variant"


# ---------------------------------------------------------------------------
# The dispatcher: ONE task, every harness, the clause still conditional
# ---------------------------------------------------------------------------
#
# The four classes above each pin one harness, which is fine when a task is
# built for a known harness. It is not fine when a SINGLE task set has to serve
# every (harness, model, effort) combo, because the task's
# KAGGLE_HARBOR_AGENT_OVERRIDE names exactly one class and the entrypoint
# resolves `AGENT="${KAGGLE_HARBOR_AGENT_OVERRIDE:-...}"` -- our name wins over
# whatever the scheduled agent slug asked for.
#
# But the slug is NOT lost. The platform still emits KAGGLE_AGENT_HARNESS from
# the scheduled Agent's HarnessVersion (entrypoint-common.sh:241-243), and the
# override only decides which class loads. So a class that reads that variable
# can discover the harness that was actually scheduled and become it.
#
# That is what this does. It is not an adapter itself: `__new__` picks the real
# adapter, composes it with the mixin above, and returns an instance of THAT.
# Python skips `__init__` when `__new__` returns something that is not an
# instance of cls, so the composed class initialises normally and Harbor --
# which only does `agent_class(logs_dir=..., model_name=..., **kwargs)` with a
# `cast`, no isinstance check (factory.py:17-21, :130-131) -- is none the wiser.
#
# Net effect: one task, `envVariables: KAGGLE_HARBOR_AGENT_OVERRIDE =
# "agents.ptb_harness:PtbHarness"`, `candidateType = agents`, schedule any
# agent slug you like. The clause fires iff the scheduled harness is Claude,
# reproducing get_prompt.py:61 against the harness that actually ran.
#
# ⚠️ Registration would be cleaner: then the scheduled slug picks the class
# directly and this indirection disappears. This exists to avoid requiring a
# HarnessVersion registration for every harness we want to run.

#: Scheduled-harness substring -> (adapter, upstream agent name). Ordered:
#: "opencode" must be tested before "codex", since it contains neither as a
#: substring of the other but the parser keys do overlap conceptually.
# Harbor already knows every harness it supports -- AgentFactory._AGENT_MAP
# maps "claude-code", "codex", "antigravity-sdk", "mini-swe-agent", ... to the
# adapter import path. Resolving against THAT instead of a table of our own
# means any harness Kaggle can schedule, we can run. An earlier revision here
# hardcoded four and raised on everything else; that was our restriction, not
# the platform's, and it would have refused antigravity or openhands for no
# reason. Deciding whether a candidate is valid is the model proxy's job -- if
# it rejects one, the run fails on its own and the failure propagates.
PTB_SCHEDULED_HARNESS_ENV = "KAGGLE_AGENT_HARNESS"

def _ptb_resolve_scheduled(harness: str) -> tuple[type, str, str]:
    """(adapter, the original's agent name) for the scheduled slug.

    No version is returned and none is pinned. The agent slug already names
    its build -- `codex-0.146.1-gpt-5.5-xhigh-reasoning` -- and the platform
    parses it out and passes `--ak version=0.146.1`, which the run logs
    confirm ("version pin '0.146.1' (--ak version=)"). A table of pinned
    builds here would be a third copy of a fact the slug already carries, kept
    in sync by hand, and read by nothing. The starter template does the same:
    `f"@{self._version}" if self._version else "@latest"`
    (opencode_binary_agent.py:91).

    A slug naming a build that does not exist is not our error to catch --
    it surfaces from the platform.

    `harness` is KAGGLE_AGENT_HARNESS, a HarnessVersion slug like
    "claude-code-2.1.223". Harbor's map is keyed by the bare harness name, so
    the version suffix is trimmed longest-match-first ("antigravity-sdk"
    before "antigravity", "claude-code" before "claude").
    """
    from harbor.agents.factory import AgentFactory
    from harbor.utils.import_path import import_class

    h = (harness or "").strip().lower()
    known = {name.value: path for name, path in AgentFactory._AGENT_MAP.items()}
    match = next((k for k in sorted(known, key=len, reverse=True)
                  if h == k or h.startswith(k + "-")), None)
    if match is None:
        raise ValueError(
            f"{PTB_SCHEDULED_HARNESS_ENV}={harness!r} is not a harness Harbor "
            f"knows ({len(known)} available). Nothing to construct."
        )

    adapter = import_class(known[match], label="agent")
    # The original's agent name, for get_prompt.py:61 / parse_trace.py:36-44.
    # Its dispatchers key on bare names, so map the harness onto one where a
    # correspondence exists; otherwise pass the harness through and let
    # select_parser fall back to copying the raw trace.
    #
    # Precedent: the same bare-name substring test the original itself uses in
    # parse_trace.py:37. Load-bearing beyond that -- this is the key into
    # _PTB_BY_AGENT_NAME below, so `gemini-cli` must become `gemini` or the
    # lookup misses and PTB_PROXY_SUFFIX="/genai" is lost.
    agent_name = next((k for k in ("claude", "codex", "gemini", "opencode")
                       if k in match), match)
    return adapter, agent_name, match


# The per-harness classes above are the single source of truth for every
# per-harness fact. The dispatcher copies from them rather than restating them.
#
# It used to restate them inline, and drifted: it set PTB_PROXY_SUFFIX by hand
# but forgot PTB_EFFORT_KWARG entirely, so a dispatched opencode inherited the
# base "reasoning_effort" instead of its own "variant" (opencode.py:65 declares
# CliFlag("variant"); entrypoint-common.sh:492-498 agrees). Nothing rejects an
# unknown kwarg -- BaseAgent.__init__ swallows it via **kwargs -- so the run
# completed at opencode's DEFAULT effort while its agent slug advertised
# otherwise. Silent, and invisible in the logs.
_PTB_DISPATCHED_ATTRS = ("PTB_AGENT_NAME", "PTB_DEFAULT_VERSION",
                         "PTB_PROXY_SUFFIX", "PTB_EFFORT_KWARG")
_PTB_BY_AGENT_NAME = {c.PTB_AGENT_NAME: c for c in (
    PtbHarness_claude, PtbHarness_codex, PtbHarness_gemini, PtbHarness_opencode)}


# Reasoning-effort kwarg per harness — a TRANSCRIPTION of the platform's own
# harbor_reasoning_effort_kwarg(), container/harbor-base/entrypoint-common.sh
# :487-511. Not our invention and not guesswork: the executor applies exactly
# this switch for built-in agent names, and a custom import path (which is how
# we get the Claude clause) bypasses it, so we replay it.
#
# Keep it in sync with that function. Anything missing here is silently run at
# the harness's DEFAULT effort while its slug advertises otherwise -- that was
# D230, where opencode's `variant` was absent and every opencode run ignored
# its requested effort without a single log line.
_PTB_EFFORT_KWARG_BY_HARNESS = {
    # `thinking` -- entrypoint-common.sh:489-491
    "cline-cli": "thinking", "openclaw": "thinking",
    "pi": "thinking", "vibe": "thinking",
    # `variant` -- :492-498. Values are per-PROVIDER for this one.
    "opencode": "variant",
    # `reasoning_effort` -- :499-503
    "aider": "reasoning_effort", "antigravity-cli": "reasoning_effort",
    "antigravity-sdk": "reasoning_effort", "claude-code": "reasoning_effort",
    "codex": "reasoning_effort", "copilot-cli": "reasoning_effort",
    "cursor-cli": "reasoning_effort", "gemini-cli": "reasoning_effort",
    "grok-build": "reasoning_effort", "mini-swe-agent": "reasoning_effort",
    "openhands": "reasoning_effort", "openhands-sdk": "reasoning_effort",
}
# Named in the platform's comment as deliberately UNHANDLED even by Kaggle:
#   deerflow    -- its `thinking` is a bool, so "high" coerces to False and
#                  silently DISABLES thinking (worse than dropping it)
#   rovodev-cli -- wants max_thinking_tokens, an int budget, not a level
_PTB_EFFORT_UNSUPPORTED = {"deerflow", "rovodev-cli"}


class PtbHarness:
    """Becomes whichever harness the run was scheduled with."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        scheduled = os.environ.get(PTB_SCHEDULED_HARNESS_ENV, "")
        adapter, agent_name, harness_key = _ptb_resolve_scheduled(scheduled)
        known = _PTB_BY_AGENT_NAME.get(agent_name)
        if known is not None:
            attrs = {k: getattr(known, k) for k in _PTB_DISPATCHED_ATTRS}
        else:
            # A harness Harbor knows but we have no class for -- antigravity,
            # say. Base defaults apply; the version comes from the slug.
            attrs = {"PTB_AGENT_NAME": agent_name}
        # The effort kwarg always comes from the platform's own switch, never
        # from our per-harness class -- the table IS that switch, so it covers
        # every harness Kaggle handles, not just the four we wrote classes for.
        if harness_key in _PTB_EFFORT_UNSUPPORTED:
            attrs["PTB_EFFORT_KWARG"] = None
            log_line = (f"PTB: {harness_key} has no usable effort kwarg "
                        f"(platform declares it unsupported); effort dropped")
        elif harness_key in _PTB_EFFORT_KWARG_BY_HARNESS:
            attrs["PTB_EFFORT_KWARG"] = _PTB_EFFORT_KWARG_BY_HARNESS[harness_key]
            log_line = None
        else:
            # Unknown to the platform switch too. Guess the common kwarg, but
            # say so -- a silent wrong guess is exactly the D230 failure.
            attrs["PTB_EFFORT_KWARG"] = "reasoning_effort"
            log_line = (f"PTB: ⚠️  {harness_key!r} is not in the platform's "
                        f"reasoning-effort switch; assuming 'reasoning_effort'. "
                        f"If this harness names it differently the effort is "
                        f"being DROPPED silently — check "
                        f"entrypoint-common.sh:harbor_reasoning_effort_kwarg")
        if log_line:
            import logging; logging.getLogger(__name__).warning(log_line)
        composed = type(
            f"PtbHarness_dispatched_{agent_name}", (_PtbHarness, adapter), attrs)
        return composed(*args, **kwargs)
