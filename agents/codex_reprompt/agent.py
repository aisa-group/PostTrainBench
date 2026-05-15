"""Codex agent with a reprompt loop, used by PostTrainBench.

This is a thin extension of harbor's built-in `Codex` agent. After codex
exits its initial pass, we poll `bash timer.sh` (provided by the
PostTrainBench task workspace). If the agent finished early and there are
more than `min_remaining_minutes` left on the budget, we resume the same
session via `codex exec resume --last` with a continuation prompt and
loop. Stop when timer.sh reports expired or remaining < threshold.

Mirrors the behavior of `agents/codex_non_api_*_reprompt/solve.sh` from
the condor pipeline.

Load with:
    harbor run --agent-import-path \\
        agents.codex_reprompt.agent:CodexReprompt \\
        --model gpt-5.3-codex \\
        --reasoning-effort high \\
        --ae OPENAI_API_KEY=$OPENAI_API_KEY \\
        ...

For ChatGPT-Pro auth instead of API key, also pass:
    --ae CODEX_AUTH_JSON_PATH=/local/path/to/auth.json
(harbor's Codex base reads the file from the host and uploads to sandbox.)
"""

import re
import shlex

from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.codex import Codex
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.env import parse_bool_env_value


# Default minimum remaining time before triggering a continuation pass.
# Below this threshold we let the agent stop instead of starting a partial
# resume that probably won't finish in time. Override via env:
#     --ae CODEX_REPROMPT_MIN_MINUTES=15
_DEFAULT_MIN_REMAINING_MINUTES = 30

# Path to the timer script inside the sandbox. PostTrainBench tasks
# generate this at /home/agent/workspace/timer.sh; the agent's cwd at
# exec time is the workspace, so a relative path works.
_TIMER_SCRIPT = "timer.sh"

# Matches output of timer.sh:
#   "Remaining time (hours:minutes):"
#   "9:30"
# Captures (hours, minutes).
_TIMER_REMAINING_RE = re.compile(r"^(\d+):(\d+)\s*$", re.MULTILINE)


class CodexReprompt(Codex):
    """Codex variant that resumes early-exiting sessions until time runs out."""

    @staticmethod
    def name() -> str:
        # Distinct from the built-in "codex" so harbor's logs / trajectory
        # agent_info correctly identify which variant ran.
        return "codex-reprompt"

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        """Run codex with a reprompt loop.

        Most of the setup (CODEX_HOME, auth resolution, base_url config,
        skills/MCP registration) is identical to the parent. To preserve
        the codex session between iterations — `codex exec resume --last`
        reads it from $CODEX_HOME/sessions — we cannot let the parent's
        cleanup (`rm -rf $CODEX_HOME`) run between iterations. So we
        replicate the parent's flow but interpose the reprompt loop
        before cleanup.
        """
        if not self.model_name:
            raise ValueError("Model name is required")

        model = self.model_name.split("/")[-1]
        cli_flags = self.build_cli_flags()
        cli_flags_arg = (cli_flags + " ") if cli_flags else ""

        min_remaining_minutes = self._resolve_min_remaining_minutes()

        # Resolve auth (CODEX_AUTH_JSON_PATH or CODEX_FORCE_AUTH_JSON; falls
        # back to OPENAI_API_KEY). Same logic as parent.
        auth_json_path = self._resolve_auth_json_path()

        remote_codex_home = self._REMOTE_CODEX_HOME.as_posix()
        remote_secrets_dir = self._REMOTE_CODEX_SECRETS_DIR.as_posix()
        remote_auth_path = (self._REMOTE_CODEX_SECRETS_DIR / "auth.json").as_posix()

        env: dict[str, str] = {"CODEX_HOME": remote_codex_home}

        await self.exec_as_agent(
            environment,
            command=(
                f'mkdir -p "$CODEX_HOME" {shlex.quote(remote_secrets_dir)} '
                f"{shlex.quote(EnvironmentPaths.agent_dir.as_posix())}"
            ),
            env=env,
        )

        # Build setup_command — auth.json or OPENAI_API_KEY path
        if auth_json_path:
            self.logger.debug(
                "Codex auth: using auth.json from %s", auth_json_path
            )
            await environment.upload_file(auth_json_path, remote_auth_path)
            if environment.default_user is not None:
                await self.exec_as_root(
                    environment,
                    command=f"chown {environment.default_user} {remote_auth_path}",
                )
            setup_command = (
                f'ln -sf {shlex.quote(remote_auth_path)} '
                '"$CODEX_HOME/auth.json"\n'
            )
        else:
            self.logger.debug("Codex auth: using OPENAI_API_KEY")
            env["OPENAI_API_KEY"] = self._get_env("OPENAI_API_KEY") or ""
            setup_command = (
                f"cat >{shlex.quote(remote_auth_path)} <<EOF\n"
                '{\n  "OPENAI_API_KEY": "${OPENAI_API_KEY}"\n}\nEOF\n'
                f"ln -sf {shlex.quote(remote_auth_path)} "
                '"$CODEX_HOME/auth.json"\n'
            )

        if openai_base_url := self._get_env("OPENAI_BASE_URL"):
            env["OPENAI_BASE_URL"] = openai_base_url
            setup_command += (
                '\ncat >>"$CODEX_HOME/config.toml" <<TOML\n'
                'openai_base_url = "${OPENAI_BASE_URL}"\n'
                "TOML"
            )

        skills_command = self._build_register_skills_command()
        if skills_command:
            setup_command += f"\n{skills_command}"

        mcp_command = self._build_register_mcp_servers_command()
        if mcp_command:
            setup_command += f"\n{mcp_command}"

        if setup_command.strip():
            await self.exec_as_agent(
                environment, command=setup_command, env=env
            )

        # Construct the codex exec command and the resume variant.
        output_path = (EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME).as_posix()
        common_codex_flags = (
            "--dangerously-bypass-approvals-and-sandbox "
            "--skip-git-repo-check "
            f"--model {model} "
            "--json "
            "--enable unified_exec "
            f"{cli_flags_arg}"
        )

        def initial_cmd(prompt: str) -> str:
            return (
                "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
                f"codex exec {common_codex_flags}-- "
                f"{shlex.quote(prompt)} "
                f"2>&1 </dev/null | tee -a {shlex.quote(output_path)}"
            )

        def resume_cmd(prompt: str) -> str:
            # `codex exec resume --last` reads the most recent session from
            # $CODEX_HOME/sessions and appends a new turn. Same flags as
            # the initial run for consistency.
            return (
                "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
                f"codex exec resume --last {common_codex_flags}-- "
                f"{shlex.quote(prompt)} "
                f"2>&1 </dev/null | tee -a {shlex.quote(output_path)}"
            )

        try:
            # 1) Initial pass.
            await self.exec_as_agent(
                environment, command=initial_cmd(instruction), env=env
            )

            # 2) Reprompt loop.
            iteration = 0
            while True:
                remaining_minutes = await self._read_remaining_minutes(environment)
                if remaining_minutes is None:
                    self.logger.info(
                        "[reprompt] timer expired or unparseable; stopping"
                    )
                    break
                if remaining_minutes < min_remaining_minutes:
                    self.logger.info(
                        "[reprompt] %d min remaining < threshold %d; stopping",
                        remaining_minutes,
                        min_remaining_minutes,
                    )
                    break

                iteration += 1
                hours, mins = divmod(remaining_minutes, 60)
                continuation = (
                    f"You still have {hours}h {mins}m remaining. Please "
                    "continue improving your result and maximize performance."
                )
                self.logger.info(
                    "[reprompt] iteration %d: %dh %dm remaining; resuming",
                    iteration,
                    hours,
                    mins,
                )
                await self.exec_as_agent(
                    environment, command=resume_cmd(continuation), env=env
                )

        finally:
            # Save sessions for postmortem and tear down CODEX_HOME — same
            # as the parent's finally block.
            try:
                await self.exec_as_agent(
                    environment,
                    command=(
                        f"mkdir -p {EnvironmentPaths.agent_dir.as_posix()}\n"
                        'if [ -d "$CODEX_HOME/sessions" ]; then\n'
                        f"  rm -rf {(EnvironmentPaths.agent_dir / 'sessions').as_posix()}\n"
                        f'  cp -R "$CODEX_HOME/sessions" '
                        f'{(EnvironmentPaths.agent_dir / "sessions").as_posix()}\n'
                        "fi"
                    ),
                    env=env,
                )
            except Exception:
                pass
            try:
                await self.exec_as_agent(
                    environment,
                    command=(
                        f'rm -rf {shlex.quote(remote_secrets_dir)} "$CODEX_HOME"'
                    ),
                    env=env,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Helpers

    def _resolve_min_remaining_minutes(self) -> int:
        """Override threshold via CODEX_REPROMPT_MIN_MINUTES env."""
        raw = self._get_env("CODEX_REPROMPT_MIN_MINUTES")
        if not raw:
            return _DEFAULT_MIN_REMAINING_MINUTES
        try:
            value = int(raw)
        except ValueError:
            self.logger.warning(
                "CODEX_REPROMPT_MIN_MINUTES=%r is not an int; "
                "using default %d",
                raw,
                _DEFAULT_MIN_REMAINING_MINUTES,
            )
            return _DEFAULT_MIN_REMAINING_MINUTES
        if value < 0:
            self.logger.warning(
                "CODEX_REPROMPT_MIN_MINUTES must be >= 0; got %d, using %d",
                value,
                _DEFAULT_MIN_REMAINING_MINUTES,
            )
            return _DEFAULT_MIN_REMAINING_MINUTES
        return value

    async def _read_remaining_minutes(
        self, environment: BaseEnvironment
    ) -> int | None:
        """Run timer.sh in the sandbox and parse remaining minutes.

        Returns:
            int: remaining minutes when the timer is still ticking
            None: when the timer has expired or output is unparseable
        """
        result = await environment.exec(
            command=f"bash {_TIMER_SCRIPT}",
            cwd=None,  # timer.sh is in the agent's workspace, default cwd
        )
        stdout = (result.stdout or "").strip()
        if "expired" in stdout.lower():
            return None
        match = _TIMER_REMAINING_RE.search(stdout)
        if not match:
            self.logger.warning(
                "[reprompt] could not parse timer.sh output: %r", stdout
            )
            return None
        hours = int(match.group(1))
        minutes = int(match.group(2))
        return hours * 60 + minutes
