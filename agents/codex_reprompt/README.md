# `codex-reprompt` — custom harbor agent

Codex with a reprompt loop: after the initial pass exits, polls
`bash timer.sh` (provided by PostTrainBench tasks). If the agent finished
early and there's more than `CODEX_REPROMPT_MIN_MINUTES` remaining,
resumes the same session via `codex exec resume --last` with a
continuation prompt and loops. Stops when the timer reports expired or
remaining drops below the threshold.

This mirrors `agents/codex_non_api_*_reprompt/solve.sh` from the condor
pipeline. It extends harbor's built-in `Codex` agent so all standard
flags (auth, reasoning_effort, MCP servers, base URL overrides) work
the same way.

## Loading

Harbor's `--agent-import-path` takes a `module:Class` form, so the
working directory needs `agents/` on `PYTHONPATH`. From the repo root:

```bash
PYTHONPATH="$(pwd):$PYTHONPATH" \
harbor run \
    --path src/harbor_adapter/tasks/posttrainbench-gsm8k-qwen3-1.7b \
    --agent-import-path agents.codex_reprompt.agent:CodexReprompt \
    --model gpt-5.3-codex \
    --ak reasoning_effort=high \
    --env modal
```

## API key auth (most variants)

Default: uses `OPENAI_API_KEY` from your shell. Pass `--reasoning-effort`
to control reasoning depth (`low` / `medium` / `high` / `xhigh`):

```bash
PYTHONPATH="$(pwd):$PYTHONPATH" \
harbor run \
    --path tasks/posttrainbench-gsm8k-qwen3-1.7b \
    --agent-import-path agents.codex_reprompt.agent:CodexReprompt \
    --model gpt-5.3-codex \
    --ak reasoning_effort=high \
    --env modal
```

Equivalent of condor's `agents/codex_non_api_high_reprompt/` minus the
auth.json (use the section below for that).

## ChatGPT-Pro subscription auth (`*_non_api_*_reprompt` parity)

Set `CODEX_AUTH_JSON_PATH` to your local `auth.json` (the one you
generated via `codex login --device-auth`, by default at
`~/.codex/auth.json`). Harbor's Codex base reads the file from your
machine and uploads it into the sandbox under `$CODEX_HOME/auth.json`:

```bash
PYTHONPATH="$(pwd):$PYTHONPATH" \
CODEX_AUTH_JSON_PATH=~/.codex/auth.json \
harbor run \
    --path tasks/posttrainbench-gsm8k-qwen3-1.7b \
    --agent-import-path agents.codex_reprompt.agent:CodexReprompt \
    --model gpt-5.3-codex \
    --ak reasoning_effort=high \
    --env modal
```

Or use the file we keep in this repo for non-API variants:

```bash
CODEX_AUTH_JSON_PATH="$(pwd)/agents/codex_non_api/auth.json"
```

## Tunables

| Env var | Default | Effect |
|---|---|---|
| `CODEX_REPROMPT_MIN_MINUTES` | `30` | Threshold for triggering another resume; below this the loop stops |
| `CODEX_AUTH_JSON_PATH` | unset | Local path to ChatGPT-Pro `auth.json`; when set, uploaded to sandbox and used in place of `OPENAI_API_KEY` |
| `CODEX_FORCE_AUTH_JSON` | unset | Truthy → use `~/.codex/auth.json` from the harbor host as the upload source |

## How it differs from the condor scripts

The condor scripts (`agents/codex_non_api_*_reprompt/solve.sh`) hard-code
reasoning effort by writing `model_reasoning_effort = "high"` (or `"xhigh"`)
into `~/.codex/config.toml`. Here the same is achieved via harbor's
built-in `--reasoning-effort` flag, which writes the equivalent `-c`
override on the codex CLI invocation. Behaviourally identical.

## Mapping to condor variants

| Condor variant | Invocation |
|---|---|
| `codex_non_api_reprompt` | `--agent-import-path …:CodexReprompt`<br>(no kwarg → harbor's `CliFlag` default of `high` is used) |
| `codex_non_api_high_reprompt` | `--ak reasoning_effort=high` |
| `codex_non_api_xhigh_reprompt` | `--ak reasoning_effort=xhigh` |

Plus `CODEX_AUTH_JSON_PATH=…` for the subscription auth path.
