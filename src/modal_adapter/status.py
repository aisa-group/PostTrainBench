#!/usr/bin/env python3
"""Show the status of submitted PostTrainBench runs on Modal.

Reads modal_runs.jsonl (written by submit.py) and reports, per run:
  - the state of the spawned run_agent call, and
  - which pipeline phases have produced their artifacts on the results
    volume (the volume is the source of truth; judge/eval are spawned
    server-side, so their progress is visible through their outputs).
"""

import json
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import APP_NAME  # noqa: E402

LEDGER = Path(__file__).resolve().parent.parent.parent / "modal_runs.jsonl"

PHASE_MARKERS = [
    ("agent", "time_taken.txt"),
    ("judge", "contamination_judgement.txt"),
    ("eval", "metrics.json"),
]


def agent_call_state(call_id: str) -> str:
    try:
        call = modal.FunctionCall.from_id(call_id)
        call.get(timeout=0)
        return "finished"
    except TimeoutError:
        return "queued/running"
    except modal.exception.OutputExpiredError:
        return "finished (output expired; see volume)"
    except Exception as e:  # noqa: BLE001
        return f"error: {type(e).__name__}: {e}"


def main() -> None:
    if not LEDGER.exists():
        sys.exit(f"no ledger at {LEDGER}; submit runs with submit.py first")

    results_vol = modal.Volume.from_name("ptb-results")
    records = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]

    for rec in records:
        rel = rec["eval_dir"]
        try:
            entries = {Path(e.path).name for e in results_vol.listdir(rel)}
        except Exception:  # noqa: BLE001 - directory not created yet
            entries = set()

        phases = " ".join(
            f"{name}:{'done' if marker in entries else '-'}"
            for name, marker in PHASE_MARKERS
        )
        attempts = "attempts.log" in entries
        print(
            f"run {rec['run_id']}  {rel}\n"
            f"    agent call: {agent_call_state(rec['call_id'])}   {phases}"
            + ("   [has attempts.log]" if attempts else "")
        )

    print(
        "\nFetch results:  modal volume get ptb-results / ./results_modal\n"
        f"Live logs:      modal app logs {APP_NAME}"
    )


if __name__ == "__main__":
    main()
