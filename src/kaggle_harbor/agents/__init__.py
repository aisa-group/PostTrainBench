"""Custom-import Harbor agents for the PostTrainBench port.

Staged to the task-definition root by build_tasks.py so that the executor's
PYTHONPATH shim (container/harbor-base/entrypoint-common.sh:531-541) can
import `agents.ptb_harness` from `KAGGLE_TASK_DEFINITION_ROOT`.

Same layout as the starter template's `agents/` package
(kaggle-benchmark-harbor-starter-template/agents/__init__.py).
"""
