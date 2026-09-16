"""Pytest bootstrap.

Adds two directories to ``sys.path``:

* ``src/kaggle_harbor`` — so ``agents.ptb_harness`` imports exactly as it does
  on the executor, where `harbor_apply_custom_import_pythonpath`
  (container/harbor-base/entrypoint-common.sh:531-541) prepends the
  task-definition root that holds the staged ``agents/`` package. Also makes
  ``build_tasks`` / ``config`` importable.
* ``src/trace_parsing`` — upstream's parsers use flat sibling imports
  (``import claude_parser``), which work for ``parse_trace.py`` only because
  Python puts a script's own directory on ``sys.path``. Tests import it as a
  module, so the directory has to be added explicitly.

``harbor`` itself must be importable in the current environment (harbor
0.21.0 is what ``.venv-harbor`` holds). Same idiom as
container/harbor-base/oneshot/tests/conftest.py.
"""

import sys
from pathlib import Path

_KAGGLE_HARBOR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _KAGGLE_HARBOR.parent.parent
_TRACE_PARSING = _REPO_ROOT / "src" / "trace_parsing"

for _path in (_KAGGLE_HARBOR, _TRACE_PARSING):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
