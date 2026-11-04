#!/usr/bin/env python3
"""Check the actual evaluator request builders without GPUs, datasets or API calls.

Run from the repo root: python3 dev_utils/smollm3_stop/test_request_stops.py

Load only the entrypoint and model/template selection functions using AST, so
unrelated scorer imports do not require the evaluation container. Execute these
functions unchanged, capturing their first Inspect call or vLLM HTTP request.
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "src" / "eval" / "tasks"
INSPECT_TASKS = ("aime2025", "gsm8k", "gpqamain", "humaneval")
HTTP_TASKS = ("healthbench", "arenahardwriting")
VARIANTS = ("evaluate.py", "evaluate_final_eval.py")
HTTP_VARIANTS = VARIANTS + ("evaluate_openrouter.py", "evaluate_openrouter_final_eval.py")


class RequestCaptured(BaseException):
    """Stop at the first request without entering retries or grading."""


def load_functions(path: Path, names: tuple[str, ...], namespace: dict) -> dict:
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names), path
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class RequestStopTests(unittest.TestCase):
    def model_cases(self, directory):
        # A final_model path contains no family name, so exercise architecture detection too.
        for family, repo, architecture in (
            ("smollm", "HuggingFaceTB/SmolLM3-3B-Base", "SmolLM3ForCausalLM"),
            ("qwen", "Qwen/Qwen3-4B-Base", "Qwen3ForCausalLM"),
            ("llama", "meta-llama/Llama-3.2-3B", "LlamaForCausalLM"),
            ("gemma", "google/gemma-3-4b-pt", "Gemma3ForConditionalGeneration"),
        ):
            yield family, repo
            checkpoint = Path(directory) / "final_model"
            checkpoint.mkdir(exist_ok=True)
            (checkpoint / "config.json").write_text(json.dumps({"architectures": [architecture]}))
            yield family, str(checkpoint)

    def args(self, model_path):
        return NS(model_path=model_path, templates_dir="templates", limit=1,
                  gpu_memory_utilization=0.3, max_tokens=32, max_new_tokens=32,
                  max_connections=1, json_output_file=None, seed=7, model_alias="test")

    def check_stops(self, payload, family):
        self.assertEqual(payload.get("stop_token_ids"), [128012] if family == "smollm" else None)
        # Base EOS and generation-config EOS IDs must remain enabled.
        self.assertNotIn("ignore_eos", payload)
        self.assertNotIn("skip_special_tokens", payload)
        self.assertNotIn("stop", payload)

    def test_inspect_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            for task in INSPECT_TASKS:
                for variant in VARIANTS:
                    path = TASKS / task / variant
                    for family, model_path in self.model_cases(directory):
                        with self.subTest(task=task, variant=variant, family=family, model=model_path):
                            capture = Mock(side_effect=RequestCaptured)
                            fake_task = NS(solver=[])
                            inspect_evals = NS(**{
                                name: NS(**{name: lambda **kwargs: fake_task}) for name in INSPECT_TASKS
                            })
                            namespace = load_functions(path, ("main", "model_type", "template_kwargs"), {
                                "os": os, "json": json, "parse_args": lambda: self.args(model_path),
                                "init_display_type": lambda value: None, "inspect_eval": capture,
                                "inspect_evals": inspect_evals, "gpqa_main": lambda **kwargs: fake_task,
                                "task_with": lambda task, **kwargs: task, "aime_scorer_exact": lambda: None,
                                "gsm8k_match_exact": lambda: None, "verify": lambda: None,
                                "per_sample_seed": lambda solver, seed: solver,
                            })
                            with self.assertRaises(RequestCaptured):
                                namespace["main"]()
                            capture.assert_called_once()
                            kwargs = capture.call_args.kwargs
                            self.check_stops(kwargs.get("extra_body") or {}, family)
                            self.assertEqual(kwargs["model"], f"vllm/{model_path}")
                            self.assertEqual(kwargs["max_tokens"], 32)
                            # Request parameters must not accidentally become vLLM CLI/server arguments.
                            self.assertNotIn("stop_token_ids", kwargs["model_args"])
                            self.assertNotIn("extra_body", kwargs["model_args"])

    def test_http_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            for task in HTTP_TASKS:
                for variant in HTTP_VARIANTS:
                    path = TASKS / task / variant
                    for family, model_path in self.model_cases(directory):
                        with self.subTest(task=task, variant=variant, family=family, model=model_path):
                            capture = Mock(side_effect=RequestCaptured)
                            session = NS(headers={}, post=capture)
                            server = NS(start=lambda: 8000, stop=Mock())
                            namespace = load_functions(path, ("generate_answers", "model_type"), {
                                "os": os, "json": json, "VLLMServer": lambda *args: server,
                                "requests": NS(Session=lambda: session, RequestException=Exception),
                                "tqdm": lambda items, **kwargs: items, "DATA_PATH": Path(directory),
                                "get_questions": lambda args: [{"prompt": "test", "uid": "sample"}],
                                "sample_seed": lambda seed, sid: seed + 123,
                                "VLLM_GENERATION_RETRY": 1, "VLLM_REQUEST_TIMEOUT": 1,
                            })
                            args = [self.args(model_path)]
                            if task == "healthbench":
                                args.append([NS(conversation=[{"role": "user", "content": "test"}],
                                                prompt_id="sample", example_id="sample")])
                            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RequestCaptured):
                                namespace["generate_answers"](*args)
                            capture.assert_called_once()
                            payload = capture.call_args.kwargs["json"]
                            self.check_stops(payload, family)
                            self.assertEqual(payload["model"], model_path)
                            self.assertEqual(payload["max_tokens"], 32)
                            if "final_eval" in variant:
                                self.assertEqual(payload["seed"], 130)
                            server.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
