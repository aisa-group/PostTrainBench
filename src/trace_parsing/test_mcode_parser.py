from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcode_parser
import parse_trace


class MCodeParserTest(unittest.TestCase):
    def test_parse_stream_json_trace(self) -> None:
        trace = """[2026-08-17T01:02:03Z] {\"type\":\"session-status\",\"status\":\"started\",\"turnId\":\"turn-test\"}
{\"type\":\"delta\",\"role\":\"assistant\",\"thinking\":\"checking\",\"content\":\"hello\",\"toolCalls\":[{\"id\":\"tool-test\",\"name\":\"bash\",\"status\":1,\"input\":{\"command\":\"pwd\"}}]}
{\"type\":\"delta\",\"role\":\"assistant\",\"toolCalls\":[{\"id\":\"tool-test\",\"name\":\"bash\",\"status\":2,\"input\":{\"command\":\"pwd\"},\"output\":{\"content\":[{\"type\":\"text\",\"text\":\"/home/ben/task\\n\"}],\"details\":{\"exitCode\":0}}}]}
{\"type\":\"message\",\"message\":{\"role\":\"assistant\",\"content\":\"done\",\"finishReason\":\"stop\",\"usage\":{\"totalTokens\":12,\"inputTokens\":8,\"outputTokens\":4,\"cacheReadTokens\":2}}}
{\"schemaVersion\":1,\"type\":\"exec.result\",\"sessionId\":\"session-test\",\"turnId\":\"turn-test\",\"status\":\"succeeded\",\"answer\":\"done\",\"durationMs\":321}
not-json
"""
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "trace.jsonl"
            output_path = Path(tmp) / "trace.txt"
            input_path.write_text(trace, encoding="utf-8")

            mcode_parser.parse(input_path, output_path)
            parsed = output_path.read_text(encoding="utf-8")

        self.assertIn("type: session-status | status: started", parsed)
        self.assertIn("ts: 2026-08-17T01:02:03Z", parsed)
        self.assertIn("Thinking:\n    checking", parsed)
        self.assertIn("Content:\n    hello", parsed)
        self.assertIn("Tool: bash | id: tool-test | status: running", parsed)
        self.assertIn('"command": "pwd"', parsed)
        self.assertIn("/home/ben/task", parsed)
        self.assertIn("exitCode", parsed)
        self.assertIn("totalTokens=12", parsed)
        self.assertIn("Answer:\n    done", parsed)
        self.assertIn("Duration: 321 ms", parsed)
        self.assertIn("NOT PARSABLE", parsed)
        self.assertIn("not-json", parsed)

    def test_parse_trace_dispatches_mcode(self) -> None:
        self.assertIs(parse_trace.select_parser("mcode"), mcode_parser.parse)
        self.assertIs(parse_trace.select_parser("mcode-minimax-m3"), mcode_parser.parse)


if __name__ == "__main__":
    unittest.main()
