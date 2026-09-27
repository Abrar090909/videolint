"""Labeled evaluation uses mocked provider decisions and no API quota."""
import unittest

from videolint.ai import Judgment
from videolint.evaluation import evaluate, heuristic, validate_examples


class Stub:
    def __init__(self, verdict):
        self.verdict = verdict
        self.last_cost_usd = .002

    def judge(self, question, context):
        return Judgment(self.verdict, .9, "Mock judgment")


class EvaluationTests(unittest.TestCase):
    def test_real_labels_required(self):
        with self.assertRaises(ValueError):
            validate_examples([{"id": "a", "kind": "ending", "question": "Incomplete?", "context": {}}])

    def test_metrics_and_missing_provider(self):
        examples = [{"id": "a", "kind": "ending", "question": "Incomplete?",
                     "context": {"finalTranscript": "And therefore", "gapAfterSpeechMs": 100}, "label": True},
                    {"id": "b", "kind": "ending", "question": "Incomplete?",
                     "context": {"finalTranscript": "All done.", "gapAfterSpeechMs": 1500}, "label": False}]
        result = evaluate(examples, {"jev": Stub("yes"), "gemini": None, "heuristic": object()})["results"]
        self.assertEqual(result["gemini"]["status"], "skipped")
        self.assertEqual(result["jev"]["metrics"]["falsePositives"], 1)
        self.assertEqual(result["jev"]["metrics"]["accuracy"], .5)
        self.assertEqual(result["jev"]["metrics"]["reportedCostUsd"], .004)
        self.assertEqual(result["heuristic"]["metrics"]["accuracy"], 1)

    def test_heuristic_is_not_a_production_finding(self):
        self.assertEqual(heuristic("interrupted_thought", {"transcriptBeforeCut": "The reason is",
                        "transcriptAfterCut": "that prices fell."}), "yes")


if __name__ == "__main__":
    unittest.main()
