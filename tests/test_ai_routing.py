"""Editorial routing and development metadata use only in-memory providers."""
import unittest
from unittest.mock import patch

from videolint import ai, server


class Stub:
    def __init__(self, name, result=None, failure=False):
        self.name = name
        self.model = name + "-test"
        self.result = result or ai.Judgment("yes", .91, "Focused evidence supports this.")
        self.failure = failure
        self.calls = 0

    def judge(self, question, context):
        self.calls += 1
        if self.failure:
            raise ai.AIProviderError(self.name + " unavailable")
        return self.result

    def connectivity_test(self):
        pass


class RoutingTests(unittest.TestCase):
    def test_jev_failure_uses_gemini_once_and_logs_actual_provider(self):
        jev, gemini = Stub("jev", failure=True), Stub("gemini")
        fallback = ai.FallbackJudgmentProvider(jev, gemini)
        job = {"debug": {"judgments": []}}
        recorded = server.RecordedJudgmentProvider(fallback, job, "interrupted_thought")
        judgment = recorded.judge("Does this edit interrupt a single semantic thought?",
                                  {"cutTimestampMs": 4200, "transcriptBeforeCut": "The reason is",
                                   "transcriptAfterCut": "prices fell"})
        self.assertEqual(judgment.verdict, "yes")
        self.assertEqual((jev.calls, gemini.calls), (1, 1))
        self.assertEqual(recorded.name, "gemini")
        self.assertEqual(job["debug"]["judgments"][0]["timestampMs"], 4200)
        self.assertEqual(job["debug"]["judgments"][0]["provider"], "gemini")
        self.assertNotIn("transcriptBeforeCut", job["debug"]["judgments"][0])

    def test_missing_jev_key_can_select_gemini(self):
        gemini = Stub("gemini")
        with patch.dict("os.environ", {"VIDEOLINT_AI_PROVIDER": "jev"}), \
             patch.object(ai, "get_jev_provider", side_effect=ai.AIProviderError("missing")), \
             patch.object(ai, "get_gemini_provider", return_value=gemini):
            self.assertIs(ai.get_provider(), gemini)


if __name__ == "__main__":
    unittest.main()
