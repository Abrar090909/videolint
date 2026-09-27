import unittest
from unittest.mock import patch
import tempfile
import wave
from pathlib import Path

from videolint import editorial, media
from videolint.ai import AIProviderError, Judgment


class StubProvider:
    name = "gemini"
    model = "gemini-3.6-flash"

    def __init__(self, result=Judgment("yes", .93, "The thought is cut short.")):
        self.result = result
        self.calls = []

    def judge(self, question, context):
        self.calls.append((question, context))
        return self.result


class EditorialTests(unittest.TestCase):
    def test_explicit_duration_only(self):
        self.assertEqual(editorial.duration_check("Make a 45-second cut", 60000)[0]["type"], "duration_goal")
        self.assertEqual(editorial.duration_check("Make a short cut", 60000), [])

    def test_repetition_retrieves_candidate_before_judgment(self):
        segments = [
            {"startMs": 0, "endMs": 1000, "text": "Battery factories got larger."},
            {"startMs": 1000, "endMs": 2000, "text": "The scale lowered costs."},
            {"startMs": 10000, "endMs": 11000, "text": "Larger plants make cells cheaper."},
            {"startMs": 11000, "endMs": 12000, "text": "That lowered battery costs."},
        ]
        provider = StubProvider()
        issues = editorial.RepetitionChecker().run(segments, provider)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["startMs"], 10000)
        self.assertEqual(issues[0]["source"], "gemini")
        self.assertGreater(issues[0]["evidence"]["metrics"]["candidateSimilarity"], .18)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(set(provider.calls[0][1]), {"earlierStartMs", "currentStartMs", "earlierPassage", "currentPassage"})

    def test_repetition_finds_adjacent_repeated_sentences(self):
        words = ["How", "old", "is", "the", "Brooklyn", "Bridge?"]
        segments = [{"startMs": i*300, "endMs": (i+1)*300,
                     "text": words[i % len(words)]} for i in range(12)]
        provider = StubProvider()
        issues = editorial.RepetitionChecker().run(segments, provider)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["startMs"], 1800)
        self.assertEqual(len(provider.calls), 1)

    def test_cut_sends_only_nearby_transcript_and_timestamp(self):
        segments = [
            {"startMs": 0, "endMs": 700, "text": "Distant opening words"},
            {"startMs": 4700, "endMs": 4900, "text": "The main reason is"},
            {"startMs": 5100, "endMs": 5500, "text": "that prices fell rapidly"},
            {"startMs": 15000, "endMs": 15500, "text": "Unrelated late passage"},
        ]
        provider = StubProvider()
        issues = editorial.InterruptedThoughtChecker().run(segments,
            [{"startMs": 0}, {"startMs": 5000}], provider)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["startMs"], 5000)
        context = provider.calls[0][1]
        self.assertEqual(context["cutTimestampMs"], 5000)
        self.assertNotIn("Distant", str(context))
        self.assertNotIn("Unrelated", str(context))
        self.assertIn("single semantic thought", provider.calls[0][0])

    def test_transcript_validation_rejects_missing_timestamps(self):
        words = [{"startMs": 1200, "endMs": 1600, "text": "Hello"}]
        self.assertEqual(editorial.validate_transcript(words, 2000), words)
        for bad in ([{"startMs": 1200, "text": "Hello"}],
                    [{"startMs": 1200, "endMs": 1200, "text": "Hello"}],
                    [{"startMs": 1200, "endMs": 1600, "text": ""}]):
            with self.assertRaises(ValueError):
                editorial.validate_transcript(bad, 2000)

    def test_uncertain_or_low_confidence_creates_no_issue(self):
        segments = [
            {"startMs": 1000, "endMs": 1500, "text": "The main reason is"},
            {"startMs": 2100, "endMs": 2600, "text": "that prices fell sharply"},
        ]
        for result in (Judgment("uncertain", .99, "Insufficient context"),
                       Judgment("yes", .7, "Maybe incomplete"),
                       Judgment("no", .99, "Natural cut")):
            provider = StubProvider(result)
            self.assertEqual(editorial.InterruptedThoughtChecker().run(
                segments, [{"startMs": 0}, {"startMs": 2000}], provider), [])

    def test_ending_and_goal_use_bounded_context(self):
        segments = [
            {"startMs": 0, "endMs": 1000, "text": "Here is the topic"},
            {"startMs": 12000, "endMs": 13500, "text": "And the final thought is"},
        ]
        provider = StubProvider()
        self.assertEqual(len(editorial.EndingCompletenessChecker().run(segments, 14000, provider)), 1)
        self.assertNotIn("Here is", str(provider.calls[0][1]))
        self.assertEqual(len(editorial.GoalAlignmentChecker().run("Explain the topic", segments, 14000, provider)), 1)
        self.assertLessEqual(len(provider.calls[1][1]["selectedTranscriptSections"]), 3)

    def test_judgment_validation(self):
        self.assertEqual(Judgment.validate({"verdict":"yes","confidence":.9,"reason":"Because."}).verdict, "yes")
        for bad in ({"verdict":"maybe","confidence":.9,"reason":"x"},
                    {"verdict":"yes","confidence":2,"reason":"x"},
                    {"verdict":"yes","confidence":True,"reason":"x"},
                    {"verdict":"yes","confidence":.9,"reason":""}):
            with self.assertRaises(AIProviderError):
                Judgment.validate(bad)

    def test_clipping_uses_decoded_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "clipped.wav"
            with wave.open(str(audio), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes((b"\xff\x7f" * 16000))
            issues, windows = media.analyze_audio(audio, 1000, True)
            self.assertTrue(windows)
            self.assertIn("audio_clipping", {i["type"] for i in issues})


if __name__ == "__main__":
    unittest.main()
