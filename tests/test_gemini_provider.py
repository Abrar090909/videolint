"""Gemini adapter contract tests use in-memory responses; no API quota is used."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from videolint.ai import AIProviderError, get_provider
from videolint.gemini_provider import GeminiProvider


class GeminiProviderTests(unittest.TestCase):
    def client(self, text):
        models = SimpleNamespace(generate_content=Mock(return_value=SimpleNamespace(text=text)))
        return SimpleNamespace(models=models, files=Mock(), interactions=Mock())

    def test_structured_verdict_and_requested_model(self):
        client = self.client(json.dumps({"verdict":"yes","confidence":.91,"reason":"A sentence breaks at the cut."}))
        provider = GeminiProvider(api_key="test-only", client=client)
        result = provider.judge("Does this edit cut interrupt a single thought?", {"cutTimestampMs": 5000})
        self.assertEqual(result.verdict, "yes")
        self.assertEqual(result.confidence, .91)
        kwargs = client.models.generate_content.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-3.6-flash")
        self.assertEqual(kwargs["config"]["response_mime_type"], "application/json")
        self.assertIn("cutTimestampMs", kwargs["contents"])

    def test_invalid_response_and_rate_limit_fail_closed(self):
        for text in ('not JSON', '{"verdict":"yes","confidence":1.8,"reason":"x"}'):
            provider = GeminiProvider(api_key="test-only", client=self.client(text))
            with self.assertRaises(AIProviderError):
                provider.judge("question", {"text":"evidence"})
        client = self.client("{}")
        error = RuntimeError("rate limit")
        error.code = 429
        client.models.generate_content.side_effect = error
        with self.assertRaisesRegex(AIProviderError, "HTTP 429"):
            GeminiProvider(api_key="test-only", client=client).judge("question", {})

    def test_word_timestamp_transcription_and_remote_cleanup(self):
        client = self.client("{}")
        client.files.upload.return_value = SimpleNamespace(name="files/test", uri="uri", mime_type="audio/mp4")
        response = {"steps": [{"content": [{"annotations": [{"type":"word_info", "text":"Hello",
                    "start_offset":"0.100s", "end_offset":"0.450s"}]}]}], "output_text":"Hello"}
        provider = GeminiProvider(api_key="test-only", client=client)
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "speech.m4a"
            audio.write_bytes(b"test")
            with patch.object(provider, "_transcription_response", return_value=response) as request:
                words = provider.transcribe(audio)
        self.assertEqual(words, [{"startMs":100,"endMs":450,"text":"Hello"}])
        request.assert_called_once_with("uri", "audio/mp4")
        client.files.delete.assert_called_once_with(name="files/test")

    def test_missing_final_word_end_uses_audio_boundary_with_provenance(self):
        client = self.client("{}")
        client.files.upload.return_value = SimpleNamespace(name="files/test", uri="uri", mime_type="audio/mp4")
        response = {"steps": [{"content": [{"annotations": [
            {"type":"word_info", "text":"The", "start_offset":"0.100s", "end_offset":"0.300s"},
            {"type":"word_info", "text":"ending", "start_offset":"0.300s"}]}]}]}
        provider = GeminiProvider(api_key="test-only", client=client)
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "speech.m4a"
            audio.write_bytes(b"test")
            with patch.object(provider, "_transcription_response", return_value=response), \
                 patch("videolint.gemini_provider._audio_duration_ms", return_value=700):
                words = provider.transcribe(audio)
        self.assertEqual(words[-1], {"startMs":300, "endMs":700, "text":"ending", "endInferred":True})

    def test_missing_key_does_not_instantiate_provider(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY":"", "VIDEOLINT_AI_PROVIDER":"gemini"}):
            with self.assertRaisesRegex(AIProviderError, "Gemini API key not configured"):
                get_provider()


if __name__ == "__main__":
    unittest.main()
