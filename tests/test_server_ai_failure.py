import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from videolint import server
from videolint.ai import AIProviderError


class FailingProvider:
    name = "gemini"
    model = "gemini-3.6-flash"

    def judge(self, question, context):
        raise AIProviderError("Gemini rate limited (HTTP 429)")


class FailureIsolationTests(unittest.TestCase):
    def test_ai_rate_limit_keeps_technical_findings(self):
        old_data = server.DATA
        try:
            with tempfile.TemporaryDirectory() as folder:
                server.DATA = Path(folder)
                job_id = "a" * 32
                directory = server.DATA / job_id
                directory.mkdir()
                (directory / "video").write_bytes(b"fixture")
                server.save_job({"id": job_id, "filename": "fixture.mp4", "goal": "Make a 6-second video",
                    "status": "queued", "createdAt": time.time(), "updatedAt": time.time(),
                    "metadata": None, "issues": [], "checkers": [], "transcript": [], "shots": [],
                    "audio": [], "feedback": {}, "error": None})
                black = {"id":"black_frames_2000", "type":"black_frames", "startMs":2000, "endMs":2500}
                words = [
                    {"startMs":1000,"endMs":1500,"text":"The main reason is"},
                    {"startMs":2100,"endMs":2600,"text":"that costs fell quickly"},
                ]
                with patch.object(server.media, "probe", return_value={"durationMs":6000,"hasAudio":True}), \
                     patch.object(server.media, "analyze_frames", return_value=([black], [{"startMs":0},{"startMs":2000}])), \
                     patch.object(server.media, "analyze_audio", return_value=([], [])), \
                     patch.object(server.editorial, "transcribe", return_value=words), \
                     patch.object(server.ai, "get_transcriber", return_value=object()), \
                     patch.object(server.ai, "get_provider", return_value=FailingProvider()):
                    server.process(job_id)
                result = server.read_job(job_id)
                self.assertEqual(result["status"], "complete")
                self.assertIn(black, result["issues"])
                self.assertEqual(next(c for c in result["checkers"] if c["checkerId"] == "interrupted_thought")["status"], "failed")
                self.assertEqual(next(c for c in result["checkers"] if c["checkerId"] == "audio")["status"], "success")
        finally:
            server.DATA = old_data
