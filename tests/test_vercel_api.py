"""Check the synchronous hosted endpoint with a real small video."""
import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from api.analysis import handler


class HostedAnalysisTest(unittest.TestCase):
    def test_small_video_returns_complete_report(self):
        video = Path(__file__).resolve().parents[1] / "data" / "speech-validation" / "bad-cut.mp4"
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/api/analysis",
                video.read_bytes(),
                {"Content-Type": "application/octet-stream", "X-Filename": "bad-cut.mp4",
                 "X-Goal": "Make a short clean video"},
                method="POST",
            )
            with patch.dict("os.environ", {"GEMINI_API_KEY": "", "VIDEOLINT_AI_PROVIDER": "gemini"}):
                with urllib.request.urlopen(request, timeout=60) as response:
                    report = json.load(response)
            self.assertEqual(report["status"], "complete", report.get("error"))
            self.assertEqual(report["filename"], "bad-cut.mp4")
            self.assertTrue(report["metadata"]["durationMs"] > 0)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()
