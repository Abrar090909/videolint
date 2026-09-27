"""A real encoded fixture checks upload, analysis, timeline timestamps, feedback, deletion."""
import json
import tempfile
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

from videolint import media, server


class VideoLintEndToEnd(unittest.TestCase):
    def test_real_video_through_api(self):
        with tempfile.TemporaryDirectory() as temp:
            server.DATA = Path(temp) / "data"
            server.DATA.mkdir()
            fixture = Path(temp) / "fixture.mp4"
            encoded = media.run(
                "-y", "-f", "lavfi", "-i", "testsrc2=duration=2:size=160x90:rate=30",
                "-f", "lavfi", "-i", "color=c=black:s=160x90:d=1:r=30",
                "-f", "lavfi", "-i", "color=c=red:s=160x90:d=3:r=30",
                "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
                "-map", "[v]", "-map", "3:a", "-t", "6", "-c:v", "mpeg4",
                "-c:a", "aac", str(fixture), timeout=60)
            self.assertEqual(encoded.returncode, 0, encoded.stderr.decode(errors="replace")[-500:])
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
            thread = Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{httpd.server_port}"
            try:
              with urllib.request.urlopen(base + "/") as response:
                  self.assertIn(b"/design.css", response.read())
              with urllib.request.urlopen(base + "/design.css") as response:
                  self.assertEqual(response.status, 200)
                  self.assertIn(b".editor-grid", response.read())
              with patch.dict("os.environ", {"GEMINI_API_KEY": "", "VIDEOLINT_AI_PROVIDER": "gemini"}):
                request = urllib.request.Request(base + "/api/analysis", fixture.read_bytes(), {
                    "Content-Type": "application/octet-stream", "X-Filename": "fixture.mp4",
                    "X-Goal": "Make a 6-second technical test video"}, method="POST")
                with urllib.request.urlopen(request) as response:
                    job_id = json.load(response)["id"]
                for _ in range(80):
                    time.sleep(.15)
                    with urllib.request.urlopen(base + "/api/analysis/" + job_id) as response:
                        job = json.load(response)
                    if job["status"] in ("complete", "failed"):
                        break
                self.assertEqual(job["status"], "complete", job.get("error"))
                self.assertAlmostEqual(job["metadata"]["durationMs"], 6000, delta=100)
                kinds = {issue["type"] for issue in job["issues"]}
                self.assertIn("black_frames", kinds)
                self.assertIn("frozen_frames", kinds)
                self.assertIn("silence", kinds)
                black = next(i for i in job["issues"] if i["type"] == "black_frames")
                self.assertAlmostEqual(black["startMs"], 2000, delta=600)
                self.assertTrue(all(c["status"] != "success" for c in job["checkers"] if c["checkerId"] in ("transcript", "repetition")))
                self.assertEqual(job["debug"]["transcript"], "NO SPEECH")
                self.assertTrue(all(c["reason"] == "No speech detected; speech-dependent checks are not applicable"
                    for c in job["checkers"] if c["checkerId"] in ("interrupted_thought", "repetition", "ending", "topic_alignment")))
                feedback = urllib.request.Request(base + f"/api/analysis/{job_id}/feedback",
                    json.dumps({"issueId": black["id"], "accepted": True}).encode(),
                    {"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(feedback) as response:
                    self.assertTrue(json.load(response)["ok"])
                with urllib.request.urlopen(base + f"/api/analysis/{job_id}") as response:
                    self.assertTrue(json.load(response)["feedback"][black["id"]])
                video = urllib.request.Request(base + f"/api/analysis/{job_id}/video", headers={"Range": "bytes=0-99"})
                with urllib.request.urlopen(video) as response:
                    self.assertEqual(response.status, 206)
                    self.assertEqual(len(response.read()), 100)
                request = urllib.request.Request(base + f"/api/analysis/{job_id}", method="DELETE")
                with urllib.request.urlopen(request) as response:
                    self.assertTrue(json.load(response)["ok"])
                self.assertFalse((server.DATA / job_id).exists())
            finally:
                httpd.shutdown()
                httpd.server_close()


if __name__ == "__main__":
    unittest.main()
