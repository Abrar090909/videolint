"""Local-first HTTP application and persisted analysis jobs."""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from . import ai, editorial, media

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WEB = ROOT / "web"
MAX_UPLOAD = 250 * 1024 * 1024
LOCK = threading.RLock()
POOL = ThreadPoolExecutor(max_workers=2)
ACTIVE: set[str] = set()


def _job_path(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("Invalid analysis id")
    return DATA / job_id


def read_job(job_id: str) -> dict:
    path = _job_path(job_id) / "report.json"
    if not path.is_file():
        raise FileNotFoundError("Analysis not found")
    with LOCK:
        return json.loads(path.read_text(encoding="utf-8"))


def save_job(job: dict) -> None:
    path = _job_path(job["id"]) / "report.json"
    with LOCK:
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)


def stage(job: dict, status: str) -> None:
    job["status"] = status
    job["updatedAt"] = time.time()
    save_job(job)


def check(job: dict, checker_id: str, callback) -> list[dict]:
    started = time.perf_counter()
    try:
        issues = callback()
        result = {"checkerId": checker_id, "status": "success", "issues": issues,
                  "latencyMs": round((time.perf_counter() - started) * 1000)}
        job["issues"].extend(issues)
    except Exception as exc:
        result = {"checkerId": checker_id, "status": "failed", "issues": [],
                  "error": str(exc)[:300], "latencyMs": round((time.perf_counter() - started) * 1000)}
    job["checkers"].append(result)
    save_job(job)
    return result["issues"]


def skipped(job: dict, checker_id: str, reason: str) -> None:
    job["checkers"].append({"checkerId": checker_id, "status": "skipped", "issues": [], "reason": reason})
    save_job(job)


class RecordedJudgmentProvider:
    """Log bounded decision metadata in development mode, never transcript or keys."""

    def __init__(self, provider: ai.AIJudgmentProvider, job: dict, checker_id: str):
        self.provider = provider
        self.job = job
        self.checker_id = checker_id

    @property
    def name(self) -> str:
        return self.provider.name

    @property
    def model(self) -> str:
        return self.provider.model

    def judge(self, question: str, context: dict) -> ai.Judgment:
        timestamp = next((context[key] for key in ("cutTimestampMs", "currentStartMs",
                          "endTimestampMs", "timestampMs") if key in context), None)
        started = time.perf_counter()
        try:
            raw = self.provider.judge(question, context)
            judgment = ai.Judgment.validate(vars(raw))
            record = {"checker": self.checker_id, "timestampMs": timestamp,
                      "provider": self.name, "model": self.model,
                      "latencyMs": round((time.perf_counter() - started) * 1000),
                      "verdict": judgment.verdict, "confidence": judgment.confidence}
            self.job["debug"]["judgments"].append(record)
            return judgment
        except Exception:
            self.job["debug"]["judgments"].append({
                "checker": self.checker_id, "timestampMs": timestamp,
                "provider": self.name, "model": self.model,
                "latencyMs": round((time.perf_counter() - started) * 1000),
                "verdict": None, "confidence": None, "status": "failed"})
            raise

    def connectivity_test(self) -> None:
        self.provider.connectivity_test()


def process(job_id: str) -> None:
    try:
        job = read_job(job_id)
        job.setdefault("debug", {"transcript": "PENDING", "gemini": "NOT CONFIGURED",
                                 "jev": "NOT CONFIGURED", "judgments": []})
        job["debug"]["gemini"] = "CONFIGURED" if ai.gemini_configured() else "NOT CONFIGURED"
        job["debug"]["jev"] = "CONFIGURED" if ai.jev_configured() else "NOT CONFIGURED"
        video = _job_path(job_id) / "video"
        stage(job, "extracting")
        metadata = media.probe(video)
        job["metadata"] = metadata
        save_job(job)
        check(job, "frames", lambda: _run_frames(job, video, metadata["durationMs"]))
        stage(job, "analyzing")
        check(job, "audio", lambda: _run_audio(job, video, metadata))
        check(job, "duration_goal", lambda: editorial.duration_check(job["goal"], metadata["durationMs"]))
        try:
            provider = ai.get_provider()
            unavailable = None
        except ai.AIProviderError as exc:
            provider = None
            unavailable = str(exc)
        silent_audio = bool(job["audio"]) and all(window["peak"] == 0 for window in job["audio"])
        if not metadata["hasAudio"] or silent_audio:
            job["debug"]["transcript"] = "NO SPEECH"
            skipped(job, "transcript", "No speech detected (silent or missing audio)")
        elif metadata["durationMs"] <= 1_800_000:
            try:
                transcriber = ai.get_transcriber()
            except ai.AIProviderError as exc:
                job["debug"]["transcript"] = "SKIPPED"
                skipped(job, "transcript", str(exc))
            else:
                stage(job, "transcribing")
                check(job, "transcript", lambda: _run_transcript(job, video, _job_path(job_id), transcriber))
                if job["checkers"][-1]["status"] == "failed":
                    job["debug"]["transcript"] = "FAILED"
                elif job["transcript"]:
                    job["debug"]["transcript"] = "SUCCESS"
                else:
                    job["debug"]["transcript"] = "NO SPEECH"
                    job["checkers"][-1]["status"] = "skipped"
                    job["checkers"][-1]["reason"] = "No speech detected in audio"
                save_job(job)
        else:
            job["debug"]["transcript"] = "SKIPPED"
            skipped(job, "transcript", "Gemini word-timestamp transcription supports up to 30 minutes")
        stage(job, "verifying")
        if not job["transcript"]:
            reason = ("No speech detected; speech-dependent checks are not applicable" if
                      job["debug"]["transcript"] == "NO SPEECH" else
                      "Transcript failed; speech-dependent checks could not run" if
                      job["debug"]["transcript"] == "FAILED" else
                      "No timestamped transcript available")
            for name in ("interrupted_thought", "repetition", "ending", "topic_alignment"):
                skipped(job, name, reason)
        elif provider is None:
            for name in ("interrupted_thought", "repetition", "ending", "topic_alignment"):
                skipped(job, name, unavailable)
        else:
            def for_checker(name: str) -> ai.AIJudgmentProvider:
                return (RecordedJudgmentProvider(provider, job, name) if
                        os.environ.get("VIDEOLINT_DEV_MODE") == "1" else provider)
            if job["shots"]:
                check(job, "interrupted_thought", lambda: editorial.InterruptedThoughtChecker().run(job["transcript"], job["shots"], for_checker("interrupted_thought")))
            else:
                skipped(job, "interrupted_thought", "Shot boundaries unavailable")
            check(job, "repetition", lambda: editorial.RepetitionChecker().run(job["transcript"], for_checker("repetition")))
            check(job, "ending", lambda: editorial.EndingCompletenessChecker().run(job["transcript"], metadata["durationMs"], for_checker("ending")))
            check(job, "topic_alignment", lambda: editorial.GoalAlignmentChecker().run(job["goal"], job["transcript"], metadata["durationMs"], for_checker("topic_alignment")))
        job["issues"].sort(key=lambda issue: (issue["startMs"], issue["id"]))
        stage(job, "complete")
    except Exception as exc:
        try:
            job = read_job(job_id)
            job["error"] = str(exc)[:400]
            stage(job, "failed")
        except FileNotFoundError:
            pass
    finally:
        ACTIVE.discard(job_id)


def _run_frames(job: dict, video: Path, duration_ms: int) -> list[dict]:
    issues, shots = media.analyze_frames(video, duration_ms)
    job["shots"] = shots
    return issues


def _run_audio(job: dict, video: Path, metadata: dict) -> list[dict]:
    issues, windows = media.analyze_audio(video, metadata["durationMs"], metadata["hasAudio"])
    job["audio"] = windows
    return issues


def _run_transcript(job: dict, video: Path, directory: Path, provider: ai.SpeechTranscriber) -> list[dict]:
    job["transcript"] = editorial.validate_transcript(
        editorial.transcribe(video, directory, provider), job["metadata"]["durationMs"])
    return []


class Handler(BaseHTTPRequestHandler):
    server_version = "VideoLint/0.1"

    def log_message(self, format, *args):
        # Requests may contain sensitive video names or goal text; don't log them.
        pass

    def json(self, status: int, payload: dict | list) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def error_json(self, status: int, message: str) -> None:
        self.json(status, {"error": message})

    def _body(self, maximum: int = 10000) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 1 or length > maximum:
            raise ValueError("Invalid request size")
        return self.rfile.read(length)

    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if path == "/api/health":
                return self.json(200, {"ok": True, "ffmpeg": media.ffmpeg(),
                    "aiProvider": ai.selected_provider_name(), "aiConfigured": ai.provider_configured(),
                    "transcriptionConfigured": ai.gemini_configured(),
                    "geminiConfigured": ai.gemini_configured(), "jevConfigured": ai.jev_configured()})
            if path == "/api/analysis":
                jobs = []
                for report in DATA.glob("*/report.json"):
                    try:
                        item = json.loads(report.read_text(encoding="utf-8"))
                        jobs.append({k: item.get(k) for k in ("id", "filename", "goal", "status", "createdAt", "metadata")})
                    except (ValueError, OSError):
                        continue
                jobs.sort(key=lambda x: x["createdAt"] or 0, reverse=True)
                return self.json(200, jobs)
            match = re.fullmatch(r"/api/analysis/([0-9a-f]{32})(?:/(video|issues|report))?", path)
            if match:
                job = read_job(match.group(1))
                if match.group(2) == "video":
                    return self.video(_job_path(job["id"]) / "video", job["filename"])
                if match.group(2) == "issues":
                    return self.json(200, job["issues"])
                return self.json(200, job)
            if path == "/" or path in ("/app.js", "/style.css", "/studio.css", "/design.css"):
                file = WEB / ("index.html" if path == "/" else path[1:])
                data = file.read_bytes()
                mime = "text/html" if file.suffix == ".html" else "text/javascript" if file.suffix == ".js" else "text/css"
                self.send_response(200)
                self.send_header("Content-Type", mime + "; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return self.wfile.write(data)
            self.error_json(404, "Not found")
        except FileNotFoundError:
            self.error_json(404, "Analysis not found")
        except (ValueError, OSError) as exc:
            self.error_json(400, str(exc))

    def video(self, file: Path, filename: str) -> None:
        size = file.stat().st_size
        start, end = 0, size - 1
        header = self.headers.get("Range")
        if header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
            if not match:
                return self.error_json(416, "Invalid range")
            start = int(match.group(1)) if match.group(1) else max(0, size - int(match.group(2)))
            end = int(match.group(2)) if match.group(1) and match.group(2) else size - 1
            if start > end or end >= size:
                return self.error_json(416, "Range outside file")
        self.send_response(206 if header else 200)
        self.send_header("Content-Type", "video/quicktime" if filename.lower().endswith(".mov") else "video/mp4")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "private, no-store")
        if header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with file.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = stream.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_POST(self):
        path = urlsplit(self.path).path
        try:
            if path == "/api/analysis":
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_UPLOAD:
                    return self.error_json(413, "Video must be under 250 MB")
                filename = unquote(self.headers.get("X-Filename", "video.mp4"))
                goal = unquote(self.headers.get("X-Goal", "")).strip()
                if not filename.lower().endswith((".mp4", ".mov")) or len(goal) < 5 or len(goal) > 2000:
                    return self.error_json(400, "Provide an MP4/MOV and a goal of 5–2000 characters")
                job_id = uuid.uuid4().hex
                directory = _job_path(job_id)
                directory.mkdir(parents=True)
                destination = directory / "video"
                try:
                    with destination.open("wb") as output:
                        remaining = length
                        while remaining:
                            chunk = self.rfile.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise ValueError("Upload ended early")
                            output.write(chunk)
                            remaining -= len(chunk)
                    job = {"id": job_id, "filename": Path(filename).name[:150], "goal": goal,
                           "status": "queued", "createdAt": time.time(), "updatedAt": time.time(),
                           "metadata": None, "issues": [], "checkers": [], "transcript": [], "shots": [],
                           "audio": [], "feedback": {}, "error": None,
                           "debug": {"transcript": "PENDING", "gemini": "NOT CONFIGURED",
                                     "jev": "NOT CONFIGURED", "judgments": []}}
                    save_job(job)
                    ACTIVE.add(job_id)
                    POOL.submit(process, job_id)
                    return self.json(202, {"id": job_id})
                except Exception:
                    shutil.rmtree(directory)
                    raise
            match = re.fullmatch(r"/api/analysis/([0-9a-f]{32})/feedback", path)
            if match:
                payload = json.loads(self._body())
                job = read_job(match.group(1))
                issue_id, accepted = payload.get("issueId"), payload.get("accepted")
                if not any(i["id"] == issue_id for i in job["issues"]) or type(accepted) is not bool:
                    return self.error_json(400, "Invalid issue feedback")
                job["feedback"][issue_id] = accepted
                save_job(job)
                return self.json(200, {"ok": True})
            self.error_json(404, "Not found")
        except FileNotFoundError:
            self.error_json(404, "Analysis not found")
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            self.error_json(400, str(exc))

    def do_DELETE(self):
        match = re.fullmatch(r"/api/analysis/([0-9a-f]{32})", urlsplit(self.path).path)
        if not match:
            return self.error_json(404, "Not found")
        job_id = match.group(1)
        if job_id in ACTIVE:
            return self.error_json(409, "Wait until analysis finishes before deleting")
        path = _job_path(job_id)
        if not path.is_dir():
            return self.error_json(404, "Analysis not found")
        shutil.rmtree(path)
        self.json(200, {"ok": True})


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    for report in DATA.glob("*/report.json"):
        try:
            job = json.loads(report.read_text(encoding="utf-8"))
            if job["status"] not in ("complete", "failed"):
                job["error"] = "Server stopped before analysis completed. Upload again to retry."
                stage(job, "failed")
        except (ValueError, OSError, KeyError):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"VideoLint running at http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
