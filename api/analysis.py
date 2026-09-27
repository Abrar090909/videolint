"""Synchronous small-video analysis for Vercel Functions."""
from __future__ import annotations

import json
import re
import tempfile
import time
import uuid
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote

from videolint import server


MAX_UPLOAD = 4 * 1024 * 1024


class handler(BaseHTTPRequestHandler):
    def respond(self, status: int, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= MAX_UPLOAD:
            return self.respond(413, {"error": "Hosted reviews support videos up to 4 MB."})
        filename = Path(unquote(self.headers.get("X-Filename", "video.mp4"))).name[:150]
        goal = unquote(self.headers.get("X-Goal", "")).strip()
        if not re.search(r"\.(mp4|mov)$", filename, re.I) or not 5 <= len(goal) <= 2000:
            return self.respond(400, {"error": "Provide an MP4/MOV and a goal of 5–2000 characters."})
        with tempfile.TemporaryDirectory() as temp:
            server.DATA = Path(temp)
            job_id = uuid.uuid4().hex
            directory = server.DATA / job_id
            directory.mkdir()
            with (directory / "video").open("wb") as output:
                remaining = length
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        return self.respond(400, {"error": "Upload ended early."})
                    output.write(chunk)
                    remaining -= len(chunk)
            now = time.time()
            server.save_job({
                "id": job_id, "filename": filename, "goal": goal,
                "status": "queued", "createdAt": now, "updatedAt": now,
                "metadata": None, "issues": [], "checkers": [], "transcript": [],
                "shots": [], "audio": [], "feedback": {}, "error": None,
                "debug": {"transcript": "PENDING", "gemini": "NOT CONFIGURED",
                          "jev": "NOT CONFIGURED", "judgments": []},
            })
            server.process(job_id)
            result = server.read_job(job_id)
            self.respond(200 if result["status"] == "complete" else 422, result)

    def do_GET(self):
        self.respond(405, {"error": "Reports are stored in this browser on the hosted site."})
